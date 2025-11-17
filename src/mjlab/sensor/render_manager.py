from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor


@wp.kernel
def unpack_rgb_kernel(
  packed: wp.array2d(dtype=wp.uint32),  # type: ignore (nworld, npixels)
  rgb: wp.array3d(dtype=wp.uint8),  # type: ignore (nworld, npixels, 3)
):
  """Unpack packed uint32 RGB into uint8 buffer for all cameras."""
  world_idx, pixel_idx = wp.tid()  # type: ignore

  # Unpack ARGB format: [31:24]=A, [23:16]=R, [15:8]=G, [7:0]=B
  b = wp.uint8(packed[world_idx, pixel_idx] & wp.uint32(0xFF))
  g = wp.uint8((packed[world_idx, pixel_idx] >> wp.uint32(8)) & wp.uint32(0xFF))
  r = wp.uint8((packed[world_idx, pixel_idx] >> wp.uint32(16)) & wp.uint32(0xFF))

  rgb[world_idx, pixel_idx, 0] = r
  rgb[world_idx, pixel_idx, 1] = g
  rgb[world_idx, pixel_idx, 2] = b


class RenderManager:
  """Manages rendering for all camera sensors in a scene."""

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    camera_sensors: list[CameraSensor],
  ):
    """Initialize the render manager.

    Args:
        mj_model: MuJoCo model (host)
        model: MJWarp model (device)
        data: MJWarp data (device)
        camera_sensors: List of camera sensors to manage
    """
    self.wp_device = (
      model.struct.device  # type: ignore[attr-defined]
      if hasattr(model.struct, "device")  # type: ignore[attr-defined]
      else wp.get_device()
    )
    self.camera_resolutions = [(s.cfg.width, s.cfg.height) for s in camera_sensors]
    render_rgb = tuple(("rgb" in s.cfg.type) for s in camera_sensors)
    render_depth = tuple(("depth" in s.cfg.type) for s in camera_sensors)
    print(render_rgb, render_depth)

    # Use first sensor's settings (could make this configurable)
    first_sensor = camera_sensors[0]
    use_textures = first_sensor.cfg.use_textures
    use_shadows = first_sensor.cfg.use_shadows
    enabled_geom_groups = list(first_sensor.cfg.enabled_geom_groups)

    # Create single RenderContext
    with wp.ScopedDevice(self.wp_device):
      self._ctx = mjwarp.create_render_context(
        mjm=mj_model,
        m=model.struct,  # type: ignore
        d=data.struct,  # type: ignore
        cam_resolutions=self.camera_resolutions,
        render_rgb=render_rgb,
        render_depth=render_depth,
        use_textures=use_textures,
        use_shadows=use_shadows,
        enabled_geom_groups=enabled_geom_groups,
      )

    self._model = model
    self._data = data
    self._render_rgb = render_rgb
    self._render_depth = render_depth
    self._rgb_adr = self._ctx.rgb_adr.numpy()
    self._depth_adr = self._ctx.depth_adr.numpy()
    self._rgb_size = self._ctx.rgb_size.numpy()
    self._depth_size = self._ctx.depth_size.numpy()

    # Allocate unpacked RGB buffer if needed
    if any(render_rgb):
      self._rgb_unpacked = wp.array3d(
        shape=(data.nworld, self._ctx.rgb_data.shape[1], 3),
        dtype=wp.uint8,
        device=self.wp_device,
      )
    else:
      self._rgb_unpacked = None

    # Link render manager back to sensors
    for sensor in camera_sensors:
      sensor.set_render_manager(self)
    
    self.use_cuda_graph = self.wp_device.is_cuda and wp.is_mempool_enabled(
      self.wp_device
    )
    self.create_graph()

  def create_graph(self) -> None:
    self.render_graph = None
    if self.use_cuda_graph:
      with wp.ScopedDevice(self.wp_device):
        with wp.ScopedCapture() as capture:
          mjwarp.render(self._model, self._data, self._ctx)
        self.render_graph = capture.graph

  def render(self) -> None:
    """Render all cameras (called once per scene.update)."""
    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.render_graph is not None:
        wp.capture_launch(self.render_graph)
      else:
        mjwarp.render(self._model, self._data, self._ctx)

    # Unpack RGB if needed
    if self._render_rgb and self._rgb_unpacked is not None:
      wp.launch(
        unpack_rgb_kernel,
        dim=(self._data.nworld, self._ctx.rgb_data.shape[1]),
        inputs=[self._ctx.rgb_data],
        outputs=[self._rgb_unpacked],
        device=self.wp_device,
      )

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    """Get RGB data for a specific camera.

    Args:
        cam_idx: Camera index

    Returns:
        RGB tensor of shape (nworld, height, width, 3) with dtype uint8
    """
    assert self._rgb_unpacked is not None, "RGB rendering not enabled"
    rgb_unpacked_torch = wp.to_torch(self._rgb_unpacked)
    start = int(self._rgb_adr[cam_idx])
    size = int(self._rgb_size[cam_idx])
    rgb_flat = rgb_unpacked_torch[:, start : start + size]

    return rgb_flat.reshape(
      self._data.nworld,
      self.camera_resolutions[cam_idx][1],
      self.camera_resolutions[cam_idx][0],
      3)

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    """Get depth data for a specific camera.

    Args:
        cam_idx: Camera index

    Returns:
        Depth tensor of shape (nworld, height, width, 1) with dtype float32
    """
    assert self._render_depth, "Depth rendering not enabled"
    depth_torch = wp.to_torch(self._ctx.depth_data)
    start = int(self._depth_adr[cam_idx])
    size = int(self._depth_size[cam_idx])
    depth_flat = depth_torch[:, start : start + size]
    return depth_flat.reshape(
      self._data.nworld,
      self.camera_resolutions[cam_idx][1],
      self.camera_resolutions[cam_idx][0],
      1)
