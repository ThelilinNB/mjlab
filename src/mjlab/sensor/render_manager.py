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
  packed: wp.array3d(dtype=wp.uint32),  # type: ignore (nworld, ncam, height*width)
  width: int,
  height: int,
  rgb: wp.array4d(dtype=wp.uint8),  # type: ignore (nworld, ncam, height*width, 3)
):
  """Unpack packed uint32 RGB into uint8 buffer for all cameras."""
  nworld_idx, cam_idx, pixel_idx = wp.tid()  # type: ignore

  if pixel_idx < width * height:
    packed_val = packed[nworld_idx, cam_idx, pixel_idx]

    # Unpack ARGB format: [31:24]=A, [23:16]=R, [15:8]=G, [7:0]=B
    b = wp.uint8(packed_val & wp.uint32(0xFF))
    g = wp.uint8((packed_val >> wp.uint32(8)) & wp.uint32(0xFF))
    r = wp.uint8((packed_val >> wp.uint32(16)) & wp.uint32(0xFF))

    rgb[nworld_idx, cam_idx, pixel_idx, 0] = r
    rgb[nworld_idx, cam_idx, pixel_idx, 1] = g
    rgb[nworld_idx, cam_idx, pixel_idx, 2] = b


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
    # Validate all sensors use same resolution
    widths = {s.cfg.width for s in camera_sensors}
    heights = {s.cfg.height for s in camera_sensors}

    if len(widths) > 1 or len(heights) > 1:
      sensor_info = [(s.cfg.name, s.cfg.width, s.cfg.height) for s in camera_sensors]
      raise ValueError(
        f"All camera sensors must have the same resolution. Found: {sensor_info}"
      )

    # Merge rendering options (use most permissive)
    self.width = camera_sensors[0].cfg.width
    self.height = camera_sensors[0].cfg.height
    render_rgb = any("rgb" in s.cfg.type for s in camera_sensors)
    render_depth = any("depth" in s.cfg.type for s in camera_sensors)

    # Use first sensor's settings (could make this configurable)
    first_sensor = camera_sensors[0]
    use_textures = first_sensor.cfg.use_textures
    use_shadows = first_sensor.cfg.use_shadows
    enabled_geom_groups = list(first_sensor.cfg.enabled_geom_groups)

    # Create single RenderContext
    self._ctx = mjwarp.create_render_context(
      mjm=mj_model,
      m=model.struct,  # type: ignore
      d=data.struct,  # type: ignore
      width=self.width,
      height=self.height,
      use_textures=use_textures,
      use_shadows=use_shadows,
      render_rgb=render_rgb,
      render_depth=render_depth,
      enabled_geom_groups=enabled_geom_groups,
    )

    self._model = model
    self._data = data
    self._render_rgb = render_rgb
    self._render_depth = render_depth

    # Allocate unpacked RGB buffer if needed
    if render_rgb:
      # Determine device from model (MJWarp model tracks its device)
      wp_device = (
        model.struct.device  # type: ignore[attr-defined]
        if hasattr(model.struct, "device")  # type: ignore[attr-defined]
        else wp.get_device()
      )
      self._rgb_unpacked = wp.array4d(
        shape=(data.nworld, mj_model.ncam, self.height * self.width, 3),
        dtype=wp.uint8,
        device=wp_device,
      )
    else:
      self._rgb_unpacked = None

    # Link render manager back to sensors
    for sensor in camera_sensors:
      sensor.set_render_manager(self)

  def render(self) -> None:
    """Render all cameras (called once per scene.update)."""
    mjwarp.render(self._model, self._data, self._ctx)

    # Unpack RGB if needed
    if self._render_rgb and self._rgb_unpacked is not None:
      wp.launch(
        unpack_rgb_kernel,
        dim=(self._data.nworld, self._ctx.ncam, self.height * self.width),
        inputs=[self._ctx.pixels, self.width, self.height],
        outputs=[self._rgb_unpacked],
        device=self._rgb_unpacked.device,
      )

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    """Get RGB data for a specific camera.

    Args:
        cam_idx: Camera index

    Returns:
        RGB tensor of shape (nworld, height, width, 3) with dtype uint8
    """
    assert self._rgb_unpacked is not None, "RGB rendering not enabled"
    rgb_flat = wp.to_torch(self._rgb_unpacked[:, cam_idx, :, :])
    return rgb_flat.reshape(self._data.nworld, self.height, self.width, 3)

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    """Get depth data for a specific camera.

    Args:
        cam_idx: Camera index

    Returns:
        Depth tensor of shape (nworld, height, width, 1) with dtype float32
    """
    assert self._render_depth, "Depth rendering not enabled"
    depth_flat = wp.to_torch(self._ctx.depth[:, cam_idx, :])
    return depth_flat.reshape(self._data.nworld, self.height, self.width, 1)
