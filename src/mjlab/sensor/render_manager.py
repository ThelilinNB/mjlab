from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor


@wp.kernel
def _unpack_rgb_kernel(
  packed: wp.array2d(dtype=wp.uint32),  # type: ignore
  rgb: wp.array3d(dtype=wp.uint8),  # type: ignore
):
  world_idx, pixel_idx = wp.tid()  # type: ignore
  b = wp.uint8(packed[world_idx, pixel_idx] & wp.uint32(0xFF))
  g = wp.uint8((packed[world_idx, pixel_idx] >> wp.uint32(8)) & wp.uint32(0xFF))
  r = wp.uint8((packed[world_idx, pixel_idx] >> wp.uint32(16)) & wp.uint32(0xFF))
  rgb[world_idx, pixel_idx, 0] = r
  rgb[world_idx, pixel_idx, 1] = g
  rgb[world_idx, pixel_idx, 2] = b


class RenderManager:
  """Manages rendering for all camera sensors in a scene.

  The RenderManager coordinates rendering across multiple camera sensors using
  a two-level control system:

  1. **Static enablement (cam_user[0])**: Cameras with cam_user[0] == 1.0 are
     included in the render context at initialization. This determines the set
     of cameras that CAN be rendered.

  2. **Dynamic toggle (render_rgb/render_depth)**: On each render() call, these
     flags control which enabled cameras ACTUALLY render based on their
     update_period, allowing efficient selective rendering.

  Update flow:
    - Sensors mark themselves as outdated based on their update_period
    - When sensor.data is accessed, it checks if outdated and calls _read()
    - _read() calls RenderManager.get_rgb()/get_depth()
    - RenderManager.render() checks which sensors are outdated and sets flags
    - Only outdated sensors render; outputs are cached until next update

  Important constraints:
    - All cameras must share the same render settings (textures, shadows, geom groups)
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    camera_sensors: list[CameraSensor],
  ):
    self.wp_device = (
      model.struct.device  # type: ignore
      if hasattr(model.struct, "device")  # type: ignore
      else wp.get_device()
    )

    # Sort sensors by camera index to match mujoco_warp ordering.
    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    camera_resolutions = [(s.cfg.width, s.cfg.height) for s in self.camera_sensors]
    render_rgb = ["rgb" in s.cfg.type for s in self.camera_sensors]
    render_depth = ["depth" in s.cfg.type for s in self.camera_sensors]

    # Create mapping from MuJoCo camera ID to sorted list index.
    self._cam_idx_to_list_idx = {
      s.camera_idx: idx for idx, s in enumerate(self.camera_sensors)
    }

    # Validate that all sensors have consistent rendering settings.
    # mujoco_warp's render context is shared across all cameras, so they must agree.
    first_sensor = self.camera_sensors[0]
    use_textures = first_sensor.cfg.use_textures
    use_shadows = first_sensor.cfg.use_shadows
    enabled_geom_groups = list(first_sensor.cfg.enabled_geom_groups)

    for sensor in self.camera_sensors[1:]:
      if sensor.cfg.use_textures != use_textures:
        raise ValueError(
          f"Camera sensor '{sensor.cfg.name}' has use_textures={sensor.cfg.use_textures}, "
          f"but '{first_sensor.cfg.name}' has use_textures={use_textures}. "
          "All camera sensors must have the same use_textures setting."
        )
      if sensor.cfg.use_shadows != use_shadows:
        raise ValueError(
          f"Camera sensor '{sensor.cfg.name}' has use_shadows={sensor.cfg.use_shadows}, "
          f"but '{first_sensor.cfg.name}' has use_shadows={use_shadows}. "
          "All camera sensors must have the same use_shadows setting."
        )
      if tuple(sensor.cfg.enabled_geom_groups) != tuple(enabled_geom_groups):
        raise ValueError(
          f"Camera sensor '{sensor.cfg.name}' has enabled_geom_groups="
          f"{sensor.cfg.enabled_geom_groups}, but '{first_sensor.cfg.name}' has "
          f"enabled_geom_groups={tuple(enabled_geom_groups)}. "
          "All camera sensors must have the same enabled_geom_groups setting."
        )

    with wp.ScopedDevice(self.wp_device):
      self._ctx = mjwarp.create_render_context(
        mjm=mj_model,
        m=model.struct,  # type: ignore
        d=data.struct,  # type: ignore
        cam_resolutions=camera_resolutions,
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

    if any(render_rgb):
      self._rgb_unpacked = wp.array3d(
        shape=(data.nworld, self._ctx.rgb_data.shape[1], 3),
        dtype=wp.uint8,
        device=self.wp_device,
      )
    else:
      self._rgb_unpacked = None

    for sensor in self.camera_sensors:
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

  def render(self, dt: float = 0.0) -> None:
    """Execute rendering for all cameras that need updates.

    This method is called by the scene/simulation loop. It checks which sensors
    are outdated (based on their update_period) and only renders those cameras.

    The render context maintains toggle flags (render_rgb, render_depth) that
    control which cameras actually render on this call, enabling efficient
    selective rendering when sensors have different update rates.

    Optimization: If no cameras need updates, the entire render call is skipped.

    Args:
      dt: Time delta (currently unused, kept for API compatibility).
    """
    del dt

    # Build toggle arrays based on which sensors need updating.
    any_render_needed = False
    for idx, sensor in enumerate(self.camera_sensors):
      should_render = sensor._is_outdated
      render_rgb = should_render and ("rgb" in sensor.cfg.type)
      render_depth = should_render and ("depth" in sensor.cfg.type)

      self._ctx.render_rgb[idx] = render_rgb
      self._ctx.render_depth[idx] = render_depth

      if render_rgb or render_depth:
        any_render_needed = True

    # Skip rendering entirely if no cameras need updates.
    if not any_render_needed:
      return

    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.render_graph is not None:
        wp.capture_launch(self.render_graph)
      else:
        mjwarp.render(self._model, self._data, self._ctx)

    if self._render_rgb and self._rgb_unpacked is not None:
      wp.launch(
        _unpack_rgb_kernel,
        dim=(self._data.nworld, self._ctx.rgb_data.shape[1]),
        inputs=[self._ctx.rgb_data],
        outputs=[self._rgb_unpacked],
        device=self.wp_device,
      )

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    """Get RGB image data for a specific camera.

    Args:
      cam_idx: MuJoCo camera ID (not the index in the sensor list).

    Returns:
      RGB image tensor of shape [num_envs, height, width, 3] (uint8).

    Raises:
      RuntimeError: If RGB rendering is not enabled for this RenderManager.
      KeyError: If cam_idx is not a valid camera ID in this RenderManager.
    """
    if self._rgb_unpacked is None:
      raise RuntimeError(
        "RGB rendering is not enabled. Ensure at least one camera sensor has "
        "'rgb' in its type configuration."
      )

    if cam_idx not in self._cam_idx_to_list_idx:
      available = list(self._cam_idx_to_list_idx.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in RenderManager. "
        f"Available camera IDs: {available}"
      )

    # Map MuJoCo camera ID to sorted list index.
    list_idx = self._cam_idx_to_list_idx[cam_idx]
    rgb_unpacked_torch = wp.to_torch(self._rgb_unpacked)
    start = int(self._rgb_adr[list_idx])
    size = int(self._rgb_size[list_idx])
    rgb_flat = rgb_unpacked_torch[:, start : start + size]
    return rgb_flat.reshape(
      self._data.nworld,
      self.camera_sensors[list_idx].cfg.height,
      self.camera_sensors[list_idx].cfg.width,
      3,
    )

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    """Get depth image data for a specific camera.

    Args:
      cam_idx: MuJoCo camera ID (not the index in the sensor list).

    Returns:
      Depth image tensor of shape [num_envs, height, width, 1] (float32).

    Raises:
      RuntimeError: If depth rendering is not enabled for this RenderManager.
      KeyError: If cam_idx is not a valid camera ID in this RenderManager.
    """
    if not any(self._render_depth):
      raise RuntimeError(
        "Depth rendering is not enabled. Ensure at least one camera sensor has "
        "'depth' in its type configuration."
      )

    if cam_idx not in self._cam_idx_to_list_idx:
      available = list(self._cam_idx_to_list_idx.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in RenderManager. "
        f"Available camera IDs: {available}"
      )

    # Map MuJoCo camera ID to sorted list index.
    list_idx = self._cam_idx_to_list_idx[cam_idx]
    depth_torch = wp.to_torch(self._ctx.depth_data)
    start = int(self._depth_adr[list_idx])
    size = int(self._depth_size[list_idx])
    depth_flat = depth_torch[:, start : start + size]
    return depth_flat.reshape(
      self._data.nworld,
      self.camera_sensors[list_idx].cfg.height,
      self.camera_sensors[list_idx].cfg.width,
      1,
    )
