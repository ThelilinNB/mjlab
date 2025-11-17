from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
  from mjlab.sensor.render_manager import RenderManager

CameraDataType = Literal["rgb", "depth"]


@dataclass
class CameraSensorCfg(SensorCfg):
  camera_name: str | None = None
  pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  fovy: float = 45
  width: int = 160
  height: int = 120
  type: tuple[CameraDataType, ...] = ("rgb",)
  use_textures: bool = True
  use_shadows: bool = False
  enabled_geom_groups: tuple[int, ...] = (0, 1, 2)

  def build(self) -> CameraSensor:
    return CameraSensor(self)


@dataclass
class CameraSensorData:
  """Camera sensor output data.

  Contains RGB and/or depth images based on sensor configuration.
  Both fields are None if not requested in the sensor's type configuration.

  Shape: [num_envs, height, width, channels]
    - rgb: channels=3 (uint8)
    - depth: channels=1 (float32)
  """

  rgb: torch.Tensor | None = None
  """RGB image data [num_envs, height, width, 3] (uint8), or None if not enabled."""
  depth: torch.Tensor | None = None
  """Depth image data [num_envs, height, width, 1] (float32), or None if not enabled."""


class CameraSensor(Sensor[CameraSensorData]):
  def __init__(self, cfg: CameraSensorCfg) -> None:
    super().__init__(cfg.update_period)
    self.cfg = cfg
    self._camera_name = cfg.camera_name if cfg.camera_name is not None else cfg.name
    self._is_wrapping_existing = cfg.camera_name is not None
    self._render_manager: RenderManager | None = None
    self._camera_idx: int = -1

  @property
  def camera_name(self) -> str:
    """The name of the MuJoCo camera this sensor wraps."""
    return self._camera_name

  @property
  def camera_idx(self) -> int:
    """The MuJoCo camera ID (index in the compiled model)."""
    return self._camera_idx

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities

    # nuser_cam is set to 1 in scene.xml to ensure all cameras have user data allocated.

    if self._is_wrapping_existing:
      return

    scene_spec.worldbody.add_camera(
      name=self.cfg.name,
      pos=self.cfg.pos,
      quat=self.cfg.quat,
      fovy=self.cfg.fovy,
      userdata=[1.0],
    )

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    """Initialize the camera sensor after model compilation.

    This resolves the camera name to its MuJoCo ID and enables it for rendering
    by setting cam_user[0] = 1.0.

    Design note: cam_user[0] acts as an enable/disable flag for cameras in the
    render context. The RenderManager creates a render context for ALL cameras
    with cam_user[0] == 1.0, then uses render_rgb/render_depth flags to control
    which cameras actually render on each frame based on their update_period.
    """
    del model, data, device

    try:
      cam = mj_model.camera(self._camera_name)
      self._camera_idx = cam.id

      # Ensure camera has user data allocated (should be set in scene.xml).
      if mj_model.cam_user.shape[1] == 0:
        raise ValueError(
          f"Camera '{self._camera_name}' requires user data, but nuser_cam=0. "
          "This should not happen - nuser_cam=1 is set in scene.xml."
        )

      # Enable camera for the render context (cam_user[0] = 1.0 means "include me").
      # For wrapped cameras, this overrides whatever was in the XML.
      # For new cameras, this was already set in edit_spec.
      mj_model.cam_user[self._camera_idx, 0] = 1.0
    except KeyError as e:
      available = [mj_model.cam(i).name for i in range(mj_model.ncam)]
      raise ValueError(
        f"Camera '{self._camera_name}' not found in model. Available: {available}"
      ) from e

  def set_render_manager(self, render_manager: RenderManager) -> None:
    self._render_manager = render_manager

  def _read(self) -> CameraSensorData:
    if self._render_manager is None:
      raise RuntimeError(
        f"Camera sensor '{self.cfg.name}' has not been initialized with a RenderManager. "
        "This should be set automatically during scene initialization."
      )

    rgb_data = None
    depth_data = None

    if "rgb" in self.cfg.type:
      rgb_data = self._render_manager.get_rgb(self._camera_idx)
    if "depth" in self.cfg.type:
      depth_data = self._render_manager.get_depth(self._camera_idx)

    return CameraSensorData(rgb=rgb_data, depth=depth_data)
