from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg
from mjlab.sim.sim_data import TorchArray

if TYPE_CHECKING:
  from mjlab.sensor.render_manager import RenderManager

CameraDataType = Literal["rgb", "depth"]


@dataclass
class CameraSpec:
  """Specification for creating a new camera."""

  name: str
  """Name of the camera."""

  pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
  """Position of the camera in world coordinates."""

  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  """Orientation quaternion (w, x, y, z)."""

  fovy: float = 45.0
  """Vertical field of view in degrees."""

  ipd: float = 0.068
  """Inter-pupillary distance for stereo rendering."""


@dataclass
class CameraSensorCfg(SensorCfg):
  """Configuration for a camera sensor (one sensor per camera)."""

  # Camera identification (exactly one must be specified)
  camera_name: str | None = None
  """Name of existing camera in XML to wrap."""

  camera_spec: CameraSpec | None = None
  """Spec for creating a new camera via edit_spec."""

  # Resolution (must match across all camera sensors in scene)
  width: int = 640
  height: int = 480

  # Output types
  type: tuple[CameraDataType, ...] = ("rgb",)

  # Rendering options (shared across all cameras via RenderManager)
  use_textures: bool = True
  use_shadows: bool = True
  enabled_geom_groups: tuple[int, ...] = (0, 1, 2)

  def __post_init__(self) -> None:
    if self.camera_name is None and self.camera_spec is None:
      raise ValueError("Must specify either camera_name or camera_spec")
    if self.camera_name is not None and self.camera_spec is not None:
      raise ValueError("Cannot specify both camera_name and camera_spec")

  def build(self) -> CameraSensor:
    return CameraSensor(self)


@dataclass
class CameraSensorData:
  """Data structure for camera sensor data."""

  output: dict[str, dict[str, torch.Tensor | TorchArray]]
  """Structure: {camera_name: {"rgb": tensor, "depth": tensor}}"""


class CameraSensor(Sensor[CameraSensorData]):
  """Camera sensor for a single camera."""

  def __init__(self, cfg: CameraSensorCfg) -> None:
    self.cfg = cfg
    self._render_manager: RenderManager | None = None
    self._camera_idx: int = -1
    self._camera_name: str = ""

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    """Add camera to spec if configured."""
    if self.cfg.camera_spec is not None:
      spec = self.cfg.camera_spec
      scene_spec.worldbody.add_camera(
        name=spec.name,
        pos=spec.pos,
        quat=spec.quat,
        fovy=spec.fovy,
        ipd=spec.ipd,
      )

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    """Find the camera index."""
    # Determine which camera to use
    if self.cfg.camera_spec is not None:
      target_name = self.cfg.camera_spec.name
    elif self.cfg.camera_name is not None:
      target_name = self.cfg.camera_name
    else:
      raise ValueError(
        f"CameraSensor '{self.cfg.name}' must specify either camera_name or camera_spec"
      )

    # Find camera index using mj_model.camera()
    try:
      self._camera_idx = mj_model.camera(target_name).id
      self._camera_name = target_name
    except KeyError as e:
      available = [mj_model.cam(i).name for i in range(mj_model.ncam)]
      raise ValueError(
        f"Camera '{target_name}' not found in model. Available cameras: {available}"
      ) from e

  def set_render_manager(self, render_manager: RenderManager) -> None:
    """Called by Scene after creating RenderManager."""
    self._render_manager = render_manager

  @property
  def data(self) -> CameraSensorData:
    """Read this camera's data from RenderManager."""
    assert self._render_manager is not None, "RenderManager not set"

    cam_output: dict[str, torch.Tensor | TorchArray] = {}

    if "rgb" in self.cfg.type:
      cam_output["rgb"] = self._render_manager.get_rgb(self._camera_idx)

    if "depth" in self.cfg.type:
      cam_output["depth"] = self._render_manager.get_depth(self._camera_idx)

    # Return data for single camera
    return CameraSensorData(output={self._camera_name: cam_output})


# if __name__ == "__main__":
#   # Test camera sensor with new API
#   camera_cfg = CameraSensorCfg(
#     name="test",
#     camera_name="robot/tracking",  # Use existing camera in XML
#     width=640,
#     height=480,
#     type=("rgb", "depth"),
#   )

#   from mjlab.tasks.tracking.config.g1 import unitree_g1_flat_tracking_env_cfg

#   cfg = unitree_g1_flat_tracking_env_cfg()

#   cfg.scene.sensors = cfg.scene.sensors + (camera_cfg,)

#   from mjlab.tasks.tracking.mdp import MotionCommandCfg

#   assert cfg.commands is not None
#   motion_cmd = cfg.commands["motion"]
#   assert isinstance(motion_cmd, MotionCommandCfg)
#   motion_cmd.motion_file = "artifacts/lafan_cartwheel:v0/motion.npz"

#   from mjlab.envs import ManagerBasedRlEnv

#   env = ManagerBasedRlEnv(cfg, device="cuda:0")
#   obs, _ = env.reset()

#   # Access camera data (renders have already been done in reset)
#   camera: CameraSensor = env.scene["test"]
#   data = camera.data

#   rgb = data.output["robot/tracking"]["rgb"]
#   depth = data.output["robot/tracking"]["depth"]

#   # Save rgb and depth to disk.
#   import numpy as np
#   from PIL import Image

#   # Convert to numpy and take first world
#   rgb_np = rgb[0].cpu().numpy().astype(np.uint8)
#   depth_np = depth[0].cpu().numpy().squeeze()

#   # Save RGB image
#   rgb_img = Image.fromarray(rgb_np)
#   rgb_img.save("camera_rgb.png")
#   print(f"Saved RGB image to camera_rgb.png (shape: {rgb_np.shape})")

#   # Normalize and save depth image (using fixed scale like mujoco_warp reference)
#   depth_scale = 5.0  # meters - adjust based on your scene scale
#   depth_normalized = np.clip(depth_np / depth_scale, 0.0, 1.0)
#   depth_img = Image.fromarray((depth_normalized * 255).astype(np.uint8))
#   depth_img.save("camera_depth.png")
#   print(f"Saved depth image to camera_depth.png (shape: {depth_np.shape})")
#   print(f"Depth range: [{depth_np.min():.3f}, {depth_np.max():.3f}]")
#   print(
#     f"Depth visualization scale: {depth_scale}m (values > {depth_scale}m clipped to white)"
#   )
