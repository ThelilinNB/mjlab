"""Test camera sensor implementation."""

import numpy as np
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.tracking.config.g1 import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

# Setup camera sensor
camera_cfg = CameraSensorCfg(
  name="test",
  camera_name="robot/tracking",  # Use existing camera in XML
  width=640,
  height=480,
  type=("rgb", "depth"),
)

# Setup environment
cfg = unitree_g1_flat_tracking_env_cfg()
cfg.scene.sensors = cfg.scene.sensors + (camera_cfg,)

assert cfg.commands is not None
motion_cmd = cfg.commands["motion"]
assert isinstance(motion_cmd, MotionCommandCfg)
motion_cmd.motion_file = "artifacts/lafan_cartwheel:v0/motion.npz"

# Create environment and reset
env = ManagerBasedRlEnv(cfg, device="cuda:0")
obs, _ = env.reset()

# Access camera data (renders have already been done in reset)
camera = env.scene["test"]
data = camera.data

rgb = data.output["robot/tracking"]["rgb"]
depth = data.output["robot/tracking"]["depth"]

# Save rgb and depth to disk
# Convert to numpy and take first world
rgb_np = rgb[0].cpu().numpy().astype(np.uint8)
depth_np = depth[0].cpu().numpy().squeeze()

# Save RGB image
rgb_img = Image.fromarray(rgb_np)
rgb_img.save("camera_rgb.png")
print(f"Saved RGB image to camera_rgb.png (shape: {rgb_np.shape})")

# Normalize and save depth image
depth_scale = 5.0  # meters - adjust based on your scene scale
depth_normalized = np.clip(depth_np / depth_scale, 0.0, 1.0)
depth_img = Image.fromarray((depth_normalized * 255).astype(np.uint8))
depth_img.save("camera_depth.png")
print(f"Saved depth image to camera_depth.png (shape: {depth_np.shape})")
print(f"Depth range: [{depth_np.min():.3f}, {depth_np.max():.3f}]")
print(
  f"Depth visualization scale: {depth_scale}m (values > {depth_scale}m clipped to white)"
)
print("\nCamera sensor test completed successfully!")
