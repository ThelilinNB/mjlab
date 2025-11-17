"""Camera image visualization for Viser viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import viser

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor


class ViserCameraViewer:
  """Handles camera image visualization for the Viser viewer."""

  def __init__(
    self,
    server: viser.ViserServer,
    camera_sensor: CameraSensor,
  ):
    """Initialize the camera viewer.

    Args:
        server: The Viser server instance
        camera_sensor: The camera sensor to visualize
    """
    self._server = server
    self._camera_sensor = camera_sensor

    # Image handles
    self._rgb_handle: viser.GuiImageHandle | None = None
    self._depth_handle: viser.GuiImageHandle | None = None

    self._camera_name = camera_sensor.camera_name

    # Determine what types of images to display
    self._has_rgb = "rgb" in self._camera_sensor.cfg.type
    self._has_depth = "depth" in self._camera_sensor.cfg.type

    # Create image display handles
    if self._has_rgb:
      self._rgb_handle = self._server.gui.add_image(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        label=f"{self._camera_name}_rgb",
        format="jpeg",
      )

    if self._has_depth:
      self._depth_handle = self._server.gui.add_image(
        image=np.zeros((480, 640), dtype=np.uint8),
        label=f"{self._camera_name}_depth",
        format="jpeg",
      )

  def update(self, env_idx: int = 0) -> None:
    """Update the camera images for a single environment.

    Args:
        env_idx: Environment index to visualize
    """
    data = self._camera_sensor.data

    if self._has_rgb and self._rgb_handle is not None and data.rgb is not None:
      rgb_np = data.rgb[env_idx].cpu().numpy()
      self._rgb_handle.image = rgb_np

    if self._has_depth and self._depth_handle is not None and data.depth is not None:
      depth_np = data.depth[env_idx].squeeze().cpu().numpy()
      depth_scale = 5.0
      depth_normalized = np.clip(depth_np / depth_scale, 0.0, 1.0)
      depth_uint8 = (depth_normalized * 255).astype(np.uint8)
      self._depth_handle.image = depth_uint8

  def cleanup(self) -> None:
    """Clean up resources."""
    if self._rgb_handle is not None:
      self._rgb_handle.remove()
    if self._depth_handle is not None:
      self._depth_handle.remove()
