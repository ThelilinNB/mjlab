"""Base sensor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


T = TypeVar("T")


@dataclass(kw_only=True)
class SensorCfg(ABC):
  """Base configuration for a sensor."""

  name: str
  update_period: float = 0.0
  """Sensor update period, in seconds.

  Controls how frequently the sensor reads fresh data based on accumulated time.
  For example, update_period=0.033 means ~30 Hz regardless of simulation rate.
  Set to 0.0 to update every step (default).
  """

  @abstractmethod
  def build(self) -> Sensor[Any]:
    """Build sensor instance from this config."""
    raise NotImplementedError


class Sensor(ABC, Generic[T]):
  """Base sensor interface with typed data.

  Type parameter T specifies the type of data returned by the sensor. For example:
  - Sensor[torch.Tensor] for sensors returning raw tensors
  - Sensor[ContactData] for sensors returning structured contact data
  """

  def __init__(self, update_period: float = 0.0) -> None:
    self._timestamp = 0.0
    self._next_update_time = 0.0
    self._is_outdated = True
    self._cached_data: T | None = None
    self._update_period = update_period

  @abstractmethod
  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    """Edit the scene spec to add this sensor.

    This is called during scene construction to add sensor elements
    to the MjSpec.

    Args:
      scene_spec: The scene MjSpec to edit.
      entities: Dictionary of entities in the scene, keyed by name.
    """
    raise NotImplementedError

  @abstractmethod
  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    """Initialize the sensor after model compilation.

    This is called after the MjSpec is compiled into an MjModel and the simulation
    is ready to run. Use this to cache sensor indices, allocate buffers, etc.

    Args:
      mj_model: The compiled MuJoCo model.
      model: The mjwarp model wrapper.
      data: The mjwarp data arrays.
      device: Device for tensor operations (e.g., "cuda", "cpu").
    """
    raise NotImplementedError

  @property
  def data(self) -> T:
    """Get the current sensor data.

    This property returns cached data if the sensor is not yet outdated based on
    update_period. Otherwise, it recomputes the data by calling _read().

    Returns:
      The sensor data in the format specified by type parameter T.
    """
    # When update_period is 0, always get fresh data (no caching).
    if self._update_period == 0.0:
      return self._read()

    # With update_period > 0, use caching and only refresh when outdated.
    if self._is_outdated:
      self._cached_data = self._read()
      self._is_outdated = False
    assert self._cached_data is not None, "Sensor data not initialized"
    return self._cached_data

  @abstractmethod
  def _read(self) -> T:
    """Read and return fresh sensor data.

    This method should be implemented by subclasses to read the actual
    sensor data. It is only called when the sensor is outdated based on
    update_period timing.

    Returns:
      Fresh sensor data in the format specified by type parameter T.
    """
    raise NotImplementedError

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset sensor state for specified environments.

    Marks sensor as outdated so it will update on next data access.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    # Reset timestamps and mark as outdated.
    self._timestamp = 0.0
    self._next_update_time = 0.0
    self._is_outdated = True

    # Hook for subclasses to add custom reset logic.
    self._on_reset(env_ids)

  def _on_reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Override this in subclasses to add custom reset logic.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    pass

  def update(self, dt: float) -> None:
    """Update sensor timestamp and check if data needs refresh.

    This method updates the internal timestamp and marks the sensor as outdated
    if enough time has passed based on update_period.

    Args:
      dt: Time step in seconds.
    """
    self._timestamp += dt

    # Check if sensor needs an update.
    if self._update_period > 0.0:
      if self._timestamp >= self._next_update_time:
        self._is_outdated = True
        self._next_update_time = self._timestamp + self._update_period
    else:
      assert self._update_period == 0.0  # Update every step.
      self._is_outdated = True

    # Hook for subclasses to add custom update logic.
    self._on_update(dt)

  def _on_update(self, dt: float) -> None:
    """Override this in subclasses to add custom update logic.

    Args:
      dt: Time step in seconds.
    """
    pass
