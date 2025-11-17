"""Tests for camera_sensor.py."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.camera_sensor import CameraSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture(scope="module")
def simple_world_xml():
  """XML for a simple world with a box."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="box" pos="0 0 0.5">
          <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
  """


@pytest.fixture(scope="module")
def world_with_camera_xml():
  """XML for world with existing camera."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="box" pos="0 0 0.5">
          <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
          <camera name="box_cam" pos="0 0 1" quat="1 0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
  """


def test_camera_sensor_rgb_output(simple_world_xml, device):
  """Verify camera sensor returns RGB data with correct shape."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    quat=(0.924, 0.0, 0.383, 0.0),
    width=64,
    height=48,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["test_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.rgb is not None
  assert data.depth is None
  assert data.rgb.shape == (2, 48, 64, 3)
  assert data.rgb.dtype == torch.uint8


def test_camera_sensor_depth_output(simple_world_xml, device):
  """Verify camera sensor returns depth data with correct shape."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("depth",),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["test_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.depth is not None
  assert data.rgb is None
  assert data.depth.shape == (2, 48, 64, 1)
  assert data.depth.dtype == torch.float32


def test_camera_sensor_rgb_and_depth(simple_world_xml, device):
  """Verify camera sensor returns both RGB and depth when configured."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb", "depth"),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["test_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.rgb is not None
  assert data.depth is not None
  assert data.rgb.shape == (1, 24, 32, 3)
  assert data.depth.shape == (1, 24, 32, 1)


def test_camera_sensor_wraps_existing(world_with_camera_xml, device):
  """Verify camera sensor can wrap an existing camera from XML."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(world_with_camera_xml)
  )

  # Wrap the existing "box_cam" camera.
  camera_cfg = CameraSensorCfg(
    name="wrapped_cam",
    camera_name="world/box_cam",
    width=64,
    height=48,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["wrapped_cam"]
  sim.step()
  scene.update(sim.mj_model.opt.timestep)
  data = sensor.data

  assert data.rgb is not None
  assert data.rgb.shape == (1, 48, 64, 3)


def test_multiple_cameras_with_different_resolutions(simple_world_xml, device):
  """Verify multiple cameras with different resolutions work correctly."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera1_cfg = CameraSensorCfg(
    name="cam1",
    pos=(2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("rgb",),
  )

  camera2_cfg = CameraSensorCfg(
    name="cam2",
    pos=(-2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera1_cfg, camera2_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor1 = scene["cam1"]
  sensor2 = scene["cam2"]

  sim.step()
  scene.update(sim.mj_model.opt.timestep)

  data1 = sensor1.data
  data2 = sensor2.data

  assert data1.rgb.shape == (2, 48, 64, 3)
  assert data2.rgb.shape == (2, 24, 32, 3)


def test_camera_update_period_caching(simple_world_xml, device):
  """Verify camera sensor respects update_period for caching."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  # Camera with 0.1s update period.
  camera_cfg = CameraSensorCfg(
    name="test_cam",
    pos=(2.0, 0.0, 1.0),
    width=32,
    height=24,
    type=("rgb",),
    update_period=0.1,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["test_cam"]
  dt = sim.mj_model.opt.timestep

  sim.step()
  scene.update(dt)

  # First access triggers rendering.
  assert sensor._is_outdated
  _ = sensor.data
  assert not sensor._is_outdated

  # Step many times (less than update_period).
  for _ in range(40):
    sim.step()
    scene.update(dt)

  # Should still be cached.
  assert not sensor._is_outdated

  # Step enough to exceed update_period.
  for _ in range(30):
    sim.step()
    scene.update(dt)

  # Should be outdated now.
  assert sensor._is_outdated


def test_error_on_mismatched_render_settings(simple_world_xml, device):
  """Verify error when cameras have inconsistent render settings."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera1_cfg = CameraSensorCfg(
    name="cam1",
    pos=(2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("rgb",),
    use_textures=True,
  )

  camera2_cfg = CameraSensorCfg(
    name="cam2",
    pos=(-2.0, 0.0, 1.0),
    width=64,
    height=48,
    type=("rgb",),
    use_textures=False,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera1_cfg, camera2_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  with pytest.raises(ValueError, match="use_textures"):
    scene.initialize(sim.mj_model, sim.model, sim.data)


def test_error_on_invalid_camera_name(simple_world_xml, device):
  """Verify error when wrapping nonexistent camera."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_world_xml))

  camera_cfg = CameraSensorCfg(
    name="bad_cam",
    camera_name="world/nonexistent_cam",
    width=64,
    height=48,
    type=("rgb",),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"world": entity_cfg},
    sensors=(camera_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=10)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  with pytest.raises(ValueError, match="not found in model"):
    scene.initialize(sim.mj_model, sim.model, sim.data)
