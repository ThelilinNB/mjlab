"""Cartpole balance task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg, mdp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.cartpole.cartpole_constants import CARTPOLE_ROBOT_CFG
from mjlab.terrains import TerrainImporterCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import sample_uniform
from mjlab.viewer import ViewerConfig

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(
    terrain_type="plane",
  ),
  num_envs=512,
  extent=1.0,
  entities={"robot": CARTPOLE_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="pole",
  distance=3.0,
  elevation=10.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.02,
    iterations=1,
  ),
)


def compute_upright_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.sim.data.qpos[:, 1].cos()


def compute_center_reward(env: ManagerBasedRlEnv, std: float = 0.3) -> torch.Tensor:
  return torch.exp(-(env.sim.data.qpos[:, 0] ** 2) / (2 * std**2))


def compute_effort_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -(env.sim.data.actuator_force[:, 0] ** 2)


def check_pole_tipped(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.sim.data.qpos[:, 1].abs() > math.radians(30)


def random_push_cart(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  force_range: tuple[float, float] = (-5, 5),
) -> None:
  n = len(env_ids)
  random_forces = (
    torch.rand(n, device=env.device) * (force_range[1] - force_range[0])
    + force_range[0]
  )
  env.sim.data.qfrc_applied[env_ids, 0] = random_forces


def reset_pole(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  # Cart position is in -2, 2.
  env.sim.data.qpos[env_ids, 0] = sample_uniform(-2, 2, len(env_ids), device=env.device)
  # Pole angle is in -0.1, 0.1 radians.
  env.sim.data.qpos[env_ids, 1] = sample_uniform(
    -0.1, 0.1, len(env_ids), device=env.device
  )
  # Reset velocities.
  env.sim.data.qvel[env_ids, :] = 0.0


OBSERVATIONS_CFG = {
  "policy": ObservationGroupCfg(
    terms={
      "cart_pos": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 0:1]),
      "angle": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 1:2]),
      "cart_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 0:1]),
      "ang_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 1:2]),
    },
    concatenate_terms=True,
  ),
}

ACTIONS_CFG: dict[str, ActionTermCfg] = {
  "joint_effort": mdp.JointEffortActionCfg(
    asset_name="robot",
    actuator_names=(".*",),
    scale=1.0,
  )
}

REWARDS_CFG = {
  "upright": RewardTermCfg(func=compute_upright_reward, weight=5.0),
  "center": RewardTermCfg(func=compute_center_reward, weight=1.0),
  "effort": RewardTermCfg(func=compute_effort_penalty, weight=1e-2),
}

TERMINATIONS_CFG = {
  "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  "tipped": TerminationTermCfg(func=check_pole_tipped, time_out=False),
}

EVENTS_CFG = {
  "reset_robot_joints": EventTermCfg(func=reset_pole, mode="reset"),
  "randomize_mass": EventTermCfg(
    func=mdp.randomize_field,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "operation": "abs",
      "field": "body_mass",
      "ranges": (0.8, 1.2),
    },
  ),
  "random_push": EventTermCfg(
    func=random_push_cart,
    mode="interval",
    interval_range_s=(1.0, 2.0),
    params={"force_range": (-20.0, 20.0)},
  ),
}

CARTPOLE_ENV_CFG = ManagerBasedRlEnvCfg(
  scene=SCENE_CFG,
  observations=OBSERVATIONS_CFG,
  actions=ACTIONS_CFG,
  rewards=REWARDS_CFG,
  terminations=TERMINATIONS_CFG,
  events=EVENTS_CFG,
  sim=SIM_CFG,
  viewer=VIEWER_CONFIG,
  decimation=1,
  episode_length_s=20.0,
)

CARTPOLE_RL_CFG = RslRlOnPolicyRunnerCfg(
  experiment_name="cartpole_balance",
  max_iterations=500,
  obs_groups={"policy": ("policy",), "critic": ("policy",)},
)
