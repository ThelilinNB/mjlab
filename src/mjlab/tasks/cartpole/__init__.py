"""CartPole balancing task."""

from mjlab.tasks.cartpole.cartpole_env_cfg import CARTPOLE_ENV_CFG, CARTPOLE_RL_CFG
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Cartpole-Balance",
  env_cfg=CARTPOLE_ENV_CFG,
  rl_cfg=CARTPOLE_RL_CFG,
)
