from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import ymboy_flat_env_cfg, ymboy_rough_env_cfg
from .rl_cfg import ymboy_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-ymboy",
  env_cfg=ymboy_rough_env_cfg(),
  play_env_cfg=ymboy_rough_env_cfg(play=True),
  rl_cfg=ymboy_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-ymboy",
  env_cfg=ymboy_flat_env_cfg(),
  play_env_cfg=ymboy_flat_env_cfg(play=True),
  rl_cfg=ymboy_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
