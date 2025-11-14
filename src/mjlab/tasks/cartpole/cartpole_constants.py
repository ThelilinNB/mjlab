"""CartPole robot configuration."""

import mujoco

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg

CARTPOLE_XML = """
<mujoco model="cartpole">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <worldbody>
    <body name="cartpole">
      <body name="cart" pos="0 0 0.5">
        <geom type="box" size="0.2 0.1 0.1" rgba="0.2 0.2 0.8 1" mass="1.0"/>
        <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
        <body name="pole" pos="0 0 0.1">
          <geom type="capsule" size="0.05 0.5" fromto="0 0 0 0 0 1" rgba="0.8 0.2 0.2 1" mass="2.0"/>
          <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide" joint="slide" ctrllimited="true" ctrlrange="-1 1" gear="20"/>
  </actuator>
</mujoco>
"""


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(CARTPOLE_XML)


CARTPOLE_ROBOT_CFG = EntityCfg(
  spec_fn=get_spec,
  articulation=EntityArticulationInfoCfg(
    actuators=(XmlMotorActuatorCfg(joint_names_expr=("slide",)),)
  ),
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  robot = Entity(CARTPOLE_ROBOT_CFG)
  viewer.launch(robot.spec.compile())
