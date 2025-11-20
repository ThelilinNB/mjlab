# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for FFTAI robots.

The following configurations are available:

* :obj:`FFTAI_GR1T1_CFG`: FFTAI GR1T1 humanoid robot

Reference: https://github.com/FFTAI
"""

import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg


from legged_lab.assets import ISAAC_ASSET_DIR
##
# Configuration
##

Ymboy_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/ps/sh_robot/TienKung-Lab/TienKung-Lab/legged_lab/assets/ymbot_e/usd/ymboy-23dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        # joint_pos={
        #     ".*_hip_pitch_joint": -0.15,
        #     ".*_knee_joint": 0.35,
        #     ".*_ankle_pitch_joint": -0.22,
        #     ".*_elbow_joint": 0.87,
        #     "left_shoulder_roll_joint": 0.18,
        #     "left_shoulder_pitch_joint": 0.35,
        #     "right_shoulder_roll_joint": -0.18,
        #     "right_shoulder_pitch_joint": 0.35,
        # },
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 60.0,
                ".*_hip_roll_joint": 94.0,
                ".*_hip_pitch_joint": 94.0,
                ".*_knee_joint": 150.0,
                ".*waist_yaw_joint": 94.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 13.0,
                ".*_hip_roll_joint": 14.0,
                ".*_hip_pitch_joint": 14.0,
                ".*_knee_joint": 13.0,
                ".*waist_yaw_joint": 14.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 75.0,
                ".*_hip_roll_joint": 75.0,
                ".*_hip_pitch_joint":100.0,
                ".*_knee_joint": 100.0,
                ".*waist_yaw_joint": 100.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 60.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 8.0,
                ".*_ankle_roll_joint": 8.0,
            },
            stiffness=20.0, #150
            damping=2.0, #150
            armature=0.01,
        ),
        "shoulders": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 12.0,
                ".*_shoulder_roll_joint": 12.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 5.0,
                ".*_shoulder_roll_joint": 5.0,
            },
            stiffness=40.0, #5
            damping=10.0,   #2
            armature=0.01,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 12.0,
                ".*_elbow_joint": 12.0,
                ".*_wrist_yaw_joint": 12.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 5.0,
                ".*_elbow_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            stiffness=40.0, #5
            damping=10.0,  #2
            armature=0.01,

        ),
    },
)

Ymboy13_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/ymbot/ymboy-13dof/ymboy_13dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_knee_joint": 0.35,
            ".*_ankle_pitch_joint": -0.22,
            # ".*_elbow_joint": 0.87,
            # "left_shoulder_roll_joint": 0.18,
            # "left_shoulder_pitch_joint": 0.35,
            # "right_shoulder_roll_joint": -0.18,
            # "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 60.0,
                ".*_hip_roll_joint": 94.0,
                ".*_hip_pitch_joint": 94.0,
                ".*_knee_joint": 150.0,
                ".*waist_yaw_joint": 94.0,
                # ".*waist_pitch_joint": 94.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 13.0,
                ".*_hip_roll_joint": 14.0,
                ".*_hip_pitch_joint": 14.0,
                ".*_knee_joint": 13.0,
                ".*waist_yaw_joint": 14.0,
                # ".*waist_pitch_joint": 14.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 75.0,
                ".*_hip_roll_joint": 75.0,
                ".*_hip_pitch_joint":100.0,
                ".*_knee_joint": 100.0,
                ".*waist.*": 100.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*waist.*": 5,
            },
            armature=0.01,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 60.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 8.0,
                ".*_ankle_roll_joint": 8.0,
            },
            stiffness=40.0, #150
            damping=4.0, #150
            armature=0.01,
        ),
        # "shoulders": IdealPDActuatorCfg(
        #     joint_names_expr=[
        #         ".*_shoulder_pitch_joint",
        #         ".*_shoulder_roll_joint",
        #     ],
        #     effort_limit_sim={
        #         ".*_shoulder_pitch_joint": 12.0,
        #         ".*_shoulder_roll_joint": 12.0,
        #     },
        #     velocity_limit_sim={
        #         ".*_shoulder_pitch_joint": 5.0,
        #         ".*_shoulder_roll_joint": 5.0,
        #     },
        #     stiffness=150.0, #5
        #     damping=10.0,   #2
        #     armature=0.01,
        # ),
        # "arms": IdealPDActuatorCfg(
        #     joint_names_expr=[
        #         ".*_shoulder_yaw_joint",
        #         ".*_elbow_joint",
        #     ],
        #     effort_limit_sim={
        #         ".*_shoulder_yaw_joint": 12.0,
        #         ".*_elbow_joint": 12.0,
        #     },
        #     velocity_limit_sim={
        #         ".*_shoulder_yaw_joint": 5.0,
        #         ".*_elbow_joint": 5.0,
        #     },
        #     stiffness=150.0, #5
        #     damping=10.0,  #2
        #     armature=0.01,
        #
        # ),
    },
)

Ymgirl_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/ymbot/ymgirl/usd/ymgirl.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),
        joint_pos={
            ".*_hip_pitch_joint": -0.14,
            ".*knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist_.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 90.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 360.0,
                ".*_knee_joint": 360.0,
                ".*waist_yaw_joint": 360.0,

            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 14.0,
                ".*_hip_roll_joint": 13.0,
                ".*_hip_pitch_joint": 10.0,
                ".*_knee_joint": 10.0,
                ".*waist_yaw_joint": 10.0,

            },
            stiffness={
                ".*_hip_yaw_joint": 300.0,#
                ".*_hip_roll_joint": 300.0,#
                ".*_hip_pitch_joint": 300.0,#
                ".*_knee_joint": 340.0,#
                ".*waist_.*": 320.0,#
            },
            damping={
                ".*_hip_yaw_joint": 10.0,#5
                ".*_hip_roll_joint": 18.0,#9
                ".*_hip_pitch_joint": 12.0,#6
                ".*_knee_joint": 14.0,#7
                ".*waist_.*": 12,#6
            },
            armature=0.01,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 100.0,
                ".*_ankle_roll_joint": 100.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint":13.0,
                ".*_ankle_roll_joint": 13.0,
            },
            stiffness={
                ".*_ankle_pitch_joint":320.0,#320
                ".*_ankle_roll_joint": 300.0,#300
            },
            damping={
                ".*_ankle_pitch_joint":14.0,#14
                ".*_ankle_roll_joint": 14.0,#14
            },
            armature=0.01,
        ),
        # "arms": IdealPDActuatorCfg(
        #     joint_names_expr=[
        #         ".*_shoulder_pitch_joint",
        #         ".*_shoulder_roll_joint",
        #         ".*_shoulder_yaw_joint",
        #         ".*_elbow_joint",
        #     ],
        #     effort_limit_sim={
        #         ".*_shoulder_pitch_joint": 12.0,
        #         ".*_shoulder_roll_joint": 12.0,
        #         ".*_shoulder_yaw_joint": 12.0,
        #         ".*_elbow_joint": 12.0,
        #     },
        #     velocity_limit_sim={
        #         ".*_shoulder_pitch_joint": 5.0,
        #         ".*_shoulder_roll_joint": 5.0,
        #         ".*_shoulder_yaw_joint": 5.0,
        #         ".*_elbow_joint": 5.0,
        #     },
        #     stiffness={
        #         ".*_shoulder_pitch_joint": 150.0,
        #         ".*_shoulder_roll_joint": 150.0,
        #         ".*_shoulder_yaw_joint": 150.0,
        #         ".*_elbow_joint": 150.0,
        #     },
        #     damping={
        #         ".*_shoulder_pitch_joint": 10.0,
        #         ".*_shoulder_roll_joint": 10.0,
        #         ".*_shoulder_yaw_joint": 10.0,
        #         ".*_elbow_joint": 10.0,
        #     },
        # ),
    },
)


