# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR
from robolab.assets.rpo_motor_catalog import (
    ANKLE_MOTOR_CHOICES,
    DEFAULT_ANKLE_MOTOR,
    DEFAULT_KNEE_MOTOR,
    KNEE_MOTOR_CHOICES,
    get_motor_spec,
    load_torque_speed_envelope,
)
from robolab.actuators import TorqueSpeedEnvelopeActuatorCfg

RPO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/urdf/rpo.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_thigh_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.06,
            "left_elbow_pitch_joint": 0.78,
            "right_thigh_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.18,
            "right_arm_roll_joint": -0.06,
            "right_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*torso.*",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.3,
                ".*_thigh_roll_joint": 3.3,
                ".*_thigh_pitch_joint": 3.3,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "knees": DelayedPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness=150.0,
            damping=5.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


def configure_rpo_motors(
    robot_cfg: ArticulationCfg,
    knee_motor: str = DEFAULT_KNEE_MOTOR,
    ankle_motor: str = DEFAULT_ANKLE_MOTOR,
    knee_envelope_csv: str | None = None,
    ankle_envelope_csv: str | None = None,
) -> dict[str, dict[str, object]]:
    """Apply selected knee and common ankle actuator models to ``robot_cfg``.

    If a selected motor has a torque-speed CSV (or an override CSV is passed),
    its delayed PD output is clipped by that curve at every simulation step.
    URDF geometry, mass, center of mass, and rigid-body inertia remain
    untouched.  Joint-side armature follows the selected motor specification.
    """

    knee = get_motor_spec(knee_motor, KNEE_MOTOR_CHOICES)
    ankle = get_motor_spec(ankle_motor, ANKLE_MOTOR_CHOICES)
    dynamic_envelope: dict[str, bool] = {}

    for actuator_name, motor, legacy_id, csv_override in (
        ("knees", knee, DEFAULT_KNEE_MOTOR, knee_envelope_csv),
        ("feet", ankle, DEFAULT_ANKLE_MOTOR, ankle_envelope_csv),
    ):
        actuator_cfg = robot_cfg.actuators[actuator_name]
        actuator_cfg.effort_limit_sim = motor.peak_torque_nm
        actuator_cfg.velocity_limit_sim = motor.max_speed_rad_s
        actuator_cfg.armature = motor.armature_kgm2
        # DelayedPDActuator is explicit: effort_limit clips its computed torque,
        # while effort_limit_sim only constrains the physics solver.  Set both
        # for real candidates.  None preserves the old URDF-resolved behavior.
        actuator_cfg.effort_limit = None if motor.motor_id == legacy_id else motor.peak_torque_nm
        actuator_cfg.velocity_limit = None if motor.motor_id == legacy_id else motor.max_speed_rad_s

        envelope = load_torque_speed_envelope(motor, csv_override)
        dynamic_envelope[actuator_name] = envelope is not None
        if envelope is not None:
            robot_cfg.actuators[actuator_name] = TorqueSpeedEnvelopeActuatorCfg(
                joint_names_expr=actuator_cfg.joint_names_expr,
                effort_limit=motor.peak_torque_nm,
                velocity_limit=motor.max_speed_rad_s,
                effort_limit_sim=motor.peak_torque_nm,
                velocity_limit_sim=motor.max_speed_rad_s,
                stiffness=actuator_cfg.stiffness,
                damping=actuator_cfg.damping,
                armature=motor.armature_kgm2,
                friction=actuator_cfg.friction,
                dynamic_friction=actuator_cfg.dynamic_friction,
                viscous_friction=actuator_cfg.viscous_friction,
                min_delay=actuator_cfg.min_delay,
                max_delay=actuator_cfg.max_delay,
                torque_speed_lookup=envelope,
            )

    result = {"knee": knee.as_dict(), "ankle": ankle.as_dict()}
    result["knee"]["dynamic_envelope"] = dynamic_envelope["knees"]
    result["ankle"]["dynamic_envelope"] = dynamic_envelope["feet"]
    return result


RPO_LINKS = [
    "base_link",
    "left_thigh_yaw_link",
    "left_thigh_roll_link",
    "left_thigh_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_thigh_yaw_link",
    "right_thigh_roll_link",
    "right_thigh_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_arm_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_arm_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
]
