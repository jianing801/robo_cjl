"""Inherit RPO-Flat unchanged except posture functions and fall contacts."""

import math

from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from robolab.tasks.direct.base.rpo_env_cfg import RPOFlatEnvCfg
from . import rewards


@configclass
class BackpackRPOFlatEnvCfg(RPOFlatEnvCfg):
    # Engineering fall proxy, not a desired lean angle. No speed/distance test.
    backpack_fall_tilt_rad: float = math.radians(60.0)

    def __post_init__(self):
        super().__post_init__()
        # Keep term names and weights for clean comparisons with old logs.
        self.reward.flat_orientation_l2.func = rewards.roll_orientation_l2
        self.reward.upward.func = rewards.roll_upward
        # RPO enables self-collision. Net forces on the backpack may therefore
        # include an arm touching it: only ground-filtered forces are terminal.
        ground_path = self.scene.terrain.prim_path + (
            "/terrain/mesh" if self.scene.terrain.terrain_type == "generator"
            else "/terrain/GroundPlane/CollisionPlane"
        )
        for sensor_name, body_name in (
            ("backpack_ground_contact", "backpack_link"),
            ("pelvis_ground_contact", "base_link"),
        ):
            setattr(self.scene, sensor_name, ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
                filter_prim_paths_expr=[ground_path],
                history_length=3,
                update_period=self.sim.dt,
            ))
        # The subclass adds a numerically safe tilt test; do not duplicate the
        # base acos check. No fixed world-height threshold on generated terrain.
        self.robot.terminate_base_orientation = None
        self.robot.terminate_base_height = None


@configclass
class BackpackRPOFlatBoundedEnvCfg(BackpackRPOFlatEnvCfg):
    # Tunable engineering starting values. They are a free range, NOT a target.
    backpack_nominal_mass_kg: float = 10.0
    backpack_pitch_deg_per_kg: float = 0.8
    backpack_pitch_maximum_deg: float = 12.0

    def __post_init__(self):
        super().__post_init__()
        self.reward.flat_orientation_l2.func = rewards.bounded_orientation_l2
        self.reward.upward.func = rewards.bounded_upward


@configclass
class BackpackRPOFlatCompliantEnvCfg(BackpackRPOFlatEnvCfg):
    # TACT-ful's reported compliance values; used as an initial transfer, not
    # claimed to be optimal for this robot or its rigid backpack.
    backpack_nominal_mass_kg: float = 10.0
    backpack_orientation_compliance_gain: float = 0.5
    backpack_rotational_stiffness_nm_per_rad: float = 50.0
    backpack_gravity_mps2: float = 9.81

    def __post_init__(self):
        super().__post_init__()
        # The baseline's broad 0.5 m/s tracking tolerance leaves a high
        # velocity reward even when the robot is nearly standing. Tighten only
        # the two command-tracking tolerances for this backpack experiment:
        # linear tracking should encourage forward progress and yaw tracking
        # should reduce heading drift. Keep their RPO weights and all other
        # baseline reward terms unchanged.
        self.reward.track_lin_vel_xy_exp.params["std"] = 0.25
        self.reward.track_ang_vel_z_exp.params["std"] = 0.25
        self.reward.flat_orientation_l2.func = rewards.compliant_orientation_l2
        self.reward.upward.func = rewards.compliant_upward
