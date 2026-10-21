"""Same RPO observations/actions, with explicit backpack-aware fall resets."""

from robolab.tasks.direct.base.base_env import BaseEnv
from .rewards import excessive_tilt, ground_contact


class BackpackRPOFlatEnv(BaseEnv):
    def _capture_terminal_state(self, env_ids):
        state = super()._capture_terminal_state(env_ids)
        ids = self.termination_contact_cfg.body_ids
        state["termination_net_forces_n"] = self.contact_sensor.data.net_forces_w_history[
            env_ids][:, :, ids].norm(dim=-1).amax(dim=1).clone()
        state["contact_fall"] = self.backpack_contact_fall[env_ids].clone()
        state["tilt_fall"] = self.backpack_tilt_fall[env_ids].clone()
        for name in ("backpack_ground_contact", "pelvis_ground_contact"):
            forces = self.scene[name].data.force_matrix_w_history[env_ids]
            state[name + "_n"] = forces.norm(dim=-1).flatten(start_dim=1).amax(dim=1).clone()
        return state

    def _get_dones(self):
        contact_fall, time_out = super()._get_dones()
        for sensor_name in ("backpack_ground_contact", "pelvis_ground_contact"):
            forces = self.scene[sensor_name].data.force_matrix_w_history
            if forces is None:
                raise RuntimeError(f"{sensor_name} must be filtered against the terrain collider")
            contact_fall = contact_fall | ground_contact(forces)
        tilt_fall = excessive_tilt(
            self.robot.data.projected_gravity_b,
            self.cfg.backpack_fall_tilt_rad,
        )
        # Diagnostic masks are captured before automatic reset. A single fall
        # can satisfy both; their counts must not be added as independent falls.
        self.backpack_contact_fall = contact_fall
        self.backpack_tilt_fall = tilt_fall
        return contact_fall | tilt_fall, time_out
