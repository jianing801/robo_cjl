"""Delayed PD actuator constrained by a measured/modelled torque-speed envelope."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class TorqueSpeedEnvelopeActuator(DelayedPDActuator):
    """Apply a symmetric, piecewise-linear output torque-speed limit.

    The lookup table is expressed at the gearbox output (joint side).  The
    absolute joint speed is used, so the same magnitude limit is applied in
    all four quadrants.  Values above the final speed use the final torque
    value (normally zero).
    """

    cfg: "TorqueSpeedEnvelopeActuatorCfg"

    def __init__(self, cfg: "TorqueSpeedEnvelopeActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        lookup = torch.as_tensor(cfg.torque_speed_lookup, dtype=torch.float32, device=self._device)
        if lookup.ndim != 2 or lookup.shape[1] != 2 or lookup.shape[0] < 2:
            raise ValueError("torque_speed_lookup must contain at least two [speed_rad_s, torque_nm] rows.")
        if torch.any(lookup[1:, 0] <= lookup[:-1, 0]):
            raise ValueError("torque_speed_lookup speeds must be strictly increasing.")
        if torch.any(lookup < 0.0):
            raise ValueError("torque_speed_lookup speed and torque values must be non-negative.")

        self._envelope_speed = lookup[:, 0].contiguous()
        self._envelope_torque = lookup[:, 1].contiguous()
        self._joint_vel = torch.zeros_like(self.computed_effort)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        speed = torch.abs(self._joint_vel)
        speed = torch.clamp(speed, max=self._envelope_speed[-1])

        upper = torch.searchsorted(self._envelope_speed, speed.contiguous(), right=False)
        upper = torch.clamp(upper, min=1, max=self._envelope_speed.numel() - 1)
        lower = upper - 1

        x0 = self._envelope_speed[lower]
        x1 = self._envelope_speed[upper]
        y0 = self._envelope_torque[lower]
        y1 = self._envelope_torque[upper]
        torque_limit = y0 + (speed - x0) * (y1 - y0) / (x1 - x0)

        # Keep the scalar actuator effort limit as an additional hard safety
        # bound.  Normally it equals the zero-speed value of the envelope.
        torque_limit = torch.minimum(torque_limit, self.effort_limit)
        return torch.clamp(effort, min=-torque_limit, max=torque_limit)


@configclass
class TorqueSpeedEnvelopeActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`TorqueSpeedEnvelopeActuator`."""

    class_type: type = TorqueSpeedEnvelopeActuator
    torque_speed_lookup: list[list[float]] = MISSING
    """Rows of ``[abs joint speed (rad/s), max abs output torque (N*m)]``."""
