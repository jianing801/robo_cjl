"""Backpack-aware variants of the two RPO posture terms."""

import math

import torch


def _lateral_gravity(gravity_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # g_y = -cos(pitch) sin(roll), g_z = -cos(pitch) cos(roll).
    # Normalization removes the pitch dependence while |pitch| < 90 degrees.
    lateral_norm = torch.linalg.vector_norm(gravity_b[:, 1:3], dim=1).clamp_min(1e-6)
    return gravity_b[:, 1] / lateral_norm, -gravity_b[:, 2] / lateral_norm


def roll_orientation_l2(env) -> torch.Tensor:
    """Original weight -1, but sin(roll)^2 instead of pitch+roll error."""
    lateral_sine, _ = _lateral_gravity(env.scene["robot"].data.projected_gravity_b)
    return lateral_sine.square()


def roll_upward(env) -> torch.Tensor:
    """Original weight +0.4, but cos(roll) rather than cos(pitch) cos(roll)."""
    _, lateral_cosine = _lateral_gravity(env.scene["robot"].data.projected_gravity_b)
    return lateral_cosine


def excessive_tilt(gravity_b: torch.Tensor, threshold_rad: float) -> torch.Tensor:
    """Severe total tilt, not a tracking/pitch target or a timeout condition."""
    if not 0.0 < threshold_rad < math.pi / 2:
        raise ValueError("fall tilt threshold must be strictly between 0 and 90 degrees")
    # Avoid acos domain errors from roundoff. Threshold equivalence holds for
    # unit projected gravity; equality remains non-terminal just as before.
    upright_cosine = (-gravity_b[:, 2]).clamp(-1.0, 1.0)
    return upright_cosine < math.cos(threshold_rad)


def ground_contact(force_history: torch.Tensor, threshold_n: float = 1.0) -> torch.Tensor:
    """Reduce [env, history, sensor_body, filtered_ground, xyz] to fall masks."""
    return force_history.norm(dim=-1).flatten(start_dim=1).amax(dim=1) > threshold_n


def bounded_posture_terms(gravity_b, mass_kg, deg_per_kg=0.8, maximum_deg=12.0):
    """Original posture terms applied to excess pitch, not a prescribed lean.

    Engineering starting parameters, not a paper-derived optimal angle.
    Zero mass exactly recovers the original flat gravity penalty/upward term.
    """
    if not (deg_per_kg >= 0 and 0 <= maximum_deg < 60):
        raise ValueError("Pitch allowance must remain below the severe-fall threshold")
    if not torch.isfinite(mass_kg).all() or (mass_kg < 0).any():
        raise ValueError("Payload masses must be finite and nonnegative")
    pitch = torch.asin(gravity_b[:, 0].clamp(-1, 1))
    sine_roll, cosine_roll = _lateral_gravity(gravity_b)
    allowance = (mass_kg * deg_per_kg).clamp(max=maximum_deg) * (math.pi / 180)
    excess_pitch = (pitch.abs() - allowance).clamp_min(0)
    effective_cosine_pitch = torch.cos(excess_pitch)
    orientation = torch.sin(excess_pitch).square() + effective_cosine_pitch.square() * sine_roll.square()
    upward = effective_cosine_pitch * cosine_roll
    return orientation, upward


def _bounded_terms(env):
    robot = env.scene["robot"]
    mass = getattr(robot, "backpack_mass_kg", None)
    if mass is None:
        # Only initialization before sampled PhysX properties are written.
        mass = torch.full((env.num_envs,), env.cfg.backpack_nominal_mass_kg, device=env.device)
    return bounded_posture_terms(robot.data.projected_gravity_b, mass,
                                 env.cfg.backpack_pitch_deg_per_kg,
                                 env.cfg.backpack_pitch_maximum_deg)


def bounded_orientation_l2(env):
    return _bounded_terms(env)[0]


def bounded_upward(env):
    return _bounded_terms(env)[1]


def compliant_posture_terms(
    gravity_b,
    mass_kg,
    com_x_m,
    compliance_gain=0.5,
    rotational_stiffness_nm_per_rad=50.0,
    gravity_mps2=9.81,
):
    """Shift, but retain, RPO's pitch penalty using a payload-moment target.

    Adapted from TACT-ful Eq. (14)-(15): theta* = alpha * tau_pitch / k_rot.
    Here tau_pitch is approximated by the static backpack gravity moment using
    each environment's sampled mass and torso-relative CoM x coordinate. The
    pitch variable follows Isaac's projected-gravity sign convention.
    """
    if not (0.0 <= compliance_gain <= 1.0):
        raise ValueError("compliance_gain must be between 0 and 1")
    if not math.isfinite(rotational_stiffness_nm_per_rad) or rotational_stiffness_nm_per_rad <= 0:
        raise ValueError("rotational stiffness must be positive and finite")
    if not math.isfinite(gravity_mps2) or gravity_mps2 <= 0:
        raise ValueError("gravity must be positive and finite")
    if (not torch.isfinite(mass_kg).all() or (mass_kg < 0).any()
            or not torch.isfinite(com_x_m).all()):
        raise ValueError("Payload mass and CoM coordinates must be finite; mass must be nonnegative")

    observed_pitch = torch.asin(gravity_b[:, 0].clamp(-1, 1))
    pitch_moment_nm = mass_kg * gravity_mps2 * com_x_m
    target_pitch = compliance_gain * pitch_moment_nm / rotational_stiffness_nm_per_rad
    pitch_error = observed_pitch - target_pitch
    sine_roll, cosine_roll = _lateral_gravity(gravity_b)
    effective_cosine_pitch = torch.cos(pitch_error)
    orientation = torch.sin(pitch_error).square() + effective_cosine_pitch.square() * sine_roll.square()
    upward = effective_cosine_pitch * cosine_roll
    return orientation, upward, target_pitch


def _compliant_terms(env):
    robot = env.scene["robot"]
    mass = getattr(robot, "backpack_mass_kg", None)
    if mass is None:
        mass = torch.full((env.num_envs,), env.cfg.backpack_nominal_mass_kg, device=env.device)
    com_x = getattr(robot, "backpack_com_x_m", None)
    if com_x is None:
        com_x = torch.full((env.num_envs,), -0.17, device=env.device)
    return compliant_posture_terms(
        robot.data.projected_gravity_b,
        mass,
        com_x,
        env.cfg.backpack_orientation_compliance_gain,
        env.cfg.backpack_rotational_stiffness_nm_per_rad,
        env.cfg.backpack_gravity_mps2,
    )


def compliant_orientation_l2(env):
    return _compliant_terms(env)[0]


def compliant_upward(env):
    return _compliant_terms(env)[1]
