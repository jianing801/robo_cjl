"""Parkour-style soft applied-torque constraint for Rough design motors only.

Uses each live actuator's per-environment peak ceiling, not the first URDF's
shared limits. The real speed-dependent hard envelope remains unchanged.
Return an unweighted N^2 m^2 cost; native RewardManager applies weight * dt.
"""
import math

GROUPS = ('knees', 'feet')
DEFAULT = dict(weight=-0.01, limit_ratio=0.8)


def validate_settings(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(DEFAULT):
        raise ValueError('torque_soft_constraint requires weight and limit_ratio only')
    if any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in DEFAULT):
        raise ValueError('torque soft constraint settings must be finite numbers')
    if value['weight'] > 0 or not 0 < value['limit_ratio'] <= 1:
        raise ValueError('torque soft constraint needs weight <= 0 and 0 < limit_ratio <= 1')
    return {k: float(value[k]) for k in DEFAULT}


def squared_torque_excess(applied, peak, limit_ratio=0.8):
    import torch
    if applied.shape != peak.shape or applied.ndim != 2:
        raise ValueError('expected matching (environments, joints) torque/peak arrays')
    # Values are validated once when the live per-design table is cached;
    # do not force a GPU->CPU synchronisation on every reward computation.
    return (applied.abs() - peak * limit_ratio).clamp(min=0).square().sum(dim=-1)


def design_motor_torque_limits(env, limit_ratio=0.8):
    """Same squared excess as RPO Parkour, restricted to the six design joints."""
    import torch
    cached = getattr(env, '_design_torque_soft_limits', None)
    if cached is None:
        ids, limits = [], []
        for name in GROUPS:
            actuator = env.robot.actuators[name]
            group_ids = list(actuator.joint_indices)
            peak = torch.as_tensor(actuator.effort_limit, device=env.device,
                                   dtype=env.robot.data.applied_torque.dtype)
            limits.append(torch.broadcast_to(peak, (env.num_envs, len(group_ids))).clone())
            ids.extend(group_ids)
        if len(ids) != 6 or len(set(ids)) != 6:
            raise ValueError('expected two knees and four distinct ankle joints')
        peak = torch.cat(limits, dim=-1)
        if not torch.isfinite(peak).all() or not (peak > 0).all():
            raise ValueError('invalid per-design peak torque')
        cached = env._design_torque_soft_limits = (ids, peak)
    ids, peak = cached
    # Read before auto-reset: native RewardManager.compute is inside env.step.
    return squared_torque_excess(env.robot.data.applied_torque[:, ids], peak, limit_ratio)


def install_constraint(cfg, value):
    settings = validate_settings(value)
    if settings is None or settings['weight'] == 0:
        return dict(enabled=False, settings=settings)
    from isaaclab.managers import RewardTermCfg
    cfg.reward.design_motor_torque_limits = RewardTermCfg(
        func=design_motor_torque_limits, weight=settings['weight'],
        params=dict(limit_ratio=settings['limit_ratio']))
    return dict(enabled=True, settings=settings, groups=list(GROUPS), joint_count=6,
        torque='actual applied torque', limit='per-environment actuator peak effort_limit',
        formula='sum_j relu(abs(applied_torque_j) - limit_ratio * peak_j)^2',
        reference='RPO Parkour applied_torque_limits_by_ratio',
        weight_and_dt='native RewardManager only; no wrapper reward addition',
        dynamic_hard_envelope_unchanged=True, action_scale_and_pd_unchanged=True)
