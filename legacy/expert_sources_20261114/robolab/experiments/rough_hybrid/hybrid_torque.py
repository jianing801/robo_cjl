"""Dynamic dimensionless torque utilization cost for six Rough design joints.

Reuse the hard-envelope ceiling; native RewardManager applies weight * dt.
The absolute-torque helper remains solely for baseline comparison tests.
"""
import math

GROUPS = ('knees', 'feet')
DEFAULT = dict(weight=-4.0, limit_ratio=0.8)


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
    """Last physics substep utilization; reuse the actual hard-envelope limit.

    Zero-capability entries are excluded from utilization, NOT declared healthy.
    Their count and pre-clip demand remain available as separate diagnostics.
    """
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
    dynamic = torch.cat([env.robot.actuators[name].dynamic_effort_limit for name in GROUPS], dim=-1)
    applied = env.robot.data.applied_torque[:, ids]
    cost, utilization, unavailable = dynamic_torque_excess(applied, dynamic, peak, limit_ratio)
    requested = torch.cat([env.robot.actuators[name].computed_effort for name in GROUPS], dim=-1)
    env._design_torque_soft_diagnostics = dict(
        utilization=utilization.detach(), unavailable=unavailable.detach(),
        requested_excess_nm=(requested.abs() - dynamic).clamp(min=0).detach())
    return cost


def dynamic_torque_excess(applied, available, peak, limit_ratio=0.8):
    """Dimensionless squared excess, with explicit zero-capability mask."""
    import torch
    if applied.shape != available.shape or applied.shape != peak.shape or applied.ndim != 2:
        raise ValueError('expected matching (environments, joints) arrays')
    valid = available > peak * 1.e-6
    safe = torch.where(valid, available, torch.ones_like(available))
    utilization = torch.where(valid, applied.abs() / safe, torch.zeros_like(applied))
    cost = torch.where(valid, (utilization - limit_ratio).clamp(min=0).square(),
                       torch.zeros_like(applied)).sum(dim=-1)
    return cost, utilization, ~valid


def install_constraint(cfg, value):
    settings = validate_settings(value)
    if settings is None or settings['weight'] == 0:
        return dict(enabled=False, settings=settings)
    from isaaclab.managers import RewardTermCfg
    cfg.reward.design_motor_torque_limits = RewardTermCfg(
        func=design_motor_torque_limits, weight=settings['weight'],
        params=dict(limit_ratio=settings['limit_ratio']))
    return dict(enabled=True, settings=settings, groups=list(GROUPS), joint_count=6,
        torque='actual applied torque at last physics substep', limit='same-substep hard-envelope dynamic_effort_limit',
        formula='sum_valid_j relu(abs(applied_torque_j) / available_j - limit_ratio)^2',
        reference='Parkour-inspired threshold, dimensionless dynamic utilization (not original Parkour formula)',
        weight_and_dt='native RewardManager only; no wrapper reward addition',
        dynamic_hard_envelope_unchanged=True, action_scale_and_pd_unchanged=True)
