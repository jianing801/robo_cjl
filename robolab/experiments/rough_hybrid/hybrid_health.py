"""Read-only per-motor health metrics, captured before automatic reset.

All reductions stay on device. Native logger averages these step scalars over
the rollout and writes TensorBoard/console (controller retains stdout.log).
"""
import torch


def motor_health(env, rewards):
    designs = env.cfg.spatial_health_designs
    groups = env.spatial_groups
    names = env.reward_manager._term_names
    # Native _step_reward is already weighted, but is a reward RATE (no dt).
    terms = env.reward_manager._step_reward
    soft_index = names.index('design_motor_torque_limits') if 'design_motor_torque_limits' in names else None
    soft = terms[:, soft_index] * env.step_dt if soft_index is not None else torch.zeros_like(rewards)
    absolute_budget = terms.abs().sum(-1) * env.step_dt
    if not hasattr(env, '_motor_health_masks'):
        env._motor_health_masks = {}
        for role, field in (('knee', 'knee_motor'), ('ankle', 'ankle_motor')):
            for model in sorted({d[field] for d in designs}):
                mask = torch.tensor([d[field] == model for d in designs], device=groups.device)
                env._motor_health_masks[role, model] = mask[groups]
    result = {}
    for role, group, field in (('knee', 'knees', 'knee_motor'), ('ankle', 'feet', 'ankle_motor')):
        actuator = env.robot.actuators[group]
        ids = list(actuator.joint_indices)
        applied = env.robot.data.applied_torque[:, ids]
        requested = actuator.computed_effort
        available = actuator.dynamic_effort_limit
        peak = torch.broadcast_to(actuator.effort_limit, applied.shape)
        valid = available > peak * 1.e-6
        denominator = torch.where(valid, available, torch.ones_like(available))
        utilization = torch.where(valid, applied.abs() / denominator, torch.zeros_like(applied))
        for model in sorted({d[field] for d in designs}):
            selected = env._motor_health_masks[role, model]
            prefix = f'Health/{role}/{model}/'
            live = valid[selected]
            count = live.sum().clamp(min=1)
            u = utilization[selected]
            result.update({prefix + 'reward_per_step': rewards[selected].mean(),
                prefix + 'forward_velocity_mps': env.robot.data.root_lin_vel_b[selected, 0].mean(),
                prefix + 'termination_fraction': env.reset_terminated[selected].float().mean(),
                prefix + 'reset_fraction': env.reset_buf[selected].float().mean(),
                prefix + 'mean_abs_raw_action': env.action_buffer.buffer[selected, -1, :][:, ids].abs().mean(),
                prefix + 'mean_abs_applied_nm': applied[selected].abs().mean(),
                prefix + 'utilization_valid_mean': u.sum() / count,
                prefix + 'over_80_fraction_valid': ((u > .8) & live).sum() / count,
                prefix + 'saturated_99_fraction_valid': ((u >= .99) & live).sum() / count,
                prefix + 'unavailable_fraction': (~live).float().mean(),
                prefix + 'requested_excess_nm': (requested[selected].abs() - available[selected]).clamp(min=0).mean(),
                prefix + 'soft_reward_per_step': soft[selected].mean(),
                prefix + 'soft_share_abs_reward_budget': soft[selected].abs().sum() / absolute_budget[selected].sum().clamp(min=1.e-12),
                prefix + 'nonfinite_fraction': (~torch.isfinite(applied[selected]) | ~torch.isfinite(requested[selected]) | ~torch.isfinite(available[selected])).float().mean()})
    return {key: value.detach().clone() for key, value in result.items()}
