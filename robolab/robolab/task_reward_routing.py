"""Strict RUN/STAIRS routing; no simulator imports or new reward weights."""

import torch

ROUTING_VERSION = "rpo_run_amp_stairs_rough_v1"


def validate_task_ids(task_ids, batch_size):
    if task_ids.shape != (batch_size,) or not torch.all((task_ids == 0) | (task_ids == 1)):
        raise ValueError("task IDs must be a vector containing only 0=RUN or 1=STAIRS")
    return task_ids == 0


def task_ids_from_context(context):
    if context.ndim != 2 or context.shape[1] != 2:
        raise ValueError("RUN/STAIRS context must have exactly two columns")
    if not torch.all((context == 0) | (context == 1)) or not torch.all(context.sum(-1) == 1):
        raise ValueError("task context must be a valid one-hot vector")
    return context.argmax(-1)


def route_task_rewards(run_reward, stairs_reward, task_ids):
    if run_reward.ndim != 1 or run_reward.shape != stairs_reward.shape:
        raise ValueError("task rewards must be equal-sized vectors")
    mask = validate_task_ids(task_ids, len(run_reward))
    result = torch.where(mask, run_reward, stairs_reward)
    if not torch.isfinite(result).all():
        raise ValueError("selected task reward is non-finite")
    return result


def mix_run_style(task_reward, run_style_reward, task_ids, task_style_lerp):
    """Original AMP interpolation for RUN; STAIRS reward is byte-for-byte unchanged."""
    mask = validate_task_ids(task_ids, len(task_reward))
    if not 0 <= task_style_lerp <= 1 or task_reward.shape != run_style_reward.shape:
        raise ValueError("invalid AMP reward interpolation")
    return torch.where(mask, task_style_lerp * task_reward + (1 - task_style_lerp) * run_style_reward,
                       task_reward)
