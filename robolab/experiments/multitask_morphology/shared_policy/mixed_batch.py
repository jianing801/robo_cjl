"""Lockstep orchestration of task subscenes that share one physics context."""

import torch


def task_counts(num_envs):
    if num_envs < 3:
        raise ValueError('The mixed scene needs at least one environment per task')
    floor, remainder = divmod(num_envs, 3)
    return tuple(floor + (i < remainder) for i in range(3))


def initialize_episode_length(counters, horizon):
    """Match RSL-RL's init_at_random_ep_len without replacing the live buffer."""
    counters.copy_(torch.randint_like(counters, high=int(horizon)))


def advance(tasks, counts, actions, substeps, physics_step):
    for task, action in zip(tasks, actions.split(counts), strict=True):
        task.begin(action)
    for _ in range(substeps):
        for task in tasks:
            task.apply()
        physics_step()
        for task in tasks:
            task.sync()
    return [task.finish() for task in tasks]
