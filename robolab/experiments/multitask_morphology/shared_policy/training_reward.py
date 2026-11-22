"""Summarize observed PPO rewards within one simulator rollout session."""
import csv
import os
from pathlib import Path
import tempfile

import torch


REWARD_FIELDS = (
    'reward_samples', 'mean_environment_step_reward', 'mean_ppo_step_reward',
    'completed_episodes', 'timeout_episodes', 'terminated_episodes',
    'mean_environment_episode_return', 'mean_ppo_episode_return',
)


class RolloutRewards:
    """Carry unfinished episodes across batches from the same workers."""

    def __init__(self):
        self.episode_returns = {}

    def summarize(self, batches, session_id):
        tasks = {}
        for batch in batches:
            rewards = torch.stack((batch['environment_reward'], batch['reward']), dim=1).double()
            running = self.episode_returns.setdefault(batch['task'], rewards.new_zeros(rewards.shape[1:]))
            finished = rewards.new_zeros(2)
            for step_rewards, done in zip(rewards, batch['done']):
                running.add_(step_rewards)
                finished.add_(running[:, done].sum(dim=1))
                running[:, done] = 0.
            completed = int(batch['done'].sum())
            timeouts = int(batch['timeout'].sum())
            environment_step, ppo_step = rewards.mean(dim=(0, 2)).tolist()
            environment_return, ppo_return = (finished / completed).tolist() if completed else (None, None)
            tasks[batch['task']] = dict(
                reward_samples=batch['reward'].numel(),
                mean_environment_step_reward=environment_step, mean_ppo_step_reward=ppo_step,
                completed_episodes=completed, timeout_episodes=timeouts, terminated_episodes=completed - timeouts,
                mean_environment_episode_return=environment_return, mean_ppo_episode_return=ppo_return,
            )
        return dict(version='ppo_training_rewards_v1', session_id=session_id, tasks=tasks)


def reward_history_rows(receipts, tasks):
    rows = []
    for receipt in receipts:
        update = receipt['completed_updates']
        if update == 0:
            continue
        rewards = receipt.get('training_reward', {})
        for task in tasks:
            metrics = rewards.get('tasks', {}).get(task, {})
            rows.append(dict(update=update, task=task, session_id=rewards.get('session_id'),
                             **{field: metrics.get(field) for field in REWARD_FIELDS}))
    return rows


def write_training_reward_csv(root, rows):
    """Atomically refresh the view of already committed checkpoint receipts."""
    path = Path(root) / 'training_reward.csv'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         mode='w', encoding='utf8', newline='', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=('update', 'task', 'session_id', *REWARD_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
