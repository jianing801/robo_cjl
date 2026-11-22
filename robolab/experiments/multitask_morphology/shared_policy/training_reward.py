"""Summarize observed PPO rewards within one simulator rollout session."""
from collections import deque
import csv
import os
from pathlib import Path
import statistics
import tempfile

import torch


REWARD_FIELDS = (
    'reward_samples', 'mean_environment_step_reward', 'mean_ppo_step_reward',
    'completed_episodes', 'timeout_episodes', 'terminated_episodes',
    'mean_environment_episode_return', 'mean_ppo_episode_return',
)


class RolloutRewards:
    """Carry unfinished episodes across batches from the same workers."""

    def __init__(self, windows=None):
        self.episode_returns = {}
        self.windows = {task: deque(values, maxlen=100) for task, values in (windows or {}).items()}

    def summarize(self, batches, session_id):
        tasks = {}
        finished_by_step = [[] for _ in range(len(batches[0]['reward']))]
        for batch in batches:
            rewards = torch.stack((batch['environment_reward'], batch['reward']), dim=1).double()
            task = batch['task']
            running = self.episode_returns.setdefault(task, rewards.new_zeros(rewards.shape[1:]))
            window = self.windows.setdefault(task, deque(maxlen=100))
            finished = rewards.new_zeros(2)
            for step, (step_rewards, done) in enumerate(zip(rewards, batch['done'], strict=True)):
                running.add_(step_rewards)
                finished.add_(running[:, done].sum(dim=1))
                returns = running[0, done].tolist()
                window.extend(returns)
                finished_by_step[step].extend(returns)
                running[:, done] = 0.
            completed = int(batch['done'].sum())
            timeouts = int(batch['timeout'].sum())
            environment_step, ppo_step = rewards.mean(dim=(0, 2)).tolist()
            environment_return, ppo_return = (finished / completed).tolist() if completed else (None, None)
            tasks[task] = dict(
                reward_samples=batch['reward'].numel(),
                mean_environment_step_reward=environment_step, mean_ppo_step_reward=ppo_step,
                completed_episodes=completed, timeout_episodes=timeouts, terminated_episodes=completed - timeouts,
                mean_environment_episode_return=environment_return, mean_ppo_episode_return=ppo_return,
            )
        overall = self.windows.setdefault('all', deque(maxlen=100))
        for returns in finished_by_step:
            overall.extend(returns)
        mean_reward = {task: dict(mean_reward=statistics.mean(window) if window else None,
                                  window_episodes=len(window)) for task, window in self.windows.items()}
        return dict(version='ppo_training_rewards_v2', session_id=session_id, tasks=tasks,
                    mean_reward=mean_reward,
                    mean_reward_windows={task: list(window) for task, window in self.windows.items()})


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


def write_reward_csv(path, fields, rows):
    """Atomically refresh the view of already committed checkpoint receipts."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         mode='w', encoding='utf8', newline='', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_training_reward_csv(root, rows):
    write_reward_csv(Path(root) / 'training_reward.csv', ('update', 'task', 'session_id', *REWARD_FIELDS), rows)


def mean_reward_rows(receipts, tasks):
    for receipt in receipts:
        if receipt['completed_updates'] == 0:
            continue
        values = receipt.get('training_reward', {})
        row = dict(update=receipt['completed_updates'], session_id=values.get('session_id'))
        for task in ('all', *tasks):
            prefix = '' if task == 'all' else task + '_'
            stats = values.get('mean_reward', {}).get(task, {})
            row[prefix + 'mean_reward'] = stats.get('mean_reward')
            row[prefix + 'window_episodes'] = stats.get('window_episodes')
        yield row


def write_mean_reward_csv(root, tasks, rows):
    fields = ['update', 'session_id', 'mean_reward', 'window_episodes']
    for task in tasks:
        fields.extend((task + '_mean_reward', task + '_window_episodes'))
    write_reward_csv(Path(root) / 'mean_reward.csv', fields, rows)
