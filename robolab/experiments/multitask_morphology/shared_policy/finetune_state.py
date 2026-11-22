"""Exact candidate mapping and committed PPO learning-state checkpoints (no simulator)."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import re
import uuid

import numpy as np
import torch

from contract import sha, structure_key, write_new
from ppo_channel import publish


def select_candidate(plan, *, design_id=None, design_task='stairs', design_key=None):
    if (design_id is None) == (design_key is None):
        raise ValueError('Select exactly one --design-id or --design-key')
    banks = {task: json.loads(Path(spec['bank']).read_text(encoding='utf8'))
             for task, spec in plan['teachers'].items()}
    if design_id is not None:
        if design_task not in banks:
            raise ValueError('Unknown candidate task')
        rows = [d for d in banks[design_task]['designs'] if d['id'] == design_id
                and d['id'] in plan['teachers'][design_task]['design_ids']]
        if len(rows) != 1:
            raise ValueError('Candidate ID must identify one allowed physical structure')
        key = structure_key(rows[0])
    else:
        if len(design_key) != 4:
            raise ValueError('Design key requires thigh_m calf_m knee_motor ankle_motor')
        lengths = [float(v) for v in design_key[:2]]
        if not all(math.isfinite(v) and v > 0 for v in lengths):
            raise ValueError('Candidate leg lengths must be finite and positive')
        key = structure_key(dict(thigh_m=lengths[0], calf_m=lengths[1],
                                 knee_motor=design_key[2], ankle_motor=design_key[3]))
    if key not in {tuple(k) for k in plan['selected_structure_keys']}:
        raise ValueError('Candidate is outside the paired structure library; no nearest-design substitution')
    selected = {}
    frozen = deepcopy(plan)
    for task, spec in frozen['teachers'].items():
        rows = [d for d in banks[task]['designs'] if d['id'] in spec['design_ids'] and structure_key(d) == key]
        if len(rows) != 1:
            raise ValueError(f'Candidate must map exactly once in {task}')
        selected[task] = rows[0]
        spec['design_ids'] = [rows[0]['id']]
        spec['copies'] = 8
    frozen.update(selected_structure_keys=[list(key)], scope='exact_candidate')
    # Keep the actor's schema/encoding of the full common bank unchanged.
    return frozen, dict(structure_key=list(key), design_ids={t: d['id'] for t, d in selected.items()})


def capture_rng():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]])


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    name, keys, pos, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))


def save_checkpoint(root, model, critic, optimizer, completed_updates, manifest_sha256, metrics=None, *,
                    rollout=None, training_reward=None):
    folder = Path(root) / 'checkpoints'
    folder.mkdir(exist_ok=True)
    path = folder / f'update_{completed_updates:06d}_{uuid.uuid4().hex}.pt'
    value = dict(kind='shared_ppo_learning_state_v2', actor=model.state_dict(), critic=critic.state_dict(),
        optimizer=optimizer.state_dict(), rng=capture_rng(), completed_updates=completed_updates,
        manifest_sha256=manifest_sha256, schema=model.schema, architecture=model.architecture,
        metrics=metrics, simulator_state_restored=False)
    if training_reward is not None:
        value['training_reward'] = training_reward
    publish(path, value)
    record = dict(completed_updates=completed_updates, manifest_sha256=manifest_sha256,
                  path=path.name, sha256=sha(path))
    if rollout is not None:
        record.update(metrics=metrics, rollout=rollout)
    if training_reward is not None:
        record['training_reward'] = training_reward
    write_new(folder / f'update_{completed_updates:06d}.json', record)
    return record


def _checkpoint_receipt(path, index, manifest_sha256):
    record = json.loads(path.read_text(encoding='utf8'))
    if (path.name != f'update_{index:06d}.json' or record['completed_updates'] != index
            or record['manifest_sha256'] != manifest_sha256):
        raise ValueError('PPO checkpoint history is incomplete or belongs to another run')
    return record


def checkpoint_history(root, manifest_sha256):
    folder = Path(root) / 'checkpoints'
    commits = sorted(folder.glob('update_*.json'))
    return [_checkpoint_receipt(path, index, manifest_sha256) for index, path in enumerate(commits)]


def load_checkpoint(root, model, critic, optimizer, manifest_sha256):
    folder = Path(root) / 'checkpoints'
    history = checkpoint_history(root, manifest_sha256)
    if not history:
        return None
    record = history[-1]
    state_path = folder / record['path']
    if state_path.parent != folder or sha(state_path) != record['sha256']:
        raise ValueError('Committed PPO checkpoint changed')
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    if (state.get('kind') != 'shared_ppo_learning_state_v2'
            or state['completed_updates'] != record['completed_updates']
            or state['manifest_sha256'] != manifest_sha256
            or state.get('training_reward') != record.get('training_reward')
            or state['schema'] != model.schema or state['architecture'] != model.architecture):
        raise ValueError('PPO learning-state identity mismatch')
    model.load_state_dict(state['actor'], strict=True)
    critic.load_state_dict(state['critic'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    tensors = [*model.state_dict().values(), *critic.state_dict().values()]
    tensors += [v for row in optimizer.state.values() for v in row.values() if isinstance(v, torch.Tensor)]
    if not all(torch.isfinite(v).all() for v in tensors):
        raise ValueError('Nonfinite restored PPO learning state')
    restore_rng(state['rng'])
    return state['completed_updates']


def _sync_directory(folder):
    if os.name != 'nt':
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _checkpoint_blob(folder, record):
    name = record['path']
    if not re.fullmatch(rf'update_{record["completed_updates"]:06d}_[0-9a-f]{{32}}\.pt', name):
        raise ValueError('Invalid compact PPO checkpoint path')
    path = folder / name
    if path.is_symlink():
        raise ValueError('Compact PPO checkpoint path cannot be a symlink')
    return path


def _consumed_rollout_paths(attempts, record):
    tasks = ('stairs', 'backpack', 'run')
    rollout = record.get('rollout', {})
    session_id = rollout.get('session_id', '')
    if not re.fullmatch(r'[0-9a-f]{32}', session_id) or set(rollout.get('task_audits', {})) != set(tasks):
        raise ValueError('Missing compact PPO rollout provenance')
    attempt = attempts / session_id
    if attempt.resolve() != attempt:
        raise ValueError('Compact PPO attempt must be inside the run')
    index = record['completed_updates'] - 1
    actor = attempt / f'actor_{index:03d}.pt'
    request = json.loads((attempt / f'request_{index:03d}.json').read_text(encoding='utf8'))
    expected = dict(kind='shared_ppo_request_v2', session_id=session_id,
                    manifest_sha256=record['manifest_sha256'], index=index, student=str(actor))
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError('Compact PPO rollout request identity mismatch')
    return attempt, [actor, *(attempt / f'{task}_{index:03d}.pt' for task in tasks)]


def prune_training_artifacts(root, manifest_sha256, completed, *, reconcile=False):
    """Reclaim compact-run tensors only after their owning update was committed."""
    root = Path(root).resolve()
    folder, attempts = root / 'checkpoints', root / 'attempts'
    if folder.resolve() != folder or attempts.resolve() != attempts:
        raise ValueError('Compact PPO storage must be inside the run')

    def receipt(index):
        return _checkpoint_receipt(folder / f'update_{index:06d}.json', index, manifest_sha256)

    latest = receipt(completed)
    if sha(_checkpoint_blob(folder, latest)) != latest['sha256']:
        raise ValueError('Committed PPO checkpoint changed')
    if reconcile:
        history = checkpoint_history(root, manifest_sha256)
        if history[-1] != latest:
            raise ValueError('Compact PPO checkpoint history changed')
        obsolete = history[1:-2]
        consumed = history[1:]
    else:
        obsolete = [receipt(completed - 2)] if completed >= 3 else []
        consumed = [latest]
    blobs = [_checkpoint_blob(folder, record) for record in obsolete]
    rollouts = [_consumed_rollout_paths(attempts, record) for record in consumed]
    # A killed process may leave consumed tensors behind. Sync the receipt and
    # request directories before unlinking, and repeat reclamation on resume.
    _sync_directory(folder)
    _sync_directory(root)
    _sync_directory(attempts)
    for attempt in {attempt for attempt, _ in rollouts}:
        _sync_directory(attempt)
    for path in blobs:
        path.unlink(missing_ok=True)
    for _, paths in rollouts:
        for path in paths:
            path.unlink(missing_ok=True)


def validate_batch(data, *, task, index, session, actor_sha256, width, envs=8, actions=23):
    expected = dict(task=task, index=index, session_id=session['session_id'],
        student_sha256=actor_sha256, manifest_sha256=session['manifest_sha256'],
        collection_protocol=session['collection_protocol'])
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('PPO batch identity mismatch')
    n = session['steps_per_task']
    shapes = dict(obs=(n, envs, width), actions=(n, envs, actions), logp=(n, envs),
                  reward=(n, envs), environment_reward=(n, envs), done=(n, envs), timeout=(n, envs),
                  next_obs=(envs, width))
    for name, shape in shapes.items():
        value = data.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid PPO batch tensor: {name}')
        if name in ('done', 'timeout'):
            if value.dtype != torch.bool:
                raise ValueError('PPO termination masks must be boolean')
        elif not value.is_floating_point():
            raise ValueError('PPO measurements must be floating point')
    if (data['timeout'] & ~data['done']).any():
        raise ValueError('PPO timeout must end an episode')
    return data
