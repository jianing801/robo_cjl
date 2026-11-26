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
from finetune_settings import NATIVE_PPO_ALGORITHM, NATIVE_PPO_SETTINGS, NATIVE_PPO_STATE_KIND
from ppo_channel import publish


NATIVE_CHECKPOINT_INTERVAL = 10
NATIVE_METRICS_RECEIPT_KIND = 'native_ppo_metrics_only_v1'


def native_checkpoint_due(update, final_update):
    """Keep full learning state at 0, every ten updates, and the final budget."""
    if (type(update) is not int or type(final_update) is not int
            or final_update < 1 or not 0 <= update <= final_update):
        raise ValueError('Invalid native PPO checkpoint update/budget')
    return update == final_update or update % NATIVE_CHECKPOINT_INTERVAL == 0


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


def _native_rng():
    state = capture_rng()
    state['cuda'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return state


def _restore_native_rng(state):
    restore_rng(state)
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Native PPO CUDA RNG device count differs on resume')
        torch.cuda.set_rng_state_all(state['cuda'])


def _ring_state(ring):
    # Native CircularBuffer sampling includes the previous <=100 transitions,
    # not just the current rollout; save its actual cursor and replay tensors.
    return dict(buffer=ring._buffer, pointer=ring._pointer,
                num_pushes=ring._num_pushes, max_length=ring.max_length,
                batch_size=ring._batch_size)


def _restore_ring(ring, state):
    pushes, buf, pointer = state['num_pushes'], state['buffer'], state['pointer']
    if (state['max_length'] != ring.max_length or state['batch_size'] != ring._batch_size
            or pushes.shape != (ring._batch_size,) or (pushes < 0).any()
            or type(pointer) != int or not -1 <= pointer < ring.max_length
            or (buf is None and (pointer != -1 or pushes.any()))
            or (buf is not None and (pointer < 0 or buf.shape[:2] != (ring.max_length, ring._batch_size)))):
        raise ValueError('Native AMP replay state differs from its ring contract')
    ring._buffer = None if buf is None else buf.to(ring._device)
    ring._pointer = pointer
    ring._num_pushes.copy_(pushes)


def _finite_state(state):
    if isinstance(state, torch.Tensor):
        return bool(torch.isfinite(state).all())
    if isinstance(state, dict):
        return all(_finite_state(v) for v in state.values())
    if isinstance(state, (tuple, list)):
        return all(_finite_state(v) for v in state)
    if isinstance(state, float):
        return math.isfinite(state)
    return True


def _v4_policy_contract(policy):
    """Record the immutable identity of all three native expert priors."""
    widths = list(policy.critic_widths)
    hashes = dict(policy.teacher_checkpoint_sha256)
    if (policy.std.shape != (3, 23) or not torch.isfinite(policy.std).all() or (policy.std <= 0).any()
            or len(policy.critics) != 3 or len(policy.critic_normalizers) != 3
            or len(widths) != 3 or any(type(width) != int or width <= 0 for width in widths)
            or set(hashes) != {'stairs', 'backpack', 'run'}
            or any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes.values())):
        raise ValueError('Invalid native PPO expert-prior identity or learned standard deviations')
    return dict(critic_widths=widths, teacher_checkpoint_sha256=hashes)


def save_native_checkpoint(root, learner, completed_updates, manifest_sha256, metrics=None, *,
                           warmup_completed=10, training_reward=None):
    """Atomically commit the v4 native state without mutating old v2/v3 archives.

    Index zero is the actor's original state *after* the separate Critic-only
    warmup. The actor never changes during that warmup.
    """
    if (type(completed_updates) != int or completed_updates < 0
            or warmup_completed != NATIVE_PPO_SETTINGS['critic_warmup_updates']):
        raise ValueError('Native checkpoint requires the completed Critic-only warmup')
    folder = Path(root).resolve()/'checkpoints'
    folder.mkdir(exist_ok=True)
    receipt_path = folder/f'update_{completed_updates:06d}.json'
    if receipt_path.exists():
        raise ValueError('Native PPO update is already committed')
    path = folder/f'update_{completed_updates:06d}_{uuid.uuid4().hex}.pt'
    actor = learner.policy.actor
    prior = _v4_policy_contract(learner.policy)
    value = dict(kind=NATIVE_PPO_STATE_KIND, ppo_algorithm=NATIVE_PPO_ALGORITHM,
        actor=actor.state_dict(), policy=learner.policy.state_dict(),
        optimizer=learner.optimizer.state_dict(), learning_rate=float(learner.learning_rate),
        amp_discriminator=learner.amp_discriminator.state_dict(), disc_optimizer=learner.disc_optimizer.state_dict(),
        disc_obs_buffer=_ring_state(learner.disc_obs_buffer),
        disc_demo_obs_buffer=_ring_state(learner.disc_demo_obs_buffer),
        counts=list(learner.counts), rng=_native_rng(), completed_updates=completed_updates,
        warmup_completed=warmup_completed, manifest_sha256=manifest_sha256,
        schema=actor.schema, architecture=actor.architecture, metrics=metrics,
        simulator_state_restored=False, **prior)
    if training_reward is not None:
        value['training_reward'] = training_reward
    publish(path, value)
    record = dict(completed_updates=completed_updates, manifest_sha256=manifest_sha256,
        ppo_algorithm=NATIVE_PPO_ALGORITHM, state_kind=NATIVE_PPO_STATE_KIND,
        teacher_checkpoint_sha256=prior['teacher_checkpoint_sha256'], warmup_completed=warmup_completed,
        path=path.name, sha256=sha(path), metrics=metrics)
    if training_reward is not None:
        record['training_reward'] = training_reward
    write_new(receipt_path, record)
    return record


def save_native_metrics(root, learner, completed_updates, manifest_sha256, metrics=None, *,
                        warmup_completed=10, training_reward=None):
    """Record every update without serializing or hashing learner/replay tensors.

    These rows make a dense metrics history, not a restore point. If a process
    dies before the next full checkpoint, only its trailing metric receipts are
    archived/replayed; committed checkpoint history is never rewritten.
    """
    if (type(completed_updates) is not int or completed_updates <= 0
            or completed_updates % NATIVE_CHECKPOINT_INTERVAL == 0
            or warmup_completed != NATIVE_PPO_SETTINGS['critic_warmup_updates']):
        raise ValueError('Native metrics-only receipt cannot replace a scheduled full checkpoint')
    folder = Path(root).resolve()/'checkpoints'
    folder.mkdir(exist_ok=True)
    prior = _v4_policy_contract(learner.policy)
    record = dict(completed_updates=completed_updates, manifest_sha256=manifest_sha256,
        ppo_algorithm=NATIVE_PPO_ALGORITHM, state_kind=NATIVE_PPO_STATE_KIND,
        teacher_checkpoint_sha256=prior['teacher_checkpoint_sha256'], warmup_completed=warmup_completed,
        record_kind=NATIVE_METRICS_RECEIPT_KIND, metrics=metrics)
    if training_reward is not None:
        record['training_reward'] = training_reward
    write_new(folder/f'update_{completed_updates:06d}.json', record)
    return record


def _native_full_history(history):
    """Recognize old full V4 receipts and new explicit metrics-only receipts."""
    full = []
    for row in history:
        if (row.get('ppo_algorithm') != NATIVE_PPO_ALGORITHM
                or row.get('state_kind') != NATIVE_PPO_STATE_KIND):
            raise ValueError('Cannot resume legacy PPO or native V3 state as a native expert-prior V4 learner')
        if row.get('record_kind') == NATIVE_METRICS_RECEIPT_KIND:
            if ('path' in row or 'sha256' in row
                    or row['completed_updates'] % NATIVE_CHECKPOINT_INTERVAL == 0):
                raise ValueError('Invalid native metrics-only receipt')
        elif (row.get('record_kind') is None and isinstance(row.get('path'), str)
              and isinstance(row.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
            full.append(row)
        else:
            raise ValueError('Invalid native full checkpoint receipt')
    if history and (not full or full[0]['completed_updates'] != 0):
        raise ValueError('Native PPO history must start with a full checkpoint')
    return full


def native_checkpoint_history_for_resume(root, manifest_sha256, *, recovery_dir):
    """Reconcile a dead session's metrics tail to the last real checkpoint.

    The prior every-update V4 format remains a valid dense prefix. Trailing
    lightweight receipts are moved, never destroyed, into a fresh recovery
    directory under this run. Highest update first leaves a contiguous active
    prefix even if recovery itself is interrupted. Verify the actual checkpoint
    before changing any history.
    """
    root = Path(root).resolve()
    folder = root/'checkpoints'
    history = checkpoint_history(root, manifest_sha256)
    if not history:
        return history
    full = _native_full_history(history)
    latest = full[-1]
    if latest is history[-1]:
        return history
    checkpoint = _checkpoint_blob(folder, latest)
    if not checkpoint.is_file() or sha(checkpoint) != latest['sha256']:
        raise ValueError('Committed native PPO checkpoint changed')
    recovered = Path(recovery_dir).absolute()
    if (folder.resolve() != folder or recovered.resolve() != recovered
            or not recovered.is_relative_to(root/'attempts')):
        raise ValueError('Native checkpoint recovery must stay in this run attempt directory')
    tail = history[latest['completed_updates']+1:]
    paths = [folder/f'update_{row["completed_updates"]:06d}.json' for row in reversed(tail)]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError('Invalid native metrics recovery receipt path')
    recovered.mkdir(parents=True, exist_ok=False)
    # Persist the archival directory's entry before removing any receipt from
    # the source. This also makes a nested newly-created attempt recoverable.
    for directory in recovered.parents:
        _sync_directory(directory)
        if directory == root:
            break
    for path in paths:
        os.replace(path, recovered/path.name)
        _sync_directory(recovered)
        _sync_directory(folder)
    return history[:latest['completed_updates']+1]


def load_native_checkpoint(root, learner, manifest_sha256):
    folder = Path(root).resolve()/'checkpoints'
    history = checkpoint_history(root, manifest_sha256)
    if not history:
        return None
    record = _native_full_history(history)[-1]
    if record is not history[-1]:
        raise ValueError('Native PPO metrics tail requires reconciliation before restore')
    state_path = _checkpoint_blob(folder, record)
    if sha(state_path) != record['sha256']:
        raise ValueError('Committed native PPO checkpoint changed')
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    actor = learner.policy.actor
    prior = _v4_policy_contract(learner.policy)
    if (state.get('kind') != NATIVE_PPO_STATE_KIND or
            any(state.get(name) != record.get(name) for name in (
                'ppo_algorithm', 'completed_updates', 'manifest_sha256', 'warmup_completed',
                'metrics', 'training_reward', 'teacher_checkpoint_sha256')) or state.get('schema') != actor.schema
            or state.get('architecture') != actor.architecture or state.get('counts') != list(learner.counts)
            or any(state.get(name) != value for name, value in prior.items())
            or state.get('warmup_completed') != NATIVE_PPO_SETTINGS['critic_warmup_updates']):
        raise ValueError('Native PPO learning-state identity mismatch')
    if not _finite_state({key: state[key] for key in (
            'actor', 'policy', 'optimizer', 'learning_rate', 'amp_discriminator', 'disc_optimizer',
            'disc_obs_buffer', 'disc_demo_obs_buffer')}):
        raise ValueError('Nonfinite restored native PPO learning state')
    learner.policy.load_state_dict(state['policy'], strict=True)
    _v4_policy_contract(learner.policy)
    if any(not torch.equal(value.cpu(), state['actor'][name]) for name, value in actor.state_dict().items()):
        raise ValueError('Native PPO exported actor does not match native policy state')
    learner.optimizer.load_state_dict(state['optimizer'])
    learner.learning_rate = float(state['learning_rate'])
    if (learner.learning_rate <= 0 or
            any(group['lr'] != learner.learning_rate for group in learner.optimizer.param_groups)):
        raise ValueError('Native PPO adaptive learning rate does not match restored Adam state')
    learner.amp_discriminator.load_state_dict(state['amp_discriminator'], strict=True)
    learner.disc_optimizer.load_state_dict(state['disc_optimizer'])
    _restore_ring(learner.disc_obs_buffer, state['disc_obs_buffer'])
    _restore_ring(learner.disc_demo_obs_buffer, state['disc_demo_obs_buffer'])
    _restore_native_rng(state['rng'])
    return {name: record[name] for name in ('completed_updates', 'warmup_completed')}


def prune_native_checkpoints(root, history, retain_updates=(0, 100, 200, 500, 800, 1000)):
    """Keep milestones + the most recent two committed full native states.

    The receipts and training metrics of *every* update are retained. Legacy
    experiments, uncommitted blobs, and files beyond this exact run are untouched.
    """
    folder = Path(root).resolve()/'checkpoints'
    if folder.resolve() != folder or not history:
        raise ValueError('Native PPO checkpoint directory/history differs')
    full = _native_full_history(history)
    latest = full[-1]
    if latest is not history[-1]:
        # No new full state has been committed; avoid repeated hashing and
        # never mistake metrics-only progress for a durable rollback point.
        return
    if sha(_checkpoint_blob(folder, latest)) != latest['sha256']:
        raise ValueError('Committed native PPO checkpoint changed')
    _sync_directory(folder)
    obsolete = {row['path']: row for row in full[:-2]
                if row['completed_updates'] not in retain_updates}
    # Only touch existing committed blobs (at most a few) rather than walking
    # a growing O(updates) list of already removed files on every update.
    for path in folder.glob('update_*.pt'):
        if path.name in obsolete:
            _checkpoint_blob(folder, obsolete[path.name]).unlink()


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


def prune_mixed_checkpoints(root, history, *, reconcile=False):
    """Keep the initial and latest two learning states for the in-process sampler."""
    folder = root/'checkpoints'
    if folder.resolve() != folder:
        raise ValueError('Mixed PPO checkpoint directory must be inside its run')
    obsolete = history[1:-2] if reconcile else history[-3:-2] if len(history) >= 4 else []
    _sync_directory(folder)
    for receipt in obsolete:
        _checkpoint_blob(folder, receipt).unlink(missing_ok=True)


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
