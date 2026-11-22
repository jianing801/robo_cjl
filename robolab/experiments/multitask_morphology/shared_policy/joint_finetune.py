"""Exact-candidate, resumable three-task PPO with optional independent evaluation."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from contract import load_plan, sha, write_new
from finetune_state import (select_candidate, save_checkpoint, load_checkpoint, checkpoint_history,
                            prune_training_artifacts, validate_batch)
from persistence import run_lock, resume_pause
from randomization import protocol_metadata, assert_protocol_unchanged
from student import load_student, save_student
from training_reward import RolloutRewards, reward_history_rows, write_training_reward_csv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
STEPS_PER_TASK = 24
ENVS_PER_TASK = 8
ACTION_STD = .05
# Exact predecessors used by the compact 200-update Linux BO runs.
LEGACY_BO_COORDINATORS = (
    '306da1b0c983a6683929c1c643d23a83f999f868be697be08a5c09f8e32d856c',
    '304cdd7c22039d3dc5eff837cf449eb1ff33c96c106dcffa92f4afed5bb4f48d',
)
LEGACY_BO_SOURCES = {
    'finetune_state.py': '7b958616ceb651b059a7fef4c40bf3356b134a3dfe56a82bf1f64448b703efad',
    'ppo_channel.py': '87d6f011074fc920b365e1b24cdaa5a05913493672a14535900df96fc3c06cfb',
    'pipeline.py': 'fffd518a08c6beeeb4348816d7baab3ba208f05572a302c86bf564da10351777',
}


def worker_command(plan, student, root, task, design_id, profile, strength, seed, full_body_torque=False):
    script = HERE / ('rollout_amp.py' if task == 'run' else 'rollout.py')
    cmd = [sys.executable, '-u', str(script), '--plan', str(plan), '--design-ids', design_id,
        '--output', str(Path(root)/f'worker_{task}'), '--mode', 'evaluate', '--student', str(student),
        '--ppo-channel', str(root), '--profile', profile, '--randomization-strength', str(strength),
        '--headless', '--device', 'cuda:0', '--seed', str(seed)]
    if task != 'run': cmd += ['--task', task]
    if full_body_torque: cmd += ['--full-body-torque']
    return cmd


def returns(reward, done, timeout, value, last, gamma=.99, lam=.95):
    result = torch.empty_like(reward); adv = torch.zeros_like(last)
    reward = reward + gamma * value * timeout.float()
    for t in reversed(range(len(reward))):
        nxt = last if t == len(reward)-1 else value[t+1]
        live = 1-done[t].float()
        adv = reward[t] + gamma*nxt*live-value[t] + gamma*lam*live*adv
        result[t] = adv+value[t]
    return result


def ppo_update(model, critic, optimizer, batches):
    prepared = []
    for data in batches:
        x = data['obs']; shape = x.shape[:2]
        with torch.no_grad():
            value = critic(x.flatten(0, 1)).reshape(shape)
            last = critic(data['next_obs']).squeeze(-1)
            target = returns(data['reward'], data['done'], data['timeout'], value, last)
            advantage = target-value
            advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        prepared.append((x.flatten(0, 1), data['actions'].flatten(0, 1), data['logp'].flatten(),
                         target.flatten(), advantage.flatten()))
    x, actions, oldlog, target, advantage = [torch.cat(parts) for parts in zip(*prepared)]
    losses = []; model.train(); critic.train()
    for _ in range(4):
        for ids in torch.randperm(len(x)).split(192):
            distribution = torch.distributions.Normal(model(x[ids]), ACTION_STD)
            ratio = (distribution.log_prob(actions[ids]).sum(-1)-oldlog[ids]).exp()
            policy = -torch.minimum(ratio*advantage[ids], ratio.clamp(.8, 1.2)*advantage[ids]).mean()
            loss = policy+(critic(x[ids]).squeeze(-1)-target[ids]).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *critic.parameters()], 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite PPO update')
            optimizer.step(); losses.append(float(loss.detach()))
    return dict(loss=sum(losses)/len(losses), samples=len(x))


def check_pause(root):
    if (Path(root)/'USER_PAUSED').exists():
        raise KeyboardInterrupt('USER_PAUSED')


def train_updates(a, root, plan, model, critic, optimizer, completed, manifest_sha256, protocol, environment):
    from pipeline import stop_child
    attempt = root/'attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    attempt_number = len(list((root/'attempts').iterdir()))-1
    worker_seed = (a.seed + completed*1009 + attempt_number*9176) % 2**32
    while worker_seed in a.evaluation_seeds: worker_seed = (worker_seed+1) % 2**32
    session = dict(kind='shared_ppo_session_v2', session_id=attempt.name, control_root=str(root),
        start_update=completed, manifest_sha256=manifest_sha256, collection_protocol=protocol,
        tasks=list(plan['teachers']), steps_per_task=STEPS_PER_TASK, std=ACTION_STD,
        worker_seed=worker_seed, simulator_state_restored=False,
        resume_scope='actor, critic, Adam, coordinator RNG and update count; fresh simulator episodes')
    write_new(attempt/'session.json', session)
    rewards = RolloutRewards()
    reward_rows = reward_history_rows(checkpoint_history(root, manifest_sha256), session['tasks'])
    workers = []; handles = []
    try:
        for task, spec in plan['teachers'].items():
            check_pause(root)
            command = worker_command(root/'plan.json', a.student, attempt, task, spec['design_ids'][0],
                a.profile, a.randomization_strength, worker_seed, a.full_body_torque)
            logs = [open(attempt/f'{task}.{suffix}.log', 'x', encoding='utf8') for suffix in ('stdout', 'stderr')]
            handles.extend(logs)
            options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
            process = subprocess.Popen(command, stdout=logs[0], stderr=logs[1],
                env=dict(environment, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'), **options)
            workers.append(process)
            write_new(attempt/f'{task}.launch.json', dict(pid=process.pid, command=command))
            time.sleep(3)
        for index in range(completed, a.updates):
            check_pause(root); assert_protocol_unchanged(protocol)
            actor = attempt/f'actor_{index:03d}.pt'
            save_student(actor, model, ppo_updates=index, collection_protocol=protocol, manifest_sha256=manifest_sha256)
            actor_sha256 = sha(actor)
            write_new(attempt/f'request_{index:03d}.json', dict(kind='shared_ppo_request_v2',
                session_id=session['session_id'], manifest_sha256=manifest_sha256, index=index,
                student=str(actor), student_sha256=actor_sha256, std=ACTION_STD, steps=STEPS_PER_TASK,
                collection_protocol=protocol))
            started = time.monotonic()
            while not all((attempt/f'{task}_{index:03d}.pt').exists() for task in plan['teachers']):
                check_pause(root)
                for process in workers:
                    if process.poll() is not None:
                        raise RuntimeError(f'Physical worker exited {process.pid}: {process.returncode}; see {attempt}')
                if time.monotonic()-started > a.timeout_seconds:
                    raise TimeoutError('Physical task batch timeout')
                time.sleep(.2)
            batches = [validate_batch(torch.load(attempt/f'{task}_{index:03d}.pt', weights_only=True),
                task=task, index=index, session=session, actor_sha256=actor_sha256, width=model.width)
                for task in plan['teachers']]
            check_pause(root)
            training_reward = rewards.summarize(batches, session['session_id'])
            metrics = ppo_update(model, critic, optimizer, batches)
            check_pause(root); assert_protocol_unchanged(protocol)
            rollout = None
            if a.compact_training_artifacts:
                rollout = dict(session_id=attempt.name,
                    task_audits={batch['task']: batch.get('full_body_audit') for batch in batches})
            receipt = save_checkpoint(root, model, critic, optimizer, index+1, manifest_sha256, metrics,
                                      rollout=rollout, training_reward=training_reward)
            reward_rows.extend(reward_history_rows([receipt], session['tasks']))
            write_training_reward_csv(root, reward_rows)
            if rollout is not None:
                prune_training_artifacts(root, manifest_sha256, index+1)
            print('JOINT_PPO_UPDATE', index+1, '/', a.updates, flush=True)
    finally:
        (attempt/'STOP').touch(exist_ok=True)
        try:
            for process in workers:
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: stop_child(process)
        finally:
            for handle in handles: handle.close()


def evaluate_candidate(a, plan, root, actor, candidate, environment):
    from pipeline import run_evaluation
    from evaluation import aggregate_evaluations, evaluation_criteria
    jobs = [(task, spec['design_ids']) for task, spec in plan['teachers'].items()]
    profiles = ('nominal', 'robust') if a.evaluation_profile == 'both' else (a.evaluation_profile,)
    criteria = dict(min_command_progress_ratio=a.min_command_progress_ratio)
    results = {}
    for profile in profiles:
        records = []
        for seed in a.evaluation_seeds:
            check_pause(root)
            records.extend(run_evaluation(plan, root/'plan.json', root, jobs, str(actor), profile,
                1. if profile == 'robust' else 0., seed,
                f'post_finetune/{profile}/seed_{seed}', environment, a.timeout_seconds, criteria))
        results[profile] = dict(task_metrics=aggregate_evaluations(records, plan['schema']['tasks']),
                                evaluations=records)
    return dict(kind='shared_candidate_evaluation_v1', candidate=candidate,
        student=str(actor), student_sha256=sha(actor), ppo_updates=a.updates,
        evaluation_criteria=evaluation_criteria(**criteria), profiles=results,
        evaluation_seeds=a.evaluation_seeds, requires_evaluation=False,
        bo_interface='Per-task post-finetuning observations; no invented multi-task scalar objective',
        scalar_objective=None, requires_bo_objective_definition=True)


def publish_finetuned_actor(a, root, model, manifest_sha256, protocol):
    actor = root/'shared_actor_finetuned.pt'
    expected = dict(ppo_updates=a.updates, initial_actor_sha256=sha(a.student),
                    collection_protocol=protocol, manifest_sha256=manifest_sha256)
    if actor.exists():
        previous, metadata = load_student(actor)
        previous_state = previous.state_dict()
        if (any(metadata.get(key) != value for key, value in expected.items())
                or previous.schema != model.schema or previous.architecture != model.architecture
                or any(not torch.equal(value, previous_state[key]) for key, value in model.state_dict().items())):
            raise ValueError('Published finetuned actor differs from the committed learning state')
    else:
        save_student(actor, model, **expected)
    return actor


def publish_training_completion(a, root, model, candidate, manifest_sha256, protocol):
    check_pause(root)
    assert_protocol_unchanged(protocol)
    actor = publish_finetuned_actor(a, root, model, manifest_sha256, protocol)
    commit = root/f'checkpoints/update_{a.updates:06d}.json'
    record = dict(result='CANDIDATE_FINETUNED', candidate=candidate, ppo_updates=a.updates,
                  manifest_sha256=manifest_sha256, student=str(actor), student_sha256=sha(actor),
                  checkpoint_sha256=sha(commit))
    receipt = root/'training_completed.json'
    if receipt.exists():
        if json.loads(receipt.read_text(encoding='utf8')) != record:
            raise ValueError('Published training completion record changed')
    else:
        write_new(receipt, record)
    print('CANDIDATE_FINETUNED', receipt, flush=True)


def run(a):
    original = load_plan(a.plan)
    if original['schema']['tasks'] != ['stairs', 'backpack', 'run']:
        raise ValueError('Candidate finetuning requires the three-task shared plan')
    plan, candidate = select_candidate(original, design_id=a.design_id,
        design_task=a.design_task, design_key=a.design_key)
    if a.full_body_torque and any(not s['full_body_torque'] for s in plan['teachers'].values()):
        raise ValueError('Cannot override frozen teacher physics')
    protocol = protocol_metadata(a.profile, a.randomization_strength)
    root = Path(a.output).resolve()
    sources = ('joint_finetune.py', 'finetune_state.py', 'ppo_channel.py', 'persistence.py', 'contract.py',
               'student.py', 'pipeline.py', 'evaluation.py', 'training_reward.py')
    manifest = dict(kind='shared_candidate_finetuning_v2', plan_sha256=sha(a.plan),
        initial_actor_sha256=sha(a.student), candidate=candidate, updates=a.updates,
        full_body_torque=a.full_body_torque, collection_protocol=protocol, seed=a.seed,
        evaluation_profile=a.evaluation_profile, evaluation_seeds=a.evaluation_seeds,
        robust_evaluation_strength=1.,
        min_command_progress_ratio=a.min_command_progress_ratio,
        sources={name: sha(HERE/name) for name in sources},
        steps_per_task=STEPS_PER_TASK, envs_per_task=ENVS_PER_TASK, std=ACTION_STD,
        actor_lr=3e-6, critic_lr=1e-3, teacher_actor_updates=0,
        method='joint clipped PPO; all three tasks collected before each shared update',
        resume_scope='Committed PPO update; all learning state restored, simulator episodes restarted')
    if a.compact_training_artifacts:
        manifest['compact_training_artifacts'] = True
    if a.resume:
        frozen_manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
        if frozen_manifest != manifest:
            previous_sources = {name: manifest['sources'][name] for name in sources if name != 'training_reward.py'}
            previous_sources.update(LEGACY_BO_SOURCES)
            if not (a.training_only and a.compact_training_artifacts and a.updates == 200
                    and any(frozen_manifest == dict(manifest, sources={**previous_sources, 'joint_finetune.py': old})
                            for old in LEGACY_BO_COORDINATORS)):
                raise ValueError('Resume candidate, source, initial actor, training/evaluation budget or protocol changed')
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_new(root/'manifest.json', manifest)
    frozen = root/'plan.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf8')) != plan:
            raise ValueError('Frozen candidate plan changed')
    else:
        write_new(frozen, plan)
    load_plan(frozen)
    manifest_sha256 = sha(root/'manifest.json')
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    model, _ = load_student(a.student)
    if model.schema != plan['schema']:
        raise ValueError('Initial actor schema differs from candidate plan')
    critic = torch.nn.Sequential(torch.nn.Linear(model.width, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 1))
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 3e-6},
                                  {'params': critic.parameters(), 'lr': 1e-3}])
    completed = load_checkpoint(root, model, critic, optimizer, manifest_sha256)
    if completed is None:
        completed = 0
        save_checkpoint(root, model, critic, optimizer, 0, manifest_sha256)
    if completed > a.updates:
        raise ValueError('Checkpoint exceeds the frozen PPO budget')
    resume_pause(root, a.resume_paused)
    if completed:
        history = checkpoint_history(root, manifest_sha256)
        write_training_reward_csv(root, reward_history_rows(history, plan['teachers']))
    if not a.training_only and (root/'completed.json').exists():
        record = json.loads((root/'completed.json').read_text(encoding='utf8'))
        if (record['manifest_sha256'] != manifest_sha256 or record['ppo_updates'] != completed
                or sha(record['student']) != record['student_sha256']
                or sha(record['evaluation']) != record['evaluation_sha256']):
            raise ValueError('Completed candidate result changed')
        print('CANDIDATE_ALREADY_COMPLETE', root); return
    if a.compact_training_artifacts and completed:
        prune_training_artifacts(root, manifest_sha256, completed, reconcile=True)
    if a.training_only and completed == a.updates:
        publish_training_completion(a, root, model, candidate, manifest_sha256, protocol)
        return
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    choices = candidates(query_gpus(), os.environ, min_free_mib=a.gpu_min_free_gb*1024,
                         max_utilization=a.gpu_max_utilization, requested=a.device)
    with gpu_lease(choices, HERE.parents[2]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        environment = child_environment(gpu, os.environ)
        if completed < a.updates:
            train_updates(a, root, plan, model, critic, optimizer, completed, manifest_sha256, protocol, environment)
        if a.training_only:
            if load_checkpoint(root, model, critic, optimizer, manifest_sha256) != a.updates:
                raise ValueError('Training has no committed checkpoint at the frozen PPO budget')
            publish_training_completion(a, root, model, candidate, manifest_sha256, protocol)
            return
        check_pause(root); assert_protocol_unchanged(protocol)
        actor = publish_finetuned_actor(a, root, model, manifest_sha256, protocol)
        result = evaluate_candidate(a, plan, root, actor, candidate, environment)
        assert_protocol_unchanged(protocol)
        result['manifest_sha256'] = manifest_sha256
        evaluation = root/'candidate_result.json'
        if evaluation.exists():
            if json.loads(evaluation.read_text(encoding='utf8')) != result:
                raise ValueError('Published candidate evaluation changed')
        else: write_new(evaluation, result)
        write_new(root/'completed.json', dict(result='CANDIDATE_FINETUNED_EVALUATED',
            candidate=candidate, ppo_updates=a.updates, manifest_sha256=manifest_sha256,
            student=str(actor), student_sha256=sha(actor), evaluation=str(evaluation),
            evaluation_sha256=sha(evaluation), requires_evaluation=False))
        print('CANDIDATE_FINETUNED_EVALUATED', evaluation, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'student', 'output'): parser.add_argument('--'+name, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--design-id', help='Exact structure ID in the selected native task bank')
    selector.add_argument('--design-key', nargs=4, metavar=('THIGH_M', 'CALF_M', 'KNEE', 'ANKLE'))
    parser.add_argument('--design-task', choices=['stairs', 'backpack', 'run'], default='stairs')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--compact-training-artifacts', action='store_true',
                        help='After committed updates, retain only initial and latest two learning states and rollout receipts')
    parser.add_argument('--training-only', action='store_true',
                        help='Publish the committed PPO actor for formal scoring without standalone gait diagnostics')
    parser.add_argument('--full-body-torque', action='store_true')
    parser.add_argument('--profile', choices=['nominal', 'robust'], default='robust')
    parser.add_argument('--randomization-strength', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-paused', action='store_true')
    parser.add_argument('--evaluation-profile', choices=['nominal', 'robust', 'both'], default='both')
    parser.add_argument('--evaluation-seeds', default='20261118')
    parser.add_argument('--tracking-rel-tolerance', type=float,
                        help='Deprecated. Accepted for compatibility and ignored.')
    parser.add_argument('--min-command-progress-ratio', type=float, default=.5)
    parser.add_argument('--device')
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--gpu-min-free-gb', type=float, default=4.)
    parser.add_argument('--gpu-max-utilization', type=int, default=30)
    a = parser.parse_args(argv)
    if a.resume_paused: a.resume = True
    a.evaluation_seeds = [int(v) for v in a.evaluation_seeds.split(',')]
    if (a.updates < 1 or not 0 <= a.randomization_strength <= 1 or not 0 <= a.seed < 2**32
            or not a.evaluation_seeds or len(set(a.evaluation_seeds)) != len(a.evaluation_seeds)
            or any(not 0 <= s < 2**32 or s == a.seed for s in a.evaluation_seeds)
            or a.timeout_seconds < 1):
        raise ValueError('Invalid PPO budget, randomization or independent evaluation seeds')
    from evaluation import evaluation_criteria
    evaluation_criteria(min_command_progress_ratio=a.min_command_progress_ratio)
    if a.profile == 'nominal': a.randomization_strength = 0.
    torch.set_num_threads(2)
    with run_lock(a.output):
        try: return run(a)
        except KeyboardInterrupt:
            if Path(a.output).is_dir(): (Path(a.output)/'USER_PAUSED').touch(exist_ok=True)
            raise


if __name__ == '__main__': main()
