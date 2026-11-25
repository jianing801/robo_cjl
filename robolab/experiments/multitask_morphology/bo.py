"""Evaluate and search exact structures after shared-policy distillation."""
import argparse
from copy import copy
import json
from pathlib import Path
from statistics import fmean, stdev
from math import isclose, isfinite
import os
import re
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED = HERE/'shared_policy'
DEFAULT_ROOT = REPO/'robolab/logs/multitask_full2500_physical'
MIGRATED_ROOT = DEFAULT_ROOT.with_name('multitask_full2500_physical_migrated')
DEFAULT_BO_UPDATES = 100
PREVIOUS_BO_UPDATES = 200
PRE_TRAINING_ONLY_BO_SHA256 = '203b9aada3b4615464b04491ba61f9d59ecd27219f0f061e8540661b4c61d053'
# LF and CRLF coordinator hashes before the measurement-import default changed.
PRE_INGEST_ALIGNMENT_BO_SHA256 = {
    'fead4dca36355e514f86981a1526cd79c1bb96ce7ee12a7c8e7d16cde9773233',
    'af10e96ee860c92e62f88f487d19d0d6984b4a32f4bbec27ef5e39f6703d336b',
}
PHYSICS_CONDITIONS = HERE/'configs/bo_conditions_physics_step.json'
FOUR_METRIC_WEIGHTS_PATH = HERE/'configs/bo_metric_weights_four.json'
ORIGINAL_RPO_ARCHIVE = REPO/'docs/reference/rpo_original_configuration'
ORIGINAL_RPO_SCORING_METHOD = 'post_finetune_original_rpo'
FORMAL_CONDITIONS = HERE/'configs/bo_conditions_single_seed.json'
LEGACY_FORMAL_CONDITIONS = HERE/'configs/bo_conditions.json'
# Exact LF and CRLF source sets preceding the fixed-reset protocol.
PRE_FIXED_BO_SOURCES = (
    (('0942320ad840a49a921bf014c57d033a6322f86e34043e50c28d6310ea06879d', PRE_TRAINING_ONLY_BO_SHA256),
     '5aacc6dc340839ed76db3a4d3e4e92f07d33b9e88911c538ceba482cad68907e',
     'c064cc6b15b03eb8869a38a567b0a1bae55f8ba93246fb13fc05e3de87e9445e'),
    (('4bfc4fc4f674b2f3b1c41b9ab888a29bb4fd998088fbdf4f28cf8b450c72d6bf',
      'ec8b7ddea5062dd043aa08f73359ea7a89aa62dccff49559d307286027f8e9dc'),
     '781e3fa5a46a4d080565a58335179502a2a0dbcb08f1d6778c8f2511a7978d08',
     '9e67388022469f694f0d0f38716379cf8f5568cf3803a6793eea9fd4416fcb5b'),
)
PILOT_CONDITIONS = HERE/'configs/bo_pilot_conditions.json'
sys.path[:0] = [str(SHARED), str(REPO/'robolab'), str(REPO)]

from shared_bo_metrics import (ABSOLUTE_COT_BASIS, DT, FORMAL_TASK_WEIGHTS, FOUR_METRICS,
                               FOUR_METRIC_SCORING_METHOD, FOUR_METRIC_SCORING_VERSION, KNEE_ANKLE, METRICS, PHYSICS_DT,
                               POSITIVE_COT_BASIS, REPORT_VERSION, SCORING_VERSION, TASKS, TRIAL_VERSION,
                               digest, objective, paired_trials, positive, read_conditions, score_trial,
                               scoring_cot_basis, trial_metric, validate_four_metric_weights, validate_scoring)
from contract import load_plan, sha, write_new
from collection_grid import SELECTION_VERSION, calibration_keys, select_collection_grid, selection_panel
from evaluation import EVALUATION_VELOCITY_FRAME
from finetune_settings import (MIXED_BO_ENVS, MIXED_BO_LAYOUT, PPO_SCHEDULE,
                               NATIVE_PPO_ALGORITHM, NATIVE_PPO_SETTINGS,
                               NATIVE_PPO_PROFILES, NATIVE_PPO_STATE_KIND)
from persistence import resume_pause, run_lock

ORIGINAL_RPO_FOUR_METRIC_METHOD = FOUR_METRIC_SCORING_METHOD
ORIGINAL_RPO_SCORING_METHODS = (ORIGINAL_RPO_SCORING_METHOD, ORIGINAL_RPO_FOUR_METRIC_METHOD)
from stair_course import COURSE, evaluation_flight, outside_flight


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def freeze(path, data):
    if path.is_file():
        if read(path) != data:
            raise ValueError(f'Frozen BO identity differs: {path}')
    else:
        write_new(path, data)


def freeze_candidate_manifest(path, identity, *, fixed_reset=False):
    if path.is_file():
        saved = read(path)
        coordinator = saved.get('sources', {}).get('bo.py')
        if (coordinator in PRE_INGEST_ALIGNMENT_BO_SHA256
                and saved == dict(identity, sources={**identity['sources'], 'bo.py': coordinator})):
            return
        if not fixed_reset:
            for coordinators, capture, metrics in PRE_FIXED_BO_SOURCES:
                if coordinator in coordinators:
                    previous_sources = {**identity['sources'], 'bo.py': coordinator,
                                        'bo_capture.py': capture, 'shared_bo_metrics.py': metrics}
                    if saved == dict(identity, sources=previous_sources):
                        return
    freeze(path, identity)


def official_conditions_path(conditions_sha256):
    for path in (PHYSICS_CONDITIONS, FORMAL_CONDITIONS, LEGACY_FORMAL_CONDITIONS):
        if digest(read_conditions(path)) == conditions_sha256:
            return path
    raise ValueError('Official BO requires the fixed three-task conditions or the archived three-seed conditions')


def uses_current_schedule(updates, conditions):
    return updates == DEFAULT_BO_UPDATES and conditions.get('reset_mode') == 'fixed'


def calibration_label(conditions, updates=DEFAULT_BO_UPDATES):
    if conditions.get('reset_mode') != 'fixed':
        return 'calibration50'
    label = 'calibration50_single_seed_ppo100_random' if uses_current_schedule(updates, conditions) else 'calibration50_single_seed'
    return label + ('_200hz' if 'measurement_dt_s' in conditions else '')


def check_formal_selection(root, plan, manifest, complete, student, *, require_horizontal_selection=False):
    protocol = manifest.get('protocol', {})
    arguments = manifest.get('arguments', {})
    version = protocol.get('selection_version')
    if version != SELECTION_VERSION:
        raise ValueError(f'Formal BO requires selection protocol {SELECTION_VERSION}, found {version!r}')
    if protocol.get('stair_evaluation') != COURSE:
        raise ValueError('Formal BO requires round selection on the timed ascending stair flight')
    if require_horizontal_selection and protocol.get('evaluation_velocity_frame') != EVALUATION_VELOCITY_FRAME:
        raise ValueError('Formal BO requires selection in the horizontal heading frame')
    if (digest(protocol) != manifest.get('protocol_sha256') or complete.get('protocol') != protocol
            or complete.get('result') != 'DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING'
            or complete.get('full_bank_evaluated') is not False or protocol.get('dagger_rounds') != 3
            or arguments.get('epochs') != 10):
        raise ValueError('Formal BO requires the completed three-round DAgger final-epoch selection')
    selection_path = root/'shared_policy/selection.json'
    if (Path(complete['selection']).resolve() != selection_path.resolve()
            or not selection_path.is_file() or complete.get('selection_sha256') != sha(selection_path)):
        raise ValueError('Selected student refers to a different selection record')
    selection = read(selection_path)
    fixed = calibration_keys()
    training = select_collection_grid(plan['selected_structure_keys'])
    panel = selection_panel(plan['selected_structure_keys'], training, fixed)
    scope = protocol['collection_scope']
    budget = protocol.get('selection_budget', {})
    expected = list(map(list,panel))
    reserved = list(map(list,fixed))
    seen = set(training)
    if (budget.get('selection_design_count') != len(panel) or selection.get('budget') != budget
            or manifest.get('selection_budget') != budget
            or scope.get('structure_keys') != list(map(list,training)) or scope.get('environments_per_task') != 4096
            or scope.get('steps_per_rollout') != 250 or selection.get('design_keys') != expected
            or protocol.get('selection_keys') != expected or protocol.get('calibration_keys') != reserved
            or selection.get('calibration_keys') != reserved
            or selection.get('training_design_keys') != [list(k) for k in panel if k in seen]
            or selection.get('profile') != 'robust' or protocol['selection'].get('profile') != 'robust'
            or protocol['selection'].get('strength') != 1.
            or selection.get('seeds') != [20261117,20261119,20261120]
            or protocol.get('selection_seeds') != selection['seeds']):
        raise ValueError('Selection panel, reserved calibration panel or collection protocol differs')
    candidates = selection.get('candidates', [])
    if ([r['round_index'] for r in candidates] != [1,2,3] or any(r['epoch'] != 9 for r in candidates)):
        raise ValueError('Expected the three final DAgger checkpoints')
    winner = max(candidates,key=lambda row: tuple(row['rank']))
    if (winner['round_index'] != selection['selected_round'] or winner['round_index'] != complete['selected_round']
            or Path(selection['selected_student']).resolve() != student or Path(winner['student']).resolve() != student
            or winner['student_sha256'] != complete['student_sha256']):
        raise ValueError('Selected DAgger checkpoint differs from the reported winner')


def shared_input(root, *, formal=False, require_horizontal_selection=False, continuation=None):
    path = root/'shared_plan/plan.json'
    plan = load_plan(path)
    manifest = read(root/'shared_policy/manifest.json')
    complete = read(root/'shared_policy/completed.json')
    student = Path(complete['student']).resolve()
    if (tuple(plan['schema']['tasks']) != TASKS or manifest.get('plan_sha256') != sha(path)
            or complete.get('protocol_sha256') != manifest.get('protocol_sha256')
            or sha(student) != complete['student_sha256']):
        raise ValueError('Shared three-task plan or selected student identity does not match')
    if formal:
        check_formal_selection(root, plan, manifest, complete, student,
                               require_horizontal_selection=require_horizontal_selection)
    if continuation is not None:
        if not formal or not require_horizontal_selection:
            raise ValueError('An appended round requires the formal horizontal selection')
        from dagger_bo_input import continuation_actor
        student = continuation_actor(continuation, root/'shared_policy', path, plan, student)
    return path, plan, student


def actor_receipt(root):
    training = root/'training_completed.json'
    return training if training.is_file() else root/'completed.json'


def completed_actor(root, source_plan, initial, candidate, updates, *, ppo_schedule=None, mixed_bo=False,
                    ppo_algorithm=None):
    return completed_actor_from_hashes(root, sha(source_plan), sha(initial), candidate, updates,
                                       ppo_schedule=ppo_schedule, mixed_bo=mixed_bo, ppo_algorithm=ppo_algorithm)


def completed_actor_from_hashes(root, source_plan_sha256, initial_actor_sha256, candidate, updates, *,
                                ppo_schedule=None, mixed_bo=False, ppo_algorithm=None):
    manifest_path = root/'manifest.json'
    receipt = actor_receipt(root)
    complete = read(receipt)
    manifest = read(manifest_path)
    actor = Path(complete['student'])
    manifest_hash = sha(manifest_path)
    training_only = receipt.name == 'training_completed.json'
    expected_result = 'CANDIDATE_FINETUNED' if training_only else 'CANDIDATE_FINETUNED_EVALUATED'
    if training_only:
        root = root.resolve()
        canonical = root/'shared_actor_finetuned.pt'
        if not actor.is_absolute() or actor.is_symlink() or canonical.is_symlink() or actor.resolve() != canonical:
            raise ValueError('Fine-tuned actor is outside its canonical tuning output')
        actor = canonical
    if (complete.get('result') != expected_result
            or complete['manifest_sha256'] != manifest_hash
            or manifest['plan_sha256'] != source_plan_sha256
            or manifest['initial_actor_sha256'] != initial_actor_sha256
            or complete['candidate'] != candidate or manifest['candidate'] != candidate
            or complete['ppo_updates'] != updates or manifest['updates'] != updates
            or sha(actor) != complete['student_sha256']):
        raise ValueError('Fine-tuned candidate/actor identity or PPO budget does not match')
    if ppo_schedule is not None and manifest.get('ppo_schedule') != ppo_schedule:
        raise ValueError('Fine-tuned candidate uses a different PPO optimizer schedule')
    settings = NATIVE_PPO_PROFILES.get(ppo_algorithm)
    if (manifest.get('ppo_algorithm') != ppo_algorithm or
            (ppo_algorithm is not None and (settings is None or
                manifest.get('native_ppo_settings') != settings))):
        raise ValueError('Fine-tuned candidate uses a different PPO algorithm')
    if mixed_bo and (manifest.get('init_at_random_ep_len') is not True
                     or manifest.get('envs_per_task') != MIXED_BO_LAYOUT):
        raise ValueError('Formal BO requires mixed 4096-environment training with random initial episode counters')
    if training_only:
        folder = root/'checkpoints'
        receipt = folder/f'update_{updates:06d}.json'
        if (folder.resolve() != folder or receipt.resolve() != receipt or not receipt.is_file()
                or sha(receipt) != complete['checkpoint_sha256']):
            raise ValueError('Fine-tuned PPO checkpoint receipt changed')
        checkpoint = read(receipt)
        blob = folder/checkpoint['path']
        if (checkpoint['completed_updates'] != updates or checkpoint['manifest_sha256'] != manifest_hash
                or blob.parent != folder or blob.resolve() != blob or not blob.is_file()
                or blob.suffix != '.pt' or sha(blob) != checkpoint['sha256']):
            raise ValueError('Fine-tuned PPO checkpoint changed')
        if (checkpoint.get('ppo_algorithm') != ppo_algorithm or
                (ppo_algorithm is not None and checkpoint.get('warmup_completed') !=
                 settings['critic_warmup_updates'])):
            raise ValueError('Fine-tuned checkpoint uses a different PPO algorithm or Critic warmup')
        if ppo_algorithm == NATIVE_PPO_ALGORITHM:
            hashes = manifest.get('teacher_checkpoint_sha256', {})
            if (checkpoint.get('state_kind') != NATIVE_PPO_STATE_KIND
                    or checkpoint.get('teacher_checkpoint_sha256') != hashes
                    or set(hashes) != set(TASKS)
                    or any(not isinstance(h, str) or not re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes.values())):
                raise ValueError('Fine-tuned V4 checkpoint has a different expert-prior identity')
    elif sha(complete['evaluation']) != complete['evaluation_sha256']:
        raise ValueError('Fine-tuned candidate evaluation changed')
    return actor


def run_worker(command, folder, environment, timeout, root):
    from pipeline import child
    child([str(v) for v in command], folder, environment, timeout, root)


def check_native_measurement(report, path, conditions, *, task, seed, design_id, actor_sha, bank_sha, plan_sha):
    dt = conditions.get('measurement_dt_s', DT)
    full_count = round(conditions['duration_s'] / dt)
    if (report['kind'] != TRIAL_VERSION or report['design_id'] != design_id
            or report['task'] != task or report['seed'] != seed
            or report['conditions_sha256'] != digest(conditions)
            or report['actor_sha256'] != actor_sha or report['bank_sha256'] != bank_sha
            or report['plan_sha256'] != plan_sha or sha(report['trace']) != report['trace_sha256']
            or sorted(row['replica'] for row in report['trials']) != list(range(conditions['copies']))
            or any(row.get('dt_s') != dt or not 0 <= row['steps'] <= full_count
                   or (row['status'] == 'success' and row['steps'] != full_count)
                   for row in report['trials'])):
        raise ValueError(f'Incomplete or mismatched BO telemetry: {path}')
    if any(row.get('reason') == 'outside_stair_course' for row in report['trials']):
        raise ValueError(f'The timed stair course boundary was reached; extend the measurement course: {path}')
    if any(row['status'] not in ('success', 'failure') for row in report['trials']):
        raise ValueError(f'Invalid trial preserved for diagnosis. Repeat the same command: {path}')
    if dt == PHYSICS_DT:
        for row in report['trials']:
            if row['status'] == 'success':
                trial_metric(row, 'cot', ABSOLUTE_COT_BASIS)


def evaluate_native(root, plan_path, plan, actor, conditions_path, conditions, gpu, timeout, *, pause_root=None):
    pause_root = Path(pause_root or root)
    if (pause_root/'USER_PAUSED').exists():
        raise KeyboardInterrupt('USER_PAUSED')
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    actor_sha = sha(actor)
    plan_sha = sha(plan_path)
    requests = [(seed, task) for seed in conditions['seeds'] for task in TASKS]
    reports = []
    for seed, task in requests:
        if (pause_root/'USER_PAUSED').exists():
            raise KeyboardInterrupt('USER_PAUSED')
        spec = plan['teachers'][task]
        design_id = spec['design_ids'][0]
        work = root/'measurements'/f'{seed}_{task}'
        work.mkdir(parents=True, exist_ok=True)
        commit = work/'accepted.json'
        if not commit.exists():
            attempt = work/f'run_{uuid4().hex[:8]}'
            child_output = attempt/'native'
            command = [sys.executable, '-u', SHARED/('rollout_amp.py' if task == 'run' else 'rollout.py'),
                       '--plan', plan_path, '--student', actor, '--design-ids', design_id,
                       '--output', child_output, '--bo-conditions', conditions_path,
                       '--mode', 'evaluate', '--profile', 'nominal', '--randomization-strength', '0',
                       '--steps', str(round(conditions['duration_s'] / DT)), '--seed', str(seed),
                       '--reset-yaw-range', '0', '--headless', '--device', 'cuda:0']
            if task != 'run':
                command += ['--task', task]
            devices = candidates(query_gpus(), os.environ, min_free_mib=4*1024,
                                 max_utilization=100, requested=f'cuda:{gpu}')
            print('BO_MEASURING', design_id, task, seed, attempt/'logs', flush=True)
            with gpu_lease(devices, REPO/'robolab/logs/rpo_rough_multimorph/gpu_locks') as device:
                command += [f"--kit_args=--/renderer/activeGpu={device['index']} --/physics/cudaDevice=0"]
                run_worker(command, attempt/'logs', child_environment(device, os.environ), timeout, pause_root)
            if not (child_output/'summary.json').is_file() or (child_output/'failure.json').exists():
                raise ValueError(f'Native BO measurement did not complete: {attempt}')
            report_path = child_output/'bo_trials.json'
            evidence = dict(report=str(report_path), sha256=sha(report_path))
        else:
            evidence = read(commit)
            report_path = Path(evidence['report'])
            if sha(report_path) != evidence['sha256']:
                raise ValueError(f'Committed trial changed: {report_path}')
        report = read(report_path)
        check_native_measurement(report, report_path, conditions, task=task, seed=seed,
                                 design_id=design_id, actor_sha=actor_sha,
                                 bank_sha=spec['bank_sha256'], plan_sha=plan_sha)
        if not commit.exists():
            write_new(commit, evidence)
        reports.append(dict(report, evidence=evidence))
    return reports


def evaluate(a, work, source, original, initial, design_id, conditions):
    pause_root = Path(getattr(a, 'search_root', None) or work)
    if (pause_root/'USER_PAUSED').exists() and pause_root != work:
        raise KeyboardInterrupt('USER_PAUSED')
    from finetune_state import select_candidate
    plan, candidate = select_candidate(original, design_id=design_id)
    if any(not spec['full_body_torque'] for spec in plan['teachers'].values()):
        raise ValueError('BO measurements require the fully configured motor curves')
    mixed = uses_current_schedule(a.updates, conditions)
    if not mixed or conditions.get('measurement_dt_s') != PHYSICS_DT:
        raise ValueError('New BO evaluations require the V4 100-update, fixed 200 Hz protocol')
    explicit = getattr(a, 'finetuned', None)
    tuning = (explicit if explicit else work/'tuning').resolve()
    # Historical reports remain readable but the retired learners do not
    # produce new BO measurements. Only the current expert-prior V4 runs.
    algorithm = getattr(a, 'ppo_algorithm', NATIVE_PPO_ALGORITHM if mixed else None)
    if explicit and not hasattr(a, 'ppo_algorithm'):
        algorithm = read(tuning/'manifest.json').get('ppo_algorithm')
    settings = NATIVE_PPO_PROFILES.get(algorithm)
    if algorithm != NATIVE_PPO_ALGORITHM or settings != NATIVE_PPO_SETTINGS:
        raise ValueError('New BO measurements require the V4 expert-prior PPO algorithm')
    identity = dict(kind='rpo_bo_candidate_v1', source_plan_sha256=sha(source), candidate=candidate,
        initial_actor_sha256=sha(initial), conditions_sha256=digest(conditions), updates=a.updates,
        sources={p.name: sha(p) for p in (Path(__file__), SHARED/'bo_capture.py', SHARED/'shared_bo_metrics.py', SHARED/'stair_course.py',
                                         SHARED/'rollout.py', SHARED/'rollout_amp.py')})
    if algorithm is not None:
        identity.update(ppo_algorithm=algorithm, native_ppo_settings=settings)
    work.mkdir(parents=True, exist_ok=True)
    freeze_candidate_manifest(work/'manifest.json', identity, fixed_reset=conditions.get('reset_mode') == 'fixed')
    resume_pause(work, a.resume_paused)
    if not actor_receipt(tuning).is_file():
        if getattr(a, 'finetuned', None):
            raise ValueError(f'Missing completed PPO run selected for reuse: {tuning}')
        entry = SHARED/'mixed_finetune.py'
        command = [sys.executable, '-u', entry, '--plan', source, '--student', initial,
                   '--output', tuning, '--design-id', design_id, '--updates', str(a.updates),
                   '--device', 'cuda:0', '--headless', '--num-envs', str(MIXED_BO_ENVS)]
        tuning_manifest = tuning/'manifest.json'
        if tuning_manifest.is_file():
            if read(tuning_manifest).get('ppo_algorithm') != algorithm:
                raise ValueError('Existing tuning belongs to a different PPO algorithm; use a new V4 output')
            command.append('--resume')
            if a.resume_paused:
                command.append('--resume-paused')
            compact = read(tuning_manifest).get('compact_training_artifacts') is True
        else:
            compact = True
        if compact:
            command.append('--compact-training-artifacts')
        logs = work/'tuning_logs'/uuid4().hex[:8]
        print('BO_FINETUNING', design_id, candidate['structure_key'], logs, flush=True)
        timeout = (a.updates + 18) * a.timeout
        from robolab.rough_gpu import candidates, child_environment, gpu_lease, query_gpus
        choices = candidates(query_gpus(), os.environ, min_free_mib=12800,
                             max_utilization=100, requested=f'cuda:{a.gpu}')
        with gpu_lease(choices, REPO/'robolab/logs/rpo_rough_multimorph/gpu_locks') as gpu:
            run_worker(command, logs, child_environment(gpu, os.environ), timeout, pause_root)
    actor = completed_actor(tuning, source, initial, candidate, a.updates,
                            ppo_schedule=PPO_SCHEDULE if mixed else None, mixed_bo=mixed,
                            ppo_algorithm=algorithm)
    plan_path = tuning/'plan.json'
    if load_plan(plan_path) != plan:
        raise ValueError('Fine-tuned physical plan differs from BO candidate')
    reports = evaluate_native(work, plan_path, plan, actor, a.conditions, conditions, a.gpu, a.timeout,
                              pause_root=pause_root)
    result = dict(kind=REPORT_VERSION, domain='simulation', candidate=candidate, ppo_updates=a.updates,
        student=str(actor), student_sha256=sha(actor), conditions=conditions, conditions_sha256=digest(conditions),
        evaluation_manifest_sha256=sha(work/'manifest.json'), pairs=paired_trials(reports, conditions),
        measurements=[r['evidence'] for r in reports])
    if algorithm is not None:
        result['ppo_algorithm'] = algorithm
    freeze(work/'bo_evaluation.json', result)
    print('BO_EVALUATION_READY', work/'bo_evaluation.json', flush=True)
    return result


def checked_report(path):
    report = read(path)
    if (report.get('kind') != REPORT_VERSION or report.get('domain') not in ('simulation', 'hardware')
            or digest(report['conditions']) != report['conditions_sha256']):
        raise ValueError('Invalid trial report or condition identity')
    return report


def ingest_measurements(spec_path, conditions_path):
    """Score an independent baseline or hardware campaign with the same math."""
    import numpy as np
    specification = read(spec_path)
    conditions = read_conditions(conditions_path)
    dt = conditions.get('measurement_dt_s', DT)
    if (specification['domain'] not in ('simulation', 'hardware')
            or specification['dt_s'] != dt or specification['measurement_convention'] != TRIAL_VERSION):
        raise ValueError(f'Imported arrays must follow the declared mechanical BO convention at {1/dt:g} Hz')
    grouped, evidence = {}, []
    base = spec_path.resolve().parent
    for trial in specification['trials']:
        task, seed, replica = (trial[k] for k in ('task', 'seed', 'replica'))
        speed = conditions['tasks'][task]['speed_mps']
        trace = (base/trial['trace']).resolve()
        motors, payload = trial['motors'], trial.get('payload')
        joints = {m['joint'] for m in motors}
        if (len(motors) != 23 or len(joints) != 23 or not KNEE_ANKLE <= joints
                or any(m.get('torque_side') != 'joint_output' or not m.get('motor_id') for m in motors)
                or type(trial['fell']) is not bool or not positive(trial['total_mass_kg'])):
            raise ValueError('Imported trial requires exact motor names, joint-output feedback and total mass')
        for motor in motors:
            if positive(motor.get('rated_torque_nm')) and not motor.get('source'):
                raise ValueError('A rated torque measurement denominator needs a source')
        if task == 'backpack' and (payload is None or
                not positive(payload.get('mass_kg')) or payload['mass_kg'] != conditions['tasks'][task]['payload_kg']
                or len(payload.get('mount_position_torso_m', [])) != 3):
            raise ValueError('Imported backpack needs the fixed actual mass and mount position')
        evidence.append(dict(trace=str(trace), trace_sha256=sha(trace), task=task, seed=seed, replica=replica))
        with np.load(trace, allow_pickle=False) as samples:
            positions = samples['position']
            forward = float(positions[-1, 0] - samples['initial_position'][0]) if len(positions) else 0.
            if positions.shape != (len(samples['vx']), 3):
                raise ValueError('Imported positions must align with the velocity series')
            outside = False
            stair = conditions['tasks']['stairs']
            if task == 'stairs' and stair.get('course') == COURSE:
                flight = evaluation_flight(conditions['duration_s'], speed, stair['step_depth_m'],
                                           stair['travel_margin_factor'])
                xy = positions[:, :2] - samples['initial_position'][:2]
                outside = bool(outside_flight(xy[:, 0], xy[:, 1], flight['origin_x_m'], flight['length_m']).any())
            score = score_trial(task, speed=speed, dt=dt, expected_steps=round(conditions['duration_s'] / dt),
                mass=trial['total_mass_kg'], forward=forward, vx=samples['vx'], omega=samples['omega'],
                torque=samples['torque'], joint_vel=samples['joint_vel'], motors=motors, fell=trial['fell'],
                reached_top=trial.get('reached_top'), route_completed=trial.get('route_completed'),
                protection_stop=trial.get('protection_stop', False), payload=payload, outside_stairs=outside,
                min_forward_progress_ratio=conditions.get('min_forward_progress_ratio'))
        grouped.setdefault((task, seed), []).append(dict(replica=replica, **score))
    return dict(kind=REPORT_VERSION, domain=specification['domain'], candidate=specification['candidate'],
        conditions=conditions, conditions_sha256=digest(conditions), source_sha256=sha(spec_path),
        pairs=paired_trials([dict(task=t, seed=s, trials=rows) for (t, s), rows in grouped.items()], conditions),
        measurements=evidence)


def calibrate(report, reference):
    pairs = [p for p in report['pairs'] if p['feasible']
             and all(p['tasks'][task]['metrics'].get(name) is not None
                     for task in TASKS for name in METRICS[task])]
    if len(pairs) < 2:
        raise ValueError('Normalization needs two or more complete joint-success repetitions')
    metrics = {}
    for task in TASKS:
        metrics[task] = {}
        for name in METRICS[task]:
            values = [pair['tasks'][task]['metrics'][name] for pair in pairs]
            metrics[task][name] = dict(baseline=fmean(values), scale=None, weight=1/3,
                                       baseline_std=stdev(values))
    if report.get('ppo_algorithm') is not None:
        reference = dict(reference, ppo_algorithm=report['ppo_algorithm'])
    return dict(kind=SCORING_VERSION, domain=report['domain'], conditions_sha256=report['conditions_sha256'],
                benchmark=reference, weighting_status='pilot_equal_weights', baseline_successful_pairs=len(pairs),
                task_weights={task: 1/3 for task in TASKS}, metrics=metrics)


def required_panel_pairs(conditions):
    seeds = conditions['seeds']
    if conditions['copies'] != 1 or len(seeds) not in (1, 3) or len(set(seeds)) != len(seeds):
        raise ValueError('Calibration needs one or three distinct seeds with one trial each')
    return {(seed, 0) for seed in seeds}


def check_panel_pairs(pairs, expected):
    if (len(pairs) != len(expected)
            or {(pair['seed'], pair['replica']) for pair in pairs} != expected
            or any(type(pair['seed']) is not int or type(pair['replica']) is not int for pair in pairs)):
        raise ValueError('Incomplete or duplicate calibration seed coverage')
    for pair in pairs:
        tasks = pair['tasks']
        if set(tasks) != set(TASKS):
            raise ValueError('Missing task in a paired BO repetition')
        statuses = {trial['status'] for trial in tasks.values()}
        if not statuses <= {'success', 'failure'}:
            raise ValueError('Invalid trial must be resolved before calibration or search')
        if pair['feasible'] is not (statuses == {'success'}):
            raise ValueError('Joint BO feasibility disagrees with task trial statuses')


def panel_scoring(reports, keys, benchmark, *, conditions=None, cot_basis=None, task_weights=None):
    """Fit each scale to structures completing every declared trial for the task."""
    import numpy as np
    if len(keys) != 50 or len(set(keys)) != 50 or [tuple(r['candidate']['structure_key']) for r in reports] != list(keys):
        raise ValueError('Calibration requires the fixed 50 distinct structures in group-21 order')
    if conditions is None:
        conditions = reports[0]['conditions']
    physics_measurement = conditions.get('measurement_dt_s') == PHYSICS_DT
    if cot_basis is None:
        cot_basis = ABSOLUTE_COT_BASIS if physics_measurement else POSITIVE_COT_BASIS
    if scoring_cot_basis(dict(cot_basis=cot_basis)) == ABSOLUTE_COT_BASIS and not physics_measurement:
        raise ValueError('Absolute mechanical CoT scoring requires 200 Hz measurements')
    absolute = cot_basis == ABSOLUTE_COT_BASIS
    if task_weights is None:
        task_weights = FORMAL_TASK_WEIGHTS if absolute else {task: 1/3 for task in TASKS}
    conditions_sha256 = digest(conditions)
    expected = required_panel_pairs(conditions)
    samples = {task: {metric: [] for metric in METRICS[task]} for task in TASKS}
    for report in reports:
        pairs = report['pairs']
        if (report['kind'] != REPORT_VERSION or report['domain'] != 'simulation'
                or report['ppo_updates'] != benchmark['updates'] or report['conditions'] != conditions
                or report['conditions_sha256'] != conditions_sha256
                or report.get('ppo_algorithm') != benchmark.get('ppo_algorithm')):
            raise ValueError('Inconsistent calibration conditions or candidate budget')
        check_panel_pairs(pairs, expected)
        for task in TASKS:
            runs = [pair['tasks'][task] for pair in pairs]
            if not all(run['status'] == 'success' for run in runs):
                continue
            for metric in METRICS[task]:
                values = [trial_metric(run, metric, cot_basis) for run in runs]
                if any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in values):
                    raise ValueError(f'Missing or invalid successful trial measurement: {task}/{metric}')
                samples[task][metric].append(fmean(values))
    metrics = {}
    for task, columns in samples.items():
        metrics[task] = {}
        weights = (1/3, 1/3, 1/3) if task == 'stairs' else (.4, .4, .2)
        for (name, values), weight in zip(columns.items(), weights, strict=True):
            if len(values) < 2:
                raise ValueError(f'Too few eligible structures for {task}/{name}: {len(values)}')
            low, high = map(float, np.percentile(values, [10, 90], method='linear'))
            if not high > low:
                raise ValueError(f'Zero P90-P10 range for {task}/{name}')
            # With an explicit scale, the scorer maps the baseline to 1.
            metrics[task][name] = dict(p10=low, p90=high, baseline=high, scale=high-low,
                                      weight=weight, eligible_designs=len(values))
    scoring = dict(kind=SCORING_VERSION,domain='simulation',conditions_sha256=conditions_sha256,
        method='post_finetune_panel50',benchmark=benchmark,metrics=metrics,task_weights=dict(task_weights))
    if absolute:
        scoring['cot_basis'] = cot_basis
    validate_scoring(scoring, dict(domain='simulation', conditions_sha256=conditions_sha256, conditions=conditions))
    return scoring


def span_only_panel(scoring):
    return scoring.get('method') == 'post_finetune_panel50' and all(
        row.get('scale') is None for metrics in scoring['metrics'].values() for row in metrics.values()
    )


def check_panel_receipt(scoring_path):
    receipt = read(scoring_path.parent/'completed.json')
    if (receipt['scoring_sha256'] != sha(scoring_path)
            or receipt['manifest_sha256'] != sha(scoring_path.parent/'manifest.json')):
        raise ValueError('Frozen calibration scoring or manifest changed')
    if scoring_cot_basis(read(scoring_path)) == ABSOLUTE_COT_BASIS:
        evidence = scoring_path.parent/'measurements_completed.json'
        if not evidence.is_file() or receipt.get('measurements_completed_sha256') != sha(evidence):
            raise ValueError('Frozen calibration measurement receipt changed')


def initial_report(path, evidence, benchmark):
    if sha(path) != evidence['sha256']:
        raise ValueError(f'Frozen initial BO report changed: {path}')
    report = checked_report(path)
    manifest_path = path.parent/'manifest.json'
    if report.get('evaluation_manifest_sha256') != sha(manifest_path):
        raise ValueError('Initial BO observation candidate manifest changed')
    identity = read(manifest_path)
    if (identity.get('source_plan_sha256') != benchmark['source_plan_sha256']
            or identity.get('initial_actor_sha256') != benchmark['initial_actor_sha256']
            or identity.get('updates') != benchmark['updates'] or identity.get('candidate') != report['candidate']
            or identity.get('conditions_sha256') != report['conditions_sha256']):
        raise ValueError('Initial BO observation has different tuning inputs')
    if (identity.get('ppo_algorithm') != benchmark.get('ppo_algorithm')
            or report.get('ppo_algorithm') != benchmark.get('ppo_algorithm')
            or (benchmark.get('ppo_algorithm') is not None and
                (benchmark['ppo_algorithm'] not in NATIVE_PPO_PROFILES or
                 identity.get('native_ppo_settings') != NATIVE_PPO_PROFILES[benchmark['ppo_algorithm']]))):
        raise ValueError('Initial BO observation has a different PPO algorithm')
    return report


def calibration_evaluation_timing(work, evidence):
    work = work.resolve()
    measured = {}
    for receipt in evidence:
        report = Path(receipt['report']).resolve()
        if not report.is_relative_to(work) or report in measured:
            raise ValueError(f'Duplicate or outside calibration measurement: {report}')
        if sha(report) != receipt['sha256']:
            raise ValueError(f'Calibration measurement changed: {report}')
        elapsed = read(report.parent.parent/'logs/exit.json')
        if elapsed['returncode'] != 0 or not positive(elapsed['seconds']):
            raise ValueError(f'Missing worker duration for accepted measurement: {report}')
        measured[report] = elapsed['seconds']
    return dict(accepted_measurements=len(measured), evaluation_seconds=sum(measured.values()))


def checked_panel_candidate(work, index, evidence, benchmark, conditions, tuning_panel):
    folder = work/f'design_{index:02d}'
    path = folder/'bo_evaluation.json'
    if (Path(evidence['report']) != path.relative_to(work) or path.resolve() != path):
        raise ValueError(f'Unexpected calibration report path at structure {index}')
    report = initial_report(path, evidence, benchmark)
    return checked_candidate_measurements(folder, report, benchmark, conditions,
                                          tuning_panel/f'design_{index:02d}'/'tuning')


def checked_candidate_measurements(folder, report, benchmark, conditions, tuning):
    if report['conditions'] != conditions or report['ppo_updates'] != benchmark['updates']:
        raise ValueError(f'Conditions or PPO budget changed: {folder}')
    candidate = report['candidate']
    mixed = uses_current_schedule(benchmark['updates'], conditions)
    actor = completed_actor_from_hashes(tuning, benchmark['source_plan_sha256'], benchmark['initial_actor_sha256'],
                                        candidate, benchmark['updates'],
                                        ppo_schedule=PPO_SCHEDULE if mixed else None, mixed_bo=mixed,
                                        ppo_algorithm=benchmark.get('ppo_algorithm')).resolve()
    actor_sha = sha(actor)
    if actor != Path(report['student']).resolve() or actor_sha != report['student_sha256']:
        raise ValueError(f'Measurement actor changed: {folder}')
    plan_path = tuning/'plan.json'
    plan = read(plan_path)
    plan_sha = sha(plan_path)
    measurements = {}
    expected = {(seed, task) for seed in conditions['seeds'] for task in TASKS}
    for entry in report['measurements']:
        native_path = Path(entry['report'])
        if (not native_path.is_absolute() or native_path.resolve() != native_path
                or not native_path.is_relative_to(folder/'measurements') or sha(native_path) != entry['sha256']):
            raise ValueError(f'Measurement receipt changed: {folder}')
        native = read(native_path)
        seed, task = native['seed'], native['task']
        trial_root = folder/'measurements'/f'{seed}_{task}'
        if ((seed, task) not in expected or (seed, task) in measurements
                or native_path.parents[2] != trial_root or native_path.name != 'bo_trials.json'
                or read(trial_root/'accepted.json') != entry
                or Path(native['trace']) != native_path.with_name('bo_samples.npz')):
            raise ValueError(f'Invalid accepted measurement: {folder}')
        spec = plan['teachers'][task]
        design_id = candidate['design_ids'][task]
        if spec['design_ids'] != [design_id]:
            raise ValueError(f'Frozen tuning design changed: {folder}/{task}')
        check_native_measurement(native, native_path, conditions, task=task, seed=seed, design_id=design_id,
                                 actor_sha=actor_sha, bank_sha=spec['bank_sha256'], plan_sha=plan_sha)
        measurements[seed, task] = native
    if measurements.keys() != expected or paired_trials(measurements.values(), conditions) != report['pairs']:
        raise ValueError(f'Incomplete or inconsistent measured trials: {folder}')
    return report


def checked_measurement_panel(work):
    work = work.resolve()
    manifest_path = work/'manifest.json'
    evidence_path = work/'measurements_completed.json'
    if not evidence_path.is_file():
        raise ValueError(f'Completed 200 Hz measurement receipt is required: {evidence_path}')
    manifest, evidence = read(manifest_path), read(evidence_path)
    conditions_hash = manifest['conditions_sha256']
    conditions_path = official_conditions_path(conditions_hash)
    if conditions_path != PHYSICS_CONDITIONS:
        raise ValueError('Scoring can only be frozen from completed official 200 Hz measurements')
    conditions = read_conditions(conditions_path)
    keys = calibration_keys()
    if (evidence['manifest_sha256'] != sha(manifest_path) or evidence['conditions_sha256'] != conditions_hash
            or manifest['structure_keys'] != list(map(list, keys)) or len(evidence['reports']) != len(keys)):
        raise ValueError('Completed measurement receipt does not match the frozen 50-structure manifest')
    benchmark = {field: manifest[field] for field in ('initial_actor_sha256', 'source_plan_sha256', 'updates')}
    mixed = uses_current_schedule(benchmark['updates'], conditions)
    if benchmark['updates'] == PREVIOUS_BO_UPDATES or mixed:
        if manifest.get('selection_velocity_frame') != EVALUATION_VELOCITY_FRAME:
            raise ValueError('Measurements require horizontal heading checkpoint selection')
        benchmark['selection_velocity_frame'] = EVALUATION_VELOCITY_FRAME
    if mixed:
        settings = dict(ppo_schedule=PPO_SCHEDULE, init_at_random_ep_len=True, envs_per_task=MIXED_BO_LAYOUT)
        if any(manifest.get(key) != value for key, value in settings.items()):
            raise ValueError('Measurements use a different PPO schedule, episode timing or environment layout')
        benchmark.update(settings)
    if manifest.get('ppo_algorithm') is not None:
        algorithm = manifest['ppo_algorithm']
        settings = NATIVE_PPO_PROFILES.get(algorithm)
        if (not mixed or settings is None or manifest.get('native_ppo_settings') != settings):
            raise ValueError('Measurements use a different native PPO algorithm')
        benchmark.update(ppo_algorithm=algorithm, native_ppo_settings=settings)
    tuning_panel = work
    if reused := manifest.get('reused_finetuned_panel'):
        tuning_panel = Path(reused['path']).resolve()
        if (tuning_panel == work or sha(tuning_panel/'manifest.json') != reused['manifest_sha256']):
            raise ValueError('Reused calibration tuning panel changed')
    reports = [checked_panel_candidate(work, index, entry, benchmark, conditions, tuning_panel)
               for index, entry in enumerate(evidence['reports'], 1)]
    return reports, keys, dict(benchmark, reports=evidence['reports']), conditions


def checked_original_rpo_reference(root, benchmark, conditions):
    archive_path = ORIGINAL_RPO_ARCHIVE/'manifest.json'
    archive = read(archive_path)
    from original_reference import ARCHIVE_KIND, check_v4_archive
    if archive.get('kind') == ARCHIVE_KIND or benchmark.get('ppo_algorithm') == NATIVE_PPO_ALGORITHM:
        # Formal V4 scoring has exactly one baseline: the checked-in measured
        # update-100 milestone. Do not launch or substitute a new RPO run.
        if (archive.get('kind') != ARCHIVE_KIND or benchmark.get('ppo_algorithm') != NATIVE_PPO_ALGORITHM
                or Path(root).resolve() != ORIGINAL_RPO_ARCHIVE.resolve()):
            raise ValueError('Formal V4 BO requires the canonical archived expert-prior update-100 original RPO reference')
        return check_v4_archive(ORIGINAL_RPO_ARCHIVE, benchmark, conditions)
    protocol_path = root/'reference_protocol.json'
    protocol = read(protocol_path)
    archived_protocol = ORIGINAL_RPO_ARCHIVE/'reference_protocol.json'
    expected = {key: benchmark[key] for key in ('source_plan_sha256', 'initial_actor_sha256', 'updates')}
    expected['conditions_sha256'] = digest(conditions)
    if benchmark.get('ppo_algorithm') is not None:
        # The new policy must be measured on the exact original structure; the
        # archived actor and raw trace hashes identify a *different* algorithm.
        algorithm = benchmark['ppo_algorithm']
        settings = NATIVE_PPO_PROFILES.get(algorithm)
        if settings is None or benchmark.get('native_ppo_settings') != settings:
            raise ValueError('Unknown native original-RPO PPO settings')
        expected.update(ppo_algorithm=algorithm, native_ppo_settings=settings)
        if protocol != expected:
            raise ValueError('Original RPO reference and initial samples must share the native PPO protocol')
        folder = root/'candidate'
        path = folder/'bo_evaluation.json'
        result_path = root/'reference_result.json'
        result = read(result_path)
        singleton = dict(benchmark, source_plan_sha256=sha(root/'reference_plan.json'))
        report = initial_report(path, dict(sha256=result['report_sha256']), singleton)
        if (report['candidate']['structure_key'] != archive['design_key']
                or result['candidate'] != report['candidate']
                or result['student_sha256'] != report['student_sha256']
                or result.get('ppo_algorithm') != algorithm):
            raise ValueError('Native original RPO physical structure or actor differs from its measured reference')
        checked_candidate_measurements(folder, report, singleton, conditions, folder/'tuning')
        return report, dict(root=str(root), reference_result_sha256=sha(result_path),
                            report_sha256=sha(path), protocol_sha256=sha(protocol_path))
    if (protocol != expected or protocol != read(archived_protocol)
            or sha(archived_protocol) != archive['files_sha256']['reference_protocol.json']):
        raise ValueError('Original RPO reference and initial samples must share the archived PPO and measurement protocol')
    folder = root/'candidate'
    path = folder/'bo_evaluation.json'
    singleton = dict(benchmark, source_plan_sha256=sha(root/'reference_plan.json'))
    report = initial_report(path, dict(sha256=archive['raw_source_report_sha256']), singleton)
    if (report['candidate']['structure_key'] != archive['design_key']
            or report['student_sha256'] != archive['final_actor_sha256']):
        raise ValueError('Original RPO structure or final actor differs from the archived reference')
    checked_candidate_measurements(folder, report, singleton, conditions, folder/'tuning')
    return report, dict(root=str(root), archive_manifest_sha256=sha(archive_path),
                        report_sha256=sha(path), protocol_sha256=sha(protocol_path))


def original_rpo_scoring(reference, source, benchmark, panel, conditions, *, metric_profile='four', metric_weights=None):
    expected = required_panel_pairs(conditions)
    check_panel_pairs(reference['pairs'], expected)
    (pair,) = reference['pairs']
    if (conditions.get('measurement_dt_s') != PHYSICS_DT or len(expected) != 1
            or pair['feasible'] is not True or any(row['status'] != 'success' for row in pair['tasks'].values())):
        raise ValueError('Original RPO ratios require three successful tasks with one fixed 200 Hz evaluation')
    if metric_profile not in ('three', 'four'):
        raise ValueError('Unknown Original RPO metric profile')
    four = metric_profile == 'four'
    if four:
        metric_weights = validate_four_metric_weights(
            read(FOUR_METRIC_WEIGHTS_PATH) if metric_weights is None else metric_weights)
    elif metric_weights is not None:
        raise ValueError('Custom metric weights require the four-metric profile')
    metrics = {}
    for task, names in (FOUR_METRICS if four else METRICS).items():
        weights = (tuple(metric_weights[task][name] for name in names) if four
                   else ((1/3, 1/3, 1/3) if task == 'stairs' else (.4, .4, .2)))
        metrics[task] = {
            name: dict(baseline=trial_metric(pair['tasks'][task], name, ABSOLUTE_COT_BASIS,
                                             metric_profile=metric_profile), weight=weight)
            for name, weight in zip(names, weights, strict=True)
        }
    scoring = dict(kind=FOUR_METRIC_SCORING_VERSION if four else SCORING_VERSION,
                   method=ORIGINAL_RPO_FOUR_METRIC_METHOD if four else ORIGINAL_RPO_SCORING_METHOD,
                   domain='simulation',
                   conditions_sha256=digest(conditions), cot_basis=ABSOLUTE_COT_BASIS,
                   task_weights=FORMAL_TASK_WEIGHTS, metrics=metrics, benchmark=benchmark,
                   measurement_panel=panel, original_rpo_reference=source)
    score = objective(reference, scoring)
    costs = score['observations'][0]['task_objectives']
    baseline_valid = (score['mean_objective'] == 1.0 and costs == dict.fromkeys(TASKS, 1.0))
    if four:
        baseline_valid = (isclose(score['mean_objective'], 1., rel_tol=0., abs_tol=1e-12)
                          and all(isclose(cost, 1., rel_tol=0., abs_tol=1e-12) for cost in costs.values()))
    if not baseline_valid:
        raise ValueError('Original RPO reference scores must all equal one')
    return scoring


def original_rpo_initial_scores(reports, evidence, scoring, conditions):
    expected = required_panel_pairs(conditions)
    candidates = []
    for report, entry in zip(reports, evidence, strict=True):
        check_panel_pairs(report['pairs'], expected)
        scored = objective(report, scoring)
        if scored['valid_pairs'] != len(expected) or scored['complete_objectives'] != scored['joint_successes']:
            raise ValueError('Initial BO sample has incomplete performance measurements')
        candidate = report['candidate']
        candidates.append(dict(design_id=candidate['design_ids']['stairs'], structure_key=candidate['structure_key'],
                               report=entry['report'], **scored))
    return candidates


def measurement_panel_source(root):
    return dict(root=str(root), manifest_sha256=sha(root/'manifest.json'),
                measurements_completed_sha256=sha(root/'measurements_completed.json'))


def freeze_original_rpo_scoring(measurements_root, reference_root, output, *, metric_profile='four', metric_weights=None):
    panel, reference_root, output = (Path(path).resolve() for path in (measurements_root, reference_root, output))
    if any(output.is_relative_to(source) or source.is_relative_to(output) for source in (panel, reference_root)):
        raise ValueError('Original RPO scoring needs a separate output directory')
    panel_source = measurement_panel_source(panel)
    reports, _, benchmark, conditions = checked_measurement_panel(panel)
    reference, source = checked_original_rpo_reference(reference_root, benchmark, conditions)
    scoring = original_rpo_scoring(reference, source, benchmark, panel_source, conditions,
                                   metric_profile=metric_profile, metric_weights=metric_weights)
    candidates = original_rpo_initial_scores(reports, benchmark['reports'], scoring, conditions)
    if measurement_panel_source(panel) != panel_source:
        raise ValueError('Measurement panel changed while freezing Original RPO scoring')
    output.mkdir(parents=True, exist_ok=True)
    path = output/'scoring.json'
    freeze(path, scoring)
    observations = output/'initial_observations.json'
    freeze(observations, dict(scoring_sha256=sha(path), reference_objective=1.0, candidates=candidates))
    freeze(output/'completed.json', dict(scoring_sha256=sha(path), initial_observations_sha256=sha(observations)))
    print('BO_ORIGINAL_RPO_SCORING_READY', path, len(candidates), flush=True)


def read_search_benchmark(scoring_path, scoring, conditions):
    if scoring['method'] not in ORIGINAL_RPO_SCORING_METHODS:
        check_panel_receipt(scoring_path)
        root = scoring_path.parent
        if read(root/'manifest.json').get('conditions_sha256') != digest(conditions):
            raise ValueError('Frozen calibration manifest uses different measurement conditions')
        if span_only_panel(scoring):
            raise ValueError('Rerun BO initialize to reuse completed measurements with P10/P90 anchors')
        return root, scoring['benchmark']
    folder = scoring_path.parent
    observations_path = folder/'initial_observations.json'
    receipt = read(folder/'completed.json')
    if (receipt['scoring_sha256'] != sha(scoring_path)
            or receipt['initial_observations_sha256'] != sha(observations_path)):
        raise ValueError('Frozen Original RPO scoring or initial observations changed')
    panel = Path(scoring['measurement_panel']['root']).resolve()
    panel_source = measurement_panel_source(panel)
    if panel_source != scoring['measurement_panel']:
        raise ValueError('Frozen Original RPO measurement panel changed')
    reports, _, benchmark, measured_conditions = checked_measurement_panel(panel)
    if measured_conditions != conditions:
        raise ValueError('Frozen Original RPO measurement conditions changed')
    reference_source = scoring['original_rpo_reference']
    root = (REPO/reference_source['root'] if reference_source.get('source_type') == 'repository_v4_milestone'
            else Path(reference_source['root']))
    reference, source = checked_original_rpo_reference(root, benchmark, conditions)
    metric_profile = 'four' if scoring['method'] == ORIGINAL_RPO_FOUR_METRIC_METHOD else 'three'
    # Existing scores retain the selected weights even if the provisional
    # default configuration was subsequently edited for a new experiment.
    weights = ({task: {name: row['weight'] for name, row in items.items()}
                for task, items in scoring['metrics'].items()} if metric_profile == 'four' else None)
    if original_rpo_scoring(reference, source, benchmark, panel_source, conditions,
                            metric_profile=metric_profile, metric_weights=weights) != scoring:
        raise ValueError('Frozen Original RPO scoring differs from the reference and measured panel')
    records = original_rpo_initial_scores(reports, benchmark['reports'], scoring, conditions)
    if read(observations_path) != dict(scoring_sha256=sha(scoring_path), reference_objective=1.0, candidates=records):
        raise ValueError('Frozen Original RPO initial observations differ from measured reports')
    return panel, benchmark


def initialize(a, work, source, plan, actor, conditions):
    """Create the fixed V4 measurement panel, never a legacy percentile score."""
    if a.updates != DEFAULT_BO_UPDATES or official_conditions_path(digest(conditions)) != PHYSICS_CONDITIONS:
        raise ValueError('New BO measurements require the V4 100-update, fixed 200 Hz protocol')
    keys=calibration_keys()
    lookup={tuple(row['key']):row for row in read(plan['shared_bank'])['structures']}
    if not set(keys) <= set(map(tuple,plan['selected_structure_keys'])):
        raise ValueError('Missing fixed calibration structures in shared plan')
    a.ppo_algorithm=NATIVE_PPO_ALGORITHM
    benchmark=dict(initial_actor_sha256=sha(actor),source_plan_sha256=sha(source),updates=a.updates,
        selection_velocity_frame=EVALUATION_VELOCITY_FRAME,ppo_schedule=PPO_SCHEDULE,
        init_at_random_ep_len=True,envs_per_task=MIXED_BO_LAYOUT,
        ppo_algorithm=NATIVE_PPO_ALGORITHM,native_ppo_settings=NATIVE_PPO_SETTINGS)
    manifest=dict(**benchmark,conditions_sha256=digest(conditions),structure_keys=list(map(list,keys)))
    reuse=getattr(a,'reuse_finetuned_panel',None)
    if reuse is not None:
        reuse=reuse.resolve()
        if reuse == work.resolve():
            raise ValueError('Reevaluation requires a separate completed calibration panel')
        # A V4 measurement panel deliberately has no local percentile scoring
        # file. Authenticate its actual measured observations and tuning receipts.
        checked_measurement_panel(reuse)
        previous=read(reuse/'manifest.json')
        if any(previous.get(key) != value for key,value in benchmark.items()) or previous.get(
                'structure_keys') != manifest['structure_keys']:
            raise ValueError('Reusable panel differs in structures, shared actor or PPO setting')
        if previous.get('reused_finetuned_panel'):
            reuse=Path(previous['reused_finetuned_panel']['path']).resolve()
        manifest['reused_finetuned_panel']=dict(path=str(reuse),manifest_sha256=sha(reuse/'manifest.json'))
    freeze(work/'manifest.json',manifest)
    if (work/'measurements_completed.json').is_file():
        checked_measurement_panel(work)
        resume_pause(work,a.resume_paused)
        print('BO_CALIBRATION_MEASUREMENTS_READY',work/'measurements_completed.json',flush=True)
        return
    resume_pause(work,a.resume_paused)
    reports,evidence=[],[]
    elapsed=0.
    for index,key in enumerate(keys,1):
        folder=work/f'design_{index:02d}'
        candidate_args=a
        if reuse is not None:
            candidate_args=copy(a)
            candidate_args.finetuned=reuse/f'design_{index:02d}'/'tuning'
        report=evaluate(candidate_args,folder,source,plan,actor,lookup[key]['design_ids']['stairs'],conditions)
        path=folder/'bo_evaluation.json'
        reports.append(report)
        evidence.append(dict(report=path.relative_to(work).as_posix(),sha256=sha(path)))
        elapsed+=calibration_evaluation_timing(work,report['measurements'])['evaluation_seconds']
        print('CALIBRATION_PROGRESS',index,'/ 50',f'evaluation_seconds={elapsed:.2f}',flush=True)
    timing=calibration_evaluation_timing(work,[row for report in reports for row in report['measurements']])
    if timing['accepted_measurements'] != len(keys)*len(TASKS)*len(conditions['seeds']):
        raise ValueError('Incomplete accepted measurement coverage for the fixed panel')
    freeze(work/'evaluation_timing.json',timing)
    freeze(work/'measurements_completed.json',dict(conditions_sha256=digest(conditions),
        manifest_sha256=sha(work/'manifest.json'),reports=evidence))
    checked_measurement_panel(work)
    print('BO_CALIBRATION_MEASUREMENTS_READY',work/'measurements_completed.json',flush=True)


def search(a, work, source, plan, actor, conditions):
    """Canonical continuous BO using the current four-metric ratios."""
    scoring = read(a.scoring)
    if (scoring.get('method') != ORIGINAL_RPO_FOUR_METRIC_METHOD or
            scoring.get('benchmark', {}).get('ppo_algorithm') != NATIVE_PPO_ALGORITHM or
            scoring['benchmark'].get('native_ppo_settings') != NATIVE_PPO_SETTINGS):
        raise ValueError('Native PPO BO requires a newly measured native panel and original-RPO reference; old data remain historical')
    a.ppo_algorithm = NATIVE_PPO_ALGORITHM
    from bo_continuous import search as continuous_search
    from bo_continuous_assets import prepare_candidate
    return continuous_search(sys.modules[__name__], a, work, source, plan, actor,
                             conditions, prepare=prepare_candidate)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('evaluate', 'initialize', 'search'):
        p = sub.add_parser(command)
        roots = p.add_mutually_exclusive_group()
        roots.add_argument('--root', type=Path)
        roots.add_argument('--migrated', action='store_true', help='Use the completed migrated common bank')
        p.add_argument('--output', type=Path)
        p.add_argument('--conditions', type=Path)
        p.add_argument('--shared-continuation', type=Path,
                       help='Use the verified winning actor through this completed DAgger continuation')
        p.add_argument('--gpu', type=int, default=0)
        p.add_argument('--updates', type=int, default=DEFAULT_BO_UPDATES)
        p.add_argument('--timeout', type=int, default=1800)
        p.add_argument('--resume-paused', action='store_true')
        if command == 'evaluate':
            p.add_argument('--candidate', required=True, help='Exact Rough (stairs) ID in the shared bank')
            p.add_argument('--finetuned', type=Path, help='Reuse a completed candidate run with the same plan and budget')
        elif command == 'initialize':
            p.add_argument('--reuse-finetuned-panel', type=Path,
                           help='Use all 50 completed PPO runs from a separate calibration panel')
            p.add_argument('--measure-only', action='store_true', default=True,
                           help='Prepare only the 50 measured observations (the only current initialize mode)')
        elif command == 'search':
            p.add_argument('--scoring', type=Path, help='Frozen V4 four-metric original-RPO scoring JSON')
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--trials', type=int, default=100, help='Total evaluations including the original 50 initial observations')
            p.add_argument('--history', type=Path, help='Completed prior search to read without modifying it')
            p.add_argument('--history-budget', type=int, help='Select a completed budget from the historical search')
            p.add_argument('--allowed-ankles', nargs='+', help='Permitted new ankle motor IDs (default: all five)')
            p.add_argument('--reference-python', type=Path, help='Dedicated HHTools Python for new AMP geometry')
            p.add_argument('--reference-device', default='cuda:0')
            p.add_argument('--warp-cache', type=Path)
            p.add_argument('--template', type=Path)
            p.add_argument('--mesh-dir', type=Path)
    p = sub.add_parser('score')
    p.add_argument('report', type=Path)
    p.add_argument('--scoring', required=True, type=Path)
    p = sub.add_parser('freeze-scoring', help='Freeze scoring from a completed 200 Hz measurement panel on CPU')
    p.add_argument('--measurements-root', required=True, type=Path)
    p.add_argument('--resume-paused', action='store_true')
    p.add_argument('--original-rpo-reference', type=Path,
                   help='Canonical repository original-RPO reference directory (default when --output is set)')
    p.add_argument('--metric-profile', choices=('four',), default='four',
                   help='Only the four-metric original-RPO profile is active')
    p.add_argument('--metric-weights', type=Path,
                   help='Four-metric weights JSON (default: configs/bo_metric_weights_four.json)')
    p.add_argument('--output', type=Path, required=True, help='Separate output directory for original RPO scoring')
    p = sub.add_parser('ingest')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--conditions', type=Path, default=FORMAL_CONDITIONS)
    p.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.command == 'freeze-scoring':
        work = a.measurements_root.resolve()
        if not work.is_dir():
            parser.error('Completed measurement panel directory does not exist')
        if a.original_rpo_reference is None:
            a.original_rpo_reference = ORIGINAL_RPO_ARCHIVE
        output = a.output.resolve()
        if output.is_relative_to(work) or output.is_relative_to(a.original_rpo_reference.resolve()):
            parser.error('Original RPO scoring requires a separate output directory')
        with run_lock(work), run_lock(output):
            resume_pause(work, a.resume_paused)
            resume_pause(output, a.resume_paused)
            freeze_original_rpo_scoring(work, a.original_rpo_reference, output,
                                        metric_profile='four',
                                        metric_weights=read(a.metric_weights) if a.metric_weights is not None else None)
        return
    if a.command == 'ingest':
        write_new(a.output, ingest_measurements(a.spec, a.conditions))
        return
    if a.command == 'score':
        scoring = read(a.scoring)
        report = checked_report(a.report)
        if (scoring.get('method') != ORIGINAL_RPO_FOUR_METRIC_METHOD
                or scoring.get('benchmark', {}).get('ppo_algorithm') != NATIVE_PPO_ALGORITHM
                or report.get('ppo_algorithm') != NATIVE_PPO_ALGORITHM):
            raise ValueError('The score CLI only accepts V4 four-metric original-RPO scoring')
        reference, source = checked_original_rpo_reference(ORIGINAL_RPO_ARCHIVE, scoring['benchmark'], report['conditions'])
        weights = {task:{name:row['weight'] for name,row in items.items()}
                   for task,items in scoring['metrics'].items()}
        expected = original_rpo_scoring(reference, source, scoring['benchmark'], scoring['measurement_panel'],
                                        report['conditions'], metric_weights=weights)
        if expected != scoring:
            raise ValueError('Scoring differs from the canonical V4 original-RPO normalization')
        print(json.dumps(objective(report, scoring), indent=2, allow_nan=False))
        return
    if min(a.updates,a.timeout) < 1 or a.gpu < 0 or (a.command == 'search' and a.trials < 50):
        parser.error('Expected a positive PPO and BO budget, timeout, and valid GPU index')
    a.root = (a.root or (MIGRATED_ROOT if a.migrated else DEFAULT_ROOT)).resolve()
    if (a.migrated or (a.root/'migration_in_progress.json').is_file()) and not (a.root/'migration_completed.json').is_file():
        parser.error('Complete migration of the selected experiment root before BO')
    formal = a.command == 'initialize'
    conditions_path = a.conditions or PHYSICS_CONDITIONS
    conditions = read_conditions(conditions_path)
    if a.command in ('initialize','evaluate') and (a.updates != DEFAULT_BO_UPDATES or
            official_conditions_path(digest(conditions)) != PHYSICS_CONDITIONS):
        raise ValueError('New BO measurements require the V4 100-update, fixed 200 Hz protocol')
    if (getattr(a, 'reuse_finetuned_panel', None) or getattr(a, 'measure_only', False)) and 'measurement_dt_s' not in conditions:
        parser.error('Panel reevaluation and measurement-only calibration require 200 Hz conditions')
    if a.command == 'search':
        if a.updates != DEFAULT_BO_UPDATES:
            parser.error('Formal continuous BO requires 100 joint PPO updates per structure')
        if a.history_budget is not None and a.history is None:
            parser.error('--history-budget requires --history')
        if a.history is not None:
            a.history = a.history.resolve()
        default_scoring = a.root/'bo'/'scoring_original_rpo_four_metrics_native_expert_prior'/'scoring.json'
        a.scoring = (a.scoring or default_scoring).resolve()
        scoring = read(a.scoring)
        if scoring.get('method') != ORIGINAL_RPO_FOUR_METRIC_METHOD:
            raise ValueError('Continuous BO requires the four-metric original-RPO ratio scoring; historical scores are read-only')
        if (scoring.get('benchmark', {}).get('ppo_algorithm') != NATIVE_PPO_ALGORITHM or
                scoring['benchmark'].get('native_ppo_settings') != NATIVE_PPO_SETTINGS):
            raise ValueError('Native PPO BO requires native original-RPO measurements and initial panel; old scoring remains available for read-only scoring')
        formal = True
        if a.conditions is None:
            conditions_path = official_conditions_path(scoring['conditions_sha256'])
            conditions = read_conditions(conditions_path)
        if (scoring.get('conditions_sha256') != digest(conditions)
                or official_conditions_path(digest(conditions)) != PHYSICS_CONDITIONS):
            raise ValueError('Continuous BO requires the same official 200 Hz conditions as the scoring and measurements')
    a.conditions = conditions_path.resolve()
    continuation = {}
    if a.shared_continuation is not None:
        if (conditions.get('reset_mode') != 'fixed' or a.updates not in (DEFAULT_BO_UPDATES, PREVIOUS_BO_UPDATES)
                or (a.command == 'search' and not formal)):
            parser.error('A continuation actor requires formal single-seed conditions and the default PPO budget')
        formal = True
        continuation['continuation'] = a.shared_continuation.resolve()
    if formal:
        official_conditions_path(digest(conditions))
    source, plan, actor = shared_input(a.root, formal=formal, **continuation,
                                      require_horizontal_selection=formal and (
                                          a.updates == PREVIOUS_BO_UPDATES or uses_current_schedule(a.updates, conditions)))
    if a.command == 'evaluate' and a.candidate not in plan['teachers']['stairs']['design_ids']:
        parser.error('Candidate must be an exact shared Rough structure ID')
    if getattr(a, 'finetuned', None) and not actor_receipt(a.finetuned).is_file():
        parser.error('--finetuned needs the completed candidate run directory')
    if a.command == 'initialize':
        label = ('calibration50_native_expert_prior_ppo100_200hz' if uses_current_schedule(a.updates, conditions)
                 and 'measurement_dt_s' in conditions else calibration_label(conditions, a.updates))
    elif a.command == 'search':
        label = 'search_continuous_four_metrics'
    else:
        label = a.candidate + '_single_seed' if conditions.get('reset_mode') == 'fixed' else a.candidate
    if a.command != 'initialize' and uses_current_schedule(a.updates, conditions):
        label += '_native_expert_prior_ppo100'
    if a.command != 'initialize' and 'measurement_dt_s' in conditions:
        label += '_200hz'
    work = (a.output or a.root/'bo'/label).resolve()
    with run_lock(work):
        work.mkdir(parents=True, exist_ok=True)
        try:
            if a.command == 'search':
                search(a, work, source, plan, actor, conditions)
            elif a.command == 'initialize':
                initialize(a, work, source, plan, actor, conditions)
            else:
                evaluate(a, work, source, plan, actor, a.candidate, conditions)
        except KeyboardInterrupt:
            (work/'USER_PAUSED').touch()
            raise SystemExit(130)


if __name__ == '__main__':
    main()
