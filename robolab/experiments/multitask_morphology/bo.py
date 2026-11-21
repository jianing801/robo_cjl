"""Evaluate and search exact structures after shared-policy distillation."""
import argparse
import json
from pathlib import Path
from statistics import fmean, stdev
from math import isfinite
import os
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED = HERE/'shared_policy'
DEFAULT_ROOT = REPO/'robolab/logs/multitask_full2500_physical'
MIGRATED_ROOT = DEFAULT_ROOT.with_name('multitask_full2500_physical_migrated')
DEFAULT_BO_UPDATES = 200
FORMAL_CONDITIONS = HERE/'configs/bo_conditions.json'
PILOT_CONDITIONS = HERE/'configs/bo_pilot_conditions.json'
sys.path[:0] = [str(SHARED), str(REPO/'robolab'), str(REPO)]

from shared_bo_metrics import (DT, KNEE_ANKLE, METRICS, REPORT_VERSION, SCORING_VERSION, TASKS, TRIAL_VERSION,
                               digest, objective, paired_trials, positive, read_conditions, score_trial,
                               validate_scoring)
from contract import load_plan, sha, write_new
from collection_grid import SELECTION_VERSION, calibration_keys, select_collection_grid, selection_panel
from evaluation import EVALUATION_VELOCITY_FRAME
from persistence import resume_pause, run_lock
from stair_course import COURSE, evaluation_flight, outside_flight


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def freeze(path, data):
    if path.is_file():
        if read(path) != data:
            raise ValueError(f'Frozen BO identity differs: {path}')
    else:
        write_new(path, data)


def check_formal_selection(root, plan, manifest, complete, student, *, require_horizontal_selection=False):
    protocol = manifest.get('protocol', {})
    arguments = manifest.get('arguments', {})
    version = protocol.get('selection_version')
    if version != SELECTION_VERSION:
        raise ValueError(f'Formal BO requires selection protocol {SELECTION_VERSION}, found {version!r}')
    if protocol.get('stair_evaluation') != COURSE:
        raise ValueError('Formal BO requires round selection on the timed ascending stair flight')
    if require_horizontal_selection and protocol.get('evaluation_velocity_frame') != EVALUATION_VELOCITY_FRAME:
        raise ValueError('Formal 200-update BO requires selection in the horizontal heading frame')
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


def shared_input(root, *, formal=False, require_horizontal_selection=False):
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
    return path, plan, student


def completed_actor(root, source_plan, initial, candidate, updates):
    manifest_path = root/'manifest.json'
    complete = read(root/'completed.json')
    manifest = read(manifest_path)
    actor = Path(complete['student'])
    if (complete.get('result') != 'CANDIDATE_FINETUNED_EVALUATED'
            or complete['manifest_sha256'] != sha(manifest_path)
            or manifest['plan_sha256'] != sha(source_plan)
            or manifest['initial_actor_sha256'] != sha(initial)
            or complete['candidate'] != candidate or manifest['candidate'] != candidate
            or complete['ppo_updates'] != updates or manifest['updates'] != updates
            or sha(actor) != complete['student_sha256']
            or sha(complete['evaluation']) != complete['evaluation_sha256']):
        raise ValueError('Fine-tuned candidate/actor identity or PPO budget does not match')
    return actor


def run_worker(command, folder, environment, timeout, root):
    from pipeline import child
    child([str(v) for v in command], folder, environment, timeout, root)


def evaluate_native(root, plan_path, plan, actor, conditions_path, conditions, gpu, timeout):
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    actor_sha = sha(actor)
    expected_conditions = digest(conditions)
    requests = [(seed, task) for seed in conditions['seeds'] for task in TASKS]
    reports = []
    for seed, task in requests:
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
                run_worker(command, attempt/'logs', child_environment(device, os.environ), timeout, root)
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
        if (report['kind'] != TRIAL_VERSION or report['design_id'] != design_id
                or report['task'] != task or report['seed'] != seed
                or report['conditions_sha256'] != expected_conditions
                or report['actor_sha256'] != actor_sha or report['bank_sha256'] != spec['bank_sha256']
                or report['plan_sha256'] != sha(plan_path) or sha(report['trace']) != report['trace_sha256']
                or sorted(row['replica'] for row in report['trials']) != list(range(conditions['copies']))):
            raise ValueError(f'Incomplete or mismatched BO telemetry: {report_path}')
        if any(row.get('reason') == 'outside_stair_course' for row in report['trials']):
            raise ValueError(f'The timed stair course boundary was reached; extend the measurement course: {report_path}')
        if any(row['status'] == 'invalid' for row in report['trials']):
            raise ValueError(f'Invalid trial preserved for diagnosis. Repeat the same command: {report_path}')
        if not commit.exists():
            write_new(commit, evidence)
        reports.append(dict(report, evidence=evidence))
    return reports


def evaluate(a, work, source, original, initial, design_id, conditions):
    from finetune_state import select_candidate
    plan, candidate = select_candidate(original, design_id=design_id)
    if any(not spec['full_body_torque'] for spec in plan['teachers'].values()):
        raise ValueError('BO measurements require the fully configured motor curves')
    identity = dict(kind='rpo_bo_candidate_v1', source_plan_sha256=sha(source), candidate=candidate,
        initial_actor_sha256=sha(initial), conditions_sha256=digest(conditions), updates=a.updates,
        sources={p.name: sha(p) for p in (Path(__file__), SHARED/'bo_capture.py', SHARED/'shared_bo_metrics.py', SHARED/'stair_course.py',
                                         SHARED/'rollout.py', SHARED/'rollout_amp.py')})
    work.mkdir(parents=True, exist_ok=True)
    freeze(work/'manifest.json', identity)
    resume_pause(work, a.resume_paused)
    tuning = (a.finetuned if getattr(a, 'finetuned', None) else work/'tuning').resolve()
    if not (tuning/'completed.json').is_file():
        command = [sys.executable, '-u', HERE/'finetune.py', '--plan', source, '--student', initial,
                   '--output', tuning, '--design-id', design_id, '--updates', str(a.updates),
                   '--device', f'cuda:{a.gpu}', '--evaluation-profile', 'both',
                   '--gpu-min-free-gb', '12.5', '--gpu-max-utilization', '100',
                   '--timeout-seconds', str(a.timeout)]
        tuning_manifest = tuning/'manifest.json'
        if tuning_manifest.is_file():
            command.append('--resume')
            if a.resume_paused:
                command.append('--resume-paused')
            compact = read(tuning_manifest).get('compact_training_artifacts') is True
        else:
            compact = a.updates == DEFAULT_BO_UPDATES
        if compact:
            command.append('--compact-training-artifacts')
        logs = work/'tuning_logs'/uuid4().hex[:8]
        print('BO_FINETUNING', design_id, candidate['structure_key'], logs, flush=True)
        # The native fine-tune coordinator owns GPU leases and per-worker timeouts.
        run_worker(command, logs, os.environ, (a.updates + 18) * a.timeout, work)
    actor = completed_actor(tuning, source, initial, candidate, a.updates)
    plan_path = tuning/'plan.json'
    if load_plan(plan_path) != plan:
        raise ValueError('Fine-tuned physical plan differs from BO candidate')
    reports = evaluate_native(work, plan_path, plan, actor, a.conditions, conditions, a.gpu, a.timeout)
    result = dict(kind=REPORT_VERSION, domain='simulation', candidate=candidate, ppo_updates=a.updates,
        student=str(actor), student_sha256=sha(actor), conditions=conditions, conditions_sha256=digest(conditions),
        evaluation_manifest_sha256=sha(work/'manifest.json'), pairs=paired_trials(reports, conditions),
        measurements=[r['evidence'] for r in reports])
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
    if (specification['domain'] not in ('simulation', 'hardware')
            or specification['dt_s'] != DT or specification['measurement_convention'] != TRIAL_VERSION):
        raise ValueError('Imported arrays must follow the declared mechanical BO convention at 50 Hz')
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
            score = score_trial(task, speed=speed, dt=DT, expected_steps=round(conditions['duration_s'] / DT),
                mass=trial['total_mass_kg'], forward=forward, vx=samples['vx'], omega=samples['omega'],
                torque=samples['torque'], joint_vel=samples['joint_vel'], motors=motors, fell=trial['fell'],
                reached_top=trial.get('reached_top'), route_completed=trial.get('route_completed'),
                protection_stop=trial.get('protection_stop', False), payload=payload, outside_stairs=outside)
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
    return dict(kind=SCORING_VERSION, domain=report['domain'], conditions_sha256=report['conditions_sha256'],
                benchmark=reference, weighting_status='pilot_equal_weights', baseline_successful_pairs=len(pairs),
                task_weights={task: 1/3 for task in TASKS}, metrics=metrics)


def panel_scoring(reports, keys, benchmark):
    """Fit each scale once to equal-weight structure means from three successful seeds."""
    import numpy as np
    if len(keys) != 50 or len(set(keys)) != 50 or [tuple(r['candidate']['structure_key']) for r in reports] != list(keys):
        raise ValueError('Calibration requires the fixed 50 distinct structures in group-21 order')
    first = reports[0]
    conditions = first['conditions']
    expected = {(seed, 0) for seed in conditions['seeds']}
    if len(expected) != 3 or conditions['copies'] != 1:
        raise ValueError('Calibration needs three distinct seeds with one trial each')
    samples = {task: {metric: [] for metric in METRICS[task]} for task in TASKS}
    for report in reports:
        pairs = report['pairs']
        if (report['kind'] != REPORT_VERSION or report['domain'] != 'simulation'
                or report['ppo_updates'] != benchmark['updates'] or report['conditions'] != conditions
                or report['conditions_sha256'] != first['conditions_sha256']
                or len(pairs) != 3 or {(r['seed'],r['replica']) for r in pairs} != expected):
            raise ValueError('Inconsistent calibration conditions, candidate budget or seed coverage')
        for task in TASKS:
            runs = [pair['tasks'][task] for pair in pairs]
            if any(run['status'] == 'invalid' for run in runs):
                raise ValueError('Invalid trial must be resolved before calibration')
            if not all(run['status'] == 'success' for run in runs):
                continue
            for metric in METRICS[task]:
                values = [run['metrics'][metric] for run in runs]
                if any(value is None or not isfinite(value) or value < 0 for value in values):
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
    return dict(kind=SCORING_VERSION,domain='simulation',conditions_sha256=first['conditions_sha256'],
        method='post_finetune_panel50',benchmark=benchmark,metrics=metrics,task_weights={t:1/3 for t in TASKS})


def span_only_panel(scoring):
    return scoring.get('method') == 'post_finetune_panel50' and all(
        row.get('scale') is None for metrics in scoring['metrics'].values() for row in metrics.values()
    )


def check_panel_receipt(scoring_path):
    receipt = read(scoring_path.parent/'completed.json')
    if (receipt['scoring_sha256'] != sha(scoring_path)
            or receipt['manifest_sha256'] != sha(scoring_path.parent/'manifest.json')):
        raise ValueError('Frozen calibration scoring or manifest changed')


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
    return report


def initialize(a, work, source, plan, actor, conditions):
    keys = calibration_keys()
    lookup = {tuple(row['key']):row for row in read(plan['shared_bank'])['structures']}
    if not set(keys) <= set(map(tuple,plan['selected_structure_keys'])):
        raise ValueError('Missing fixed calibration structures in shared plan')
    benchmark = dict(initial_actor_sha256=sha(actor),source_plan_sha256=sha(source),updates=a.updates)
    if a.updates == DEFAULT_BO_UPDATES:
        benchmark['selection_velocity_frame'] = EVALUATION_VELOCITY_FRAME
    manifest = dict(**benchmark, conditions_sha256=digest(conditions), structure_keys=list(map(list, keys)))
    freeze(work/'manifest.json', manifest)
    resume_pause(work,a.resume_paused)
    reports, evidence = [], []
    old_scoring_path = work/'scoring.json'
    previous = read(old_scoring_path) if old_scoring_path.is_file() else None
    if previous is not None and (a.updates == 100 or span_only_panel(previous)):
        reanchor = span_only_panel(previous)
        if reanchor or (work/'completed.json').is_file():
            check_panel_receipt(old_scoring_path)
        previous_benchmark = previous['benchmark']
        if any(previous_benchmark.get(key) != value for key, value in benchmark.items()):
            raise ValueError('Frozen calibration uses a different starting policy, plan or PPO budget')
        for entry in previous_benchmark['reports']:
            relative = Path(entry['report'])
            path = work/relative
            reports.append(initial_report(path, entry, benchmark))
            evidence.append(dict(entry, report=(Path('..')/relative).as_posix()) if reanchor else entry)
        if reanchor:
            work = work/'percentile_scoring'
            work.mkdir(exist_ok=True)
            freeze(work/'manifest.json', manifest)
    else:
        for index, key in enumerate(keys, 1):
            folder = work/f'design_{index:02d}'
            report = evaluate(a, folder, source, plan, actor, lookup[key]['design_ids']['stairs'], conditions)
            path = folder/'bo_evaluation.json'
            reports.append(report)
            evidence.append(dict(report=str(path.relative_to(work)), sha256=sha(path)))
            print('CALIBRATION_PROGRESS', index, '/ 50', flush=True)
    scoring = panel_scoring(reports,keys,dict(benchmark,reports=evidence))
    freeze(work/'scoring.json',scoring)
    freeze(work/'completed.json',dict(scoring_sha256=sha(work/'scoring.json'),
                                      manifest_sha256=sha(work/'manifest.json')))
    print('BO_CALIBRATION_READY',work/'scoring.json',flush=True)


def parameters(row):
    return dict(zip(('thigh_m', 'calf_m', 'knee_motor', 'ankle_motor'), row['key'], strict=True))


def propose(pool, observations, evaluated, schema, seed):
    import numpy as np
    from rough_bo.constrained_gp import ConstrainedSurrogate
    candidates = [r for r in pool if r['design_ids']['stairs'] not in evaluated]
    if not candidates:
        raise ValueError('All exact shared structures have been evaluated')
    model = ConstrainedSurrogate(schema['ranges_m'], schema['knees'], schema['ankles'],
                                 seed=seed, objective_key='objective')
    X = model.encode([parameters(r) for r in candidates])
    if evaluated:
        old = model.encode([parameters(r) for r in pool if r['design_ids']['stairs'] in evaluated])
        distance = np.min(np.sum((X[:, None, :2] - old[None, :, :2]) ** 2, axis=-1)
                          + np.sum(X[:, None, 2:] != old[None, :, 2:], axis=-1), axis=1)
    else:
        distance = -np.sum((X[:, :2] - .5) ** 2, axis=-1)
    if not observations or len(evaluated) < 4:
        index = int(np.argmax(distance))
        return candidates[index], dict(method='initial_space_filling', pool_size=len(candidates))
    model.fit(observations)
    acq = model.acquisition(X)
    if model.regressor is None:
        values, method = acq * (1 + distance), 'feasibility_exploration'
    elif acq.max() <= 1e-16:
        _, deviation, probability = model.predict(X)
        values, method = probability * deviation, 'uncertainty_feasibility_fallback'
    else:
        values, method = acq, 'EI_times_probability_feasible'
    index = int(np.argmax(values))
    mean, std, probability = model.predict(X[index:index+1])
    return candidates[index], dict(method=method, pool_size=len(candidates),
        acquisition=float(values[index]), probability_feasible=float(probability[0]),
        expected_objective=float(mean[0]) if model.regressor is not None else None,
        objective_std=float(std[0]) if model.regressor is not None else None,
        performance_rows=model.performance_count, feasibility_rows=model.feasibility_count)


def search(a, work, source, plan, actor, conditions):
    scoring = read(a.scoring)
    validate_scoring(scoring, dict(domain='simulation', conditions_sha256=digest(conditions)))
    pool = read(plan['shared_bank'])['structures']
    keys = {tuple(k) for k in plan['selected_structure_keys']}
    pool = [r for r in pool if tuple(r['key']) in keys]
    observations, evaluated, results = [], set(), []
    if scoring.get('method') == 'post_finetune_panel50':
        check_panel_receipt(a.scoring)
        if span_only_panel(scoring):
            raise ValueError('Rerun BO initialize to reuse completed measurements with P10/P90 anchors')
        benchmark = scoring['benchmark']
        keys = calibration_keys()
        if (benchmark['initial_actor_sha256'] != sha(actor) or benchmark['source_plan_sha256'] != sha(source)
                or benchmark['updates'] != a.updates or len(benchmark['reports']) != len(keys)):
            raise ValueError('Initial observations must share the selected actor, source plan and PPO budget')
        if (a.updates == DEFAULT_BO_UPDATES
                and benchmark.get('selection_velocity_frame') != EVALUATION_VELOCITY_FRAME):
            raise ValueError('Initial observations must come from horizontal heading checkpoint selection')
        if a.trials < len(keys):
            raise ValueError('The total BO budget must include the 50 initial observations')
        lookup = {tuple(row['key']): row for row in pool}
        for key, evidence in zip(keys,benchmark['reports'],strict=True):
            path = a.scoring.parent/evidence['report']
            report = initial_report(path, evidence, benchmark)
            row = lookup[key]
            design_id = row['design_ids']['stairs']
            if (report['candidate']['design_ids'] != row['design_ids'] or tuple(report['candidate']['structure_key']) != key
                    or report['ppo_updates'] != a.updates):
                raise ValueError('Initial BO observation is attached to another candidate or budget')
            scored = objective(report,scoring)
            observations.extend(dict(obs,params=parameters(row)) for obs in scored['observations'])
            evaluated.add(design_id)
            results.append(dict(design_id=design_id,structure_key=row['key'],evaluation=str(path),**scored))
    freeze(work/'manifest.json', dict(kind='rpo_shared_bo_search_v1', source_plan_sha256=sha(source),
        initial_actor_sha256=sha(actor), conditions_sha256=digest(conditions), scoring_sha256=sha(a.scoring),
        coordinator_sha256=sha(__file__), surrogate_sha256=sha(REPO/'rough_bo/constrained_gp.py'),
        updates=a.updates, seed=a.seed))
    resume_pause(work, a.resume_paused)
    for i in range(len(evaluated), a.trials):
        trial = work/f'trial_{i:04d}'
        trial.mkdir(exist_ok=True)
        request = trial/'proposal.json'
        if request.is_file():
            proposal = read(request)
        else:
            row, audit = propose(pool, observations, evaluated, plan['schema'], a.seed)
            proposal = dict(design_id=row['design_ids']['stairs'], structure_key=row['key'], acquisition=audit)
            write_new(request, proposal)
        design_id = proposal['design_id']
        matched = [r for r in pool if r['design_ids']['stairs'] == design_id and r['key'] == proposal['structure_key']]
        if len(matched) != 1 or design_id in evaluated:
            raise ValueError('Repeated or mismatched frozen BO proposal')
        print('BO_TRIAL', i + 1, '/', a.trials, design_id, flush=True)
        report = evaluate(a, trial/'candidate', source, plan, actor, design_id, conditions)
        scored = objective(report, scoring)
        result = dict(proposal, evaluation=str(trial/'candidate/bo_evaluation.json'), **scored)
        freeze(trial/'observation.json', result)
        observations.extend(dict(row, params=parameters(matched[0])) for row in scored['observations'])
        evaluated.add(design_id)
        results.append(result)
    complete = [row for row in results if row['mean_objective'] is not None
                and (scoring.get('method') != 'post_finetune_panel50' or row['complete_objectives'] == 3)]
    best_observed = min(complete, key=lambda r: r['mean_objective']) if complete else None
    freeze(work/f'budget_{a.trials:04d}.json', dict(trials=len(results), best_observed=best_observed,
        candidates=[{k: r[k] for k in ('design_id', 'structure_key', 'evaluation',
                    'valid_pairs', 'joint_successes', 'complete_objectives', 'mean_objective')} for r in results]))


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
        p.add_argument('--gpu', type=int, default=0)
        p.add_argument('--updates', type=int, default=DEFAULT_BO_UPDATES)
        p.add_argument('--timeout', type=int, default=1800)
        p.add_argument('--resume-paused', action='store_true')
        if command == 'evaluate':
            p.add_argument('--candidate', required=True, help='Exact Rough (stairs) ID in the shared bank')
            p.add_argument('--finetuned', type=Path, help='Reuse a completed candidate run with the same plan and budget')
        elif command == 'search':
            p.add_argument('--scoring', type=Path)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--trials', type=int, required=True, help='Total candidate evaluations including previous commits')
    p = sub.add_parser('calibrate')
    p.add_argument('report', type=Path)
    p.add_argument('--benchmark-name', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('score')
    p.add_argument('report', type=Path)
    p.add_argument('--scoring', required=True, type=Path)
    p = sub.add_parser('ingest')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--conditions', type=Path, default=HERE/'configs/bo_conditions.json')
    p.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.command == 'ingest':
        write_new(a.output, ingest_measurements(a.spec, a.conditions))
        return
    if a.command == 'calibrate':
        report = checked_report(a.report)
        write_new(a.output, calibrate(report, dict(name=a.benchmark_name, report=str(a.report.resolve()),
                                                   report_sha256=sha(a.report))))
        return
    if a.command == 'score':
        print(json.dumps(objective(checked_report(a.report), read(a.scoring)), indent=2, allow_nan=False))
        return
    if min(a.updates,a.timeout) < 1 or a.gpu < 0 or (a.command == 'search' and a.trials < 1):
        parser.error('Expected a positive PPO and BO budget, timeout, and valid GPU index')
    a.root = (a.root or (MIGRATED_ROOT if a.migrated else DEFAULT_ROOT)).resolve()
    if (a.migrated or (a.root/'migration_in_progress.json').is_file()) and not (a.root/'migration_completed.json').is_file():
        parser.error('Complete migration of the selected experiment root before BO')
    formal = a.command == 'initialize'
    default_conditions = FORMAL_CONDITIONS
    if a.command == 'search':
        default_scoring = a.root/'bo/calibration50/scoring.json'
        anchored_scoring = default_scoring.parent/'percentile_scoring/scoring.json'
        if (anchored_scoring.parent/'completed.json').is_file():
            default_scoring = anchored_scoring
        a.scoring = (a.scoring or default_scoring).resolve()
        scoring = read(a.scoring)
        formal = scoring.get('method') == 'post_finetune_panel50'
        if not formal and (a.scoring == default_scoring or 'reports' in scoring.get('benchmark', {})):
            raise ValueError('Official initial observations require the frozen panel scoring method')
        if not formal and scoring.get('conditions_sha256') == digest(read_conditions(PILOT_CONDITIONS)):
            default_conditions = PILOT_CONDITIONS
    a.conditions = (a.conditions or default_conditions).resolve()
    conditions = read_conditions(a.conditions)
    if formal and conditions != read_conditions(FORMAL_CONDITIONS):
        raise ValueError('Official BO calibration and search require the fixed three-task conditions')
    source, plan, actor = shared_input(a.root, formal=formal,
                                      require_horizontal_selection=formal and a.updates == DEFAULT_BO_UPDATES)
    if a.command == 'evaluate' and a.candidate not in plan['teachers']['stairs']['design_ids']:
        parser.error('Candidate must be an exact shared Rough structure ID')
    if getattr(a, 'finetuned', None) and not (a.finetuned/'completed.json').is_file():
        parser.error('--finetuned needs the completed candidate run directory')
    label = {'search':'search','initialize':'calibration50'}.get(a.command,getattr(a,'candidate',None))
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
