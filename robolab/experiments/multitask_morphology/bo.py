"""Evaluate and search exact structures after shared-policy distillation."""
import argparse
import json
from pathlib import Path
from statistics import fmean, stdev
import os
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED = HERE/'shared_policy'
DEFAULT_ROOT = REPO/'robolab/logs/multitask_full2500_physical_v1'
sys.path[:0] = [str(SHARED), str(REPO/'robolab'), str(REPO)]

from shared_bo_metrics import (DT, KNEE_ANKLE, METRICS, REPORT_VERSION, SCORING_VERSION, TASKS, TRIAL_VERSION,
                               digest, objective, paired_trials, positive, read_conditions, score_trial,
                               validate_scoring)
from contract import load_plan, sha, write_new
from persistence import resume_pause, run_lock


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def freeze(path, data):
    if path.is_file():
        if read(path) != data:
            raise ValueError(f'Frozen BO identity differs: {path}')
    else:
        write_new(path, data)


def shared_input(root):
    path = root/'shared_plan/plan.json'
    plan = load_plan(path)
    manifest = read(root/'shared_policy/manifest.json')
    complete = read(root/'shared_policy/completed.json')
    student = Path(complete['student']).resolve()
    if (tuple(plan['schema']['tasks']) != TASKS or manifest.get('plan_sha256') != sha(path)
            or complete.get('protocol_sha256') != manifest.get('protocol_sha256')
            or sha(student) != complete['student_sha256']):
        raise ValueError('Shared three-task plan or selected student identity does not match')
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
        sources={p.name: sha(p) for p in (Path(__file__), SHARED/'bo_capture.py', SHARED/'shared_bo_metrics.py',
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
        if (tuning/'manifest.json').is_file():
            command.append('--resume')
        if a.resume_paused:
            command.append('--resume-paused')
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
            score = score_trial(task, speed=speed, dt=DT, expected_steps=round(conditions['duration_s'] / DT),
                mass=trial['total_mass_kg'], forward=forward, vx=samples['vx'], omega=samples['omega'],
                torque=samples['torque'], joint_vel=samples['joint_vel'], motors=motors, fell=trial['fell'],
                reached_top=trial.get('reached_top'), route_completed=trial.get('route_completed'),
                protection_stop=trial.get('protection_stop', False), payload=payload)
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
    freeze(work/'manifest.json', dict(kind='rpo_shared_bo_search_v1', source_plan_sha256=sha(source),
        initial_actor_sha256=sha(actor), conditions_sha256=digest(conditions), scoring_sha256=sha(a.scoring),
        coordinator_sha256=sha(__file__), surrogate_sha256=sha(REPO/'rough_bo/constrained_gp.py'),
        updates=a.updates, seed=a.seed))
    resume_pause(work, a.resume_paused)
    observations, evaluated, results = [], set(), []
    for i in range(a.trials):
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
    complete = [row for row in results if row['mean_objective'] is not None]
    best_observed = min(complete, key=lambda r: r['mean_objective']) if complete else None
    freeze(work/f'budget_{a.trials:04d}.json', dict(trials=len(results), best_observed=best_observed,
        candidates=[{k: r[k] for k in ('design_id', 'structure_key', 'evaluation',
                    'valid_pairs', 'joint_successes', 'complete_objectives', 'mean_objective')} for r in results]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('evaluate', 'search'):
        p = sub.add_parser(command)
        p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
        p.add_argument('--output', type=Path)
        p.add_argument('--conditions', type=Path, default=HERE/'configs/bo_pilot_conditions.json')
        p.add_argument('--gpu', type=int, default=0)
        p.add_argument('--updates', type=int, default=100)
        p.add_argument('--timeout', type=int, default=1800)
        p.add_argument('--resume-paused', action='store_true')
        if command == 'evaluate':
            p.add_argument('--candidate', required=True, help='Exact Rough (stairs) ID in the shared bank')
            p.add_argument('--finetuned', type=Path, help='Reuse a completed candidate run with the same plan and budget')
        else:
            p.add_argument('--scoring', type=Path, required=True)
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
    p.add_argument('--conditions', type=Path, default=HERE/'configs/bo_pilot_conditions.json')
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
    a.conditions = a.conditions.resolve()
    conditions = read_conditions(a.conditions)
    source, plan, actor = shared_input(a.root)
    if a.command == 'evaluate' and a.candidate not in plan['teachers']['stairs']['design_ids']:
        parser.error('Candidate must be an exact shared Rough structure ID')
    if getattr(a, 'finetuned', None) and not (a.finetuned/'completed.json').is_file():
        parser.error('--finetuned needs the completed candidate run directory')
    work = (a.output or a.root/'bo'/('search' if a.command == 'search' else a.candidate)).resolve()
    with run_lock(work):
        work.mkdir(parents=True, exist_ok=True)
        try:
            if a.command == 'search':
                search(a, work, source, plan, actor, conditions)
            else:
                evaluate(a, work, source, plan, actor, a.candidate, conditions)
        except KeyboardInterrupt:
            (work/'USER_PAUSED').touch()
            raise SystemExit(130)


if __name__ == '__main__':
    main()
