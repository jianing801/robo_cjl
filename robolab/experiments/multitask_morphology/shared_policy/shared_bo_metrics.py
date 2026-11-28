"""Three-task performance metrics, independent of training and simulator APIs."""
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np
from stair_course import COURSE

TASKS = ('stairs', 'backpack', 'run')
KNEE_ANKLE = frozenset(f'{side}_{joint}_joint' for side in ('left', 'right')
                      for joint in ('knee', 'ankle_pitch', 'ankle_roll'))
METRICS = {task: ('cot', 'forward_rmse_mps',
                  'roll_pitch_rms_rad_s' if task == 'stairs' else 'knee_ankle_max_rated_ratio')
           for task in TASKS}
# Keep the native v1 report and its three-metric payload unchanged. Every
# required quantity already exists in `observed`, including metrics not used
# by the former task-specific score.
FOUR_METRICS = {task: ('cot', 'forward_rmse_mps', 'roll_pitch_rms_rad_s',
                       'knee_ankle_max_rated_ratio') for task in TASKS}
PRICE_METRIC = 'motor_total_price_cny'
FIVE_METRICS = {task: (*names, PRICE_METRIC) for task, names in FOUR_METRICS.items()}
ORIGINAL_PRICE_KEY = (.25, .30, 'DM-J10010L-2EC', 'DM-J4340P-2EC')
MOTOR_PRICE_IDS = {
    'knee': {'RS04', 'DM-J10010L-2EC', 'DM-J10010-2EC', 'DM-J6248P-2EC', 'DM-J10422P-2EC'},
    'ankle': {'RS06', 'DM-J4340P-2EC', 'DM-J8009P-2EC-V1.1', 'DM-J4340P-2EC-V1.1-48V', 'DM-J6216P-2EC'},
}
TRIAL_VERSION = 'rpo_three_task_mechanical_positive_v1'
REPORT_VERSION = 'rpo_three_task_bo_evaluation_v1'
SCORING_VERSION = 'rpo_three_task_bo_scoring_v1'
FOUR_METRIC_SCORING_VERSION = 'rpo_three_task_bo_scoring_v2'
FOUR_METRIC_SCORING_METHOD = 'post_finetune_original_rpo_four_metrics'
FIVE_METRIC_SCORING_VERSION = 'rpo_three_task_bo_scoring_v3'
FIVE_METRIC_SCORING_METHOD = 'post_finetune_original_rpo_five_metrics'
DT = .02
PHYSICS_DT = .005
POSITIVE_COT_BASIS = 'positive_mechanical'
ABSOLUTE_COT_BASIS = 'absolute_mechanical'
FORMAL_TASK_WEIGHTS = dict(stairs=.4, backpack=.3, run=.3)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_four_metric_weights(value):
    """Validate an explicit four-metric profile without altering its weights."""
    if not isinstance(value, dict) or set(value) != set(TASKS):
        raise ValueError('Four-metric weights require exactly stairs, backpack and run')
    result = {}
    for task, names in FOUR_METRICS.items():
        row = value[task]
        if (not isinstance(row, dict) or set(row) != set(names)
                or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0
                       for w in row.values())
                or not math.isclose(math.fsum(row.values()), 1., rel_tol=0., abs_tol=1e-12)):
            raise ValueError(f'Invalid {task} four-metric weights: exact metrics, nonnegative finite values, sum one required')
        result[task] = {name: row[name] for name in names}
    return result


def validate_five_metric_weights(value):
    if not isinstance(value, dict) or set(value) != set(TASKS):
        raise ValueError('Five-metric weights require exactly three tasks')
    for task, names in FIVE_METRICS.items():
        row = value[task]
        if (not isinstance(row, dict) or set(row) != set(names)
                or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in row.values())
                or not math.isclose(math.fsum(row.values()), 1., rel_tol=0., abs_tol=1e-12)):
            raise ValueError(f'Invalid {task} five-metric weights')
    return {task: {name: value[task][name] for name in names} for task, names in FIVE_METRICS.items()}


def validate_motor_prices(prices):
    if not isinstance(prices, dict) or set(prices) != set(MOTOR_PRICE_IDS):
        raise ValueError('Invalid motor price catalog')
    for group, ids in MOTOR_PRICE_IDS.items():
        row = prices[group]
        if (not isinstance(row, dict) or set(row) != ids
                or any(type(v) is not int or v <= 0 for v in row.values())):
            raise ValueError(f'Invalid {group} motor price catalog')
    return {group: dict(row) for group, row in prices.items()}


def motor_total_price(key, prices):
    if not isinstance(key, (tuple, list)) or len(key) != 4:
        raise ValueError('Invalid motor structure key')
    try:
        return 2 * prices['knee'][key[2]] + 4 * prices['ankle'][key[3]]
    except (KeyError, TypeError) as exc:
        raise ValueError('Unknown motor in price catalog') from exc


def read_conditions(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    fields = {'kind', 'duration_s', 'seeds', 'copies', 'tasks'}
    if (not fields <= value.keys() <= fields | {'reset_mode', 'min_forward_progress_ratio', 'measurement_dt_s'}
            or ('reset_mode' in value and value['reset_mode'] != 'fixed')
            or value['kind'] != TRIAL_VERSION or set(value['tasks']) != set(TASKS)):
        raise ValueError('Invalid three-task conditions')
    if 'measurement_dt_s' in value and value['measurement_dt_s'] != PHYSICS_DT:
        raise ValueError('Physics-step BO measurements require a 5 ms interval')
    if 'min_forward_progress_ratio' in value:
        ratio = value['min_forward_progress_ratio']
        if not positive(ratio) or ratio > 1:
            raise ValueError('Minimum forward progress ratio must be in (0, 1]')
    seconds, seeds, copies = value['duration_s'], value['seeds'], value['copies']
    if (not positive(seconds) or not 0 < seconds <= 20 or abs(seconds / DT - round(seconds / DT)) > 1e-8
            or type(copies) is not int or not 1 <= copies <= 32
            or not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32-1_000_000 for s in seeds)):
        raise ValueError('Conditions require integral control steps, unique seeds and 1–32 copies')
    stair = value['tasks']['stairs']
    timed = stair.get('course') == COURSE
    geometry = {'course', 'travel_margin_factor'} if timed else {'step_count', 'approach_m'}
    fields = dict(run={'route_distance_m'}, backpack={'payload_kg', 'route_distance_m'},
                  stairs={'step_height_m', 'step_depth_m'} | geometry)
    for task, spec in value['tasks'].items():
        if (spec.keys() != fields[task] | {'speed_mps'}
                or any(not positive(v) for k, v in spec.items() if k != 'course')):
            raise ValueError(f'Invalid fixed {task} conditions')
    if (value['tasks']['backpack']['payload_kg'] > 15 or (timed and stair['travel_margin_factor'] <= 1)
            or (not timed and (type(stair['step_count']) is not int or stair['step_count'] > 64
                or stair['approach_m'] + stair['step_count'] * stair['step_depth_m'] >= 30))):
        raise ValueError('Conditions exceed the supported payload or stair geometry')
    return value


def score_trial(task, *, speed, dt, expected_steps, mass, forward, vx, omega, torque,
                joint_vel, motors, fell, timeout=False, reached_top=None, route_completed=None,
                protection_stop=False, payload=None, outside_stairs=False, min_forward_progress_ratio=None):
    """Samples use world +X velocity/displacement and joint-output torque/velocity."""
    count = len(vx)
    torque, joint_vel = (np.asarray(x, dtype=float) for x in (torque, joint_vel))
    vx, omega = (np.asarray(x, dtype=float) for x in (vx, omega))
    frames = (vx, omega, torque, joint_vel)
    row = dict(status='invalid', reason=None, steps=count, dt_s=dt, fell=bool(fell),
               forward_m=None, mean_forward_mps=None, command_mps=speed,
               reached_top=reached_top, route_completed=route_completed, timed_out=bool(timeout),
               protection_stop=bool(protection_stop), payload=payload, total_mass_kg=mass,
               metrics=None, observed=None, motors=None)
    if outside_stairs:
        return dict(row, reason='outside_stair_course')
    if (not 0 < count <= expected_steps or (count != expected_steps and not fell)
            or (timeout and count < expected_steps and not fell)):
        if fell and count == 0:
            return dict(row, status='failure', reason='fell')
        return dict(row, reason='incomplete_measurement')
    if (not math.isfinite(forward) or not positive(speed) or not positive(dt)
            or vx.shape != (count,) or omega.shape != (count, 2)
            or torque.shape != (count, len(motors)) or joint_vel.shape != torque.shape
            or not all(np.isfinite(x).all() for x in frames)):
        if fell:
            return dict(row, status='failure', reason='fell')
        return dict(row, reason='nonfinite_or_malformed_measurement')
    mean_v = fmean(vx)
    if min_forward_progress_ratio is None:
        progress_failure = 'below_minimum_forward_speed' if mean_v < .10 * speed else None
    else:
        # Compute the cutoff from the full window, keeping exact boundaries such as 6 m.
        commanded_m = Decimal(str(speed)) * Decimal(str(dt)) * expected_steps
        minimum_m = float(commanded_m * Decimal(str(min_forward_progress_ratio)))
        progress_failure = 'below_minimum_forward_progress' if forward < minimum_m else None
    reason = 'fell' if fell else progress_failure
    status = 'failure' if reason else 'success'
    power = torque * joint_vel
    energy = float(np.maximum(power, 0).sum() * dt)
    rmse = float(np.sqrt(np.mean((vx - speed) ** 2)))
    sway = float(np.sqrt(np.mean(np.sum(omega ** 2, axis=1))))
    rms = np.sqrt(np.mean(torque ** 2, axis=0))
    peaks = np.abs(torque).max(axis=0)
    loads = {}
    for spec, joint_rms, peak in zip(motors, rms, peaks, strict=True):
        rated = spec.get('rated_torque_nm')
        loads[spec['joint']] = dict(spec, rms_torque_nm=float(joint_rms),
            peak_torque_observed_nm=float(peak),
            rated_ratio=float(joint_rms / rated) if positive(rated) else None)
    six = [loads[j]['rated_ratio'] for j in KNEE_ANKLE if j in loads]
    motor_cost = max(six) if len(six) == 6 and all(x is not None for x in six) else None
    row.update(status=status, reason=reason,
               forward_m=forward, mean_forward_mps=mean_v, motors=loads)
    observed = dict(cot=energy / (mass * 9.81 * forward) if positive(mass) and forward > 0 else None,
        forward_rmse_mps=rmse, roll_pitch_rms_rad_s=sway, knee_ankle_max_rated_ratio=motor_cost,
        positive_mechanical_work_j=energy)
    if dt == PHYSICS_DT:
        absolute_work = float(np.abs(power).sum() * dt)
        observed.update(absolute_mechanical_work_j=absolute_work,
                        absolute_mechanical_cot=absolute_work / (mass * 9.81 * forward)
                        if positive(mass) and forward > 0 else None)
    row['observed'] = observed
    if status == 'success':
        row['metrics'] = {name: observed[name] for name in METRICS[task]}
    return row


def paired_trials(reports, conditions):
    """Pair by seed and replica ordinal; keep each task and infrastructure failure visible."""
    pairs = {(seed, replica): {} for seed in conditions['seeds']
             for replica in range(conditions['copies'])}
    for report in reports:
        task, seed = report['task'], report['seed']
        for row in report['trials']:
            group = pairs[seed, row['replica']]
            if task in group or task not in TASKS or row['status'] not in ('success', 'failure', 'invalid'):
                raise ValueError('Duplicate or malformed candidate trial')
            group[task] = row
    result = []
    for (seed, replica), tasks in pairs.items():
        if set(tasks) != set(TASKS):
            raise ValueError('Missing task in a paired BO repetition')
        statuses = {task: tasks[task]['status'] for task in TASKS}
        joint = None if 'invalid' in statuses.values() else all(v == 'success' for v in statuses.values())
        result.append(dict(seed=seed, replica=replica, feasible=joint, tasks=tasks))
    return result


def scoring_cot_basis(scoring):
    basis = scoring.get('cot_basis', POSITIVE_COT_BASIS)
    if basis not in (POSITIVE_COT_BASIS, ABSOLUTE_COT_BASIS):
        raise ValueError('Unknown mechanical CoT scoring basis')
    return basis


def trial_metric(trial, name, cot_basis, *, metric_profile='three'):
    if metric_profile == 'four':
        if trial.get('dt_s') != PHYSICS_DT or cot_basis != ABSOLUTE_COT_BASIS:
            raise ValueError('Four-metric scoring requires 200 Hz absolute mechanical measurements')
        if trial.get('status') != 'success':
            raise ValueError('Four-metric scoring requires a successful task trial')
        raw_name = 'absolute_mechanical_cot' if name == 'cot' else name
        observed = trial.get('observed')
        value = observed.get(raw_name) if isinstance(observed, dict) else None
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Missing or invalid observed metric in successful trial: {raw_name}')
        return value
    if metric_profile != 'three':
        raise ValueError(f'Unknown metric profile: {metric_profile}')
    if name == 'cot' and cot_basis == ABSOLUTE_COT_BASIS:
        value = (trial.get('observed') or {}).get('absolute_mechanical_cot')
        if (trial.get('dt_s') != PHYSICS_DT or type(value) not in (int, float)
                or not math.isfinite(value) or value < 0):
            raise ValueError('Missing or invalid 200 Hz absolute mechanical CoT')
        return value
    return (trial['metrics'] or {}).get(name)


def validate_scoring(value, report):
    five = value.get('method') == FIVE_METRIC_SCORING_METHOD
    four = value.get('method') == FOUR_METRIC_SCORING_METHOD
    ratio = four or five
    version = FIVE_METRIC_SCORING_VERSION if five else FOUR_METRIC_SCORING_VERSION if four else SCORING_VERSION
    if (value.get('kind') != version or value.get('domain') != report.get('domain')
            or value.get('conditions_sha256') != report.get('conditions_sha256')):
        raise ValueError('Scoring and evaluation must have the same domain and conditions')
    def weights(values, keys):
        return (set(values) == set(keys)
            and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values.values())
            and math.isclose(sum(values.values()), 1., abs_tol=1e-12))
    if not weights(value['task_weights'], TASKS) or set(value['metrics']) != set(TASKS):
        raise ValueError('Task weights must be nonnegative and sum to one')
    cot_basis = scoring_cot_basis(value)
    if ratio and (value.get('domain') != 'simulation' or cot_basis != ABSOLUTE_COT_BASIS
                 or value['task_weights'] != FORMAL_TASK_WEIGHTS):
        raise ValueError('Four-metric Original RPO scoring requires absolute mechanical CoT and 0.4/0.3/0.3 task weights')
    if ratio:
        (validate_five_metric_weights if five else validate_four_metric_weights)(
            {task: {name: row['weight'] for name, row in items.items()} for task, items in value['metrics'].items()})
    if five:
        prices = validate_motor_prices(value.get('motor_prices_cny'))
        base_price = motor_total_price(ORIGINAL_PRICE_KEY, prices)
        if any(items[PRICE_METRIC]['baseline'] != base_price for items in value['metrics'].values()):
            raise ValueError('Original RPO motor price baseline differs from frozen catalog')
    if cot_basis == ABSOLUTE_COT_BASIS:
        if 'conditions' in report and report['conditions'].get('measurement_dt_s') != PHYSICS_DT:
            raise ValueError('Absolute mechanical CoT scoring requires 200 Hz measurements')
        if value.get('method') == 'post_finetune_panel50' and value['task_weights'] != FORMAL_TASK_WEIGHTS:
            raise ValueError('Formal 200 Hz scoring requires 0.4/0.3/0.3 task weights')
    for task, items in value['metrics'].items():
        task_weights = {k: row['weight'] for k, row in items.items()}
        names = FIVE_METRICS[task] if five else FOUR_METRICS[task] if four else METRICS[task]
        if not weights(task_weights, names):
            raise ValueError(f'Invalid {task} metric weights')
        for row in items.values():
            baseline, scale = row['baseline'], row.get('scale')
            if ratio and (not positive(baseline) or scale is not None):
                raise ValueError('Four-metric Original RPO scoring requires positive fixed ratio denominators')
            if (type(baseline) not in (int, float) or not math.isfinite(baseline) or baseline < 0
                    or (scale is None and not positive(baseline)) or (scale is not None and not positive(scale))):
                raise ValueError('Zero or unreliable baseline requires an explicit positive physical scale')


def objective(report, scoring):
    if (report.get('domain') == 'simulation' and
            report.get('ppo_algorithm') != scoring.get('benchmark', {}).get('ppo_algorithm')):
        raise ValueError('Cannot mix BO observations from different PPO algorithms')
    validate_scoring(scoring, report)
    cot_basis = scoring_cot_basis(scoring)
    five = scoring.get('method') == FIVE_METRIC_SCORING_METHOD
    metric_profile = 'four' if five or scoring.get('method') == FOUR_METRIC_SCORING_METHOD else 'three'
    price = motor_total_price(report.get('candidate', {}).get('structure_key'),
                              scoring['motor_prices_cny']) if five else None
    observations = []
    for pair in report['pairs']:
        feasible = pair['feasible']
        if feasible is None:
            continue
        result = dict(seed=pair['seed'], replica=pair['replica'], feasible=feasible, objective=None)
        if five:
            result[PRICE_METRIC] = price
        if feasible:
            normalized = {}
            for task, specs in scoring['metrics'].items():
                raw = {key: price if five and key == PRICE_METRIC else
                       trial_metric(pair['tasks'][task], key, cot_basis, metric_profile=metric_profile)
                       for key in specs}
                if any(value is None for value in raw.values()):
                    continue
                normalized[task] = {}
                for key, spec in specs.items():
                    x, b = raw[key], spec['baseline']
                    if not math.isfinite(x) or x < 0:
                        raise ValueError('Invalid metric in successful trial')
                    scale = spec.get('scale')
                    normalized[task][key] = x / b if scale is None else 1 + (x - b) / scale
            if len(normalized) == len(TASKS):
                costs = {task: sum(scoring['metrics'][task][k]['weight'] * v for k, v in values.items())
                         for task, values in normalized.items()}
                result.update(normalized=normalized, task_objectives=costs,
                              objective=sum(scoring['task_weights'][task] * x for task, x in costs.items()))
        observations.append(result)
    scores = [r['objective'] for r in observations if r['objective'] is not None]
    successes = sum(row['feasible'] for row in observations)
    return dict(observations=observations, valid_pairs=len(observations), joint_successes=successes,
                invalid_pairs=len(report['pairs']) - len(observations), complete_objectives=len(scores),
                mean_objective=fmean(scores) if scores else None)
