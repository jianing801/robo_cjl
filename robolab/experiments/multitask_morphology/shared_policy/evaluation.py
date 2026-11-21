"""Closed-loop checkpoint selection from first-episode survival and forward progress."""
from copy import deepcopy
import json
import math

GATE_VERSION = "continuous_locomotion_progress_v4"
EVALUATION_VELOCITY_FRAME = "horizontal_heading"
FIXED_COURSE_VELOCITY_FRAME = "world_horizontal_x"
COMPARISON_VERSION = "paired_teacher_design_gap_v2"
RANK_DECIMAL_PLACES = 12  # Remove floating-point tail noise, not a performance tolerance.
SELECTION_RULE = ("Apply the same student-minus-teacher comparison to every task and structure; "
    "first minimize the worst task's mean per-structure success shortfall, then the mean task shortfall. "
    "Structure gains cannot cancel another structure's shortfall. Then maximize equal-task/equal-structure "
    "student success and survival fraction. Ties retain the earlier checkpoint.")


def evaluation_criteria(*, duration_s=20., time_tolerance_s=.1, min_command_progress_ratio=.5,
                        max_missing_tracking_steps=1):
    values = (duration_s, time_tolerance_s, min_command_progress_ratio)
    if (any(isinstance(v, bool) or not math.isfinite(v) for v in values)
            or duration_s <= 0 or not 0 <= time_tolerance_s < duration_s
            or not 0 <= min_command_progress_ratio <= 1
            or type(max_missing_tracking_steps) is not int or max_missing_tracking_steps < 0):
        raise ValueError('Invalid engineering locomotion acceptance thresholds')
    return dict(version=GATE_VERSION, duration_s=float(duration_s), time_tolerance_s=float(time_tolerance_s),
        min_command_progress_ratio=float(min_command_progress_ratio),
        max_missing_tracking_steps=max_missing_tracking_steps,
        success_scope='initial-episode survival and continuous forward progress',
        command_scope='continuous locomotion; standing commands are disabled',
        progress_scope='integrated forward speed along commanded direction, only while abs(commanded vx) > 0.05 m/s',
        active_command_threshold_m_s=.05,
        speed_summary='signed progress divided by measured time and commanded distance divided by command exposure time',
        profiles='same survival and progress criteria for nominal and robust')


def _number(row, key, *, nonnegative=False):
    try:
        value = row[key]
    except KeyError as error:
        raise ValueError(f'Missing closed-loop metric: {key}') from error
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Nonfinite or invalid closed-loop metric: {key}')
    if nonnegative and value < 0:
        raise ValueError(f'Negative closed-loop metric: {key}')
    return float(value)


def _count(row, key):
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f'{key} must be a nonnegative integer')
    return value


def _check_duration(seconds, expected, label):
    if not math.isclose(seconds, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f'Inconsistent {label}')


def _check_motion_samples(row, samples, seconds, requested, active, dt):
    command_samples = _count(row, 'command_samples')
    moving_command_samples = _count(row, 'moving_command_samples')
    standing_command_samples = _count(row, 'standing_command_samples')
    if moving_command_samples + standing_command_samples != command_samples:
        raise ValueError('Segmented command counts do not sum to total command_samples')
    _check_duration(requested, command_samples * dt, 'command duration/count')
    _check_duration(active, moving_command_samples * dt, 'active command duration/count')
    if standing_command_samples:
        raise ValueError('Continuous-locomotion evaluation does not accept standing commands')
    if (samples > moving_command_samples or _count(row, 'moving_tracking_samples') != samples
            or _count(row, 'standing_tracking_samples')):
        raise ValueError('Segmented measurement counts do not match the moving episode')
    _check_duration(_number(row, 'moving_tracking_seconds', nonnegative=True), seconds,
                    'moving measurement duration/count')
    _check_duration(_number(row, 'moving_command_seconds', nonnegative=True), active,
                    'moving command duration/count')
    _check_duration(_number(row, 'standing_command_seconds', nonnegative=True), 0., 'standing command duration')
    _check_duration(_number(row, 'standing_tracking_seconds', nonnegative=True), 0., 'standing measurement duration')


def _mean_speed(distance, seconds):
    return distance / seconds if seconds > 0 else None


def _episode_statistics(task, rows, profile, criteria):
    if task not in ('stairs', 'backpack', 'run') or profile not in ('nominal', 'robust') or not rows:
        raise ValueError('A known task/profile and nonempty closed-loop evaluation are required')
    duration = criteria['duration_s']
    time_tolerance = criteria['time_tolerance_s']
    progress_threshold = criteria['min_command_progress_ratio']
    allowed_missing = criteria['max_missing_tracking_steps']
    failed, details = [], []
    survived = fully_covered = 0
    survival_sum = commanded_distance = measured_distance = command_seconds = measured_seconds = 0.
    for row in rows:
        environment = row.get('environment')
        if type(environment) is not int or environment < 0 or type(row.get('fell')) is not bool:
            raise ValueError('Evaluation requires integer environment IDs and a boolean fell flag')
        seconds = _number(row, 'seconds', nonnegative=True)
        samples = _count(row, 'tracking_samples')
        measured = _number(row, 'tracking_seconds', nonnegative=True)
        requested = _number(row, 'command_seconds', nonnegative=True)
        active = _number(row, 'active_command_seconds', nonnegative=True)
        distance = _number(row, 'commanded_distance_m', nonnegative=True)
        aligned = _number(row, 'command_aligned_distance_m')
        dt = _number(row, 'step_dt_s', nonnegative=True)
        if dt <= 0:
            raise ValueError('A positive control timestep is required')
        _check_duration(measured, samples * dt, 'measurement duration/count')
        if (measured > requested + 1e-3 or active > requested + 1e-3
                or abs(requested - seconds) > 1e-3):
            raise ValueError('Measurement and command exposure must describe the same initial episode')
        _check_motion_samples(row, samples, measured, requested, active, dt)
        if (active > 0) != (distance > 0) or (not samples and abs(aligned) > 1e-8):
            raise ValueError('Distance is inconsistent with command or measurement exposure')

        survival = not row['fell'] and seconds >= duration - time_tolerance
        progress = aligned / distance if distance > 0 else None
        coverage = samples > 0 and requested - measured <= allowed_missing * dt + 1e-6
        survived += int(survival)
        fully_covered += int(coverage)
        survival_sum += min(seconds / duration, 1.)
        commanded_distance += distance
        measured_distance += aligned
        command_seconds += active
        measured_seconds += measured

        reasons = []
        if not survival:
            reasons.append('fell_or_incomplete_horizon')
        if not coverage:
            reasons.append('missing_progress_measurement')
        if progress is not None and progress < progress_threshold:
            reasons.append('insufficient_command_relative_progress')
        if reasons:
            failed.append(environment)
        details.append(dict(environment=environment, success=not reasons, survival=survival,
            reasons=reasons, command_progress_ratio=progress, progress_measurement_complete=coverage,
            mean_command_aligned_speed_m_s=_mean_speed(aligned, measured),
            mean_commanded_speed_m_s=_mean_speed(distance, active)))

    count = len(rows)
    return dict(environments=count, failed=failed, success_rate=1 - len(failed) / count,
        survival_rate=survived / count, mean_survival_fraction=survival_sum / count,
        survival_fraction_sum=survival_sum, complete_measurement_environments=fully_covered,
        measurement_coverage_fraction=fully_covered / count, commanded_distance_m=commanded_distance,
        command_aligned_distance_m=measured_distance, command_seconds=command_seconds,
        measured_seconds=measured_seconds,
        mean_command_aligned_speed_m_s=_mean_speed(measured_distance, measured_seconds),
        mean_commanded_speed_m_s=_mean_speed(commanded_distance, command_seconds),
        criteria=criteria, environment_results=details)


def evaluation_statistics(task, rows, profile, *, duration_s=20., time_tolerance_s=.1,
                          min_command_progress_ratio=.5, max_missing_tracking_steps=1):
    """Retain every structure instead of hiding regressions in a job average."""
    groups = {}
    for row in rows:
        design = row.get('design')
        if not isinstance(design, str) or not design.strip():
            raise ValueError('Every evaluation row requires a nonempty design identity')
        groups.setdefault(design, []).append(row)
    criteria = evaluation_criteria(duration_s=duration_s, time_tolerance_s=time_tolerance_s,
        min_command_progress_ratio=min_command_progress_ratio, max_missing_tracking_steps=max_missing_tracking_steps)
    result = _episode_statistics(task, rows, profile, criteria)
    result['design_metrics'] = {design: dict(profile=profile,
        **_episode_statistics(task, selected, profile, criteria))
        for design, selected in sorted(groups.items())}
    return result


def _aggregate_statistics(current):
    if any(row['criteria'] != current[0]['criteria'] for row in current):
        raise ValueError('Cannot aggregate evaluations with different acceptance criteria')
    count = sum(row['environments'] for row in current)
    failed = sum(len(row['failed']) for row in current)
    measured_seconds = sum(row['measured_seconds'] for row in current)
    command_seconds = sum(row['command_seconds'] for row in current)
    aligned = sum(row['command_aligned_distance_m'] for row in current)
    commanded = sum(row['commanded_distance_m'] for row in current)
    return dict(environments=count, success_rate=1 - failed / count,
        survival_rate=sum(row['survival_rate'] * row['environments'] for row in current) / count,
        mean_survival_fraction=sum(row['survival_fraction_sum'] for row in current) / count,
        measurement_coverage_fraction=sum(row['complete_measurement_environments'] for row in current) / count,
        command_aligned_distance_m=aligned, commanded_distance_m=commanded,
        mean_command_aligned_speed_m_s=_mean_speed(aligned, measured_seconds),
        mean_commanded_speed_m_s=_mean_speed(commanded, command_seconds),
        measured_seconds=measured_seconds, command_seconds=command_seconds, criteria=current[0]['criteria'])


def _protocol(value, profile):
    if (not isinstance(value, dict) or value.get('profile') != profile
            or not isinstance(value.get('version'), str) or not value['version']
            or not isinstance(value.get('protocol_hash'), str) or not value['protocol_hash']):
        raise ValueError('Evaluation protocol identity/version/profile/hash is required')
    strength = _number(value, 'strength', nonnegative=True)
    if strength > 1:
        raise ValueError('Invalid evaluation protocol strength')
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError('Evaluation protocol must contain finite JSON metadata') from error
    return deepcopy(value)


def _exposure(record, environments):
    profile, seed = record.get('profile'), record.get('seed')
    if profile not in ('nominal', 'robust') or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Every evaluation exposure requires a valid profile and seed')
    if type(environments) is not int or environments <= 0:
        raise ValueError('Evaluation exposure requires a positive environment count')
    return dict(profile=profile, seed=seed, environments=environments,
        evaluation_protocol=_protocol(record.get('evaluation_protocol'), profile))


def aggregate_evaluations(records, tasks):
    """Pool matching trials within a structure; preserve full exposure identity.

    One design belongs to one job at a given profile/seed in this pipeline.
    Repeating such a record is rejected rather than counted as extra trials.
    """
    if not tasks or len(set(tasks)) != len(tasks) or {row.get('task') for row in records} != set(tasks):
        raise ValueError('Evaluation records must contain exactly the requested task set')
    result = {}
    for task in tasks:
        current = [row for row in records if row['task'] == task]
        designs = {}
        exposures = {}
        for record in current:
            selected = record.get('design_metrics')
            if not isinstance(selected, dict) or not selected:
                raise ValueError('Evaluation records require per-design metrics')
            if sum(row['environments'] for row in selected.values()) != record['environments']:
                raise ValueError('Per-design environment counts differ from the task record')
            for design, metric in selected.items():
                if not isinstance(design, str) or not design.strip():
                    raise ValueError('Invalid design identity in evaluation metrics')
                if metric['criteria'] != record['criteria'] or metric.get('profile') != record.get('profile'):
                    raise ValueError('Per-design criteria/profile differ from the evaluation record')
                exposure = _exposure(record, metric['environments'])
                key = (exposure['profile'], exposure['seed'])
                coverage = exposures.setdefault(design, {})
                if key in coverage:
                    raise ValueError(f'Duplicate design/profile/seed exposure: {task}/{design}/{key}')
                coverage[key] = exposure
                designs.setdefault(design, []).append(metric)
        result[task] = _aggregate_statistics(current)
        result[task]['design_metrics'] = {}
        for design, metrics in sorted(designs.items()):
            combined = _aggregate_statistics(metrics)
            combined['exposure_coverage'] = [value for _, value in sorted(exposures[design].items())]
            result[task]['design_metrics'][design] = combined
    return result


def _validate_metric(row):
    if type(row.get('environments')) is not int or row['environments'] <= 0:
        raise ValueError('Compared metrics require a positive environment count')
    for key in ('success_rate', 'survival_rate', 'mean_survival_fraction', 'measurement_coverage_fraction'):
        value = _number(row, key, nonnegative=True)
        if value > 1 + 1e-8:
            raise ValueError(f'Invalid evaluation fraction: {key}')


def _coverage(metric):
    entries = metric.get('exposure_coverage')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Compared design metrics require profile/seed/protocol exposure coverage')
    result = {}
    for entry in entries:
        exposure = _exposure(entry, entry.get('environments'))
        key = (exposure['profile'], exposure['seed'])
        if key in result:
            raise ValueError('Duplicate profile/seed in compared exposure coverage')
        result[key] = exposure
    if sum(entry['environments'] for entry in result.values()) != metric['environments']:
        raise ValueError('Exposure coverage count differs from the design environment count')
    return result


def teacher_comparison(student, teacher):
    """Uniform observed teacher differences, with no teacher quality boundary.

    For each structure, first pool its matching-seed trials, then calculate the
    positive shortfall. A gain on another structure cannot erase that shortfall.
    """
    if not student or student.keys() != teacher.keys():
        raise ValueError('Comparison requires the same complete task set')
    result = dict(version=COMPARISON_VERSION,
        scope='Matched-protocol/seed evaluation estimates only; not statistical significance, teacher quality categories, or proof of mastery',
        weighting='pool matched seeds within each structure; equal structures within task; equal tasks',
        shortfall_rule='max(teacher success rate - student success rate, 0) separately for every structure', tasks={})
    for task in sorted(teacher):
        s_task, t_task = student[task], teacher[task]
        _validate_metric(s_task)
        _validate_metric(t_task)
        if s_task.get('criteria') != t_task.get('criteria') or not isinstance(s_task.get('criteria'), dict):
            raise ValueError('Student and teacher acceptance criteria differ or are missing')
        s_designs, t_designs = s_task.get('design_metrics'), t_task.get('design_metrics')
        if not s_designs or not t_designs or s_designs.keys() != t_designs.keys():
            raise ValueError(f'Student and teacher design sets differ or are missing: {task}')
        for current in (s_task, t_task):
            if sum(row['environments'] for row in current['design_metrics'].values()) != current['environments']:
                raise ValueError('Compared per-design counts differ from task environment count')
        comparisons = {}
        for design in sorted(t_designs):
            s, t = s_designs[design], t_designs[design]
            _validate_metric(s)
            _validate_metric(t)
            if s.get('criteria') != s_task['criteria'] or t.get('criteria') != t_task['criteria']:
                raise ValueError('Per-design acceptance criteria differ from task criteria')
            if s['environments'] != t['environments']:
                raise ValueError(f'Student and teacher environment counts differ: {task}/{design}')
            if _coverage(s) != _coverage(t):
                raise ValueError(f'Student and teacher profile/seed/protocol exposure coverage differs: {task}/{design}')
            gap = s['success_rate'] - t['success_rate']
            comparisons[design] = dict(student=deepcopy(s), teacher=deepcopy(t), success_rate_gap=gap,
                success_rate_shortfall=max(-gap, 0.), success_rate_gain=max(gap, 0.),
                survival_fraction_gap=s['mean_survival_fraction'] - t['mean_survival_fraction'])
        n = len(comparisons)
        result['tasks'][task] = dict(design_count=n, designs=comparisons,
            mean_success_rate_shortfall=sum(row['success_rate_shortfall'] for row in comparisons.values()) / n,
            mean_success_rate_gap=sum(row['success_rate_gap'] for row in comparisons.values()) / n,
            mean_success_rate_gain=sum(row['success_rate_gain'] for row in comparisons.values()) / n,
            mean_student_success_rate=sum(row['student']['success_rate'] for row in comparisons.values()) / n,
            mean_student_survival_fraction=sum(row['student']['mean_survival_fraction'] for row in comparisons.values()) / n)
    values = list(result['tasks'].values())
    result.update(worst_task_mean_shortfall=max(row['mean_success_rate_shortfall'] for row in values),
        mean_task_shortfall=sum(row['mean_success_rate_shortfall'] for row in values) / len(values),
        mean_task_gap=sum(row['mean_success_rate_gap'] for row in values) / len(values),
        mean_task_gain=sum(row['mean_success_rate_gain'] for row in values) / len(values),
        mean_student_success_rate=sum(row['mean_student_success_rate'] for row in values) / len(values),
        mean_student_survival_fraction=sum(row['mean_student_survival_fraction'] for row in values) / len(values))
    return result


def candidate_rank(student, teacher):
    """Minimize per-structure shortfalls without a teacher pass/fail boundary."""
    comparison = teacher_comparison(student, teacher)
    rank = (-comparison['worst_task_mean_shortfall'], -comparison['mean_task_shortfall'],
        comparison['mean_student_success_rate'], comparison['mean_student_survival_fraction'])
    # Reports retain the unrounded observed differences. Ranking alone removes
    # numerical tails so mathematically tied shortfalls reach later criteria.
    return tuple(round(value, RANK_DECIMAL_PLACES) if isinstance(value, float) else value for value in rank)
