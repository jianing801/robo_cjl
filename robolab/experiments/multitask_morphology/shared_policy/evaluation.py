"""Declared engineering gates for closed-loop checkpoint selection.

These tolerances are configurable engineering acceptance settings, not paper-
derived optimal thresholds. Success means complete-episode survival and forward
command tracking, not a full stair-climbing or deployment certification.
"""
from copy import deepcopy
import json
import math

GATE_VERSION = "continuous_locomotion_evaluation_v3"
COMPARISON_VERSION = "paired_teacher_design_gap_v1"
RANK_DECIMAL_PLACES = 12  # Remove floating-point tail noise, not a performance tolerance.
SELECTION_RULE = ("Apply the same student-minus-teacher comparison to every task and structure; "
    "first minimize the worst task's mean per-structure success shortfall, then the mean task shortfall. "
    "Structure gains cannot cancel another structure's shortfall. Then maximize equal-task/equal-structure "
    "student success and survival fraction. "
    "Prefer complete tracking measurements before lower equal-task/equal-structure mean episode vx RMSE; "
    "ties retain the earlier checkpoint. These are observed estimates, not significance or mastery claims.")


def evaluation_criteria(*, duration_s=20., time_tolerance_s=.1, tracking_rel_tol=.25,
                        min_command_progress_ratio=.5,
                        max_missing_tracking_steps=1):
    values = (duration_s, time_tolerance_s, tracking_rel_tol, min_command_progress_ratio)
    if (any(isinstance(v, bool) or not math.isfinite(v) for v in values)
            or duration_s <= 0 or not 0 <= time_tolerance_s < duration_s
            or tracking_rel_tol < 0
            or not 0 <= min_command_progress_ratio <= 1
            or type(max_missing_tracking_steps) is not int or max_missing_tracking_steps < 0):
        raise ValueError('Invalid engineering command-tracking acceptance thresholds')
    return dict(version=GATE_VERSION, duration_s=float(duration_s), time_tolerance_s=float(time_tolerance_s),
        tracking_rel_tol=float(tracking_rel_tol),
        min_command_progress_ratio=float(min_command_progress_ratio),
        max_missing_tracking_steps=max_missing_tracking_steps,
        tolerance_source='configurable engineering acceptance settings; not paper-optimal thresholds',
        success_scope='initial-episode survival and continuous forward-command tracking; not full stair traversal or stop-and-stand validation',
        command_scope='continuous locomotion; standing commands are disabled',
        unexpected_standing='protocol error; never silently omit zero-command periods or count them as success',
        progress_scope='integrated body vx along commanded direction, only while abs(commanded vx) > 0.05 m/s',
        active_command_threshold_m_s=.05,
        tracking_gate='moving-phase vx RMSE <= relative tolerance * moving-phase commanded vx RMS; no absolute floor',
        ranking_tracking_metric='overall vx tracking RMSE remains the diagnostic and ranking tie-breaker',
        empty_tracking='null RMSE and unsuccessful episode',
        profiles='same command-aware criteria for nominal and robust')


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


def _rms(row, key, count):
    if key not in row:
        raise ValueError(f'Missing closed-loop metric: {key}')
    if count:
        return _number(row, key, nonnegative=True)
    if row[key] is not None:
        raise ValueError(f'Zero samples requires null {key}, never fabricated zero')
    return None


def _close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f'Inconsistent segmented {label}')


def _phases(row, count, error, requested, command_rms, dt, active):
    """Check disjoint command/measurement phases and their second moments."""
    if 'command_zero_tolerance_m_s' in row:
        _close(_number(row, 'command_zero_tolerance_m_s', nonnegative=True), .05,
            'standing-command classification threshold')
    command_count = _count(row, 'command_samples')
    _close(requested, command_count * dt, 'command duration/count')
    phases = {}
    for name in ('moving', 'standing'):
        nc = _count(row, f'{name}_command_samples')
        nv = _count(row, f'{name}_tracking_samples')
        if nv > nc:
            raise ValueError(f'{name} tracking samples exceed command samples')
        tc = _number(row, f'{name}_command_seconds', nonnegative=True)
        tv = _number(row, f'{name}_tracking_seconds', nonnegative=True)
        _close(tc, nc * dt, f'{name} command duration/count')
        _close(tv, nv * dt, f'{name} tracking duration/count')
        rms = _rms(row, f'{name}_commanded_vx_rms_m_s', nc)
        err = _rms(row, f'{name}_tracking_rmse', nv)
        if rms is not None and ((name == 'moving' and rms < .05 - 1e-8)
                or (name == 'standing' and rms > .05 + 1e-8)):
            raise ValueError(f'{name} command RMS contradicts the standing-command band')
        phases[name] = dict(command_samples=nc, tracking_samples=nv, command_seconds=tc,
            tracking_seconds=tv, commanded_vx_rms_m_s=rms, tracking_rmse=err)
    if sum(phase['command_samples'] for phase in phases.values()) != command_count:
        raise ValueError('Segmented command counts do not sum to total command_samples')
    if sum(phase['tracking_samples'] for phase in phases.values()) != count:
        raise ValueError('Segmented tracking counts do not sum to total tracking_samples')
    _close(active, phases['moving']['command_seconds'], 'active command seconds')
    _close((command_rms or 0.) ** 2 * requested,
        sum((phase['commanded_vx_rms_m_s'] or 0.) ** 2 * phase['command_seconds'] for phase in phases.values()),
        'command RMS second moments')
    _close((error or 0.) ** 2 * count,
        sum((phase['tracking_rmse'] or 0.) ** 2 * phase['tracking_samples'] for phase in phases.values()),
        'tracking RMSE second moments')
    standing = phases['standing']
    standing['actual_vx_rms_m_s'] = _rms(row, 'standing_vx_rms_m_s', standing['tracking_samples'])
    if standing['tracking_samples']:
        # Only a subset of commanded samples may have measurements. Its command
        # energy cannot exceed all command energy; this bounds the possible
        # difference between residual-speed RMS and command-error RMS.
        max_command_rms = (standing['commanded_vx_rms_m_s'] or 0.) * math.sqrt(
            standing['command_samples'] / standing['tracking_samples'])
        if abs(standing['actual_vx_rms_m_s'] - standing['tracking_rmse']) > max_command_rms + 1e-8:
            raise ValueError('Standing residual speed and tracking error are inconsistent')
    return phases


def _episode_statistics(task, rows, profile, *, duration_s=20., time_tolerance_s=.1,
                          tracking_rel_tol=.25,
                          min_command_progress_ratio=.5, max_missing_tracking_steps=1):
    if task not in ('stairs', 'backpack', 'run') or profile not in ('nominal', 'robust') or not rows:
        raise ValueError('A known task/profile and nonempty closed-loop evaluation are required')
    criteria = evaluation_criteria(duration_s=duration_s, time_tolerance_s=time_tolerance_s,
        tracking_rel_tol=tracking_rel_tol,
        min_command_progress_ratio=min_command_progress_ratio,
        max_missing_tracking_steps=max_missing_tracking_steps)
    failed, details, errors = [], [], []
    survived = fully_covered = legacy_successes = legacy_available = 0
    survival_sum = squared_error_sum = command_seconds_sum = tracking_seconds_sum = 0.
    sample_count = 0
    for row in rows:
        environment = row.get('environment')
        if type(environment) is not int or environment < 0 or type(row.get('fell')) is not bool:
            raise ValueError('Evaluation requires integer environment IDs and a boolean fell flag')
        seconds = _number(row, 'seconds', nonnegative=True)
        count = row.get('tracking_samples')
        if type(count) is not int or count < 0:
            raise ValueError('tracking_samples must be a nonnegative integer')
        if 'tracking_rmse' not in row:
            raise ValueError('Missing closed-loop metric: tracking_rmse')
        error = row['tracking_rmse']
        if count:
            error = _number(row, 'tracking_rmse', nonnegative=True)
            errors.append(error)
            squared_error_sum += error ** 2 * count
        elif error is not None:
            raise ValueError('Zero tracking_samples requires null tracking_rmse, never fabricated zero')
        measured = _number(row, 'tracking_seconds', nonnegative=True)
        requested = _number(row, 'command_seconds', nonnegative=True)
        active = _number(row, 'active_command_seconds', nonnegative=True)
        distance = _number(row, 'commanded_distance_m', nonnegative=True)
        aligned = _number(row, 'command_aligned_distance_m')
        dt = _number(row, 'step_dt_s', nonnegative=True)
        if dt <= 0 or abs(measured - count * dt) > 1e-5:
            raise ValueError('Tracking duration/sample/control-dt mismatch')
        if (measured > requested + 1e-3 or active > requested + 1e-3
                or abs(requested - seconds) > 1e-3):
            raise ValueError('Tracking/command exposure must describe the same initial episode')
        if requested:
            command_rms = _number(row, 'commanded_vx_rms_m_s', nonnegative=True)
        elif row.get('commanded_vx_rms_m_s') is not None:
            raise ValueError('Zero command duration requires null commanded_vx_rms_m_s')
        else:
            command_rms = None
        phases = _phases(row, count, error, requested, command_rms, dt, active)
        if phases['standing']['command_samples']:
            raise ValueError('Continuous-locomotion evaluation does not accept standing commands; use a new motion-only evaluation')
        if active == 0 and (distance > 1e-8 or abs(aligned) > 1e-8):
            raise ValueError('Standing commands cannot accumulate moving-command progress')
        if active > 0 and distance <= 0:
            raise ValueError('Moving command exposure requires positive commanded distance')
        survival = not row['fell'] and seconds >= duration_s - time_tolerance_s
        survived += int(survival)
        # Preserve the exact former gate as a comparison column only. It is
        # intentionally not reused for command-aware acceptance or selection.
        legacy_fields = ('mean_forward_speed', 'ascent_m') if task == 'stairs' else ('mean_forward_speed',)
        if all(field in row for field in legacy_fields):
            speed = _number(row, 'mean_forward_speed')
            ascent = _number(row, 'ascent_m') if task == 'stairs' else None
            legacy_available += 1
            legacy_successes += int(not row['fell'] and seconds >= 19.9 and speed >= .1
                and (task != 'stairs' or ascent >= .1))
        survival_sum += min(seconds / duration_s, 1.)
        command_seconds_sum += requested
        tracking_seconds_sum += measured
        sample_count += count
        moving = phases['moving']
        tolerance = tracking_rel_tol * moving['commanded_vx_rms_m_s'] if moving['command_samples'] else None
        progress = aligned / distance if active > 0 else None
        coverage = (count > 0 and requested - measured <= max_missing_tracking_steps * dt + 1e-6
            and all(phase['tracking_samples'] > 0 for phase in phases.values() if phase['command_samples']))
        fully_covered += int(coverage)
        reasons = []
        if not survival:
            reasons.append('fell_or_incomplete_horizon')
        if not coverage:
            reasons.append('missing_tracking_coverage')
        if moving['command_samples'] and (moving['tracking_rmse'] is None or moving['tracking_rmse'] > tolerance + 1e-8):
            reasons.append('moving_forward_command_tracking')
        if progress is not None and progress < min_command_progress_ratio - 1e-8:
            reasons.append('insufficient_command_relative_progress')
        if reasons:
            failed.append(environment)
        details.append(dict(environment=environment, success=not reasons, survival=survival,
            reasons=reasons, tracking_rmse=error, tracking_tolerance_m_s=tolerance,
            moving_tracking_tolerance_m_s=tolerance,
            phase_measurements={'moving': moving},
            command_progress_ratio=progress,
            tracking_coverage_verified=coverage))
    count = len(rows)
    success_rate = 1 - len(failed) / count
    return dict(environments=count, failed=failed, success_rate=success_rate,
        command_tracking_success_rate=success_rate, survival_rate=survived / count,
        legacy_basic_gait_success_rate=legacy_successes / count if legacy_available == count else None,
        legacy_basic_gait_successes=legacy_successes, legacy_basic_gait_environments=legacy_available,
        legacy_basic_gait_rule='comparison only: no fall, seconds>=19.9, mean_forward_speed>=0.1 m/s; stairs ascent>=0.1 m',
        mean_survival_fraction=survival_sum / count, survival_fraction_sum=survival_sum,
        tracking_rmse_mean=sum(errors) / len(errors) if errors else None,
        tracking_rmse_sum=sum(errors), tracking_squared_error_sum=squared_error_sum,
        tracking_rmse_pooled=math.sqrt(squared_error_sum / sample_count) if sample_count else None,
        tracking_valid_environments=len(errors), tracking_valid_fraction=len(errors) / count,
        tracking_fully_covered_environments=fully_covered, tracking_fully_covered_fraction=fully_covered / count,
        tracking_samples=sample_count, tracking_seconds=tracking_seconds_sum,
        command_seconds=command_seconds_sum, criteria=criteria, environment_results=details)


def evaluation_statistics(task, rows, profile, *, duration_s=20., time_tolerance_s=.1,
                          tracking_rel_tol=.25,
                          min_command_progress_ratio=.5, max_missing_tracking_steps=1):
    """Retain every structure instead of hiding regressions in a job average."""
    groups = {}
    for row in rows:
        design = row.get('design')
        if not isinstance(design, str) or not design.strip():
            raise ValueError('Every evaluation row requires a nonempty design identity')
        groups.setdefault(design, []).append(row)
    parameters = dict(duration_s=duration_s, time_tolerance_s=time_tolerance_s,
        tracking_rel_tol=tracking_rel_tol,
        min_command_progress_ratio=min_command_progress_ratio,
        max_missing_tracking_steps=max_missing_tracking_steps)
    result = _episode_statistics(task, rows, profile, **parameters)
    result['design_metrics'] = {design: dict(profile=profile,
        **_episode_statistics(task, selected, profile, **parameters))
        for design, selected in sorted(groups.items())}
    return result


def _aggregate_statistics(current):
    if any(row['criteria'] != current[0]['criteria'] for row in current):
        raise ValueError('Cannot aggregate evaluations with different acceptance criteria')
    count = sum(row['environments'] for row in current)
    failed = sum(len(row['failed']) for row in current)
    valid = sum(row['tracking_valid_environments'] for row in current)
    samples = sum(row['tracking_samples'] for row in current)
    squared = sum(row['tracking_squared_error_sum'] for row in current)
    success = 1 - failed / count
    legacy_count = sum(row['legacy_basic_gait_environments'] for row in current)
    legacy_successes = sum(row['legacy_basic_gait_successes'] for row in current)
    return dict(environments=count, success_rate=success,
            command_tracking_success_rate=success,
            legacy_basic_gait_success_rate=legacy_successes / count if legacy_count == count else None,
            survival_rate=sum(row['survival_rate'] * row['environments'] for row in current) / count,
            mean_survival_fraction=sum(row['survival_fraction_sum'] for row in current) / count,
            tracking_rmse_mean=sum(row['tracking_rmse_sum'] for row in current) / valid if valid else None,
            tracking_rmse_pooled=math.sqrt(squared / samples) if samples else None,
            tracking_valid_environments=valid, tracking_valid_fraction=valid / count,
            tracking_fully_covered_fraction=sum(row['tracking_fully_covered_environments'] for row in current) / count,
            tracking_samples=samples, tracking_seconds=sum(row['tracking_seconds'] for row in current),
            command_seconds=sum(row['command_seconds'] for row in current), criteria=current[0]['criteria'])


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
    for key in ('success_rate', 'survival_rate', 'mean_survival_fraction', 'tracking_valid_fraction',
            'tracking_fully_covered_fraction'):
        value = _number(row, key, nonnegative=True)
        if value > 1 + 1e-8:
            raise ValueError(f'Invalid evaluation fraction: {key}')
    if row.get('tracking_rmse_mean') is not None:
        _number(row, 'tracking_rmse_mean', nonnegative=True)
    elif row['tracking_valid_fraction'] > 0:
        raise ValueError('Valid tracking fraction requires measured RMSE')


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
                survival_fraction_gap=s['mean_survival_fraction'] - t['mean_survival_fraction'],
                tracking_rmse_gap=(s['tracking_rmse_mean'] - t['tracking_rmse_mean'])
                    if s['tracking_rmse_mean'] is not None and t['tracking_rmse_mean'] is not None else None)
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
    groups = [list(task['design_metrics'].values()) for task in student.values()]
    all_measured = all(design['tracking_rmse_mean'] is not None for group in groups for design in group)
    task_errors = []
    for group in groups:
        measured = [design['tracking_rmse_mean'] for design in group if design['tracking_rmse_mean'] is not None]
        if measured:
            task_errors.append(sum(measured) / len(measured))
    # Missing observations are never imputed as zero error. The final finite
    # sentinel applies only when every error is absent; completeness/coverage
    # precede it, so the sentinel cannot reward a no-data checkpoint.
    rank = (-comparison['worst_task_mean_shortfall'], -comparison['mean_task_shortfall'],
        comparison['mean_student_success_rate'], comparison['mean_student_survival_fraction'],
        int(all_measured),
        sum(sum(row['tracking_fully_covered_fraction'] for row in group) / len(group) for group in groups) / len(groups),
        sum(sum(row['tracking_valid_fraction'] for row in group) / len(group) for group in groups) / len(groups),
        -sum(task_errors) / len(task_errors) if task_errors else 0.)
    # Reports retain the unrounded observed differences. Ranking alone removes
    # numerical tails so mathematically tied shortfalls reach later criteria.
    return tuple(round(value, RANK_DECIMAL_PLACES) if isinstance(value, float) else value for value in rank)
