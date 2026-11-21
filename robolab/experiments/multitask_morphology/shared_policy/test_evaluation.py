"""Pure metric tests: no locomotion or native simulation claims."""
import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (evaluation_statistics, aggregate_evaluations, candidate_rank,
    evaluation_criteria, teacher_comparison, COMPARISON_VERSION)
from student_observations import TrackingError


def row(environment=0, **changes):
    value = dict(environment=environment, design='design_a', seconds=20., fell=False, forward_m=9.5,
        mean_forward_speed=.475, ascent_m=0., tracking_samples=999, tracking_rmse=.1,
        tracking_seconds=19.98, command_seconds=20., active_command_seconds=20.,
        commanded_distance_m=10., command_aligned_distance_m=9.5,
        commanded_vx_rms_m_s=.5, step_dt_s=.02)
    value.update(changes)
    command_count = round(value['command_seconds'] / value['step_dt_s'])
    moving = value['active_command_seconds'] > 0
    value['command_samples'] = command_count
    for name, selected in (('moving', moving), ('standing', not moving)):
        value.update({f'{name}_command_samples': command_count if selected else 0,
            f'{name}_command_seconds': value['command_seconds'] if selected else 0.,
            f'{name}_commanded_vx_rms_m_s': value['commanded_vx_rms_m_s'] if selected and command_count else None,
            f'{name}_tracking_samples': value['tracking_samples'] if selected else 0,
            f'{name}_tracking_seconds': value['tracking_seconds'] if selected else 0.,
            f'{name}_tracking_rmse': value['tracking_rmse'] if selected else None})
    value['standing_vx_rms_m_s'] = value['tracking_rmse'] if not moving and value['tracking_samples'] else None
    return value


def record(task, rows, *, profile='robust', seed=1):
    return dict(task=task, profile=profile, seed=seed,
        evaluation_protocol=dict(profile=profile, version='test_protocol', protocol_hash='test_hash',
            strength=1. if profile == 'robust' else 0.),
        **evaluation_statistics(task, rows, profile))


def metrics(rows):
    records = [record(task, rows)
        for task in ('stairs', 'backpack', 'run')]
    return aggregate_evaluations(records, ('stairs', 'backpack', 'run'))


@pytest.mark.parametrize('task', ['stairs', 'backpack', 'run'])
def test_acceptance_reports_progress_with_no_instantaneous_error_metric(task):
    sample = row(command_aligned_distance_m=5., tracking_rmse=.9)
    for name in ('tracking_rmse', 'moving_tracking_rmse', 'standing_tracking_rmse'):
        del sample[name]
    report = evaluation_statistics(task, [sample], 'robust')
    assert report['success_rate'] == 1.
    assert report['mean_command_aligned_speed_m_s'] == pytest.approx(5 / 19.98)
    assert report['mean_commanded_speed_m_s'] == .5
    assert 'tracking_rmse_mean' not in report
    assert 'tracking_rel_tol' not in report['criteria']


@pytest.mark.parametrize('profile', ['nominal', 'robust'])
def test_standing_commands_are_outside_continuous_locomotion_protocol(profile):
    standing = row(tracking_samples=1000, tracking_seconds=20., tracking_rmse=.03,
        commanded_distance_m=0., command_aligned_distance_m=0., active_command_seconds=0.,
        commanded_vx_rms_m_s=0., mean_forward_speed=-.02, forward_m=-.4, ascent_m=-.1)
    with pytest.raises(ValueError, match='does not accept standing commands'):
        evaluation_statistics('stairs', [standing], profile)


def test_surviving_without_following_running_command_fails_and_loses_selection():
    stationary = row(tracking_samples=1000, tracking_seconds=20., tracking_rmse=2.,
        commanded_distance_m=40., command_aligned_distance_m=0., commanded_vx_rms_m_s=2.)
    report = evaluation_statistics('run', [stationary], 'robust')
    assert report['survival_rate'] == 1. and report['success_rate'] == 0.
    teacher = metrics([row() for _ in range(20)])
    running = metrics([row(i, fell=(i == 0)) for i in range(20)])
    standing = metrics([dict(stationary, environment=i) for i in range(20)])
    assert candidate_rank(running, teacher) > candidate_rank(standing, teacher)


def test_empty_measurements_are_null_and_cannot_pass():
    empty = row(seconds=.02, fell=True, tracking_samples=0, tracking_seconds=0., tracking_rmse=None,
        command_seconds=.02, active_command_seconds=.02, commanded_distance_m=.01,
        command_aligned_distance_m=0.)
    report = evaluation_statistics('run', [empty], 'robust')
    assert report['success_rate'] == 0. and report['mean_command_aligned_speed_m_s'] is None
    assert report['measurement_coverage_fraction'] == 0.
    teacher = metrics([row()])
    assert candidate_rank(metrics([row(fell=True)]), teacher) > candidate_rank(metrics([empty]), teacher)
    json.dumps(dict(report=report, rank=candidate_rank(metrics([empty]), teacher)), allow_nan=False)
    with pytest.raises(ValueError, match='inconsistent with command or measurement'):
        evaluation_statistics('run', [dict(empty, command_aligned_distance_m=.1)], 'robust')


def test_only_one_terminal_measurement_may_be_missing():
    assert evaluation_statistics('run', [row()], 'robust')['success_rate'] == 1.
    short = row(tracking_samples=998, tracking_seconds=19.96)
    report = evaluation_statistics('run', [short], 'robust')
    assert report['survival_rate'] == 1. and report['success_rate'] == 0.
    assert 'missing_progress_measurement' in report['environment_results'][0]['reasons']
    assert evaluation_statistics('run', [short], 'robust', max_missing_tracking_steps=2)['success_rate'] == 1.


def test_sparse_measurement_cannot_beat_complete_progress():
    sparse = row(tracking_samples=1, tracking_seconds=.02, tracking_rmse=.0, command_aligned_distance_m=.01)
    complete = row(tracking_rmse=.5)
    teacher = metrics([row()])
    assert candidate_rank(metrics([complete]), teacher) > candidate_rank(metrics([sparse]), teacher)


def test_stairs_progress_does_not_require_ascent_measurement_or_reaching_top():
    result = evaluation_statistics('stairs', [row(ascent_m=.2), row(1)], 'nominal')
    assert result['success_rate'] == 1.


def test_legacy_rmse_values_cannot_change_survival_or_progress_acceptance():
    first = row(tracking_rmse=.06, commanded_vx_rms_m_s=.2)
    second = row(tracking_rmse=.9, commanded_vx_rms_m_s=.2)
    assert evaluation_statistics('stairs', [first], 'robust') == evaluation_statistics('stairs', [second], 'robust')
    with pytest.raises(TypeError):
        evaluation_criteria(tracking_rel_tol=.3)


def test_motion_progress_gate_is_relative_and_standing_drift_cannot_substitute():
    slow_progress = row(commanded_distance_m=10., command_aligned_distance_m=4.9,
        forward_m=100., tracking_rmse=.01)
    assert evaluation_statistics('backpack', [slow_progress], 'robust')['success_rate'] == 0.
    assert evaluation_statistics('backpack', [slow_progress], 'robust',
        min_command_progress_ratio=.4)['success_rate'] == 1.


def test_aggregation_pools_actual_distance_and_control_time():
    empty = row(1, tracking_samples=0, tracking_seconds=0., tracking_rmse=None, command_aligned_distance_m=0.)
    first = record('run', [row(command_aligned_distance_m=6.), empty], seed=1)
    second = record('run', [row(command_aligned_distance_m=8., commanded_distance_m=12.)], seed=2)
    result = aggregate_evaluations([first, second], ['run'])['run']
    assert result['success_rate'] == pytest.approx(2/3)
    assert result['measurement_coverage_fraction'] == pytest.approx(2/3)
    assert result['mean_command_aligned_speed_m_s'] == pytest.approx(14 / 39.96)
    assert result['mean_commanded_speed_m_s'] == pytest.approx(32 / 60)


def test_inconsistent_exposure_and_missing_legacy_metrics_are_rejected():
    with pytest.raises(ValueError, match='same initial episode'):
        evaluation_statistics('run', [row(command_seconds=10.)], 'robust')
    old = row()
    del old['tracking_samples']
    with pytest.raises(ValueError, match='tracking_samples'):
        evaluation_statistics('run', [old], 'robust')
    with pytest.raises(ValueError, match='Invalid engineering'):
        evaluation_criteria(min_command_progress_ratio=float('nan'))


def test_acceptance_protocol_cannot_change_between_teacher_and_candidate():
    teacher = metrics([row()])
    student = metrics([row()])
    student['run']['criteria'] = evaluation_criteria(min_command_progress_ratio=.6)
    with pytest.raises(ValueError, match='criteria differ'):
        candidate_rank(student, teacher)


def recorded_row(commands, actual, *, missing_last=False):
    tracker = TrackingError(1, 'cpu')
    for i, (command, speed) in enumerate(zip(commands, actual)):
        tracker.update(torch.tensor([command], dtype=torch.float64), torch.tensor([speed], dtype=torch.float64),
            torch.tensor([not (missing_last and i == len(commands)-1)]), dt=.02,
            command_valid=torch.tensor([True]))
    value = dict(environment=0, design='design_a', seconds=len(commands)*.02, fell=False,
        mean_forward_speed=0., ascent_m=0.)
    value.update(tracker.rows()[0])
    return value


@pytest.mark.parametrize('residual_speed', [0., .09, .2])
def test_mixed_old_episodes_are_rejected_instead_of_silently_dropping_standing(residual_speed):
    mixed = recorded_row([2., 2., 0., 0.], [2., 2., residual_speed, residual_speed])
    with pytest.raises(ValueError, match='does not accept standing commands'):
        evaluation_statistics('run', [mixed], 'robust', duration_s=.08, time_tolerance_s=0.)


def test_good_standing_cannot_dilute_bad_moving_tracking():
    mixed = recorded_row([.2, 0., 0., 0.], [.12, 0., 0., 0.])
    assert mixed['tracking_rmse'] == pytest.approx(.04)
    assert mixed['moving_tracking_rmse'] == pytest.approx(.08)
    with pytest.raises(ValueError, match='does not accept standing commands'):
        evaluation_statistics('run', [mixed], 'robust', duration_s=.08, time_tolerance_s=0.)


def test_standing_acceptance_setting_is_removed_and_progress_threshold_is_unchanged():
    with pytest.raises(TypeError):
        evaluation_criteria(standing_speed_tol_m_s=.1)
    criteria = evaluation_criteria()
    assert 'standing_gate' not in criteria and 'standing_speed_tol_m_s' not in criteria
    assert criteria['min_command_progress_ratio'] == .5
    assert 'forward progress' in criteria['success_scope']


def test_terminal_reset_velocity_is_excluded_but_last_command_is_counted():
    value = recorded_row([.2, .2, .2, .2], [.2, .2, .2, 999.], missing_last=True)
    assert value['command_samples'] == 4 and value['tracking_samples'] == 3
    assert value['moving_command_samples'] == 4 and value['moving_tracking_samples'] == 3
    assert value['moving_tracking_rmse'] == 0.
    report = evaluation_statistics('run', [value], 'robust', duration_s=.08, time_tolerance_s=0.)
    assert report['success_rate'] == 1.
    reset_standing = recorded_row([.2, .2, 0.], [.2, .2, 999.], missing_last=True)
    assert reset_standing['standing_tracking_samples'] == 0
    with pytest.raises(ValueError, match='does not accept standing commands'):
        evaluation_statistics('run', [reset_standing], 'robust', duration_s=.06, time_tolerance_s=0.)


@pytest.mark.parametrize('field,value,match', [
    ('moving_tracking_samples', 998, 'measurement counts'),
    ('moving_command_samples', 999, 'command counts'),
    ('moving_command_seconds', 19., 'duration/count'),
    ('moving_tracking_seconds', 19., 'duration/count'),
    ('standing_tracking_seconds', .02, 'standing measurement duration'),
])
def test_segmented_metrics_must_be_consistent(field, value, match):
    value_row = row()
    value_row[field] = value
    with pytest.raises(ValueError, match=match):
        evaluation_statistics('run', [value_row], 'robust')



def outcome_rows(outcomes):
    return [row(environment=i, design=design, command_aligned_distance_m=9.5 if success else 4.)
        for i, (design, success) in enumerate((design, success)
            for design, successes in outcomes.items() for success in successes)]


def one_task(outcomes, *, profile='robust', seed=1):
    return aggregate_evaluations([record('run', outcome_rows(outcomes), profile=profile, seed=seed)], ['run'])


def test_design_statistics_are_retained_and_require_design_identity():
    stats = evaluation_statistics('run', outcome_rows({'a': [True, False], 'b': [True]}), 'robust')
    assert stats['design_metrics']['a']['environments'] == 2
    assert stats['design_metrics']['a']['success_rate'] == .5
    assert stats['design_metrics']['b']['success_rate'] == 1.
    missing = row()
    del missing['design']
    with pytest.raises(ValueError, match='design identity'):
        evaluation_statistics('run', [missing], 'robust')


def test_structure_gain_cannot_cancel_another_structure_shortfall():
    teacher = one_task({'a': [True, True, False, False], 'b': [True, True, False, False]})
    student = one_task({'a': [True, False, False, False], 'b': [True, True, True, False]})
    assert teacher['run']['success_rate'] == student['run']['success_rate'] == .5
    compared = teacher_comparison(student, teacher)
    task = compared['tasks']['run']
    assert task['designs']['a']['success_rate_gap'] == -.25
    assert task['designs']['a']['success_rate_shortfall'] == .25
    assert task['designs']['b']['success_rate_gain'] == .25
    assert task['mean_success_rate_shortfall'] == .125
    assert task['mean_success_rate_gap'] == 0.
    assert candidate_rank(teacher, teacher) > candidate_rank(student, teacher)
    assert candidate_rank(student, teacher)[0] == -.125
    assert compared['version'] == COMPARISON_VERSION


def test_zero_teacher_uses_the_same_finite_difference_formula():
    teacher = one_task({'a': [False, False]})
    student = one_task({'a': [True, False]})
    compared = teacher_comparison(student, teacher)
    design = compared['tasks']['run']['designs']['a']
    assert design['success_rate_gap'] == design['success_rate_gain'] == .5
    assert design['success_rate_shortfall'] == 0.
    assert compared['mean_task_shortfall'] == 0.
    json.dumps(compared, allow_nan=False)
    assert 'Matched-protocol/seed' in compared['scope']


def test_structure_weight_is_equal_even_with_different_environment_counts():
    teacher = one_task({'a': [True], 'b': [False] * 10})
    student = one_task({'a': [False], 'b': [True] * 10})
    comparison = teacher_comparison(student, teacher)
    assert student['run']['success_rate'] == pytest.approx(10/11)
    assert comparison['tasks']['run']['mean_success_rate_shortfall'] == .5
    assert comparison['mean_student_success_rate'] == .5
    assert comparison['mean_task_gap'] == 0.


def test_same_structure_pools_matched_seeds_before_positive_shortfall():
    teacher = aggregate_evaluations([record('run', outcome_rows({'a': [True]}), seed=1),
        record('run', outcome_rows({'a': [False]}), seed=2)], ['run'])
    student = aggregate_evaluations([record('run', outcome_rows({'a': [False]}), seed=1),
        record('run', outcome_rows({'a': [True]}), seed=2)], ['run'])
    result = teacher_comparison(student, teacher)
    assert result['worst_task_mean_shortfall'] == 0.
    assert teacher['run']['design_metrics']['a']['environments'] == 2
    assert [entry['seed'] for entry in teacher['run']['design_metrics']['a']['exposure_coverage']] == [1, 2]


def test_different_design_batches_merge_without_losing_structure_counts():
    split = aggregate_evaluations([record('run', outcome_rows({'a': [True]})),
        record('run', outcome_rows({'b': [False, True]}))], ['run'])
    combined = one_task({'a': [True], 'b': [False, True]})
    assert teacher_comparison(split, combined)['mean_task_gap'] == 0.
    assert split['run']['design_metrics']['b']['exposure_coverage'][0]['environments'] == 2


@pytest.mark.parametrize('kind', ['task', 'design', 'count', 'seed', 'profile', 'strength', 'source'])
def test_unmatched_teacher_student_exposure_is_rejected(kind):
    teacher = one_task({'a': [True, False]})
    student = deepcopy(teacher)
    if kind == 'task':
        student = {'stairs': student['run']}
    elif kind == 'design':
        student = one_task({'b': [True, False]})
    elif kind == 'count':
        student = one_task({'a': [True]})
    elif kind == 'seed':
        student = one_task({'a': [True, False]}, seed=2)
    elif kind == 'profile':
        student = one_task({'a': [True, False]}, profile='nominal')
    else:
        protocol = student['run']['design_metrics']['a']['exposure_coverage'][0]['evaluation_protocol']
        if kind == 'strength':
            protocol['strength'] = .5
        else:
            protocol['source_sha256'] = 'different_source'
    with pytest.raises(ValueError, match='task set|design sets|counts differ|coverage differs'):
        teacher_comparison(student, teacher)


def test_missing_speed_remains_null_without_replacing_it_with_zero():
    empty = row(tracking_samples=0, tracking_seconds=0., tracking_rmse=None, command_aligned_distance_m=0.)
    student = aggregate_evaluations([record('run', [empty])], ['run'])
    teacher = one_task({'design_a': [True]})
    comparison = teacher_comparison(student, teacher)
    compared = comparison['tasks']['run']['designs']['design_a']
    assert compared['student']['mean_command_aligned_speed_m_s'] is None
    assert 'tracking_rmse_gap' not in compared
    assert candidate_rank(student, teacher) < candidate_rank(teacher, teacher)
    json.dumps(comparison, allow_nan=False)


def test_duplicate_exposure_and_missing_protocol_are_rejected():
    first = record('run', [row()])
    with pytest.raises(ValueError, match='Duplicate design/profile/seed'):
        aggregate_evaluations([first, deepcopy(first)], ['run'])
    del first['evaluation_protocol']
    with pytest.raises(ValueError, match='protocol identity'):
        aggregate_evaluations([first], ['run'])


def test_speed_variations_do_not_break_ties_between_equal_progress_outcomes():
    def errors(first, second):
        rows = [row(0, design='a', tracking_rmse=first)]
        rows += [row(i + 1, design='b', tracking_rmse=second) for i in range(9)]
        return aggregate_evaluations([record('run', rows)], ['run'])
    teacher = errors(.2, .2)
    first, second = errors(.2, .4), errors(.5, .25)
    assert candidate_rank(first, teacher) == candidate_rank(second, teacher)


def test_complete_measurements_are_checked_per_design_not_only_per_task():
    empty = row(0, design='a', tracking_samples=0, tracking_seconds=0., tracking_rmse=None,
        command_aligned_distance_m=0.)
    student = aggregate_evaluations([record('run', [empty,
        *[row(i + 1, design='b') for i in range(9)]])], ['run'])
    teacher = one_task({'a': [False], 'b': [False] * 9})
    rank = candidate_rank(student, teacher)
    assert student['run']['measurement_coverage_fraction'] == pytest.approx(.9)
    assert student['run']['design_metrics']['a']['measurement_coverage_fraction'] == 0.
    assert student['run']['design_metrics']['b']['measurement_coverage_fraction'] == 1.
    assert rank[2] == .5
    json.dumps(rank, allow_nan=False)


def test_equal_theoretical_shortfalls_reach_absolute_success_tie_breaker():
    teacher = one_task({'a': [True] * 8 + [False] * 2, 'b': [True] * 7 + [False] * 3})
    high_success = one_task({'a': [True] * 6 + [False] * 4, 'b': [True] * 9 + [False]})
    low_success = one_task({'a': [True] * 8 + [False] * 2, 'b': [True] * 5 + [False] * 5})
    # 0.8-0.6 and 0.7-0.5 are both 20 percentage points. Binary float tails
    # differ, but must not override the later absolute student success term.
    raw_high = teacher_comparison(high_success, teacher)['worst_task_mean_shortfall']
    raw_low = teacher_comparison(low_success, teacher)['worst_task_mean_shortfall']
    assert raw_high > raw_low
    first, second = candidate_rank(high_success, teacher), candidate_rank(low_success, teacher)
    assert first[:2] == second[:2] == (-.1, -.1)
    assert first[2] > second[2] and first > second
