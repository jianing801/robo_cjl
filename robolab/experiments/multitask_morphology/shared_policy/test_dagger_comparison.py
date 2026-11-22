"""Paired DAgger comparisons use the recorded design and episode populations."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_comparison import summarize_comparison
from evaluation import EVALUATION_VELOCITY_FRAME, aggregate_evaluations, evaluation_statistics


SEEDS = (20261117, 20261119, 20261120)
STRUCTURE_KEYS = [[.2, .3, 'K1', 'A1'], [.3, .4, 'K1', 'A1']]
DESIGN_IDS = {task: [task + '_seen', task + '_held'] for task in ('stairs', 'run')}


def _episode(environment, design, speed, seconds):
    steps = int(seconds * 50)
    return dict(
        environment=environment, design=design, fell=seconds < 20, seconds=seconds,
        tracking_samples=steps, tracking_seconds=seconds, command_seconds=seconds,
        active_command_seconds=seconds, commanded_distance_m=.5 * seconds,
        command_aligned_distance_m=speed * seconds, step_dt_s=.02,
        command_samples=steps, moving_command_samples=steps, standing_command_samples=0,
        moving_tracking_samples=steps, standing_tracking_samples=0,
        moving_tracking_seconds=seconds, moving_command_seconds=seconds,
        standing_command_seconds=0., standing_tracking_seconds=0.,
    )


def _report(round_index, observations):
    protocol = dict(profile='robust', version='test', protocol_hash='test', strength=1.)
    records = []
    for seed, samples in zip(SEEDS, observations, strict=True):
        for task, designs in DESIGN_IDS.items():
            rows = [_episode(i, design, *sample)
                    for i, (design, sample) in enumerate(zip(designs, samples, strict=True))]
            records.append(dict(
                task=task, seed=seed, profile='robust', evaluation_protocol=protocol,
                velocity_frame=EVALUATION_VELOCITY_FRAME, **evaluation_statistics(task, rows, 'robust'),
            ))
    return dict(
        round_index=round_index, student=f'round_{round_index}.pt',
        student_sha256=f'round_{round_index}_sha256', evaluations=records,
        task_metrics=aggregate_evaluations(records, tuple(DESIGN_IDS)),
        by_seed={str(seed): aggregate_evaluations(
            [record for record in records if record['seed'] == seed], tuple(DESIGN_IDS),
        ) for seed in SEEDS},
    )


def _compare(candidate, previous, incumbent=None):
    teacher = _report(0, [((.5, 20), (.5, 20))] * len(SEEDS))
    return summarize_comparison(
        round_index=candidate['round_index'], previous_report=previous, candidate_report=candidate,
        incumbent_report=previous if incumbent is None else incumbent, teacher=teacher['task_metrics'],
        structure_keys=STRUCTURE_KEYS, training_keys=STRUCTURE_KEYS[:1], design_ids=DESIGN_IDS,
    )


def test_reports_seen_held_out_and_full_horizontal_gains_with_paired_counts():
    previous = _report(3, [((.4, 20), (.1, 20))] * len(SEEDS))
    candidate = _report(4, [((.5, 20), (.5, 20)), ((.5, 20), (.3, 10)), ((.5, 20), (.4, 20))])

    result = _compare(candidate, previous)

    assert result['selected_round'] == 4
    assert result['strictly_improves_incumbent'] is True
    assert result['velocity_frame'] == EVALUATION_VELOCITY_FRAME
    for task, metrics in result['tasks'].items():
        assert metrics['all']['design_count'] == 2
        assert metrics['all']['episodes_per_policy'] == 6
        assert metrics['all']['candidate']['mean_command_aligned_speed_m_s'] == pytest.approx(51 / 110)
        assert metrics['seen']['candidate']['success_rate'] == 1.
        assert metrics['seen']['delta_vs_previous']['mean_command_aligned_speed_m_s'] == pytest.approx(.1)
        held_out = metrics['held_out']
        assert held_out['episodes_per_policy'] == 3
        assert held_out['candidate']['success_rate'] == pytest.approx(2 / 3)
        assert held_out['delta_vs_previous']['success_rate'] == pytest.approx(2 / 3)
        assert held_out['delta_vs_previous']['survival_rate'] == pytest.approx(-1 / 3)
        assert held_out['delta_vs_previous']['mean_command_aligned_speed_m_s'] == pytest.approx(.32)
        details = metrics['designs'][task + '_held']
        assert details['structure_key'] == STRUCTURE_KEYS[1]
        assert details['split'] == 'held_out'
        assert [row['seed'] for row in details['exposure_coverage']] == list(SEEDS)


def test_tied_rank_retains_the_incumbent_even_with_higher_speed():
    previous = _report(3, [((.4, 20), (.1, 20))] * len(SEEDS))
    candidate = _report(4, [((.6, 20), (.15, 20))] * len(SEEDS))

    result = _compare(candidate, previous)

    assert result['ranks']['candidate'] == result['ranks']['incumbent']
    assert result['selected_round'] == 3
    assert result['selected_student'] == previous['student']
    assert result['selected_student_sha256'] == previous['student_sha256']
    assert result['strictly_improves_incumbent'] is False


def test_candidate_can_improve_on_previous_round_without_beating_incumbent():
    incumbent = _report(3, [((.4, 20), (.4, 20))] * len(SEEDS))
    previous = _report(4, [((.1, 20), (.1, 20))] * len(SEEDS))
    candidate = _report(5, [((.4, 20), (.1, 20))] * len(SEEDS))

    result = _compare(candidate, previous, incumbent)

    assert result['strictly_improves_previous'] is True
    assert result['strictly_improves_incumbent'] is False
    assert result['selected_round'] == 3
    observed = result['tasks']['stairs']['all']
    assert observed['delta_vs_previous']['success_rate'] == .5
    assert observed['delta_vs_incumbent']['success_rate'] == -.5


def test_rejects_changed_velocity_frame():
    previous = _report(3, [((.4, 20), (.1, 20))] * len(SEEDS))
    candidate = _report(4, [((.5, 20), (.5, 20))] * len(SEEDS))
    candidate['evaluations'][0]['velocity_frame'] = 'legacy'

    with pytest.raises(ValueError, match='velocity frame'):
        _compare(candidate, previous)


def test_rejects_report_metrics_that_disagree_with_the_recorded_episodes():
    previous = _report(3, [((.4, 20), (.1, 20))] * len(SEEDS))
    candidate = _report(4, [((.5, 20), (.5, 20))] * len(SEEDS))
    candidate['task_metrics']['stairs']['success_rate'] = 0.

    with pytest.raises(ValueError, match='recorded episodes'):
        _compare(candidate, previous)
