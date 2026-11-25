import pytest

from native_ppo_convergence import diagnostic_performance, evaluations_enabled


def test_performance_not_gated_by_progress_ratio_but_missing_cot_is_not_fabricated():
    scoring = {'task_weights': {'stairs': .4, 'backpack': .3, 'run': .3},
               'metrics': {task: {'cot': {'baseline': 2., 'weight': .5},
                                  'forward_rmse_mps': {'baseline': .2, 'weight': .5}}
                           for task in ('stairs', 'backpack', 'run')}}
    tasks = {task: dict(observed={'absolute_mechanical_cot': 2., 'forward_rmse_mps': .2},
                        steps=4000, dt_s=.005, forward_m=.1, status='failed', reason='progress')
             for task in scoring['metrics']}
    result = diagnostic_performance({'tasks': tasks}, scoring, 4000)
    assert result['mean_objective'] == pytest.approx(1.)
    tasks['run']['observed']['absolute_mechanical_cot'] = None
    tasks['run']['forward_m'] = -2.
    result = diagnostic_performance({'tasks': tasks}, scoring, 4000)
    assert result['mean_objective'] is None
    assert result['tasks']['run']['task_objective'] is None
    assert result['tasks']['run']['raw']['forward_rmse_mps'] == .2


def test_training_only_and_dynamic_skip_marker(tmp_path):
    assert evaluations_enabled(tmp_path, False)
    assert not evaluations_enabled(tmp_path, True)
    (tmp_path/'SKIP_EVALUATION').touch()
    assert not evaluations_enabled(tmp_path, False)
