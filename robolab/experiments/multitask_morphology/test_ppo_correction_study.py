import csv
from ppo_correction_study import curve_assessment


def make_curve(decline=False):
    return [dict(update=i,mean_reward=30,
                 stairs_mean_reward=34+min(i/200,1)*3,
                 backpack_mean_reward=30,
                 run_mean_reward=29-(7*(i-150)/850 if decline and i>150 else 0))
            for i in range(1,1001)]


def test_all_task_gate_rejects_cancellation_even_when_total_is_flat():
    prior=make_curve(False)
    assert curve_assessment(prior,prior)['passed']
    result=curve_assessment(make_curve(True),prior)
    assert not result['passed'] and not result['tasks']['run']['passed']
    assert result['tasks']['stairs']['passed'] and result['tasks']['backpack']['passed']
