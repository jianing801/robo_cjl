import pytest
from full_body_torque_calibration import equivalent_sum_weight,mean_utilization_reward

def test_joint_count_does_not_change_global_utilization_budget():
    for u in (.5,.8,.9,1.):
        old=-4*.02*6*max(u-.8,0.)**2
        assert mean_utilization_reward([u]*23)==pytest.approx(old)
        assert equivalent_sum_weight(23)*.02*23*max(u-.8,0.)**2==pytest.approx(old)

def test_partial_saturation_and_dt():
    assert mean_utilization_reward([1.]*6+[.5]*17)==pytest.approx(-24*.02*6/23*.04)
    assert mean_utilization_reward([.9]*23,dt=.04)==pytest.approx(2*mean_utilization_reward([.9]*23))
    assert mean_utilization_reward([.8]*23)==0

def test_reject_implicit_or_empty_joint_count():
    for count in (0,-1,True,23.):
        with pytest.raises(ValueError):equivalent_sum_weight(count)
