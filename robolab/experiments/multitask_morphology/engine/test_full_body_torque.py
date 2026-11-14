import pytest
import torch
from full_body_torque import full_body_cost,WEIGHT

def test_all_23_joints_share_same_dimensionless_coefficient():
    peak=torch.arange(1,24,dtype=torch.float32).unsqueeze(0)*5
    cost,util,missing=full_body_cost(.9*peak,peak,peak)
    torch.testing.assert_close(cost,torch.tensor([23*.1**2]))
    assert not missing.any() and WEIGHT==-12

def test_zero_limit_is_excluded_and_reported_without_nan():
    peak=torch.ones(1,23)*40;available=peak.clone();available[0,0]=0
    applied=available*.9;cost,util,missing=full_body_cost(applied,available,peak)
    assert missing.sum()==1 and torch.isfinite(util).all()
    torch.testing.assert_close(cost,torch.tensor([22*.1**2]))

def test_reject_wrong_coverage():
    with pytest.raises(ValueError):full_body_cost(torch.ones(1,22),torch.ones(1,22),torch.ones(1,22))

def test_dynamic_speed_capacity_changes_cost_even_at_same_peak():
    peak=torch.ones(1,23)*40
    applied=torch.zeros_like(peak);applied[:,0]=18
    fast=peak.clone();fast[:,0]=20
    slow_cost,_,_=full_body_cost(applied,peak,peak)
    fast_cost,_,_=full_body_cost(applied,fast,peak)
    assert slow_cost.item()==0
    torch.testing.assert_close(fast_cost,torch.tensor([.01]))

def test_fixed_groups_export_limit_too(monkeypatch):
    import full_body_torque as module
    from types import SimpleNamespace as NS
    class Motor:
        def _clip_effort(self,effort):return effort.clamp(-3,3)
    robot=NS(actuators={name:NS(class_type=Motor) for name in module.GROUP_MOTORS})
    monkeypatch.setattr(module,'install_fixed_motor_curves',lambda *args:{})
    monkeypatch.setattr(module,'install_full_body_reward',lambda cfg:None)
    module.activate_for_config(NS(scene=NS(robot=robot)),None,1)
    for _ in module.GROUP_MOTORS:
        motor=Motor()
        torch.testing.assert_close(motor._clip_effort(torch.tensor([[8.,-9.]])),torch.tensor([[3.,-3.]]))
        torch.testing.assert_close(motor.dynamic_effort_limit,torch.tensor([[3.,3.]]))

def test_source_identity_binds_real_candidate_curves():
    from full_body_torque import source_identity
    sources=source_identity()
    assert 'robolab/assets/rpo_motor_catalog.py' in sources
    assert sum(name.endswith('.csv') for name in sources)>=12
    assert all(len(value)==64 for value in sources.values())

def test_subclass_override_does_not_inherit_false_export_marker(monkeypatch):
    import full_body_torque as module
    from types import SimpleNamespace as NS
    class Base:
        def _clip_effort(self,effort):return effort.clamp(-3,3)
    class Sub(Base):
        def _clip_effort(self,effort):return effort.clamp(-2,2)
    monkeypatch.setattr(module,'install_fixed_motor_curves',lambda *args:{})
    monkeypatch.setattr(module,'install_full_body_reward',lambda cfg:None)
    for cls in (Base,Sub):
        cfg=NS(scene=NS(robot=NS(actuators={group:NS(class_type=cls) for group in module.GROUP_MOTORS})))
        module.activate_for_config(cfg,None,1)
    actuator=Sub();actuator._clip_effort(torch.tensor([[10.]]))
    torch.testing.assert_close(actuator.dynamic_effort_limit,torch.tensor([[2.]]))

def test_invalid_physics_is_not_silently_converted_to_zero_penalty():
    peak=torch.ones(2,23)*40;available=peak.clone();applied=peak*.9
    available[0,0]=float('nan');applied[1,1]=float('nan')
    cost,_,_=full_body_cost(applied,available,peak)
    assert torch.isnan(cost).all()

def test_full_pool_configs_share_exact_grid_and_disable_old_penalty():
    import json
    from pathlib import Path
    from hybrid_bank import select_grid
    here=Path(__file__).resolve().parent
    rough=json.loads((here.parent/'configs/rough.json').read_text())
    amp=json.loads((here.parent/'configs/amp.json').read_text())
    assert rough['grid']==amp['grid']
    for cfg in (rough,amp):
        assert cfg['full_body_torque'] is True and 'torque_soft_constraint' not in cfg
        grid=cfg['grid'];assert not any('4310' in m for m in grid['ankle_pool'])
        selected=select_grid(grid,grid['knee_pool'],grid['ankle_pool'])
        assert selected['design_count']==3000 and selected['num_envs']==4096
