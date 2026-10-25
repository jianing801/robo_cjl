"""CPU checks of the exact Parkour cost, native dt scaling and new motor pool."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from hybrid_torque import design_motor_torque_limits, install_constraint, squared_torque_excess, validate_settings
from hybrid_contract import validate_config, experiment_identity
from test_hybrid import bank, config

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def fake_env():
    data=SimpleNamespace(applied_torque=torch.tensor([[9.,11.,8.,10.,12.,20.,999.],
                                                     [9.,11.,8.,10.,12.,20.,999.]]))
    actuators=dict(knees=SimpleNamespace(joint_indices=[0,1],effort_limit=torch.tensor([[10.,10.],[20.,20.]])),
                   feet=SimpleNamespace(joint_indices=[2,3,4,5],effort_limit=torch.tensor([[10.]*4,[40.]*4])))
    return SimpleNamespace(robot=SimpleNamespace(data=data,actuators=actuators),num_envs=2,device='cpu')


def test_threshold_absolute_sign_and_sum():
    applied=torch.tensor([[7.,8.,-9.],[0.,0.,0.]])
    peak=torch.full_like(applied,10.)
    assert torch.equal(squared_torque_excess(applied,peak),torch.tensor([1.,0.]))


def test_matches_actual_parkour_formula():
    path=ROOT/'robolab/tasks/manager_based/parkour/mdp/rewards.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='applied_torque_limits_by_ratio')
    node.returns=None
    for arg in node.args.args:arg.annotation=None
    node.args.defaults=[ast.Constant(None),ast.Constant(.8)]
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    torch.manual_seed(42)
    torque=torch.randn(16,6)*100
    peak=torch.rand(16,6)*100+1
    asset=SimpleNamespace(data=SimpleNamespace(applied_torque=torque,joint_effort_limits=peak))
    env=SimpleNamespace(scene={'robot':asset})
    ref=scope['applied_torque_limits_by_ratio'](env,SimpleNamespace(name='robot',joint_ids=slice(None)),.8)
    assert torch.equal(ref,squared_torque_excess(torque,peak,.8))


def test_live_limits_are_per_environment_and_other_joints_ignored():
    env=fake_env()
    cost=design_motor_torque_limits(env)
    assert torch.equal(cost,torch.tensor([174.,0.]))
    assert env._design_torque_soft_limits[1][0,0]==10
    assert env._design_torque_soft_limits[1][1,0]==20
    old=env._design_torque_soft_limits
    env.robot.data.applied_torque.zero_()
    assert not design_motor_torque_limits(env).any()
    assert env._design_torque_soft_limits is old
    # New physical group gets its own table; no old-group cache is reused.
    other=fake_env()
    other.robot.actuators['feet'].effort_limit.fill_(100)
    assert torch.equal(design_motor_torque_limits(other),torch.tensor([10.,0.]))


def test_native_reward_manager_applies_weight_and_dt_once():
    path=Path('C:/Roboparty/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py')
    if not path.exists():pytest.skip('local Isaac Lab source unavailable')
    tree=ast.parse(path.read_text(encoding='utf8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RewardManager')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute')
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),scope)
    term=SimpleNamespace(func=design_motor_torque_limits,weight=-.01,params={'limit_ratio':.8})
    manager=SimpleNamespace(_env=fake_env(),_reward_buf=torch.zeros(2),_term_names=['torque'],
        _term_cfgs=[term],_episode_sums={'torque':torch.zeros(2)},_step_reward=torch.zeros(2,1))
    assert torch.allclose(scope['compute'](manager,.02),torch.tensor([-.0348,0.]))
    assert torch.equal(manager._episode_sums['torque'],manager._reward_buf)
    assert torch.allclose(manager._step_reward[:,0],torch.tensor([-1.74,0.]))


@pytest.mark.parametrize('value',[{'weight':.1,'limit_ratio':.8},{'weight':-.01,'limit_ratio':0},
    {'weight':float('nan'),'limit_ratio':.8},{'weight':-.01,'limit_ratio':1.1},
    {'weight':-.01,'limit_ratio':.8,'other':1}])
def test_invalid_settings_rejected(value):
    with pytest.raises(ValueError):validate_settings(value)


def test_zero_or_missing_term_does_not_touch_source_rewards():
    cfg=SimpleNamespace(reward=SimpleNamespace(original=object()))
    before=dict(vars(cfg.reward))
    assert not install_constraint(cfg,None)['enabled']
    assert not install_constraint(cfg,dict(weight=0,limit_ratio=.8))['enabled']
    assert vars(cfg.reward)==before


def test_installs_only_one_native_reward_term(monkeypatch):
    monkeypatch.setitem(sys.modules,'isaaclab.managers',SimpleNamespace(RewardTermCfg=SimpleNamespace))
    original=object();cfg=SimpleNamespace(reward=SimpleNamespace(original=original))
    audit=install_constraint(cfg,dict(weight=-.01,limit_ratio=.8))
    assert set(vars(cfg.reward))=={'original','design_motor_torque_limits'}
    assert cfg.reward.original is original
    assert cfg.reward.design_motor_torque_limits.weight==-.01 and audit['joint_count']==6


def test_new_pool_and_frozen_settings_identity(tmp_path):
    c=json.loads((HERE/'config.local900.json').read_text())
    assert c['grid']['ankle_pool']==['DM-J4340P-2EC','DM-J6216P-2EC','RS06']
    assert c['torque_soft_constraint']==dict(weight=-.01,limit_ratio=.8)
    b=bank();base=config(b);validate_config(base,b)
    base['torque_soft_constraint']=c['torque_soft_constraint'];validate_config(base,b)
    bp=tmp_path/'bank.json';bp.write_text(json.dumps(b))
    first=experiment_identity(bp,base)
    changed=deepcopy(base);changed['torque_soft_constraint']['weight']=0
    assert first!=experiment_identity(bp,changed)
    assert 'hybrid_torque.py' in first['hybrid_sources']


def test_persistent_no_preflight_and_unchanged_action_path():
    text=(HERE/'train.py').read_text()
    assert "default='persistent'" in text and 'preflight_updates_per_new_layout=0' in text
    assert '--preflight-only' not in text
    node=next(n for n in ast.parse((HERE/'hybrid_runtime.py').read_text()).body
              if isinstance(n,ast.ClassDef) and n.name=='SpatialRoughEnv')
    assert not any(isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step' for n in node.body)
