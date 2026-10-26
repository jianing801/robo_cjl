"""CPU checks: correct motor grouping, rates, masks and pre-reset capture."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from hybrid_health import motor_health


def fake_env():
    available=torch.tensor([[10.]*6,[20.]*6,[0.]*6,[10.]*6])
    applied=available*.9
    actuators={}
    for name, ids in [('knees',[0,1]),('feet',[2,3,4,5])]:
        actuators[name]=NS(joint_indices=ids,computed_effort=available[:,ids]+5,
            dynamic_effort_limit=available[:,ids],effort_limit=torch.full((4,len(ids)),40.))
    return NS(cfg=NS(spatial_health_designs=[dict(knee_motor='K0',ankle_motor='A0'),
        dict(knee_motor='K1',ankle_motor='A0')]),spatial_groups=torch.tensor([0,1,0,1]),
        reward_manager=NS(_term_names=['tracking','design_motor_torque_limits'],
            _step_reward=torch.tensor([[2.,-1.]]*4)),step_dt=.02,
        robot=NS(actuators=actuators,data=NS(applied_torque=applied,
            root_lin_vel_b=torch.tensor([[1.,0.,0.],[2.,0.,0.],[3.,0.,0.],[4.,0.,0.]]))),
        reset_terminated=torch.tensor([1,0,0,0]),reset_buf=torch.tensor([1,1,0,0]),
        action_buffer=NS(buffer=torch.ones(4,1,6)))


def test_grouping_zero_capability_and_dt_no_reward_change():
    env=fake_env();rewards=torch.tensor([1.,2.,3.,4.]);before=rewards.clone()
    log=motor_health(env,rewards);p='Health/knee/K0/'
    assert log[p+'reward_per_step']==2 and log[p+'forward_velocity_mps']==2
    assert log[p+'termination_fraction']==.5 and log[p+'reset_fraction']==.5
    assert log[p+'unavailable_fraction']==.5
    assert torch.allclose(log[p+'utilization_valid_mean'],torch.tensor(.9))
    assert log[p+'over_80_fraction_valid']==1 and log[p+'saturated_99_fraction_valid']==0
    assert log[p+'requested_excess_nm']==5
    assert torch.allclose(log[p+'soft_reward_per_step'],torch.tensor(-.02))
    assert torch.allclose(log[p+'soft_share_abs_reward_budget'],torch.tensor(1/3))
    assert log['Health/ankle/A0/reward_per_step']==2.5
    assert all(torch.isfinite(v) and not v.requires_grad for v in log.values())
    assert torch.equal(rewards,before)
    # Detached snapshots cannot become reset-state zeros later.
    env.robot.data.applied_torque.zero_();env.robot.data.root_lin_vel_b.zero_()
    assert log[p+'forward_velocity_mps']==2


def test_disabled_penalty_and_new_scene_masks():
    env=fake_env();env.reward_manager._term_names=['tracking']
    env.reward_manager._step_reward=torch.zeros(4,1)
    a=motor_health(env,torch.zeros(4))
    assert a['Health/knee/K0/soft_reward_per_step']==0
    assert a['Health/knee/K0/soft_share_abs_reward_budget']==0
    first=env._motor_health_masks
    motor_health(env,torch.zeros(4));assert env._motor_health_masks is first
    other=fake_env();motor_health(other,torch.zeros(4))
    assert other._motor_health_masks is not first


def test_native_reward_hook_captures_terminal_state_before_reset():
    source=Path(__file__).with_name('hybrid_runtime.py').read_text()
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpatialRoughEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
    class Base:
        def _get_rewards(self):return torch.ones(4)
    scope={'Base':Base}
    node=ast.ClassDef(name='Probe',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual-hook','exec'),scope)
    probe=scope['Probe']();probe.__dict__.update(vars(fake_env()))
    result=probe._get_rewards()
    assert torch.equal(result,torch.ones(4))
    assert probe._motor_health_log['Health/knee/K0/forward_velocity_mps']==2
