"""Cross-group logging must not carry completed physical episodes forward."""
from collections import deque
from types import SimpleNamespace
import torch
from train_stage import reset_group_logger


def test_new_group_does_not_report_old_completed_episodes():
    logger=SimpleNamespace(cur_reward_sum=torch.tensor([2.,3.]),
        cur_episode_length=torch.tensor([20.,30.]),ep_extras=[{'old':1}],
        rewbuffer=deque([-4.]*100,maxlen=100),lenbuffer=deque([30.]*100,maxlen=100))
    reset_group_logger(logger)
    assert not logger.rewbuffer and not logger.lenbuffer and not logger.ep_extras
    assert not logger.cur_reward_sum.any() and not logger.cur_episode_length.any()
    # The first newly completed episode must stand alone, not blend with 99 old ones.
    logger.rewbuffer.append(12.)
    assert sum(logger.rewbuffer)/len(logger.rewbuffer)==12.


def test_inference_allocated_counters_can_be_cleared_at_group_boundary():
    with torch.inference_mode():
        reward=torch.ones(3);length=torch.ones(3)
    logger=SimpleNamespace(cur_reward_sum=reward,cur_episode_length=length,
        ep_extras=[],rewbuffer=deque(),lenbuffer=deque())
    reset_group_logger(logger)
    assert not reward.any() and not length.any()


def test_per_design_failure_frequency_keeps_timeout_and_reward_semantics():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(__file__).with_name('hybrid_runtime.py').read_text())
    wrapper=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpatialVecEnv')
    step=next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    rewards=torch.tensor([1.,2.,3.,4.]);dones=torch.tensor([1,1,0,1])
    obs={'policy':torch.ones(4,2)}
    class Base:
        def step(self,actions):return obs,rewards,dones,{}
    cls=ast.ClassDef(name='DiagnosticWrapper',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[step],decorator_list=[])
    scope={'Base':Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'actual_spatial_step','exec'),scope)
    env=scope['DiagnosticWrapper']()
    env.unwrapped=SimpleNamespace(cfg=SimpleNamespace(spatial_design_ids=['a','b']),
        spatial_groups=torch.tensor([0,0,1,1]),
        robot=SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=torch.zeros(4,3))),
        reset_terminated=torch.tensor([True,False,False,True]))
    env._extend=lambda x:x
    actual_obs,actual_reward,actual_done,extras=env.step(torch.zeros(4,1))
    assert actual_obs is obs and actual_reward is rewards and actual_done is dones
    log=extras['log']
    assert log['Spatial/a/reward']==1.5 and log['Spatial/b/reward']==3.5
    assert log['Spatial/a/termination_fraction']==.5 and log['Spatial/b/termination_fraction']==.5
    assert log['Spatial/a/reset_fraction']==1. and log['Spatial/b/reset_fraction']==.5
