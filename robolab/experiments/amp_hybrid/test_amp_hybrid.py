"""CPU-only AMP hybrid tests; no Isaac, GPU, reference generation or long training."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT.parent/'rsl_rl')]
from amp_bank import select_grid
from amp_contract import cycle_groups,stage_spec,validate_checkpoint,validate_config
from amp_layout import identity,balanced_indices,make_layout
from amp_checkpoint import AmpCheckpointMixin
from amp_policy import SpatialActorCritic
from reference_router import ReferenceRouter


def extracted(path,name,namespace):
    node=next(n for n in ast.parse((HERE/path).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(HERE/path),'exec'),namespace)
    return namespace[name]


def full_bank():
    cfg=json.loads((HERE/'config.local900.json').read_text())
    grid=select_grid(cfg['grid'],cfg['grid']['knee_pool'],cfg['grid']['ankle_pool'])
    rows=[dict(d,context=[0.]*24,graph={'joint_names':['j']}) for d in grid['designs']]
    return cfg,dict(configuration=cfg['grid'],designs=rows,context_width=24)


def test_full900_schedule_balance_and_native_amp_budget():
    from collections import Counter
    cfg,bank=full_bank();validate_config(cfg,bank)
    assert cfg['grid']['num_envs']==4096 and len(bank['designs'])==900
    assert cfg['rotation']==dict(active_designs=30,updates_per_stage=50,total_updates=5000,seed=42)
    lookup={d['id']:d for d in bank['designs']}
    for cycle in range(4):
        groups=cycle_groups(bank,cfg,cycle)
        assert len(groups)==30 and len(set(sum(groups,[])))==900
        for ids in groups:
            rows=[lookup[i] for i in ids]
            assert set(Counter(d['thigh_m'] for d in rows).values())=={3}
            assert set(Counter(d['calf_m'] for d in rows).values())=={3}
            assert sorted(Counter((d['knee_motor'],d['ankle_motor']) for d in rows).values())==[3]*6+[4]*3
            layout=make_layout(bank,4096,ids)
            assert set(layout['counts'].values())=={136,137}
    assert stage_spec(bank,cfg,99)['end_update']==5000


def observations():
    return TensorDict(dict(policy=torch.randn(8,28),critic=torch.randn(8,30),
                           disc=torch.randn(8,3,5),disc_demo=torch.randn(8,3,5)),batch_size=[8])


def new_runner(experiment,stage):
    from rsl_rl.storage import RolloutStorage,CircularBuffer
    from rsl_rl.algorithms import PPOAMP
    obs=observations()
    groups=dict(policy=['policy'],critic=['critic'],discriminator=['disc'],discriminator_demonstration=['disc_demo'])
    policy=SpatialActorCritic(obs,groups,2,native_config=dict(role='run',context_width=24,
        spatial_identity=identity(experiment)),actor_hidden_dims=[16],critic_hidden_dims=[16],
        actor_obs_normalization=True,critic_obs_normalization=True,init_noise_std=.5)
    storage=RolloutStorage('rl',8,4,obs,[2],'cpu')
    storage_cls=extracted('amp_mixed_runtime.py','SpatialStorage',
                          dict(RolloutStorage=RolloutStorage,balanced_indices=balanced_indices))
    storage.__class__=storage_cls;storage.spatial_groups=torch.arange(8)%2
    algorithm=extracted('amp_runtime.py','HybridPPOAMP',dict(PPOAMP=PPOAMP,CircularBuffer=CircularBuffer))
    amp=dict(disc_obs_dim=5,disc_obs_steps=3,disc_obs_buffer_size=4,loss_type='LSGAN',
        disc_trunk_weight_decay=.001,disc_linear_weight_decay=.1,disc_learning_rate=1e-4,
        grad_penalty_scale=10.,step_dt=.02,
        amp_discriminator=dict(hidden_dims=[16],activation='elu',style_reward_scale=1.5,task_style_lerp=.6))
    alg=algorithm(policy,storage,amp_cfg=amp,device='cpu',num_learning_epochs=1,num_mini_batches=2,
                  learning_rate=1e-4,schedule='fixed')
    runner=AmpCheckpointMixin()
    runner.alg=alg;runner.device='cpu';runner.hybrid_identity=experiment;runner.hybrid_stage=stage
    runner.current_learning_iteration=0
    return runner


def assert_state_equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_state_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state_equal(x,y)
    else:assert a==b


def learn_update(runner,it):
    alg=runner.alg;obs=observations()
    for _ in range(4):
        with torch.inference_mode():
            alg.act(obs)
            obs=observations()
            alg.process_env_step(obs,torch.randn(8),torch.zeros(8,dtype=torch.long),{})
    with torch.inference_mode():alg.compute_returns(obs)
    losses=alg.update()
    assert all(torch.isfinite(torch.as_tensor(v)) for v in losses.values() if v is not None)
    assert alg.policy.actor[0].weight.grad is not None
    assert alg.amp_discriminator.disc_linear.weight.grad is not None
    runner.current_learning_iteration=it


def test_three_groups_real_cpu_ppo_amp_and_complete_state_continuation(tmp_path):
    experiment=dict(kind='amp_hybrid_test',bank_sha256='fixed')
    previous=None
    for index in range(3):
        stage=dict(index=index,first_update=index*2,end_update=index*2+2)
        runner=new_runner(experiment,stage)
        assert not runner.alg.disc_obs_buffer.current_length.any()
        assert not runner.alg.disc_demo_obs_buffer.current_length.any()
        if previous:
            saved=validate_checkpoint(previous,experiment,stage['first_update'])
            runner.restore_spatial(previous)
            assert_state_equal(runner.alg.policy.state_dict(),saved['model_state_dict'])
            assert_state_equal(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict'])
            assert_state_equal(runner.alg.amp_discriminator.state_dict(),saved['amp_discriminator_state_dict'])
            assert_state_equal(runner.alg.disc_optimizer.state_dict(),saved['amp_discriminator_optimizer_state_dict'])
            assert runner.current_learning_iteration==stage['first_update']
            # No stale previous-morphology motion data are restored.
            assert not runner.alg.disc_obs_buffer.current_length.any()
        before=runner.alg.policy.actor[0].weight.detach().clone()
        before_disc=runner.alg.amp_discriminator.disc_linear.weight.detach().clone()
        for it in range(stage['first_update'],stage['end_update']):learn_update(runner,it)
        assert not torch.equal(before,runner.alg.policy.actor[0].weight)
        assert not torch.equal(before_disc,runner.alg.amp_discriminator.disc_linear.weight)
        previous=tmp_path/f'model_{stage["end_update"]-1}.pt';runner.save(previous)
        saved=validate_checkpoint(previous,experiment,stage['end_update'])
        assert all(v['step'].item()==stage['end_update']*2
                   for v in saved['amp_discriminator_optimizer_state_dict']['state'].values())
    with pytest.raises(ValueError):validate_checkpoint(previous,dict(kind='rough'),6)
    broken=deepcopy(saved);broken.pop('amp_discriminator_optimizer_state_dict')
    path=tmp_path/'missing.pt';torch.save(broken,path)
    with pytest.raises(ValueError,match='missing AMP'):validate_checkpoint(path,experiment,6)
    broken=deepcopy(saved)
    next(iter(broken['amp_discriminator_optimizer_state_dict']['state'].values()))['exp_avg'].fill_(float('nan'))
    path=tmp_path/'nan.pt';torch.save(broken,path)
    with pytest.raises(ValueError,match='nonfinite'):validate_checkpoint(path,experiment,6)


def test_morphology_suffix_never_changes_discriminator_inputs():
    from rsl_rl.storage import RolloutStorage
    cls=extracted('amp_mixed_runtime.py','SpatialVecEnv',dict(RslRlVecEnvWrapper=object,
        TensorDict=TensorDict,torch=torch))
    wrapper=cls.__new__(cls);wrapper.num_envs=8
    wrapper.unwrapped=SimpleNamespace(spatial_context=torch.ones(8,24))
    obs=observations();extended=wrapper._extend(obs)
    assert extended['policy'].shape[-1]==52 and extended['critic'].shape[-1]==54
    assert torch.equal(extended['disc'],obs['disc'])
    assert torch.equal(extended['disc_demo'],obs['disc_demo'])
    assert obs['policy'].shape[-1]==28


class MotionTerm:
    num_dofs=23;num_key_bodies=6
    def __init__(self,label):self.label=label
    def get_num_motions(self):return 8
    def sample_motions(self,n):return torch.arange(n)%8
    def get_motion_durations(self,motion_ids):return motion_ids.float()+self.label*100
    def get_motion_state(self,motion_ids,motion_times):
        return {'dof_pos':torch.full((len(motion_ids),23),float(self.label))}
    def sample_times(self,motion_ids,**kwargs):return torch.zeros(len(motion_ids))


def test_current_group_reference_routing_not_previous_group():
    router=ReferenceRouter([MotionTerm(1),MotionTerm(2)],[0,1,0,1],'cpu')
    ids=router.sample_motions_for_envs(torch.arange(4))
    assert torch.equal(ids,torch.tensor([0,8,1,9]))
    result=router.get_motion_state(ids,torch.zeros(4))['dof_pos']
    assert torch.equal(result[:,0],torch.tensor([1.,2.,1.,2.]))
    with pytest.raises(RuntimeError):router.sample_motions(4)
    with pytest.raises(ValueError):router.get_motion_durations(torch.tensor([16]))


def test_native_agent_parameters_unchanged_and_amp_only():
    from amp_support import extend_agent,source_identity
    source=dict(policy={'actor_hidden_dims':[512,256,128]},
        algorithm=dict(learning_rate=1e-4,symmetry_cfg=dict(use_mirror_loss=True,mirror_loss_coeff=.2),
                       amp_cfg=dict(disc_learning_rate=1e-4,grad_penalty_scale=10)))
    original=deepcopy(source);agent=extend_agent(source,bank_sha='bank',context_width=24)
    assert source==original
    assert agent['algorithm']['amp_cfg']==original['algorithm']['amp_cfg']
    assert agent['algorithm']['symmetry_cfg']['use_mirror_loss'] is True
    assert agent['algorithm']['learning_rate']==1e-4
    assert agent['algorithm']['class_name']=='amp_runtime:HybridPPOAMP'
    sources=source_identity()
    assert any('/manager_based/amp/' in p for p in sources)
    assert not any('backpack' in p or '/experiments/rough_hybrid/' in p for p in sources)
    with pytest.raises(ValueError):source_identity('stairs')


def test_duplicate_bank_protection_without_blocking_other_expert(tmp_path):
    from amp_processes import reject_duplicate_bank_safely
    bank=tmp_path/'bank.json'
    other=SimpleNamespace(pid=1,info=dict(cmdline=['python','train_stage.py','--bank',str(tmp_path/'other.json')],cwd=None))
    same=SimpleNamespace(pid=2,info=dict(cmdline=['python','train_stage.py','--bank=bank.json'],cwd=str(tmp_path)))
    reject_duplicate_bank_safely(bank,snapshots=[other],own_tree=set())
    with pytest.raises(RuntimeError,match='same AMP bank'):
        reject_duplicate_bank_safely(bank,snapshots=[other,same],own_tree=set())
    reject_duplicate_bank_safely(bank,snapshots=[same],own_tree={2})


def test_no_preflight_no_rough_or_backpack_task_router():
    for name in ('train.py','train_stage.py'):
        text=(HERE/name).read_text()
        assert '--preflight-only' not in text and '--preflight-report' not in text
    for file in HERE.glob('*.py'):
        if file.name.startswith('test_'):continue
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ('spatial_task_runtime','spatial_rough_runtime',
                                          'robolab.spatial_expert_tasks','hybrid_contract')


def test_native_key_body_order_and_weight_mixture():
    from amp_references import native_weights,KEY_BODIES
    tree=ast.parse((ROOT/'robolab/tasks/manager_based/amp/rpo_amp_env_cfg.py').read_text())
    bodies=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='KEY_BODY_NAMES' for t in n.targets))
    assert bodies==list(KEY_BODIES) and len(native_weights())==8


def test_native_amp_runner_constructor_resolves_buffers_and_logger(monkeypatch):
    from types import ModuleType
    from rsl_rl.runners import OnPolicyRunner,AMPRunner
    from rsl_rl.modules import resolve_amp_config
    from rsl_rl.algorithms import PPOAMP
    from rsl_rl.storage import RolloutStorage,CircularBuffer
    storage=extracted('amp_mixed_runtime.py','SpatialStorage',
                      dict(RolloutStorage=RolloutStorage,balanced_indices=balanced_indices))
    base=extracted('amp_mixed_runtime.py','SpatialRunner',dict(AmpCheckpointMixin=AmpCheckpointMixin,
                    OnPolicyRunner=OnPolicyRunner,SpatialStorage=storage))
    cls=extracted('amp_runtime.py','SpatialAMPRunner',dict(SpatialRunner=base,
                    AMPRunner=AMPRunner,resolve_amp_config=resolve_amp_config))
    algorithm=extracted('amp_runtime.py','HybridPPOAMP',dict(PPOAMP=PPOAMP,CircularBuffer=CircularBuffer))
    module=ModuleType('amp_runtime');module.HybridPPOAMP=algorithm
    monkeypatch.setitem(sys.modules,'amp_runtime',module)
    env=SimpleNamespace(num_envs=8,num_actions=2,max_episode_length=100,step_dt=.02,
                        spatial_groups=torch.arange(8)%2,cfg={},get_observations=observations)
    env.unwrapped=env;env.env=SimpleNamespace(unwrapped=env)
    agent=dict(obs_groups=dict(policy=['policy'],critic=['critic'],discriminator=['disc'],
                               discriminator_demonstration=['disc_demo']),
        num_steps_per_env=4,save_interval=100,logger='tensorboard',
        policy=dict(class_name='amp_policy:SpatialActorCritic',native_config=dict(role='run',context_width=24),
            actor_hidden_dims=[16],critic_hidden_dims=[16],actor_obs_normalization=True,critic_obs_normalization=True),
        algorithm=dict(class_name='amp_runtime:HybridPPOAMP',rnd_cfg=None,symmetry_cfg=None,
            amp_cfg=dict(disc_obs_buffer_size=4,loss_type='LSGAN',disc_trunk_weight_decay=.001,
                         disc_linear_weight_decay=.1,disc_learning_rate=1e-4,grad_penalty_scale=10.,
                         amp_discriminator=dict(hidden_dims=[16]))))
    runner=cls(env,agent,log_dir=None,device='cpu')
    assert runner.alg.amp_discriminator.disc_obs_dim==5
    assert runner.alg.amp_discriminator.disc_obs_steps==3
    assert runner.alg.disc_obs_buffer.batch_size==8
    assert runner.logger.__class__.__name__=='LoggerAMP'
    assert isinstance(runner,AmpCheckpointMixin)


def test_reference_library_binds_bank_geometry_six_bodies_and_eight_clips(tmp_path):
    import numpy as np
    from amp_references import (sha,kinematic_signature,validate_library,native_weights,ROOT as REFROOT,KEY_BODIES)
    from robolab.amp_reference_serialization import dump_portable
    bank_path=tmp_path/'bank.json';bank_path.write_text('{}')
    urdf=tmp_path/'robot.urdf';urdf.write_text('<robot name="test"><link name="base"/></robot>')
    names=[f'joint{i}' for i in range(23)]
    clips=tmp_path/'clips';clips.mkdir()
    data=dict(fps=30.,root_pos=np.zeros((3,3)),root_rot=np.tile([1.,0.,0.,0.],(3,1)),
              dof_pos=np.zeros((3,23)),key_body_pos=np.zeros((3,6,3)))
    weights=native_weights();hashes={}
    for name in weights:
        path=clips/(name+'.pkl');dump_portable(path,data);hashes[name]=sha(path)
    design=dict(id='a',urdf=str(urdf),urdf_sha256=sha(urdf))
    bank=dict(designs=[design],canonical_semantic_contract=dict(joint_names=names))
    binding=dict(directory=str(clips),clips=hashes,geometry_key=kinematic_signature(urdf),urdf_sha256=sha(urdf))
    sources={name:dict(path=str(REFROOT/'data/motions/rpo_lab'/(name+'.pkl')),
                      sha256=sha(REFROOT/'data/motions/rpo_lab'/(name+'.pkl'))) for name in weights}
    library=dict(kind='spatial_hhtools_library_v1',bank_sha256=sha(bank_path),weights=weights,
                 joint_names=names,key_bodies=list(KEY_BODIES),designs={'a':binding},source_clips=sources)
    manifest=tmp_path/'library.json';manifest.write_text(json.dumps(library))
    assert validate_library(bank,bank_path,manifest,['a'])==library
    with pytest.raises(ValueError,match='subset'):validate_library(bank,bank_path,manifest,['unknown'])
    bad=deepcopy(library);bad['key_bodies'][2]='wrong_elbow';manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='semantics'):validate_library(bank,bank_path,manifest)
    bad=deepcopy(library);bad['designs']['a']['urdf_sha256']='wrong';manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='morphology'):validate_library(bank,bank_path,manifest)
    manifest.write_text(json.dumps(library))
    (clips/(next(iter(weights))+'.pkl')).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='identity'):validate_library(bank,bank_path,manifest)


def test_controller_direct_training_interrupt_resume_and_absolute_budget(tmp_path,monkeypatch):
    from contextlib import nullcontext
    import psutil
    import amp_bank,amp_references
    import robolab.rough_gpu as gpu
    spec=importlib.util.spec_from_file_location('amp_controller_test',HERE/'train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg,bank=full_bank()
    cfg['rotation'].update(active_designs=30,updates_per_stage=2,total_updates=5)
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    bankfile=tmp_path/'bank.json';bankfile.write_text('{}')
    refs=tmp_path/'library.json';refs.write_text('{}');root=tmp_path/'run'
    monkeypatch.setattr(module,'load_configurable_bank',lambda p:bank)
    monkeypatch.setattr(amp_bank,'load_configurable_bank',lambda p:bank)
    monkeypatch.setattr(amp_references,'validate_library',lambda *a,**k:{})
    monkeypatch.setattr(psutil,'process_iter',lambda *a,**k:[])
    monkeypatch.setattr(gpu,'query_gpus',lambda:[])
    monkeypatch.setattr(gpu,'candidates',lambda *a,**k:[{'index':0,'uuid':'test'}])
    monkeypatch.setattr(gpu,'gpu_lease',lambda *a,**k:nullcontext({'index':0,'uuid':'test'}))
    monkeypatch.setattr(gpu,'child_environment',lambda _,environment:dict(environment))
    monkeypatch.setattr(gpu,'reject_duplicate_bank',lambda *a,**k:None)
    calls=[];fail=[True]
    def child(command,folder,env):
        assert '--preflight-only' not in command and '--preflight-report' not in command
        assert command[command.index('--reference-library')+1]==str(refs)
        index=int(command[command.index('--stage-index')+1]);calls.append(index)
        if index==1 and fail[0]:
            folder.mkdir();(folder/'launch.json').write_text('{}');raise RuntimeError('injected interruption')
        out=Path(command[command.index('--output-dir')+1]);out.mkdir(parents=True)
        plan=json.loads((root/'plan.json').read_text());stage=stage_spec(bank,cfg,index)
        runner=new_runner(plan['experiment'],stage)
        if '--resume-checkpoint' in command:
            runner.restore_spatial(command[command.index('--resume-checkpoint')+1])
            assert runner.current_learning_iteration==stage['first_update']
        for it in range(stage['first_update'],stage['end_update']):learn_update(runner,it)
        (out/'models').mkdir();runner.save(out/f'models/model_{stage["end_update"]-1}.pt')
        (out/'summary.json').write_text(json.dumps(dict(result='AMP_HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED',
                                                     completed_updates=stage['end_update'])))
    monkeypatch.setattr(module,'run_child',child)
    args=['--config',str(config),'--bank',str(bankfile),'--reference-library',str(refs),'--output-root',str(root),'--execution-mode','per-group']
    module.main(args);assert calls==[]
    with pytest.raises(RuntimeError,match='injected'):module.main(args+['--resume','--execute'])
    fail[0]=False;module.main(args+['--resume','--execute'])
    assert calls==[0,1,1,2]
    assert json.loads((root/'completed.json').read_text())['completed_updates']==5
    saved=list(calls);module.main(args+['--resume','--execute']);assert calls==saved
    (root/'USER_PAUSED').touch()
    with pytest.raises(RuntimeError,match='USER_PAUSED'):module.main(args+['--resume','--execute'])


def test_persistent_amp_rebind_preserves_learned_state_and_clears_replay():
    from amp_continuity import rebind_group, learned_objects, capture_rng, restore_rng
    from collections import deque
    runner=new_runner(dict(kind='amp_test'),dict(index=0,first_update=0,end_update=2))
    runner.logger=SimpleNamespace(ep_extras=[],rewbuffer=deque(),lenbuffer=deque(),
        total_rewbuffer=deque(),style_rewbuffer=deque())
    with torch.inference_mode():
        for name in ('cur_reward_sum','cur_episode_length','cur_total_reward_sum','cur_style_reward_sum'):
            setattr(runner.logger,name,torch.ones(8))
    objects=learned_objects(runner)
    for index in range(3):
        for it in range(index*2,index*2+2):learn_update(runner,it)
        assert learned_objects(runner)==objects
        states=deepcopy((runner.alg.policy.state_dict(),runner.alg.optimizer.state_dict(),
            runner.alg.amp_discriminator.state_dict(),runner.alg.disc_optimizer.state_dict()))
        buffers=(runner.alg.disc_obs_buffer._buffer,runner.alg.disc_demo_obs_buffer._buffer)
        env=SimpleNamespace(get_observations=observations)
        rebind_group(runner,env,torch.arange(8)%2)
        assert runner.env is env and learned_objects(runner)==objects
        assert buffers[0] is runner.alg.disc_obs_buffer._buffer
        assert buffers[1] is runner.alg.disc_demo_obs_buffer._buffer
        assert not runner.alg.disc_obs_buffer.current_length.any()
        assert not runner.alg.disc_demo_obs_buffer.current_length.any()
        assert_state_equal(states,(runner.alg.policy.state_dict(),runner.alg.optimizer.state_dict(),
            runner.alg.amp_discriminator.state_dict(),runner.alg.disc_optimizer.state_dict()))
        assert not runner.logger.cur_style_reward_sum.any()
    assert all(v['step']==12 for v in runner.alg.disc_optimizer.state.values())
    session={};capture_rng(session);expected=torch.rand(4)
    torch.manual_seed(123);restore_rng(session)
    assert torch.equal(expected,torch.rand(4))
    bad=SimpleNamespace(get_observations=lambda:observations().exclude('disc'))
    with pytest.raises(ValueError,match='shapes'):rebind_group(runner,bad,torch.arange(8))


def test_persistent_group_loop_commit_failure_resume_and_tail(tmp_path,monkeypatch):
    import amp_bank
    from amp_persistent_worker import run_groups
    from amp_continuity import rebind_group
    cfg,bank=full_bank()
    cfg['rotation'].update(updates_per_stage=2,total_updates=5)
    root=tmp_path/'run';root.mkdir()
    (root/'attempts').mkdir();(root/'commits').mkdir()
    bankfile=tmp_path/'bank.json';bankfile.write_text('{}')
    # load_plan is separately identity-tested; isolate orchestration from source/GP IO.
    plan=dict(configuration=cfg,experiment=dict(kind='amp_loop'),bank=str(bankfile),
        reference_library=str(tmp_path/'refs'),execution_protocol='amp_persistent_v1')
    (root/'plan.json').write_text(json.dumps(plan))
    import amp_contract
    monkeypatch.setattr(amp_contract,'load_plan',lambda *a:plan)
    monkeypatch.setattr(amp_bank,'load_configurable_bank',lambda *a:bank)
    calls=[];fail=[True];sessions=[]
    def stage_main(args,session):
        assert sys.argv[1:]==args
        assert Path(sys.argv[0]).name=='train_stage.py'
        index=int(args[args.index('--stage-index')+1]);calls.append(index)
        if index==1 and fail[0]:raise RuntimeError('injected AMP transition')
        stage=stage_spec(bank,cfg,index)
        output=Path(args[args.index('--output-dir')+1]);output.mkdir(parents=True)
        env=SimpleNamespace(get_observations=observations)
        if 'runner' not in session:
            runner=new_runner(plan['experiment'],stage)
            (output/'models').mkdir()
            runner.logger=SimpleNamespace(log_dir=str(output/'models'),ep_extras=[])
            if '--resume-checkpoint' in args:
                runner.restore_spatial(args[args.index('--resume-checkpoint')+1])
            session['runner']=runner;sessions.append(runner)
        else:
            runner=session['runner'];rebind_group(runner,env,torch.arange(8)%2)
        assert runner.current_learning_iteration==stage['first_update']
        runner.hybrid_stage=stage
        for it in range(stage['first_update'],stage['end_update']):learn_update(runner,it)
        runner.save(Path(runner.logger.log_dir)/f'model_{stage["end_update"]-1}.pt')
        (output/'summary.json').write_text(json.dumps(dict(result='AMP_HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED',
            completed_updates=stage['end_update'])))
    with pytest.raises(RuntimeError,match='injected'):
        run_groups(root,bankfile,tmp_path/'first',stage_main=stage_main)
    assert len(list((root/'commits').glob('*.json')))==1
    fail[0]=False
    run_groups(root,bankfile,tmp_path/'second',stage_main=stage_main)
    assert calls==[0,1,1,2] and len(sessions)==2
    assert json.loads((tmp_path/'second/summary.json').read_text())['completed_updates']==5
    assert all(v['step']==10 for v in sessions[-1].alg.disc_optimizer.state.values())


def test_child_rejects_masked_zero_exit(tmp_path):
    import os
    spec=importlib.util.spec_from_file_location('amp_child_test',HERE/'train.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    command=[sys.executable,'-c','pass']
    with pytest.raises(RuntimeError,match='missing summary'):
        controller.run_child(command,tmp_path/'child',os.environ)
    result=json.loads((tmp_path/'child/exit.json').read_text())
    assert result['raw_returncode']==0 and result['returncode']==1
