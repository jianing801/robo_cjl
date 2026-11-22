"""Real CPU learning/IPC, synthetic environments; not Isaac gait evidence."""
from contextlib import nullcontext
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from contract import schema_for, structure_key, sha, write_new, load_plan
from finetune_state import checkpoint_history, prune_mixed_checkpoints, select_candidate, save_checkpoint, load_checkpoint, validate_batch
import joint_finetune as joint
import pipeline
from ppo_channel import validate_request
from student import SharedMLPActor, save_student, load_student
from test_shared import banks
from test_pipeline_protocol import row


def fixture_plan(tmp_path, designs=None, *, full_body_torque=False):
    native = banks(); native['run'] = deepcopy(native['stairs'])
    keys = designs or [dict(thigh_m=.25, calf_m=.3, knee_motor='K1', ankle_motor='RS06'),
                       dict(thigh_m=.4, calf_m=.45, knee_motor='K2', ankle_motor='RS06')]
    teachers = {}
    for task, bank in native.items():
        bank['designs'] = [dict(d, id=f'{task}_{i}') for i, d in enumerate(keys)]
        path = tmp_path/f'{task}.json'; write_new(path, bank)
        checkpoint = tmp_path/f'{task}.pt'; checkpoint.write_bytes(b'frozen teacher')
        teachers[task] = dict(bank=str(path), bank_sha256=sha(path), checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint), design_ids=[d['id'] for d in bank['designs']],
            copies=8, full_body_torque=full_body_torque)
    schema = schema_for(native)
    schema['shared_structure_keys'] = [list(structure_key(d)) for d in keys]
    shared = tmp_path/'shared.json'
    write_new(shared, dict(structures=[dict(key=key, design_ids={t:f'{t}_{i}' for t in native})
        for i, key in enumerate(schema['shared_structure_keys'])]))
    plan = dict(kind='rough_backpack_amp_distillation_plan_v3', structure_scope='teacher_intersection_only',
        shared_bank=str(shared), shared_bank_sha256=sha(shared), teachers=teachers, schema=schema,
        task_ids={t:i for i,t in enumerate(native)}, selected_structure_keys=schema['shared_structure_keys'], scope='all')
    path = tmp_path/'plan.json'; write_new(path, plan)
    return path, load_plan(path)


def learning_state(schema):
    model = SharedMLPActor(schema, dict(hidden_dims=[8]))
    critic = torch.nn.Sequential(torch.nn.Linear(model.width, 8), torch.nn.ELU(), torch.nn.Linear(8, 1))
    optimizer = torch.optim.Adam([{'params':model.parameters(), 'lr':3e-6},
                                  {'params':critic.parameters(), 'lr':1e-3}])
    return model, critic, optimizer


def batches_for(model):
    x = torch.randn(24, 8, model.width)*.1
    with torch.no_grad():
        distribution = torch.distributions.Normal(model(x.flatten(0,1)).reshape(24,8,23), joint.ACTION_STD)
        actions = distribution.sample()
        logp = distribution.log_prob(actions).sum(-1)
    return [dict(obs=x, actions=actions, logp=logp, reward=torch.ones(24,8), environment_reward=torch.ones(24,8),
                 done=torch.zeros(24,8,dtype=torch.bool), timeout=torch.zeros(24,8,dtype=torch.bool),
                 next_obs=x[-1])] * 3


def test_candidate_maps_formal_2500_grid_exactly(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(joint.HERE.parent/'engine'))
    from hybrid_bank import select_grid
    config=json.loads((joint.HERE.parent/'configs/rough.json').read_text(encoding='utf8'))['grid']
    grid=select_grid(config,config['knee_pool'],config['ankle_pool'])
    assert grid['design_count']==2500 and grid['context_width']==28
    _, plan = fixture_plan(tmp_path, grid['designs'])
    before = deepcopy(plan)
    selected, candidate = select_candidate(plan,design_id='backpack_1234',design_task='backpack')
    assert candidate['design_ids']=={t:f'{t}_1234' for t in plan['teachers']}
    assert selected['schema']==plan['schema'] and plan==before
    assert all(s['design_ids']==[f'{t}_1234'] for t,s in selected['teachers'].items())
    assert select_candidate(plan,design_key=candidate['structure_key'])[1]==candidate
    with pytest.raises(ValueError,match='outside'):
        select_candidate(plan,design_key=[.25,.3,config['knee_pool'][0],config['ankle_pool'][0]])
    with pytest.raises(ValueError,match='Candidate ID'):
        select_candidate(plan,design_id='missing')
    broken=deepcopy(plan); broken['teachers']['run']['design_ids'].remove('run_1234')
    with pytest.raises(ValueError,match='exactly once'):
        select_candidate(broken,design_id='stairs_1234')


def test_learning_resume_restores_actor_critic_adam_and_all_coordinator_rng(tmp_path):
    _, plan=fixture_plan(tmp_path)
    torch.set_num_threads(2); torch.manual_seed(4); random.seed(4); np.random.seed(4)
    model,critic,optimizer=learning_state(plan['schema']); data=batches_for(model)
    save_checkpoint(tmp_path,model,critic,optimizer,0,'identity')
    joint.ppo_update(model,critic,optimizer,data)
    save_checkpoint(tmp_path,model,critic,optimizer,1,'identity')
    expected_rng=(random.random(),np.random.rand(),torch.rand(3))
    joint.ppo_update(model,critic,optimizer,data)
    resumed,c2,o2=learning_state(plan['schema'])
    assert load_checkpoint(tmp_path,resumed,c2,o2,'identity')==1
    assert random.random()==expected_rng[0] and np.random.rand()==expected_rng[1]
    assert torch.equal(torch.rand(3),expected_rng[2])
    joint.ppo_update(resumed,c2,o2,data)
    for left,right in zip([*model.parameters(),*critic.parameters()],[*resumed.parameters(),*c2.parameters()]):
        assert torch.equal(left,right)
    for a,b in zip(optimizer.state.values(),o2.state.values()):
        for key in a: assert torch.equal(a[key],b[key])
    # An uncommitted tensor file cannot change the resume point.
    (tmp_path/'checkpoints/update_000002_orphan.pt').write_bytes(b'partial')
    assert load_checkpoint(tmp_path,resumed,c2,o2,'identity')==1
    with pytest.raises(ValueError,match='another run'): load_checkpoint(tmp_path,resumed,c2,o2,'other')
    record=json.loads((tmp_path/'checkpoints/update_000001.json').read_text())
    (tmp_path/'checkpoints'/record['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'): load_checkpoint(tmp_path,resumed,c2,o2,'identity')


def test_mixed_compaction_keeps_receipts_and_the_resume_checkpoint(tmp_path):
    _, plan = fixture_plan(tmp_path)
    model, critic, optimizer = learning_state(plan['schema'])
    root = tmp_path/'candidate'
    root.mkdir()
    for update in range(5):
        save_checkpoint(root, model, critic, optimizer, update, 'mixed')
    history = checkpoint_history(root, 'mixed')
    prune_mixed_checkpoints(root, history, reconcile=True)
    assert len(checkpoint_history(root, 'mixed')) == 5
    assert {path.name for path in (root/'checkpoints').glob('*.pt')} == {
        history[i]['path'] for i in (0, 3, 4)}
    assert load_checkpoint(root, model, critic, optimizer, 'mixed') == 4


@pytest.mark.parametrize('field,value', [('session_id','stale'),('index',3),('steps',0),('std',float('nan')),
    ('student_sha256','changed'),('manifest_sha256','wrong'),('collection_protocol',{})])
def test_request_rejects_stale_or_corrupt_identity(tmp_path,field,value):
    actor=tmp_path/'actor'; actor.write_bytes(b'actor')
    session=dict(session_id='session',manifest_sha256='manifest',steps_per_task=24,std=.05)
    spec=dict(kind='shared_ppo_request_v2',session_id='session',manifest_sha256='manifest',index=1,
        steps=24,std=.05,student=str(actor),student_sha256=sha(actor),collection_protocol={'profile':'robust'})
    validate_request(spec,session,1,spec['collection_protocol'])
    spec[field]=value
    with pytest.raises(ValueError): validate_request(spec,session,1,{'profile':'robust'})


@pytest.mark.parametrize('corruption', ['stale','nan','shape','mask','timeout','native_nan','missing_native'])
def test_batch_rejects_corruption(tmp_path,corruption):
    _,plan=fixture_plan(tmp_path); model,_,_=learning_state(plan['schema'])
    data=deepcopy(batches_for(model)[0])
    session=dict(session_id='new',manifest_sha256='manifest',collection_protocol={},steps_per_task=24)
    data.update(task='stairs',index=1,session_id='new',manifest_sha256='manifest',collection_protocol={},student_sha256='actor')
    if corruption=='stale': data['session_id']='old'
    elif corruption=='nan': data['reward'][0,0]=float('nan')
    elif corruption=='shape': data['obs']=data['obs'][:-1]
    elif corruption=='mask': data['done']=data['done'].long()
    elif corruption=='native_nan': data['environment_reward'][0,0]=float('nan')
    elif corruption=='missing_native': del data['environment_reward']
    else: data['timeout'][0,0]=True
    with pytest.raises(ValueError):
        validate_batch(data,task='stairs',index=1,session=session,actor_sha256='actor',width=model.width)


@pytest.mark.parametrize('compact', [False, True], ids=['full-retention', 'compact'])
def test_real_ipc_pause_resume_then_independent_evaluation(tmp_path, monkeypatch, compact):
    plan_path,plan=fixture_plan(tmp_path, full_body_torque=compact)
    model,_,_=learning_state(plan['schema']); initial=tmp_path/'actor.pt'; save_student(initial,model)
    output=tmp_path/'candidate'
    gpu=ModuleType('robolab.rough_gpu')
    gpu.query_gpus=lambda:[]; gpu.candidates=lambda *args,**kw:[]
    gpu.gpu_lease=lambda *args:nullcontext('cpu'); gpu.child_environment=lambda *args:dict(os.environ)
    monkeypatch.setitem(sys.modules,'robolab.rough_gpu',gpu)
    # Exercise real serve(), atomic requests and tensors across OS processes.
    stub=tmp_path/'sampler.py'
    stub.write_text('''import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,sys.argv[1])
from ppo_channel import serve
from student import load_student
torch.set_num_threads(1)
channel,task,actor,*options=sys.argv[2:]
model,_=load_student(actor)
obs=torch.zeros(8,model.width)
raw=SimpleNamespace(device='cpu',episode_length_buf=torch.zeros(8,dtype=torch.long),max_episode_length=1000)
if '--full-body-torque' in options:
 raw._full_body_torque_diagnostics=dict(cost=torch.full((8,),.25),
     hard_clip_excess_nm=torch.zeros(8,23),unavailable=torch.zeros(8,23),
     utilization=torch.ones(8,23))
 raw.reward_manager=SimpleNamespace(_term_names=['full_body_torque_limits'],
     _term_cfgs=[SimpleNamespace(weight=-12.)],_step_reward=torch.full((8,1),-3.))
 raw.step_dt=.02
 raw.robot=SimpleNamespace(actuators={})
class Env:
 steps=0
 def step(self,action):
  self.steps+=1
  done=torch.zeros(8,dtype=torch.int64)
  timeout=torch.zeros(8,dtype=torch.bool)
  if self.steps%30==0:
   done[:2]=1
   timeout[0]=True
  reward=float({'stairs':1,'backpack':2,'run':3}[task])
  return obs,torch.full((8,),reward),done,{'time_outs':timeout}
session=json.loads((Path(channel)/'session.json').read_text())
adapter=(lambda obs,reward:reward.mul_(2)) if task=='run' else None
serve(channel,task,Env(),raw,obs,lambda x,reset:x,None,adapter,protocol=session['collection_protocol'])
''')
    original_popen=joint.subprocess.Popen; launches=[]
    def popen(command,**kwargs):
        channel=command[command.index('--ppo-channel')+1]
        task=command[command.index('--task')+1] if '--task' in command else 'run'
        launches.append((channel,task))
        torque_options=['--full-body-torque'] if '--full-body-torque' in command else []
        return original_popen([sys.executable,str(stub),str(joint.HERE),channel,task,str(initial),*torque_options],**kwargs)
    monkeypatch.setattr(joint.subprocess,'Popen',popen)
    original_sleep=joint.time.sleep
    monkeypatch.setattr(joint.time,'sleep',lambda seconds:None if seconds==3 else original_sleep(seconds))
    original_save=joint.save_checkpoint
    paused=[]
    def checkpoint(*args,**kwargs):
        result=original_save(*args,**kwargs)
        if args[4]==1 and not paused:
            paused.append(True)
            if compact:
                # Interrupt after the receipt is committed, just before reclamation.
                raise KeyboardInterrupt('checkpoint committed before compaction')
            (output/'USER_PAUSED').touch()
        return result
    monkeypatch.setattr(joint,'save_checkpoint',checkpoint)
    evaluations=[]; evaluation_paused=[]
    def evaluate(command,*args):
        if len(evaluations)==1 and not evaluation_paused:
            evaluation_paused.append(True)
            raise KeyboardInterrupt('pause after a committed final evaluation')
        folder=Path(command[command.index('--output')+1]); folder.mkdir(parents=True)
        task=command[command.index('--task')+1] if '--task' in command else 'run'
        profile=command[command.index('--profile')+1]; seed=int(command[command.index('--seed')+1])
        actor=command[command.index('--student')+1]; design=command[command.index('--design-ids')+1]
        evaluations.append((task,profile,seed))
        rows=[row(i,design=design) for i in range(8)]
        for record in rows:
            record['mean_horizontal_forward_speed_m_s']=(
                record['command_aligned_distance_m']/record['moving_tracking_seconds'])
        write_new(folder/'summary.json',dict(task=task,teacher=plan['teachers'][task]['checkpoint_sha256'],
            student=sha(actor),design_ids=[design],steps=1000,rows=rows,
            velocity_frame=pipeline.EVALUATION_VELOCITY_FRAME,
            collection_protocol=pipeline.protocol_metadata(profile,1. if profile=='robust' else 0.)))
    monkeypatch.setattr(pipeline,'child',evaluate)
    updates=4 if compact else 2
    args=['--plan',str(plan_path),'--student',str(initial),'--output',str(output),
          '--design-id','stairs_1','--updates',str(updates),'--timeout-seconds','60']
    if compact:
        args.extend(['--compact-training-artifacts','--full-body-torque'])
    with pytest.raises(KeyboardInterrupt): joint.main(args)
    assert (output/'checkpoints/update_000001.json').exists() and not (output/'completed.json').exists()
    saved_reward=json.loads((output/'checkpoints/update_000001.json').read_text())['training_reward']
    assert saved_reward['tasks']['run']['mean_environment_step_reward']==3.
    assert saved_reward['tasks']['run']['mean_ppo_step_reward']==6.
    if compact:
        first_attempt=next((output/'attempts').iterdir())
        assert {p.name for p in first_attempt.glob('*.pt')}=={
            'actor_000.pt','stairs_000.pt','backpack_000.pt','run_000.pt'}
        # A payload without a corresponding committed receipt is not consumed.
        (first_attempt/'actor_001.pt').write_bytes(b'unconsumed')
    with pytest.raises(RuntimeError,match='resume-paused'): joint.main(args+['--resume'])
    with pytest.raises(ValueError,match='changed'): joint.main(args+['--resume-paused','--updates','5'])
    assert (output/'USER_PAUSED').exists()
    with pytest.raises(KeyboardInterrupt): joint.main(args+['--resume-paused'])
    assert (output/f'checkpoints/update_{updates:06d}.json').exists()
    assert (output/'shared_actor_finetuned.pt').exists() and not (output/'completed.json').exists()
    reward_csv=output/'training_reward.csv'
    reward_csv.unlink()  # Restore the derived view from checkpoint receipts on resume.
    mean_csv = output/'mean_reward.csv'
    expected_mean_rows = mean_csv.read_bytes()
    mean_csv.unlink()
    joint.main(args+['--resume-paused'])
    assert mean_csv.read_bytes() == expected_mean_rows
    with mean_csv.open(newline='', encoding='utf8') as stream:
        mean_rows = list(csv.DictReader(stream))
    assert len(mean_rows) == updates and set(mean_rows[0]) >= {
        'mean_reward', 'stairs_mean_reward', 'backpack_mean_reward', 'run_mean_reward'}
    assert not (output/'USER_PAUSED').exists() and len(launches)==6
    sessions=[json.loads(p.read_text()) for p in (output/'attempts').glob('*/session.json')]
    assert all(s['init_at_random_ep_len'] is True for s in sessions)
    assert sorted(s['start_update'] for s in sessions)==[0,1]
    resumed_session=next(s for s in sessions if s['start_update']==1)
    resumed_folder=output/'attempts'/resumed_session['session_id']
    assert not (resumed_folder/'request_000.json').exists() and (resumed_folder/'request_001.json').exists()
    actor,meta=load_student(output/'shared_actor_finetuned.pt')
    assert meta['ppo_updates']==updates and meta['critic_trained'] and 'PPO_updates' not in meta
    manifest=json.loads((output/'manifest.json').read_text())
    if compact:
        assert manifest['compact_training_artifacts'] is True
    else:
        assert 'compact_training_artifacts' not in manifest
    receipts=[json.loads(p.read_text()) for p in sorted((output/'checkpoints').glob('update_*.json'))]
    assert [r['completed_updates'] for r in receipts]==list(range(updates+1))
    assert 'training_reward' not in receipts[0]
    with reward_csv.open(newline='', encoding='utf8') as stream:
        rewards={(int(r['update']),r['task']):r for r in csv.DictReader(stream)}
    assert set(rewards)=={(i,task) for i in range(1,updates+1) for task in plan['teachers']}
    retained={r['completed_updates'] for r in receipts if (output/'checkpoints'/r['path']).is_file()}
    assert retained==({0,3,4} if compact else {0,1,2})
    for index in range(updates):
        folder=next(p.parent for p in (output/'attempts').glob(f'*/request_{index:03d}.json'))
        blobs=[folder/f'{name}_{index:03d}.pt' for name in ('actor','stairs','backpack','run')]
        assert all(p.exists() is (not compact) for p in blobs)
        assert all((folder/f'{task}.stdout.log').exists() for task in plan['teachers'])
        reward_receipt=receipts[index+1]['training_reward']
        assert reward_receipt['session_id']==folder.name
        for task, native_reward in {'stairs':1.,'backpack':2.,'run':3.}.items():
            ppo_reward=native_reward*2 if task=='run' else native_reward
            task_reward=reward_receipt['tasks'][task]
            completed_episodes=2 if index>=2 else 0
            assert task_reward==dict(
                reward_samples=192, mean_environment_step_reward=native_reward, mean_ppo_step_reward=ppo_reward,
                completed_episodes=completed_episodes, timeout_episodes=completed_episodes//2,
                terminated_episodes=completed_episodes//2,
                mean_environment_episode_return=native_reward*30 if completed_episodes else None,
                mean_ppo_episode_return=ppo_reward*30 if completed_episodes else None)
            recorded=rewards[index+1,task]
            assert recorded['session_id']==folder.name
            assert all(recorded[field]==('' if value is None else str(value))
                       for field,value in task_reward.items())
        if compact:
            assert receipts[index+1]['rollout']['session_id']==folder.name
            audits=receipts[index+1]['rollout']['task_audits']
            assert set(audits)==set(plan['teachers'])
            for audit in audits.values():
                assert audit['dt_verified'] and audit['joint_count']==23 and len(audit['steps'])==24
                assert audit['steps'][0]['mean_penalty']==pytest.approx(-.06)
            assert receipts[index+1]['metrics']['samples']==576
    if compact:
        assert (first_attempt/'actor_001.pt').read_bytes()==b'unconsumed'
    assert any(not torch.equal(v,actor.state_dict()[k]) for k,v in model.state_dict().items())
    result=json.loads((output/'candidate_result.json').read_text())
    assert result['candidate']['design_ids']=={t:f'{t}_1' for t in plan['teachers']}
    assert result['scalar_objective'] is None and result['requires_bo_objective_definition']
    assert len(evaluations)==6 and {s for _,_,s in evaluations}=={20261118}
    for profile in ('nominal','robust'):
        assert set(result['profiles'][profile]['task_metrics'])==set(plan['teachers'])
        for evaluation in result['profiles'][profile]['evaluations']:
            summary=json.loads(Path(evaluation['summary']).read_text())
            assert summary['velocity_frame']==pipeline.EVALUATION_VELOCITY_FRAME
            assert summary['student']==sha(output/'shared_actor_finetuned.pt')
    joint.main(args+['--resume'])
    assert len(launches)==6 and len(evaluations)==6
    if compact:
        joint.main(args+['--resume', '--training-only'])
        receipt=json.loads((output/'training_completed.json').read_text(encoding='utf8'))
        assert receipt['ppo_updates']==updates and receipt['student_sha256']==sha(output/'shared_actor_finetuned.pt')
        assert len(launches)==6 and len(evaluations)==6
    (output/'candidate_result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'): joint.main(args+['--resume'])


def test_training_only_writes_checkpoint_receipt_without_diagnostics(tmp_path, monkeypatch):
    plan_path, plan = fixture_plan(tmp_path)
    model, _, _ = learning_state(plan['schema'])
    initial, output = tmp_path/'actor.pt', tmp_path/'candidate'
    save_student(initial, model)
    gpu = ModuleType('robolab.rough_gpu')
    gpu.query_gpus = lambda: []
    gpu.candidates = lambda *args, **kwargs: []
    gpu.gpu_lease = lambda *args: nullcontext('cpu')
    gpu.child_environment = lambda *args: dict(os.environ)
    monkeypatch.setitem(sys.modules, 'robolab.rough_gpu', gpu)
    monkeypatch.setattr(joint, 'evaluate_candidate',
                        lambda *args: pytest.fail('Standalone gait diagnostics ran for BO'))
    runs = []

    def train(a, root, plan, actor, critic, optimizer, completed, digest, protocol, env):
        runs.append(completed)
        with torch.no_grad():
            next(actor.parameters()).add_(.125)
        save_checkpoint(root, actor, critic, optimizer, a.updates, digest)

    monkeypatch.setattr(joint, 'train_updates', train)
    args = ['--plan', str(plan_path), '--student', str(initial), '--output', str(output),
            '--design-id', 'stairs_1', '--updates', '1', '--training-only']
    joint.main(args)
    actor_path = output/'shared_actor_finetuned.pt'
    commit = output/'checkpoints/update_000001.json'
    manifest = output/'manifest.json'
    receipt = output/'training_completed.json'
    record = dict(result='CANDIDATE_FINETUNED', candidate=select_candidate(plan, design_id='stairs_1')[1],
                  ppo_updates=1, manifest_sha256=sha(manifest), student=str(actor_path),
                  student_sha256=sha(actor_path), checkpoint_sha256=sha(commit))
    assert json.loads(receipt.read_text(encoding='utf8')) == record
    assert not (output/'completed.json').exists() and not (output/'candidate_result.json').exists()
    actor, _ = load_student(actor_path)
    state_path = commit.parent/json.loads(commit.read_text(encoding='utf8'))['path']
    committed = torch.load(state_path, weights_only=True)['actor']
    assert all(torch.equal(param, committed[name]) for name, param in actor.state_dict().items())
    assert not torch.equal(next(actor.parameters()), next(model.parameters()))

    gpu.query_gpus = lambda: pytest.fail('Completed training requested a GPU')
    joint.main(args+['--resume'])
    assert runs == [0]
    receipt.write_text('{}', encoding='utf8')
    with pytest.raises(ValueError, match='training completion record changed'):
        joint.main(args+['--resume'])
    state_path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Committed PPO checkpoint changed'):
        joint.main(args+['--resume'])


@pytest.mark.parametrize('legacy_source,has_receipt', [
    ('306da1b0c983a6683929c1c643d23a83f999f868be697be08a5c09f8e32d856c', False),
    ('304cdd7c22039d3dc5eff837cf449eb1ff33c96c106dcffa92f4afed5bb4f48d', True),
])
def test_training_only_adopts_exact_legacy_compact_200_commit(tmp_path, monkeypatch, legacy_source, has_receipt):
    plan_path, plan = fixture_plan(tmp_path)
    model, _, _ = learning_state(plan['schema'])
    initial, output = tmp_path/'actor.pt', tmp_path/'candidate'
    save_student(initial, model)
    gpu = ModuleType('robolab.rough_gpu')
    gpu.query_gpus = lambda: []
    gpu.candidates = lambda *args, **kwargs: []
    gpu.gpu_lease = lambda *args: nullcontext('cpu')
    gpu.child_environment = lambda *args: dict(os.environ)
    monkeypatch.setitem(sys.modules, 'robolab.rough_gpu', gpu)
    monkeypatch.setattr(joint, 'evaluate_candidate',
                        lambda *args: pytest.fail('Legacy BO diagnostics ran again'))
    # The unit fixture omits collection receipts and keeps only the learning state.
    monkeypatch.setattr(joint, 'prune_training_artifacts', lambda *args, **kwargs: None)
    def train(a, root, plan, actor, critic, optimizer, completed, digest, protocol, env):
        for index in range(1, a.updates):
            write_new(root/f'checkpoints/update_{index:06d}.json',
                      dict(completed_updates=index, manifest_sha256=digest))
        save_checkpoint(root, actor, critic, optimizer, a.updates, digest)

    def write_legacy_manifest(path, value):
        if path == output/'manifest.json':
            sources = dict(value['sources'])
            sources.pop('training_reward.py', None)
            sources.pop('finetune_settings.py')
            sources.pop('mixed_batch.py')
            sources.update({
                'joint_finetune.py': legacy_source,
                'finetune_state.py': '7b958616ceb651b059a7fef4c40bf3356b134a3dfe56a82bf1f64448b703efad',
                'ppo_channel.py': '87d6f011074fc920b365e1b24cdaa5a05913493672a14535900df96fc3c06cfb',
                'pipeline.py': 'fffd518a08c6beeeb4348816d7baab3ba208f05572a302c86bf564da10351777',
            })
            value = dict(value, sources=sources)
            value.pop('ppo_schedule')
            value.pop('init_at_random_ep_len')
        write_new(path, value)

    args = ['--plan', str(plan_path), '--student', str(initial), '--output', str(output),
            '--design-id', 'stairs_0', '--updates', '200', '--compact-training-artifacts']
    with monkeypatch.context() as legacy:
        legacy.setattr(joint, 'train_updates', train)
        legacy.setattr(joint, 'write_new', write_legacy_manifest)
        joint.main(args+['--training-only'])
    receipt = output/'training_completed.json'
    saved_receipt = receipt.read_bytes()
    if not has_receipt:
        receipt.unlink()
    manifest = output/'manifest.json'
    old_manifest = manifest.read_bytes()
    commit = output/'checkpoints/update_000200.json'
    old_commit = commit.read_bytes()
    actor_path = output/'shared_actor_finetuned.pt'
    old_actor = sha(actor_path)

    gpu.query_gpus = lambda: pytest.fail('The existing 200 updates should be reused on CPU')
    with pytest.raises(ValueError, match='changed'):
        joint.main(args+['--resume'])
    with pytest.raises(ValueError, match='changed'):
        joint.main(args+['--resume', '--training-only', '--seed', '2'])
    wrong_sources = json.loads(old_manifest)
    wrong_sources['sources']['ppo_channel.py'] = 'a'*64
    manifest.write_text(json.dumps(wrong_sources), encoding='utf8')
    with pytest.raises(ValueError, match='changed'):
        joint.main(args+['--resume', '--training-only'])
    manifest.write_bytes(old_manifest)
    joint.main(args+['--resume', '--training-only'])
    result = json.loads(receipt.read_text(encoding='utf8'))
    assert receipt.read_bytes() == saved_receipt
    assert result['ppo_updates'] == 200 and result['student_sha256'] == sha(actor_path) == old_actor
    assert manifest.read_bytes() == old_manifest and result['manifest_sha256'] == sha(manifest)
    assert commit.read_bytes() == old_commit and result['checkpoint_sha256'] == sha(commit)
    with (output/'training_reward.csv').open(newline='', encoding='utf8') as stream:
        rewards = list(csv.DictReader(stream))
    assert len(rewards) == 600
    assert rewards[-1]['update'] == '200' and rewards[-1]['task'] == 'run'
    assert all(row['session_id'] == row['reward_samples'] == row['mean_ppo_step_reward'] == ''
               for row in rewards)


def test_launcher_selects_candidate_and_keeps_robust_finetuning(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(joint.HERE.parents[3]))
    monkeypatch.delitem(sys.modules,'start',raising=False)  # rough_bo also provides a start.py.
    import start
    plan_path,plan=fixture_plan(tmp_path)
    (tmp_path/'shared_plan').mkdir(); (tmp_path/'shared_policy').mkdir()
    write_new(tmp_path/'shared_plan/plan.json',plan)
    write_new(tmp_path/'shared_policy/completed.json',dict(student='selected_actor.pt'))
    commands=[]
    monkeypatch.setattr(start,'run',lambda command,**kw:commands.append([str(x) for x in command]))
    start.main(['微调','--root',str(tmp_path),'--gpu','2','--candidate-design-id','run_1',
                '--candidate-task','run','--finetune-updates','9','--dry-run','--resume-paused',
                '--tracking-rel-tolerance','.2','--min-command-progress-ratio','.6'])
    assert len(commands)==1
    command=commands[0]
    for flag,value in {'--design-id':'run_1','--design-task':'run','--updates':'9','--profile':'robust',
                       '--randomization-strength':'1','--device':'cuda:2',
                       '--tracking-rel-tolerance':'0.2','--min-command-progress-ratio':'0.6'}.items():
        assert command[command.index(flag)+1]==value
    assert '--resume-paused' in command
    assert not (tmp_path/'candidate_finetuning').exists()
