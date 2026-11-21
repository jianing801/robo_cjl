"""Real CPU learning/IPC, synthetic environments; not Isaac gait evidence."""
from contextlib import nullcontext
from copy import deepcopy
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
from finetune_state import select_candidate, save_checkpoint, load_checkpoint, validate_batch
import joint_finetune as joint
import pipeline
from ppo_channel import validate_request
from student import SharedMLPActor, save_student, load_student
from test_shared import banks
from test_pipeline_protocol import row


def fixture_plan(tmp_path, designs=None):
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
            copies=8, full_body_torque=False)
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
    return [dict(obs=x, actions=actions, logp=logp, reward=torch.ones(24,8),
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


@pytest.mark.parametrize('corruption', ['stale','nan','shape','mask','timeout'])
def test_batch_rejects_corruption(tmp_path,corruption):
    _,plan=fixture_plan(tmp_path); model,_,_=learning_state(plan['schema'])
    data=deepcopy(batches_for(model)[0])
    session=dict(session_id='new',manifest_sha256='manifest',collection_protocol={},steps_per_task=24)
    data.update(task='stairs',index=1,session_id='new',manifest_sha256='manifest',collection_protocol={},student_sha256='actor')
    if corruption=='stale': data['session_id']='old'
    elif corruption=='nan': data['reward'][0,0]=float('nan')
    elif corruption=='shape': data['obs']=data['obs'][:-1]
    elif corruption=='mask': data['done']=data['done'].long()
    else: data['timeout'][0,0]=True
    with pytest.raises(ValueError):
        validate_batch(data,task='stairs',index=1,session=session,actor_sha256='actor',width=model.width)


def test_real_ipc_pause_resume_then_independent_evaluation(tmp_path, monkeypatch):
    plan_path,plan=fixture_plan(tmp_path)
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
channel,task,actor=sys.argv[2:]
model,_=load_student(actor)
obs=torch.zeros(8,model.width)
class Env:
 def step(self,action):
  return obs,torch.ones(8),torch.zeros(8,dtype=torch.int64),{}
session=json.loads((Path(channel)/'session.json').read_text())
serve(channel,task,Env(),SimpleNamespace(device='cpu'),obs,lambda x,reset:x,None,protocol=session['collection_protocol'])
''')
    original_popen=joint.subprocess.Popen; launches=[]
    def popen(command,**kwargs):
        channel=command[command.index('--ppo-channel')+1]
        task=command[command.index('--task')+1] if '--task' in command else 'run'
        launches.append((channel,task))
        return original_popen([sys.executable,str(stub),str(joint.HERE),channel,task,str(initial)],**kwargs)
    monkeypatch.setattr(joint.subprocess,'Popen',popen)
    original_sleep=joint.time.sleep
    monkeypatch.setattr(joint.time,'sleep',lambda seconds:None if seconds==3 else original_sleep(seconds))
    original_save=joint.save_checkpoint
    paused=[]
    def checkpoint(*args,**kwargs):
        result=original_save(*args,**kwargs)
        if args[4]==1 and not paused:
            paused.append(True); (output/'USER_PAUSED').touch()
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
        write_new(folder/'summary.json',dict(task=task,teacher=plan['teachers'][task]['checkpoint_sha256'],
            student=sha(actor),design_ids=[design],steps=1000,rows=[row(i,design=design) for i in range(8)],
            collection_protocol=pipeline.protocol_metadata(profile,1. if profile=='robust' else 0.)))
    monkeypatch.setattr(pipeline,'child',evaluate)
    args=['--plan',str(plan_path),'--student',str(initial),'--output',str(output),
          '--design-id','stairs_1','--updates','2','--timeout-seconds','60']
    with pytest.raises(KeyboardInterrupt): joint.main(args)
    assert (output/'checkpoints/update_000001.json').exists() and not (output/'completed.json').exists()
    with pytest.raises(RuntimeError,match='resume-paused'): joint.main(args+['--resume'])
    with pytest.raises(ValueError,match='changed'): joint.main(args+['--resume-paused','--updates','3'])
    assert (output/'USER_PAUSED').exists()
    with pytest.raises(KeyboardInterrupt): joint.main(args+['--resume-paused'])
    assert (output/'checkpoints/update_000002.json').exists()
    assert (output/'shared_actor_finetuned.pt').exists() and not (output/'completed.json').exists()
    joint.main(args+['--resume-paused'])
    assert not (output/'USER_PAUSED').exists() and len(launches)==6
    sessions=[json.loads(p.read_text()) for p in (output/'attempts').glob('*/session.json')]
    assert sorted(s['start_update'] for s in sessions)==[0,1]
    resumed_session=next(s for s in sessions if s['start_update']==1)
    resumed_folder=output/'attempts'/resumed_session['session_id']
    assert not (resumed_folder/'request_000.json').exists() and (resumed_folder/'request_001.json').exists()
    actor,meta=load_student(output/'shared_actor_finetuned.pt')
    assert meta['ppo_updates']==2 and meta['critic_trained'] and 'PPO_updates' not in meta
    assert any(not torch.equal(v,actor.state_dict()[k]) for k,v in model.state_dict().items())
    result=json.loads((output/'candidate_result.json').read_text())
    assert result['candidate']['design_ids']=={t:f'{t}_1' for t in plan['teachers']}
    assert result['scalar_objective'] is None and result['requires_bo_objective_definition']
    assert len(evaluations)==6 and {s for _,_,s in evaluations}=={20261118}
    for profile in ('nominal','robust'):
        assert set(result['profiles'][profile]['task_metrics'])==set(plan['teachers'])
    joint.main(args+['--resume'])
    assert len(launches)==6 and len(evaluations)==6
    (output/'candidate_result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'): joint.main(args+['--resume'])


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
