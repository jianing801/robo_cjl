"""Retain the candidate, publication and reward contracts for native V4."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contract import schema_for, structure_key, sha, write_new, load_plan
from finetune_state import select_candidate
from student import SharedMLPActor, save_student, load_student
from test_shared import banks
from training_reward import RolloutRewards


HERE = Path(__file__).resolve().parent


def fixture_plan(tmp_path, designs=None):
    native=banks(); native['run']=deepcopy(native['stairs'])
    keys=designs or [dict(thigh_m=.25,calf_m=.3,knee_motor='K1',ankle_motor='RS06'),
                     dict(thigh_m=.4,calf_m=.45,knee_motor='K2',ankle_motor='RS06')]
    teachers={}
    for task,bank in native.items():
        bank['designs']=[dict(d,id=f'{task}_{i}') for i,d in enumerate(keys)]
        path=tmp_path/f'{task}.json'; write_new(path,bank)
        checkpoint=tmp_path/f'{task}.pt'; checkpoint.write_bytes(b'frozen teacher')
        teachers[task]=dict(bank=str(path),bank_sha256=sha(path),checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),design_ids=[d['id'] for d in bank['designs']],
            copies=8,full_body_torque=True)
    schema=schema_for(native)
    schema['shared_structure_keys']=[list(structure_key(d)) for d in keys]
    shared=tmp_path/'shared.json'
    write_new(shared,dict(structures=[dict(key=key,design_ids={t:f'{t}_{i}' for t in native})
        for i,key in enumerate(schema['shared_structure_keys'])]))
    plan=dict(kind='rough_backpack_amp_distillation_plan_v3',structure_scope='teacher_intersection_only',
        shared_bank=str(shared),shared_bank_sha256=sha(shared),teachers=teachers,schema=schema,
        task_ids={t:i for i,t in enumerate(native)},selected_structure_keys=schema['shared_structure_keys'],scope='all')
    path=tmp_path/'plan.json';write_new(path,plan)
    return path,load_plan(path)


def test_candidate_maps_formal_2500_grid_exactly(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent/'engine'))
    from hybrid_bank import select_grid
    config=json.loads((HERE.parent/'configs/rough.json').read_text(encoding='utf8'))['grid']
    grid=select_grid(config,config['knee_pool'],config['ankle_pool'])
    assert grid['design_count']==2500 and grid['context_width']==28
    _,plan=fixture_plan(tmp_path,grid['designs'])
    before=deepcopy(plan)
    selected,candidate=select_candidate(plan,design_id='backpack_1234',design_task='backpack')
    assert candidate['design_ids']=={t:f'{t}_1234' for t in plan['teachers']}
    assert selected['schema']==plan['schema'] and plan==before
    assert all(s['design_ids']==[f'{t}_1234'] for t,s in selected['teachers'].items())
    assert select_candidate(plan,design_key=candidate['structure_key'])[1]==candidate
    with pytest.raises(ValueError,match='outside'):
        select_candidate(plan,design_key=[.25,.3,config['knee_pool'][0],config['ankle_pool'][0]])
    with pytest.raises(ValueError,match='Candidate ID'):
        select_candidate(plan,design_id='missing')
    broken=deepcopy(plan);broken['teachers']['run']['design_ids'].remove('run_1234')
    with pytest.raises(ValueError,match='exactly once'):
        select_candidate(broken,design_id='stairs_1234')


def test_native_completion_export_cannot_edit_committed_actor_or_pause(tmp_path):
    from finetune_common import publish_training_completion,STEPS_PER_TASK
    from randomization import protocol_metadata
    _,plan=fixture_plan(tmp_path)
    _,candidate=select_candidate(plan,design_id='stairs_1')
    model=SharedMLPActor(plan['schema'],dict(hidden_dims=[8]))
    initial=tmp_path/'initial.pt';save_student(initial,model)
    args=SimpleNamespace(student=initial,updates=1)
    root=tmp_path/'native';root.mkdir()
    (root/'checkpoints').mkdir()
    commit=root/'checkpoints/update_000001.json';write_new(commit,dict(completed_updates=1))
    protocol=protocol_metadata('robust',1.)
    assert STEPS_PER_TASK==24
    publish_training_completion(args,root,model,candidate,'manifest',protocol)
    final=root/'shared_actor_finetuned.pt';receipt=root/'training_completed.json'
    assert receipt.is_file()
    saved,metadata=load_student(final)
    assert metadata['ppo_updates']==1 and metadata['initial_actor_sha256']==sha(initial)
    assert all(torch.equal(value,saved.state_dict()[key]) for key,value in model.state_dict().items())
    expected=json.loads(receipt.read_text(encoding='utf8'))
    publish_training_completion(args,root,model,candidate,'manifest',protocol)
    assert json.loads(receipt.read_text(encoding='utf8'))==expected
    (root/'USER_PAUSED').touch()
    with pytest.raises(KeyboardInterrupt,match='USER_PAUSED'):
        publish_training_completion(args,root,model,candidate,'manifest',protocol)


def test_mean_reward_tracks_global_and_task_episode_returns():
    logger=RolloutRewards()
    def rollout(task,step_reward,done):
        reward=torch.tensor([step_reward])
        return dict(task=task,environment_reward=reward,reward=reward*2,
                    done=torch.tensor([done]),timeout=torch.zeros(1,len(done),dtype=torch.bool))
    first=logger.summarize([rollout('stairs',[2.,4.],[True,False]),
                            rollout('run',[20.],[False])],'session')
    assert first['mean_reward']['all']['mean_reward']==2.
    assert first['mean_reward']['run']['mean_reward'] is None
    second=logger.summarize([rollout('stairs',[4.,4.],[True,True]),
                             rollout('run',[22.],[True])],'session')
    assert second['mean_reward']['all']==dict(mean_reward=14.,window_episodes=4)
    assert second['mean_reward']['stairs']==dict(mean_reward=14./3,window_episodes=3)
    assert second['mean_reward']['run']==dict(mean_reward=42.,window_episodes=1)
