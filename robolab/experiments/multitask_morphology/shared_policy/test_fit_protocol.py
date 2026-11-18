"""CPU replay/protocol checks; these are not locomotion validation."""
from pathlib import Path
from copy import deepcopy
import json
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fit
from fit import batch,collection_protocol,load_groups,stratified_rows,validation
from randomization import protocol_metadata,PROTOCOL_SOURCE_FILES


def test_fixed_budget_preserves_rare_scenarios_and_episode_balance():
    # The common scenario has 20x more rows; each scenario has two environments.
    scenario=torch.tensor([0]*400+[1]*20,dtype=torch.int64)
    environment=torch.tensor([0]*200+[1]*200+[2]*10+[3]*10,dtype=torch.int64)
    episode=torch.zeros(420,dtype=torch.int64)
    selected,report=stratified_rows(torch.arange(420),environment,scenario,episode,max_rows=32,seed=17)
    assert len(selected)==len(selected.unique())==32
    assert torch.bincount(scenario[selected]).tolist()==[16,16]
    assert torch.bincount(environment[selected]).tolist()==[8,8,8,8]
    assert report['all_episodes_retained'] and not report['legacy_environment_only']
    replay,_=stratified_rows(torch.arange(420),environment,scenario,episode,max_rows=32,seed=17)
    assert torch.equal(selected,replay)


def test_episode_identity_includes_environment_and_short_strata_redistribute():
    scenario=torch.zeros(41,dtype=torch.int64)
    environment=torch.tensor([0]+[1]*20+[2]*20,dtype=torch.int64)
    episode=torch.zeros(41,dtype=torch.int64)
    selected,report=stratified_rows(torch.arange(41),environment,scenario,episode,max_rows=9,seed=8)
    assert torch.bincount(environment[selected]).tolist()==[1,4,4]
    assert report['scenario_coverage'][0]['available_episodes']==3


def test_insufficient_scenario_budget_refused_episode_shortfall_reported():
    ids=torch.arange(5)
    with pytest.raises(ValueError,match='cannot cover 5 scenario bins'):
        stratified_rows(ids,ids,ids,torch.zeros_like(ids),max_rows=4)
    selected,report=stratified_rows(ids,ids,torch.zeros_like(ids),torch.zeros_like(ids),max_rows=4)
    assert len(selected)==4 and report['all_scenarios_retained']
    assert not report['all_episodes_retained']
    assert report['scenario_coverage'][0]['retained_episodes']==4


def test_legacy_sampling_balances_environments_without_inventing_scenarios():
    env=torch.tensor([0]*40+[1]*10,dtype=torch.int64)
    selected,report=stratified_rows(torch.arange(50),env,max_rows=16,seed=99)
    assert torch.bincount(env[selected]).tolist()==[8,8]
    assert report['legacy_environment_only']
    assert len(report['scenario_coverage'])==1
    full,_=stratified_rows(torch.arange(50),env,max_rows=0)
    assert torch.equal(full.sort().values,torch.arange(50))


def test_sampling_never_leaks_rows_from_another_design_or_split():
    env=torch.arange(100)%8
    scenario=torch.arange(100)%3
    episode=torch.arange(100)//16
    eligible=torch.arange(1,100,2)
    selected,_=stratified_rows(eligible,env,scenario,episode,max_rows=15,seed=3)
    assert len(selected)==15 and (selected%2==1).all()


def test_three_task_batches_still_have_128_rows_per_task():
    tasks=['stairs','backpack','run']
    plan=dict(schema=dict(tasks=tasks),task_ids={name:i for i,name in enumerate(tasks)})
    groups={}
    for task_id,task in enumerate(tasks):
        for design in range(task_id+1):
            groups[task,str(design)]={'train':[(torch.zeros(2+task_id,780),
                torch.full((2+task_id,23),float(task_id)),torch.zeros(379))]}
    x,y=batch(groups,torch.Generator().manual_seed(4),plan)
    assert x.shape==(384,1162)
    assert x[:,780:783].sum(0).tolist()==[128,128,128]
    assert torch.equal(x[:,780:783].argmax(1),y[:,0].long())


@pytest.fixture
def dataset(tmp_path,monkeypatch):
    """Small physical identities; only URDF-to-static-feature calculation is stubbed."""
    urdf=tmp_path/'robot.urdf';urdf.write_text('<robot name="synthetic"/>')
    design=dict(id='design',urdf=str(urdf),urdf_sha256=fit.sha(urdf))
    bank=tmp_path/'bank.json';bank.write_text(json.dumps(dict(designs=[design])))
    semantic=dict(joint_names=[f'j{i}' for i in range(23)],default_joint_pos=[0.]*23)
    schema=dict(tasks=['stairs'],semantic_contract=semantic)
    plan=dict(schema=schema,teachers={'stairs':dict(bank=str(bank),design_ids=['design'],
        checkpoint_sha256='teacher',bank_sha256=fit.sha(bank))},task_ids={'stairs':0})
    io=dict(**semantic,step_dt=.02,action_scale=.25,history_width=780,
        history_layout='frame_major_oldest_to_newest',observation_scales={'joint_pos':1.},
        student_action_clip=100.,student_observation_clip=100.,previous_action='executed_unscaled')
    data=dict(kind='real_rough_backpack_teacher_data_v1',task='stairs',plan_sha256='plan',schema=schema,
        teacher_sha256='teacher',bank_sha256=fit.sha(bank),actual_payload_not_in_actor=True,
        io_contract=io,base_obs=torch.zeros(16,780),teacher_actions=torch.zeros(16,23),
        validation=torch.arange(16)%2==1,environment_ids=torch.arange(16)%2,
        design_ids=['design'],design_index=torch.zeros(16,dtype=torch.int64),static_context=torch.zeros(1,379),
        source='synthetic',scenario_bin=torch.arange(16)//8,episode_id=torch.zeros(16,dtype=torch.int64),
        actual_command=torch.tensor([[.5,0.,0.]]).repeat(16,1),native_boundaries={'action_clip':None},
        collection_protocol=protocol_metadata('robust',.25))
    monkeypatch.setattr(fit,'graph_features',lambda *args:[[0.]*14 for _ in range(25)])
    monkeypatch.setattr(fit,'unified_context',lambda *args:[0.]*29)
    def save(value,name='data.pt'):
        path=tmp_path/name;torch.save(value,path);return str(path)
    return plan,data,save


def test_loader_allows_curriculum_strength_and_native_teacher_boundary_changes(dataset):
    plan,data,save=dataset
    other=deepcopy(data);other['collection_protocol']['strength']=1.
    other['native_boundaries']['action_clip']=100.
    paths=[save(data),save(other,'later.pt')]
    groups,sources,io=load_groups(paths,plan,'plan',validation_rows=4,sampling_seed=19)
    assert len(groups['stairs','design']['train'])==2
    assert sum(len(shard) for shard in groups['stairs','design']['train'])==16
    assert [source['collection_protocol']['strength'] for source in sources]==[.25,1.]
    assert [source['native_boundaries']['action_clip'] for source in sources]==[None,100.]
    assert io['student_action_clip']==100.
    assert all(r['retained_rows']==(8 if r['split']=='train' else 4) for source in sources for r in source['sampling'])


@pytest.mark.parametrize('profile',['nominal','robust'])
def test_actual_collector_protocol_metadata_is_accepted_by_fit(dataset,profile):
    plan,data,save=dataset
    metadata=protocol_metadata(profile,.5)
    assert set(metadata)=={'version','profile','protocol_hash','strength','source_sha256'}
    assert set(metadata['source_sha256'])==set(PROTOCOL_SOURCE_FILES)
    data['collection_protocol']=metadata
    assert collection_protocol(data)==metadata
    _,sources,_=load_groups([save(data)],plan,'plan',validation_rows=4)
    assert sources[0]['collection_protocol']==metadata


@pytest.mark.parametrize('sources',[{},None,{'rollout.py':'not-a-sha256'}])
def test_collection_source_identity_requires_complete_valid_hashes(dataset,sources):
    _,data,_=dataset;data['collection_protocol']['source_sha256']=sources
    with pytest.raises(ValueError,match='source_sha256'):
        collection_protocol(data)


def test_source_changes_cannot_hide_behind_an_unchanged_protocol_hash(dataset):
    plan,data,save=dataset;other=deepcopy(data)
    other['collection_protocol']['source_sha256']['rollout.py']='f'*64
    with pytest.raises(ValueError,match='Incompatible collection protocols'):
        load_groups([save(data),save(other,'changed_source.pt')],plan,'plan')


@pytest.mark.parametrize('key,value',[('profile','nominal'),('protocol_hash','b'*64)])
def test_loader_refuses_incompatible_protocol_mixture(dataset,key,value):
    plan,data,save=dataset
    other=deepcopy(data);other['collection_protocol'][key]=value
    with pytest.raises(ValueError,match='Incompatible collection protocols'):
        load_groups([save(data),save(other,'other.pt')],plan,'plan')


def test_all_legacy_still_loads_but_cannot_be_mixed_into_new_run(dataset):
    plan,data,save=dataset;legacy=deepcopy(data)
    for key in ('collection_protocol','scenario_bin','episode_id','actual_command','native_boundaries'):
        legacy.pop(key)
    for key in ('student_action_clip','student_observation_clip','previous_action'):
        legacy['io_contract'].pop(key)
    legacy_path=save(legacy,'legacy.pt')
    _,sources,_=load_groups([legacy_path],plan,'plan',validation_rows=4)
    assert sources[0]['sampling'][0]['legacy_environment_only']
    for paths in ([legacy_path,save(data)],[save(data,'new.pt'),legacy_path]):
        with pytest.raises(ValueError,match='Incompatible collection protocols'):
            load_groups(paths,plan,'plan')


def test_old_standing_protocol_is_not_relabelled_as_motion_only(dataset):
    _, data, _ = dataset
    data['collection_protocol']['version'] = 'shared_randomization_v1'
    with pytest.raises(ValueError, match='Invalid collection_protocol'):
        collection_protocol(data)


@pytest.mark.parametrize('speed', [0., .04, -1.])
def test_motion_only_replay_refuses_standing_or_nonforward_samples(dataset, speed):
    plan, data, save = dataset
    data['actual_command'][3,0] = speed
    with pytest.raises(ValueError, match='cannot contain standing'):
        load_groups([save(data)], plan, 'plan')


@pytest.mark.parametrize('key,value',[('student_action_clip',10.),('student_observation_clip',None),
    ('previous_action','raw_unclipped')])
def test_student_boundary_contract_is_enforced(dataset,key,value):
    _,data,_=dataset;data['io_contract'][key]=value
    with pytest.raises(ValueError,match='boundary semantics'):
        collection_protocol(data)


@pytest.mark.parametrize('strength',[float('nan'),1.01,True])
def test_collection_strength_matches_the_versioned_profile_range(dataset,strength):
    _,data,_=dataset;data['collection_protocol']['strength']=strength
    with pytest.raises(ValueError,match='Invalid collection_protocol'):
        collection_protocol(data)


@pytest.mark.parametrize('key,value',[('scenario_bin',torch.zeros(16)),
    ('episode_id',torch.full((16,),-1,dtype=torch.int64)),('actual_command',torch.zeros(16,2))])
def test_new_row_metadata_is_checked(dataset,key,value):
    plan,data,save=dataset;data[key]=value
    with pytest.raises(ValueError,match='annotations'):
        load_groups([save(data)],plan,'plan')


@pytest.mark.parametrize('field',['base_obs','teacher_actions'])
def test_new_actual_tensors_must_obey_declared_numerical_boundaries(dataset,field):
    plan,data,save=dataset;data[field][0,0]=100.01
    with pytest.raises(ValueError,match='exceed the declared numerical boundary'):
        load_groups([save(data)],plan,'plan')


def test_validation_chunks_all_retained_rows_without_second_truncation():
    class Zero(torch.nn.Module):
        def forward(self,x):return torch.zeros(len(x),23)
    targets=torch.zeros(100,23);targets[1]=10.
    groups={('stairs','design'):{'validation':[(torch.zeros(100,780),targets,torch.zeros(379))]}}
    plan=dict(schema=dict(tasks=['stairs']),task_ids={'stairs':0})
    details,means=validation(Zero(),groups,plan,'cpu',cap=4)
    assert details['stairs/design']==pytest.approx(1.)
    assert means['stairs']==pytest.approx(1.)
