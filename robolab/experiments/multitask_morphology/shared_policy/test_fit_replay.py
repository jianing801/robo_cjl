"""CPU full-replay regressions, including the former 32-row information loss."""
import gc
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fit
from fit import ReplayShard,ReplayStore,load_groups,replay_exposure,replay_memory,training_budget
from test_fit_protocol import dataset


def archive(path,rows):
    x=torch.zeros(rows,780);x[:,0]=torch.arange(rows)
    torch.save(dict(base_obs=x,teacher_actions=torch.zeros(rows,23)),path)
    return str(path)


def test_all_rows_remain_eligible_and_sampling_is_not_a_fixed_32_row_subset(tmp_path):
    n=1200;path=archive(tmp_path/'recovery.pt',n)
    env=torch.arange(n)//200;time=torch.arange(n)%200
    scenario=env*3+(time*3//200);episode=torch.zeros(n,dtype=torch.int64)
    old,_=fit.stratified_rows(torch.arange(n),env,scenario,episode,max_rows=32,seed=42)
    shard=ReplayShard(ReplayStore(1),path,torch.arange(n),torch.zeros(379),env,scenario,episode)
    assert torch.equal(shard.indices.sort().values,torch.arange(n))
    generator=torch.Generator().manual_seed(42)
    epochs=[shard.sample_indices(32,generator) for _ in range(4)]
    observed=torch.cat(epochs).unique()
    assert len(observed)>32 and not torch.equal(epochs[0],epochs[1])
    assert (~torch.isin(observed,old)).any()
    # A longer deterministic coverage probe exercises every original row,
    # including the recovery labels removed by the old one-time cap.
    shard.sample_indices(30000,generator)
    assert shard.seen.all()
    recovery=(time>=100)&(time<110)
    assert shard.seen[torch.isin(shard.indices,torch.arange(n)[recovery])].all()
    report=replay_exposure({('stairs','design'):{'train':[shard]}})['stairs']
    assert report==dict(eligible_rows=n,sampled_draws=30128,unique_sampled_rows=n,unique_coverage=1.)


def test_on_demand_draws_are_balanced_by_scenario_then_physical_episode(tmp_path):
    path=archive(tmp_path/'imbalanced.pt',420)
    scenario=torch.tensor([0]*400+[1]*20)
    env=torch.tensor([0]*200+[1]*200+[2]*10+[3]*10)
    shard=ReplayShard(ReplayStore(),path,torch.arange(420),torch.zeros(379),env,scenario,torch.zeros(420,dtype=torch.int64))
    rows=shard.sample_indices(24000,torch.Generator().manual_seed(73))
    fractions=torch.bincount(env[rows],minlength=4)/len(rows)
    assert (fractions-.25).abs().max()<.015


def test_loader_retains_full_training_split_and_separate_fixed_validation(dataset):
    plan,data,save=dataset;path=save(data)
    groups,sources,_=load_groups([path],plan,'plan',validation_rows=4,replay_cache_shards=1)
    group=groups['stairs','design'];train=group['train'][0];heldout=group['validation'][0]
    assert torch.equal(train.indices.sort().values,torch.arange(0,16,2))
    assert len(heldout)==4 and (heldout.indices%2==1).all()
    fixed=heldout.indices.clone()
    train.sample_indices(1000,torch.Generator().manual_seed(2))
    assert torch.equal(heldout.indices,fixed) and heldout.seen is None
    assert sources[0]['sampling'][0]['all_original_rows_eligible']
    again,_,_=load_groups([path],plan,'plan',validation_rows=4,replay_cache_shards=1)
    assert torch.equal(again['stairs','design']['validation'][0].indices,fixed)


def test_diagnostic_draws_do_not_count_as_training_exposure(dataset):
    plan,data,save=dataset
    groups,_,_=load_groups([save(data)],plan,'plan',validation_rows=4)
    fit.batch(groups,torch.Generator().manual_seed(7),plan,record_exposure=False)
    assert replay_exposure(groups)['stairs']['sampled_draws']==0
    fit.batch(groups,torch.Generator().manual_seed(7),plan)
    assert replay_exposure(groups)['stairs']['sampled_draws']==128


@pytest.mark.parametrize('routing',[torch.full((16,),-1,dtype=torch.int64),torch.full((16,),1,dtype=torch.int64)])
def test_invalid_design_routing_cannot_silently_drop_original_rows(dataset,routing):
    plan,data,save=dataset;data['design_index']=routing
    with pytest.raises(ValueError,match='every original row'):
        load_groups([save(data)],plan,'plan')


def test_mmap_cache_evicts_archive_mappings_and_metadata_remains_small(tmp_path):
    paths=[archive(tmp_path/f'archive_{i}.pt',4096) for i in range(4)]
    store=ReplayStore(2);shards=[]
    for path in paths:
        shard=ReplayShard(store,path,torch.arange(4096),torch.zeros(379),torch.zeros(4096,dtype=torch.int64))
        x,y=shard.sample(16,torch.Generator().manual_seed(0));shards.append(shard)
        assert x.shape==(16,780) and y.shape==(16,23) and len(store.cache)<=2
    assert list(store.cache)==paths[-2:]
    groups={('stairs',str(i)):{'train':[shard],'validation':[]} for i,shard in enumerate(shards)}
    memory=replay_memory(groups)
    original_bytes=4*4096*(780+23)*4
    assert memory['full_replay_rows']==16384 and memory['cache_limit']==2 and memory['mmap']
    assert memory['access_bitmap_bytes']==16384
    assert memory['replay_index_bytes']+memory['access_bitmap_bytes']+memory['static_context_bytes']<original_bytes/100
    del x,y;gc.collect()
    maps=Path('/proc/self/maps')
    if maps.exists():
        mapped=maps.read_text()
        assert paths[0] not in mapped and paths[1] not in mapped
        assert paths[2] in mapped and paths[3] in mapped


def test_unsupported_archive_never_falls_back_to_eager_ram(tmp_path):
    path=tmp_path/'legacy_format.pt'
    torch.save({'base_obs':torch.zeros(2,780)},path,_use_new_zipfile_serialization=False)
    with pytest.raises(ValueError,match='No full-RAM fallback'):
        ReplayStore().load(path)


def test_archives_cannot_change_after_initial_validation(tmp_path):
    path=archive(tmp_path/'changed.pt',4);store=ReplayStore();store.load(path)
    # Windows forbids overwriting an open mmap; release mapping, retain validated identity.
    store.cache.clear()
    import gc
    gc.collect()
    archive(Path(path),5)
    with pytest.raises(ValueError,match='changed during fitting'):store.load(path)


def test_full_tensor_validation_reads_bounded_chunks_and_catches_the_last_row(monkeypatch):
    x=torch.zeros(1025,780);y=torch.zeros(1025,23);x[-1,-1]=float('nan')
    original=torch.isfinite;shapes=[]
    def record(value):
        shapes.append(value.shape);return original(value)
    monkeypatch.setattr(fit.torch,'isfinite',record)
    with pytest.raises(ValueError,match='Bad actual teacher tensors'):
        fit._validate_teacher_tensors(x,y,None,chunk_rows=256)
    assert max(shape[0] for shape in shapes)<=256 and shapes[-1][0]==1


def test_optimization_budget_is_independent_of_raw_replay_and_validation_size():
    def groups(n,validation_n):
        return {(task,str(i)):{'train':[(torch.zeros(n,1),None,None)]*4,
                               'validation':[(torch.zeros(validation_n,1),None,None)]}
                for task in ('stairs','backpack','run') for i in range(100)}
    small=groups(1200,32);large=groups(12000,128)
    assert training_budget(small,0)[0]==training_budget(large,0)[0]==100
    assert training_budget(large,7)[0]==7


def test_every_epoch_is_saved_even_when_mse_gets_worse(dataset,tmp_path,monkeypatch):
    plan,data,save=dataset
    plan['schema'].update(knees=[f'k{i}' for i in range(6)],ankles=[f'a{i}' for i in range(5)])
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    data['plan_sha256']=fit.sha(plan_path)
    data_path=save(data);listing=tmp_path/'data_list.json';listing.write_text(json.dumps([data_path]))
    output=tmp_path/'fit_run'
    from student import SharedMLPActor
    class TinyActor(SharedMLPActor):
        def __init__(self,schema):super().__init__(schema,{'hidden_dims':[8]})
    monkeypatch.setattr(fit,'SharedMLPActor',TinyActor)
    monkeypatch.setattr(fit,'load_plan',lambda _:plan)
    scores=iter([.1,2.,1.])
    def validation(*args):
        score=next(scores);return {'stairs/design':score},{'stairs':score}
    monkeypatch.setattr(fit,'validation',validation)
    monkeypatch.setattr(sys,'argv',['fit.py','--plan',str(plan_path),'--data-list',str(listing),
        '--output',str(output),'--architecture','mlp','--device','cpu','--epochs','3',
        '--steps-per-epoch','1','--validation-rows-per-design-shard','4'])
    fit.main()
    summary=json.loads((output/'summary.json').read_text())
    assert [row['epoch'] for row in summary['checkpoints']]==[0,1,2]
    assert summary['student']==summary['checkpoints'][-1]['student']
    assert summary['balanced_validation_MSE']==1.0
    assert summary['closed_loop_required'] and summary['full_replay_rows']==8
    for row in summary['checkpoints']:
        assert Path(row['student']).is_file() and fit.sha(row['student'])==row['student_sha256']
        assert row['validation_MSE']['stairs']==row['balanced_validation_MSE']


def test_retired_training_row_cap_fails_with_migration_instructions(monkeypatch):
    monkeypatch.setattr(sys,'argv',['fit.py','--plan','unused','--data-list','unused','--output','unused',
        '--max-rows-per-design-shard','32'])
    with pytest.raises(ValueError,match='retired; use --samples-per-design-per-epoch'):fit.main()


def test_real_three_task_fit_subprocess_mmap_to_all_epoch_checkpoints(tmp_path):
    """No production-function stubs: actual plan/URDF checks, replay and Adam."""
    from contract import graph_features,schema_for,unified_context
    from randomization import protocol_metadata
    from student_observations import boundary_contract
    names=[f'j{i}' for i in range(23)];q0=[0.]*23
    links=['<link name="base"><inertial><mass value="1"/></inertial></link>']
    joints=[]
    for i,name in enumerate(names):
        links.append(f'<link name="link_{i}"><inertial><mass value=".1"/></inertial></link>')
        joints.append(f'<joint name="{name}" type="revolute"><parent link="base"/><child link="link_{i}"/>'
            '<origin xyz="0 0 .1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2"/></joint>')
    urdf=tmp_path/'synthetic.urdf';urdf.write_text('<robot name="synthetic">'+''.join(links+joints)+'</robot>')
    tasks=['stairs','backpack','run'];banks={};teachers={}
    key=[.3,.35,'K1','A1']
    for task in tasks:
        design=dict(id=task+'_design',thigh_m=.3,calf_m=.35,knee_motor='K1',ankle_motor='A1',
            context=[-1/3,-1/3,1.,1.]+[0.]*16,urdf=str(urdf),urdf_sha256=fit.sha(urdf))
        banks[task]=dict(canonical_semantic_contract=dict(joint_names=names,default_joint_pos=q0),
            ranges_m=dict(thigh=[.25,.4],calf=[.3,.45]),knees=['K1'],ankles=['A1'],designs=[design])
        bank_path=tmp_path/(task+'_bank.json');bank_path.write_text(json.dumps(banks[task]))
        teacher_path=tmp_path/(task+'_teacher.pt');teacher_path.write_bytes(b'synthetic frozen teacher identity')
        teachers[task]=dict(bank=str(bank_path),bank_sha256=fit.sha(bank_path),design_ids=[design['id']],
            checkpoint=str(teacher_path),checkpoint_sha256=fit.sha(teacher_path))
    schema=schema_for(banks);schema['shared_structure_keys']=[key]
    shared=tmp_path/'shared.json';shared.write_text(json.dumps({'structures':[key]}))
    plan=dict(kind='rough_backpack_amp_distillation_plan_v3',structure_scope='teacher_intersection_only',
        teachers=teachers,schema=schema,task_ids={task:i for i,task in enumerate(tasks)},
        shared_bank=str(shared),shared_bank_sha256=fit.sha(shared),selected_structure_keys=[key])
    plan_path=tmp_path/'real_plan.json';plan_path.write_text(json.dumps(plan))
    paths=[];generator=torch.Generator().manual_seed(884)
    for task_id,task in enumerate(tasks):
        rows=512;env=torch.arange(rows)%2;time=torch.arange(rows)//2
        x=torch.randn(rows,780,generator=generator)*.1
        command=torch.zeros(rows,3);command[:,0]=(1.5 if task=='run' else .3)+.1*(time%3)
        x[:,-72:-69]=command
        y=x[:,-69:-46]*.2+task_id*.01
        design=banks[task]['designs'][0]
        static=[v for node in graph_features(str(urdf),names,q0) for v in node]+unified_context(banks[task],design,schema)
        io=dict(joint_names=names,default_joint_pos=q0,step_dt=.02,action_scale=.25,history_width=780,
            history_layout='frame_major_oldest_to_newest',
            observation_scales={name:1. for name in ('lin_vel','ang_vel','projected_gravity','commands','joint_pos','joint_vel','actions','height_scan')},
            **boundary_contract())
        data=dict(kind='real_rough_backpack_teacher_data_v1',task=task,plan_sha256=fit.sha(plan_path),
            teacher_sha256=teachers[task]['checkpoint_sha256'],bank_sha256=teachers[task]['bank_sha256'],schema=schema,
            actual_payload_not_in_actor=True,io_contract=io,base_obs=x,teacher_actions=y,
            validation=env.bool(),environment_ids=env,design_ids=[design['id']],design_index=torch.zeros(rows,dtype=torch.int64),
            static_context=torch.tensor([static]),source='synthetic_cpu_smoke',scenario_bin=time%3,episode_id=time//64,
            actual_command=command,collection_protocol=protocol_metadata('robust',.5),
            native_boundaries={'action_clip':None if task=='run' else 100.})
        path=tmp_path/(task+'_data.pt');torch.save(data,path);paths.append(str(path))
    listing=tmp_path/'data.json';listing.write_text(json.dumps(paths));output=tmp_path/'three_task_fit'
    process=subprocess.run([sys.executable,str(Path(fit.__file__)),'--plan',str(plan_path),'--data-list',str(listing),
        '--output',str(output),'--architecture','mlp','--device','cpu','--epochs','2'],
        capture_output=True,text=True,timeout=90)
    assert process.returncode==0,process.stdout+'\n'+process.stderr
    summary=json.loads((output/'summary.json').read_text());manifest=json.loads((output/'manifest.json').read_text())
    assert summary['full_replay_rows']==3*256 and summary['optimizer_steps']==2
    assert summary['mmap'] and summary['cache_limit']==2 and summary['access_bitmap_bytes']==3*256
    assert manifest['replay_row_cap_per_design_source_split'] is None
    assert manifest['validation_rows_per_design_source_split']==128
    assert [row['epoch'] for row in summary['checkpoints']]==[0,1]
    for candidate in summary['checkpoints']:
        assert fit.sha(candidate['student'])==candidate['student_sha256']
        assert set(candidate['validation_MSE'])==set(tasks)
    for task,row in summary['cumulative_training_exposure'].items():
        assert row['eligible_rows']==256 and row['sampled_draws']==256
        assert 32<row['unique_sampled_rows']<=256
        assert row['unique_coverage']==row['unique_sampled_rows']/256


def test_collection_grid_replay_requires_exact_native_ids_for_every_teacher(dataset):
    from itertools import product

    plan, data, save = dataset
    original = plan["teachers"]["stairs"]
    sample = json.loads(Path(original["bank"]).read_text())["designs"][0]
    tasks = ("stairs", "backpack", "run")
    keys = list(product((.20, .25, .30, .35, .40), (.25, .30, .35, .40, .45),
                        (f"k{i}" for i in range(5)), (f"a{i}" for i in range(5))))
    expected = [i for i, key in enumerate(keys) if key[0] != .30 and key[1] != .35]
    outside = next(i for i in range(len(keys)) if i not in expected)
    assert len(expected) == 400
    plan["selected_structure_keys"] = [list(key) for key in keys]
    plan["schema"].update(tasks=list(tasks), shared_structure_keys=plan["selected_structure_keys"])
    plan["task_ids"] = {task: i for i, task in enumerate(tasks)}
    parent = Path(original["bank"]).parent
    for task in tasks:
        designs = [dict(sample, id=f"{task}_{i}", thigh_m=key[0], calf_m=key[1],
                        knee_motor=key[2], ankle_motor=key[3]) for i, key in enumerate(keys)]
        bank = parent / f"{task}_grid_bank.json"
        bank.write_text(json.dumps({"designs": designs}))
        plan["teachers"][task] = dict(original, bank=str(bank), bank_sha256=fit.sha(bank),
                                      design_ids=[design["id"] for design in designs])

    def replay(task, indices, name):
        count = len(indices)
        return save(dict(data, task=task, bank_sha256=plan["teachers"][task]["bank_sha256"],
                         design_ids=[f"{task}_{i}" for i in indices],
                         base_obs=torch.zeros(count * 2, 780), teacher_actions=torch.zeros(count * 2, 23),
                         static_context=torch.zeros(count, 379), design_index=torch.arange(count).repeat_interleave(2),
                         validation=torch.tensor([False, True]).repeat(count),
                         environment_ids=torch.tensor([0, 1]).repeat(count),
                         scenario_bin=torch.zeros(count * 2, dtype=torch.int64),
                         episode_id=torch.zeros(count * 2, dtype=torch.int64),
                         actual_command=torch.tensor([[.5, 0., 0.]]).repeat(count * 2, 1)), name)

    paths = [replay(task, expected, f"{task}_grid.pt") for task in tasks]
    groups, _, _ = load_groups(paths, plan, "plan", collection_grid=True, validation_rows=1)
    assert set(groups) == {(task, f"{task}_{i}") for task in tasks for i in expected}
    assert all(len(group["train"]) == len(group["validation"]) == 1 for group in groups.values())
    with pytest.raises(ValueError, match="(?i)design|coverage|grid"):
        mixed = replay("stairs", [outside, *expected[1:]], "stairs_mixed.pt")
        load_groups([mixed, *paths[1:]], plan, "plan", collection_grid=True, validation_rows=1)
