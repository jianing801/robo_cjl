"""Synthetic CPU interface tests only, never locomotion evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import schema_for,unified_context,input_width,paired_structures
from student import SharedActor,save_student,load_student
from fit import batch


def banks():
    semantic=dict(joint_names=[f'j{i}' for i in range(23)],default_joint_pos=[0.]*23)
    return {'stairs':dict(canonical_semantic_contract=semantic,ranges_m={'thigh':[.25,.4],'calf':[.3,.45]},
                          knees=['K1','K2','K3'],ankles=['4340','6216','RS06']),
            'backpack':dict(canonical_semantic_contract=deepcopy(semantic),ranges_m={'thigh':[.25,.4],'calf':[.3,.45]},
                            knees=['K2','K1'],ankles=['RS06','4340-V1.1'])}


def test_intersection_motor_ids_not_aliases_and_native_context_unchanged():
    b=banks();s=schema_for(b)
    assert s['ankles']==['RS06'] and s['knees']==['K1','K2'] and input_width(s)==1153
    row=dict(knee_motor='K1',ankle_motor='RS06',context=[-.2,.4,0.,1.,1.,0.]+list(range(16)))
    original=deepcopy(row)
    x=unified_context(b['backpack'],row,s)
    assert row==original and len(x)==21
    assert x[2+s['knees'].index('K1')]==1
    assert x[2+len(s['knees'])+s['ankles'].index('RS06')]==1
    assert x[-16:]==list(range(16))
    row['context'][2]=1.
    with pytest.raises(ValueError,match='category'):unified_context(b['backpack'],row,s)
    row['ankle_motor']='4340-V1.1'
    with pytest.raises(ValueError,match='intersection'):unified_context(b['backpack'],row,s)


def test_three_teacher_physical_context_keeps_order_and_rejects_legacy_width():
    from robolab.morphology_context import FIELDS, SCHEMA, WIDTH

    native = banks()['stairs']
    old = dict(native, knees=['K1', 'K2', 'K3', 'K4', 'K5'],
               ankles=['A1', 'A2', 'A3', 'A4', 'RS06'])
    current = dict(old, context_schema=SCHEMA, context_fields=list(FIELDS),
                   context_width=WIDTH, context_normalization={'kind': 'training_bank_minmax_v1'})
    teachers = {task: deepcopy(current) for task in ('stairs', 'backpack', 'run')}
    schema = schema_for(teachers)
    row = dict(knee_motor='K1', ankle_motor='RS06', context=[i / WIDTH for i in range(WIDTH)])
    assert input_width(schema) == 1161
    for bank in teachers.values():
        assert unified_context(bank, row, schema) == row['context']
    with pytest.raises(ValueError, match='schemas differ'):
        unified_context(old, row, schema)
    teachers['run'] = old
    with pytest.raises(ValueError, match='shape description differs'):
        schema_for(teachers)


def test_pairing_uses_actual_leg_motor_tuple_not_names_or_motor_family():
    b=banks()
    def row(t,c,k,a,name):return dict(thigh_m=t,calf_m=c,knee_motor=k,ankle_motor=a,id=name)
    b['stairs']['designs']=[row(.25,.3,'K1','RS06','rough_short'),
        row(.4,.45,'K2','RS06','rough_long'),row(.3,.35,'K1','RS06','rough_only_length'),
        row(.25,.3,'K1','4340','rough_4340')]
    b['backpack']['designs']=[row(.25,.3,'K1','RS06','backpack_short'),
        row(.4,.45,'K2','RS06','backpack_long'),row(.25,.3,'K1','4340-V1.1','backpack_4340')]
    pairs=paired_structures(b)
    assert len(pairs)==2
    assert {key[3] for key,_ in pairs}=={'RS06'}
    assert pairs[0][1]['stairs']['id']=='rough_short'
    assert pairs[0][1]['backpack']['id']=='backpack_short'


def test_joint_and_length_contract_mismatches_refused():
    b=banks();b['backpack']['canonical_semantic_contract']['joint_names'].reverse()
    with pytest.raises(ValueError,match='joint'):schema_for(b)
    b=banks();b['backpack']['ranges_m']['thigh']=[.2,.4]
    with pytest.raises(ValueError,match='scaling'):schema_for(b)


def model_and_input():
    s=schema_for(banks());m=SharedActor(s,dict(d_model=32,num_heads=2,num_layers=1,feedforward_dim=64))
    x=torch.zeros(4,input_width(s));x[:,780]=1
    return m,x


def test_history_token_order_action_order_and_task_only_goal():
    m,x=model_and_input()
    x[:,:780]=torch.arange(780).float()
    n=m.nodes(x)
    assert n.shape==(4,25,162)
    assert n[0,0,0]==0 and n[0,0,6]==78
    assert n[0,1,60]==9 and n[0,1,61]==32 and n[0,1,62]==55
    assert n[0,-1,120]==1 and n[0,0,120]==0
    assert n[0,1:24,139].eq(1).all() and n[0,-1,139]==0
    assert m(x).shape==(4,23)


def test_real_gradient_adam_update_and_no_tanh():
    m,x=model_and_input();before=m.encoder.weight.detach().clone()
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    loss=(m(x)-torch.ones(4,23)).square().mean();loss.backward()
    assert m.encoder.weight.grad.abs().sum()>0
    opt.step();assert not torch.equal(before,m.encoder.weight)
    with torch.no_grad():m.decoder.weight.zero_();m.decoder.bias.fill_(2.)
    assert m(x).eq(2).all()


def test_strict_student_roundtrip_and_wrong_kind(tmp_path):
    m,x=model_and_input();m.eval();path=tmp_path/'student.pt'
    save_student(path,m,teacher_updates=0)
    restored,data=load_student(path)
    assert torch.equal(m(x),restored(x)) and data['ppo_updates']==0 and not data['critic_trained']
    assert 'PPO_updates' not in data
    wrong=tmp_path/'wrong.pt';torch.save({'kind':'old_fixed_run_stairs'},wrong)
    with pytest.raises(ValueError,match='kind'):load_student(wrong)


def test_extra_privileged_payload_input_rejected():
    m,x=model_and_input()
    with pytest.raises(ValueError,match='input'):m(torch.cat((x,torch.ones(4,1)),1))


def test_task_balance_independent_of_design_counts():
    groups={};s=schema_for(banks());plan=dict(schema=s,task_ids={'stairs':0,'backpack':1})
    for task,count in [('stairs',9),('backpack',4)]:
        for i in range(count):
            groups[(task,str(i))]=dict(train=[(torch.zeros(5,780),torch.zeros(5,23),torch.zeros(371))])
    x,y=batch(groups,torch.Generator().manual_seed(1),plan)
    assert x.shape==(256,1153) and y.shape==(256,23)
    assert x[:,780].sum()==128 and x[:,781].sum()==128
def test_epoch_budget_scales_with_design_sources_not_raw_rows_and_preserves_explicit_cap():
    from fit import training_budget
    import torch
    def group(n):
        return {'train':[(torch.zeros(n,1),None,None)]}
    small={('stairs','a'):group(128),('backpack','a'):group(256)}
    large={('stairs',str(i)):group(128) for i in range(200)}
    large.update({('backpack',str(i)):group(256) for i in range(200)})
    assert training_budget(small,0)[0]==1
    assert training_budget(large,0)[0]==50
    assert training_budget(large,500)[0]==500


def test_shared_mlp_native_input_gradient_and_checkpoint(tmp_path):
    from student import SharedMLPActor
    schema=schema_for(banks());m=SharedMLPActor(schema,dict(hidden_dims=[32,16]))
    x=torch.arange(input_width(schema)).float().expand(3,-1).clone()/100
    f=m.features(x)
    assert f.shape==(3,803)
    assert torch.equal(f[:,:782],x[:,:782])
    assert torch.equal(f[:,782:],x[:,1132:])
    before=m.network[0].weight.detach().clone()
    optimizer=torch.optim.Adam(m.parameters(),lr=.001)
    m(x).square().mean().backward();optimizer.step()
    assert not torch.equal(before,m.network[0].weight)
    path=tmp_path/'mlp.pt';save_student(path,m)
    restored,_=load_student(path)
    assert isinstance(restored,SharedMLPActor) and torch.equal(m(x),restored(x))
    with pytest.raises(ValueError,match='input'):m(torch.cat((x,torch.ones(3,1)),1))


def test_three_task_warm_start_preserves_existing_tasks(tmp_path):
    from student import SharedMLPActor,expand_mlp_tasks
    b=banks();old=schema_for(b);old['shared_structure_keys']=[]
    m=SharedMLPActor(old,dict(hidden_dims=[32,16]));path=tmp_path/'two.pt';save_student(path,m)
    b['run']=deepcopy(b['stairs']);new=schema_for(b);new['shared_structure_keys']=[]
    expanded=expand_mlp_tasks(path,new)
    assert input_width(new)==1154
    from fit import assemble
    x=torch.randn(5,780);static=torch.randn(5,371)
    for task in old['tasks']:
        before=assemble(x,static,old['tasks'].index(task))
        after=assemble(x,static,new['tasks'].index(task),3)
        assert torch.allclose(m(before),expanded(after),atol=2e-6,rtol=1e-6)
    assert expanded.network[0].weight[:,782].eq(0).all()


def test_amp_term_history_and_real_streaming_reset():
    from amp_adapter import decode_amp_frames,StreamingHistory,TERM_WIDTHS
    frames=torch.arange(3*78).float().reshape(1,3,78)
    term_major=torch.cat([v.flatten(1) for v in frames.split(TERM_WIDTHS,dim=-1)],dim=1)
    assert torch.equal(decode_amp_frames(term_major),frames)
    history=StreamingHistory(2,'cpu')
    for i in range(12):result=history.append(torch.full((2,78),float(i)))
    assert torch.equal(result.reshape(2,10,78)[0,:,0],torch.arange(2.,12.))
    result=history.append(torch.full((2,78),12.),torch.tensor([True,False])).reshape(2,10,78)
    assert result[0].eq(12.).all() and result[1,0,0]==3.


def test_reference_routing_cache_keeps_order_reset_groups_and_rng():
    import importlib.util
    source=Path(__file__).resolve().parent.parent/'amp_engine/reference_router.py'
    spec=importlib.util.spec_from_file_location('reference_test_native',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Term:
        num_dofs=23;num_key_bodies=6
        def get_num_motions(self):return 3
        def get_motion_state(self,ids,motion_times):return {'q':ids[:,None].float()+motion_times[:,None]}
        def sample_times(self,ids):return torch.rand(len(ids))
    from amp_fast_reference import CachedReferenceRouting
    router=module.ReferenceRouter([Term(),Term(),Term()],[0,1,2],'cpu');cache=CachedReferenceRouting(router)
    for ids in [torch.tensor([0,4,8,2,5,6]),torch.tensor([1,3,7,0,5,8]),torch.tensor([7,0,3])]:
        times=torch.arange(len(ids)).float()
        assert torch.equal(router.get_motion_state(ids,times)['q'],cache.route('get_motion_state',ids,motion_times=times)['q'])
        torch.manual_seed(123);native=router._route('sample_times',ids)
        torch.manual_seed(123);optimized=cache.route('sample_times',ids)
        assert torch.equal(native,optimized)
    assert cache.hits>0 and cache.misses>0
