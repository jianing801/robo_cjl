"""Prepare strict three-teacher intersection and audited shared-MLP warm start."""
from copy import deepcopy
import json
from pathlib import Path
import torch
from contract import load_plan,sha,write_new,schema_for,paired_structures,unified_context,graph_features
from student import expand_mlp_tasks,load_student,save_student
from fit import assemble

def main():
    w=Path(__file__).resolve().parents[4];out=w/'tri30';out.mkdir(exist_ok=False)
    old_path=w/'rbc2/plan.json';old=load_plan(old_path)
    teachers=deepcopy(old['teachers'])
    checkpoint=w/'am30/bundle/model_4999.pt'
    if sha(checkpoint)!='11721ba425188a68a4265f4fe45989097aac2b0f10db8ce23f29e8839996ff7b':raise ValueError('AMP checkpoint differs')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=state['model_state_dict']['_extra_state']['config']
    if state['iter']!=4999 or config['role']!='run' or config['bank_sha256']!=sha(w/'am30/bundle/bank.json'):
        raise ValueError('Wrong AMP role, iteration or original bank')
    teachers['run']=dict(code_root=str(w/'am30/code3/robolab'),bank=str(w/'am30/bank_local3.json'),
        original_bank=str(w/'am30/bundle/bank.json'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        expected_iter=4999,context_width=24,copies=4,full_design_count=900,
        reference_library=str(w/'am30/library_local3.json'),reference_library_sha256=sha(w/'am30/library_local3.json'),
        asset_cache=[],native_history_width=234,native_history_layout='term_major',
        actor_normalizer_restored=True,nominal_command_speed=2.)
    teachers['run']['bank_sha256']=sha(teachers['run']['bank'])
    banks={t:json.loads(Path(s['bank']).read_text(encoding='utf8')) for t,s in teachers.items()}
    pairs=paired_structures(banks);schema=schema_for(banks)
    schema['shared_structure_keys']=[list(k) for k,_ in pairs]
    if schema['shared_structure_keys']!=old['schema']['shared_structure_keys'] or len(pairs)!=200:
        raise ValueError('Three-task intersection differs; explicit scope review required')
    rows=[]
    for key,designs in pairs:
        rows.append(dict(thigh_m=key[0],calf_m=key[1],knee_motor=key[2],ankle_motor=key[3],
            native_design_ids={t:d['id'] for t,d in designs.items()},
            physical_assets={t:dict(urdf=d['urdf'],urdf_sha256=d['urdf_sha256']) for t,d in designs.items()}))
    shared=out/'shared_bank.json';write_new(shared,dict(kind='three_teacher_intersection_bank_v1',design_count=200,designs=rows))
    for task,spec in teachers.items():spec['design_ids']=[d[task]['id'] for _,d in pairs]
    plan=dict(kind='rough_backpack_amp_distillation_plan_v3',schema=schema,teachers=teachers,
        structure_scope='teacher_intersection_only',shared_bank=str(shared),shared_bank_sha256=sha(shared),
        selected_structure_keys=schema['shared_structure_keys'],shared_library_design_count=200,
        workspace=str(w),scope='all',task_ids={t:i for i,t in enumerate(schema['tasks'])},
        compatible_legacy_data={sha(old_path):old['schema']},teacher_updates=0,
        reference_policy='Validate exact per-structure references first; generate missing new kinematics before sampling')
    path=out/'plan.json';write_new(path,plan);load_plan(path)
    baseline=w/'rbm1/student_epoch_0018.pt';before,_=load_student(baseline)
    after=expand_mlp_tasks(baseline,schema)
    generator=torch.Generator().manual_seed(20261030);x=torch.randn(32,780,generator=generator)
    static=torch.randn(32,371,generator=generator);errors={}
    with torch.inference_mode():
        for task in old['schema']['tasks']:
            a=before(assemble(x,static,old['task_ids'][task]))
            b=after(assemble(x,static,plan['task_ids'][task],3))
            error=float((a-b).abs().max());errors[task]=error
            if error>2e-5:raise ValueError('Existing task warm-start behavior changed')
    student=out/'bootstrap.pt';save_student(student,after,plan_sha256=sha(path),
        teachers={t:s['checkpoint_sha256'] for t,s in teachers.items()},
        source_student_sha256=sha(baseline),task_expansion_parity_max_abs=errors,
        closed_loop_required=True)
    write_new(out/'warm_start_audit.json',dict(source=str(baseline),source_sha256=sha(baseline),
        student_sha256=sha(student),single_shared_network=True,old_task_parity_max_abs=errors,
        new_task_column='zero initialized; existing task columns and all other weights retained',
        optimizer='Fresh supervised Adam; no teacher or PPO updates'))
    print('THREE_TASK_PLAN',path,'shared_designs',len(pairs),'conditions',len(pairs)*3,'parity',errors,flush=True)

if __name__=='__main__':main()
