"""Bind two existing frozen teachers; no expert training or physical asset generation."""
import argparse
import json
from pathlib import Path
import torch
from contract import sha,write_new,schema_for,paired_structures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--scope',choices=['smoke','pilot','all'],default='all')
    a=p.parse_args();w=Path(a.workspace).resolve()
    specifications={
        'stairs':dict(code_root=w/'rough_training_source_recovery_20261029/exact_source_snapshot_20261029/robolab',
            bank=w/'rough_motor_soft_20261026/bank/bank.json',
            original_bank=w/'rough_training_source_recovery_20261029/bank.json',
            checkpoint=w/'rough_final_import_20261028/model_11999.pt',
            checkpoint_sha256='a0cd01d59c47588eb23a52b440beac41d626f7b4a85473fb7135743930face26',
            expected_iter=11999,asset_cache=[w/'rough_final_import_20261028',w/'rough_pretrained_bo_pilot_exactsource_20261029',
                w/'rough_motor_soft_20261026',w/'amp_real_smoke_20261028/run_v2/attempts']),
        'backpack':dict(code_root=w/'bpcode5/robolab',bank=w/'bp26/bank/bank.json',
            original_bank=w/'bp26/bank/bank.json',
            checkpoint=w/'bp26/train_resume_20261024_232253_425700/models/model_9000.pt',
            checkpoint_sha256='2ca69637be2ab8d10d711dfa5df1ddef5651d7b7a239987a255072fabc02ab13',
            expected_iter=9000,asset_cache=[w/'bpiso'])}
    banks={task:json.loads(spec['bank'].read_text()) for task,spec in specifications.items()}
    schema=schema_for(banks)
    pairs=paired_structures(banks)
    schema['shared_structure_keys']=[list(k) for k,_ in pairs]
    chosen=pairs
    if a.scope=='pilot':
        thighs=sorted({k[0] for k,_ in pairs});calves=sorted({k[1] for k,_ in pairs})
        diagonal={(thighs[i],calves[i]) for i in (0,(len(thighs)-1)//2,len(thighs)-1)}
        chosen=[(k,rows) for k,rows in pairs if k[:2] in diagonal]
    elif a.scope=='smoke':chosen=[pairs[0],pairs[-1]]
    for task,spec in specifications.items():
        if sha(spec['checkpoint'])!=spec['checkpoint_sha256']:raise ValueError('Wrong real teacher')
        state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        config=state['model_state_dict']['_extra_state']['config']
        if state['iter']!=spec['expected_iter'] or config['role']!=task:raise ValueError('Wrong teacher role/update')
        if config['bank_sha256']!=sha(spec['original_bank']):raise ValueError('Original teacher bank mismatch')
        original=json.loads(spec['original_bank'].read_text())
        by_id={d['id']:d for d in original['designs']}
        for d in banks[task]['designs']:
            other=by_id[d['id']]
            for field in ('thigh_m','calf_m','knee_motor','ankle_motor','context'):
                if d[field]!=other[field]:raise ValueError('Local teacher asset binding changed')
            if abs(d['total_mass_kg']-other['total_mass_kg'])>1e-6:raise ValueError('Local base mass mismatch')
        if banks[task]['canonical_semantic_contract']!=original['canonical_semantic_contract']:
            raise ValueError('Original physical joint contract mismatch')
        spec.update(bank_sha256=sha(spec['bank']),context_width=banks[task]['context_width'],
            full_design_count=len(banks[task]['designs']),copies=4 if task=='stairs' else 8,
            original_urdf_identity_boundary='Rough local regenerated assets match design/context/base mass; original server URDF bytes were not supplied' if task=='stairs' else 'Original frozen backpack assets')
        rows=[rows[task] for _,rows in chosen]
        spec['design_ids']=[d['id'] for d in rows]
        for key,value in list(spec.items()):
            if isinstance(value,Path):spec[key]=str(value)
            elif isinstance(value,list) and value and isinstance(value[0],Path):spec[key]=[str(v) for v in value]
    output=Path(a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    shared_bank=output.with_name(output.stem+'_shared_bank.json')
    shared_rows=[]
    for key,rows in pairs:
        identity='shared_'+__import__('hashlib').sha256(json.dumps(key).encode()).hexdigest()[:12]
        shared_rows.append(dict(id=identity,thigh_m=key[0],calf_m=key[1],knee_motor=key[2],ankle_motor=key[3],
            native_design_ids={t:d['id'] for t,d in rows.items()},
            physical_assets={t:dict(urdf=d['urdf'],urdf_sha256=d['urdf_sha256']) for t,d in rows.items()}))
    write_new(shared_bank,dict(kind='rough_backpack_shared_intersection_bank_v1',schema=schema,
        design_count=len(shared_rows),designs=shared_rows,
        definition='Exact same thigh/calf lengths and exact knee/ankle motor IDs in BOTH teachers',
        task_physics='Keep each original asset; backpack link/runtime load remains task-specific'))
    plan=dict(kind='rough_backpack_distillation_plan_v2',schema=schema,teachers=specifications,
        structure_scope='teacher_intersection_only',shared_bank=str(shared_bank),shared_bank_sha256=sha(shared_bank),
        selected_structure_keys=[list(k) for k,_ in chosen],shared_library_design_count=len(pairs),
        workspace=str(w),scope=a.scope,task_ids={'stairs':0,'backpack':1},
        earlier_method='fixed-RPO MxT BC then real student roll-in DAgger; no old fixed-RUN weights imported',
        teacher_updates=0,task_and_design_balanced=True,
        validation='Independent physical environment streams; never random-split adjacent frames',
        sensing='780 original proprioceptive history + task ID + known geometry/motor configuration; no actual payload/terrain scan/true velocity in actor')
    write_new(output,plan)
    print(json.dumps({t:dict(designs=len(s['design_ids']),teacher_iteration=s['expected_iter']) for t,s in specifications.items()},indent=2))
    print('DISTILLATION_PLAN',output,flush=True)


if __name__=='__main__':main()
