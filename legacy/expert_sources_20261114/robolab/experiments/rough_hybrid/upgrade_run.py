"""Explicit, read-only-old-run upgrade from reviewed b313646 to persistent Rough.

Creates a NEW output and a metadata-rebound checkpoint. No training is started.
"""
import argparse
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]


def upgrade(old_root, bank_path, output_root):
    import torch
    from hybrid_contract import experiment_identity,validate_config,validate_checkpoint,sha,stage_spec
    from hybrid_layout import identity
    from hybrid_bank import load_configurable_bank
    from hybrid_support import write_new
    from hybrid_processes import reject_duplicate_bank_safely
    from train import committed
    old_root=Path(old_root).resolve();bank_path=Path(bank_path).resolve();output_root=Path(output_root).resolve()
    if output_root.exists() or old_root in output_root.parents:
        raise ValueError('upgrade requires a separate NEW output outside old run')
    if (old_root/'USER_PAUSED').exists():raise ValueError('old run is USER_PAUSED; explicit review required before upgrade')
    reject_duplicate_bank_safely(bank_path)
    old=json.loads((old_root/'plan.json').read_text(encoding='utf8'))
    if old.get('execution_protocol')!='direct_stage_v2' or old.get('start_stage',0)!=0:
        raise ValueError('only reviewed original per-group v2 run is supported')
    bank=load_configurable_bank(bank_path);config=old['configuration'];validate_config(config,bank)
    new_identity=experiment_identity(bank_path,config)
    legacy=old['experiment']
    allowed=json.loads((HERE/'legacy_sources.json').read_text())
    if legacy.get('hybrid_sources') not in allowed.values():
        raise ValueError('legacy sources not an exact reviewed b313646 manifest; no automatic migration')
    if {k:v for k,v in legacy.items() if k!='hybrid_sources'} != {k:v for k,v in new_identity.items() if k!='hybrid_sources'}:
        raise ValueError('bank/native source/config/physics identity changed; migration forbidden')
    rows=committed(old_root,old,bank)
    if not rows:raise ValueError('no fully committed old group; no scratch fallback')
    last=json.loads(rows[-1].read_text());stage=last['stage'];completed=stage['end_update']
    if completed>=config['rotation']['total_updates']:raise ValueError('old training already complete')
    checkpoint=Path(last['checkpoint'])
    if not checkpoint.is_file() or checkpoint.stat().st_size==0:raise ValueError('missing/empty committed checkpoint')
    before=(checkpoint.stat().st_size,sha(checkpoint),[sha(x) for x in rows])
    time.sleep(30)
    if before!=(checkpoint.stat().st_size,sha(checkpoint),[sha(x) for x in rows]) or rows!=sorted((old_root/'commits').glob('stage_*.json')):
        raise ValueError('old checkpoint/commits changed during stability window')
    if before[1]!=last['checkpoint_sha256']:raise ValueError('old checkpoint SHA mismatch')
    state=validate_checkpoint(checkpoint,legacy,completed)
    if state['hybrid_stage']!=stage:raise ValueError('checkpoint group differs from committed schedule')
    # All tensor values and all training states remain untouched. Only two
    # explicit provenance bindings change; originals remain at the old path.
    state['hybrid_identity']=new_identity
    state['model_state_dict']['_extra_state']['config']['spatial_identity']=identity(new_identity)
    output_root.mkdir(parents=True,exist_ok=False)
    (output_root/'commits').mkdir();(output_root/'attempts').mkdir()
    rebound=output_root/'bootstrap.pt'
    with rebound.open('xb') as stream:torch.save(state,stream)
    verified=validate_checkpoint(rebound,new_identity,completed)
    original=validate_checkpoint(checkpoint,legacy,completed)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    for key in original:
        if key in ('hybrid_identity','model_state_dict'):continue
        if not equal(original[key],verified[key]):raise ValueError(f'migration altered training state {key}')
    for key in original['model_state_dict']:
        if key=='_extra_state':continue
        if not equal(original['model_state_dict'][key],verified['model_state_dict'][key]):
            raise ValueError(f'migration altered policy/normalization tensor {key}')
    old_extra=original['model_state_dict']['_extra_state']['config'].copy()
    new_extra=verified['model_state_dict']['_extra_state']['config'].copy()
    old_extra.pop('spatial_identity');new_extra.pop('spatial_identity')
    if old_extra!=new_extra:raise ValueError('migration altered context/normalization schema')
    audit=dict(old_root=str(old_root),old_plan_sha256=sha(old_root/'plan.json'),
               old_checkpoint=str(checkpoint),old_checkpoint_sha256=before[1],
               bootstrap_sha256=sha(rebound),old_experiment=legacy,
               completed_updates=completed,next_stage=stage['index']+1 if 'index' in stage else len(rows),
               exact_policy_normalization_adam_lr_rng=True,
               changed_metadata_only=['hybrid_identity','model_state_dict._extra_state.config.spatial_identity'])
    write_new(output_root/'upgrade_audit.json',audit)
    write_new(output_root/'plan.json',dict(configuration=config,experiment=new_identity,bank=str(bank_path),
              preflight_updates_per_new_layout=0,execution_protocol='persistent_v1',total_updates_are_absolute=True,
              start_stage=len(rows),bootstrap_checkpoint=str(rebound),bootstrap_sha256=sha(rebound),
              upgrade_audit_sha256=sha(output_root/'upgrade_audit.json')))
    print('UPGRADED_NO_TRAINING_STARTED',str(output_root),'next_update',completed,flush=True)
    return output_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root',required=True);p.add_argument('--bank',required=True);p.add_argument('--output-root',required=True)
    a=p.parse_args();upgrade(a.old_root,a.bank,a.output_root)


if __name__=='__main__':main()
