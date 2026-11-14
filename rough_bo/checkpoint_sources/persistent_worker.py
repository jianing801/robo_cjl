"""Formal persistent Rough worker. Launched only by the locked GPU controller."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
import time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]


def run_groups(root, bank_path, result, kit_args='', asset_cache=None, stage_main=None):
    from hybrid_contract import load_plan,stage_spec,validate_checkpoint,sha
    from hybrid_bank import load_configurable_bank
    from hybrid_support import write_new
    from train import committed
    plan=load_plan(root/'plan.json',bank_path)
    if plan.get('execution_protocol')!='persistent_v1':raise ValueError('persistent plan required')
    bank=load_configurable_bank(bank_path)
    rows=committed(root,plan,bank)
    index=plan.get('start_stage',0)+len(rows)
    previous=json.loads(rows[-1].read_text())['checkpoint'] if rows else plan.get('bootstrap_checkpoint')
    if stage_main is None:
        from train_stage import main as stage_main
    session={}
    failed=False
    result.mkdir(parents=True,exist_ok=False)
    try:
        while index*plan['configuration']['rotation']['updates_per_stage']<plan['configuration']['rotation']['total_updates']:
            if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED; no next group')
            stage=stage_spec(bank,plan['configuration'],index)
            if previous:validate_checkpoint(previous,plan['experiment'],stage['first_update'])
            output=root/'attempts'/f'group_{index:06d}_{datetime.now():%Y%m%d_%H%M%S_%f}'/'result'
            args=['--bank',str(bank_path),'--hybrid-plan',str(root/'plan.json'),'--stage-index',str(index),
                  '--output-dir',str(output),'--execute','--updates',str(stage['end_update']),
                  '--headless','--device','cuda:0','--kit_args',kit_args,
                  '--asset-cache',str(asset_cache or root/'attempts')]
            if not asset_cache:args+=['--allow-cache-misses']
            if previous:args+=['--resume-checkpoint',str(previous)]
            before=None
            if session.get('runner'):
                r=session['runner'];before=(id(r.alg.policy),id(r.alg.optimizer),id(r.alg.storage))
            print(f'GROUP {index+1}: global updates {stage["first_update"]}..{stage["end_update"]-1}; persistent PID={os.getpid()}',flush=True)
            original=sys.argv
            try:
                sys.argv=[str(HERE/'train_stage.py'),*args]
                stage_main(args,session=session)
            finally:sys.argv=original
            if (output/'failure.json').exists():raise RuntimeError(f'group failure: {output}')
            summary=json.loads((output/'summary.json').read_text())
            if summary['result']!='HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED' or summary['completed_updates']!=stage['end_update']:
                raise ValueError('group did not complete exact target')
            r=session['runner']
            objects=(id(r.alg.policy),id(r.alg.optimizer),id(r.alg.storage))
            if before is not None and before!=objects:raise ValueError('policy/Adam/rollout allocations replaced')
            checkpoint=Path(r.logger.log_dir)/f'model_{stage["end_update"]-1}.pt'
            checkpoint_state=validate_checkpoint(checkpoint,plan['experiment'],stage['end_update'])
            if checkpoint_state['hybrid_stage']!=stage:raise ValueError('checkpoint group differs from current schedule')
            import torch
            memory = dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                          peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else {}
            write_new(root/'commits'/f'stage_{index:06d}.json',dict(stage=stage,checkpoint=str(checkpoint),
                      checkpoint_sha256=sha(checkpoint),summary=str(output/'summary.json'),pid=os.getpid(),objects=objects,memory=memory))
            print('STAGE_COMMITTED',index,stage['end_update'],flush=True)
            r.current_learning_iteration=stage['end_update']
            previous=str(checkpoint);index+=1
        write_new(result/'summary.json',dict(result='PERSISTENT_ROUGH_COMPLETE_NOT_GAIT_VALIDATED',
                  completed_updates=plan['configuration']['rotation']['total_updates'],checkpoint=previous))
    except BaseException:
        failed=True
        write_new(result/'failure.json',dict(traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
    finally:
        if session.get('app'):
            if failed:
                import atexit
                atexit.register(lambda:os._exit(1))
            session['app'].close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--root',required=True)
    p.add_argument('--result',required=True);p.add_argument('--kit-args',default='')
    p.add_argument('--asset-cache')
    a=p.parse_args()
    run_groups(Path(a.root).resolve(),Path(a.bank).resolve(),Path(a.result).resolve(),a.kit_args,a.asset_cache)


if __name__=='__main__':main()
