"""Isolated 200-update scheme-B adaptive-KL pilot; never changes official BO."""
import argparse
import math
import os
from pathlib import Path
import signal
from statistics import fmean
import sys
import time
from uuid import uuid4

from ppo_correction_study import read_csv, verified_corrected_actor, measured_pairs

PILOT_UPDATES = 200
TASKS = ('stairs', 'backpack', 'run')


def pilot_assessment(rows, comparison):
    if ([int(r['update']) for r in rows] != list(range(1, PILOT_UPDATES+1))
            or [int(r['update']) for r in comparison] != list(range(1, 1001))):
        raise ValueError('Expected complete, ordered 200/1000-update series')
    def mean(data, task, lo, hi):
        values=[float(r[f'{task}_mean_reward']) for r in data[lo-1:hi]]
        if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite reward evidence')
        return fmean(values)
    tasks={}
    for task in TASKS:
        early=mean(rows,task,51,100); late=mean(rows,task,151,200)
        b_early=mean(comparison,task,51,100); b_late=mean(comparison,task,151,200)
        b_final=mean(comparison,task,901,1000)
        tasks[task]=dict(first_late50=early,last50=late,b_first_late50=b_early,b_same_last50=b_late,
                         b_final100=b_final,early_gain=early-b_early,late_gain=late-b_late,
                         late_vs_early=late-early,retains_b_same_window=late>=b_late-.05*max(1,abs(b_late)))
    s=tasks['stairs']; floor=.95*s['b_final100']
    material_gain=(s['early_gain']>=.05*max(1,abs(s['b_first_late50'])) or
                   s['late_gain']>=.05*max(1,abs(s['b_same_last50'])))
    preserves=all(tasks[t]['retains_b_same_window'] for t in TASKS)
    early_plateau=(s['first_late50']>=floor and s['last50']>=floor and
                   abs(s['late_vs_early'])<=.05*max(1,abs(s['first_late50'])))
    return dict(tasks=tasks,material_stairs_gain=material_gain,other_tasks_preserved=preserves,
                early_plateau_candidate=early_plateau,promising=material_gain and preserves,
                note='A 200-update pilot cannot prove long-horizon convergence; raw rewards are unchanged.')


def plot_curves(path,rows,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
    for ax,key,title in zip(axes.flat,('mean_reward',*[t+'_mean_reward' for t in TASKS]),('Overall',*TASKS),strict=True):
        ax.plot(range(1,201),[float(r[key]) for r in comparison[:200]],label='B: target KL 0.01',lw=.9,color='#777777')
        ax.plot(range(1,201),[float(r[key]) for r in rows],label='B: target KL 0.02',lw=1.,color='#0072b2')
        ax.set_title(title);ax.set_xlabel('PPO updates');ax.set_ylabel('Mean reward');ax.grid(alpha=.2);ax.legend()
    fig.savefig(path,dpi=170);plt.close(fig)


def main():
    import bo
    from finetune_settings import KL_PILOT_PROFILE, correction_settings
    from finetune_state import checkpoint_history
    from original_expert_comparison import run_child, wait_gpu, check_pause, status
    from robolab.rough_gpu import gpu_lease, child_environment
    from singleton_native import publish
    from persistence import run_lock
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True,help='Completed scheme B tuning directory')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();a.original=a.original.resolve();a.comparison=a.comparison.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True);tuning=a.output/'tuning'
    def pause(*_): (a.output/'USER_PAUSED').touch(exist_ok=True)
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    with run_lock(a.output):
        try:
            check_pause(a.output)
            origin=bo.read(a.original/'tuning/manifest.json')
            verified_corrected_actor(a.comparison,origin,'fixed_amp_std_pcgrad')
            comparison=read_csv(a.comparison/'mean_reward.csv')
            provenance=bo.read(bo.ORIGINAL_RPO_ARCHIVE/'source_legacy_manifest.json')
            source=Path(provenance['source_plan']);initial=Path(provenance['initial_actor'])
            if bo.sha(source)!=origin['plan_sha256'] or bo.sha(initial)!=origin['initial_actor_sha256']:
                raise ValueError('Original source or initial Actor changed')
            bo.freeze(a.output/'manifest.json',dict(original=str(a.original),comparison=str(a.comparison),
                source_plan_sha256=bo.sha(source),initial_actor_sha256=bo.sha(initial),
                comparison_manifest_sha256=bo.sha(a.comparison/'manifest.json'),
                profile=KL_PILOT_PROFILE,settings=correction_settings(KL_PILOT_PROFILE),updates=PILOT_UPDATES))
            for retry in range(3):
                check_pause(a.output);check_pause(tuning)
                if (tuning/'training_completed.json').is_file():break
                status(a.output,'training_kl_pilot',target=PILOT_UPDATES,attempt=retry+1)
                cmd=[sys.executable,'-B','-u',bo.SHARED/'mixed_finetune.py','--plan',source,'--student',initial,
                     '--output',tuning,'--design-id',origin['candidate']['design_ids']['stairs'],
                     '--updates',str(PILOT_UPDATES),'--num-envs','4096','--seed',str(origin['seed']),
                     '--profile','robust','--randomization-strength','1','--init-at-random-ep-len','--headless',
                     '--device','cuda:0','--experimental-profile',KL_PILOT_PROFILE]
                if (tuning/'manifest.json').is_file():cmd.append('--resume')
                try:
                    with gpu_lease(wait_gpu(a.output,a.gpu,12800),bo.REPO/'robolab/logs/rpo_rough_multimorph/gpu_locks') as gpu:
                        cmd += [f'--kit_args=--/renderer/activeGpu={gpu["index"]} --/physics/cudaDevice=0']
                        run_child(cmd,a.output/'training_logs'/uuid4().hex,child_environment(gpu,os.environ),
                                  a.output,grace=15,heartbeat=tuning/'mean_reward.csv',timeout=6*3600)
                    if not (tuning/'training_completed.json').exists():raise RuntimeError('Missing final pilot Actor')
                except KeyboardInterrupt:raise
                except Exception as error:
                    status(a.output,'retry_kl_pilot',error=str(error),attempt=retry+1);time.sleep(20)
            actor=verified_corrected_actor(tuning,origin,KL_PILOT_PROFILE,updates=PILOT_UPDATES)
            history=checkpoint_history(tuning,bo.sha(tuning/'manifest.json'))[1:]
            if len(history)!=PILOT_UPDATES or any(r['metrics'].get('desired_kl')!=.02 for r in history):
                raise ValueError('KL target differs from the pilot at one or more updates')
            result=pilot_assessment(read_csv(tuning/'mean_reward.csv'),comparison)
            result.update(actor_sha256=bo.sha(actor),manifest_sha256=bo.sha(tuning/'manifest.json'))
            plot_curves(a.output/'mean_reward.png',read_csv(tuning/'mean_reward.csv'),comparison)
            if result['promising']:
                status(a.output,'evaluating_kl_pilot')
                cond=bo.read_conditions(bo.PHYSICS_CONDITIONS);plan=bo.load_plan(tuning/'plan.json')
                reports=bo.evaluate_native(a.output/'evaluation',tuning/'plan.json',plan,actor,
                    bo.PHYSICS_CONDITIONS,cond,a.gpu,1800,pause_root=a.output)
                result['pairs'],result['evidence']=measured_pairs(bo,reports,cond)
            publish(a.output/'pilot_assessment.json',result)
            status(a.output,'complete',promising=result['promising'],early_plateau_candidate=result['early_plateau_candidate'])
        except KeyboardInterrupt:
            (a.output/'USER_PAUSED').touch(exist_ok=True);status(a.output,'paused');raise SystemExit(130)
        except Exception as error:
            status(a.output,'failed',error=str(error));raise


if __name__=='__main__': main()
