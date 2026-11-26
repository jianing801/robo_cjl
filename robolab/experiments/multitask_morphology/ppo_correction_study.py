"""Sequential, isolated original-RPO PPO correction experiments."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from uuid import uuid4

HERE=Path(__file__).resolve().parent
TASKS=('stairs','backpack','run')
UPDATES=1000


def curve_assessment(rows, baseline):
    """No task can trade its decline for another's improvement.

    Operational stability criterion, not a mathematical PPO convergence proof:
    final 100 >= 95% of baseline 51:150 and own best 100-window after 150;
    last 200 absolute fitted drift and difference between its halves <= 5%.
    """
    import numpy as np
    if [int(x['update']) for x in rows]!=list(range(1,UPDATES+1)):
        raise ValueError('All 1000 ordered updates are required to assess convergence')
    results={}
    for task in TASKS:
        key=task+'_mean_reward'
        y=np.asarray([float(x[key]) for x in rows]);prior=np.asarray([float(x[key]) for x in baseline])
        if len(prior)!=UPDATES or not np.isfinite(y).all() or not np.isfinite(prior).all():
            raise ValueError('Nonfinite or incomplete task reward evidence')
        early=float(prior[50:150].mean());last=float(y[-100:].mean());previous=float(y[-200:-100].mean())
        best=float(np.convolve(y[50:],np.ones(100)/100,mode='valid').max())
        scale=max(abs(last),abs(early),1.)
        drift=float(np.polyfit(np.arange(200),y[-200:],1)[0]*199)
        checks=dict(baseline_floor=last>=early-.05*max(abs(early),1.),peak_retention=last>=best-.05*max(abs(best),1.),
                    final_window_change=abs(last-previous)<=.05*scale,final_trend=abs(drift)<=.05*scale)
        results[task]=dict(passed=all(checks.values()),checks=checks,baseline_early=early,last100=last,
                           preceding100=previous,best100=best,last200_fitted_change=drift)
    return dict(passed=all(r['passed'] for r in results.values()),tasks=results,tolerance=.05)


def choose_best(candidates, normalization):
    """Among separately accepted curves, prefer the best worst-task raw score ratio."""
    from original_expert_comparison import comparison_scores
    ranked=[]
    for candidate in candidates:
        pair=candidate['pairs'][0]
        if pair['feasible'] is not True: raise ValueError('Unaccepted correction cannot compete')
        tasks=pair['tasks'];score=comparison_scores(tasks,tasks,normalization)
        by_task={t:score['shared'][t]['objective'] for t in TASKS}
        ranked.append((max(by_task.values()),score['shared_objective'],candidate['profile'],candidate,by_task))
    if not ranked:return None
    ranked.sort(key=lambda row:row[:3]);winner=dict(ranked[0][3]);winner['selection_scores']=[dict(profile=r[2],worst_task=r[0],overall=r[1],task_objectives=r[4]) for r in ranked]
    return winner


def read_csv(path):
    with Path(path).open(encoding='utf8',newline='') as stream: return list(csv.DictReader(stream))


def measured_pairs(bo,reports,conditions):
    """Native evaluation returns flat task reports with a separate evidence field."""
    return bo.paired_trials(reports,conditions),[r['evidence'] for r in reports]


def write_diagnostics(bo,folder,tuning,previous):
    from finetune_state import checkpoint_history
    from singleton_native import publish
    rows=read_csv(tuning/'mean_reward.csv');result=curve_assessment(rows,previous)
    publish(folder/'curve_assessment.json',result)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
    for ax,key,label in zip(axes.flat,('mean_reward',*[t+'_mean_reward' for t in TASKS]),('Overall',*TASKS),strict=True):
        ax.plot([int(r['update']) for r in rows],[float(r[key]) for r in rows],lw=.8)
        ax.set_title(label);ax.set_xlabel('PPO updates');ax.set_ylabel('Mean reward');ax.grid(alpha=.2)
    fig.savefig(folder/'mean_reward.png',dpi=170);plt.close(fig)
    history=checkpoint_history(tuning,bo.sha(tuning/'manifest.json'))[1:]
    expected_weights=bo.read(tuning/'manifest.json')['native_ppo_settings'].get('actor_gradient_weights')
    if expected_weights and (len(history)!=UPDATES or any(r['metrics'].get('actor_gradient_weights')!=expected_weights for r in history)):
        raise ValueError('Weighted Actor task gradients were not used on every formal update')
    with (folder/'ppo_diagnostics.csv').open('w',newline='',encoding='utf8') as stream:
        keys=('learning_rate','mean_action_std','value','surrogate','entropy','collection_seconds','update_seconds')
        subs=('kl','clip_fraction','explained_variance','mean_action_std')
        weights=[f'{t}_actor_gradient_weight' for t in TASKS] if expected_weights else []
        writer=csv.DictWriter(stream,fieldnames=['update',*keys,*[f'{t}_{k}' for t in TASKS for k in subs],*weights])
        writer.writeheader()
        for rec in history:
            m=rec['metrics'];writer.writerow(dict(update=rec['completed_updates'],**{k:m.get(k) for k in keys},
                **{f'{t}_{k}':m['tasks'][t].get(k) for t in TASKS for k in subs},
                **{f'{t}_actor_gradient_weight':m['actor_gradient_weights'][t] for t in TASKS} if expected_weights else {}))
    publish(folder/'actor_gradient_samples.json',[dict(update=r['completed_updates'],**r['metrics']['actor_gradient_sample'])
         for r in history if r['metrics'].get('actor_gradient_sample')])
    return result


def verified_corrected_actor(tuning, origin_manifest, profile):
    """Detached experimental output, never accepted by official BO checks."""
    import bo, torch
    from finetune_settings import correction_settings,NATIVE_PPO_SETTINGS
    from finetune_state import checkpoint_history
    from native_ppo_convergence import verify_blob
    from student import load_student
    tuning=Path(tuning).resolve();manifest=bo.read(tuning/'manifest.json');digest=bo.sha(tuning/'manifest.json')
    if (manifest.get('experimental_profile')!=profile or manifest['native_ppo_settings']!=correction_settings(profile)
            or manifest['native_ppo_settings']==NATIVE_PPO_SETTINGS or manifest['updates']!=UPDATES):
        raise ValueError('Unexpected experimental training identity')
    for k in ('plan_sha256','initial_actor_sha256','candidate','seed','envs_per_task','collection_protocol','ppo_schedule','init_at_random_ep_len','teacher_checkpoint_sha256'):
        if manifest[k]!=origin_manifest[k]: raise ValueError(f'Correction changed frozen control setting {k}')
    receipt=bo.read(tuning/'training_completed.json');actor=tuning/'shared_actor_finetuned.pt'
    if (receipt['manifest_sha256']!=digest or receipt['ppo_updates']!=UPDATES or Path(receipt['student']).resolve()!=actor
            or receipt['student_sha256']!=bo.sha(actor) or receipt['candidate']!=manifest['candidate']):
        raise ValueError('Experimental final Actor receipt differs')
    history=checkpoint_history(tuning,digest)
    if len(history)!=UPDATES+1 or bo.sha(tuning/'checkpoints/update_001000.json')!=receipt['checkpoint_sha256']:
        raise ValueError('Incomplete experimental PPO commits')
    state=verify_blob(bo,tuning,history[-1],digest)
    model,_=load_student(actor,'cpu')
    if any(not torch.equal(value,state['actor'][key]) for key,value in model.state_dict().items()):
        raise ValueError('Published experimental Actor differs from final checkpoint')
    return actor


def verified_selection(selection_path, original):
    import bo
    selection=bo.read(selection_path);tuning=Path(selection['tuning']).resolve();folder=tuning.parent
    origin=bo.read(Path(original)/'tuning/manifest.json')
    actor=verified_corrected_actor(tuning,origin,selection['profile'])
    if (selection['actor_sha256']!=bo.sha(actor) or selection['manifest_sha256']!=bo.sha(tuning/'manifest.json')
            or selection['curve_assessment_sha256']!=bo.sha(folder/'curve_assessment.json')):
        raise ValueError('Selected correction identity changed')
    assessment=curve_assessment(read_csv(tuning/'mean_reward.csv'), read_csv(Path(original)/'tuning/mean_reward.csv'))
    if not assessment['passed'] or assessment!=bo.read(folder/'curve_assessment.json'):
        raise ValueError('Selected correction does not have stable results in each task')
    cond=bo.read_conditions(bo.PHYSICS_CONDITIONS);plan=bo.load_plan(tuning/'plan.json');reports=[]
    if selection['conditions_sha256']!=bo.digest(cond): raise ValueError('Correction conditions changed')
    for evidence in selection['evidence']:
        path=Path(evidence['report']);report=bo.read(path);task=report['task'];spec=plan['teachers'][task]
        if bo.sha(path)!=evidence['sha256']:raise ValueError('Selected correction measurement changed')
        bo.check_native_measurement(report,path,cond,task=task,seed=cond['seeds'][0],design_id=spec['design_ids'][0],
                                   actor_sha=bo.sha(actor),bank_sha=spec['bank_sha256'],plan_sha=bo.sha(tuning/'plan.json'))
        reports.append(report)
    pairs=bo.paired_trials(reports,cond)
    if pairs!=selection['pairs'] or len(pairs)!=1 or not pairs[0]['feasible']:
        raise ValueError('Selected correction did not pass all fixed-condition gait trials')
    return actor


def run_profile(a,bo,profile,source,initial,origin,previous):
    from original_expert_comparison import run_child,wait_gpu,status,check_pause
    from robolab.rough_gpu import gpu_lease,child_environment
    from singleton_native import publish
    folder=a.output/profile;folder.mkdir(exist_ok=True);tuning=folder/'tuning'
    for retry in range(3):
        check_pause(a.output)
        if (tuning/'USER_PAUSED').exists(): raise KeyboardInterrupt('Correction worker was paused')
        if (tuning/'training_completed.json').is_file():break
        status(a.output,'training_correction',profile=profile,target=UPDATES,attempt=retry+1)
        cmd=[sys.executable,'-B','-u',bo.SHARED/'mixed_finetune.py','--plan',source,'--student',initial,
             '--output',tuning,'--design-id',origin['candidate']['design_ids']['stairs'],'--updates',str(UPDATES),
             '--num-envs','4096','--seed',str(origin['seed']),'--profile','robust','--randomization-strength','1',
             '--init-at-random-ep-len','--headless','--device','cuda:0','--experimental-profile',profile]
        if (tuning/'manifest.json').exists():cmd.append('--resume')
        try:
            with gpu_lease(wait_gpu(a.output,a.gpu,12800),bo.REPO/'robolab/logs/rpo_rough_multimorph/gpu_locks') as gpu:
                cmd += [f'--kit_args=--/renderer/activeGpu={gpu["index"]} --/physics/cudaDevice=0']
                run_child(cmd,folder/'training_logs'/uuid4().hex,child_environment(gpu,os.environ),a.output,
                          grace=15,heartbeat=tuning/'mean_reward.csv',timeout=12*3600)
            if not (tuning/'training_completed.json').exists():raise RuntimeError('Training worker did not publish its final actor')
        except KeyboardInterrupt:raise
        except Exception as error:
            status(a.output,'retry_correction',profile=profile,error=str(error));time.sleep(20)
    actor=verified_corrected_actor(tuning,origin,profile)
    assessment=write_diagnostics(bo,folder,tuning,previous)
    if not assessment['passed']:
        status(a.output,'curve_rejected',profile=profile,tasks=assessment['tasks']);return None
    # Only a separately stable curve reaches fixed-condition real gait tests.
    cond=bo.read_conditions(bo.PHYSICS_CONDITIONS);plan=bo.load_plan(tuning/'plan.json')
    status(a.output,'evaluating_correction',profile=profile)
    for retry in range(3):
        try:
            measured=bo.evaluate_native(folder/'evaluation',tuning/'plan.json',plan,actor,bo.PHYSICS_CONDITIONS,cond,a.gpu,1800,pause_root=a.output)
            break
        except KeyboardInterrupt:raise
        except Exception as error:
            status(a.output,'retry_correction_evaluation',profile=profile,error=str(error))
            if retry==2: raise
            time.sleep(20)
    pairs,evidence=measured_pairs(bo,measured,cond)
    if len(pairs)!=1 or pairs[0]['feasible'] is not True:
        publish(folder/'gait_rejected.json',dict(pairs=pairs,evidence=evidence));return None
    result=dict(profile=profile,tuning=str(tuning),actor=str(actor),actor_sha256=bo.sha(actor),
                curve_assessment_sha256=bo.sha(folder/'curve_assessment.json'),
                manifest_sha256=bo.sha(tuning/'manifest.json'),conditions_sha256=bo.digest(cond),
                pairs=pairs,evidence=evidence)
    publish(folder/'accepted.json',result)
    return result


def record_expert_resumed(root,expert_pid,current):
    from original_expert_comparison import status
    status(root,'expert_resumed',expert_pid=expert_pid,task=current.get('task'),
           checkpoint_update=current.get('completed_updates'))


def resume_experts(a):
    """The user authorized one conditional resume after a validated correction."""
    import subprocess,psutil
    from original_expert_comparison import status,check_pause,matching_process
    from singleton_native import publish
    import bo
    selection=a.output/'selected.json';verified_selection(selection,a.original)
    check_pause(a.output)
    handoff=a.output/'expert_handoff.json'
    if handoff.exists():
        previous=bo.read(handoff)
        process=matching_process(previous['pid'],previous['create_time'])
        if process is not None:
            current=bo.read(a.expert_root/'status.json')
            if (current.get('pid')!=previous['pid'] or str(selection) not in process.cmdline()
                    or (a.expert_root/'USER_PAUSED').exists()):
                raise RuntimeError('Existing expert handoff identity or pause state changed')
            record_expert_resumed(a.output,previous['pid'],current);return
        raise RuntimeError('Expert handoff already attempted; inspect its status rather than duplicating it')
    if not (a.expert_root/'USER_PAUSED').exists():
        raise RuntimeError('Expected the preserved expert pause marker before the conditional handoff')
    logs=a.output/'expert_handoff_logs';logs.mkdir()
    command=[sys.executable,'-B','-u',str(HERE/'original_expert_comparison.py'),'--shared',str(a.original),
             '--eval100',str(a.eval100),'--native-checkout',str(a.native_checkout),'--output',str(a.expert_root),
             '--gpu',str(a.gpu),'--resume-paused','--corrected-selection',str(selection)]
    with (logs/'stdout.log').open('x') as out,(logs/'stderr.log').open('x') as err:
        proc=subprocess.Popen(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
    publish(handoff,dict(pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),command=command))
    for _ in range(120):
        check_pause(a.output)
        if proc.poll() is not None:raise RuntimeError('Conditional expert resume exited; inspect handoff logs')
        current=bo.read(a.expert_root/'status.json')
        if current.get('pid')==proc.pid and current.get('phase')=='training_expert':
            record_expert_resumed(a.output,proc.pid,current);return
        time.sleep(2)
    raise RuntimeError('Expert resume has not reached its training launch state')


def main():
    import bo
    from original_expert_comparison import status,check_pause
    from persistence import run_lock,resume_pause
    from singleton_native import publish
    from finetune_settings import CORRECTION_PROFILES, TRAINING_PROFILES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--profiles',nargs='+',choices=TRAINING_PROFILES[1:],default=list(CORRECTION_PROFILES[1:]))
    p.add_argument('--expert-root',type=Path);p.add_argument('--eval100',type=Path);p.add_argument('--native-checkout',type=Path)
    p.add_argument('--resume-paused',action='store_true');a=p.parse_args()
    a.original=a.original.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if any((a.expert_root,a.eval100,a.native_checkout)) and not all((a.expert_root,a.eval100,a.native_checkout)):
        p.error('Conditional expert handoff requires all three preserved paths')
    def pause(*_): (a.output/'USER_PAUSED').touch(exist_ok=True)
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    with run_lock(a.output):
        try:
            resume_pause(a.output,a.resume_paused);check_pause(a.output)
            origin=bo.read(a.original/'tuning/manifest.json');previous=read_csv(a.original/'tuning/mean_reward.csv')
            provenance=bo.read(bo.ORIGINAL_RPO_ARCHIVE/'source_legacy_manifest.json')
            source=Path(provenance['source_plan']);initial=Path(provenance['initial_actor'])
            if bo.sha(source)!=origin['plan_sha256'] or bo.sha(initial)!=origin['initial_actor_sha256'] or origin['updates']!=UPDATES:
                raise ValueError('Correction control starting policy or original structure plan changed')
            bo.freeze(a.output/'manifest.json',dict(original=str(a.original),original_manifest_sha256=bo.sha(a.original/'tuning/manifest.json'),
                        source_plan_sha256=bo.sha(source),initial_actor_sha256=bo.sha(initial),profiles=a.profiles,updates=UPDATES))
            accepted=[]
            for profile in a.profiles:
                result=run_profile(a,bo,profile,source,initial,origin,previous)
                if result is not None:
                    accepted.append(result)
                    status(a.output,'profile_accepted',profile=profile,actor_sha256=result['actor_sha256'])
            result=choose_best(accepted,bo.read(bo.ORIGINAL_RPO_ARCHIVE/'normalization.json'))
            if result is not None:
                publish(a.output/'selected.json',result);status(a.output,'accepted',profile=result['profile'],actor_sha256=result['actor_sha256'])
                if a.expert_root is not None:resume_experts(a)
                return
            status(a.output,'next_correction_needed')
        except KeyboardInterrupt:
            (a.output/'USER_PAUSED').touch(exist_ok=True);status(a.output,'paused');raise SystemExit(130)
        except Exception as e:
            status(a.output,'failed',error=str(e));raise


if __name__=='__main__':main()
