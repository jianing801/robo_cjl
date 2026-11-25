"""Queued original-RPO single-structure experts and paired comparison, GPU sequential."""
import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from singleton_native import DESIGN_KEY, TASK_BUDGETS, singleton_config, file_sha, publish

HERE=Path(__file__).resolve().parent
TASKS=tuple(TASK_BUDGETS)
METRICS=('cot','forward_rmse_mps','roll_pitch_rms_rad_s','knee_ankle_max_rated_ratio')


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def freeze(path, value):
    if path.exists():
        if read(path)!=value: raise ValueError(f'Frozen identity changed: {path}')
    else: publish(path,value)


def comparison_scores(shared, expert, normalization):
    result={}
    for label, trials in (('shared',shared),('expert',expert)):
        costs={}; rows={}
        for task in TASKS:
            trial=trials[task]; raw=trial.get('observed'); values={}; cost=None
            if trial['status']=='success':
                for name in METRICS:
                    value=raw['absolute_mechanical_cot' if name=='cot' else name]
                    denom=normalization['metrics'][task][name]['baseline']
                    if not math.isfinite(value) or value<0 or not math.isfinite(denom) or denom<=0:
                        raise ValueError('Nonfinite metric or invalid frozen denominator')
                    values[name]=value/denom
                cost=sum(values[k]*normalization['metrics'][task][k]['weight'] for k in METRICS)
            costs[task]=cost;rows[task]=dict(status=trial['status'],raw=raw,normalized=values,objective=cost)
        result[label]=rows
        result[label+'_objective']=(sum(costs[t]*normalization['task_weights'][t] for t in TASKS)
                                    if all(v is not None for v in costs.values()) else None)
    def percent(a,b): return (a/b-1)*100 if a is not None and b is not None and b>0 else None
    result['shared_vs_expert_percent']=percent(result['shared_objective'],result['expert_objective'])
    result['task_differences_percent']={t:percent(result['shared'][t]['objective'],result['expert'][t]['objective']) for t in TASKS}
    return result


def check_pause(root):
    if (root/'USER_PAUSED').exists(): raise KeyboardInterrupt('USER_PAUSED')


def status(root, phase, **data):
    publish(root/'status.json',dict(phase=phase,pid=os.getpid(),timestamp=time.time(),**data))
    print('ORIGINAL_EXPERT_COMPARISON',phase,json.dumps(data,ensure_ascii=False),flush=True)


def matching_process(pid, created):
    import psutil
    try:
        process=psutil.Process(pid)
        return process if abs(process.create_time()-created)<.05 and process.is_running() and process.status()!=psutil.STATUS_ZOMBIE else None
    except psutil.NoSuchProcess: return None


def wait_shared(root, shared, pid=None, created=None):
    status(root,'waiting_shared_1000',shared=str(shared),shared_pid=pid)
    while True:
        check_pause(root)
        if (shared/'USER_PAUSED').exists() or (shared/'tuning/USER_PAUSED').exists():
            raise KeyboardInterrupt('Existing shared training is paused')
        ready=(shared/'tuning/training_completed.json').is_file() and read(shared/'status.json').get('phase')=='complete'
        process=matching_process(pid,created) if pid is not None else None
        if ready and process is None: return
        if process is None: raise RuntimeError('Shared training stopped before validated update 1000; no experts launched')
        if str(shared) not in process.cmdline(): raise ValueError('Shared supervisor process identity changed')
        time.sleep(20)


def native_env(native, environment=None):
    return dict(environment or os.environ,PYTHONPATH=os.pathsep.join(map(str,(native/'robolab',native/'rsl_rl'))),
                PYTHONUTF8='1',PYTHONUNBUFFERED='1')


def run_child(command, logs, environment, root, *, grace=0, heartbeat=None, timeout=7*86400, cwd=None):
    """Own and monitor only this child; preserve all failed attempts."""
    import psutil
    logs.mkdir(parents=True,exist_ok=False)
    args=list(map(str,command)); began=time.monotonic()
    with (logs/'stdout.log').open('x',encoding='utf8') as out,(logs/'stderr.log').open('x',encoding='utf8') as err:
        process=subprocess.Popen(args,stdout=out,stderr=err,env=environment,cwd=cwd or HERE.parents[2])
        identity=psutil.Process(process.pid)
        publish(logs/'launch.json',dict(pid=process.pid,create_time=identity.create_time(),command=args))
        stopped=None
        try:
            while process.poll() is None:
                if (root/'USER_PAUSED').exists():
                    if stopped is None: stopped=time.monotonic()
                    if time.monotonic()-stopped>=grace: raise KeyboardInterrupt('USER_PAUSED')
                elapsed=time.monotonic()-began
                if elapsed>timeout: raise TimeoutError('Child exceeded bounded runtime')
                if heartbeat is not None and elapsed>1800:
                    age=time.time()-heartbeat.stat().st_mtime if heartbeat.exists() else elapsed
                    if age>1800: raise TimeoutError('Native worker made no update progress for 30 minutes')
                time.sleep(2)
        except BaseException:
            from pipeline import stop_child
            stop_child(process);raise
    publish(logs/'exit.json',dict(returncode=process.returncode,elapsed_seconds=time.monotonic()-began))
    check_pause(root)
    if process.returncode!=0: raise RuntimeError(f'Child failed ({process.returncode}): {logs}')


def wait_gpu(root,gpu,min_free):
    from robolab.rough_gpu import candidates,query_gpus
    while True:
        check_pause(root)
        choices=candidates(query_gpus(),os.environ,min_free_mib=min_free,max_utilization=100,requested=f'cuda:{gpu}')
        if choices: return choices
        time.sleep(30)


def latest_receipt(task_root):
    last=None
    for path in sorted(task_root.glob('attempts/*/result/checkpoint_receipts/update_*.json')):
        row=read(path)
        if last is not None and row['completed_updates']<last['completed_updates']: continue
        checkpoint=Path(row['checkpoint']).resolve()
        if not checkpoint.is_relative_to(path.parent.parent.resolve()) or not checkpoint.is_file() or file_sha(checkpoint)!=row['sha256']:
            raise ValueError(f'Committed native checkpoint missing or changed: {path}')
        if last is not None and row['completed_updates']==last['completed_updates'] and row['sha256']!=last['sha256']:
            raise ValueError('Duplicate committed update has different contents')
        last=row
    return last


def assert_no_live_workers(root):
    import psutil
    for process in psutil.process_iter(['cmdline']):
        command=process.info['cmdline'] or []
        if '--singleton-root' in command and str(root) in command:
            raise RuntimeError(f'Existing singleton worker {process.pid}; no duplicate launched')
    for path in root.rglob('launch.json'):
        row=read(path)
        if 'create_time' in row and matching_process(row['pid'],row['create_time']) is not None:
            raise RuntimeError(f'Existing comparison child {row["pid"]}; no duplicate launched')


def exact_cache_source(eval100, task, design, bank_sha):
    from bo import read_conditions,PHYSICS_CONDITIONS,check_native_measurement
    conditions=read_conditions(PHYSICS_CONDITIONS); seed=conditions['seeds'][0]
    accepted=read(eval100/'measurements'/f'{seed}_{task}'/'accepted.json')
    report_path=Path(accepted['report']); report=read(report_path)
    if file_sha(report_path)!=accepted['sha256']: raise ValueError('Source baseline measurement changed')
    check_native_measurement(report,report_path,conditions,task=task,seed=seed,design_id=design,
                             actor_sha=report['actor_sha256'],bank_sha=bank_sha,plan_sha=report['plan_sha256'])
    source=report_path.parent/'native_assets'/task/design
    audit=read(report_path.parent/'asset_cache_audit.json')
    hit=[row for row in audit if row['design']==design]
    if len(hit)!=1 or (source/'.asset_hash').read_text().strip()!=hit[0]['exact_hash']:
        raise ValueError('Previously verified singleton converted asset does not match')
    return source


def prepare_all(a,plan):
    for task in TASKS:
        spec=plan['teachers'][task]; native=a.native_checkout
        bank=Path(spec['bank']); bank_json=read(bank); singleton_config(bank_json,task)
        if file_sha(bank)!=spec['bank_sha256']: raise ValueError('Physical bank changed')
        task_root=a.output/'experts'/task
        task_root.mkdir(parents=True,exist_ok=True)
        source=exact_cache_source(a.eval100,task,spec['design_ids'][0],spec['bank_sha256'])
        command=[sys.executable,native/'robolab/experiments/multitask_morphology/singleton_native.py','prepare',
                 '--task',task,'--bank',bank,'--root',task_root,'--source-cache',source]
        if task=='run': command+=['--library',spec['reference_library']]
        run_child(command,task_root/'prepare_logs'/uuid4().hex,native_env(native),a.output,timeout=600,cwd=native/'robolab')


def verify_native(a,task,task_root,receipt):
    cmd=[sys.executable,a.native_checkout/'robolab/experiments/multitask_morphology/singleton_native.py','verify',
         '--task',task,'--plan',task_root/'plan.json','--checkpoint',receipt['checkpoint'],
         '--completed',receipt['completed_updates']]
    run_child(cmd,task_root/'verify_logs'/uuid4().hex,native_env(a.native_checkout),a.output,timeout=600,cwd=a.native_checkout/'robolab')


def summarize_training(task_root,budget):
    rows={}; attempted_seconds=0.; segments=[]
    for path in sorted(task_root.glob('attempts/*/result/native_mean_reward.csv')):
        times=0.; updates=[]
        with path.open(newline='',encoding='utf8') as stream:
            for row in csv.DictReader(stream):
                step=int(row['update']); duration=float(row['collect_seconds'])+float(row['learn_seconds'])
                times+=duration;updates.append(step);rows[step]=row
        attempted_seconds+=times
        if updates: segments.append(dict(first=updates[0],last=updates[-1],seconds=times,source=str(path)))
    if set(rows)!=set(range(1,budget+1)): raise ValueError('Expert logs do not cover all committed training updates')
    ordered=[rows[i] for i in range(1,budget+1)]
    with (task_root/'mean_reward.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(ordered[0]));writer.writeheader();writer.writerows(ordered)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));x=[i for i,row in enumerate(ordered,1) if row['mean_reward']!='']
    y=[float(ordered[i-1]['mean_reward']) for i in x];ax.plot(x,y,lw=.7);ax.set(xlabel='Native PPO update',ylabel='Mean reward')
    for seg in segments[1:]: ax.axvline(seg['first'],color='gray',ls='--',lw=.6)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(task_root/'mean_reward.png',dpi=150);plt.close(fig)
    publish(task_root/'timing.json',dict(training_seconds=attempted_seconds,
            committed_update_compute_seconds=sum(float(r['collect_seconds'])+float(r['learn_seconds']) for r in ordered),segments=segments))


def train_one(a,task,plan):
    from robolab.rough_gpu import child_environment,gpu_lease
    root=a.output; task_root=root/'experts'/task; budget=TASK_BUDGETS[task]
    if (task_root/'USER_PAUSED').exists(): raise KeyboardInterrupt('Task paused')
    spec=plan['teachers'][task]; native=a.native_checkout
    runtime=native/'robolab/experiments/multitask_morphology'/('amp_engine' if task=='run' else 'engine')
    complete=task_root/'completed.json'
    failures=0
    while True:
        check_pause(root); receipt=latest_receipt(task_root)
        if receipt:
            verify_native(a,task,task_root,receipt)
            if receipt['completed_updates']==budget:
                freeze(complete,dict(task=task,completed_updates=budget,checkpoint=receipt['checkpoint'],checkpoint_sha256=receipt['sha256'],plan_sha256=file_sha(task_root/'plan.json')))
                summarize_training(task_root,budget);return read(complete)
            if receipt['completed_updates']>budget: raise ValueError('Expert budget exceeded')
        if complete.exists(): raise ValueError('Completion receipt has no matching final checkpoint')
        if failures>=3: raise RuntimeError(f'Three failed attempts for {task}; inspect logs before resuming')
        if shutil.disk_usage(root).free<10*1024**3: raise RuntimeError('Less than 10 GiB available on experiment disk')
        status(root,'training_expert',task=task,completed_updates=receipt['completed_updates'] if receipt else 0,target=budget)
        attempt=task_root/'attempts'/f'{time.time_ns()}_{uuid4().hex[:8]}';attempt.mkdir(parents=True)
        output=attempt/'result'
        command=[sys.executable,'-B','-u',runtime/'train_stage.py','--bank',spec['bank'],'--hybrid-plan',task_root/'plan.json',
                 '--stage-index','0','--output-dir',output,'--execute','--updates',str(budget),'--num-envs','4096',
                 '--asset-cache',task_root/'cache','--singleton-root',root,'--headless','--device','cuda:0']
        if task=='run': command+=['--reference-library',spec['reference_library']]
        if receipt: command+=['--resume-checkpoint',receipt['checkpoint']]
        try:
            choices=wait_gpu(root,a.gpu,12800)
            with gpu_lease(choices,HERE.parents[1]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
                command += [f'--kit_args=--/renderer/activeGpu={gpu["index"]} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0']
                run_child(command,attempt/'logs',native_env(native,child_environment(gpu,os.environ)),root,
                          grace=60,heartbeat=output/'native_mean_reward.csv',cwd=native/'robolab')
            if not (output/'summary.json').exists(): raise RuntimeError('Native expert worker exited without summary')
            if read(output/'summary.json')['completed_updates']!=budget: raise ValueError('Native expert worker did not finish exact budget')
            final_receipt=latest_receipt(task_root)
            if not final_receipt or final_receipt['completed_updates']!=budget: raise RuntimeError('Missing final native checkpoint receipt')
        except KeyboardInterrupt: raise
        except Exception as error:
            failures+=1;status(root,'retry_expert',task=task,error=str(error),attempts=failures);time.sleep(20)
        # The next loop independently checks the committed checkpoint, not the exit message.


def evaluate_one(a,label,task,plan_path,actor):
    import bo
    from robolab.rough_gpu import child_environment,gpu_lease
    root=a.output; conditions=bo.read_conditions(bo.PHYSICS_CONDITIONS);seed=conditions['seeds'][0]
    plan=bo.load_plan(plan_path);spec=plan['teachers'][task]
    folder=root/'evaluation'/label/task;folder.mkdir(parents=True,exist_ok=True)
    accepted=folder/'accepted.json'
    actor_sha=file_sha(actor) if actor else spec['checkpoint_sha256']
    def verified(path):
        report=read(path)
        bo.check_native_measurement(report,path,conditions,task=task,seed=seed,design_id=spec['design_ids'][0],
            actor_sha=actor_sha,bank_sha=spec['bank_sha256'],plan_sha=file_sha(plan_path))
        return report['trials'][0]
    if accepted.exists():
        record=read(accepted);path=Path(record['report'])
        if file_sha(path)!=record['sha256'] or file_sha(record['video'])!=record['video_sha256']:
            raise ValueError('Accepted comparison evaluation changed')
        return verified(path),record
    for retry in range(3):
        check_pause(root);status(root,'evaluating',model=label,task=task,attempt=retry+1)
        attempt=folder/f'run_{uuid4().hex[:8]}';output=attempt/'native'
        command=[sys.executable,'-B','-u',bo.SHARED/('rollout_amp.py' if task=='run' else 'rollout.py'),
                 '--plan',plan_path,'--design-ids',spec['design_ids'][0],'--output',output,
                 '--bo-conditions',bo.PHYSICS_CONDITIONS,'--mode','evaluate','--profile','nominal',
                 '--randomization-strength','0','--steps','1000','--seed',str(seed),'--reset-yaw-range','0',
                 '--headless','--device','cuda:0','--video']
        if actor: command+=['--student',actor]
        if task!='run': command+=['--task',task]
        try:
            with gpu_lease(wait_gpu(root,a.gpu,5*1024),HERE.parents[1]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
                command += [f'--kit_args=--/renderer/activeGpu={gpu["index"]} --/physics/cudaDevice=0']
                run_child(command,attempt/'logs',child_environment(gpu,os.environ),root,timeout=1800)
            path=output/'bo_trials.json';trial=verified(path);video=output/'evaluation.mp4'
            if not video.is_file() or not (output/'summary.json').is_file(): raise RuntimeError('Missing recording or rollout summary')
            record=dict(report=str(path),sha256=file_sha(path),video=str(video),video_sha256=file_sha(video))
            freeze(accepted,record);return trial,record
        except KeyboardInterrupt: raise
        except Exception as error: status(root,'retry_evaluation',model=label,task=task,error=str(error))
    raise RuntimeError(f'No validated {label} {task} measurement and recording after three attempts')


def write_comparison(a,results,evidence,normalization):
    scores=comparison_scores(results['shared'],results['expert'],normalization)
    scores['evidence']=evidence
    scores['training_seconds']={t:read(a.output/'experts'/t/'timing.json')['training_seconds'] for t in TASKS}
    scores['normalization_sha256']=file_sha(HERE.parents[2]/'docs/reference/rpo_original_configuration/normalization.json')
    publish(a.output/'comparison.json',scores)
    with (a.output/'comparison.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['task','metric','shared_1000','expert_final','shared_vs_expert_percent'])
        for task in TASKS:
            for key in (*METRICS,'objective'):
                def value(label):
                    row=scores[label][task]
                    return row['objective'] if key=='objective' else (row['raw'] or {}).get('absolute_mechanical_cot' if key=='cot' else key)
                x,y=value('shared'),value('expert');writer.writerow([task,key,x,y,(x/y-1)*100 if x is not None and y is not None and y>0 else None])
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True);x=np.arange(4)
    for ax,task in zip(axes,TASKS):
        for index,label in enumerate(('shared','expert')):
            row=scores[label][task];values=[row['normalized'].get(k,float('nan')) for k in METRICS]
            ax.bar(x+(index-.5)*.35,values,width=.35,label='Shared 1000' if label=='shared' else 'Expert final')
        ax.set_xticks(x,['CoT','RMSE','Sway','Torque']);ax.set_title(task);ax.axhline(1,color='gray',lw=.8);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Normalized cost (lower is better)');axes[-1].legend();fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=170);plt.close(fig)


def main(argv=None):
    import bo
    from persistence import run_lock,resume_pause
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('shared','eval100','native-checkout','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--wait-pid',type=int);parser.add_argument('--wait-create-time',type=float)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume-paused',action='store_true')
    a=parser.parse_args(argv)
    for name in ('shared','eval100','native_checkout','output'): setattr(a,name,getattr(a,name).resolve())
    if (a.wait_pid is None)!=(a.wait_create_time is None): parser.error('Both shared process PID and creation time are required')
    a.output.mkdir(parents=True,exist_ok=True)
    def pause(*_): (a.output/'USER_PAUSED').touch(exist_ok=True)
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    with run_lock(a.output):
        try:
            resume_pause(a.output,a.resume_paused);check_pause(a.output)
            assert_no_live_workers(a.output)
            tuning=a.shared/'tuning'; origin=read(a.shared/'manifest.json');manifest=read(tuning/'manifest.json')
            if (origin['requested_updates']!=1000 or origin['candidate']['structure_key']!=DESIGN_KEY
                    or manifest['ppo_algorithm']!=bo.NATIVE_PPO_ALGORITHM or manifest['updates']!=1000):
                raise ValueError('Not the requested original-RPO V4 training')
            plan_path=tuning/'plan.json';plan=bo.load_plan(plan_path)
            norm_path=bo.ORIGINAL_RPO_ARCHIVE/'normalization.json';normalization=read(norm_path)
            archive=read(bo.ORIGINAL_RPO_ARCHIVE/'manifest.json')
            if file_sha(norm_path)!=archive['files_sha256']['normalization.json']: raise ValueError('Frozen normalization changed')
            commit=subprocess.check_output(['git','-C',str(a.native_checkout),'rev-parse','HEAD'],text=True).strip()
            controller_commit=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip()
            freeze(a.output/'manifest.json',dict(kind='rpo_original_singleton_expert_comparison',shared=str(a.shared),
                   shared_manifest_sha256=file_sha(tuning/'manifest.json'),candidate_plan_sha256=file_sha(plan_path),
                   native_checkout=str(a.native_checkout),native_commit=commit,budgets=TASK_BUDGETS,
                   controller_commit=controller_commit,
                   normalization_sha256=file_sha(norm_path),conditions_sha256=bo.digest(bo.read_conditions(bo.PHYSICS_CONDITIONS))))
            prepare_all(a,plan)
            if a.prepare_only: status(a.output,'prepared_no_training');return
            if (a.output/'completed.json').is_file(): status(a.output,'already_complete');return
            wait_shared(a.output,a.shared,a.wait_pid,a.wait_create_time)
            actor=bo.completed_actor_from_hashes(tuning,manifest['plan_sha256'],manifest['initial_actor_sha256'],
                   manifest['candidate'],1000,ppo_schedule=bo.PPO_SCHEDULE,mixed_bo=True,ppo_algorithm=bo.NATIVE_PPO_ALGORITHM)
            if file_sha(actor)!=read(a.shared/'status.json')['final_actor_sha256']: raise ValueError('Shared final Actor identity changed')
            experts={t:train_one(a,t,plan) for t in TASKS}
            expert_plan=deepcopy(plan)
            for task,record in experts.items():
                expert_plan['teachers'][task].update(checkpoint=record['checkpoint'],checkpoint_sha256=record['checkpoint_sha256'],expected_iter=TASK_BUDGETS[task]-1)
            for task in TASKS:
                cache=str(a.output/'experts'/task/'cache');expert_plan['teachers'][task]['asset_cache']=[cache]
            expert_plan_path=a.output/'expert_evaluation_plan.json';freeze(expert_plan_path,expert_plan)
            shared_plan=deepcopy(plan)
            for task in TASKS: shared_plan['teachers'][task]['asset_cache']=[str(a.output/'experts'/task/'cache')]
            shared_plan_path=a.output/'shared_evaluation_plan.json';freeze(shared_plan_path,shared_plan)
            results={};evidence={}
            for label,model_path,policy in (('shared',shared_plan_path,actor),('expert',expert_plan_path,None)):
                results[label]={};evidence[label]={}
                for task in TASKS: results[label][task],evidence[label][task]=evaluate_one(a,label,task,model_path,policy)
            write_comparison(a,results,evidence,normalization)
            freeze(a.output/'completed.json',dict(comparison_sha256=file_sha(a.output/'comparison.json'),shared_actor_sha256=file_sha(actor),expert_checkpoints=experts))
            status(a.output,'complete')
        except KeyboardInterrupt:
            (a.output/'USER_PAUSED').touch(exist_ok=True);status(a.output,'paused');raise SystemExit(130)
        except Exception as error:
            status(a.output,'failed',error=str(error));raise


if __name__=='__main__': main()
