"""Bounded serial real-data collection, balanced distillation and paired evaluation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[2]),str(HERE)]
from contract import load_plan,sha,write_new


def stop_child(process):
    import psutil
    try:
        parent=psutil.Process(process.pid);children=parent.children(recursive=True)
    except psutil.NoSuchProcess:return
    for item in [parent,*children]:
        try:item.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs([parent,*children],timeout=5)
    for item in alive:
        try:item.kill()
        except psutil.NoSuchProcess:pass


def child(command,log_dir,environment,timeout,root):
    log_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with (log_dir/'stdout.log').open('x',encoding='utf8') as stdout,(log_dir/'stderr.log').open('x',encoding='utf8') as stderr:
        kwargs=dict(stdout=stdout,stderr=stderr,cwd=HERE.parents[3],env=dict(environment,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
        if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
        process=subprocess.Popen(command,**kwargs)
        write_new(log_dir/'launch.json',dict(pid=process.pid,command=command,started=datetime.now().isoformat()))
        try:
            while process.poll() is None:
                if (root/'USER_PAUSED').exists():raise KeyboardInterrupt('USER_PAUSED')
                if time.monotonic()-started>timeout:raise TimeoutError(f'Bounded data/BC job timed out: {log_dir}')
                time.sleep(1)
        except BaseException:
            (root/'USER_PAUSED').touch() if isinstance(sys.exc_info()[1],KeyboardInterrupt) else None
            stop_child(process)
            raise
    write_new(log_dir/'exit.json',dict(returncode=process.returncode,seconds=time.monotonic()-started))
    if process.returncode:raise RuntimeError(f'Child failed {process.returncode}: {log_dir}')


def jobs_for(plan,batch_size):
    result=[]
    for task,spec in plan['teachers'].items():
        ids=spec['design_ids']
        for offset in range(0,len(ids),batch_size):result.append((task,ids[offset:offset+batch_size]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-designs',type=int,default=20)
    p.add_argument('--collect-steps',type=int,default=1000);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--steps-per-epoch',type=int,default=0,
        help='0 scales optimization to real task data size; positive values are explicit update caps')
    p.add_argument('--max-rows-per-design-shard',type=int,default=32,help='Bound CPU replay RAM per design/source/split; 0 keeps every row')
    p.add_argument('--dagger-rounds',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--device')
    p.add_argument('--gpu-min-free-gb',type=float,default=4.)
    p.add_argument('--gpu-max-utilization',type=int,default=30)
    p.add_argument('--timeout-seconds',type=int,default=900)
    a=p.parse_args()
    if min(a.batch_designs,a.collect_steps,a.stride,a.epochs,a.timeout_seconds)<1 or a.steps_per_epoch<0 or a.dagger_rounds<0 or a.max_rows_per_design_shard<0:
        raise ValueError('Positive finite collection/BC budgets required')
    plan=load_plan(a.plan);root=Path(a.output).resolve()
    if a.resume:
        previous=json.loads((root/'manifest.json').read_text())
        expected={k:v for k,v in vars(a).items() if k!='resume'}
        if previous['arguments']!=expected or previous['plan_sha256']!=sha(a.plan):raise ValueError('Resume plan/budget changed')
    else:
        root.mkdir(parents=True,exist_ok=False)
        write_new(root/'manifest.json',dict(plan_sha256=sha(a.plan),arguments={k:v for k,v in vars(a).items() if k!='resume'},
            teacher_updates=0,initial_collection='teacher actions',later_collection='student actions with frozen teacher labels'))
    if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED; no automatic restart')
    if (root/'completed.json').exists():
        complete=json.loads((root/'completed.json').read_text())
        if sha(complete['student'])!=complete['student_sha256']:raise ValueError('Completed student changed')
        print('SHARED_STUDENT_ALREADY_COMPLETE',complete['student']);return
    from robolab.rough_gpu import query_gpus,candidates,gpu_lease,child_environment
    choices=candidates(query_gpus(),os.environ,min_free_mib=a.gpu_min_free_gb*1024,
                       max_utilization=a.gpu_max_utilization,requested=a.device)
    with gpu_lease(choices,HERE.parents[2]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        environment=child_environment(gpu,os.environ)
        all_paths=[];student=None;round_reports=[]
        jobs=jobs_for(plan,a.batch_designs)
        for round_index in range(a.dagger_rounds+1):
            for index,(task,ids) in enumerate(jobs):
                commit=root/'data_commits'/f'round_{round_index}'/f'{index:04d}_{task}.json'
                if commit.exists():
                    saved=json.loads(commit.read_text());folder=Path(saved['folder'])
                    if sha(folder/'data.pt')!=saved['sha256']:raise ValueError('Committed data changed')
                    all_paths.append(str(folder/'data.pt'));continue
                folder=root/'data'/f'round_{round_index}'/f'{index:04d}_{task}'/f'attempt_{datetime.now():%Y%m%d_%H%M%S_%f}'
                attempt=root/'processes'/f'collect_{round_index}_{index:04d}_{datetime.now():%Y%m%d_%H%M%S_%f}'
                command=[sys.executable,'-X','faulthandler','-u',str(HERE/('rollout_amp.py' if task=='run' else 'rollout.py')),'--plan',str(Path(a.plan).resolve()),
                    '--task',task,'--design-ids',','.join(ids),'--output',str(folder),'--asset-cache-root',str(root),
                    '--steps',str(a.collect_steps),'--stride',str(a.stride),'--seed',str(42+round_index*10000+index),
                    '--headless','--device','cuda:0']
                if task=='run':
                    pos=command.index('--task');del command[pos:pos+2]
                    pos=command.index('--asset-cache-root');del command[pos:pos+2]
                if student:command+=['--student',student]
                print('COLLECT_BEGIN',round_index,index+1,'/',len(jobs),task,len(ids),flush=True)
                child(command,attempt,environment,a.timeout_seconds,root)
                if not (folder/'data.pt').exists() or (folder/'failure.json').exists():raise ValueError('Missing real teacher data')
                commit.parent.mkdir(parents=True,exist_ok=True)
                write_new(commit,dict(folder=str(folder),sha256=sha(folder/'data.pt')))
                all_paths.append(str(folder/'data.pt'))
                print('COLLECT_COMPLETE',round_index,index+1,'/',len(jobs),flush=True)
            data_list=root/f'data_round_{round_index}.json'
            if not data_list.exists():write_new(data_list,all_paths)
            fit_commit=root/f'fit_round_{round_index}.json'
            if not fit_commit.exists():
                fit=root/f'bc_round_{round_index}'/f'attempt_{datetime.now():%Y%m%d_%H%M%S_%f}'
                command=[sys.executable,'-u',str(HERE/'fit.py'),'--plan',str(Path(a.plan).resolve()),
                    '--data-list',str(data_list),'--output',str(fit),'--architecture','mlp','--epochs',str(a.epochs),
                    '--steps-per-epoch',str(a.steps_per_epoch),'--max-rows-per-design-shard',str(a.max_rows_per_design_shard),'--device','cuda:0']
                if student:command+=['--init-student',student]
                child(command,root/'processes'/f'fit_{round_index}_{datetime.now():%Y%m%d_%H%M%S_%f}',
                      environment,max(a.timeout_seconds,7200),root)
                report=json.loads((fit/'summary.json').read_text())
                write_new(fit_commit,dict(summary=str(fit/'summary.json'),student_sha256=report['student_sha256']))
            saved=json.loads(fit_commit.read_text());report=json.loads(Path(saved['summary']).read_text());student=report['student']
            if sha(student)!=saved['student_sha256']:raise ValueError('Committed student changed')
            round_reports.append(report)
            print('SHARED_STUDENT_FIT',round_index,student,flush=True)
        # Full paired coverage. Every task/design is evaluated with both actors.
        evaluations=[];nominal=[]
        for index,(task,ids) in enumerate(jobs):
            for kind in ('teacher','student'):
                eval_commit=root/f'evaluation_{index:04d}_{task}_{kind}.json'
                if not eval_commit.exists():
                    folder=root/'evaluation'/f'{index:04d}_{task}_{kind}'/f'attempt_{datetime.now():%Y%m%d_%H%M%S_%f}'
                    command=[sys.executable,'-X','faulthandler','-u',str(HERE/('rollout_amp.py' if task=='run' else 'rollout.py')),
                        '--plan',str(Path(a.plan).resolve()),'--design-ids',','.join(ids),'--output',str(folder),
                        '--mode','evaluate','--steps','1000','--seed','20261029','--headless','--device','cuda:0']
                    if task!='run':command+=['--task',task,'--asset-cache-root',str(root)]
                    if kind=='student':command+=['--student',student]
                    child(command,root/'processes'/f'eval_{index}_{kind}_{datetime.now():%Y%m%d_%H%M%S_%f}',environment,a.timeout_seconds,root)
                    if not (folder/'summary.json').exists() or (folder/'failure.json').exists():raise ValueError('Evaluation did not complete')
                    write_new(eval_commit,dict(folder=str(folder)))
                folder=Path(json.loads(eval_commit.read_text())['folder'])
                report=json.loads((folder/'summary.json').read_text())
                rows=report['rows']
                if set(row['design'] for row in rows)!=set(ids):raise ValueError('Incomplete gait evaluation coverage')
                evaluations.append(dict(task=task,policy=kind,summary=str(folder/'summary.json')))
                failed=[row['environment'] for row in rows if row['fell'] or row['seconds']<19.9
                        or row['mean_forward_speed']<.1
                        or (task=='stairs' and row['ascent_m']<.1)]
                nominal.append(dict(task=task,policy=kind,designs=ids,environments=len(rows),failed=failed))
        passed=bool(nominal) and all(not report['failed'] for report in nominal)
        result=('SHARED_ACTOR_FULL_COMMON_BANK_NOMINAL_GAIT_PASSED' if passed
                else 'DISTILLATION_FINISHED_NOMINAL_EVALUATION_FAILED')
        write_new(root/'completed.json',dict(result=result,nominal_checks=nominal,nominal_passed=passed,
            student=student,student_sha256=sha(student),rounds=round_reports,evaluations=evaluations,
            teacher_updates=0,PPO_updates=0,critic_trained=False,scope=plan['scope'],
            gate='No falls, full 20 seconds, forward speed >=0.1m/s; stairs ascent >=0.1m; lateral drift ignored'))



if __name__=='__main__':main()
