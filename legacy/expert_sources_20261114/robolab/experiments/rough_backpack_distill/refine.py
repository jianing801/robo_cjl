"""One bounded additional DAgger round, preserving all prior runs and data."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from pipeline import HERE,child
from contract import load_plan,sha,write_new

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','data-list','student','output','wait-for-evaluation','reference-run'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();plan=load_plan(a.plan);root=Path(a.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    paths=json.loads(Path(a.data_list).read_text());initial_sha=sha(a.student)
    write_new(root/'manifest.json',dict(arguments=vars(a),plan_sha256=sha(a.plan),
        initial_student_sha256=initial_sha,teacher_updates=0,PPO_updates=0,
        collection_steps=1000,fit_epochs=4,fit_steps_per_epoch='automatic from task data size'))
    started=time.monotonic()
    while not Path(a.wait_for_evaluation).is_file():
        if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
        if time.monotonic()-started>1200:raise TimeoutError('Prior paired evaluation did not complete')
        time.sleep(3)
    time.sleep(3)
    if sha(a.student)!=initial_sha:raise ValueError('Initial student changed')
    from robolab.rough_gpu import query_gpus,candidates,gpu_lease,child_environment
    choices=candidates(query_gpus(),os.environ,min_free_mib=4096,max_utilization=30,requested='cuda:0')
    with gpu_lease(choices,HERE.parents[1]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        environment=child_environment(gpu,os.environ)
        for task,spec in plan['teachers'].items():
            out=root/('data_'+task)
            command=[sys.executable,'-u',str(HERE/'rollout.py'),'--plan',a.plan,
                '--task',task,'--design-ids',','.join(spec['design_ids']),
                '--output',str(out),'--student',a.student,'--mode','collect',
                '--steps','1000','--stride','2','--seed','20261030','--headless','--device','cuda:0']
            print('REFINE_COLLECT_BEGIN',task,flush=True)
            child(command,root/('log_'+task),environment,1200,root)
            if (out/'failure.json').exists() or not (out/'data.pt').exists():raise RuntimeError('Invalid DAgger collection')
            paths.append(str(out/'data.pt'))
            print('REFINE_COLLECT_COMPLETE',task,flush=True)
        listing=root/'data.json';write_new(listing,paths)
        command=[sys.executable,'-u',str(HERE/'fit.py'),'--plan',a.plan,
            '--data-list',str(listing),'--output',str(root/'fit'),
            '--init-student',a.student,'--epochs','4','--steps-per-epoch','0',
            '--learning-rate','0.0001','--device','cuda:0']
        child(command,root/'log_fit',environment,7200,root)
        fit=json.loads((root/'fit/summary.json').read_text());student=fit['student']
        previous=json.loads((Path(a.reference_run)/'completed.json').read_text())
        reports=[]
        for item in previous['evaluations']:
            if item['policy']!='teacher':continue
            task=item['task'];reference=json.loads(Path(item['summary']).read_text());out=root/('eval_'+task)
            command=[sys.executable,'-u',str(HERE/'rollout.py'),'--plan',a.plan,
                '--task',task,'--design-ids',','.join(reference['design_ids']),
                '--output',str(out),'--student',student,'--mode','evaluate',
                '--steps','1000','--seed','20261029','--headless','--device','cuda:0']
            child(command,root/('log_eval_'+task),environment,1200,root)
            if (out/'failure.json').exists():raise RuntimeError('Recorded evaluation failure')
            rows=json.loads((out/'summary.json').read_text())['rows']
            reports.append(dict(task=task,count=len(rows),falls=sum(r['fell'] for r in rows),
                mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
                mean_forward_speed=sum(r['mean_forward_speed'] for r in rows)/len(rows),
                mean_ascent_m=sum(r['ascent_m'] for r in rows)/len(rows)))
            print('REFINE_EVALUATION',json.dumps(reports[-1]),flush=True)
        write_new(root/'summary.json',dict(student=student,student_sha256=sha(student),
            evaluations=reports,full_robustness_validated=False,teacher_updates=0,PPO_updates=0,
            finished=datetime.now().isoformat()))

if __name__=='__main__':main()
