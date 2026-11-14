"""Evaluate a completed candidate after another bounded GPU workflow finishes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from pipeline import HERE,child
from contract import load_plan,sha,write_new

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','fit-summary','after-summary','reference-run','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    load_plan(a.plan);start=time.monotonic()
    while not all(Path(path).is_file() for path in (a.fit_summary,a.after_summary)):
        if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
        if time.monotonic()-start>7200:raise TimeoutError('Prerequisite bounded workflow did not finish')
        time.sleep(3)
    time.sleep(5)
    fit=json.loads(Path(a.fit_summary).read_text());student=fit['student']
    if sha(student)!=fit['student_sha256']:raise ValueError('Candidate changed')
    from robolab.rough_gpu import query_gpus,candidates,gpu_lease,child_environment
    choices=candidates(query_gpus(),os.environ,min_free_mib=4096,max_utilization=30,requested='cuda:0')
    with gpu_lease(choices,HERE.parents[1]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        environment=child_environment(gpu,os.environ)
        previous=json.loads((Path(a.reference_run)/'completed.json').read_text());reports=[]
        for item in previous['evaluations']:
            if item['policy']!='teacher':continue
            task=item['task'];reference=json.loads(Path(item['summary']).read_text());out=root/task
            command=[sys.executable,'-u',str(HERE/'rollout.py'),'--plan',a.plan,
                '--task',task,'--design-ids',','.join(reference['design_ids']),
                '--output',str(out),'--student',student,'--mode','evaluate',
                '--steps','1000','--seed','20261029','--headless','--device','cuda:0']
            child(command,root/(task+'_logs'),environment,1200,root)
            if (out/'failure.json').exists():raise RuntimeError('Recorded candidate evaluation failure')
            rows=json.loads((out/'summary.json').read_text())['rows']
            reports.append(dict(task=task,count=len(rows),falls=sum(r['fell'] for r in rows),
                mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
                mean_forward_speed=sum(r['mean_forward_speed'] for r in rows)/len(rows),
                mean_ascent_m=sum(r['ascent_m'] for r in rows)/len(rows)))
            print('CANDIDATE_EVALUATION',json.dumps(reports[-1]),flush=True)
        write_new(root/'summary.json',dict(student=student,student_sha256=sha(student),
            evaluations=reports,full_robustness_validated=False))

if __name__=='__main__':main()
