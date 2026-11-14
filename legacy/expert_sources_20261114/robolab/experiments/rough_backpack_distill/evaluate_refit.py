"""Wait for a bounded offline refit, then repeat the original paired student tests."""
import json
import os
from pathlib import Path
import sys
import time
from pipeline import HERE,child
from contract import load_plan,write_new,sha

def main():
    w=HERE.parents[3];root=w/'rbv1';root.mkdir(exist_ok=False)
    started=time.monotonic();summary=w/'rbf1/summary.json'
    while not summary.exists():
        if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
        if time.monotonic()-started>3600:raise TimeoutError('Refit did not finish within one hour')
        time.sleep(5)
    time.sleep(3)
    fit=json.loads(summary.read_text());student=fit['student']
    if sha(student)!=fit['student_sha256']:raise ValueError('Refit checkpoint changed')
    plan=w/'rbc2/plan.json';load_plan(plan)
    reports=[]
    old=json.loads((w/'rbs2/completed.json').read_text())
    for item in old['evaluations']:
        if item['policy']!='teacher':continue
        task=item['task'];reference=json.loads(Path(item['summary']).read_text())
        out=root/task
        command=[sys.executable,'-u',str(HERE/'rollout.py'),'--plan',str(plan),
            '--task',task,'--design-ids',','.join(reference['design_ids']),
            '--output',str(out),'--student',student,'--mode','evaluate',
            '--steps','1000','--seed','20261029','--headless','--device','cuda:0']
        child(command,root/(task+'_logs'),os.environ,900,root)
        result=json.loads((out/'summary.json').read_text())
        if (out/'failure.json').exists():raise RuntimeError('Evaluation recorded failure')
        rows=result['rows'];reports.append(dict(task=task,count=len(rows),
            falls=sum(row['fell'] for row in rows),
            mean_seconds=sum(row['seconds'] for row in rows)/len(rows),
            mean_ascent_m=sum(row['ascent_m'] for row in rows)/len(rows)))
        print('REFIT_EVALUATION',json.dumps(reports[-1]),flush=True)
    write_new(root/'summary.json',dict(student=student,student_sha256=sha(student),
        evaluations=reports,validated_success=False,
        note='Inspect survival and task progress before promoting this student'))

if __name__=='__main__':main()
