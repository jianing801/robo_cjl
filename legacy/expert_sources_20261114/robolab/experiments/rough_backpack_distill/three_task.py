"""Audited three-task BC/DAgger, real nominal gates, bounded automatic cycles."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from pipeline import HERE,child
from contract import load_plan,sha,write_new,structure_key

def assess(rows,reference,task):
    if len(rows)!=len(reference):raise ValueError('Unpaired evaluation row count')
    failed=[]
    for row,teacher in zip(rows,reference):
        if (row['design'],row['environment'])!=(teacher['design'],teacher['environment']):
            raise ValueError('Unpaired physical design/environment')
        speed=teacher['mean_forward_speed']
        passed=(not row['fell'] and row['seconds']>=19.9 and row['mean_forward_speed']>=max(.1,.8*speed)
            and row['mean_forward_speed']<=max(.3,1.2*speed)
            and abs(row['lateral_m'])<=max(1.,abs(teacher['lateral_m'])+.5))
        if task=='stairs':passed=passed and row['ascent_m']>=max(.1,.8*teacher['ascent_m'])
        if not passed:failed.append(dict(design=row['design'],environment=row['environment'],
            fell=row['fell'],seconds=row['seconds'],speed=row['mean_forward_speed'],lateral_m=row['lateral_m']))
    return dict(task=task,total=len(rows),falls=sum(r['fell'] for r in rows),
        passed=len(rows)-len(failed),pass_fraction=1-len(failed)/len(rows),failed=failed,
        mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
        mean_forward_speed=sum(r['mean_forward_speed'] for r in rows)/len(rows))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--bootstrap',required=True)
    p.add_argument('--rounds',type=int,default=4);p.add_argument('--prepare-workers',type=int,choices=[1,2],default=2)
    p.add_argument('--resume',action='store_true',help='Reuse verified completed stages; preserve failed attempts')
    p.add_argument('--heading-replay',action='store_true',help='From round 4 add physical frozen-teacher yaw recovery demonstrations; evaluation unchanged')
    p.add_argument('--late-learning-rate',type=float,default=.0001,help='Learning rate for rounds 6 onward only')
    p.add_argument('--tail-replay',action='store_true',help='Round10 adds audited failing-design teacher recovery trajectories')
    a=p.parse_args()
    if not 1<=a.rounds<=12:raise ValueError('Finite audited cycles required')
    if not 0<a.late_learning_rate<=.0001:raise ValueError('Late learning rate must be positive and no larger than baseline')
    root=Path(a.output).resolve()
    if not a.resume:root.mkdir(parents=True,exist_ok=False)
    original=load_plan(a.plan);w=Path(original['workspace'])
    manifest=dict(arguments=vars(a),plan_sha256=sha(a.plan),
        bootstrap_sha256=sha(a.bootstrap),teacher_updates=0,PPO_updates=0,
        preparation='At most two isolated 4-asset converters; physical sampling and optimization serial',
        nominal_gate='20 s; no fall; speed within 20% of paired teacher; bounded lateral drift; stairs ascent >=80% teacher',
        full_bank_gate='At least 99% nominal passes separately in every task; not real-world robustness certification')
    if a.resume:
        previous=json.loads((root/'manifest.json').read_text())
        if previous['plan_sha256']!=sha(a.plan) or previous['bootstrap_sha256']!=sha(a.bootstrap):raise ValueError('Resume identities changed')
        for key in ('plan','output','bootstrap'):
            if Path(previous['arguments'][key]).resolve()!=Path(getattr(a,key)).resolve():raise ValueError('Resume paths changed')
        if a.rounds<previous['arguments']['rounds']:raise ValueError('Cannot silently reduce the planned cycle')
        write_new(root/f'continuation_{datetime.now():%Y%m%d_%H%M%S_%f}.json',manifest)
    else:write_new(root/'manifest.json',manifest)
    if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED; explicit human resume required')
    if (root/'completed.json').exists():
        result=json.loads((root/'completed.json').read_text())
        if sha(result['student'])!=result['student_sha256']:raise ValueError('Completed checkpoint changed')
        print('THREE_TASK_ALREADY_COMPLETE',result['student'],flush=True);return
    from robolab.rough_gpu import query_gpus,candidates,gpu_lease,child_environment
    choices=candidates(query_gpus(),os.environ,min_free_mib=6144,max_utilization=30,requested='cuda:0')
    with gpu_lease(choices,HERE.parents[1]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        environment=child_environment(gpu,os.environ)
        ids=original['teachers']['run']['design_ids'];chunks=[ids[i:i+4] for i in range(0,len(ids),4)]
        def prepare(index,selected):
            if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
            previous=sorted((root/'cache').glob(f'c{index:02d}a*/summary.json')) if (root/'cache').exists() else []
            for path in previous:
                saved=json.loads(path.read_text())
                if saved['remaining']==0 and {r['design'] for r in saved['assets']}==set(selected):
                    if all((Path(r['cache'])/'config.yaml').is_file() and
                           (Path(r['cache'])/'.asset_hash').read_text().strip()==r['exact_hash'] and
                           any(Path(r['cache']).glob('*.usd')) for r in saved['assets']):return
            first_attempt=0
            while (root/'cache'/f'c{index:02d}a{first_attempt}').exists():first_attempt+=1
            for attempt in range(2):
                actual_attempt=first_attempt+attempt
                target=root/'cache'/f'c{index:02d}a{actual_attempt}'
                command=[sys.executable,'-u',str(HERE/'prepare_assets.py'),'--plan',a.plan,'--task','run',
                    '--design-ids',','.join(selected),'--output',str(target),'--max-conversions','4',
                    '--cache-root',str(root/'cache'),'--cache-root',str(w/'ta00'),'--headless','--device','cuda:0']
                try:
                    child(command,root/'prepare_logs'/f'c{index:02d}a{actual_attempt}',environment,240,root)
                    result=json.loads((target/'summary.json').read_text())
                    if result['remaining']!=0:raise RuntimeError('Asset subset incomplete')
                    print('AMP_ASSET_BATCH_COMPLETE',index+1,'/',len(chunks),flush=True);return
                except (TimeoutError,RuntimeError) as error:
                    failure=target/'failure.json'
                    write_new(root/f'prepare_failure_{index:02d}_{actual_attempt}.json',dict(error=str(error),
                        native_failure=json.loads(failure.read_text()) if failure.exists() else None,
                        stderr_tail=(root/'prepare_logs'/f'c{index:02d}a{actual_attempt}'/'stderr.log').read_text(encoding='utf8',errors='replace')[-4000:]))
                    if attempt or (root/'USER_PAUSED').exists():raise
        with ThreadPoolExecutor(max_workers=a.prepare_workers) as pool:
            cursor=0;pending=set()
            while cursor<len(chunks) or pending:
                while cursor<len(chunks) and len(pending)<a.prepare_workers:
                    pending.add(pool.submit(prepare,cursor,chunks[cursor]));cursor+=1
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:future.result()
        plan=deepcopy(original)
        plan['teachers']['run']['asset_cache']=[str(root/'cache'),str(w/'ta00')]
        frozen=root/'plan.json'
        if frozen.exists():
            if json.loads(frozen.read_text())!=plan:raise ValueError('Frozen collection plan changed')
        else:write_new(frozen,plan)
        load_plan(frozen)
        banks={t:json.loads(Path(s['bank']).read_text(encoding='utf8')) for t,s in plan['teachers'].items()}
        old=json.loads((w/'rbs2/completed.json').read_text())
        reference={}
        selected={}
        for task in ('stairs','backpack'):
            item=next(e for e in old['evaluations'] if e['task']==task and e['policy']=='teacher')
            reference[task]=Path(item['summary']);selected[task]=json.loads(reference[task].read_text())['design_ids']
        keys={structure_key(d) for d in banks['stairs']['designs'] if d['id'] in selected['stairs']}
        selected['run']=[d['id'] for d in banks['run']['designs'] if structure_key(d) in keys]
        def rollout(task,designs,out,student=None,mode='evaluate',seed=20261029,yaw_range=.05):
            expected=sha(student) if student else None
            candidates=[out,*sorted(out.parent.glob(out.name+'_retry_*'),reverse=True)]
            for candidate in candidates:
                record=candidate/'summary.json'
                if not record.is_file() or (candidate/'failure.json').exists():continue
                saved=json.loads(record.read_text())
                if (saved['task']==task and saved['design_ids']==designs and saved['steps']==1000 and
                    saved['teacher']==plan['teachers'][task]['checkpoint_sha256'] and saved['student']==expected and
                    saved.get('reset_yaw_range_rad',.05)==yaw_range and
                    (mode!='collect' or (candidate/'data.pt').is_file())):return record
            if out.exists():out=out.with_name(out.name+f'_retry_{datetime.now():%m%d_%H%M%S_%f}')
            script=HERE/('rollout_amp.py' if task=='run' else 'rollout.py')
            command=[sys.executable,'-u',str(script),'--plan',str(frozen),'--design-ids',','.join(designs),
                '--output',str(out),'--mode',mode,'--steps','1000','--stride','2','--seed',str(seed),'--headless','--device','cuda:0']
            if task!='run':command+=['--task',task]
            command+=['--reset-yaw-range',str(yaw_range)]
            if student:command+=['--student',str(student)]
            print('THREE_TASK_ROLLOUT_BEGIN',task,mode,len(designs),'student',bool(student),flush=True)
            child(command,root/'processes'/out.name,environment,1800,root)
            if (out/'failure.json').exists() or not (out/'summary.json').exists():raise RuntimeError('Invalid native rollout')
            return out/'summary.json'
        reference['run']=rollout('run',selected['run'],root/'ref_run')
        baseline_paths=json.loads((w/'rbs2/data_round_1.json').read_text())
        amp_teacher=rollout('run',ids,root/'data_run_teacher',mode='collect',seed=20261030)
        paths=[*baseline_paths,str(amp_teacher.parent/'data.pt')];student=Path(a.bootstrap)
        rounds=[];full_reference={}
        for index in range(a.rounds):
            if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
            if index==10 and a.tail_replay:
                diagnostic=json.loads((root/'audit9/summary.json').read_text())
                if diagnostic['student_sha256']!=sha(student):raise ValueError('Tail diagnostic checkpoint mismatch')
                for check in diagnostic['checks']:
                    task=check['task'];failed_ids={row['design'] for row in check['failed']}
                    designs=[d for d in plan['teachers'][task]['design_ids'] if d in failed_ids]
                    if not designs:continue
                    for label,policy,yaw in [('teacher',None,.2),('student',student,.05)]:
                        saved=rollout(task,designs,root/f'tail10_{task}_{label}',
                            student=policy,mode='collect',seed=20261040,yaw_range=yaw)
                        paths.append(str(saved.parent/'data.pt'))
            if index==4 and a.heading_replay:
                for task,spec in plan['teachers'].items():
                    saved=rollout(task,spec['design_ids'],root/f'heading_teacher_{task}',
                        mode='collect',seed=20261034,yaw_range=.3)
                    paths.append(str(saved.parent/'data.pt'))
            if index:
                for task,spec in plan['teachers'].items():
                    out=root/f'd{index}_{task}'
                    saved=rollout(task,spec['design_ids'],out,student,'collect',20261030+index)
                    paths.append(str(saved.parent/'data.pt'))
            listing=root/f'data_{index}.json'
            if listing.exists():
                if json.loads(listing.read_text())!=paths:raise ValueError('Committed replay list changed')
            else:write_new(listing,paths)
            fit=root/f'fit{index}'
            if fit.exists() and not (fit/'summary.json').exists():fit=fit.with_name(fit.name+f'_retry_{datetime.now():%m%d_%H%M%S_%f}')
            command=[sys.executable,'-u',str(HERE/'fit.py'),'--plan',str(frozen),'--data-list',str(listing),
                '--output',str(fit),'--init-student',str(student),'--architecture','mlp','--epochs','6',
                '--steps-per-epoch','0','--max-rows-per-design-shard','512','--learning-rate',str(a.late_learning_rate if index>=6 else .0001),'--device','cuda:0']
            if not (fit/'summary.json').exists():
                print('THREE_TASK_FIT_BEGIN',index,flush=True)
                child(command,root/'processes'/fit.name,environment,7200,root)
            fitted=json.loads((fit/'summary.json').read_text());student=Path(fitted['student'])
            if sha(student)!=fitted['student_sha256']:raise ValueError('Committed fit checkpoint changed')
            checks=[]
            for task in plan['schema']['tasks']:
                summary=rollout(task,selected[task],root/f'e{index}_{task}',student)
                checks.append(assess(json.loads(summary.read_text())['rows'],json.loads(reference[task].read_text())['rows'],task))
            round_record=dict(round=index,student=str(student),student_sha256=sha(student),checks=checks,
                fit_summary=str(fit/'summary.json'))
            record=root/f'round_{index}.json'
            if record.exists():
                if json.loads(record.read_text())!=round_record:raise ValueError('Committed round changed')
            else:write_new(record,round_record)
            rounds.append(round_record)
            print('THREE_TASK_NOMINAL_RESULTS',json.dumps([{k:v for k,v in c.items() if k!='failed'} for c in checks]),flush=True)
            if all(c['pass_fraction']==1. for c in checks):
                full=[]
                for task,spec in plan['teachers'].items():
                    if task not in full_reference:
                        full_reference[task]=rollout(task,spec['design_ids'],root/f'full_ref_{task}')
                    result=rollout(task,spec['design_ids'],root/f'f{index}_{task}',student)
                    full.append(assess(json.loads(result.read_text())['rows'],json.loads(full_reference[task].read_text())['rows'],task))
                full_path=root/f'full_{index}.json'
                if not full_path.exists():write_new(full_path,full)
                if all(c['pass_fraction']>=.99 for c in full):
                    write_new(root/'completed.json',dict(result='THREE_TASK_SHARED_ACTOR_NOMINAL_VALIDATED',
                        student=str(student),student_sha256=sha(student),rounds=rounds,full_bank_checks=full,
                        scope='200 exact common structures; 20 s nominal paired tasks; no full robustness claim',
                        teacher_updates=0,PPO_updates=0,finished=datetime.now().isoformat()))
                    print('THREE_TASK_NOMINAL_COMPLETE',student,flush=True);return
        finished=root/'cycle_finished.json'
        if finished.exists():finished=root/f'cycle_finished_{a.rounds}_{datetime.now():%m%d_%H%M%S_%f}.json'
        write_new(finished,dict(result='BOUNDED_CYCLE_FINISHED_REQUIRES_FURTHER_IMPROVEMENT',
            student=str(student),student_sha256=sha(student),rounds=rounds,data_list=str(listing),
            note='Preserve outputs; inspect failure patterns before another bounded cycle'))

if __name__=='__main__':main()
