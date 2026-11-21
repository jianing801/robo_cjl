"""Bounded serial real-data collection, balanced distillation and paired evaluation."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[2]),str(HERE)]
from contract import load_plan,sha,write_new,design_ids_for_keys
from collection_grid import calibration_keys, select_collection_grid, selection_panel
from randomization import protocol_metadata
from persistence import run_lock,resume_pause
from evaluation import (evaluation_statistics, aggregate_evaluations, candidate_rank,
                        evaluation_criteria, teacher_comparison, SELECTION_RULE, GATE_VERSION,
                        COMPARISON_VERSION)


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


def jobs_for(plan,batch_size,design_ids=None):
    result=[]
    for task,spec in plan['teachers'].items():
        ids=spec['design_ids'] if design_ids is None else design_ids[task]
        for offset in range(0,len(ids),batch_size):result.append((task,ids[offset:offset+batch_size]))
    return result


def round_strength(index, rounds, initial, target, profile):
    if (profile not in ('nominal', 'robust') or not 0 <= index <= rounds
            or not 0 <= initial <= 1 or not 0 <= target <= 1
            or (profile == 'robust' and initial > target)):
        raise ValueError('Require strengths in [0,1], a nondecreasing robust curriculum and a valid round/profile')
    if profile == 'nominal':
        return 0.
    return target if rounds == 0 else initial + (target-initial)*index/rounds


def last_checkpoint(report, epochs):
    checkpoints = report.get('checkpoints', [])
    by_epoch = {row['epoch']: row for row in checkpoints}
    if len(checkpoints) != epochs or set(by_epoch) != set(range(epochs)):
        raise ValueError('Missing or duplicate epoch checkpoints')
    last = by_epoch[epochs - 1]
    if sha(last['student']) != last['student_sha256']:
        raise ValueError('Final epoch checkpoint changed')
    return last


def evaluation_copies(task, spec, profile):
    return max(spec['copies'], 6) if task == 'stairs' and profile == 'robust' else spec['copies']


def validate_evaluation_report(report, task, ids, spec, student_sha, protocol):
    if (report.get('collection_protocol') != protocol or report.get('task') != task
            or report.get('teacher') != spec['checkpoint_sha256']
            or report.get('student') != student_sha or report.get('design_ids') != ids
            or report.get('steps') != 1000):
        raise ValueError('Evaluation protocol or actor/task identity differs from the request')
    copies = evaluation_copies(task, spec, protocol['profile'])
    rows = report['rows']
    environments = [row['environment'] for row in rows]
    if (len(rows) != len(ids)*copies
            or any(isinstance(env, bool) or not isinstance(env, int) for env in environments)
            or sorted(environments) != list(range(len(ids)*copies))
            or any(row['design'] != ids[row['environment'] % len(ids)] for row in rows)):
        raise ValueError('Incomplete or duplicate gait evaluation environment/design coverage')
    return rows


def run_evaluation(plan, plan_path, root, jobs, student, profile, strength, seed,
                   suite, environment, timeout, criteria=None):
    """Persist fixed-seed evaluation separately from training and from final tests."""
    policy = 'student' if student else 'teacher'
    evaluations = []
    protocol = protocol_metadata(profile, strength)
    criteria_parameters = criteria or {}
    declared_criteria = evaluation_criteria(**criteria_parameters)
    for index, (task, ids) in enumerate(jobs):
        expected = dict(profile=profile, protocol=protocol, seed=seed, design_ids=ids,evaluation_criteria=declared_criteria,
                        policy=policy, actor_sha256=sha(student) if student else plan['teachers'][task]['checkpoint_sha256'])
        commit = root/'evaluation_commits'/suite/f'{index:04d}_{task}_{policy}.json'
        new_commit = not commit.exists()
        if new_commit:
            folder = root/'evaluation'/suite/f'{index:04d}_{task}_{policy}'/f'attempt_{datetime.now():%Y%m%d_%H%M%S_%f}'
            command = [sys.executable, '-X', 'faulthandler', '-u',
                       str(HERE/('rollout_amp.py' if task == 'run' else 'rollout.py')),
                       '--plan', str(plan_path), '--design-ids', ','.join(ids), '--output', str(folder),
                       '--mode', 'evaluate', '--profile', profile, '--randomization-strength', str(strength),
                       '--steps', '1000', '--seed', str(seed), '--headless', '--device', 'cuda:0']
            if task != 'run':
                command += ['--task', task, '--asset-cache-root', str(root)]
            if student:
                command += ['--student', student]
            child(command, root/'processes'/f'eval_{suite.replace("/", "_")}_{index}_{policy}_{datetime.now():%Y%m%d_%H%M%S_%f}',
                  environment, timeout, root)
            if not (folder/'summary.json').exists() or (folder/'failure.json').exists():
                raise ValueError('Evaluation did not complete')
        else:
            saved = json.loads(commit.read_text())
            folder = Path(saved['folder'])
            if saved['identity'] != expected or sha(folder/'summary.json') != saved['summary_sha256']:
                raise ValueError('Committed evaluation profile, policy or results changed')
        report = json.loads((folder/'summary.json').read_text())
        rows = validate_evaluation_report(report, task, ids, plan['teachers'][task],
            expected['actor_sha256'] if student else None, protocol)
        statistics = evaluation_statistics(task, rows, profile, **criteria_parameters)
        if new_commit:
            commit.parent.mkdir(parents=True, exist_ok=True)
            write_new(commit, dict(folder=str(folder), identity=expected, summary_sha256=sha(folder/'summary.json')))
        evaluations.append(dict(task=task, policy=policy, profile=profile, seed=seed, designs=ids,
                                evaluation_protocol=protocol,
                                summary=str(folder/'summary.json'), **statistics))
    return evaluations


def run_selection_suite(plan, plan_path, root, jobs, student, profile, strength, seeds,
                        suite, environment, timeout, criteria=None):
    records, by_seed = [], {}
    for seed in seeds:
        current = run_evaluation(plan, plan_path, root, jobs, student, profile, strength,
                                 seed, f'{suite}/seed_{seed}', environment, timeout, criteria)
        records.extend(current)
        by_seed[str(seed)] = aggregate_evaluations(current, plan['schema']['tasks'])
    return dict(task_metrics=aggregate_evaluations(records, plan['schema']['tasks']),
                by_seed=by_seed, evaluations=records)


def summarize_final_evaluations(records, tasks, profiles):
    """Describe paired policy results; no teacher-ability or distillation pass gate."""
    if (not records or not tasks or not profiles or len(set(profiles)) != len(profiles)
            or {row.get('profile') for row in records} != set(profiles)
            or any(row.get('policy') not in ('teacher', 'student') or row.get('task') not in tasks
                   for row in records)):
        raise ValueError('Final comparison requires nonempty declared profiles, tasks and paired policies')
    result = {}
    for profile in profiles:
        policy_metrics, all_episodes_passed = {}, {}
        for policy in ('teacher', 'student'):
            current = [row for row in records if row['profile'] == profile and row['policy'] == policy]
            if not current:
                raise ValueError(f'Missing final {profile}/{policy} evaluation records')
            policy_metrics[policy] = aggregate_evaluations(current, tasks)
            all_episodes_passed[policy] = all(not row['failed'] for row in current)
        result[profile] = dict(policy_metrics=policy_metrics,
            teacher_comparison=teacher_comparison(policy_metrics['student'], policy_metrics['teacher']),
            comparison_version=COMPARISON_VERSION,
            policy_all_episodes_command_tracking_passed=all_episodes_passed,
            scope='Observed per-policy episodes only; no distillation acceptance or optimal-control claim')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--collect-steps',type=int,default=250)
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--steps-per-epoch',type=int,default=0,
        help='0 derives steps from the per-design/source sampling budget; positive values set exact update counts')
    p.add_argument('--max-rows-per-design-shard',type=int,default=None,help='Retired; use --samples-per-design-per-epoch')
    p.add_argument('--samples-per-design-per-epoch',type=int,default=32,help='Optimization exposure budget; does not discard collected training rows')
    p.add_argument('--validation-rows-per-design-shard',type=int,default=128)
    p.add_argument('--replay-cache-shards',type=int,default=12)
    p.add_argument('--dagger-rounds',type=int,default=3)
    p.add_argument('--collection-profile',choices=['nominal','robust'],default='robust')
    p.add_argument('--randomization-strength',type=float,default=1.)
    p.add_argument('--initial-randomization-strength',type=float,default=.25,
                   help='First BC round strength; linearly reaches target at the last DAgger round')
    p.add_argument('--selection-profile',choices=['nominal','robust'],default='robust')
    p.add_argument('--selection-seeds',default='20261117,20261119,20261120')
    p.add_argument('--tracking-abs-tolerance',type=float,default=None,help='Retired; moving tracking has no absolute tolerance floor')
    p.add_argument('--standing-speed-tolerance',type=float,default=None,help='Retired; standing commands and their acceptance gate have been removed')
    p.add_argument('--tracking-rel-tolerance',type=float,default=.25,help='Moving-only RMSE tolerance relative to moving commanded vx RMS')
    p.add_argument('--min-command-progress-ratio',type=float,default=.5,help='Minimum realized/commanded forward progress for moving commands')
    p.add_argument('--resume',action='store_true');p.add_argument('--device')
    p.add_argument('--resume-paused',action='store_true',help='Resume a manually paused run after identity checks')
    p.add_argument('--gpu-min-free-gb',type=float,default=4.)
    p.add_argument('--gpu-max-utilization',type=int,default=30)
    p.add_argument('--timeout-seconds',type=int,default=900)
    a=p.parse_args(argv)
    if a.resume_paused:a.resume=True
    if a.tracking_abs_tolerance is not None:
        raise ValueError('--tracking-abs-tolerance is retired; moving tracking uses only --tracking-rel-tolerance.')
    if a.standing_speed_tolerance is not None:
        raise ValueError('--standing-speed-tolerance is retired; standing commands and their acceptance gate have been removed.')
    if a.max_rows_per_design_shard is not None:
        raise ValueError('--max-rows-per-design-shard is retired; use --samples-per-design-per-epoch for a compute budget without discarding training data')
    if min(a.collect_steps,a.stride,a.epochs,a.timeout_seconds,a.samples_per_design_per_epoch,a.validation_rows_per_design_shard,a.replay_cache_shards)<1 or a.steps_per_epoch<0 or a.dagger_rounds<1:
        raise ValueError('Positive finite collection/BC budgets required')
    round_strength(0,a.dagger_rounds,a.initial_randomization_strength,a.randomization_strength,a.collection_profile)
    seeds=[int(value.strip()) for value in a.selection_seeds.split(',')]
    if len(seeds)!=3 or len(set(seeds))!=3 or any(not 0<=seed<2**32 for seed in seeds):
        raise ValueError('Round comparison requires three distinct valid seeds')
    root=Path(a.output).resolve()
    with run_lock(root):
        try:
            return execute_pipeline(a,seeds)
        except KeyboardInterrupt:
            if root.is_dir():(root/'USER_PAUSED').touch(exist_ok=True)
            raise


def execute_pipeline(a,seeds):
    plan=load_plan(a.plan);root=Path(a.output).resolve()
    criteria_parameters=dict(duration_s=20.,time_tolerance_s=.1,
        tracking_rel_tol=a.tracking_rel_tolerance,
        min_command_progress_ratio=a.min_command_progress_ratio,max_missing_tracking_steps=1)
    criteria=evaluation_criteria(**criteria_parameters)
    collection_keys=select_collection_grid(plan['selected_structure_keys'])
    collection_ids=design_ids_for_keys(plan,collection_keys)
    collection_jobs=jobs_for(plan,400,collection_ids)
    reserved=calibration_keys()
    selected_keys=selection_panel(plan['selected_structure_keys'],collection_keys,reserved)
    selected_jobs=jobs_for(plan,50,design_ids_for_keys(plan,selected_keys))
    collection_scope=dict(design_count=len(collection_keys),structure_keys=collection_keys,
        design_ids_by_task=collection_ids,steps_per_rollout=a.collect_steps,environments_per_task=4096,
        batches_per_round=len(collection_jobs),selector_sha256=sha(HERE/'collection_grid.py'))
    budget=dict(selection_design_count=len(selected_keys),teacher_rollout_processes=len(selected_jobs)*len(seeds),
        candidate_rollout_processes_per_round=len(selected_jobs)*len(seeds),
        total_selection_rollout_processes=(1+a.dagger_rounds)*len(selected_jobs)*len(seeds),final_test_rollout_processes=0)
    protocol=dict(collection=protocol_metadata(a.collection_profile,a.randomization_strength if a.collection_profile=='robust' else 0.),
        collection_scope=collection_scope,
        initial_strength=a.initial_randomization_strength,dagger_rounds=a.dagger_rounds,
        selection=protocol_metadata(a.selection_profile,a.randomization_strength if a.selection_profile=='robust' else 0.),
        selection_seeds=seeds,selection_keys=selected_keys,calibration_keys=reserved,
        selection_version='last_epoch_dagger_paired_50',selection_budget=budget,
        comparison_version=COMPARISON_VERSION,
        evaluation_criteria=criteria,evaluation_version=GATE_VERSION,
        implementation_sha256=sha(__file__),fit_implementation_sha256=sha(HERE/'fit.py'),
        evaluation_implementation_sha256=sha(HERE/'evaluation.py'),
        rollout_implementation={p:sha(HERE/p) for p in ('rollout.py','rollout_amp.py','scenario_sampling.py','physical_audit.py')})
    protocol_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if a.resume:
        previous=json.loads((root/'manifest.json').read_text())
        expected={k:v for k,v in vars(a).items() if k not in ('resume','resume_paused')}
        if previous.get('protocol_sha256')!=protocol_sha256:
            raise ValueError('Collection/selection protocol changed or this is a legacy run. Keep old data; use a new output directory.')
        if previous['arguments']!=expected or previous['plan_sha256']!=sha(a.plan):raise ValueError('Resume plan/budget changed')
    else:
        root.mkdir(parents=True,exist_ok=False)
        write_new(root/'manifest.json',dict(plan_sha256=sha(a.plan),arguments={k:v for k,v in vars(a).items() if k not in ('resume','resume_paused')},
            protocol=protocol,protocol_sha256=protocol_sha256,selection_rule=SELECTION_RULE,selection_budget=budget,
            teacher_updates=0,initial_collection='teacher actions',later_collection='student actions with frozen teacher labels'))
    resume_pause(root,a.resume_paused)
    if (root/'completed.json').exists():
        complete=json.loads((root/'completed.json').read_text())
        if (sha(complete['student']) != complete['student_sha256']
                or sha(complete['selection']) != complete['selection_sha256']
                or complete['protocol_sha256'] != protocol_sha256):
            raise ValueError('Completed student or its selection evidence changed')
        print('SHARED_STUDENT_ALREADY_COMPLETE',complete['student']);return
    from robolab.rough_gpu import query_gpus,candidates,gpu_lease,child_environment
    choices=candidates(query_gpus(),os.environ,min_free_mib=a.gpu_min_free_gb*1024,
                       max_utilization=a.gpu_max_utilization,requested=a.device)
    with gpu_lease(choices,HERE.parents[2]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        environment=child_environment(gpu,os.environ)
        print('DISTILL_COLLECTION',collection_scope['design_count'],'structures per task',
            '400 structures per task batch, 4096 environments',len(collection_jobs),'batches per round',
            a.collect_steps,'control steps per environment',flush=True)
        all_paths=[];student=None;round_reports=[];selection_reports=[]
        selection_strength=a.randomization_strength if a.selection_profile=='robust' else 0.
        teacher_baseline=None
        for round_index in range(a.dagger_rounds+1):
            strength=round_strength(round_index,a.dagger_rounds,a.initial_randomization_strength,
                                    a.randomization_strength,a.collection_profile)
            for index,(task,ids) in enumerate(collection_jobs):
                commit=root/'data_commits'/f'round_{round_index}'/f'{index:04d}_{task}.json'
                execution_sha=sha(student) if student else None
                if commit.exists():
                    saved=json.loads(commit.read_text());folder=Path(saved['folder'])
                    if (sha(folder/'data.pt')!=saved['sha256']
                            or saved.get('protocol')!=protocol_metadata(a.collection_profile,strength)
                            or saved.get('round_index')!=round_index
                            or saved.get('execution_student_sha256')!=execution_sha):
                        raise ValueError('Committed data or collection protocol changed')
                    all_paths.append(str(folder/'data.pt'));continue
                folder=root/'data'/f'round_{round_index}'/f'{index:04d}_{task}'/f'attempt_{datetime.now():%Y%m%d_%H%M%S_%f}'
                attempt=root/'processes'/f'collect_{round_index}_{index:04d}_{datetime.now():%Y%m%d_%H%M%S_%f}'
                command=[sys.executable,'-X','faulthandler','-u',str(HERE/('rollout_amp.py' if task=='run' else 'rollout.py')),'--plan',str(Path(a.plan).resolve()),
                    '--task',task,'--design-ids',','.join(ids),'--output',str(folder),'--asset-cache-root',str(root),
                    '--num-envs','4096','--steps',str(a.collect_steps),'--stride',str(a.stride),'--seed',str(42+round_index*10000+index),
                    '--profile',a.collection_profile,'--randomization-strength',str(strength),
                    '--headless','--device','cuda:0']
                if task=='run':
                    pos=command.index('--task');del command[pos:pos+2]
                    pos=command.index('--asset-cache-root');del command[pos:pos+2]
                if student:command+=['--student',student]
                print('COLLECT_BEGIN',round_index,index+1,'/',len(collection_jobs),task,len(ids),flush=True)
                child(command,attempt,environment,a.timeout_seconds,root)
                if not (folder/'data.pt').exists() or (folder/'failure.json').exists():raise ValueError('Missing real teacher data')
                commit.parent.mkdir(parents=True,exist_ok=True)
                write_new(commit,dict(folder=str(folder),sha256=sha(folder/'data.pt'),
                    protocol=protocol_metadata(a.collection_profile,strength),round_index=round_index,
                    execution_student_sha256=execution_sha))
                all_paths.append(str(folder/'data.pt'))
                print('COLLECT_COMPLETE',round_index,index+1,'/',len(collection_jobs),flush=True)
            data_list=root/f'data_round_{round_index}.json'
            if not data_list.exists():write_new(data_list,all_paths)
            elif json.loads(data_list.read_text())!=all_paths:raise ValueError('Committed round data list changed')
            fit_commit=root/f'fit_round_{round_index}.json'
            fit_identity=dict(data_list_sha256=sha(data_list),init_student_sha256=sha(student) if student else None,
                implementation_sha256=sha(HERE/'fit.py'))
            if not fit_commit.exists():
                fit=root/f'bc_round_{round_index}'/f'attempt_{datetime.now():%Y%m%d_%H%M%S_%f}'
                command=[sys.executable,'-u',str(HERE/'fit.py'),'--plan',str(Path(a.plan).resolve()),
                    '--data-list',str(data_list),'--output',str(fit),'--architecture','mlp','--epochs',str(a.epochs),
                    '--steps-per-epoch',str(a.steps_per_epoch),
                    '--samples-per-design-per-epoch',str(a.samples_per_design_per_epoch),
                    '--validation-rows-per-design-shard',str(a.validation_rows_per_design_shard),
                    '--replay-cache-shards',str(a.replay_cache_shards),'--device','cuda:0']
                command.append('--collection-grid')
                if student:command+=['--init-student',student]
                child(command,root/'processes'/f'fit_{round_index}_{datetime.now():%Y%m%d_%H%M%S_%f}',
                      environment,max(a.timeout_seconds,7200),root)
                report=json.loads((fit/'summary.json').read_text())
                last_checkpoint(report,a.epochs)
                write_new(fit_commit,dict(summary=str(fit/'summary.json'),summary_sha256=sha(fit/'summary.json'),identity=fit_identity))
            saved=json.loads(fit_commit.read_text())
            if saved['identity']!=fit_identity or sha(saved['summary'])!=saved['summary_sha256']:
                raise ValueError('Committed fit input, initialization or summary changed')
            report=json.loads(Path(saved['summary']).read_text())
            last=last_checkpoint(report,a.epochs)
            student=last['student']
            round_report=dict(round_index=round_index,epoch=last['epoch'],student=student,
                student_sha256=last['student_sha256'],fit_summary=saved['summary'])
            round_reports.append(round_report)
            if round_index == 0:
                continue
            if teacher_baseline is None:
                teacher_reference=run_selection_suite(plan,Path(a.plan).resolve(),root,selected_jobs,None,
                    a.selection_profile,selection_strength,seeds,'selection/teacher',environment,a.timeout_seconds,criteria_parameters)
                teacher_baseline=teacher_reference['task_metrics']
            assessment=run_selection_suite(plan,Path(a.plan).resolve(),root,selected_jobs,student,
                a.selection_profile,selection_strength,seeds,f'selection/round_{round_index}',environment,
                a.timeout_seconds,criteria_parameters)
            measured=assessment['task_metrics']
            selection_reports.append(dict(round_report,**assessment,rank=candidate_rank(measured,teacher_baseline),
                teacher_comparison=teacher_comparison(measured,teacher_baseline)))
            print('DAGGER_ROUND_EVALUATED',round_index,student,flush=True)
        selected=max(selection_reports,key=lambda row:tuple(row['rank']))
        student=selected['student']
        training_keys=set(collection_keys)
        selection=dict(rule=SELECTION_RULE,profile=a.selection_profile,seeds=seeds,budget=budget,
            design_keys=selected_keys,teacher=teacher_baseline,candidates=selection_reports,
            selected_round=selected['round_index'],selected_student=student,
            teacher_comparison=selected['teacher_comparison'],comparison_version=COMPARISON_VERSION,
            scope='DAgger final epochs compared on paired seen/held-out morphologies',
            training_design_keys=[k for k in selected_keys if k in training_keys],calibration_keys=reserved)
        selection_path=root/'selection.json'
        if not selection_path.exists():write_new(selection_path,selection)
        elif json.loads(selection_path.read_text())!=json.loads(json.dumps(selection)):
            raise ValueError('Committed selection changed')
        write_new(root/'completed.json',dict(result='DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING',
            requires_candidate_finetuning=True,requires_post_finetuning_evaluation=True,full_bank_evaluated=False,
            evaluation_criteria=criteria,comparison_version=COMPARISON_VERSION,
            protocol=protocol,protocol_sha256=protocol_sha256,selection=str(selection_path),selection_sha256=sha(selection_path),
            student=student,student_sha256=sha(student),selected_round=selected['round_index'],
            rounds=round_reports,
            teacher_updates=0,ppo_updates=0,critic_trained=False,scope=plan['scope'],
            gate=criteria,scope_note='Per-task/per-design teacher comparison of a shared initialization. '
                'Candidate-specific finetuning and independent evaluation remain required; '
                'no distillation acceptance, optimal-control or real-hardware claim.'))



if __name__=='__main__':main()
