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
from contract import load_plan,sha,write_new,structure_key
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


def jobs_for(plan,batch_size):
    result=[]
    for task,spec in plan['teachers'].items():
        ids=spec['design_ids']
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


def diverse_structure_keys(keys, count):
    """Deterministic nested subsets covering leg extremes and motor categories."""
    if count < 1:
        raise ValueError('At least one representative selection structure is required')
    keys = sorted(set(tuple(key) for key in keys))
    if not keys:
        raise ValueError('No common structures for selection')
    lows = [min(key[i] for key in keys) for i in (0, 1)]
    highs = [max(key[i] for key in keys) for i in (0, 1)]
    def features(key):
        result = {('knee', key[2]), ('ankle', key[3])}
        for i in (0, 1):
            if key[i] == lows[i]: result.add((i, 'short'))
            if key[i] == highs[i]: result.add((i, 'long'))
        return result
    def distance(left, right):
        geometry = sum(((left[i]-right[i])/(highs[i]-lows[i]))**2
                       for i in (0, 1) if highs[i] != lows[i])
        return geometry + .25*sum(left[i] != right[i] for i in (2, 3))
    selected, covered = [], set()
    remaining = list(keys)
    for _ in range(min(count, len(keys))):
        # max() retains the lexicographically first key on a complete tie.
        chosen = max(remaining, key=lambda key: (len(features(key)-covered),
                     min(distance(key, old) for old in selected) if selected else 0.))
        selected.append(chosen);covered.update(features(chosen));remaining.remove(chosen)
    return selected


def selection_jobs(plan, count, batch_size):
    """Select identical physical tuples across tasks, even when task IDs/order differ."""
    selected = diverse_structure_keys(plan['selected_structure_keys'], count)
    subset = {'teachers': {}}
    for task, spec in plan['teachers'].items():
        bank = json.loads(Path(spec['bank']).read_text(encoding='utf8'))
        allowed = set(spec['design_ids'])
        by_key = {structure_key(d): d['id'] for d in bank['designs'] if d['id'] in allowed}
        if any(key not in by_key for key in selected):
            raise ValueError(f'Selection structures not paired in {task}')
        subset['teachers'][task] = {'design_ids': [by_key[key] for key in selected]}
    return jobs_for(subset, batch_size), selected


def checkpoint_candidates(report, every_epochs, epochs):
    """Choose by epoch schedule only; validation MSE never nominates actors."""
    checkpoints = report.get('checkpoints', [])
    if every_epochs < 1 or epochs < 1 or len(checkpoints) != epochs:
        raise ValueError('Closed-loop selection requires every epoch checkpoint and a positive schedule')
    by_epoch = {row['epoch']: row for row in checkpoints}
    if set(by_epoch) != set(range(epochs)):
        raise ValueError('Missing or duplicate epoch checkpoints')
    chosen = [by_epoch[i] for i in range(epochs) if (i+1) % every_epochs == 0 or i == epochs-1]
    for candidate in chosen:
        if sha(candidate['student']) != candidate['student_sha256']:
            raise ValueError('Epoch candidate checkpoint changed')
    return chosen


def select_round_candidates(candidates, teacher_screen, teacher_selection, evaluate, promotion_count):
    """Screen all scheduled epochs, then promote by measured closed-loop rank."""
    if not candidates or promotion_count < 1:
        raise ValueError('Selection requires candidates and a positive promotion budget')
    screened = []
    for candidate in sorted(candidates, key=lambda row: row['epoch']):
        screen = dict(evaluate(candidate, 'screen'))
        screen['teacher_comparison'] = teacher_comparison(screen['task_metrics'], teacher_screen)
        screened.append(dict(candidate, screen=screen,
            screen_rank=list(candidate_rank(screen['task_metrics'], teacher_screen))))
    promoted = sorted(screened, key=lambda row: tuple(row['screen_rank']), reverse=True)[:promotion_count]
    for candidate in promoted:
        selection = dict(evaluate(candidate, 'selection'))
        selection['teacher_comparison'] = teacher_comparison(selection['task_metrics'], teacher_selection)
        candidate.update(selection=selection,
            selection_rank=list(candidate_rank(selection['task_metrics'], teacher_selection)))
    # Ascending epoch tie-breaking is independent of the screening rank/order.
    selected = max(sorted(promoted, key=lambda row: row['epoch']), key=lambda row: tuple(row['selection_rank']))
    return dict(candidates=screened, promoted_epochs=[row['epoch'] for row in promoted],
        selected_epoch=selected['epoch'], selected_student=selected['student'],
        selected_student_sha256=selected['student_sha256'], task_metrics=selected['selection']['task_metrics'],
        teacher_comparison=selected['selection']['teacher_comparison'],comparison_version=COMPARISON_VERSION,
        rank=selected['selection_rank'], evaluations=selected['selection']['evaluations'])


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


def selection_budget(plan, screen_jobs, expanded_jobs, all_jobs, *, epochs, every_epochs,
                     rounds, promotion_count, seeds, selection_profile, evaluation_profile,
                     preflight_profile):
    candidate_count = sum((i+1) % every_epochs == 0 or i == epochs-1 for i in range(epochs))
    promotions = min(candidate_count, promotion_count)
    def environment_count(jobs, profile):
        return sum(len(ids)*evaluation_copies(task, plan['teachers'][task], profile) for task, ids in jobs)
    teacher_processes = len(screen_jobs)+len(expanded_jobs)*len(seeds)
    teacher_envs = environment_count(screen_jobs, selection_profile)+environment_count(expanded_jobs, selection_profile)*len(seeds)
    if preflight_profile != selection_profile:
        teacher_processes += len(expanded_jobs)*len(seeds)
        teacher_envs += environment_count(expanded_jobs, preflight_profile)*len(seeds)
    round_processes = candidate_count*len(screen_jobs)+promotions*len(expanded_jobs)*len(seeds)
    round_envs = candidate_count*environment_count(screen_jobs, selection_profile)+promotions*environment_count(expanded_jobs, selection_profile)*len(seeds)
    profiles = ('nominal', 'robust') if evaluation_profile == 'both' else (evaluation_profile,)
    return dict(epochs_per_round=epochs, epoch_candidate_count=candidate_count,
        epoch_candidate_frequency=every_epochs, promoted_candidates=promotions, rounds=rounds,
        screen_design_count=sum(len(ids) for _, ids in screen_jobs)//len(plan['teachers']),
        expanded_design_count=sum(len(ids) for _, ids in expanded_jobs)//len(plan['teachers']),
        screen_seed=seeds[0], expanded_seeds=seeds, steps_per_rollout=1000,
        teacher_rollout_processes=teacher_processes, candidate_rollout_processes_per_round=round_processes,
        total_selection_rollout_processes=teacher_processes+rounds*round_processes,
        selection_environment_steps=1000*(teacher_envs+rounds*round_envs),
        final_test_rollout_processes=2*len(all_jobs)*len(profiles),
        final_test_environment_steps=2000*sum(environment_count(all_jobs, profile) for profile in profiles),
        scope='Exact configured process/environment-step budgets; wall time and memory depend on Isaac runtime')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-designs',type=int,default=20)
    p.add_argument('--collect-steps',type=int,default=1000);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--steps-per-epoch',type=int,default=0,
        help='0 derives steps from the per-design/source sampling budget; positive values set exact update counts')
    p.add_argument('--max-rows-per-design-shard',type=int,default=None,help='Retired; use --samples-per-design-per-epoch')
    p.add_argument('--samples-per-design-per-epoch',type=int,default=32,help='Optimization exposure budget; does not discard collected training rows')
    p.add_argument('--validation-rows-per-design-shard',type=int,default=128)
    p.add_argument('--replay-cache-shards',type=int,default=2)
    p.add_argument('--dagger-rounds',type=int,default=0)
    p.add_argument('--collection-profile',choices=['nominal','robust'],default='robust')
    p.add_argument('--randomization-strength',type=float,default=1.)
    p.add_argument('--initial-randomization-strength',type=float,default=.25,
                   help='First BC round strength; linearly reaches target at the last DAgger round')
    p.add_argument('--selection-profile',choices=['nominal','robust'],default='robust')
    p.add_argument('--candidate-every-epochs',type=int,default=5,help='Screen every Nth epoch and the final epoch; never nominate by action MSE')
    p.add_argument('--screen-designs',type=int,default=6,help='Diverse paired structures for the first candidate screen')
    p.add_argument('--promotion-count',type=int,default=2,help='Screen winners promoted to the expanded selection suite')
    p.add_argument('--selection-designs',type=int,default=12,
                   help='Diverse paired structures for expanded and cross-round closed-loop selection')
    seeds_group=p.add_mutually_exclusive_group()
    seeds_group.add_argument('--selection-seeds',default='20261117,20261119,20261120',help='Comma-separated expanded selection seeds; first seed is also used for screening')
    seeds_group.add_argument('--selection-seed',type=int,help='Compatibility alias for an explicitly single-seed selection budget')
    p.add_argument('--evaluation-profile',choices=['nominal','robust','both'],default='both')
    p.add_argument('--test-seed',type=int,default=20261118)
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
    if min(a.batch_designs,a.collect_steps,a.stride,a.epochs,a.timeout_seconds,a.samples_per_design_per_epoch,a.validation_rows_per_design_shard,a.replay_cache_shards,a.candidate_every_epochs,a.screen_designs,a.promotion_count)<1 or a.steps_per_epoch<0 or a.dagger_rounds<0:
        raise ValueError('Positive finite collection/BC budgets required')
    round_strength(0,a.dagger_rounds,a.initial_randomization_strength,a.randomization_strength,a.collection_profile)
    seeds=[a.selection_seed] if a.selection_seed is not None else [int(value.strip()) for value in a.selection_seeds.split(',')]
    if (a.selection_designs < a.screen_designs or not seeds or len(set(seeds))!=len(seeds)
            or any(seed in (a.test_seed,20261029) for seed in seeds)
            or any(not 0<=seed<2**32 for seed in (*seeds,a.test_seed))):
        raise ValueError('Expanded selection must cover the screen and use distinct seeds independent from final testing')
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
    jobs=jobs_for(plan,a.batch_designs)
    screen_jobs,screen_keys=selection_jobs(plan,a.screen_designs,a.batch_designs)
    selected_jobs,selected_keys=selection_jobs(plan,a.selection_designs,a.batch_designs)
    preflight_profile='robust' if a.collection_profile=='robust' else a.selection_profile
    budget=selection_budget(plan,screen_jobs,selected_jobs,jobs,epochs=a.epochs,every_epochs=a.candidate_every_epochs,
        rounds=a.dagger_rounds+1,promotion_count=a.promotion_count,seeds=seeds,selection_profile=a.selection_profile,
        evaluation_profile=a.evaluation_profile,preflight_profile=preflight_profile)
    protocol=dict(collection=protocol_metadata(a.collection_profile,a.randomization_strength if a.collection_profile=='robust' else 0.),
        initial_strength=a.initial_randomization_strength,dagger_rounds=a.dagger_rounds,
        selection=protocol_metadata(a.selection_profile,a.randomization_strength if a.selection_profile=='robust' else 0.),
        evaluation_profile=a.evaluation_profile,selection_seeds=seeds,test_seed=a.test_seed,
        selection_version='closed_loop_epoch_promotion_design_comparison_v2',selection_budget=budget,
        comparison_version=COMPARISON_VERSION,
        evaluation_criteria=criteria,evaluation_version=GATE_VERSION,
        implementation_sha256=sha(__file__),fit_implementation_sha256=sha(HERE/'fit.py'),
        evaluation_implementation_sha256=sha(HERE/'evaluation.py'))
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
        if sha(complete['student'])!=complete['student_sha256']:raise ValueError('Completed student changed')
        print('SHARED_STUDENT_ALREADY_COMPLETE',complete['student']);return
    from robolab.rough_gpu import query_gpus,candidates,gpu_lease,child_environment
    choices=candidates(query_gpus(),os.environ,min_free_mib=a.gpu_min_free_gb*1024,
                       max_utilization=a.gpu_max_utilization,requested=a.device)
    with gpu_lease(choices,HERE.parents[2]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        environment=child_environment(gpu,os.environ)
        all_paths=[];student=None;round_reports=[];selection_reports=[]
        selection_strength=a.randomization_strength if a.selection_profile=='robust' else 0.
        teacher_screen=run_selection_suite(plan,Path(a.plan).resolve(),root,screen_jobs,None,
            a.selection_profile,selection_strength,seeds[:1],'screen/teacher',environment,a.timeout_seconds,criteria_parameters)
        teacher_selection=run_selection_suite(plan,Path(a.plan).resolve(),root,selected_jobs,None,
            a.selection_profile,selection_strength,seeds,'selection/teacher',environment,a.timeout_seconds,criteria_parameters)
        teacher_baseline=teacher_selection['task_metrics']
        preflight_strength=a.randomization_strength if preflight_profile=='robust' else 0.
        preflight_result=teacher_selection if preflight_profile==a.selection_profile else run_selection_suite(
            plan,Path(a.plan).resolve(),root,selected_jobs,None,preflight_profile,preflight_strength,
            seeds,'preflight/robust',environment,a.timeout_seconds,criteria_parameters)
        preflight=dict(profile=preflight_profile,strength=preflight_strength,seeds=seeds,
            design_keys=selected_keys,**preflight_result,
            result='TEACHER_REFERENCE_EVALUATED',ability_threshold_applied=False,
            limitations='Reference measurements only; low teacher success does not block distillation. '
                        'Execution and data completeness remain required; task competence is not guaranteed.')
        preflight_path=root/'teacher_preflight.json'
        if not preflight_path.exists():write_new(preflight_path,preflight)
        for round_index in range(a.dagger_rounds+1):
            strength=round_strength(round_index,a.dagger_rounds,a.initial_randomization_strength,
                                    a.randomization_strength,a.collection_profile)
            for index,(task,ids) in enumerate(jobs):
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
                    '--steps',str(a.collect_steps),'--stride',str(a.stride),'--seed',str(42+round_index*10000+index),
                    '--profile',a.collection_profile,'--randomization-strength',str(strength),
                    '--headless','--device','cuda:0']
                if task=='run':
                    pos=command.index('--task');del command[pos:pos+2]
                    pos=command.index('--asset-cache-root');del command[pos:pos+2]
                if student:command+=['--student',student]
                print('COLLECT_BEGIN',round_index,index+1,'/',len(jobs),task,len(ids),flush=True)
                child(command,attempt,environment,a.timeout_seconds,root)
                if not (folder/'data.pt').exists() or (folder/'failure.json').exists():raise ValueError('Missing real teacher data')
                commit.parent.mkdir(parents=True,exist_ok=True)
                write_new(commit,dict(folder=str(folder),sha256=sha(folder/'data.pt'),
                    protocol=protocol_metadata(a.collection_profile,strength),round_index=round_index,
                    execution_student_sha256=execution_sha))
                all_paths.append(str(folder/'data.pt'))
                print('COLLECT_COMPLETE',round_index,index+1,'/',len(jobs),flush=True)
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
                if student:command+=['--init-student',student]
                child(command,root/'processes'/f'fit_{round_index}_{datetime.now():%Y%m%d_%H%M%S_%f}',
                      environment,max(a.timeout_seconds,7200),root)
                report=json.loads((fit/'summary.json').read_text())
                checkpoint_candidates(report,a.candidate_every_epochs,a.epochs)
                write_new(fit_commit,dict(summary=str(fit/'summary.json'),summary_sha256=sha(fit/'summary.json'),identity=fit_identity))
            saved=json.loads(fit_commit.read_text())
            if saved['identity']!=fit_identity or sha(saved['summary'])!=saved['summary_sha256']:
                raise ValueError('Committed fit input, initialization or summary changed')
            report=json.loads(Path(saved['summary']).read_text())
            candidates_for_round=checkpoint_candidates(report,a.candidate_every_epochs,a.epochs)
            def evaluate(candidate,tier):
                return run_selection_suite(plan,Path(a.plan).resolve(),root,
                    screen_jobs if tier=='screen' else selected_jobs,candidate['student'],
                    a.selection_profile,selection_strength,seeds[:1] if tier=='screen' else seeds,
                    f'{tier}/round_{round_index}/epoch_{candidate["epoch"]}',environment,a.timeout_seconds,criteria_parameters)
            choice=select_round_candidates(candidates_for_round,teacher_screen['task_metrics'],
                teacher_baseline,evaluate,a.promotion_count)
            student=choice['selected_student']
            choice.update(round_index=round_index,fit_summary=saved['summary'],fit_summary_sha256=saved['summary_sha256'],
                screen_design_keys=screen_keys,selection_design_keys=selected_keys,selection_seeds=seeds)
            choice_path=root/f'selection_round_{round_index}.json'
            if not choice_path.exists():write_new(choice_path,choice)
            elif json.loads(choice_path.read_text())!=json.loads(json.dumps(choice)):
                raise ValueError('Committed within-round closed-loop selection changed')
            round_reports.append(dict(round_index=round_index,fit_summary=saved['summary'],
                selection=str(choice_path),selected_epoch=choice['selected_epoch'],student=student,
                teacher_comparison=choice['teacher_comparison'],comparison_version=COMPARISON_VERSION))
            print('SHARED_STUDENT_CLOSED_LOOP_SELECTED',round_index,choice['selected_epoch'],student,flush=True)
            selection_reports.append(dict(round_index=round_index,epoch=choice['selected_epoch'],student=student,
                student_sha256=sha(student),task_metrics=choice['task_metrics'],rank=choice['rank'],
                teacher_comparison=choice['teacher_comparison'],comparison_version=COMPARISON_VERSION,
                evaluations=choice['evaluations'],round_selection=str(choice_path)))
        selected=max(selection_reports,key=lambda row:tuple(row['rank']))
        student=selected['student']
        selection=dict(rule=SELECTION_RULE,profile=a.selection_profile,seeds=seeds,budget=budget,
            design_keys=selected_keys,teacher=teacher_baseline,candidates=selection_reports,
            selected_round=selected['round_index'],selected_student=student,
            teacher_comparison=selected['teacher_comparison'],comparison_version=COMPARISON_VERSION,
            scope='Each round winner ranked on the same expanded structures/seeds; final test results never select an actor')
        selection_path=root/'selection.json'
        if not selection_path.exists():write_new(selection_path,selection)
        elif json.loads(selection_path.read_text())!=json.loads(json.dumps(selection)):
            raise ValueError('Committed selection changed')
        # Full paired testing uses an independent seed and never changes the selected actor.
        evaluations=[]
        profiles=('nominal','robust') if a.evaluation_profile=='both' else (a.evaluation_profile,)
        for profile in profiles:
            for actor in (None,student):
                evaluations.extend(run_evaluation(plan,Path(a.plan).resolve(),root,jobs,actor,profile,
                    a.randomization_strength if profile=='robust' else 0.,
                    20261029 if profile=='nominal' else a.test_seed,
                    f'test/{profile}',environment,a.timeout_seconds,criteria_parameters))
        nominal=[row for row in evaluations if row['profile']=='nominal']
        robust=[row for row in evaluations if row['profile']=='robust']
        profile_results=summarize_final_evaluations(evaluations,plan['schema']['tasks'],profiles)
        write_new(root/'completed.json',dict(result='DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING',
            requires_candidate_finetuning=True,requires_post_finetuning_evaluation=True,
            nominal_checks=nominal,robust_checks=robust,profile_results=profile_results,
            policy_all_episodes_command_tracking_passed={profile:row['policy_all_episodes_command_tracking_passed']
                for profile,row in profile_results.items()},comparison_version=COMPARISON_VERSION,
            evaluated_profiles=list(profiles),evaluation_criteria=criteria,
            protocol=protocol,protocol_sha256=protocol_sha256,selection=str(selection_path),
            student=student,student_sha256=sha(student),selected_round=selected['round_index'],
            rounds=round_reports,evaluations=evaluations,
            teacher_updates=0,ppo_updates=0,critic_trained=False,scope=plan['scope'],
            gate=criteria,scope_note='Per-task/per-design teacher comparison of a shared initialization. '
                'Candidate-specific finetuning and independent evaluation remain required; '
                'no distillation acceptance, optimal-control or real-hardware claim.'))



if __name__=='__main__':main()
