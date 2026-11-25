"""Compare original-RPO native joint PPO to its preserved legacy experiment.

Only a new output directory is written. Historical measurements supply fixed
diagnostic denominators, not additional observations for an active BO search.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


TASKS = ('stairs', 'backpack', 'run')
MILESTONES = (0, 100, 200, 500, 800, 1000)


def diagnostic_performance(pair, scoring, expected_steps):
    rows = {}
    for task in TASKS:
        trial = pair['tasks'][task]
        raw = {}
        for metric in scoring['metrics'][task]:
            # A mechanical CoT with nonpositive travel is undefined, never zero.
            value = (trial.get('observed') or {}).get('absolute_mechanical_cot' if metric == 'cot' else metric)
            raw[metric] = (value if type(value) in (int, float) and math.isfinite(value) and value >= 0
                           else None)
            if metric == 'cot' and trial['forward_m'] <= 0:
                raw[metric] = None
        normalized = {k: (v / scoring['metrics'][task][k]['baseline'] if v is not None else None)
                      for k, v in raw.items()}
        full = trial['steps'] == expected_steps
        cost = (sum(normalized[k] * scoring['metrics'][task][k]['weight'] for k in normalized)
                if full and all(v is not None for v in normalized.values()) else None)
        rows[task] = dict(raw=raw, normalized=normalized, task_objective=cost,
                         measured_steps=trial['steps'], duration_s=trial['steps']*trial['dt_s'],
                         full_duration=full, forward_m=trial['forward_m'], reason=trial.get('reason'),
                         measurement_status=trial['status'])
    total = (sum(scoring['task_weights'][task]*rows[task]['task_objective'] for task in TASKS)
             if all(rows[task]['task_objective'] is not None for task in TASKS) else None)
    return dict(tasks=rows, mean_objective=total)


def status(root, phase, **fields):
    data = dict(phase=phase, pid=os.getpid(), timestamp=time.time(), **fields)
    tmp = root/'.status.writing'
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(root/'status.json')
    print('NATIVE_CONVERGENCE', json.dumps(data, ensure_ascii=False), flush=True)


def check_pause(root, tuning=None):
    if (root/'USER_PAUSED').exists() or (tuning is not None and (tuning/'USER_PAUSED').exists()):
        raise KeyboardInterrupt('USER_PAUSED')


def verify_blob(bo, tuning, record, manifest_digest):
    import torch
    folder = tuning/'checkpoints'
    path = folder/record['path']
    if path.parent != folder or path.is_symlink() or record['manifest_sha256'] != manifest_digest or bo.sha(path) != record['sha256']:
        raise ValueError('Native milestone checkpoint identity changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['kind'] != 'shared_ppo_learning_state_v3' or state['manifest_sha256'] != manifest_digest
            or state['completed_updates'] != record['completed_updates']):
        raise ValueError('Native milestone checkpoint metadata differs')
    return state


def run_training(bo, a, root, tuning, source, initial, candidate, updates):
    from robolab.rough_gpu import candidates, child_environment, gpu_lease, query_gpus
    from finetune_settings import NATIVE_PPO_ALGORITHM, PPO_SCHEDULE
    from finetune_state import checkpoint_history
    check_pause(root, tuning)
    if not (tuning/'training_completed.json').exists():
        log_dir = root/'training_logs'/uuid4().hex
        command = [sys.executable, '-B', '-u', bo.SHARED/'mixed_finetune.py', '--plan', source,
                   '--student', initial, '--output', tuning, '--design-id', candidate['design_ids']['stairs'],
                   '--updates', str(updates), '--num-envs', '4096', '--seed', str(a.seed),
                   '--profile', 'robust', '--randomization-strength', '1', '--init-at-random-ep-len',
                   '--headless', '--device', 'cuda:0']
        if (tuning/'manifest.json').exists():
            command.append('--resume')
        status(root, 'smoke' if updates == 1 else 'training', log_dir=str(log_dir), updates=updates)
        devices = candidates(query_gpus(), os.environ, min_free_mib=12800,
                             max_utilization=100, requested=f'cuda:{a.gpu}')
        with gpu_lease(devices, bo.REPO/'robolab/logs/rpo_rough_multimorph/gpu_locks') as gpu:
            bo.run_worker(command, log_dir, child_environment(gpu, os.environ), 12*3600, root)
    final = bo.completed_actor(tuning, source, initial, candidate, updates,
                               ppo_schedule=PPO_SCHEDULE, mixed_bo=True, ppo_algorithm=NATIVE_PPO_ALGORITHM)
    manifest = bo.read(tuning/'manifest.json')
    digest = bo.sha(tuning/'manifest.json')
    history = checkpoint_history(tuning, digest)
    if len(history) != updates+1 or manifest.get('ppo_algorithm') != NATIVE_PPO_ALGORITHM:
        raise ValueError('Native PPO update history or algorithm identity differs')
    return final, manifest, digest, history


def write_report(bo, root, rows, scoring, legacy, tuning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = dict(milestones=rows, scoring_sha256=bo.sha(root/'comparison_scoring.json'),
                   task_weights=scoring['task_weights'], metrics=scoring['metrics'],
                   note='Diagnostic ratios only, using the same historical RPO 100-update denominators. '
                        'Missing CoT/full-duration values stay undefined. Not a new BO scoring library.')
    bo.freeze(root/'convergence_four_metrics.json', summary)
    keys = list(scoring['metrics']['stairs'])
    with (root/'convergence_four_metrics.csv').open('w', encoding='utf8', newline='') as stream:
        fields = ['update','task','forward_m','duration_s',*keys,*['ratio_'+k for k in keys], 'task_cost','overall_cost']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            for task, item in row['tasks'].items():
                writer.writerow(dict(update=row['updates'], task=task, forward_m=item['forward_m'],
                    duration_s=item['duration_s'], **item['raw'], **{'ratio_'+k:v for k,v in item['normalized'].items()},
                    task_cost=item['task_objective'], overall_cost=row['mean_objective']))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    old = bo.read(legacy/'analysis_four_metrics/convergence_four_metrics.json')['milestones']
    for ax, task in zip(axes.flat, (*TASKS, 'overall'), strict=True):
        def val(row): return row['mean_objective'] if task == 'overall' else row['tasks'][task]['task_objective']
        for points, label in ((old, 'Legacy simplified PPO'), (rows, 'Native joint PPOAMP')):
            ax.plot([r['updates'] for r in points], [math.nan if val(r) is None else val(r) for r in points],
                    marker='o', label=label)
        ax.axhline(1, color='gray', linestyle=':'); ax.set_title(task); ax.set_xlabel('PPO updates')
        ax.set_ylabel('Cost / original reference'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(root/'convergence_four_metrics.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for run, label in ((legacy/'tuning','Legacy simplified PPO'), (tuning,'Native joint PPOAMP')):
        with (run/'mean_reward.csv').open(encoding='utf8', newline='') as stream:
            values = list(csv.DictReader(stream))
        for ax, task in zip(axes.flat, ('all',*TASKS), strict=True):
            key = ('mean_reward' if task=='all' else task+'_mean_reward')
            ax.plot([int(r['update']) for r in values], [float(r[key]) if r[key] else math.nan for r in values],
                    alpha=.85,label=label,lw=.8)
            ax.set_title(task); ax.set_xlabel('PPO updates'); ax.set_ylabel('Mean reward'); ax.grid(alpha=.2)
    for ax in axes.flat: ax.legend()
    fig.savefig(root/'mean_reward.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax, task in zip(axes, TASKS, strict=True):
        for key in keys:
            values=[r['tasks'][task]['normalized'][key] for r in rows]
            ax.plot([r['updates'] for r in rows],[math.nan if v is None else v for v in values],
                    marker='o',label=key)
        ax.axhline(1,color='gray',linestyle=':'); ax.set_title(task); ax.set_xlabel('PPO updates')
        ax.set_ylabel('Metric / original reference'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(root/'four_metric_details.png',dpi=170); plt.close(fig)
    from finetune_state import checkpoint_history
    history=checkpoint_history(tuning,bo.sha(tuning/'manifest.json'))[1:]
    with (root/'ppo_diagnostics.csv').open('w',encoding='utf8',newline='') as stream:
        keys=['learning_rate','mean_action_std','value','surrogate','entropy','collection_seconds','update_seconds','torch_peak_mib']
        task_keys=('kl','clip_fraction','explained_variance')
        fields=['update',*keys,*[f'{task}_{k}' for task in TASKS for k in task_keys]]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for receipt in history:
            metrics=receipt['metrics']
            writer.writerow(dict(update=receipt['completed_updates'],**{key:metrics.get(key) for key in keys},
                **{f'{task}_{key}':metrics['tasks'][task][key] for task in TASKS for key in task_keys}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--resume-paused', action='store_true')
    a = parser.parse_args()
    root, legacy = a.output.resolve(), a.legacy_run.resolve()
    if root == legacy or root.is_relative_to(legacy):
        raise ValueError('Use a new output outside the preserved legacy experiment')
    import bo
    import torch
    from finetune_state import select_candidate
    from finetune_settings import NATIVE_PPO_ALGORITHM
    from student import load_student, save_student
    old = bo.read(legacy/'manifest.json')
    source, initial = Path(old['source_plan']), Path(old['initial_actor'])
    a.seed = old['seed']
    if bo.sha(source) != old['source_plan_sha256'] or bo.sha(initial) != old['initial_actor_sha256']:
        raise ValueError('Legacy plan or round-five student changed')
    plan, candidate = select_candidate(bo.load_plan(source), design_id=old['candidate']['design_ids']['stairs'])
    if candidate != old['candidate'] or candidate['structure_key'] != [.25,.3,'DM-J10010L-2EC','DM-J4340P-2EC']:
        raise ValueError('Original RPO morphology changed')
    conditions = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    scoring = bo.read(legacy/'analysis_four_metrics/scoring.json')
    if (bo.digest(conditions) != old['conditions_sha256'] or scoring['conditions_sha256'] != old['conditions_sha256']
            or any(scoring['metrics'][task][k]['weight'] != v
                   for task, weights in bo.read(bo.FOUR_METRIC_WEIGHTS_PATH).items() for k,v in weights.items())):
        raise ValueError('Evaluation conditions or weights changed')
    a.output = root
    root.mkdir(parents=True, exist_ok=True)
    with bo.run_lock(root):
        if a.resume_paused:
            for folder in (root, root/'smoke', root/'tuning'):
                if folder.exists(): bo.resume_pause(folder,True)
        check_pause(root)
        bo.freeze(root/'comparison_scoring.json', scoring)
        commit = subprocess.check_output(['git','-C',str(bo.REPO),'rev-parse','HEAD'], text=True).strip()
        bo.freeze(root/'manifest.json', dict(kind='native_joint_ppo_convergence',runtime_commit=commit,
            ppo_algorithm=NATIVE_PPO_ALGORITHM, candidate=candidate, source_plan_sha256=bo.sha(source),
            initial_actor_sha256=bo.sha(initial), conditions_sha256=bo.digest(conditions), seed=a.seed,
            comparison_scoring_sha256=bo.sha(root/'comparison_scoring.json'), legacy_manifest_sha256=bo.sha(legacy/'manifest.json'),
            requested_updates=1000, evaluation_updates=list(MILESTONES)))
        smoke, sm, sd, sh = run_training(bo,a,root,root/'smoke',source,initial,candidate,1)
        smoke_state = verify_blob(bo,root/'smoke',sh[-1],sd)
        if smoke_state['completed_updates'] != 1 or not sh[-1].get('metrics'):
            raise ValueError('Native smoke did not commit one real PPO update')
        metrics=sh[-1]['metrics']
        if (metrics['samples'] != 4096*24 or set(metrics['tasks']) != set(TASKS)
                or not all(math.isfinite(metrics['tasks'][task]['kl']) for task in TASKS)):
            raise ValueError('Native smoke has incomplete rollout or KL evidence')
        del smoke_state
        status(root,'smoke_passed',actor_sha256=bo.sha(smoke),metrics=sh[-1]['metrics'])
        if a.smoke_only: return
        tuning = root/'tuning'
        final, manifest, digest, history = run_training(bo,a,root,tuning,source,initial,candidate,1000)
        model, _ = load_student(initial,'cpu')
        start = verify_blob(bo,tuning,history[0],digest)
        if any(not torch.equal(value,start['actor'][key]) for key,value in model.state_dict().items()):
            raise ValueError('Critic warmup changed the distilled Actor')
        del start
        rows=[]
        for update in MILESTONES:
            check_pause(root,tuning)
            state=verify_blob(bo,tuning,history[update],digest)
            model.load_state_dict(state['actor'],strict=True); del state
            folder=root/'evaluation'/f'update_{update:06d}'; folder.mkdir(parents=True,exist_ok=True)
            actor=final if update==1000 else folder/'actor.pt'
            if not actor.exists():
                save_student(actor,model,ppo_updates=update,initial_actor_sha256=bo.sha(initial),
                    collection_protocol=manifest['collection_protocol'],manifest_sha256=digest)
            exported,metadata=load_student(actor,'cpu')
            if metadata['ppo_updates']!=update or any(not torch.equal(value,exported.state_dict()[key]) for key,value in model.state_dict().items()):
                raise ValueError('Published milestone differs from committed Actor')
            status(root,'evaluating',update=update)
            bo.freeze(folder/'manifest.json',dict(updates=update,actor_sha256=bo.sha(actor),
                checkpoint_sha256=history[update]['sha256'],conditions_sha256=bo.digest(conditions),candidate=candidate))
            measured=bo.evaluate_native(folder,tuning/'plan.json',plan,actor,bo.PHYSICS_CONDITIONS,conditions,a.gpu,1800,pause_root=root)
            pairs=bo.paired_trials(measured,conditions)
            bo.check_panel_pairs(pairs,bo.required_panel_pairs(conditions))
            (pair,)=pairs
            report=dict(kind=bo.REPORT_VERSION, domain='simulation',candidate=candidate,ppo_updates=update,
                student=str(actor),student_sha256=bo.sha(actor),conditions=conditions,conditions_sha256=bo.digest(conditions),
                evaluation_manifest_sha256=bo.sha(folder/'manifest.json'),pairs=pairs,
                measurements=[r['evidence'] for r in measured])
            bo.freeze(folder/'bo_evaluation.json',report)
            row=dict(updates=update,**diagnostic_performance(pair,scoring,
                round(conditions['duration_s']/conditions['measurement_dt_s'])),actor_sha256=bo.sha(actor),
                report_sha256=bo.sha(folder/'bo_evaluation.json'))
            bo.freeze(folder/'performance.json',row); rows.append(row)
        write_report(bo,root,rows,scoring,legacy,tuning)
        status(root,'complete',updates=1000,evaluated=list(MILESTONES),final_actor_sha256=bo.sha(final))


if __name__ == '__main__':
    main()
