"""Compare the isolated stairs-biased V4 experiment to the archived V4 control.

Reading the curves never starts a simulator, evaluates a policy, rewrites
training data, or publishes observations to BO.  The two training protocols
differ in both environment allocation and epochs, so this is a joint contrast.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
DEFAULT_CONTROL = HERE.parents[2]/'docs/reference/rpo_ppo_correction_20261126/v4_control'
PROFILE = 'v4_stairs5120_epoch10'
ALGORITHM = 'shared_native_ppoamp_expert_prior_v4'
UPDATES = 1000
STEPS = 24
TASKS = ('stairs', 'backpack', 'run')
CURVES = ('all', *TASKS)
CONTROL_LAYOUT = {'stairs': 1366, 'backpack': 1365, 'run': 1365}
EXPERIMENT_LAYOUT = {'stairs': 5120, 'backpack': 512, 'run': 512}
RPO_KEY = [.25, .30, 'DM-J10010L-2EC', 'DM-J4340P-2EC']
CONTROL_PLAN_SHA = 'f04cd26293b0d5ce286b09316b98dc282a7b3ac4244af99e71fe3de570e1a3e9'
CONTROL_ACTOR_SHA = 'ccc8fae759cf0eff5db8df43b908fd29eb8cc12ce4216c467ed1d2a8759d4c90'
CONTROL_CSV_SHA = {
    'mean_reward.csv':'5f9efcb7014a3cdcba729bc4d96ddf50f8466da6c56344358330c302ad10661d',
    'ppo_diagnostics.csv':'31a5f64a2170263bdb5df03e1167689a4c905de06778050e6077ffcd3c3567af',
    'training_reward.csv':'8a8833941ef5872703622806fe39fbb07c204df612df4bcc08c3028a057f8dae',
}
MEAN_FIELDS = ['update', 'session_id', 'mean_reward', 'window_episodes']
for task in TASKS:
    MEAN_FIELDS.extend([task+'_mean_reward',task+'_window_episodes'])
REWARD_FIELDS = ['update','task','session_id','reward_samples',
    'mean_environment_step_reward','mean_ppo_step_reward','completed_episodes',
    'timeout_episodes','terminated_episodes','mean_environment_episode_return',
    'mean_ppo_episode_return']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def csv_rows(path, fields=None):
    with Path(path).open(encoding='utf8', newline='') as stream:
        reader=csv.DictReader(stream)
        if fields is not None and reader.fieldnames!=fields:
            raise ValueError(f'Unexpected raw CSV columns: {path}')
        return list(reader)


def number(value, *, allow_blank=False):
    if allow_blank and value in ('',None):
        return None
    try:
        result=float(value)
    except (TypeError,ValueError):
        raise ValueError(f'Invalid numeric value: {value!r}') from None
    if not math.isfinite(result):
        raise ValueError(f'Non-finite numeric value: {value!r}')
    return result


def check_sequence(rows, label):
    if [int(row['update']) for row in rows]!=list(range(1, UPDATES+1)):
        raise ValueError(f'{label}: expected exactly updates 1..1000 in order')


def read_reward_rows(root, layout):
    means=csv_rows(root/'mean_reward.csv', MEAN_FIELDS)
    check_sequence(means, 'Mean reward')
    steps=csv_rows(root/'training_reward.csv', REWARD_FIELDS)
    actual={}
    if len(steps)!=UPDATES*len(TASKS):
        raise ValueError('Expected exactly 3000 task/update sample records')
    for row in steps:
        key=(int(row['update']),row['task'])
        if key in actual or not 1 <= key[0] <= UPDATES or key[1] not in layout:
            raise ValueError('Invalid duplicate/unknown sample record')
        if int(row['reward_samples'])!=layout[key[1]]*STEPS:
            raise ValueError(f'Measured sample count differs from allocation: {key}')
        if row['session_id']!=means[key[0]-1]['session_id']:
            raise ValueError(f'Sample and reward session differ: {key}')
        actual[key]=int(row['reward_samples'])
    return means,actual


def rows_with_time(means,actual,timings):
    out=[]
    samples=dict.fromkeys(CURVES,0)
    compute=0.
    for row, durations in zip(means,timings,strict=True):
        update=int(row['update'])
        rewards={}
        windows={}
        for task in CURVES:
            p='' if task=='all' else task+'_'
            rewards[task]=number(row[p+'mean_reward'],allow_blank=True)
            windows[task]=int(row[p+'window_episodes'])
            if task!='all':
                samples[task]+=actual[update,task]
        samples['all']=sum(samples[t] for t in TASKS)
        if any(not 0<=v<=100 for v in windows.values()):
            raise ValueError(f'Invalid episode-window count at update {update}')
        collect,train=(number(v) for v in durations)
        if min(collect,train)<0:
            raise ValueError(f'Negative compute duration at update {update}')
        compute+=collect+train
        equal_mean=(statistics.mean(rewards[t] for t in TASKS)
                    if all(rewards[t] is not None for t in TASKS) else None)
        out.append(dict(update=update,mean_reward=rewards,window_episodes=windows,
            equal_task_mean_reward=equal_mean,cumulative_samples=dict(samples),
            cumulative_compute_seconds=compute,collection_seconds=collect,update_seconds=train))
    return out


def load_control(root):
    root=Path(root).resolve()
    # Normalize checkout line endings only; no alternate run can silently
    # become the archived reference just by matching its CSV schema.
    for name,digest in CONTROL_CSV_SHA.items():
        content=(root/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(content).hexdigest()!=digest:
            raise ValueError(f'Archived V4 control CSV differs: {name}')
    means,actual=read_reward_rows(root,CONTROL_LAYOUT)
    diag=csv_rows(root/'ppo_diagnostics.csv')
    check_sequence(diag,'V4 control diagnostics')
    if any(k not in diag[0] for k in ('collection_seconds','update_seconds')):
        raise ValueError('Control lacks measured optimizer/collection duration')
    rows=rows_with_time(means,actual,[(r['collection_seconds'],r['update_seconds']) for r in diag])
    return dict(root=str(root),layout=CONTROL_LAYOUT,epochs=5,rows=rows,
                hashes={name:sha(root/name) for name in CONTROL_CSV_SHA})


def validate_experiment_manifest(manifest):
    expected={'kind':'shared_candidate_finetuning_mixed','ppo_algorithm':ALGORITHM,
        'experimental_profile':PROFILE,'updates':UPDATES,'seed':20261104,
        'plan_sha256':CONTROL_PLAN_SHA,'initial_actor_sha256':CONTROL_ACTOR_SHA,
        'envs_per_task':EXPERIMENT_LAYOUT,'ppo_schedule':{'epochs':10,'minibatches':4},
        'steps_per_task':STEPS,'init_at_random_ep_len':True}
    for field,value in expected.items():
        if manifest.get(field)!=value:
            raise ValueError(f'Experiment manifest differs at {field}')
    candidate=manifest.get('candidate',{})
    if candidate.get('structure_key')!=RPO_KEY or candidate.get('design_ids')!=dict.fromkeys(TASKS,'tr_792ca9f47877'):
        raise ValueError('Experiment must use the same predicted original-RPO structure')
    settings=manifest.get('native_ppo_settings',{})
    for field,value in {'epochs':10,'minibatches':4,'steps_per_env':STEPS,
                        'envs_per_task':EXPERIMENT_LAYOUT,'checkpoint_interval':100}.items():
        if settings.get(field)!=value:
            raise ValueError(f'Experiment training settings differ at {field}')
    if any(field in settings for field in ('fixed_expert_std_tasks','actor_gradient_projection','actor_gradient_weights')):
        raise ValueError('Experiment unexpectedly enabled a different PPO correction')
    protocol=manifest.get('collection_protocol',{})
    if protocol.get('profile')!='robust' or protocol.get('strength')!=1.:
        raise ValueError('Experiment changed the robust randomization protocol')


def load_experiment(root):
    root=Path(root).resolve()
    mp=root/'manifest.json'
    manifest=read_json(mp)
    validate_experiment_manifest(manifest)
    digest=sha(mp)
    completed=read_json(root/'training_completed.json')
    actor=root/'shared_actor_finetuned.pt'
    expected={'result':'CANDIDATE_FINETUNED','candidate':manifest['candidate'],'ppo_updates':UPDATES,
              'manifest_sha256':digest,'student':str(actor),'student_sha256':sha(actor),
              'checkpoint_sha256':sha(root/'checkpoints/update_001000.json')}
    if completed!=expected:
        raise ValueError('Completed training receipt or final Actor identity differs')
    means,actual=read_reward_rows(root,EXPERIMENT_LAYOUT)
    folder=root/'checkpoints'
    paths=sorted(folder.glob('update_*.json'))
    if [p.name for p in paths]!=[f'update_{i:06d}.json' for i in range(UPDATES+1)]:
        raise ValueError('Expected 1001 dense committed update receipts including zero')
    timings=[]
    milestone_hashes={}
    receipts=hashlib.sha256()
    for i,path in enumerate(paths):
        raw=path.read_bytes()
        receipts.update(hashlib.sha256(raw).digest())
        record=json.loads(raw)
        if (record.get('completed_updates')!=i or record.get('manifest_sha256')!=digest
                or record.get('ppo_algorithm')!=ALGORITHM or record.get('warmup_completed')!=10
                or record.get('state_kind')!='shared_ppo_learning_state_v4'):
            raise ValueError(f'Wrong committed training receipt: {path.name}')
        if i%100==0:
            blob=folder/record.get('path','')
            if (blob.parent!=folder or not blob.is_file() or blob.is_symlink()
                    or record.get('sha256')!=sha(blob)):
                raise ValueError(f'Missing or changed 100-update full checkpoint: {i}')
            milestone_hashes[str(i)]=record['sha256']
        elif (record.get('record_kind')!='native_ppo_metrics_only_v1'
              or 'path' in record or 'sha256' in record):
            raise ValueError(f'Expected a metrics-only receipt at update {i}')
        if not i:
            continue
        metrics=record.get('metrics',{})
        if metrics.get('samples')!=sum(EXPERIMENT_LAYOUT.values())*STEPS:
            raise ValueError(f'Measured total sample count differs at update {i}')
        reward=record.get('training_reward',{})
        if reward.get('session_id')!=means[i-1]['session_id']:
            raise ValueError(f'Reward receipt session differs at update {i}')
        for task in TASKS:
            if reward.get('tasks',{}).get(task,{}).get('reward_samples')!=actual[i,task]:
                raise ValueError(f'Measured task sample receipt differs at {i}/{task}')
        for task in CURVES:
            prefix='' if task=='all' else task+'_'
            observed=reward.get('mean_reward',{}).get(task,{})
            row=means[i-1]
            if (observed.get('mean_reward')!=number(row[prefix+'mean_reward'],allow_blank=True)
                    or observed.get('window_episodes',0)!=int(row[prefix+'window_episodes'])):
                raise ValueError(f'Mean reward CSV differs from committed receipt at {i}/{task}')
        timings.append((metrics.get('collection_seconds'),metrics.get('update_seconds')))
    rows=rows_with_time(means,actual,timings)
    return dict(root=str(root),layout=EXPERIMENT_LAYOUT,epochs=10,rows=rows,
        milestone_hashes=milestone_hashes,manifest_sha256=digest,
        receipts_digest=receipts.hexdigest(),final_actor_sha256=completed['student_sha256'],
        hashes={name:sha(root/name) for name in ('mean_reward.csv','training_reward.csv','training_completed.json')})


def convergence(values, baseline_last100):
    """First 50 consecutive trailing-50 means above the fixed control target."""
    target=None if baseline_last100 is None else baseline_last100-.05*max(abs(baseline_last100),1.e-12)
    streak=0
    if target is not None:
        for i in range(49,len(values)):
            window=values[i-49:i+1]
            qualifies=(all(x is not None for x in window) and statistics.mean(window)>=target)
            streak=streak+1 if qualifies else 0
            if streak==50:
                return dict(threshold=target,confirmed_update=i+1)
    return dict(threshold=target,confirmed_update=None)


def end_statistics(values, window=100):
    tail=values[-window:]
    if len(tail)<window or any(x is None for x in tail):
        return dict(last100_mean=None,last100_std=None)
    return dict(last100_mean=statistics.mean(tail),last100_std=statistics.pstdev(tail))


def curve_values(run, key):
    return [r['equal_task_mean_reward'] if key=='equal_task' else r['mean_reward'][key] for r in run['rows']]


def run_summary(run,wall_seconds):
    last=run['rows'][-1]
    duration=None if wall_seconds is None else number(wall_seconds)
    if duration is not None and duration<=0:
        raise ValueError('Measured wall time must be positive')
    return dict({k:v for k,v in run.items() if k!='rows'},
        formal_compute_seconds=last['cumulative_compute_seconds'],
        formal_collection_seconds=sum(r['collection_seconds'] for r in run['rows']),
        formal_update_seconds=sum(r['update_seconds'] for r in run['rows']),
        wall_seconds=duration,formal_total_samples=last['cumulative_samples'])


def compare(control, experiment, control_wall_seconds=None, experiment_wall_seconds=None):
    summaries={}
    for curve in (*CURVES,'equal_task'):
        values=[curve_values(run,curve) for run in (control,experiment)]
        base=end_statistics(values[0])['last100_mean']
        entry={}
        for run,vs,label in zip((control,experiment),values,('control','experiment'),strict=True):
            result=dict(end_statistics(vs),**convergence(vs,base))
            index=result['confirmed_update']
            if index is not None:
                row=run['rows'][index-1]
                result.update(confirmed_compute_seconds=row['cumulative_compute_seconds'],
                              confirmed_samples=row['cumulative_samples']['all' if curve=='equal_task' else curve])
            else:
                result.update(confirmed_compute_seconds=None,confirmed_samples=None)
            entry[label]=result
        summaries[curve]=entry
    return dict(kind='rpo_v4_stairs5120_epoch10_training_curve_comparison',
        comparison_only=True,environment_and_epochs_changed_together=True,
        runs={'control':run_summary(control,control_wall_seconds),
              'experiment':run_summary(experiment,experiment_wall_seconds)},
        criterion='Control last-100 mean minus 5% of its absolute value; 50 consecutive trailing-50 means reaching threshold. Confirmation update is the last qualifying window. This measures reaching a reference reward level, not proof of convergence or fixed-condition performance.',
        notes=['Raw Mean reward is the original unmodified rolling completed-episode statistic.',
               'Overall raw Mean reward uses a mixture of tasks; changed task population makes it not a like-for-like metric. Per-task curves are primary, equal-task mean is only a derived diagnostic.',
               'Sample/time axes count formal updates only. Compute seconds are collection + PPO optimizer timings; exclude initialization, critic warmup, checkpoint I/O and idle intervals. Independently measured worker wall time may include those costs.',
               'A single paired experiment with both layout and epochs changed cannot identify either effect independently or prove statistical significance. No fixed-condition evaluation was performed.'],
        curves=summaries)


def write_report(out, control, experiment, summary):
    out=Path(out).resolve()
    for run in (control,experiment):
        root=Path(run['root']).resolve()
        if out==root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError('Use independent report output outside training/control input directories')
    out.mkdir(parents=True,exist_ok=False)
    columns=['update']
    for label in ('control','experiment'):
        columns += [label+'_'+task+'_mean_reward' for task in (*CURVES,'equal_task')]
        columns += [label+'_'+task+'_cumulative_samples' for task in CURVES]
        columns += [label+'_cumulative_compute_seconds',label+'_collection_seconds',label+'_update_seconds']
    with (out/'curve_comparison.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for c,e in zip(control['rows'],experiment['rows'],strict=True):
            result={'update':c['update']}
            for label,row in (('control',c),('experiment',e)):
                result.update({label+'_'+t+'_mean_reward':row['mean_reward'][t] for t in CURVES})
                result[label+'_equal_task_mean_reward']=row['equal_task_mean_reward']
                result.update({label+'_'+t+'_cumulative_samples':row['cumulative_samples'][t] for t in CURVES})
                for key in ('cumulative_compute_seconds','collection_seconds','update_seconds'):
                    result[label+'_'+key]=row[key]
            writer.writerow(result)
    (out/'convergence_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for axis,filename in (('update','mean_reward_comparison.png'),('compute','mean_reward_compute_time.png'),('samples','mean_reward_samples.png')):
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for ax,curve in zip(axes.flat,CURVES,strict=True):
            for run,label in ((control,'V4 4096 env / 5 epochs'),(experiment,'V4 6144 env / 10 epochs')):
                if axis=='update':
                    xs=[r['update'] for r in run['rows']]
                    xlabel='Formal PPO updates'
                elif axis=='compute':
                    xs=[r['cumulative_compute_seconds']/60 for r in run['rows']]
                    xlabel='Collection + optimizer time (min)'
                else:
                    xs=[r['cumulative_samples'][curve]/1.e6 for r in run['rows']]
                    xlabel='Cumulative formal task samples (million)'
                ax.plot(xs,curve_values(run,curve),label=label,linewidth=.85,alpha=.85)
            ax.set_title('Overall (changed task mixture)' if curve=='all' else curve.capitalize())
            ax.set_xlabel(xlabel);ax.set_ylabel('Original Mean reward');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(out/filename,dpi=170)
        plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning',type=Path,required=True)
    parser.add_argument('--control',type=Path,default=DEFAULT_CONTROL)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control-wall-seconds',type=float)
    parser.add_argument('--experiment-wall-seconds',type=float)
    a=parser.parse_args(argv)
    control,experiment=load_control(a.control),load_experiment(a.tuning)
    summary=compare(control,experiment,a.control_wall_seconds,a.experiment_wall_seconds)
    write_report(a.output,control,experiment,summary)
    print('RPO_EXPERIMENT_CURVES_READY',a.output.resolve(),flush=True)


if __name__=='__main__':
    main()
