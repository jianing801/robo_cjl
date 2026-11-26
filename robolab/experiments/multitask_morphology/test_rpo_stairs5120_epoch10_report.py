from pathlib import Path
import csv
import hashlib
import json

import pytest

import rpo_stairs5120_epoch10_report as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_tuning(root):
    root.mkdir()
    settings = dict(epochs=10, minibatches=4, steps_per_env=24,
                    envs_per_task=report.EXPERIMENT_LAYOUT, checkpoint_interval=100)
    manifest = dict(kind='shared_candidate_finetuning_mixed',
                    ppo_algorithm=report.ALGORITHM, experimental_profile=report.PROFILE,
                    updates=1000, seed=20261104,
                    candidate={'structure_key': report.RPO_KEY, 'design_ids': dict.fromkeys(report.TASKS, 'tr_792ca9f47877')},
                    plan_sha256=report.CONTROL_PLAN_SHA,
                    initial_actor_sha256=report.CONTROL_ACTOR_SHA,
                    envs_per_task=report.EXPERIMENT_LAYOUT,
                    ppo_schedule={'epochs': 10, 'minibatches': 4},
                    steps_per_task=24, native_ppo_settings=settings,
                    init_at_random_ep_len=True,
                    collection_protocol={'profile': 'robust', 'strength': 1.})
    write_json(root/'manifest.json', manifest)
    digest = sha(root/'manifest.json')
    rows, step_rows = [], []
    for update in range(1001):
        state = dict(completed_updates=update, manifest_sha256=digest,
                     ppo_algorithm=report.ALGORITHM, state_kind='shared_ppo_learning_state_v4',
                     warmup_completed=10,
                     metrics={'samples': 6144*24, 'collection_seconds': 2., 'update_seconds': 3.})
        means = {task: {'mean_reward': float(update)/100., 'window_episodes': 100}
                 for task in report.CURVES}
        reward = {'session_id': 's', 'mean_reward': means, 'tasks': {
            t: {'reward_samples': n*24} for t,n in report.EXPERIMENT_LAYOUT.items()}}
        state['training_reward'] = reward
        if update % 100 == 0:
            blob = root/'checkpoints'/f'update_{update:06d}_0123456789abcdef0123456789abcdef.pt'
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(str(update).encode())
            state.update(path=blob.name, sha256=sha(blob))
        else:
            state['record_kind']='native_ppo_metrics_only_v1'
        write_json(root/'checkpoints'/f'update_{update:06d}.json', state)
        if update:
            row = {'update': update, 'session_id': 's'}
            for task in report.CURVES:
                prefix = '' if task == 'all' else task+'_'
                row[prefix+'mean_reward'] = float(update)/100.
                row[prefix+'window_episodes'] = 100
            rows.append(row)
            for task,n in report.EXPERIMENT_LAYOUT.items():
                step_rows.append({'update': update, 'task': task, 'session_id':'s',
                                  'reward_samples': n*24})
    write_csv(root/'mean_reward.csv', report.MEAN_FIELDS, rows)
    write_csv(root/'training_reward.csv', report.REWARD_FIELDS, step_rows)
    actor = root/'shared_actor_finetuned.pt'
    actor.write_bytes(b'actor')
    write_json(root/'training_completed.json', dict(result='CANDIDATE_FINETUNED',
        candidate=manifest['candidate'], ppo_updates=1000,
        manifest_sha256=digest, student=str(actor.resolve()), student_sha256=sha(actor),
        checkpoint_sha256=sha(root/'checkpoints/update_001000.json')))
    return root


@pytest.fixture(scope='module')
def training(tmp_path_factory):
    return make_tuning(tmp_path_factory.mktemp('stairs5120_test')/'tuning')


def test_archived_control_and_exact_experiment_evidence(training, tmp_path):
    control = report.load_control(report.DEFAULT_CONTROL)
    experiment = report.load_experiment(training)
    assert len(control['rows']) == len(experiment['rows']) == 1000
    assert control['rows'][-1]['cumulative_samples']['stairs'] == 32784000
    assert experiment['rows'][-1]['cumulative_samples']['stairs'] == 122880000
    assert experiment['rows'][-1]['cumulative_samples']['backpack'] == 12288000
    assert experiment['rows'][-1]['cumulative_samples']['all'] == 147456000
    assert experiment['rows'][-1]['cumulative_compute_seconds'] == 5000
    summary=report.compare(control, experiment, 8227.0, 6000.0)
    assert summary['runs']['experiment']['wall_seconds'] == 6000
    assert summary['runs']['control']['wall_seconds'] == 8227.0
    assert summary['curves']['stairs']['experiment']['last100_mean'] == pytest.approx(9.505)
    out=tmp_path/'report'
    before=sha(training/'mean_reward.csv')
    report.write_report(out, control, experiment, summary)
    assert len(list(csv.DictReader((out/'curve_comparison.csv').open(encoding='utf8')))) == 1000
    assert (out/'mean_reward_comparison.png').stat().st_size > 1000
    assert (out/'mean_reward_compute_time.png').stat().st_size > 1000
    assert (out/'mean_reward_samples.png').stat().st_size > 1000
    assert sha(training/'mean_reward.csv') == before


def test_missing_metric_or_mismatched_rows_rejected(training):
    p=training/'mean_reward.csv'
    original=p.read_bytes()
    try:
        lines=original.splitlines(keepends=True)
        p.write_bytes(b''.join(lines[:-1]))
        with pytest.raises(ValueError, match='1000|1..1000|update'):
            report.load_experiment(training)
        p.write_bytes(original)
        lines=original.splitlines(keepends=True)
        lines[15]=lines[15].replace(b',0.15,100', b',99,100', 1)
        p.write_bytes(b''.join(lines))
        with pytest.raises(ValueError, match='receipt'):
            report.load_experiment(training)
    finally:
        p.write_bytes(original)


def test_mismatched_sample_allocation_rejected(training):
    p=training/'training_reward.csv'
    original=p.read_bytes()
    try:
        p.write_bytes(original.replace(b',122880,', b',1228801,', 1))
        with pytest.raises(ValueError, match='sample'):
            report.load_experiment(training)
    finally:
        p.write_bytes(original)


def test_negative_reference_and_stable_windows():
    result=report.convergence([(-10.)]*100 + [-4.] * 100, -5.)
    assert result['threshold'] == pytest.approx(-5.25)
    assert result['confirmed_update']==189
    assert report.convergence([-10.]*100,-5.)['confirmed_update'] is None

