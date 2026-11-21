"""Formula, paired failure handling and BO proposal regressions."""
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bo
from shared_bo_metrics import DT, KNEE_ANKLE, TRIAL_VERSION, digest, objective, paired_trials, score_trial


def trial(task, speed=.5, steps=4, actual=.25, fell=False, **options):
    motors = [dict(joint=j, rated_torque_nm=4.) for j in sorted(KNEE_ANKLE)]
    motors.append(dict(joint='hip', rated_torque_nm=.5))
    torque = np.column_stack([np.full(steps, 2.) for _ in motors])
    velocity = np.tile([-1., 2., 2., 2., 2., 2., 2.], (steps, 1))
    return score_trial(task, speed=speed, dt=DT, expected_steps=4, mass=20.,
        forward=.08, vx=np.full(steps, actual), omega=np.tile([3., 4.], (steps, 1)),
        torque=torque, joint_vel=velocity, motors=motors, fell=fell, **options)


def test_fixed_three_task_metrics_and_only_six_motors_in_objective():
    load = trial('backpack')
    assert load['status'] == 'success'
    assert load['metrics'] == pytest.approx(dict(cot=1.92 / (20*9.81*.08),
        forward_rmse_mps=.25, knee_ankle_max_rated_ratio=.5))
    assert load['motors']['hip']['rated_ratio'] == 4
    assert load['motors']['hip']['peak_torque_observed_nm'] == 2
    stairs = trial('stairs', reached_top=False)
    assert stairs['status'] == 'success' and not stairs['reached_top']
    assert stairs['metrics']['roll_pitch_rms_rad_s'] == 5
    assert 'knee_ankle_max_rated_ratio' not in stairs['metrics']


def test_fall_strict_10_percent_boundary_and_invalid_data():
    assert trial('run', actual=.05)['status'] == 'success'
    assert trial('run', actual=np.nextafter(.05, 0))['reason'] == 'below_minimum_forward_speed'
    assert trial('run', actual=-1)['status'] == 'failure'
    fall = trial('run', steps=2, actual=2, fell=True)
    assert fall['status'] == 'failure' and fall['metrics'] is None and fall['observed']
    assert trial('run', steps=2, timeout=True)['status'] == 'invalid'
    assert trial('run', actual=np.nan)['status'] == 'invalid'


def fixtures():
    conditions = dict(seeds=[7], copies=4)
    reports = []
    for task in bo.TASKS:
        rows = [trial(task, actual=.1 + i*.05) for i in range(4)]
        if task == 'run':
            rows[-2] = trial(task, fell=True)
            rows[-1] = trial(task, steps=2)
        reports.append(dict(task=task, seed=7, trials=[dict(replica=i, **r) for i, r in enumerate(rows)]))
    report = dict(kind=bo.REPORT_VERSION, domain='simulation', conditions=conditions,
                  conditions_sha256=digest(conditions), pairs=paired_trials(reports, conditions))
    return report


def test_calibration_and_repetition_keeps_failure_out_of_performance():
    report = fixtures()
    scoring = bo.calibrate(report, dict(name='pilot'))
    result = objective(report, scoring)
    assert result['valid_pairs'] == 3 and result['invalid_pairs'] == 1
    assert result['complete_objectives'] == result['joint_successes'] == 2
    assert result['mean_objective'] == pytest.approx(1)
    assert result['observations'][2]['objective'] is None
    report['pairs'][0]['tasks']['run']['metrics']['cot'] = None
    censored = objective(report, scoring)
    assert censored['complete_objectives'] == 1 and censored['joint_successes'] == 2
    with pytest.raises(ValueError, match='domain'):
        objective(dict(report, domain='hardware'), scoring)


def test_search_makes_exact_unique_proposals_and_preserves_scores_on_resume(tmp_path, monkeypatch):
    report = fixtures()
    conditions = report['conditions']
    scoring = tmp_path/'scoring.json'
    scoring.write_text(json.dumps(bo.calibrate(report, dict(name='pilot'))))
    source, actor = tmp_path/'source.json', tmp_path/'actor.pt'
    source.write_bytes(b'source')
    actor.write_bytes(b'actor')
    rows = [dict(key=[thigh, calf, 'K', 'A'], design_ids={'stairs': f'id{i}'})
            for i, (thigh, calf) in enumerate([(x,y) for x in (.2,.3,.4) for y in (.25,.35,.45)])]
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps(dict(structures=rows)))
    plan = dict(schema={'ranges_m': {'thigh': [.2, .4], 'calf': [.25, .45]},
                        'knees': ['K'], 'ankles': ['A']},
                shared_bank=str(bank), selected_structure_keys=[r['key'] for r in rows])
    path = tmp_path/'search'
    path.mkdir()
    def fake_evaluate(*_):
        return report
    monkeypatch.setattr(bo, 'evaluate', fake_evaluate)
    args = SimpleNamespace(scoring=scoring, updates=2, seed=42, trials=5, resume_paused=False)
    bo.search(args, path, source, plan, actor, conditions)
    first = bo.read(path/'budget_0005.json')
    assert len({row['design_id'] for row in first['candidates']}) == 5
    fifth = bo.read(path/'trial_0004/proposal.json')['acquisition']
    assert fifth['feasibility_rows'] == 12 and fifth['performance_rows'] == 8
    args.trials = 6
    bo.search(args, path, source, plan, actor, conditions)
    assert bo.read(path/'budget_0006.json')['candidates'][:5] == first['candidates']


def test_imported_baseline_uses_same_measurements_and_fixed_conditions(tmp_path):
    conditions = dict(kind=TRIAL_VERSION, duration_s=.08, seeds=[10], copies=2, tasks={
        'stairs': dict(speed_mps=.5, step_height_m=.075, step_depth_m=.25, step_count=12, approach_m=1.),
        'backpack': dict(speed_mps=.5, payload_kg=10., route_distance_m=8.),
        'run': dict(speed_mps=2., route_distance_m=30.)})
    condition_path = tmp_path/'conditions.json'
    condition_path.write_text(json.dumps(conditions))
    motors = [dict(joint=j, motor_id='test', rated_torque_nm=2., torque_side='joint_output',
                   source='fixture') for j in sorted(KNEE_ANKLE) + [f'j{i}' for i in range(17)]]
    trace = tmp_path/'trace.npz'
    np.savez(trace, position=np.full((4, 3), .25), initial_position=np.zeros(3),
             vx=np.full(4, .25), omega=np.ones((4, 2)), torque=np.ones((4, 23)),
             joint_vel=np.ones((4, 23)))
    spec = tmp_path/'measurements.json'
    trials = [dict(task=task, seed=10, replica=i, trace=trace.name, motors=motors,
                   total_mass_kg=35 if task == 'backpack' else 25., fell=False,
                   payload=dict(mass_kg=10., mount_position_torso_m=[-.1, 0., 0.]) if task == 'backpack' else None)
              for task in bo.TASKS for i in range(2)]
    spec.write_text(json.dumps(dict(domain='hardware', measurement_convention=TRIAL_VERSION, dt_s=.02,
                                    candidate='benchmark', trials=trials)))
    report = bo.ingest_measurements(spec, condition_path)
    assert report['domain'] == 'hardware'
    assert objective(report, bo.calibrate(report, dict(name='fixture')))['mean_objective'] == pytest.approx(1)


def test_candidate_evaluation_checks_handoff_and_reuses_measurements(tmp_path, monkeypatch):
    import finetune_state
    from robolab import rough_gpu

    source, initial, conditions_path = (tmp_path/name for name in ('shared_plan.json', 'actor.pt', 'conditions.json'))
    source.write_text('{}')
    initial.write_bytes(b'initial')
    conditions = dict(seeds=[9], copies=2, duration_s=.08)
    conditions_path.write_text(json.dumps(conditions))
    plan = dict(teachers={task: dict(design_ids=[task+'id'], bank_sha256=task, full_body_torque=True)
                          for task in bo.TASKS})
    candidate = dict(structure_key=[.3, .35, 'K', 'A'], design_ids={t: t+'id' for t in bo.TASKS})
    monkeypatch.setattr(finetune_state, 'select_candidate', lambda *_args, **_kw: (plan, candidate))
    monkeypatch.setattr(bo, 'load_plan', lambda path: bo.read(path))
    monkeypatch.setattr(rough_gpu, 'query_gpus', lambda: [])
    monkeypatch.setattr(rough_gpu, 'gpu_lease', lambda *_: nullcontext(dict(index=3, uuid='GPU-test')))
    calls = []

    def worker(command, _folder, _env, _timeout, _root):
        def arg(name):
            return command[command.index(name)+1]
        out = Path(arg('--output'))
        out.mkdir(parents=True)
        if Path(command[2]).name == 'finetune.py':
            calls.append('finetune')
            manifest = dict(candidate=candidate, updates=2, plan_sha256=bo.sha(source),
                            initial_actor_sha256=bo.sha(initial))
            (out/'manifest.json').write_text(json.dumps(manifest))
            (out/'plan.json').write_text(json.dumps(plan))
            actor, tracking = out/'finetuned.pt', out/'candidate_result.json'
            actor.write_bytes(b'finetuned')
            tracking.write_text('{}')
            (out/'completed.json').write_text(json.dumps(dict(result='CANDIDATE_FINETUNED_EVALUATED',
                candidate=candidate, ppo_updates=2, manifest_sha256=bo.sha(out/'manifest.json'),
                student=str(actor), student_sha256=bo.sha(actor), evaluation=str(tracking),
                evaluation_sha256=bo.sha(tracking))))
        else:
            task = arg('--task') if '--task' in command else 'run'
            calls.append(task)
            assert arg('--bo-conditions') == conditions_path and arg('--profile') == 'nominal'
            trace = out/'bo_samples.npz'
            trace.write_bytes(b'test-trace')
            (out/'summary.json').write_text('{}')
            (out/'bo_trials.json').write_text(json.dumps(dict(kind=TRIAL_VERSION, task=task,
                design_id=arg('--design-ids'), bank_sha256=task, seed=9, trace=str(trace), trace_sha256=bo.sha(trace),
                conditions_sha256=digest(conditions), actor_sha256=bo.sha(arg('--student')),
                plan_sha256=bo.sha(arg('--plan')), trials=[dict(replica=i, **trial(task)) for i in range(2)])))

    monkeypatch.setattr(bo, 'run_worker', worker)
    args = SimpleNamespace(updates=2, resume_paused=False, gpu=3, timeout=15, conditions=conditions_path)
    out = tmp_path/'candidate'
    first = bo.evaluate(args, out, source, plan, initial, 'stairsid', conditions)
    assert calls == ['finetune', *bo.TASKS]
    assert all(pair['feasible'] for pair in first['pairs'])
    assert bo.evaluate(args, out, source, plan, initial, 'stairsid', conditions) == first
    assert calls == ['finetune', *bo.TASKS]


def test_post_finetune_panel_uses_structure_means_and_frozen_weights():
    import copy
    from collection_grid import calibration_keys
    keys=calibration_keys()
    reports=[]
    for i,key in enumerate(keys):
        pairs=[]
        for seed in (11,22,33):
            tasks={task:dict(status='success',reached_top=True,metrics={m:i+1. for m in bo.METRICS[task]})
                   for task in bo.TASKS}
            pairs.append(dict(tasks=tasks,seed=seed,replica=0,feasible=True))
        reports.append(dict(candidate=dict(structure_key=list(key)), pairs=pairs,
            kind=bo.REPORT_VERSION, domain='simulation', conditions=dict(seeds=[11,22,33],copies=1),
            conditions_sha256='conditions',ppo_updates=100))
    scoring=bo.panel_scoring(reports,keys,{'initial_actor_sha256':'actor','updates':100})
    for task in bo.TASKS:
        metric=scoring['metrics'][task]['cot']
        assert metric['p10']==pytest.approx(5.9)
        assert metric['p90']==pytest.approx(45.1)
        assert metric['baseline']==pytest.approx(39.2)
        assert metric['weight']==pytest.approx(1/3 if task=='stairs' else .4)
    assert objective(reports[0],scoring)['mean_objective']==pytest.approx(1/39.2)
    incomplete=copy.deepcopy(reports)
    incomplete[0]['pairs'][0]['tasks']['stairs']['status']='failure'
    revised=bo.panel_scoring(incomplete,keys,{'initial_actor_sha256':'actor','updates':100})
    assert revised['metrics']['stairs']['cot']['eligible_designs']==49
    assert revised['metrics']['run']['cot']['eligible_designs']==50
    with pytest.raises(ValueError):
        bo.panel_scoring(reports[:-1]+reports[:1],keys,{'updates':100})

@pytest.fixture
def official_panel(tmp_path, monkeypatch):
    from itertools import product
    from collection_grid import calibration_keys, select_collection_grid, selection_panel

    fixed = calibration_keys()
    axes = [sorted(set(axis)) for axis in zip(*fixed)]
    keys = list(product(*axes))
    training = select_collection_grid(keys)
    selected = selection_panel(keys, training, fixed)
    ids = {key: {task: f'{task}_{i}' for task in bo.TASKS} for i, key in enumerate(keys)}
    rows = [dict(key=list(key), design_ids=ids[key]) for key in keys]
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps(dict(structures=rows)))
    source = tmp_path/'shared_plan/plan.json'
    source.parent.mkdir()
    source.write_text(json.dumps(dict(schema=dict(tasks=list(bo.TASKS)), shared_bank=str(bank),
                                     selected_structure_keys=keys)))
    monkeypatch.setattr(bo, 'load_plan', bo.read)
    out = tmp_path/'shared_policy'
    out.mkdir()
    actor = out/'selected.pt'
    actor.write_bytes(b'selected student')
    seeds = [20261117, 20261119, 20261120]
    budget = dict(selection_design_count=200)
    protocol = dict(selection_version='last_epoch_dagger_paired_200', selection_keys=selected,
                    selection_budget=budget,
                    selection=dict(profile='robust',strength=1.0),
                    calibration_keys=fixed, selection_seeds=seeds, dagger_rounds=3,
                    collection_scope=dict(structure_keys=training, design_count=400,
                                          steps_per_rollout=250, environments_per_task=4096))
    (out/'manifest.json').write_text(json.dumps(dict(plan_sha256=bo.sha(source), protocol=protocol, selection_budget=budget,
        protocol_sha256=digest(protocol),arguments=dict(epochs=10))))
    candidates = [dict(round_index=i, epoch=9, student=str(actor) if i==2 else str(out/f'dagger_{i}.pt'),
                       student_sha256=bo.sha(actor) if i==2 else str(i), rank=[2 if i==2 else 1])
                  for i in (1,2,3)]
    selection = dict(candidates=candidates, profile='robust', selected_round=2, selected_student=str(actor), budget=budget,
                     design_keys=selected, calibration_keys=fixed, seeds=seeds,
                     training_design_keys=[k for k in selected if k in set(training)])
    selection_path=out/'selection.json'
    selection_path.write_text(json.dumps(selection))
    (out/'completed.json').write_text(json.dumps(dict(
        result='DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING', full_bank_evaluated=False,
        protocol=protocol, protocol_sha256=digest(protocol), selection=str(selection_path),
        selected_round=2, rounds=candidates, selection_sha256=bo.sha(selection_path), student=str(actor), student_sha256=bo.sha(actor))))
    (tmp_path/'migration_completed.json').write_text('{}')
    monkeypatch.setattr(bo,'MIGRATED_ROOT',tmp_path,raising=False)
    calls = []
    next_key = next(k for k in keys if k not in fixed)
    by_id = {ids[k]['stairs']:k for k in keys}
    def synthetic(a, folder, source, plan, initial, candidate_id, conditions):
        calls.append((candidate_id,bo.sha(initial),a.updates))
        key = by_id[candidate_id]
        n = fixed.index(key) if key in fixed else -1
        candidate = dict(structure_key=list(key),design_ids=ids[key])
        manifest = dict(kind='rpo_bo_candidate_v1',candidate=candidate,updates=a.updates,
                        source_plan_sha256=bo.sha(source),initial_actor_sha256=bo.sha(initial),
                        conditions_sha256=digest(conditions))
        folder.mkdir(parents=True,exist_ok=True)
        bo.freeze(folder/'manifest.json',manifest)
        pairs = []
        for j, seed in enumerate(conditions['seeds']):
            tasks = {t: dict(status='success',metrics={m:n+2 for m in bo.METRICS[t]}) for t in bo.TASKS}
            tasks['stairs']['reached_top'] = n not in (-1,0) or j != 2
            pairs.append(dict(tasks=tasks,seed=seed,replica=0,feasible=True))
        report = dict(candidate=candidate,conditions=conditions,conditions_sha256=digest(conditions),
                      ppo_updates=a.updates,kind=bo.REPORT_VERSION,domain='simulation',pairs=pairs,
                      evaluation_manifest_sha256=bo.sha(folder/'manifest.json'))
        bo.freeze(folder/'bo_evaluation.json',report)
        return report
    monkeypatch.setattr(bo,'evaluate',synthetic)

    def choose(pool, observations, evaluated, _schema, _seed):
        assert evaluated == {ids[k]['stairs'] for k in fixed}
        assert len(observations) == 150 and sum(o['feasible'] for o in observations) == 149
        return next(r for r in pool if r['key'] == list(next_key)), dict(method='fixture')
    monkeypatch.setattr(bo,'propose',choose)
    return SimpleNamespace(root=tmp_path,calls=calls,selection=selection_path,student=actor,
                           fixed_ids=[ids[k]['stairs'] for k in fixed],next_id=ids[next_key]['stairs'])

def test_official_panel_initializes_once_and_proposes_candidate_51(official_panel):
    p=official_panel
    bo.main(['initialize','--migrated'])
    assert p.calls == [(k,bo.sha(p.student),100) for k in p.fixed_ids]
    bo.main(['search','--migrated','--trials','50'])
    bo.main(['search','--migrated','--trials','51'])
    assert p.calls[50:] == [(p.next_id,bo.sha(p.student),100)]
    saved=bo.read(p.root/'bo/search/budget_0051.json')
    assert [r['design_id'] for r in saved['candidates']] == [*p.fixed_ids,p.next_id]
    assert saved['best_observed']['design_id'] == p.fixed_ids[1]
    assert saved['candidates'][-1]['complete_objectives'] == 2
    assert not (p.root/'bo/search/trial_0000').exists()

@pytest.mark.parametrize('change', ['actor','budget','report','substituted_manifest','scale','downgrade'])
def test_official_panel_rejects_changed_evidence_before_proposal(official_panel,change):
    p=official_panel
    bo.main(['initialize','--root',str(p.root)])
    calib=p.root/'bo/calibration50'
    args=['search','--root',str(p.root),'--trials','51']
    if change=='actor':
        p.student.write_bytes(b'changed')
    elif change=='budget':
        args+=['--updates','99']
    elif change=='report':
        with (calib/'design_01/bo_evaluation.json').open('a') as f: f.write(' ')
    elif change=='substituted_manifest':
        manifest=calib/'design_01/manifest.json'
        value=bo.read(manifest)
        value['initial_actor_sha256']='different student'
        manifest.write_text(json.dumps(value))
    else:
        scoring=calib/'scoring.json'
        value=bo.read(scoring)
        if change=='scale': value['metrics']['run']['cot']['baseline']*=2
        else: value.pop('method')
        scoring.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        bo.main(args)
    assert len(p.calls)==50
    assert not (p.root/'bo/search/manifest.json').exists()

@pytest.mark.parametrize('change', ['winner','reserved','old_version','relabelled_50','budget_50',
                                    'conditions','selection_hash','nominal'])
def test_official_initialization_checks_selection_before_tuning(official_panel,change):
    p=official_panel
    args=['initialize','--root',str(p.root)]
    if change=='conditions':
        custom=bo.read(bo.HERE/'configs/bo_conditions.json')
        custom['tasks']['run']['speed_mps']=2.
        path=p.root/'conditions.json'
        path.write_text(json.dumps(custom))
        args+=['--conditions',str(path)]
    elif change=='selection_hash':
        with p.selection.open('a') as f: f.write(' ')
    else:
        data=bo.read(p.selection)
        manifest_path=p.selection.parent/'manifest.json'
        completed_path=p.selection.parent/'completed.json'
        manifest=bo.read(manifest_path)
        completed=bo.read(completed_path)
        protocol=manifest['protocol']
        if change=='winner':
            data['selected_round']=1
        elif change=='nominal':
            data['profile']='nominal'
        elif change=='old_version':
            protocol['selection_version']='last_epoch_dagger_paired_50'
        elif change=='budget_50':
            data['budget']['selection_design_count']=50
            protocol['selection_budget']=manifest['selection_budget']=data['budget']
        elif change=='relabelled_50':
            # Keep one seen and one held-out design from each motor pair.
            reduced=data['design_keys'][::4]
            data['design_keys']=protocol['selection_keys']=reduced
            data['training_design_keys']=[key for key in reduced if key in data['training_design_keys']]
        else:
            data['design_keys'][0]=data['calibration_keys'][0]
            protocol['selection_keys']=data['design_keys']
        p.selection.write_text(json.dumps(data))
        manifest['protocol_sha256']=digest(protocol)
        manifest_path.write_text(json.dumps(manifest))
        completed.update(protocol=protocol,protocol_sha256=digest(protocol),selection_sha256=bo.sha(p.selection))
        completed_path.write_text(json.dumps(completed))
    version_error = 'last_epoch_dagger_paired_200.*last_epoch_dagger_paired_50' if change == 'old_version' else None
    with pytest.raises(ValueError, match=version_error):
        bo.main(args)
    assert not p.calls

def test_migrated_root_and_explicit_root_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        bo.main(['search','--migrated','--root',str(tmp_path),'--trials','51'])
    assert error.value.code==2
