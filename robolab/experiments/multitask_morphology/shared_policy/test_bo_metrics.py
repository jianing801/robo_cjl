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


def trial(task, speed=.5, steps=4, expected_steps=4, actual=.25, forward=.08, fell=False, **options):
    motors = [dict(joint=j, rated_torque_nm=4.) for j in sorted(KNEE_ANKLE)]
    motors.append(dict(joint='hip', rated_torque_nm=.5))
    torque = np.column_stack([np.full(steps, 2.) for _ in motors])
    velocity = np.tile([-1., 2., 2., 2., 2., 2., 2.], (steps, 1))
    return score_trial(task, speed=speed, dt=DT, expected_steps=expected_steps, mass=20.,
        forward=forward, vx=np.full(steps, actual), omega=np.tile([3., 4.], (steps, 1)),
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
    assert trial('stairs', actual=.05, reached_top=None)['status'] == 'success'
    assert trial('stairs', actual=np.nextafter(.05, 0))['reason'] == 'below_minimum_forward_speed'
    assert trial('stairs', fell=True)['status'] == 'failure'
    edge = trial('stairs', actual=.6, outside_stairs=True)
    assert edge['status'] == 'invalid' and edge['metrics'] is None
    assert trial('run', actual=.05)['status'] == 'success'
    assert trial('run', actual=np.nextafter(.05, 0))['reason'] == 'below_minimum_forward_speed'
    assert trial('run', actual=-1)['status'] == 'failure'
    fall = trial('run', steps=2, actual=2, fell=True)
    assert fall['status'] == 'failure' and fall['metrics'] is None and fall['observed']
    assert trial('run', steps=2, timeout=True)['status'] == 'invalid'
    assert trial('run', actual=np.nan)['status'] == 'invalid'


@pytest.mark.parametrize(('task', 'minimum'), [('stairs', 2.), ('backpack', 2.), ('run', 6.)])
def test_formal_bo_enforces_twenty_percent_of_commanded_distance(task, minimum):
    conditions = bo.read_conditions(bo.FORMAL_CONDITIONS)
    steps = round(conditions['duration_s'] / DT)
    inputs = dict(speed=conditions['tasks'][task]['speed_mps'], steps=steps, expected_steps=steps)
    if 'min_forward_progress_ratio' in conditions:
        inputs['min_forward_progress_ratio'] = conditions['min_forward_progress_ratio']
    short = trial(task, actual=1.5, forward=minimum - .001, **inputs)
    assert short['status'] == 'failure'
    assert short['reason'] == 'below_minimum_forward_progress' and short['metrics'] is None
    assert trial(task, actual=.01, forward=minimum, **inputs)['status'] == 'success'
    assert trial(task, actual=1.5, forward=minimum, fell=True, **inputs)['reason'] == 'fell'
    assert digest(conditions) != digest({k: v for k, v in conditions.items()
                                        if k != 'min_forward_progress_ratio'})


@pytest.mark.parametrize('task', bo.TASKS)
@pytest.mark.parametrize('legacy', [False, True])
def test_bo_reset_matches_declared_conditions(task, legacy, monkeypatch):
    import bo_capture

    path = bo.LEGACY_FORMAL_CONDITIONS if legacy else bo.FORMAL_CONDITIONS
    conditions = bo.read_conditions(path)
    assert conditions['seeds'] == ([20261118, 20261119, 20261120] if legacy else [20261118])
    assert conditions['copies'] == 1
    assert conditions.get('reset_mode') == (None if legacy else 'fixed')
    terrains = SimpleNamespace(MeshPlaneTerrainCfg=SimpleNamespace, TerrainGeneratorCfg=SimpleNamespace)
    monkeypatch.setitem(sys.modules, 'isaaclab', SimpleNamespace(terrains=terrains))
    monkeypatch.setitem(sys.modules, 'isaaclab.terrains', terrains)
    monkeypatch.setattr(bo_capture, 'flight_tile', lambda *_: None)
    command = SimpleNamespace(ranges=SimpleNamespace(), heading_command=True)
    pose = {'x': (-.01, .01), 'yaw': (-.02, .02)}
    velocities = {'x': (-.1, .1), 'yaw': (-.2, .2)}
    joint_position = (.9, 1.1) if task == 'run' else (-.1, .1)
    base = dict(pose_range=pose, velocity_range=velocities)
    joint = dict(position_range=joint_position, velocity_range=(-.1, .1))
    cfg = SimpleNamespace(commands=SimpleNamespace(base_velocity=command) if task == 'run' else command,
                          episode_length_s=10., scene=SimpleNamespace(terrain=SimpleNamespace(terrain_type='plane')),
                          scene_context=SimpleNamespace(),
                          events=SimpleNamespace(reset_base=SimpleNamespace(params=base),
                                                 reset_robot_joints=SimpleNamespace(params=joint)))
    audit = dict(reset={}, commands={})
    bo_capture.configure_conditions(cfg, task, conditions, 20261118, audit)
    speed = conditions['tasks'][task]['speed_mps']
    assert command.ranges.lin_vel_x == (speed, speed)
    assert command.ranges.lin_vel_y == command.ranges.ang_vel_z == (0., 0.)
    assert cfg.episode_length_s > conditions['duration_s']
    if legacy and task != 'stairs':
        assert base == dict(pose_range=pose, velocity_range=velocities)
        assert joint == dict(position_range=joint_position, velocity_range=(-.1, .1))
        assert audit['reset'] == {}
    else:
        assert base == dict(pose_range=dict.fromkeys(pose, (0., 0.)),
                            velocity_range=dict.fromkeys(velocities, (0., 0.)))
        position = (1., 1.) if task == 'run' else (0., 0.)
        assert joint == dict(position_range=position, velocity_range=(0., 0.))
        assert audit['reset']['joint_position'] == position


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
    np.savez(trace, position=np.full((4, 3), .025), initial_position=np.zeros(3),
             vx=np.full(4, .45), omega=np.ones((4, 2)), torque=np.ones((4, 23)),
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

    condition_path.write_text(json.dumps(dict(conditions, min_forward_progress_ratio=.20)))
    stricter = bo.ingest_measurements(spec, condition_path)
    with pytest.raises(ValueError, match='conditions'):
        objective(stricter, bo.calibrate(report, dict(name='fixture')))
    for pair in stricter['pairs']:
        assert pair['tasks']['run']['reason'] == 'below_minimum_forward_progress'
        assert pair['feasible'] is False


def test_bo_ingest_defaults_to_the_formal_forward_progress_gate(tmp_path):
    motors = [dict(joint=joint, motor_id='test', torque_side='joint_output')
              for joint in sorted(KNEE_ANKLE) + [f'j{i}' for i in range(17)]]
    steps = 1000
    position = np.zeros((steps, 3))
    position[:, 0] = np.linspace(.003, 3., steps)
    np.savez(tmp_path/'trace.npz', position=position, initial_position=np.zeros(3),
             vx=np.full(steps, .15), omega=np.zeros((steps, 2)),
             torque=np.ones((steps, 23)), joint_vel=np.ones((steps, 23)))
    trials = [dict(task=task, seed=20261118, replica=0, trace='trace.npz', motors=motors,
                   total_mass_kg=35. if task == 'backpack' else 25., fell=False,
                   reached_top=False, route_completed=False,
                   payload=dict(mass_kg=10., mount_position_torso_m=[-.1, 0., 0.]) if task == 'backpack' else None)
              for task in bo.TASKS]
    spec, output = tmp_path/'measurements.json', tmp_path/'report.json'
    spec.write_text(json.dumps(dict(domain='hardware', measurement_convention=TRIAL_VERSION,
                                    dt_s=DT, candidate='fixture', trials=trials)))

    bo.main(['ingest', '--spec', str(spec), '--output', str(output)])

    report = bo.read(output)
    pair, = report['pairs']
    assert report['conditions']['min_forward_progress_ratio'] == .2
    assert pair['tasks']['stairs']['status'] == pair['tasks']['backpack']['status'] == 'success'
    assert pair['tasks']['run']['reason'] == 'below_minimum_forward_progress'
    assert pair['feasible'] is False


@pytest.mark.parametrize(('updates', 'frozen_compaction', 'expect_compact', 'resume_paused', 'old_bo'), [
    (100, None, False, False, False), (200, None, True, False, False),
    (200, False, False, False, False), (200, True, True, False, False),
    pytest.param(200, None, True, True, False, id='manual-resume-fresh-tuning'),
    pytest.param(200, True, True, True, False, id='manual-resume-existing-tuning'),
    pytest.param(200, True, True, False, 'pending', id='reuse-frozen-bo-manifest'),
    pytest.param(200, True, True, False, 'three-seed', id='reuse-three-seed-coordinator-manifest'),
    pytest.param(200, True, True, False, 'legacy', id='reuse-legacy-diagnostics-and-formal-measurements'),
])
def test_candidate_evaluation_checks_handoff_and_reuses_measurements(
        tmp_path, monkeypatch, updates, frozen_compaction, expect_compact, resume_paused, old_bo):
    import finetune_state
    from robolab import rough_gpu

    source, initial, conditions_path = (tmp_path/name for name in ('shared_plan.json', 'actor.pt', 'conditions.json'))
    source.write_text('{}')
    initial.write_bytes(b'initial')
    conditions = dict(seeds=[9, 19, 29], copies=2, duration_s=.08)
    conditions_path.write_text(json.dumps(conditions))
    plan = dict(teachers={task: dict(design_ids=[task+'id'], bank_sha256=task, full_body_torque=True)
                          for task in bo.TASKS})
    candidate = dict(structure_key=[.3, .35, 'K', 'A'], design_ids={t: t+'id' for t in bo.TASKS})
    monkeypatch.setattr(finetune_state, 'select_candidate', lambda *_args, **_kw: (plan, candidate))
    monkeypatch.setattr(bo, 'load_plan', lambda path: bo.read(path))
    monkeypatch.setattr(bo, 'shared_input', lambda *_args, **_kw: (source, plan, initial))
    monkeypatch.setattr(bo, 'read_conditions', bo.read)
    monkeypatch.setattr(rough_gpu, 'query_gpus', lambda: [])
    monkeypatch.setattr(rough_gpu, 'gpu_lease', lambda *_: nullcontext(dict(index=3, uuid='GPU-test')))
    calls = []
    out = tmp_path/'candidate'
    tuning = out/'tuning'
    manifest = dict(candidate=candidate, updates=updates, plan_sha256=bo.sha(source),
                    initial_actor_sha256=bo.sha(initial))
    if expect_compact:
        manifest['compact_training_artifacts'] = True
    if frozen_compaction is not None:
        tuning.mkdir(parents=True)
        (tuning/'manifest.json').write_text(json.dumps(manifest))
    if resume_paused:
        out.mkdir(exist_ok=True)
        (out/'USER_PAUSED').touch()
    if old_bo:
        sources = {name: bo.sha(bo.SHARED/name) for name in (
            'bo_capture.py', 'shared_bo_metrics.py', 'stair_course.py', 'rollout.py', 'rollout_amp.py')}
        sources.update({
            'bo.py': ('0942320ad840a49a921bf014c57d033a6322f86e34043e50c28d6310ea06879d'
                      if old_bo == 'three-seed' else
                      '203b9aada3b4615464b04491ba61f9d59ecd27219f0f061e8540661b4c61d053'),
            'bo_capture.py': '5aacc6dc340839ed76db3a4d3e4e92f07d33b9e88911c538ceba482cad68907e',
            'shared_bo_metrics.py': 'c064cc6b15b03eb8869a38a567b0a1bae55f8ba93246fb13fc05e3de87e9445e',
        })
        (out/'manifest.json').write_text(json.dumps(dict(kind='rpo_bo_candidate_v1', candidate=candidate,
            source_plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(initial), conditions_sha256=digest(conditions),
            updates=updates, sources=sources), indent=3))
        original_manifest = (out/'manifest.json').read_bytes()
    if old_bo == 'legacy':
        (tuning/'plan.json').write_text(json.dumps(plan))
        actor = tuning/'shared_actor_finetuned.pt'
        actor.write_bytes(b'finetuned')
        write_legacy_completion(tuning, candidate, updates, actor)

    def worker(command, _folder, _env, _timeout, _root):
        def arg(name):
            return command[command.index(name)+1]
        out = Path(arg('--output'))
        out.mkdir(parents=True, exist_ok=True)
        if Path(command[2]).name == 'finetune.py':
            calls.append('finetune')
            assert arg('--updates') == str(updates)
            assert '--training-only' in command
            assert ('--compact-training-artifacts' in command) is expect_compact
            assert ('--resume' in command) is (frozen_compaction is not None)
            assert ('--resume-paused' in command) is (resume_paused and frozen_compaction is not None)
            (out/'manifest.json').write_text(json.dumps(manifest))
            (out/'plan.json').write_text(json.dumps(plan))
            actor = out/'shared_actor_finetuned.pt'
            actor.write_bytes(b'finetuned')
            write_training_completion(out, candidate, updates, actor)
        else:
            task = arg('--task') if '--task' in command else 'run'
            calls.append(task)
            assert arg('--bo-conditions') == conditions_path and arg('--profile') == 'nominal'
            trace = out/'bo_samples.npz'
            trace.write_bytes(b'test-trace')
            (out/'summary.json').write_text('{}')
            (out/'bo_trials.json').write_text(json.dumps(dict(kind=TRIAL_VERSION, task=task,
                design_id=arg('--design-ids'), bank_sha256=task, seed=int(arg('--seed')), trace=str(trace),
                trace_sha256=bo.sha(trace),
                conditions_sha256=digest(conditions), actor_sha256=bo.sha(arg('--student')),
                plan_sha256=bo.sha(arg('--plan')), trials=[dict(replica=i, **trial(task)) for i in range(2)])))

    monkeypatch.setattr(bo, 'run_worker', worker)
    args = ['evaluate', '--root', str(tmp_path), '--output', str(out), '--candidate', 'stairsid',
            '--conditions', str(conditions_path), '--gpu', '3', '--timeout', '15']
    args += ['--updates', str(updates)]
    if resume_paused:
        args.append('--resume-paused')
    bo.main(args)
    assert not (out/'USER_PAUSED').exists()
    first = bo.read(out/'bo_evaluation.json')
    if old_bo:
        assert (out/'manifest.json').read_bytes() == original_manifest
        assert first['evaluation_manifest_sha256'] == bo.sha(out/'manifest.json')
    expected_calls = [] if old_bo == 'legacy' else ['finetune']
    expected_calls.extend(task for seed in conditions['seeds'] for task in bo.TASKS)
    assert calls == expected_calls
    assert len(list((out/'measurements').glob('*/accepted.json'))) == 9
    assert first['ppo_updates'] == updates
    assert all(pair['feasible'] for pair in first['pairs'])
    frozen_report = (out/'bo_evaluation.json').read_bytes()
    bo.main([*args, '--finetuned', str(tuning)])
    assert bo.read(out/'bo_evaluation.json') == first
    assert (out/'bo_evaluation.json').read_bytes() == frozen_report
    assert calls == expected_calls


def test_frozen_bo_manifest_only_allows_the_previous_bo_source(tmp_path):
    path = tmp_path/'manifest.json'
    identity = dict(updates=200, sources={'bo.py': 'current', 'rollout.py': 'unchanged'})
    path.write_text(json.dumps(dict(identity, sources={
        'bo.py': bo.PRE_TRAINING_ONLY_BO_SHA256, 'rollout.py': 'different'})))
    with pytest.raises(ValueError, match='Frozen BO identity differs'):
        bo.freeze_candidate_manifest(path, identity)


def test_formal_candidate_accepts_the_previous_ingest_coordinator_only(tmp_path):
    path = tmp_path/'manifest.json'
    identity = dict(updates=200, conditions_sha256='formal', sources={'bo.py': 'current', 'rollout.py': 'unchanged'})
    previous = dict(identity, sources={**identity['sources'],
        'bo.py': 'fead4dca36355e514f86981a1526cd79c1bb96ce7ee12a7c8e7d16cde9773233'})
    path.write_text(json.dumps(previous))
    bo.freeze_candidate_manifest(path, identity, fixed_reset=True)
    assert bo.read(path) == previous
    with pytest.raises(ValueError, match='Frozen BO identity differs'):
        bo.freeze_candidate_manifest(path, dict(identity, conditions_sha256='different'), fixed_reset=True)


def write_legacy_completion(root, candidate, updates, actor):
    evaluation = root/'candidate_result.json'
    evaluation.write_bytes(b'diagnostics')
    (root/'completed.json').write_text(json.dumps(dict(result='CANDIDATE_FINETUNED_EVALUATED',
        candidate=candidate, ppo_updates=updates, manifest_sha256=bo.sha(root/'manifest.json'),
        student=str(actor), student_sha256=bo.sha(actor), evaluation=str(evaluation),
        evaluation_sha256=bo.sha(evaluation))))
    return evaluation


def write_training_completion(root, candidate, updates, actor):
    checkpoint_dir = root/'checkpoints'
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir/f'update_{updates:06d}.json'
    blob = checkpoint_dir/f'update_{updates:06d}_fixed.pt'
    blob.write_bytes(b'learning state')
    manifest_hash = bo.sha(root/'manifest.json')
    checkpoint.write_text(json.dumps(dict(completed_updates=updates, manifest_sha256=manifest_hash,
                                          path=blob.name, sha256=bo.sha(blob))))
    (root/'training_completed.json').write_text(json.dumps(dict(result='CANDIDATE_FINETUNED',
        candidate=candidate, ppo_updates=updates, manifest_sha256=manifest_hash, student=str(actor),
        student_sha256=bo.sha(actor), checkpoint_sha256=bo.sha(checkpoint))))
    return checkpoint, blob


@pytest.mark.parametrize('tampered', [None, 'state', 'outside', 'actor', 'legacy', 'legacy-diagnostics'])
def test_completed_actor_validates_checkpoint_and_keeps_legacy_diagnostics_compatible(tmp_path, tampered):
    root = tmp_path/'tuning'
    root.mkdir()
    source, initial, actor = tmp_path/'plan.json', tmp_path/'initial.pt', root/'shared_actor_finetuned.pt'
    source.write_bytes(b'plan')
    initial.write_bytes(b'initial')
    actor.write_bytes(b'actor')
    candidate = dict(design_ids={'stairs': 'one'})
    updates = 200
    manifest = dict(candidate=candidate, updates=updates, plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(initial))
    (root/'manifest.json').write_text(json.dumps(manifest))
    if tampered in ('legacy', 'legacy-diagnostics'):
        evaluation = write_legacy_completion(root, candidate, updates, actor)
        if tampered == 'legacy-diagnostics':
            evaluation.write_bytes(b'replaced diagnostics')
    else:
        checkpoint, blob = write_training_completion(root, candidate, updates, initial if tampered == 'actor' else actor)
        if tampered == 'state':
            blob.write_bytes(b'replaced state')
        elif tampered == 'outside':
            outside = root/'outside.pt'
            outside.write_bytes(blob.read_bytes())
            checkpoint.write_text(json.dumps(dict(bo.read(checkpoint), path='../outside.pt')))
            complete = root/'training_completed.json'
            complete.write_text(json.dumps(dict(bo.read(complete), checkpoint_sha256=bo.sha(checkpoint))))
    expected_error = {'state': 'checkpoint', 'outside': 'checkpoint', 'actor': 'canonical',
                      'legacy-diagnostics': 'candidate evaluation'}.get(tampered)
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            bo.completed_actor(root, source, initial, candidate, updates)
    else:
        assert bo.completed_actor(root, source, initial, candidate, updates) == actor


@pytest.mark.parametrize('seeds', [[20261118], [11, 22, 33]])
def test_post_finetune_panel_uses_structure_means_and_frozen_weights(seeds):
    import copy
    from collection_grid import calibration_keys
    keys=calibration_keys()
    conditions = dict(seeds=seeds, copies=1)
    reports=[]
    for i,key in enumerate(keys):
        pairs=[]
        for seed in seeds:
            tasks={task:dict(status='success',reached_top=True,metrics={m:i+1. for m in bo.METRICS[task]})
                   for task in bo.TASKS}
            pairs.append(dict(tasks=tasks,seed=seed,replica=0,feasible=True))
        reports.append(dict(candidate=dict(structure_key=list(key)), pairs=pairs,
            kind=bo.REPORT_VERSION, domain='simulation', conditions=conditions,
            conditions_sha256=digest(conditions),ppo_updates=100))
    scoring=bo.panel_scoring(reports,keys,{'initial_actor_sha256':'actor','updates':100})
    for task in bo.TASKS:
        metric=scoring['metrics'][task]['cot']
        assert metric['p10']==pytest.approx(5.9)
        assert metric['p90']==pytest.approx(45.1)
        assert metric['baseline']==pytest.approx(45.1)
        assert metric['scale']==pytest.approx(39.2)
        assert metric['weight']==pytest.approx(1/3 if task=='stairs' else .4)
    assert objective(reports[0],scoring)['mean_objective']==pytest.approx((1-5.9)/39.2)
    for value, normalized in ((1., (1-5.9)/39.2), (5.9, 0.), (45.1, 1.), (50., (50-5.9)/39.2)):
        sample = copy.deepcopy(reports[0])
        for pair in sample['pairs']:
            for task, trial in pair['tasks'].items():
                trial['metrics'] = dict.fromkeys(bo.METRICS[task], value)
        for observation in objective(sample, scoring)['observations']:
            for task, metrics in observation['normalized'].items():
                assert metrics == pytest.approx(dict.fromkeys(bo.METRICS[task], normalized))
    missed_top=copy.deepcopy(reports)
    missed_top[0]['pairs'][0]['tasks']['stairs']['reached_top']=False
    assert bo.panel_scoring(missed_top,keys,scoring['benchmark']) == scoring
    assert objective(missed_top[0],scoring) == objective(reports[0],scoring)
    incomplete=copy.deepcopy(reports)
    incomplete[0]['pairs'][0]['tasks']['stairs']['status']='failure'
    incomplete[0]['pairs'][0]['feasible'] = False
    revised=bo.panel_scoring(incomplete,keys,{'initial_actor_sha256':'actor','updates':100})
    assert revised['metrics']['stairs']['cot']['eligible_designs']==49
    assert revised['metrics']['run']['cot']['eligible_designs']==50
    with pytest.raises(ValueError):
        bo.panel_scoring(reports[:-1]+reports[:1],keys,{'updates':100})
    missing = copy.deepcopy(reports)
    missing[0]['pairs'].pop()
    with pytest.raises(ValueError, match='seed coverage'):
        bo.panel_scoring(missing, keys, {'updates':100})
    incomplete[0]['pairs'][0]['feasible'] = True
    with pytest.raises(ValueError, match='feasibility'):
        bo.panel_scoring(incomplete, keys, {'updates':100})

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
    protocol = dict(selection_version='last_epoch_dagger_paired_200', stair_evaluation=bo.COURSE, selection_keys=selected,
                    evaluation_velocity_frame='horizontal_heading',
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
            if n == 0 and j == len(conditions['seeds']) - 1:
                tasks['stairs'].update(status='failure', reason='below_minimum_forward_speed', metrics=None)
            pairs.append(dict(tasks=tasks,seed=seed,replica=0,
                              feasible=all(t['status']=='success' for t in tasks.values())))
        report = dict(candidate=candidate,conditions=conditions,conditions_sha256=digest(conditions),
                      ppo_updates=a.updates,kind=bo.REPORT_VERSION,domain='simulation',pairs=pairs,
                      evaluation_manifest_sha256=bo.sha(folder/'manifest.json'))
        bo.freeze(folder/'bo_evaluation.json',report)
        return report
    monkeypatch.setattr(bo,'evaluate',synthetic)

    def choose(pool, observations, evaluated, _schema, _seed):
        assert evaluated == {ids[k]['stairs'] for k in fixed}
        assert len(observations) in (50, 150)
        assert sum(o['feasible'] for o in observations) == len(observations) - 1
        return next(r for r in pool if r['key'] == list(next_key)), dict(method='fixture')
    monkeypatch.setattr(bo,'propose',choose)
    return SimpleNamespace(root=tmp_path,calls=calls,selection=selection_path,student=actor,
                           fixed_ids=[ids[k]['stairs'] for k in fixed],next_id=ids[next_key]['stairs'])

def test_official_panel_initializes_once_and_proposes_candidate_51(official_panel):
    p=official_panel
    bo.main(['initialize','--migrated'])
    assert p.calls == [(k,bo.sha(p.student),100) for k in p.fixed_ids]
    manifest = bo.read(p.root/'bo/calibration50_single_seed_ppo100_random/manifest.json')
    assert manifest['updates'] == 100 and manifest['selection_velocity_frame'] == 'horizontal_heading'
    assert manifest['ppo_schedule'] == dict(epochs=5, minibatches=4)
    assert manifest['init_at_random_ep_len'] is True
    assert manifest['envs_per_task'] == dict(stairs=1366, backpack=1365, run=1365)
    bo.main(['search','--migrated','--trials','50'])
    bo.main(['search','--migrated','--trials','51'])
    assert p.calls[50:] == [(p.next_id,bo.sha(p.student),100)]
    saved=bo.read(p.root/'bo/search_single_seed_ppo100_random/budget_0051.json')
    assert [r['design_id'] for r in saved['candidates']] == [*p.fixed_ids,p.next_id]
    assert saved['candidates'][0]['complete_objectives'] == 0
    assert saved['candidates'][0]['mean_objective'] is None
    assert saved['candidates'][-1]['complete_objectives'] == 1
    assert saved['best_observed']['design_id'] == p.next_id
    assert not (p.root/'bo/search_single_seed_ppo100_random/trial_0000').exists()

def test_initialize_reuses_frozen_legacy_reports_with_anchored_scoring(official_panel):
    p = official_panel
    legacy = ['--conditions', str(bo.LEGACY_FORMAL_CONDITIONS)]
    initialize = ['initialize', '--migrated', '--updates', '100', *legacy]
    search = ['search', '--migrated', '--updates', '100', '--trials', '50', *legacy]
    bo.main(initialize)
    assert p.calls == [(k,bo.sha(p.student),100) for k in p.fixed_ids]
    calibration = p.root/'bo/calibration50'
    old_scoring = calibration/'scoring.json'
    old_receipt = calibration/'completed.json'
    data = bo.read(old_scoring)
    for metrics in data['metrics'].values():
        for metric in metrics.values():
            metric.update(baseline=metric['p90']-metric['p10'], scale=None)
    old_scoring.write_text(json.dumps(data), encoding='utf8')
    receipt = bo.read(old_receipt)
    old_receipt.write_text(json.dumps(dict(receipt, scoring_sha256=bo.sha(old_scoring))), encoding='utf8')
    old_hashes = (bo.sha(old_scoring), bo.sha(old_receipt))
    report_path = calibration/'design_02/bo_evaluation.json'
    legacy_report = bo.read(report_path)
    assert objective(legacy_report, data)['mean_objective'] > 0

    with pytest.raises(ValueError, match='initialize'):
        bo.main(search)
    original_report = report_path.read_bytes()
    report_path.write_bytes(original_report + b' ')
    with pytest.raises(ValueError, match='Frozen initial BO report changed'):
        bo.main(initialize)
    report_path.write_bytes(original_report)

    bo.main(initialize)
    bo.main(initialize)
    assert len(p.calls) == 50
    assert (bo.sha(old_scoring), bo.sha(old_receipt)) == old_hashes

    updated = calibration/'percentile_scoring/scoring.json'
    assert objective(legacy_report, bo.read(updated))['mean_objective'] < 0
    bo.main(search)
    assert bo.read(p.root/'bo/search/manifest.json')['scoring_sha256'] == bo.sha(updated)


def test_initialize_reuses_completed_anchored_100_update_panel(official_panel):
    p = official_panel
    args = ['initialize', '--migrated', '--updates', '100', '--conditions', str(bo.LEGACY_FORMAL_CONDITIONS)]
    bo.main(args)
    manifest = p.root/'bo/calibration50/manifest.json'
    original_manifest = manifest.read_bytes()
    bo.main(args)
    assert len(p.calls) == 50
    assert manifest.read_bytes() == original_manifest
    with pytest.raises(ValueError, match='Frozen BO identity differs'):
        bo.main(['initialize', '--migrated', '--updates', '200', '--conditions', str(bo.LEGACY_FORMAL_CONDITIONS)])
    assert len(p.calls) == 50
    assert manifest.read_bytes() == original_manifest


def test_explicit_legacy_calibration_refuses_current_conditions(official_panel):
    p = official_panel
    legacy = ['--conditions', str(bo.LEGACY_FORMAL_CONDITIONS)]
    bo.main(['initialize', '--migrated', *legacy])
    calibration = p.root/'bo/calibration50'
    original_manifest = (calibration/'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='Frozen BO identity differs'):
        bo.main(['initialize', '--migrated', '--output', str(calibration)])
    with pytest.raises(ValueError, match='same conditions'):
        bo.main(['search', '--migrated', '--trials', '50', '--scoring', str(calibration/'scoring.json'),
                 '--conditions', str(bo.FORMAL_CONDITIONS)])
    assert len(p.calls) == 50
    assert (calibration/'manifest.json').read_bytes() == original_manifest


@pytest.mark.parametrize('change', ['actor','budget','report','substituted_manifest','scale','downgrade'])
def test_official_panel_rejects_changed_evidence_before_proposal(official_panel,change):
    p=official_panel
    bo.main(['initialize','--root',str(p.root)])
    calib=p.root/'bo/calibration50_single_seed_ppo100_random'
    args=['search','--root',str(p.root),'--trials','51']
    if change=='actor':
        p.student.write_bytes(b'changed')
    elif change=='budget':
        args+=['--updates','99','--scoring',str(calib/'scoring.json')]
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
    assert not (p.root/'bo/search_single_seed_ppo100_random/manifest.json').exists()

@pytest.mark.parametrize('change', ['winner','reserved','old_version','relabelled_50','budget_50',
                                    'conditions','selection_hash','nominal','body_axis'])
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
        elif change=='body_axis':
            protocol['evaluation_velocity_frame']='body_axis'
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
