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


@pytest.mark.parametrize('missing', ['measurement_rate', 'trial_rate', 'absolute_cot'])
def test_absolute_scoring_rejects_positive_only_or_wrong_rate_evidence(missing):
    report = fixtures()
    if missing != 'measurement_rate':
        report['conditions']['measurement_dt_s'] = .005
    report['conditions_sha256'] = digest(report['conditions'])
    scoring = dict(bo.calibrate(report, dict(name='pilot')), cot_basis='absolute_mechanical')
    for pair in report['pairs']:
        for task_trial in pair['tasks'].values():
            if missing != 'trial_rate':
                task_trial['dt_s'] = .005
            if missing != 'absolute_cot' and task_trial['status'] == 'success':
                task_trial['observed']['absolute_mechanical_cot'] = 0.2
    with pytest.raises(ValueError, match='200 Hz'):
        objective(report, scoring)


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


@pytest.mark.parametrize('frozen_compaction', [None,False,True])
@pytest.mark.parametrize('resume_paused', [False,True])
def test_candidate_evaluation_checks_handoff_and_reuses_measurements(
        tmp_path, monkeypatch, frozen_compaction, resume_paused):
    import finetune_state
    from robolab import rough_gpu

    source, initial = (tmp_path/name for name in ('shared_plan.json','actor.pt'))
    source.write_text('{}')
    initial.write_bytes(b'initial')
    conditions_path = bo.PHYSICS_CONDITIONS
    conditions = bo.read_conditions(conditions_path)
    updates = bo.DEFAULT_BO_UPDATES
    plan = dict(teachers={task: dict(design_ids=[task+'id'], bank_sha256=task, full_body_torque=True)
                          for task in bo.TASKS})
    candidate = dict(structure_key=[.3,.35,'K','A'], design_ids={t:t+'id' for t in bo.TASKS})
    monkeypatch.setattr(finetune_state,'select_candidate',lambda *_args,**_kw:(plan,candidate))
    monkeypatch.setattr(bo,'load_plan',lambda path:bo.read(path))
    monkeypatch.setattr(bo,'shared_input',lambda *_args,**_kw:(source,plan,initial))
    monkeypatch.setattr(rough_gpu,'query_gpus',lambda:[])
    monkeypatch.setattr(rough_gpu,'gpu_lease',lambda *_:nullcontext(dict(index=3,uuid='GPU-test')))
    calls=[]
    out=tmp_path/'candidate'
    tuning=out/'tuning'
    expect_compact=frozen_compaction is not False
    manifest=dict(candidate=candidate,updates=updates,plan_sha256=bo.sha(source),
        initial_actor_sha256=bo.sha(initial),ppo_schedule=bo.PPO_SCHEDULE,
        init_at_random_ep_len=True,envs_per_task=bo.MIXED_BO_LAYOUT,
        ppo_algorithm=bo.CURRENT_PPO_ALGORITHM,native_ppo_settings=bo.CURRENT_PPO_SETTINGS,
        teacher_checkpoint_sha256=dict(stairs='1'*64,backpack='2'*64,run='3'*64),
        compact_training_artifacts=expect_compact)
    if frozen_compaction is not None:
        tuning.mkdir(parents=True)
        (tuning/'manifest.json').write_text(json.dumps(manifest))
    if resume_paused:
        out.mkdir(exist_ok=True)
        (out/'USER_PAUSED').touch()

    def worker(command,_folder,_env,_timeout,_root):
        def arg(name): return command[command.index(name)+1]
        dest=Path(arg('--output')); dest.mkdir(parents=True,exist_ok=True)
        if Path(command[2]).name=='mixed_finetune.py':
            calls.append('finetune')
            assert arg('--updates')=='200' and arg('--num-envs')=='4096'
            assert '--training-only' not in command and arg('--device')=='cuda:0'
            assert ('--compact-training-artifacts' in command) is expect_compact
            assert ('--resume' in command) is (frozen_compaction is not None)
            assert ('--resume-paused' in command) is (resume_paused and frozen_compaction is not None)
            (dest/'manifest.json').write_text(json.dumps(manifest))
            (dest/'plan.json').write_text(json.dumps(plan))
            actor=dest/'shared_actor_finetuned.pt'; actor.write_bytes(b'finetuned')
            write_training_completion(dest,candidate,updates,actor)
        else:
            task=arg('--task') if '--task' in command else 'run'
            calls.append(task)
            assert arg('--bo-conditions')==conditions_path and arg('--profile')=='nominal'
            trace=dest/'bo_samples.npz'; trace.write_bytes(b'test-trace')
            (dest/'summary.json').write_text('{}')
            speed=conditions['tasks'][task]['speed_mps']
            motors=[dict(joint=j,rated_torque_nm=4.) for j in sorted(KNEE_ANKLE)]
            row=score_trial(task,speed=speed,dt=.005,expected_steps=4000,mass=20.,forward=speed*20,
                vx=np.full(4000,speed),omega=np.ones((4000,2)),torque=np.ones((4000,6)),
                joint_vel=np.ones((4000,6)),motors=motors,fell=False,min_forward_progress_ratio=.2)
            (dest/'bo_trials.json').write_text(json.dumps(dict(kind=TRIAL_VERSION,task=task,
                design_id=arg('--design-ids'),bank_sha256=task,seed=int(arg('--seed')),trace=str(trace),
                trace_sha256=bo.sha(trace),conditions_sha256=digest(conditions),actor_sha256=bo.sha(arg('--student')),
                plan_sha256=bo.sha(arg('--plan')),trials=[dict(replica=0,**row)])))

    monkeypatch.setattr(bo,'run_worker',worker)
    args=['evaluate','--root',str(tmp_path),'--output',str(out),'--candidate','stairsid','--gpu','3','--timeout','15']
    if resume_paused: args.append('--resume-paused')
    bo.main(args)
    assert not (out/'USER_PAUSED').exists()
    first=bo.read(out/'bo_evaluation.json')
    assert first['ppo_updates']==200 and first['ppo_algorithm']==bo.CURRENT_PPO_ALGORITHM
    assert all(pair['feasible'] for pair in first['pairs'])
    assert calls==['finetune',*bo.TASKS]
    assert len(list((out/'measurements').glob('*/accepted.json')))==3
    frozen_report=(out/'bo_evaluation.json').read_bytes()
    bo.main([*args,'--finetuned',str(tuning)])
    assert (out/'bo_evaluation.json').read_bytes()==frozen_report
    assert calls==['finetune',*bo.TASKS]


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
    receipt = dict(completed_updates=updates, manifest_sha256=manifest_hash,
                   path=blob.name, sha256=bo.sha(blob))
    manifest = bo.read(root/'manifest.json')
    if manifest.get('ppo_algorithm') is not None:
        receipt.update(ppo_algorithm=manifest['ppo_algorithm'], warmup_completed=10)
    if manifest.get('ppo_algorithm') in bo.EXPERT_PRIOR_ALGORITHMS:
        receipt.update(state_kind=bo.NATIVE_PPO_STATE_KIND,
                       teacher_checkpoint_sha256=manifest['teacher_checkpoint_sha256'])
    checkpoint.write_text(json.dumps(receipt))
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
    assert scoring['task_weights'] == pytest.approx(dict.fromkeys(bo.TASKS, 1/3))
    assert scoring.get('cot_basis', 'positive_mechanical') == 'positive_mechanical'
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
    by_id = {ids[k]['stairs']:k for k in keys}
    def synthetic(a, folder, source, plan, initial, candidate_id, conditions):
        calls.append((candidate_id,bo.sha(initial),a.updates))
        key = by_id[candidate_id]
        n = fixed.index(key) if key in fixed else -1
        candidate = dict(structure_key=list(key),design_ids=ids[key])
        manifest = dict(kind='rpo_bo_candidate_v1',candidate=candidate,updates=a.updates,
                        source_plan_sha256=bo.sha(source),initial_actor_sha256=bo.sha(initial),
                        conditions_sha256=digest(conditions))
        native = getattr(a, 'ppo_algorithm', None) is not None
        extra = dict(ppo_algorithm=a.ppo_algorithm, native_ppo_settings=bo.NATIVE_PPO_PROFILES[a.ppo_algorithm]) if native else {}
        manifest.update(extra)
        folder.mkdir(parents=True,exist_ok=True)
        bo.freeze(folder/'manifest.json',manifest)
        reused = getattr(a,'finetuned',None)
        actor = ((Path(reused) if reused else folder/'tuning')/'shared_actor_finetuned.pt').resolve()
        tuning_plan = actor.parent/'plan.json'
        teachers = {task: dict(design_ids=[candidate['design_ids'][task]], bank_sha256=f'bank-{task}')
                    for task in bo.TASKS}
        fixed_dt = conditions.get('measurement_dt_s', DT)
        physics_rate = fixed_dt == .005
        if physics_rate and not reused:
            actor.parent.mkdir()
            actor.write_bytes(f'fitted {candidate_id}'.encode())
            bo.freeze(tuning_plan, dict(teachers=teachers))
            prior = dict(teacher_checkpoint_sha256=dict(stairs='1'*64,backpack='2'*64,run='3'*64)) if native else {}
            bo.freeze(actor.parent/'manifest.json', dict(
                candidate=candidate, updates=a.updates, plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(initial),
                ppo_schedule=bo.PPO_SCHEDULE, init_at_random_ep_len=True, envs_per_task=bo.MIXED_BO_LAYOUT, **extra, **prior))
            write_training_completion(actor.parent, candidate, a.updates, actor)
        elif not physics_rate:
            actor = initial
        pairs = []
        for j, seed in enumerate(conditions['seeds']):
            tasks = {t: dict(replica=0, status='success', mean_forward_mps=.5,
                             metrics={m:n+2 for m in bo.METRICS[t]},
                             dt_s=fixed_dt, steps=round(conditions['duration_s'] / fixed_dt),
                             observed=dict(absolute_mechanical_cot=3 * (n+2) + 2 + 4 * (n == -1 and t == 'stairs'),
                                           cot=n+2,forward_rmse_mps=n+2,roll_pitch_rms_rad_s=n+2,knee_ankle_max_rated_ratio=n+2))
                     for t in bo.TASKS}
            tasks['stairs']['reached_top'] = n not in (-1,0) or j != 2
            if n == 0 and j == len(conditions['seeds']) - 1:
                tasks['stairs'].update(status='failure', reason='below_minimum_forward_speed', metrics=None)
            pairs.append(dict(tasks=tasks,seed=seed,replica=0,
                              feasible=all(t['status']=='success' for t in tasks.values())))
        measurements = []
        for pair in pairs:
            seed = pair['seed']
            for task in bo.TASKS:
                attempt = folder/'measurements'/f'{seed}_{task}'/'synthetic'
                (attempt/'logs').mkdir(parents=True, exist_ok=True)
                (attempt/'native').mkdir(parents=True, exist_ok=True)
                trace = attempt/'native/bo_trials.json'
                if physics_rate:
                    samples = trace.with_name('bo_samples.npz')
                    samples.write_bytes(b'raw physics trace fixture')
                    bo.freeze(trace, dict(
                        kind=bo.TRIAL_VERSION, task=task, seed=seed, actor_sha256=bo.sha(actor),
                        design_id=candidate['design_ids'][task], bank_sha256=teachers[task]['bank_sha256'],
                        plan_sha256=bo.sha(tuning_plan), conditions_sha256=digest(conditions),
                        trace=str(samples), trace_sha256=bo.sha(samples),
                        trials=[pair['tasks'][task]]))
                else:
                    bo.freeze(trace, {})
                bo.freeze(attempt/'logs/exit.json', dict(returncode=0, seconds=3.))
                measurement = dict(report=str(trace), sha256=bo.sha(trace))
                if physics_rate:
                    bo.freeze(attempt.parent/'accepted.json', measurement)
                measurements.append(measurement)
        report = dict(candidate=candidate,conditions=conditions,conditions_sha256=digest(conditions),
                      measurements=measurements,
                      student=str(actor), student_sha256=bo.sha(actor),
                      ppo_updates=a.updates,kind=bo.REPORT_VERSION,domain='simulation',pairs=pairs,
                      evaluation_manifest_sha256=bo.sha(folder/'manifest.json'))
        if native:
            report['ppo_algorithm'] = a.ppo_algorithm
        bo.freeze(folder/'bo_evaluation.json',report)
        return report
    monkeypatch.setattr(bo,'evaluate',synthetic)
    # The real continuation chain is independently exercised by test_dagger_bo_input.
    import dagger_bo_input
    monkeypatch.setattr(dagger_bo_input, 'continuation_actor', lambda *_: actor)

    return SimpleNamespace(root=tmp_path,calls=calls,selection=selection_path,student=actor,
                           fixed_ids=[ids[k]['stairs'] for k in fixed])

def test_official_panel_initializes_once_and_only_measures(official_panel):
    p=official_panel
    bo.main(['initialize','--migrated'])
    assert p.calls==[(k,bo.sha(p.student),200) for k in p.fixed_ids]
    panel=p.root/'bo/calibration50_pcgrad_ppo200_fixed_all_std_200hz'
    manifest=bo.read(panel/'manifest.json')
    assert manifest['updates']==200 and manifest['selection_velocity_frame']=='horizontal_heading'
    assert manifest['ppo_schedule']==dict(epochs=5,minibatches=4)
    assert manifest['init_at_random_ep_len'] is True
    assert manifest['envs_per_task']==dict(stairs=1366,backpack=1365,run=1365)
    assert manifest['ppo_algorithm']==bo.CURRENT_PPO_ALGORITHM
    assert manifest['native_ppo_settings']==bo.CURRENT_PPO_SETTINGS
    bo.main(['initialize','--migrated'])
    assert len(p.calls)==50 and not (panel/'scoring.json').exists()
    assert len(bo.read(panel/'measurements_completed.json')['reports'])==50
    assert len(bo.checked_measurement_panel(panel)[0])==50


def test_scheme_b_reference_freezes_and_validates_matching_native_measurements(official_panel,monkeypatch):
    from types import SimpleNamespace
    import finetune_state
    p=official_panel; source=p.root/'shared_plan/plan.json'
    row=next(r for r in bo.read(p.root/'bank.json')['structures'] if r['design_ids']['stairs']==p.fixed_ids[1])
    candidate=dict(structure_key=row['key'],design_ids=row['design_ids'])
    # A new reference must not depend on the obsolete V4 archive.
    monkeypatch.setattr(bo,'ORIGINAL_RPO_ARCHIVE',p.root/'absent_historical_reference')
    monkeypatch.setattr(bo,'RPO_REFERENCE_KEY',tuple(row['key']))
    shared=bo.read(source)
    shared['teachers']={t:dict(checkpoint_sha256=str(n)*64) for n,t in enumerate(bo.TASKS,1)}
    source.write_text(json.dumps(shared))
    plan=dict(schema=shared['schema'],selected_structure_keys=[row['key']],
              teachers={t:dict(design_ids=[i],**shared['teachers'][t]) for t,i in row['design_ids'].items()})
    plan_path=p.root/'singleton.json'; plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(finetune_state,'select_candidate',lambda *_args,**_kw:(plan,candidate))
    conditions=bo.read_conditions(bo.PHYSICS_CONDITIONS)
    dest=p.root/'reference_b'; dest.mkdir()
    args=SimpleNamespace(reference_plan=plan_path,resume_paused=False,updates=200)
    bo.prepare_scheme_b_reference(args,dest,source,p.student,conditions)
    protocol=bo.read(dest/'reference_protocol.json')
    assert protocol['ppo_algorithm']==bo.CURRENT_PPO_ALGORITHM
    report,_=bo.checked_original_rpo_reference(dest,protocol,conditions)
    assert report['candidate']==candidate and report['pairs'][0]['feasible']
    assert bo.original_rpo_scoring(report,{},protocol,{},conditions)['benchmark']==protocol


@pytest.mark.parametrize('flags', [
    ['--conditions',str(bo.LEGACY_FORMAL_CONDITIONS)],
    ['--conditions',str(bo.FORMAL_CONDITIONS)],
    ['--updates','100'],
])
def test_formal_cli_retires_old_measurement_protocols(official_panel,flags):
    with pytest.raises(ValueError,match='200-update'):
        bo.main(['initialize','--migrated',*flags])
    assert not official_panel.calls


@pytest.mark.parametrize('change', ['actor','budget','report','substituted_manifest','receipt','algorithm'])
def test_official_panel_rejects_changed_evidence_on_reuse(official_panel,change):
    p=official_panel
    bo.main(['initialize','--root',str(p.root)])
    calib=p.root/'bo/calibration50_pcgrad_ppo200_fixed_all_std_200hz'
    args=['initialize','--root',str(p.root),'--output',str(calib)]
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
        changed=calib/('measurements_completed.json' if change=='receipt' else 'manifest.json')
        value=bo.read(changed)
        if change=='receipt': value['conditions_sha256']='wrong'
        else: value['ppo_algorithm']='old'
        changed.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        bo.main(args)
    assert len(p.calls)==50
    assert not (p.root/'bo/search_single_seed_ppo100_random_200hz/manifest.json').exists()

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


def test_physics_protocol_preserves_tasks_and_separates_scores():
    old = bo.read_conditions(bo.FORMAL_CONDITIONS)
    current = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    assert current == dict(old, measurement_dt_s=.005)
    assert bo.calibration_label(current) == bo.calibration_label(old) + '_200hz'
    assert bo.official_conditions_path(digest(current)) == bo.PHYSICS_CONDITIONS


@pytest.mark.parametrize('fell', [False, True])
def test_physics_step_capture_records_braking_and_excludes_reset_state(tmp_path, fell):
    import torch
    from bo_capture import TrialCapture

    cfg = bo.read_conditions(bo.FORMAL_CONDITIONS)
    cfg.update(measurement_dt_s=.005, duration_s=.04)
    motors = [dict(joint=j, rated_torque_nm=4.) for j in sorted(KNEE_ANKLE)]
    data = SimpleNamespace(root_pos_w=torch.tensor([[0., 0., 1.]]),
        root_lin_vel_w=torch.tensor([[.5, 0., 0.]]), root_ang_vel_b=torch.zeros(1, 3),
        applied_torque=torch.zeros(1, 6), joint_vel=torch.full((1, 6), 2.))
    robot = SimpleNamespace(data=data, root_physx_view=SimpleNamespace(get_masses=lambda: torch.tensor([[20.]])))
    scene = SimpleNamespace(env_origins=torch.zeros(1, 3))
    raw = SimpleNamespace(robot=robot, scene=scene, step_dt=.02, physics_dt=.005,
                          num_envs=1, reset_terminated=torch.tensor([fell]))
    substeps = iter(range(1, 9))

    def physics_update(dt):
        i = next(substeps)
        data.root_pos_w[0, 0] = i*.005
        data.root_lin_vel_w[0, 0] = 1.5 if i % 4 == 1 else .5
        data.root_ang_vel_b[0, :2] = torch.tensor([3., 4.]) if i % 4 == 1 else 0.
        data.applied_torque[0, 0] = {1: 8., 2: -4.}.get(i % 4, 0.)

    scene.update = physics_update
    recorder = TrialCapture(raw, 'stairs', cfg, 20261118, motors)
    for done in (fell, False):
        for _ in range(4):
            scene.update(dt=.005)
        if done:
            recorder.before_reset(torch.tensor([0]))
            data.root_pos_w[0, 0] = 99.
        recorder.capture(torch.tensor([[.5, 0., 0.]]), torch.tensor([done]))
    recorder.publish(tmp_path, {})
    assert scene.update is physics_update
    row = bo.read(tmp_path/'bo_trials.json')['trials'][0]
    assert (row['dt_s'], row['steps'], row['status']) == (.005, 4 if fell else 8, 'failure' if fell else 'success')
    assert row['forward_m'] == pytest.approx(.02 if fell else .04)
    assert row['observed']['forward_rmse_mps'] == pytest.approx(.5)
    assert row['observed']['roll_pitch_rms_rad_s'] == pytest.approx(2.5)
    assert row['observed']['positive_mechanical_work_j'] == pytest.approx(.08 if fell else .16)
    assert row['observed']['absolute_mechanical_work_j'] == pytest.approx(.12 if fell else .24)
    assert row['observed']['absolute_mechanical_cot'] == pytest.approx(1.5 * row['observed']['cot'])
    with np.load(tmp_path/'bo_samples.npz') as trace:
        assert (trace['dt_s'], trace['control_dt_s']) == (.005, .02)


def test_calibration_evaluation_timer_counts_only_accepted_new_measurements(tmp_path):
    reports = []
    for index, task in enumerate(bo.TASKS, 1):
        for name in ('accepted', 'discarded'):
            attempt = tmp_path/'design_01'/'measurements'/task/name
            (attempt/'logs').mkdir(parents=True)
            (attempt/'native').mkdir()
            (attempt/'logs/exit.json').write_text(json.dumps(dict(returncode=0, seconds=index if name == 'accepted' else 999)))
            p = attempt/'native/bo_trials.json'
            p.write_text('{}', encoding='utf8')
            if name == 'accepted':
                reports.append(dict(report=str(p), sha256=bo.sha(p)))
    assert bo.calibration_evaluation_timing(tmp_path, reports) == dict(accepted_measurements=3, evaluation_seconds=6)
    with pytest.raises(ValueError, match='Duplicate'):
        bo.calibration_evaluation_timing(tmp_path, reports + reports)



def test_measurement_only_panel_defers_scoring(official_panel):
    p = official_panel
    bo.main(['initialize', '--migrated', '--measure-only'])
    out = p.root/'bo/calibration50_pcgrad_ppo200_fixed_all_std_200hz'
    assert len(bo.read(out/'measurements_completed.json')['reports']) == 50
    assert bo.read(out/'evaluation_timing.json') == dict(accepted_measurements=150, evaluation_seconds=450.)
    assert not (out/'scoring.json').exists()


def test_scheme_b_panel_reuses_tuning_without_legacy_scoring_file(official_panel):
    p=official_panel
    bo.main(['initialize','--migrated'])
    original=p.root/'bo/calibration50_pcgrad_ppo200_fixed_all_std_200hz'
    frozen=bo.sha(original/'manifest.json'),bo.sha(original/'measurements_completed.json')
    target=p.root/'remeasured'
    assert not (original/'scoring.json').exists()
    bo.main(['initialize','--migrated','--reuse-finetuned-panel',str(original),'--output',str(target)])
    assert len(p.calls)==100
    assert not (target/'design_01/tuning').exists()
    reports,*_=bo.checked_measurement_panel(target)
    assert Path(reports[0]['student']).is_relative_to(original/'design_01/tuning')
    assert (bo.sha(original/'manifest.json'),bo.sha(original/'measurements_completed.json'))==frozen


def test_freeze_scoring_only_reads_verified_native_measurements(official_panel, monkeypatch):
    p = official_panel
    bo.main(['initialize', '--migrated', '--measure-only'])
    out = p.root/'bo/calibration50_pcgrad_ppo200_fixed_all_std_200hz'
    scoring_out=p.root/'new_scoring'
    args=['freeze-scoring','--measurements-root',str(out),'--output',str(scoring_out),'--original-rpo-reference',str(p.root/'reference')]
    reference=bo.read(out/'design_02/bo_evaluation.json')
    monkeypatch.setattr(bo,'checked_original_rpo_reference',lambda *_:(reference,dict(root='fixture')))

    def forbidden(*args, **kwargs):
        pytest.fail('Freezing existing measurements must not start a simulator or train an actor')

    monkeypatch.setattr(bo, 'evaluate', forbidden)
    monkeypatch.setattr(bo, 'evaluate_native', forbidden)
    monkeypatch.setattr(bo, 'run_worker', forbidden)
    report = out/'design_01/bo_evaluation.json'
    receipt = out/'measurements_completed.json'
    native = Path(bo.read(report)['measurements'][0]['report'])
    accepted = native.parents[2]/'accepted.json'
    raw = Path(bo.read(native)['trace'])
    changes = {
        receipt: json.dumps(dict(bo.read(receipt), conditions_sha256='wrong')).encode(),
        report: report.read_bytes() + b' ',
        accepted: json.dumps(dict(bo.read(accepted), sha256='wrong')).encode(),
        native: native.read_bytes() + b' ',
        raw: b'other physics trace',
        out/'design_01/tuning/shared_actor_finetuned.pt': b'other fitted actor',
    }
    for changed, replacement in changes.items():
        original = changed.read_bytes()
        changed.write_bytes(replacement)
        with pytest.raises(ValueError):
            bo.main(args)
        assert not (scoring_out/'scoring.json').exists()
        changed.write_bytes(original)

    frozen = {path: bo.sha(path) for path in (*changes, out/'manifest.json')}
    paused = out/'USER_PAUSED'
    paused.touch()
    with pytest.raises(RuntimeError, match='USER_PAUSED'):
        bo.main(args)
    assert not (scoring_out/'scoring.json').exists()
    assert paused.is_file()
    bo.main([*args, '--resume-paused'])
    assert not paused.exists()
    bo.main(args)
    assert len(p.calls) == 50
    assert {path: bo.sha(path) for path in frozen} == frozen
    scoring = bo.read(scoring_out/'scoring.json')
    assert scoring['task_weights'] == dict(stairs=.4, backpack=.3, run=.3)
    assert scoring['cot_basis'] == 'absolute_mechanical'
    assert bo.read(scoring_out/'completed.json')['scoring_sha256'] == bo.sha(scoring_out/'scoring.json')
    assert len(bo.read(scoring_out/'initial_observations.json')['candidates'])==50
