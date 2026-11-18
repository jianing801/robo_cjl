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
