"""CPU checks for protocol isolation and closed-loop selection, not gait validation."""
import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline
from evaluation import candidate_rank


def test_curriculum_reaches_target_and_nominal_never_randomizes():
    assert [pipeline.round_strength(i, 3, .25, 1., 'robust') for i in range(4)] == [.25, .5, .75, 1.]
    assert pipeline.round_strength(0, 0, .25, 1., 'robust') == 1.
    assert pipeline.round_strength(2, 3, .25, 1., 'nominal') == 0.
    assert pipeline.round_strength(0, 0, .25, 0., 'nominal') == 0.
    for initial, target in [(1., .5), (-.1, 1.), (.25, float('nan'))]:
        with pytest.raises(ValueError):
            pipeline.round_strength(0, 3, initial, target, 'robust')


def test_balanced_selection_excludes_calibration_and_spans_seen_and_unseen():
    from itertools import product
    from collections import Counter
    from collection_grid import select_collection_grid, selection_panel
    keys=list(product(range(20,30), range(25,35), 'ABCDE', 'abcde'))
    grid=select_collection_grid(keys)
    reserved=[(t,c,k,a) for k in 'ABCDE' for a in 'abcde' for t,c in [(20,25),(21,26)]]
    panel=selection_panel(keys, grid, reserved)
    assert len(panel) == len(set(panel)) == 200 and set(panel).isdisjoint(reserved)
    assert Counter((row[2:], row in grid) for row in panel) == {
        ((k,a),seen):4 for k,a,seen in product('ABCDE','abcde',(True,False))}
    for pair in product('ABCDE','abcde'):
        seen_lengths = [row[:2] for row in panel if row[2:] == pair and row in grid]
        assert all(len(set(axis)) == 4 for axis in zip(*seen_lengths))
    assert selection_panel(keys[::-1],grid[::-1],reserved[::-1]) == panel
    with pytest.raises(ValueError):
        selection_panel(keys, grid, reserved[:-1])
    limited = [row for row in grid if row[2:] != ('A','a') or row[0] == 20]
    with pytest.raises(ValueError, match='motor pair'):
        selection_panel(keys, limited, reserved)


def row(index=0, **changes):
    result = dict(environment=index, design='d', seconds=20., mean_forward_speed=.5,
                  ascent_m=.3, fell=False, tracking_rmse=.1, tracking_samples=999,
                  tracking_seconds=19.98, commanded_distance_m=10., command_aligned_distance_m=10.,
                  active_command_seconds=20., command_seconds=20., commanded_vx_rms_m_s=.5, step_dt_s=.02)
    result.update(changes)
    result.update(command_samples=round(result['command_seconds']/result['step_dt_s']),
        moving_command_samples=round(result['command_seconds']/result['step_dt_s']),standing_command_samples=0,
        moving_command_seconds=result['command_seconds'],standing_command_seconds=0.,
        moving_commanded_vx_rms_m_s=result['commanded_vx_rms_m_s'],standing_commanded_vx_rms_m_s=None,
        moving_tracking_samples=result['tracking_samples'],standing_tracking_samples=0,
        moving_tracking_seconds=result['tracking_seconds'],standing_tracking_seconds=0.,
        moving_tracking_rmse=result['tracking_rmse'],standing_tracking_rmse=None,standing_vx_rms_m_s=None)
    return result


def test_evaluation_requires_observed_tracking_and_records_falls():
    result = pipeline.evaluation_statistics('stairs', [row(), row(1, fell=True)], 'robust')
    assert result['failed'] == [1] and result['success_rate'] == .5
    assert 'tracking_rmse_mean' not in result
    missing = row(); del missing['command_aligned_distance_m']
    with pytest.raises(ValueError, match='command_aligned_distance_m'):
        pipeline.evaluation_statistics('run', [missing], 'robust')


def comparison_protocol(profile='robust', strength=1.):
    return dict(version='fixture', profile=profile, strength=strength, protocol_hash='fixture')


def metrics(rates, error=.1, *, profile='robust', seeds=(7,)):
    tasks = ('stairs', 'backpack', 'run')
    records = [dict(task=task, profile=profile, seed=seed, evaluation_protocol=comparison_protocol(profile),
        **pipeline.evaluation_statistics(task,
        [row(i, fell=i>=round(10*rate), tracking_rmse=error) for i in range(10)], profile))
        for task, rate in zip(tasks, rates) for seed in seeds]
    return pipeline.aggregate_evaluations(records, tasks)


def test_selection_does_not_hide_one_task_regression_in_average():
    teacher = metrics([.8, .8, .8])
    retained = metrics([.8, .8, .8], .1)
    better_average_but_regression = metrics([1., 1., .6], .01)
    assert candidate_rank(retained, teacher) > candidate_rank(better_average_but_regression, teacher)
    assert candidate_rank(metrics([.8, .8, .8], .08), teacher) == candidate_rank(retained, teacher)


def test_distill_selection_defaults_to_twenty_percent_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'execute_pipeline', lambda args, seeds: args.min_command_progress_ratio)
    assert pipeline.main(['--plan', 'plan.json', '--output', str(tmp_path/'new')]) == .2


def test_task_success_selects_the_highest_equal_task_average():
    from dagger_comparison import TASK_SUCCESS_SELECTION_RULE, selection_rank

    teacher = metrics([.8, .8, .8])
    high_success = metrics([1., 1., .6])
    balanced = metrics([.8, .8, .8])
    assert selection_rank(high_success, teacher, TASK_SUCCESS_SELECTION_RULE) == (pytest.approx(2.6/3),)
    assert selection_rank(high_success, teacher, TASK_SUCCESS_SELECTION_RULE) > selection_rank(
        balanced, teacher, TASK_SUCCESS_SELECTION_RULE)


def report_for(command):
    profile = command[command.index('--profile')+1]
    strength = float(command[command.index('--randomization-strength')+1])
    actor = command[command.index('--student')+1] if '--student' in command else None
    return dict(rows=[row(i, mean_horizontal_forward_speed_m_s=10./19.98)
                      for i in range(6 if profile == 'robust' else 1)],
        collection_protocol=pipeline.protocol_metadata(profile, strength), task='stairs',
        teacher='teacher', student=pipeline.sha(actor) if actor else None, design_ids=['d'], steps=1000,
        velocity_frame='horizontal_heading')


@pytest.mark.parametrize('corruption', ['protocol', 'actor', 'missing_copy', 'duplicate_env', 'wrong_design'])
def test_evaluation_rejects_wrong_protocol_actor_or_environment_coverage(corruption):
    protocol = {'profile': 'robust'}
    report = dict(rows=[row(i) for i in range(6)], collection_protocol=protocol,
        task='stairs', teacher='teacher', student=None, design_ids=['d'], steps=1000)
    report = deepcopy(report)
    if corruption == 'protocol': report['collection_protocol']['profile'] = 'nominal'
    elif corruption == 'actor': report['student'] = 'unexpected_actor'
    elif corruption == 'missing_copy': report['rows'].pop()
    elif corruption == 'duplicate_env': report['rows'][-1]['environment'] = 0
    else: report['rows'][-1]['design'] = 'wrong_design'
    with pytest.raises(ValueError):
        pipeline.validate_evaluation_report(report, 'stairs', ['d'],
            {'checkpoint_sha256': 'teacher', 'copies': 4}, None, protocol)


def test_legacy_stair_plan_evaluates_all_six_robust_strata_but_keeps_nominal_copies():
    spec = {'checkpoint_sha256': 'teacher', 'copies': 4}
    for profile, copies in [('nominal', 4), ('robust', 6)]:
        protocol = {'profile': profile}
        rows = [row(i, design=['a', 'b'][i % 2]) for i in range(2*copies)]
        report = dict(rows=rows, collection_protocol=protocol, task='stairs', teacher='teacher',
            student=None, design_ids=['a', 'b'], steps=1000)
        assert pipeline.validate_evaluation_report(report, 'stairs', ['a', 'b'], spec, None, protocol) == rows


@pytest.mark.parametrize('task', ['stairs', 'backpack', 'run'])
def test_selection_refuses_body_frame_and_requires_recorded_horizontal_speed(task):
    command = ['--profile', 'robust', '--randomization-strength', '1.0']
    report = report_for(command)
    report['task']=task
    spec = {'checkpoint_sha256': 'teacher', 'copies': 6}
    args = (task, ['d'], spec, None, report['collection_protocol'])
    with pytest.raises(ValueError, match='velocity frame'):
        pipeline.validate_evaluation_report(dict(report, velocity_frame='body'), *args,
            velocity_frame='horizontal_heading')
    report['rows'][0].pop('mean_horizontal_forward_speed_m_s')
    with pytest.raises(ValueError, match='horizontal speed'):
        pipeline.validate_evaluation_report(report, *args, velocity_frame='horizontal_heading')


def test_evaluation_resume_binds_policy_profile_seed_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'protocol_metadata', lambda p, s: {'version': 'test', 'profile': p, 'strength': s})
    calls = []
    def fake_child(command, *args):
        calls.append(command)
        folder = Path(command[command.index('--output')+1]); folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps(report_for(command)))
    monkeypatch.setattr(pipeline, 'child', fake_child)
    actor = tmp_path/'actor.pt'; actor.write_bytes(b'actor-one')
    plan = {'teachers': {'stairs': {'checkpoint_sha256': 'teacher', 'copies': 1}}}
    kwargs = dict(plan=plan, plan_path=tmp_path/'plan.json', root=tmp_path, jobs=[('stairs', ['d'])],
                  student=str(actor), profile='robust', strength=1., seed=7, suite='selection/round_0',
                  environment={}, timeout=30)
    first = pipeline.run_evaluation(**kwargs)
    assert first[0]['success_rate'] == 1.
    assert calls[0][calls[0].index('--profile')+1] == 'robust'
    assert pipeline.run_evaluation(**kwargs) == first and len(calls) == 1
    with pytest.raises(ValueError, match='changed'):
        pipeline.run_evaluation(**dict(kwargs, seed=8))
    with pytest.raises(ValueError, match='changed'):
        pipeline.run_evaluation(**dict(kwargs, criteria={'min_command_progress_ratio': .3}))
    actor.write_bytes(b'actor-two')
    with pytest.raises(ValueError, match='changed'):
        pipeline.run_evaluation(**kwargs)


def test_rescore_reuses_frozen_episodes_without_changing_the_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'protocol_metadata', comparison_protocol)

    def child(command, *_):
        folder = Path(command[command.index('--output')+1])
        folder.mkdir(parents=True)
        report = report_for(command)
        for observed in report['rows']:
            observed.update(command_aligned_distance_m=3., mean_horizontal_forward_speed_m_s=3./19.98)
        report['rows'][0]['fell'] = True
        (folder/'summary.json').write_text(json.dumps(report))

    monkeypatch.setattr(pipeline, 'child', child)
    kwargs = dict(plan={'teachers': {'stairs': {'checkpoint_sha256': 'teacher', 'copies': 1}}},
                  plan_path=tmp_path/'plan.json', root=tmp_path, jobs=[('stairs', ['d'])],
                  student=None, profile='robust', strength=1., seed=7, suite='selection/teacher',
                  environment={}, timeout=30, criteria={'min_command_progress_ratio': .5})
    assert pipeline.run_evaluation(**kwargs)[0]['success_rate'] == 0.
    receipt, = (tmp_path/'evaluation_commits').rglob('*.json')
    original = receipt.read_bytes()
    rescored, = pipeline.run_evaluation(**kwargs, reuse_only=True, rescore={'min_command_progress_ratio': .2})
    assert rescored['success_rate'] == pytest.approx(5/6)
    assert receipt.read_bytes() == original
    archived = json.loads(original)['identity']['protocol']
    monkeypatch.setattr(pipeline, 'protocol_metadata', lambda *_: pytest.fail('Live protocol was read'))
    assert pipeline.run_evaluation(**kwargs, reuse_only=True, frozen_protocol=archived)[0]['success_rate'] == 0.
    with pytest.raises(ValueError, match='matching committed evidence'):
        pipeline.run_evaluation(**kwargs, frozen_protocol=archived)
    Path(rescored['summary']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        pipeline.run_evaluation(**kwargs, reuse_only=True, frozen_protocol=archived,
                                rescore={'min_command_progress_ratio': .2})


def test_final_test_suite_does_not_reuse_selection_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'protocol_metadata', lambda p, s: {'profile': p, 'strength': s})
    seen = []
    def fake_child(command, *args):
        folder = Path(command[command.index('--output')+1]); folder.mkdir(parents=True)
        seen.append((command[command.index('--seed')+1], folder))
        (folder/'summary.json').write_text(json.dumps(report_for(command)))
    monkeypatch.setattr(pipeline, 'child', fake_child)
    kwargs = dict(plan={'teachers': {'stairs': {'checkpoint_sha256': 'teacher', 'copies': 1}}}, plan_path=tmp_path/'plan.json',
                  root=tmp_path, jobs=[('stairs', ['d'])], student=None, profile='robust', strength=1.,
                  environment={}, timeout=30)
    pipeline.run_evaluation(**kwargs, seed=7, suite='selection/teacher')
    pipeline.run_evaluation(**kwargs, seed=8, suite='test/robust')
    assert len(seen) == 2 and seen[0][0] != seen[1][0] and seen[0][1] != seen[1][1]


def test_last_epoch_ignores_supervised_best_and_checks_identity(tmp_path):
    reports=[]
    for i in range(10):
        path=tmp_path/f'{i}.pt'
        path.write_text(str(i))
        reports.append(dict(epoch=i,student=str(path),student_sha256=pipeline.sha(path)))
    report=dict(checkpoints=list(reversed(reports)),student=reports[0]['student'])
    assert pipeline.last_checkpoint(report,10) == reports[-1]
    path.write_text('changed')
    with pytest.raises(ValueError):
        pipeline.last_checkpoint(report,10)
    with pytest.raises(ValueError):
        pipeline.last_checkpoint(dict(checkpoints=reports[:-1]),10)


def test_multiseed_suite_uses_all_common_seeds_and_separate_cache_names(tmp_path, monkeypatch):
    calls = []
    def evaluate(plan, plan_path, root, jobs, student, profile, strength, seed, suite, *args):
        calls.append((student, seed, suite))
        return [dict(task=task, profile=profile, seed=seed, evaluation_protocol=comparison_protocol(profile,strength),
                     **pipeline.evaluation_statistics(task, [row()], profile))
                for task in plan['schema']['tasks']]
    monkeypatch.setattr(pipeline, 'run_evaluation', evaluate)
    plan = {'schema': {'tasks': ['stairs', 'backpack', 'run']}}
    result = pipeline.run_selection_suite(plan, tmp_path/'plan', tmp_path, [], 'actor', 'robust', 1.,
        [11, 12, 13], 'selection/round_0/epoch_4', {}, 30)
    assert [call[1] for call in calls] == [11, 12, 13]
    assert len({call[2] for call in calls}) == 3
    assert result['task_metrics']['stairs']['environments'] == 3
    assert set(result['by_seed']) == {'11', '12', '13'}


def test_start_uses_approved_distillation_protocol(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    monkeypatch.delitem(sys.modules,'start',raising=False)  # rough_bo also provides a start.py.
    import start
    folders = {}
    teachers = {}
    for public, task in [('rough', 'stairs'), ('backpack', 'backpack'), ('amp', 'run')]:
        folder = tmp_path/public; folder.mkdir(); folders[public] = folder
        checkpoint = str(folder/'actor.pt')
        (folder/'completed.json').write_text(json.dumps({'checkpoint': checkpoint}))
        teachers[task] = {'checkpoint': checkpoint}
    (tmp_path/'shared_plan').mkdir()
    (tmp_path/'shared_plan'/'plan.json').write_text(json.dumps({'teachers': teachers}))
    calls = []
    monkeypatch.setattr(start, 'select_run', lambda root, task: (folders[task], True))
    monkeypatch.setattr(start, 'ensure_banks', lambda *args: None)
    monkeypatch.setattr(start, 'run', lambda command, **kwargs: calls.append([str(value) for value in command]))
    start.main(['蒸馏','--root',str(tmp_path),'--dry-run','--resume-paused'])
    command, = calls
    for flag,value in {'--epochs':'10','--dagger-rounds':'3','--collect-steps':'250',
                       '--replay-cache-shards':'12','--output':str(tmp_path/'shared_policy')}.items():
        assert command[command.index(flag)+1] == value
    assert '--screen-designs' not in command and '--resume-paused' in command


def test_retired_absolute_motion_tolerance_has_explicit_migration_errors(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--plan', 'unused', '--output', 'unused',
        '--tracking-abs-tolerance', '.1'])
    with pytest.raises(ValueError, match='retired.*forward progress'):
        pipeline.main()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    monkeypatch.delitem(sys.modules,'start',raising=False)  # rough_bo also provides a start.py.
    import start
    with pytest.raises(SystemExit):
        start.main(['蒸馏', '--tracking-abs-tolerance', '.1'])
    assert '--tracking-abs-tolerance' in capsys.readouterr().err


def test_removed_standing_tolerance_is_rejected_by_both_entrypoints(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--plan', 'unused', '--output', 'unused',
        '--standing-speed-tolerance', '.1'])
    with pytest.raises(ValueError, match='standing-speed-tolerance.*retired'):
        pipeline.main()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    monkeypatch.delitem(sys.modules,'start',raising=False)  # rough_bo also provides a start.py.
    import start
    with pytest.raises(SystemExit):
        start.main(['蒸馏', '--standing-speed-tolerance', '.1'])
    assert '关闭站立指令' in capsys.readouterr().err


def test_pipeline_resumes_and_selects_only_last_epoch_of_three_dagger_rounds(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from itertools import product
    from types import ModuleType
    tasks = ['stairs', 'backpack', 'run']
    keys=list(product([.2,.25,.3,.35,.4],[.25,.30,.32,.38,.45],[f'K{i}' for i in range(5)],[f'A{i}' for i in range(5)]))
    reserved=[(t,c,k,a) for k,a in product([f'K{i}' for i in range(5)],[f'A{i}' for i in range(5)]) for t,c in [(.2,.25),(.25,.32)]]
    monkeypatch.setattr(pipeline, 'calibration_keys',lambda:reserved)
    plan = {'selected_structure_keys': keys, 'teachers': {}, 'schema': {'tasks': tasks}, 'scope': 'all'}
    for task in tasks:
        bank = tmp_path/f'{task}.json'
        entries = [dict(id=f'{task}_{i}', thigh_m=k[0], calf_m=k[1], knee_motor=k[2], ankle_motor=k[3])
                   for i,k in enumerate(keys)]
        bank.write_text(json.dumps({'designs': entries}))
        plan['teachers'][task] = {'bank': str(bank), 'design_ids': [row['id'] for row in entries],
                                  'copies': 8, 'checkpoint_sha256': task}
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(pipeline, 'load_plan', lambda path: plan)
    monkeypatch.setattr(pipeline, 'protocol_metadata', lambda profile, strength: {'profile': profile, 'strength': strength})
    gpu = ModuleType('robolab.rough_gpu')
    gpu.query_gpus = lambda: []
    gpu.candidates = lambda *args, **kwargs: []
    gpu.gpu_lease = lambda *args: nullcontext('fake_gpu')
    gpu.child_environment = lambda *args: {}
    monkeypatch.setitem(sys.modules, 'robolab.rough_gpu', gpu)
    children = []; paused = []
    def child(command, *args):
        if any(str(part).endswith('fit.py') for part in command) and not paused:
            paused.append(True)
            raise KeyboardInterrupt('simulated manual pause after collection')
        children.append(command)
        folder = Path(command[command.index('--output')+1]); folder.mkdir(parents=True)
        if any(str(part).endswith('fit.py') for part in command):
            checkpoints = []
            for epoch in range(10):
                actor = folder/f'epoch_{epoch}.pt'; actor.write_text(f'{folder.parent.name}:{epoch}')
                checkpoints.append(dict(epoch=epoch, student=str(actor), student_sha256=pipeline.sha(actor),
                    balanced_validation_MSE=0. if epoch == 0 else 100.))
            (folder/'summary.json').write_text(json.dumps(dict(checkpoints=checkpoints,
                student=checkpoints[0]['student'], student_sha256=checkpoints[0]['student_sha256'])))
        else:
            (folder/'data.pt').write_bytes(b'fake collection; real serialization is tested separately')
    monkeypatch.setattr(pipeline, 'child', child)
    suites = []
    def suite(plan, plan_path, root, jobs, actor, profile, strength, seeds, name, *args):
        assert [(task, len(ids)) for task, ids in jobs] == [(task, 200) for task in tasks]
        suites.append((actor, tuple(seeds), name))
        if actor is None:
            rate = 1.
        else:
            epoch = int(Path(actor).stem.split('_')[-1])
            round_index = int(Path(actor).parents[1].name.split('_')[-1])
            assert epoch == 9
            rate = {1:.5, 2:.9, 3:.6}[round_index]
        return {'task_metrics': metrics([rate]*3,profile=profile,seeds=seeds), 'evaluations': [{'suite': name}]}
    monkeypatch.setattr(pipeline, 'run_selection_suite', suite)
    output=tmp_path/'shared_policy'
    argv=['pipeline','--plan',str(plan_path),'--output',str(output)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(KeyboardInterrupt): pipeline.main()
    assert (output/'USER_PAUSED').exists()
    original_argv = sys.argv.copy()
    monkeypatch.setattr(sys, 'argv', original_argv+['--resume'])
    with pytest.raises(RuntimeError,match='resume-paused'): pipeline.main()
    monkeypatch.setattr(sys, 'argv', original_argv+['--resume-paused','--epochs','7'])
    with pytest.raises(ValueError,match='changed'): pipeline.main()
    assert (output/'USER_PAUSED').exists()
    monkeypatch.setattr(sys, 'argv', original_argv+['--resume-paused'])
    pipeline.main()
    assert not (output/'USER_PAUSED').exists()
    first_collect = [cmd for cmd in children if any('/data/round_0/' in str(v).replace('\\','/') for v in cmd)]
    assert len(first_collect)==3
    assert all(c[c.index('--num-envs')+1]=='4096' and len(c[c.index('--design-ids')+1].split(','))==400 for c in first_collect)
    completed=json.loads((output/'completed.json').read_text())
    selection=json.loads((output/'selection.json').read_text())
    assert len(suites)==4 and sum(actor is None for actor,_,_ in suites)==1
    assert [x['round_index'] for x in selection['candidates']]==[1,2,3]
    assert completed['selected_round']==2
    assert pipeline.sha(completed['student'])==completed['student_sha256']
    for r in range(1,4):
        prior=completed['rounds'][r-1]['student']
        commands=[c for c in children if any(f'/data/round_{r}/' in str(v).replace(chr(92),'/') for v in c)]
        assert len(commands)==3 and all(c[c.index('--student')+1]==prior for c in commands)
        assert Path(prior).name=='epoch_9.pt'
    assert len(selection['design_keys'])==200 and len(selection['training_design_keys'])==100
    assert not completed['full_bank_evaluated']
    assert selection['budget']['environments_per_task_batch'] == dict.fromkeys(tasks,1600)
    assert selection['budget']['total_selection_rollout_processes'] == 36
    assert completed['protocol']['selection_version'] == 'last_epoch_dagger_paired_200'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import bo
    monkeypatch.setattr(bo,'calibration_keys',lambda:reserved)
    manifest=json.loads((output/'manifest.json').read_text())
    bo.check_formal_selection(tmp_path,plan,manifest,completed,Path(completed['student']).resolve())


def test_final_comparison_rejects_missing_policies_and_empty_records():
    with pytest.raises(ValueError, match='nonempty'):
        pipeline.summarize_final_evaluations([], ['stairs'], ['robust'])
    teacher = dict(task='stairs', policy='teacher', profile='robust', seed=11,
        evaluation_protocol=comparison_protocol(),
        **pipeline.evaluation_statistics('stairs', [row()], 'robust'))
    with pytest.raises(ValueError, match='Missing final robust/student'):
        pipeline.summarize_final_evaluations([teacher], ['stairs'], ['robust'])


def test_final_comparison_keeps_design_regression_when_task_success_rates_match():
    records = []
    for policy, failed_design in [('teacher', 'b'), ('student', 'a')]:
        rows = [row(i, design=design, fell=design == failed_design) for i, design in enumerate(('a', 'b'))]
        records.append(dict(task='stairs', policy=policy, profile='robust', seed=11,
            evaluation_protocol=comparison_protocol(),
            **pipeline.evaluation_statistics('stairs', rows, 'robust')))
    result = pipeline.summarize_final_evaluations(records, ['stairs'], ['robust'])['robust']
    assert result['policy_metrics']['teacher']['stairs']['success_rate'] == .5
    assert result['policy_metrics']['student']['stairs']['success_rate'] == .5
    comparison = result['teacher_comparison']['tasks']['stairs']
    assert comparison['mean_success_rate_gap'] == 0.
    assert comparison['mean_success_rate_shortfall'] == .5
    assert comparison['designs']['a']['success_rate_gap'] == -1.
    assert comparison['designs']['b']['success_rate_gap'] == 1.
