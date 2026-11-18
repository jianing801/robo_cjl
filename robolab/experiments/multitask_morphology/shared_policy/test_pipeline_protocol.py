"""CPU checks for protocol isolation and closed-loop selection, not gait validation."""
import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline


def test_curriculum_reaches_target_and_nominal_never_randomizes():
    assert [pipeline.round_strength(i, 3, .25, 1., 'robust') for i in range(4)] == [.25, .5, .75, 1.]
    assert pipeline.round_strength(0, 0, .25, 1., 'robust') == 1.
    assert pipeline.round_strength(2, 3, .25, 1., 'nominal') == 0.
    assert pipeline.round_strength(0, 0, .25, 0., 'nominal') == 0.
    for initial, target in [(1., .5), (-.1, 1.), (.25, float('nan'))]:
        with pytest.raises(ValueError):
            pipeline.round_strength(0, 3, initial, target, 'robust')


def test_selection_pairs_physical_structures_with_different_task_ids(tmp_path):
    keys = [(round(.2+i*.01, 10), .3, 'K', 'A') for i in range(7)]
    plan = {'selected_structure_keys': keys, 'teachers': {}}
    for task in ('stairs', 'backpack', 'run'):
        designs = [dict(id=f'{task}_{i}', thigh_m=key[0], calf_m=key[1], knee_motor=key[2], ankle_motor=key[3])
                   for i, key in enumerate(keys)]
        bank = tmp_path/f'{task}.json'
        bank.write_text(json.dumps({'designs': list(reversed(designs))}))
        plan['teachers'][task] = {'bank': str(bank), 'design_ids': [d['id'] for d in reversed(designs)]}
    jobs, selected = pipeline.selection_jobs(plan, 3, 2)
    assert selected == [keys[0], keys[6], keys[3]]
    assert jobs == [(task, [f'{task}_0', f'{task}_6']) if i % 2 == 0 else (task, [f'{task}_3'])
                    for i, task in enumerate(['stairs', 'stairs', 'backpack', 'backpack', 'run', 'run'])]


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
    assert result['tracking_rmse_mean'] == .1
    missing = row(); del missing['tracking_rmse']
    with pytest.raises(ValueError, match='tracking_rmse'):
        pipeline.evaluation_statistics('run', [missing], 'robust')
    with pytest.raises(ValueError, match='tracking_rmse'):
        pipeline.evaluation_statistics('run', [row(tracking_rmse=float('nan'))], 'robust')


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
    assert pipeline.candidate_rank(retained, teacher) > pipeline.candidate_rank(better_average_but_regression, teacher)
    assert pipeline.candidate_rank(metrics([.8, .8, .8], .08), teacher) > pipeline.candidate_rank(retained, teacher)


def report_for(command):
    profile = command[command.index('--profile')+1]
    strength = float(command[command.index('--randomization-strength')+1])
    actor = command[command.index('--student')+1] if '--student' in command else None
    return dict(rows=[row(i) for i in range(6 if profile == 'robust' else 1)],
        collection_protocol=pipeline.protocol_metadata(profile, strength), task='stairs',
        teacher='teacher', student=pipeline.sha(actor) if actor else None, design_ids=['d'], steps=1000)


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
        pipeline.run_evaluation(**dict(kwargs, criteria={'tracking_rel_tol': .3}))
    actor.write_bytes(b'actor-two')
    with pytest.raises(ValueError, match='changed'):
        pipeline.run_evaluation(**kwargs)


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


def test_diverse_subsets_cover_leg_extremes_and_motor_categories_and_are_nested():
    keys = [(thigh, calf, knee, ankle) for thigh in (.2, .25, .3)
            for calf in (.2, .3, .4) for knee in ('K0', 'K1', 'K2') for ankle in ('A0', 'A1')]
    screen = pipeline.diverse_structure_keys(keys, 6)
    expanded = pipeline.diverse_structure_keys(list(reversed(keys)), 12)
    assert expanded[:6] == screen
    assert {key[2] for key in screen} == {'K0', 'K1', 'K2'}
    assert {key[3] for key in screen} == {'A0', 'A1'}
    assert {min(key[i] for key in screen) for i in (0, 1)} == {.2}
    assert max(key[0] for key in screen) == .3 and max(key[1] for key in screen) == .4
    assert len(pipeline.diverse_structure_keys(keys, len(keys)+1)) == len(keys)


def test_epoch_nomination_uses_schedule_and_keeps_final_epoch_without_mse_filter(tmp_path):
    checkpoints = []
    for epoch in range(12):
        actor = tmp_path/f'epoch_{epoch}.pt'; actor.write_text(str(epoch))
        checkpoints.append(dict(epoch=epoch, student=str(actor), student_sha256=pipeline.sha(actor),
                                balanced_validation_MSE=0. if epoch == 0 else 100.))
    report = {'checkpoints': checkpoints, 'student': checkpoints[0]['student']}
    assert [row['epoch'] for row in pipeline.checkpoint_candidates(report, 5, 12)] == [4, 9, 11]
    Path(checkpoints[9]['student']).write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        pipeline.checkpoint_candidates(report, 5, 12)
    with pytest.raises(ValueError, match='every epoch'):
        pipeline.checkpoint_candidates({'checkpoints': checkpoints[:-1]}, 5, 12)


def test_two_stage_selection_uses_closed_loop_metrics_not_supervised_mse():
    candidates = [dict(epoch=e, student=f'actor_{e}', student_sha256=str(e), balanced_validation_MSE=mse)
                  for e, mse in [(4, .001), (9, 20.), (14, 30.), (19, .0001)]]
    screen_rates = {4: .4, 9: .9, 14: .8, 19: .7}
    expanded_rates = {9: .5, 14: .9}
    calls = []
    def evaluate(candidate, tier):
        epoch = candidate['epoch']; calls.append((epoch, tier))
        rate = (screen_rates if tier == 'screen' else expanded_rates)[epoch]
        return {'task_metrics': metrics([rate]*3), 'evaluations': [dict(epoch=epoch, tier=tier)]}
    result = pipeline.select_round_candidates(candidates, metrics([1.]*3), metrics([1.]*3), evaluate, 2)
    assert calls == [(4, 'screen'), (9, 'screen'), (14, 'screen'), (19, 'screen'), (9, 'selection'), (14, 'selection')]
    assert result['promoted_epochs'] == [9, 14] and result['selected_epoch'] == 14
    assert result['selected_student'] == 'actor_14'
    assert result['comparison_version'] == pipeline.COMPARISON_VERSION
    assert result['teacher_comparison']['version'] == pipeline.COMPARISON_VERSION
    assert all('teacher_comparison' in candidate['screen'] for candidate in result['candidates'])
    assert all('teacher_comparison' in candidate['selection'] for candidate in result['candidates']
               if 'selection' in candidate)


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


def test_selection_budget_accounts_for_both_tiers_seeds_rounds_and_final_tests():
    tasks = ('stairs', 'backpack', 'run')
    plan = {'teachers': {task: {'copies': 8} for task in tasks}}
    screen = [(task, list(range(6))) for task in tasks]
    expanded = [(task, list(range(offset, min(offset+10, 12)))) for task in tasks for offset in (0, 10)]
    full = [(task, list(range(10))) for task in tasks for _ in range(300)]
    budget = pipeline.selection_budget(plan, screen, expanded, full, epochs=20, every_epochs=5,
        rounds=4, promotion_count=2, seeds=[1, 2, 3], selection_profile='robust',
        evaluation_profile='both', preflight_profile='robust')
    assert budget['epoch_candidate_count'] == 4 and budget['promoted_candidates'] == 2
    assert budget['screen_design_count'] == 6 and budget['expanded_design_count'] == 12
    assert budget['teacher_rollout_processes'] == 21
    assert budget['candidate_rollout_processes_per_round'] == 48
    assert budget['total_selection_rollout_processes'] == 213
    assert budget['final_test_rollout_processes'] == 3600


def test_start_passes_training_and_multistage_selection_budgets(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
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
    start.main(['蒸馏', '--root', str(tmp_path), '--dry-run', '--candidate-every-epochs', '2',
        '--screen-designs', '4', '--selection-designs', '8', '--promotion-count', '3',
        '--selection-seeds', '101,102', '--samples-per-design-per-epoch', '64',
        '--validation-rows-per-design-shard', '256', '--replay-cache-shards', '3',
        '--tracking-rel-tolerance', '.2', '--resume-paused'])
    assert len(calls) == 1
    command = calls[0]
    for flag, expected in {'--candidate-every-epochs': '2', '--screen-designs': '4',
        '--selection-designs': '8', '--promotion-count': '3', '--selection-seeds': '101,102',
        '--samples-per-design-per-epoch': '64', '--validation-rows-per-design-shard': '256',
        '--replay-cache-shards': '3',
        '--tracking-rel-tolerance': '0.2'}.items():
        assert command[command.index(flag)+1] == expected
    assert '--max-rows-per-design-shard' not in command
    assert '--tracking-abs-tolerance' not in command
    assert '--standing-speed-tolerance' not in command
    assert '--resume-paused' in command


def test_retired_absolute_motion_tolerance_has_explicit_migration_errors(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--plan', 'unused', '--output', 'unused',
        '--tracking-abs-tolerance', '.1'])
    with pytest.raises(ValueError, match='retired.*tracking-rel-tolerance'):
        pipeline.main()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import start
    with pytest.raises(SystemExit):
        start.main(['蒸馏', '--tracking-abs-tolerance', '.1'])
    assert '--tracking-rel-tolerance' in capsys.readouterr().err


def test_removed_standing_tolerance_is_rejected_by_both_entrypoints(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--plan', 'unused', '--output', 'unused',
        '--standing-speed-tolerance', '.1'])
    with pytest.raises(ValueError, match='standing-speed-tolerance.*retired'):
        pipeline.main()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import start
    with pytest.raises(SystemExit):
        start.main(['蒸馏', '--standing-speed-tolerance', '.1'])
    assert '关闭站立指令' in capsys.readouterr().err


@pytest.mark.parametrize('teacher_rate,teacher_final_pass,student_final_pass', [
    (1., True, True), (0., False, False), (1., True, False), (0., False, True)])
def test_pipeline_dagger_uses_round_closed_loop_winner_and_final_test_cannot_select(
        tmp_path, monkeypatch, teacher_rate, teacher_final_pass, student_final_pass):
    from contextlib import nullcontext
    from types import ModuleType
    tasks = ['stairs', 'backpack', 'run']
    key = (.2, .3, 'K', 'A')
    plan = {'selected_structure_keys': [key], 'teachers': {}, 'schema': {'tasks': tasks}, 'scope': 'all'}
    for task in tasks:
        bank = tmp_path/f'{task}.json'
        bank.write_text(json.dumps({'designs': [dict(id=task, thigh_m=.2, calf_m=.3, knee_motor='K', ankle_motor='A')]}))
        plan['teachers'][task] = {'bank': str(bank), 'design_ids': [task], 'copies': 8, 'checkpoint_sha256': task}
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
            for epoch in range(6):
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
        suites.append((actor, tuple(seeds), name))
        if actor is None:
            rate = teacher_rate
        else:
            epoch = int(Path(actor).stem.split('_')[-1])
            round_index = int(Path(actor).parents[1].name.split('_')[-1])
            rate = ({2: .8, 5: .9}[epoch] if name.startswith('screen/') else
                    {2: 1. if round_index == 0 else .9, 5: .8}[epoch])
        return {'task_metrics': metrics([rate]*3,profile=profile,seeds=seeds), 'evaluations': [{'suite': name}]}
    monkeypatch.setattr(pipeline, 'run_selection_suite', suite)
    final_calls = []
    def final_evaluation(plan, plan_path, root, jobs, actor, profile, strength, seed, suite, *args):
        final_calls.append((actor, profile, seed))
        passed = student_final_pass if actor else teacher_final_pass
        return [dict(task=task, profile=profile, seed=seed, policy='student' if actor else 'teacher',
            evaluation_protocol=comparison_protocol(profile,strength),
            **pipeline.evaluation_statistics(task, [row(fell=not passed)], profile)) for task in tasks]
    monkeypatch.setattr(pipeline, 'run_evaluation', final_evaluation)
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--plan', str(plan_path), '--output', str(output),
        '--epochs', '6', '--dagger-rounds', '1', '--candidate-every-epochs', '3',
        '--screen-designs', '1', '--selection-designs', '1', '--selection-seeds', '101,102'])
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
    preflight = json.loads((output/'teacher_preflight.json').read_text())
    assert preflight['result'] == 'TEACHER_REFERENCE_EVALUATED'
    assert preflight['ability_threshold_applied'] is False and 'gate' not in preflight
    assert preflight['task_metrics']['stairs']['success_rate'] == teacher_rate
    round0 = json.loads((output/'selection_round_0.json').read_text())
    winner = round0['selected_student']
    assert round0['selected_epoch'] == 2 and 'epoch_0.pt' not in winner
    round1_collect = [command for command in children if any('/data/round_1/' in str(value).replace('\\','/') for value in command)]
    assert len(round1_collect) == 3
    assert all(command[command.index('--student')+1] == winner for command in round1_collect)
    second_fit = [command for command in children if any('/bc_round_1/' in str(value).replace('\\','/') for value in command)][0]
    assert second_fit[second_fit.index('--init-student')+1] == winner
    completed = json.loads((output/'completed.json').read_text())
    criteria = completed['evaluation_criteria']
    assert criteria['tracking_rel_tol'] == .25
    assert 'standing_speed_tol_m_s' not in criteria and 'standing_gate' not in criteria
    assert 'standing commands are disabled' in criteria['command_scope']
    assert 'tracking_abs_tol_m_s' not in criteria
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['protocol']['evaluation_criteria'] == criteria
    assert manifest['protocol']['evaluation_version'] == pipeline.GATE_VERSION
    assert manifest['protocol']['comparison_version'] == pipeline.COMPARISON_VERSION
    assert completed['selected_round'] == 0 and completed['student'] == winner
    assert {actor for actor, _, _ in final_calls if actor} == {winner}
    assert {seed for _, _, seed in final_calls}.isdisjoint({101, 102})
    assert completed['result'] == 'DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING'
    assert completed['requires_candidate_finetuning'] and completed['requires_post_finetuning_evaluation']
    for retired in ('robust_basic_gait_passed', 'command_tracking_passed',
                    'nominal_command_tracking_passed', 'robust_command_tracking_passed',
                    'profile_command_tracking_passed'):
        assert retired not in completed
    expected_pass = {'teacher': teacher_final_pass, 'student': student_final_pass}
    assert completed['policy_all_episodes_command_tracking_passed'] == {
        'nominal': expected_pass, 'robust': expected_pass}
    for profile in ('nominal', 'robust'):
        comparison = completed['profile_results'][profile]['teacher_comparison']
        assert comparison['version'] == pipeline.COMPARISON_VERSION
        design = comparison['tasks']['stairs']['designs']['d']
        assert design['success_rate_gap'] == int(student_final_pass)-int(teacher_final_pass)
        assert design['success_rate_shortfall'] == max(int(teacher_final_pass)-int(student_final_pass), 0)
    selection = json.loads((output/'selection.json').read_text())
    assert 'matches_teacher_on_each_selection_task' not in selection
    assert selection['comparison_version'] == pipeline.COMPARISON_VERSION
    assert selection['teacher_comparison'] == round0['teacher_comparison']
    assert all(entry['comparison_version'] == pipeline.COMPARISON_VERSION for entry in completed['rounds'])
    assert all(seeds == (101, 102) for _, seeds, name in suites if name.startswith('selection/'))


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
