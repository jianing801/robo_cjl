"""Regression tests for frozen original-RPO ratios in the 200 Hz BO search."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bo


def test_reference_protocol_accepts_archived_trailing_newline(tmp_path, monkeypatch):
    archive = tmp_path/'archive'
    root = tmp_path/'run'
    archive.mkdir()
    root.mkdir()
    conditions = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    protocol = dict(source_plan_sha256='original', initial_actor_sha256='round5',
                    updates=100, conditions_sha256=bo.digest(conditions))
    live_protocol = root/'reference_protocol.json'
    bo.freeze(live_protocol, protocol)
    archived_protocol = archive/'reference_protocol.json'
    archived_protocol.write_bytes(live_protocol.read_bytes() + b'\n')
    bo.freeze(root/'reference_plan.json', {})
    key = [.25, .30, 'DM-J10010L-2EC', 'DM-J4340P-2EC']
    bo.freeze(archive/'manifest.json', dict(
        design_key=key, final_actor_sha256='final', raw_source_report_sha256='raw-report',
        files_sha256={'reference_protocol.json': bo.sha(archived_protocol)},
    ))
    report_path = root/'candidate/bo_evaluation.json'
    report_path.parent.mkdir()
    report = dict(candidate=dict(structure_key=key), student_sha256='final')
    bo.freeze(report_path, report)
    monkeypatch.setattr(bo, 'ORIGINAL_RPO_ARCHIVE', archive)
    monkeypatch.setattr(bo, 'initial_report', lambda *args: report)
    monkeypatch.setattr(bo, 'checked_candidate_measurements', lambda *args: report)

    checked, source = bo.checked_original_rpo_reference(root, protocol, conditions)
    assert checked == report and source['protocol_sha256'] == bo.sha(live_protocol)
    archived_protocol.write_bytes(b'{}')
    with pytest.raises(ValueError, match='archived PPO'):
        bo.checked_original_rpo_reference(root, protocol, conditions)


@pytest.fixture
def original_rpo_panel(tmp_path, monkeypatch):
    conditions = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    conditions_sha256 = bo.digest(conditions)
    archive = bo.REPO/'docs/reference/rpo_original_configuration'
    pairs = bo.paired_trials((bo.read(archive/task/'bo_trials.json') for task in bo.TASKS), conditions)
    summary = bo.read(archive/'reference_result.json')
    reference_report = dict(
        kind=bo.REPORT_VERSION, domain='simulation', candidate=summary['candidate'],
        conditions=conditions, conditions_sha256=conditions_sha256, ppo_updates=100, pairs=pairs,
    )
    reference = tmp_path/'reference'
    (reference/'candidate').mkdir(parents=True)
    reference_path = reference/'candidate/bo_evaluation.json'
    bo.freeze(reference_path, reference_report)
    source = tmp_path/'plan.json'
    bo.freeze(source, {'kind': 'test_source_plan'})
    actor = tmp_path/'selected.pt'
    actor.write_bytes(b'test selected actor')
    protocol = dict(bo.read(archive/'reference_protocol.json'),
                    source_plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(actor))
    protocol_path = reference/'reference_protocol.json'
    bo.freeze(protocol_path, protocol)
    provenance = dict(root=str(reference), report_sha256=bo.sha(reference_path),
                      protocol_sha256=bo.sha(protocol_path))

    panel = tmp_path/'original_panel'
    panel.mkdir()
    legacy_path = panel/'scoring.json'
    legacy_path.write_text('{"method": "post_finetune_panel50"}', encoding='utf-8')
    old_receipt = panel/'completed.json'
    bo.freeze(old_receipt, dict(scoring_sha256=bo.sha(legacy_path)))
    keys = bo.calibration_keys()
    evidence = []
    reports = []
    for index, key in enumerate(keys, 1):
        report = deepcopy(reference_report)
        report['candidate'] = dict(structure_key=list(key), design_ids=dict.fromkeys(bo.TASKS, f'tr_{index:012d}'))
        (pair,) = report['pairs']
        for task, trial in pair['tasks'].items():
            factor = {'stairs': .65, 'backpack': .85, 'run': .95}[task]
            for metric in bo.METRICS[task]:
                trial['metrics'][metric] *= factor
            trial['observed']['absolute_mechanical_cot'] *= factor
        if index == 50:
            pair['feasible'] = False
            pair['tasks']['run']['status'] = 'failure'
            pair['tasks']['run']['metrics'] = None
        path = panel/f'design_{index:02d}'/'bo_evaluation.json'
        path.parent.mkdir()
        candidate_manifest = path.with_name('manifest.json')
        bo.freeze(candidate_manifest, dict(protocol, candidate=report['candidate']))
        report['evaluation_manifest_sha256'] = bo.sha(candidate_manifest)
        bo.freeze(path, report)
        evidence.append(dict(report=path.relative_to(panel).as_posix(), sha256=bo.sha(path)))
        reports.append(report)
    manifest_path = panel/'manifest.json'
    bo.freeze(manifest_path, dict(protocol, structure_keys=list(map(list, keys))))
    evidence_path = panel/'measurements_completed.json'
    bo.freeze(evidence_path, dict(conditions_sha256=conditions_sha256,
                                  manifest_sha256=bo.sha(manifest_path), reports=evidence))
    benchmark = dict(protocol, reports=evidence, selection_velocity_frame=bo.EVALUATION_VELOCITY_FRAME,
                     ppo_schedule=bo.PPO_SCHEDULE, init_at_random_ep_len=True, envs_per_task=bo.MIXED_BO_LAYOUT)
    monkeypatch.setattr(bo, 'checked_measurement_panel',
                        lambda root: (reports, keys, benchmark, conditions))
    monkeypatch.setattr(bo, 'checked_original_rpo_reference',
                        lambda root, benchmark, conditions: (reference_report, provenance), raising=False)
    return panel, reference, tmp_path/'original_rpo_scoring', conditions, benchmark, reports, reference_report


def test_freezes_reference_ratios_from_verified_200_hz_measurements(original_rpo_panel):
    panel, reference, output, conditions, benchmark, reports, reference_report = original_rpo_panel
    old = {path: bo.sha(path) for path in (panel/'scoring.json', panel/'completed.json',
                                         panel/'manifest.json', panel/'measurements_completed.json')}
    bo.freeze_original_rpo_scoring(panel, reference, output)
    scoring_path = output/'scoring.json'
    scoring = bo.read(scoring_path)
    assert scoring['method'] == bo.ORIGINAL_RPO_SCORING_METHOD
    assert scoring['cot_basis'] == bo.ABSOLUTE_COT_BASIS
    assert scoring['task_weights'] == dict(stairs=.4, backpack=.3, run=.3)
    assert [scoring['metrics']['stairs'][metric]['weight'] for metric in bo.METRICS['stairs']] == [1/3]*3
    for task in ('backpack', 'run'):
        assert [scoring['metrics'][task][metric]['weight'] for metric in bo.METRICS[task]] == [.4, .4, .2]
    assert all(spec.get('scale') is None and spec['baseline'] > 0
               for task in scoring['metrics'].values() for spec in task.values())
    baseline = bo.objective(reference_report, scoring)
    assert baseline['mean_objective'] == 1.
    assert baseline['observations'][0]['task_objectives'] == dict.fromkeys(bo.TASKS, 1.)
    assert scoring['metrics']['stairs']['cot']['baseline'] == pytest.approx(1.4626239251672708)
    assert bo.objective(reports[0], scoring)['mean_objective'] == pytest.approx(.4*.65 + .3*.85 + .3*.95)
    failure = bo.objective(reports[-1], scoring)
    assert failure['mean_objective'] is None
    assert failure['observations'] == [dict(seed=20261118, replica=0, feasible=False, objective=None)]
    assert bo.read_search_benchmark(scoring_path, scoring, conditions)[0] == panel
    assert bo.read_search_benchmark(scoring_path, scoring, conditions)[1]['reports'] == benchmark['reports']
    assert (output/'completed.json').is_file()
    saved = bo.read(output/'initial_observations.json')
    assert saved['scoring_sha256'] == bo.sha(scoring_path)
    assert saved['reference_objective'] == 1.
    assert len(saved['candidates']) == 50
    assert saved['candidates'][0]['mean_objective'] == pytest.approx(.4*.65 + .3*.85 + .3*.95)
    assert saved['candidates'][-1]['mean_objective'] is None
    assert all(bo.sha(path) == digest for path, digest in old.items())


@pytest.mark.parametrize('artifact', ['panel_manifest', 'panel_receipt', 'scoring', 'observations'])
def test_search_refuses_tampered_score_or_panel_receipt(original_rpo_panel, artifact):
    panel, reference, output, conditions, *_ = original_rpo_panel
    bo.freeze_original_rpo_scoring(panel, reference, output)
    scoring_path = output/'scoring.json'
    paths = dict(panel_manifest=panel/'manifest.json', panel_receipt=panel/'measurements_completed.json',
                 scoring=scoring_path, observations=output/'initial_observations.json')
    before = paths[artifact].read_bytes()
    paths[artifact].write_bytes(before + b' ')
    with pytest.raises(ValueError):
        bo.read_search_benchmark(scoring_path, bo.read(scoring_path), conditions)


def test_search_refuses_a_different_evaluation_protocol(original_rpo_panel):
    panel, reference, output, conditions, *_ = original_rpo_panel
    bo.freeze_original_rpo_scoring(panel, reference, output)
    scoring_path = output/'scoring.json'
    changed = dict(conditions, seeds=[42])
    with pytest.raises(ValueError):
        bo.read_search_benchmark(scoring_path, bo.read(scoring_path), changed)


def test_search_starts_at_51_with_original_rpo_scores(original_rpo_panel, monkeypatch, capsys):
    panel, reference, output, conditions, _, reports, _ = original_rpo_panel
    bo.freeze_original_rpo_scoring(panel, reference, output)
    source = panel.parent/'plan.json'
    actor = panel.parent/'selected.pt'
    bank = panel.parent/'bank.json'
    candidates = [dict(key=r['candidate']['structure_key'], design_ids=r['candidate']['design_ids'])
                  for r in reports]
    next_candidate = dict(key=[.3333333333, .3777777778, 'RS04', 'DM-J8009P-2EC-V1.1'],
                          design_ids=dict.fromkeys(bo.TASKS, 'tr_new'))
    rows = [*candidates, next_candidate]
    bo.freeze(bank, dict(structures=rows))
    plan = dict(shared_bank=str(bank), selected_structure_keys=[row['key'] for row in rows], schema={})
    evaluated = []

    def propose(pool, observations, seen, schema, seed):
        assert [r for r in pool if r['design_ids']['stairs'] not in seen] == [next_candidate]
        assert len(observations) == 50
        assert observations[0]['objective'] == pytest.approx(.8)
        assert observations[-1]['feasible'] is False and observations[-1]['objective'] is None
        return next_candidate, dict(method='test')

    def evaluate(args, work, source, plan, actor, design_id, conditions):
        evaluated.append(design_id)
        return dict(reports[0], candidate=dict(structure_key=next_candidate['key'],
                                               design_ids=next_candidate['design_ids']))

    monkeypatch.setattr(bo, 'propose', propose)
    monkeypatch.setattr(bo, 'evaluate', evaluate)
    work = panel.parent/'new_search'
    work.mkdir()
    args = SimpleNamespace(scoring=output/'scoring.json', updates=100, trials=51, seed=42, resume_paused=False)
    bo.search(args, work, source, plan, actor, conditions)
    assert evaluated == ['tr_new']
    assert 'BO_TRIAL 51 / 51 tr_new' in capsys.readouterr().out
    assert bo.read(work/'manifest.json')['scoring_sha256'] == bo.sha(args.scoring)
    assert bo.read(work/'budget_0051.json')['trials'] == 51
