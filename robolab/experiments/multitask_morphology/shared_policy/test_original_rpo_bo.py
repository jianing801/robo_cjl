"""Regression tests for frozen original-RPO ratios in the 200 Hz BO search."""

from copy import deepcopy
import json
from pathlib import Path
import sys

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
        conditions=conditions, conditions_sha256=conditions_sha256, ppo_updates=200, pairs=pairs,
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
                    source_plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(actor),
                    updates=200, ppo_algorithm=bo.CURRENT_PPO_ALGORITHM, native_ppo_settings=bo.CURRENT_PPO_SETTINGS)
    if protocol.get('ppo_algorithm') is not None:
        reference_report['ppo_algorithm'] = protocol['ppo_algorithm']
        # Synthetic fixture authority, not the repository's byte-proven archive.
        reference_path.write_text(json.dumps(reference_report), encoding='utf8')
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
    bo.freeze_original_rpo_scoring(panel, reference, output, metric_profile='three')
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
    assert scoring['metrics']['stairs']['cot']['baseline'] == pytest.approx(reference_report['pairs'][0]['tasks']['stairs']['observed']['absolute_mechanical_cot'])
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


FOUR_FIELDS = ('cot', 'forward_rmse_mps', 'roll_pitch_rms_rad_s', 'knee_ankle_max_rated_ratio')
FOUR_WEIGHTS = dict(stairs=(.25, .25, .40, .10), backpack=(.40, .25, .20, .15),
                    run=(.25, .40, .15, .20))


def test_four_metric_profile_freezes_all_observed_ratios_without_touching_old_data(original_rpo_panel):
    panel, reference, output, conditions, benchmark, reports, reference_report = original_rpo_panel
    # Preserve the existing three-metric scoring and every native panel file.
    bo.freeze_original_rpo_scoring(panel, reference, output, metric_profile='three')
    old = {path: bo.sha(path) for path in (*panel.rglob('*.json'), *output.rglob('*.json'))}
    fresh = output.with_name('four_metrics')
    bo.freeze_original_rpo_scoring(panel, reference, fresh, metric_profile='four')
    scoring_path = fresh/'scoring.json'
    scoring = bo.read(scoring_path)
    assert scoring['kind'] == 'rpo_three_task_bo_scoring_v2'
    assert scoring['method'] == 'post_finetune_original_rpo_four_metrics'
    assert scoring['cot_basis'] == bo.ABSOLUTE_COT_BASIS
    assert scoring['task_weights'] == dict(stairs=.4, backpack=.3, run=.3)
    source = reference_report['pairs'][0]['tasks']
    trial = reports[0]['pairs'][0]['tasks']
    expected_costs = {}
    for task in bo.TASKS:
        assert tuple(scoring['metrics'][task]) == FOUR_FIELDS
        assert [scoring['metrics'][task][name]['weight'] for name in FOUR_FIELDS] == list(FOUR_WEIGHTS[task])
        costs = []
        for name, weight in zip(FOUR_FIELDS, FOUR_WEIGHTS[task], strict=True):
            raw_name = 'absolute_mechanical_cot' if name == 'cot' else name
            baseline = source[task]['observed'][raw_name]
            assert scoring['metrics'][task][name] == dict(baseline=baseline, weight=weight)
            costs.append(weight * trial[task]['observed'][raw_name] / baseline)
        expected_costs[task] = sum(costs)
    baseline = bo.objective(reference_report, scoring)
    assert baseline['mean_objective'] == 1.
    assert baseline['observations'][0]['task_objectives'] == dict.fromkeys(bo.TASKS, 1.)
    scored = bo.objective(reports[0], scoring)
    assert scored['observations'][0]['task_objectives'] == pytest.approx(expected_costs)
    assert scored['mean_objective'] == pytest.approx(sum(scoring['task_weights'][task] * cost
                                                          for task, cost in expected_costs.items()))
    assert bo.objective(reports[-1], scoring)['observations'] == [
        dict(seed=20261118, replica=0, feasible=False, objective=None)]
    assert bo.read_search_benchmark(scoring_path, scoring, conditions)[0] == panel
    assert len(bo.read(fresh/'initial_observations.json')['candidates']) == 50
    assert bo.read_search_benchmark(output/'scoring.json', bo.read(output/'scoring.json'), conditions)[0] == panel
    assert all(bo.sha(path) == digest for path, digest in old.items())


@pytest.mark.parametrize('missing', ['knee_ankle_max_rated_ratio', 'roll_pitch_rms_rad_s', 'forward_rmse_mps',
                                    'absolute_mechanical_cot'])
def test_four_metric_profile_rejects_missing_observed_inputs(original_rpo_panel, missing):
    panel, reference, output, _, _, reports, _ = original_rpo_panel
    bo.freeze_original_rpo_scoring(panel, reference, output, metric_profile='four')
    scoring = bo.read(output/'scoring.json')
    report = deepcopy(reports[0])
    report['pairs'][0]['tasks']['stairs']['observed'].pop(missing)
    with pytest.raises(ValueError, match='Missing or invalid'):
        bo.objective(report, scoring)


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), '0.1', -0.1, True])
def test_four_metric_profile_rejects_invalid_observed_inputs(original_rpo_panel, bad):
    panel, reference, output, _, _, reports, _ = original_rpo_panel
    bo.freeze_original_rpo_scoring(panel, reference, output, metric_profile='four')
    report = deepcopy(reports[0])
    report['pairs'][0]['tasks']['backpack']['observed']['roll_pitch_rms_rad_s'] = bad
    with pytest.raises(ValueError, match='Missing or invalid'):
        bo.objective(report, bo.read(output/'scoring.json'))


def test_four_metric_profile_rejects_rate_weights_scale_or_kind_changes(original_rpo_panel):
    panel, reference, output, _, _, reports, _ = original_rpo_panel
    bo.freeze_original_rpo_scoring(panel, reference, output, metric_profile='four')
    scoring = bo.read(output/'scoring.json')
    wrong_rate = deepcopy(reports[0])
    wrong_rate['pairs'][0]['tasks']['run']['dt_s'] = .02
    with pytest.raises(ValueError, match='200 Hz'):
        bo.objective(wrong_rate, scoring)
    changed_weights = deepcopy(scoring)
    changed_weights['metrics']['stairs']['cot']['weight'] = .26
    changed_scale = deepcopy(scoring)
    changed_scale['metrics']['stairs']['cot']['scale'] = 1.
    changed_kind = dict(scoring, kind=bo.SCORING_VERSION)
    for changed in (changed_weights, changed_scale, changed_kind):
        with pytest.raises(ValueError):
            bo.objective(reports[0], changed)


@pytest.mark.parametrize('profile', [None, 'four'])
def test_freeze_scoring_cli_metric_profile_default_is_four(original_rpo_panel, profile):
    panel, reference, output, *_ = original_rpo_panel
    flags = ['freeze-scoring', '--measurements-root', str(panel), '--original-rpo-reference', str(reference),
             '--output', str(output)]
    if profile is not None:
        flags += ['--metric-profile', profile]
    bo.main(flags)
    scoring = bo.read(output/'scoring.json')
    assert scoring['method'] == 'post_finetune_original_rpo_four_metrics'


def test_freeze_scoring_cli_retires_three_metric_formal_profile(original_rpo_panel):
    panel, reference, output, *_ = original_rpo_panel
    with pytest.raises(SystemExit):
        bo.main(['freeze-scoring','--measurements-root',str(panel),'--output',str(output),'--metric-profile','three'])


def test_freeze_scoring_cli_requires_new_measured_reference(original_rpo_panel, monkeypatch):
    panel, _, output, _, _, _, _ = original_rpo_panel
    received = []
    monkeypatch.setattr(bo, 'freeze_original_rpo_scoring', lambda panel,root,output,**kwargs: received.append((root,kwargs)))
    with pytest.raises(SystemExit):
        bo.main(['freeze-scoring','--measurements-root',str(panel),'--output',str(output)])
    assert not received


def test_four_metric_custom_weights_are_frozen_and_independent_of_current_default(original_rpo_panel, monkeypatch):
    panel, reference, output, conditions, _, _, reference_report = original_rpo_panel
    config = output.with_name('custom_weights.json')
    original_weights = {task: dict(zip(FOUR_FIELDS, weights, strict=True))
                        for task, weights in FOUR_WEIGHTS.items()}
    custom = deepcopy(original_weights)
    custom['stairs'].update(cot=.4, forward_rmse_mps=.1, roll_pitch_rms_rad_s=.3, knee_ankle_max_rated_ratio=.2)
    bo.freeze(config, custom)
    bo.main(['freeze-scoring', '--measurements-root', str(panel), '--original-rpo-reference', str(reference),
             '--output', str(output), '--metric-profile', 'four', '--metric-weights', str(config)])
    scoring_path = output/'scoring.json'
    scoring = bo.read(scoring_path)
    assert {task: {metric: row['weight'] for metric, row in values.items()}
            for task, values in scoring['metrics'].items()} == custom
    assert bo.objective(reference_report, scoring)['mean_objective'] == pytest.approx(1.)
    # A later edit to the provisional default file cannot alter an already
    # frozen experiment, even when the new default uses other valid weights.
    changed_default = output.with_name('changed_default.json')
    bo.freeze(changed_default, original_weights)
    monkeypatch.setattr(bo, 'FOUR_METRIC_WEIGHTS_PATH', changed_default)
    before = {p: bo.sha(p) for p in output.rglob('*.json')}
    assert bo.read_search_benchmark(scoring_path, scoring, conditions)[0] == panel
    assert all(bo.sha(p) == digest for p, digest in before.items())


@pytest.mark.parametrize('bad', ['missing_task', 'missing_metric', 'extra_metric', 'negative', 'nonfinite', 'bool', 'sum'])
def test_four_metric_weight_configuration_has_exact_fields_and_valid_weights(bad):
    from shared_bo_metrics import validate_four_metric_weights
    config = {task: dict(zip(FOUR_FIELDS, weights, strict=True)) for task, weights in FOUR_WEIGHTS.items()}
    if bad == 'missing_task':
        config.pop('stairs')
    elif bad == 'missing_metric':
        config['run'].pop('cot')
    elif bad == 'extra_metric':
        config['stairs']['unknown'] = 0.
    elif bad == 'negative':
        config['backpack'].update(cot=-.1, forward_rmse_mps=.7)
    elif bad == 'nonfinite':
        config['stairs']['cot'] = float('nan')
    elif bad == 'bool':
        config['stairs']['cot'] = True
    else:
        config['stairs']['cot'] = .26
    with pytest.raises(ValueError):
        validate_four_metric_weights(config)


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

def test_five_metric_price_profile_freezes_and_reuses_same_measurements(original_rpo_panel):
    panel, reference, output, conditions, _, reports, baseline = original_rpo_panel
    before = {p: bo.sha(p) for p in panel.rglob('*.json')}
    four = output.with_name('four')
    bo.freeze_original_rpo_scoring(panel, reference, four)
    old = bo.sha(four/'scoring.json')
    bo.main(['freeze-scoring', '--measurements-root', str(panel),
             '--original-rpo-reference', str(reference), '--output', str(output),
             '--metric-profile', 'five'])
    scoring = bo.read(output/'scoring.json')
    assert scoring['method'] == 'post_finetune_original_rpo_five_metrics'
    assert scoring['task_weights'] == {'stairs': .4, 'backpack': .3, 'run': .3}
    expected = {
        'stairs': (.25, .25, .35, .10, .05),
        'backpack': (.35, .25, .20, .15, .05),
        'run': (.25, .35, .15, .20, .05),
    }
    fields = ('cot','forward_rmse_mps','roll_pitch_rms_rad_s',
              'knee_ankle_max_rated_ratio','motor_total_price_cny')
    for task, weights in expected.items():
        assert {name: row['weight'] for name,row in scoring['metrics'][task].items()} == dict(zip(fields,weights))
        assert scoring['metrics'][task]['motor_total_price_cny']['baseline'] == 5900
    orig = bo.objective(baseline, scoring)
    assert orig['mean_objective'] == pytest.approx(1.)
    assert orig['observations'][0]['task_objectives'] == pytest.approx(dict.fromkeys(bo.TASKS,1.))
    expensive = deepcopy(baseline)
    expensive['candidate']['structure_key'] = [.4,.45,'DM-J10010-2EC','DM-J8009P-2EC-V1.1']
    score = bo.objective(expensive,scoring)
    assert score['mean_objective'] == pytest.approx(.95 + .05 * 10800/5900)
    assert score['observations'][0]['motor_total_price_cny'] == 10800
    assert bo.objective(reports[-1],scoring)['mean_objective'] is None
    assert bo.read_search_benchmark(output/'scoring.json',scoring,conditions)[0] == panel
    assert bo.sha(four/'scoring.json') == old
    assert all(bo.sha(p)==h for p,h in before.items())
    invalid = deepcopy(expensive)
    invalid['candidate']['structure_key'][3] = 'unknown'
    with pytest.raises(ValueError,match='motor'):
        bo.objective(invalid,scoring)
    changed = deepcopy(scoring)
    changed['metrics']['stairs']['motor_total_price_cny']['baseline'] = 6000
    with pytest.raises(ValueError,match='price'):
        bo.objective(expensive,changed)

def test_five_metric_prices_are_frozen_and_continuous_protocol_accepts_them(original_rpo_panel, monkeypatch):
    panel, reference, output, conditions, _, reports, _ = original_rpo_panel
    bo.freeze_original_rpo_scoring(panel, reference, output, metric_profile='five')
    path = output/'scoring.json'
    scoring = bo.read(path)
    from bo_continuous import scoring_protocol
    scoring_protocol(bo, scoring, conditions)
    prices = deepcopy(scoring['motor_prices_cny'])
    prices['knee']['DM-J10010L-2EC'] += 100
    next_catalog = output.with_name('future_prices.json')
    bo.freeze(next_catalog, prices)
    monkeypatch.setattr(bo, 'MOTOR_PRICES_PATH', next_catalog)
    assert bo.read_search_benchmark(path, scoring, conditions)[0] == panel
    invalid = deepcopy(scoring)
    invalid['motor_prices_cny']['ankle'].pop('RS06')
    with pytest.raises(ValueError,match='motor price'):
        bo.objective(reports[0],invalid)
    changed = deepcopy(scoring)
    changed['metrics']['run']['motor_total_price_cny']['weight'] = .1
    with pytest.raises(ValueError,match='weights'):
        bo.objective(reports[0],changed)
