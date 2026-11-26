"""Real archived V4 milestone telemetry is the portable BO denominator."""
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bo
from original_reference import check_v4_archive


def protocol(root=None):
    return bo.read((Path(root) if root is not None else bo.ORIGINAL_RPO_ARCHIVE)/'reference_protocol.json')


def conditions():
    return bo.read_conditions(bo.PHYSICS_CONDITIONS)


def test_active_reference_rederives_native_physical_metrics():
    reference, source = bo.checked_original_rpo_reference(bo.ORIGINAL_RPO_ARCHIVE, protocol(), conditions())
    assert source['source_type'] == 'repository_v4_milestone'
    assert source['root'] == 'docs/reference/rpo_original_configuration'
    assert reference['ppo_updates'] == 100 and reference['ppo_algorithm'] == bo.NATIVE_PPO_ALGORITHM
    assert reference['pairs'][0]['feasible'] is True
    scoring = bo.original_rpo_scoring(reference, source, protocol(), {}, conditions(), metric_profile='four')
    result = bo.objective(reference, scoring)
    assert result['mean_objective'] == pytest.approx(1.)
    assert result['observations'][0]['task_objectives'] == pytest.approx(dict.fromkeys(bo.TASKS,1.))
    assert scoring['metrics']['stairs']['cot']['baseline'] == pytest.approx(.9647208003381651)
    assert scoring['metrics']['backpack']['forward_rmse_mps']['baseline'] == pytest.approx(.05971545500186304)


def test_active_reference_rejects_noncanonical_path_and_incompatible_panel(tmp_path):
    with pytest.raises(ValueError,match='canonical'):
        bo.checked_original_rpo_reference(tmp_path,protocol(),conditions())
    wrong = dict(protocol(),initial_actor_sha256='f'*64)
    with pytest.raises(ValueError,match='protocol|identity'):
        bo.checked_original_rpo_reference(bo.ORIGINAL_RPO_ARCHIVE, wrong, conditions())
    wrong = dict(protocol(),updates=1000)
    with pytest.raises(ValueError,match='protocol|identity'):
        bo.checked_original_rpo_reference(bo.ORIGINAL_RPO_ARCHIVE, wrong, conditions())


def test_active_reference_rejects_hash_consistent_trial_tampering(tmp_path):
    archive=tmp_path/'archive'; shutil.copytree(bo.ORIGINAL_RPO_ARCHIVE,archive)
    manifest=bo.read(archive/'manifest.json')
    native_path=archive/'stairs/bo_trials.json'
    native=bo.read(native_path)
    row=native['trials'][0]
    row['observed']['absolute_mechanical_cot'] *= 2.
    native_path.write_text(json.dumps(native),encoding='utf8')
    # Even if the file hash and the parent receipt are changed together, the
    # independently measured raw trace cannot reproduce the fabricated CoT.
    manifest['files_sha256']['stairs/bo_trials.json']=bo.sha(native_path)
    manifest['tasks']['stairs']['native_report_sha256']=bo.sha(native_path)
    report_path=archive/'bo_evaluation.json'
    report=bo.read(report_path)
    report['pairs'][0]['tasks']['stairs']=row
    report['measurements'][0]['sha256']=bo.sha(native_path)
    report_path.write_text(json.dumps(report),encoding='utf8')
    manifest['files_sha256']['bo_evaluation.json']=bo.sha(report_path)
    manifest['raw_source_report_sha256']=bo.sha(report_path)
    (archive/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    with pytest.raises(ValueError,match='recomputation|raw trace'):
        check_v4_archive(archive,protocol(archive),conditions())


def test_active_reference_rejects_hash_inconsistent_trace(tmp_path):
    archive=tmp_path/'archive'; shutil.copytree(bo.ORIGINAL_RPO_ARCHIVE,archive)
    with (archive/'run/bo_samples.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='hash'):
        check_v4_archive(archive,protocol(archive),conditions())


def test_frozen_reference_is_repo_relative_across_worktrees(tmp_path, monkeypatch):
    reference, source = bo.checked_original_rpo_reference(bo.ORIGINAL_RPO_ARCHIVE, protocol(), conditions())
    relocated_repo = tmp_path/'other_checkout'
    relocated_archive = relocated_repo/source['root']
    shutil.copytree(bo.ORIGINAL_RPO_ARCHIVE,relocated_archive)
    panel=tmp_path/'panel'; panel.mkdir()
    bo.freeze(panel/'manifest.json',{})
    bo.freeze(panel/'measurements_completed.json',{})
    entry=dict(report='design_01/bo_evaluation.json',sha256='test-report')
    benchmark=dict(protocol(),reports=[entry])
    scoring=bo.original_rpo_scoring(reference,source,benchmark,bo.measurement_panel_source(panel),conditions())
    out=tmp_path/'scoring'; out.mkdir()
    path=out/'scoring.json'; bo.freeze(path,scoring)
    observations=out/'initial_observations.json'
    records=bo.original_rpo_initial_scores([reference],[entry],scoring,conditions())
    bo.freeze(observations,dict(scoring_sha256=bo.sha(path),reference_objective=1.,candidates=records))
    bo.freeze(out/'completed.json',dict(scoring_sha256=bo.sha(path),initial_observations_sha256=bo.sha(observations)))
    monkeypatch.setattr(bo,'REPO',relocated_repo)
    monkeypatch.setattr(bo,'ORIGINAL_RPO_ARCHIVE',relocated_archive)
    monkeypatch.setattr(bo,'checked_measurement_panel',lambda _:( [reference], [],benchmark,conditions()))
    assert bo.read_search_benchmark(path,scoring,conditions())==(panel,benchmark)


def test_score_cli_refuses_old_profile_or_fabricated_reference_scale(tmp_path,capsys):
    reference,source=bo.checked_original_rpo_reference(bo.ORIGINAL_RPO_ARCHIVE,protocol(),conditions())
    scoring=bo.original_rpo_scoring(reference,source,protocol(),{},conditions())
    path=tmp_path/'scoring.json'; bo.freeze(path,scoring)
    report_path=bo.ORIGINAL_RPO_ARCHIVE/'bo_evaluation.json'
    flags=['score',str(report_path),'--scoring',str(path)]
    bo.main(flags)
    assert json.loads(capsys.readouterr().out)['mean_objective']==pytest.approx(1.)
    scoring['metrics']['stairs']['cot']['baseline']*=2
    path.write_text(json.dumps(scoring),encoding='utf8')
    with pytest.raises(ValueError,match='measured original-RPO'):
        bo.main(flags)
    scoring['method']=bo.ORIGINAL_RPO_SCORING_METHOD
    path.write_text(json.dumps(scoring),encoding='utf8')
    with pytest.raises(ValueError,match='four-metric scoring'):
        bo.main(flags)


def test_scheme_b_refuses_v4_reference_archive():
    new_protocol=dict(protocol(),ppo_algorithm=bo.CURRENT_PPO_ALGORITHM,
                      native_ppo_settings=bo.CURRENT_PPO_SETTINGS)
    with pytest.raises(ValueError,match='Scheme B requires its own measured'):
        bo.checked_original_rpo_reference(bo.ORIGINAL_RPO_ARCHIVE,new_protocol,conditions())
