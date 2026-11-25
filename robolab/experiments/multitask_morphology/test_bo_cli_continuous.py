"""The canonical BO CLI routes only the V4 four-metric original-RPO scoring."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bo
import bo_continuous


def test_search_cli_defaults_to_100_and_routes_to_continuous(tmp_path, monkeypatch):
    conditions = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    scoring = tmp_path/'scoring.json'
    scoring.write_text(json.dumps({'method': bo.ORIGINAL_RPO_FOUR_METRIC_METHOD, 'conditions_sha256': bo.digest(conditions),
        'benchmark': dict(ppo_algorithm=bo.NATIVE_PPO_ALGORITHM, native_ppo_settings=bo.NATIVE_PPO_SETTINGS)}))
    source, actor = tmp_path/'plan', tmp_path/'actor'
    plan = {'marker': 'frozen plan'}
    output = tmp_path/'new_continuous'
    monkeypatch.setattr(bo, 'shared_input', lambda *args, **kwargs: (source, plan, actor))
    calls = []

    def continuous(b, a, work, donor_source, donor_plan, student, conditions_, **kwargs):
        calls.append((b, a, work))
        assert donor_source == source and donor_plan == plan and student == actor
        assert conditions_ == conditions
        assert callable(kwargs['prepare'])

    monkeypatch.setattr(bo_continuous, 'search', continuous)
    bo.main(['search', '--root', str(tmp_path), '--scoring', str(scoring), '--output', str(output)])
    (b, a, work), = calls
    assert b is bo and work == output.resolve()
    assert a.trials == 100 and a.updates == 100
    assert a.ppo_algorithm == bo.NATIVE_PPO_ALGORITHM
    assert a.history is None and a.history_budget is None and a.allowed_ankles is None


@pytest.mark.parametrize('method', ['post_finetune_panel50','post_finetune_original_rpo'])
def test_search_cli_rejects_retired_percentile_default(tmp_path, method):
    conditions = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    scoring = tmp_path/'scoring.json'
    scoring.write_text(json.dumps({'method': method, 'conditions_sha256': bo.digest(conditions)}))
    with pytest.raises((ValueError, SystemExit)):
        bo.main(['search', '--root', str(tmp_path), '--scoring', str(scoring), '--trials', '100'])


@pytest.mark.parametrize('old_native', [False, True])
def test_search_cli_rejects_old_ppo_before_any_worker(tmp_path, monkeypatch, old_native):
    from finetune_settings import NATIVE_PPO_ALGORITHM_V3, NATIVE_PPO_SETTINGS_V3
    conditions = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    scoring = tmp_path/'scoring.json'
    benchmark = (dict(ppo_algorithm=NATIVE_PPO_ALGORITHM_V3,
                      native_ppo_settings=NATIVE_PPO_SETTINGS_V3) if old_native else {})
    scoring.write_text(json.dumps({'method': bo.ORIGINAL_RPO_FOUR_METRIC_METHOD,
                                  'conditions_sha256': bo.digest(conditions), 'benchmark': benchmark}))
    monkeypatch.setattr(bo, 'shared_input', lambda *args, **kwargs: pytest.fail('old scoring cannot launch new BO'))
    with pytest.raises(ValueError, match='Native PPO BO'):
        bo.main(['search', '--root', str(tmp_path), '--scoring', str(scoring)])


def test_search_pause_prevents_candidate_and_native_measurement_start(tmp_path):
    search_root = tmp_path/'search'
    search_root.mkdir()
    (search_root/'USER_PAUSED').touch()
    candidate = search_root/'trial_0050'/'candidate'
    a = SimpleNamespace(search_root=search_root, resume_paused=False)
    with pytest.raises(KeyboardInterrupt):
        bo.evaluate(a, candidate, None, None, None, None, None)
    assert not candidate.exists()
    with pytest.raises(KeyboardInterrupt):
        bo.evaluate_native(candidate, None, None, None, None, None, 0, 1800, pause_root=search_root)
    assert not candidate.exists()
