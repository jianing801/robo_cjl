"""The extension is a new search restriction, not a change to BO evidence."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bo
import bo_compact_ankles as extension

ALLOWED = sorted(extension.ALLOWED_ANKLES)
EXCLUDED = ['RS06', 'DM-J8009P-2EC-V1.1']


def test_proposal_retains_full_fit_and_uses_global_incumbent(monkeypatch):
    from rough_bo import constrained_gp

    captured = {}

    class Surrogate:
        def __init__(self, ranges, knees, ankles, **kwargs):
            assert ankles == ALLOWED + EXCLUDED
            self.ankles = ankles
            self.regressor = object()

        def encode(self, params):
            return np.asarray([[v['thigh_m'], v['calf_m'], 0, self.ankles.index(v['ankle_motor'])]
                               for v in params])

        def fit(self, observations):
            captured['fit'] = observations
            self.best = min(o['objective'] for o in observations if o['feasible'])
            self.performance_count = self.feasibility_count = len(observations)

        def acquisition(self, X):
            captured['incumbent'] = self.best
            assert len(X) == 3
            assert all(int(v[3]) < 3 for v in X)
            return np.array([1., 2., 3.])

        def predict(self, X):
            return np.zeros(len(X)), np.ones(len(X)), np.ones(len(X))

    monkeypatch.setattr(constrained_gp, 'ConstrainedSurrogate', Surrogate)
    pool = [dict(key=[.3, .3, 'knee', ankle], design_ids={'stairs': f'old-{i}'})
            for i, ankle in enumerate(ALLOWED + EXCLUDED)]
    pool += [dict(key=[.4, .3, 'knee', ankle], design_ids={'stairs': f'new-{i}'})
             for i, ankle in enumerate(ALLOWED + EXCLUDED)]
    observations = [dict(params=bo.parameters(row), feasible=True, objective=x)
                    for row, x in zip(pool[:5], [.8, .9, .79, .75, .70])]
    row, audit = extension.propose(bo, pool, observations, {f'old-{i}' for i in range(5)},
                                   dict(ranges_m={}, knees=['knee'], ankles=ALLOWED + EXCLUDED), 42)
    assert row['design_ids']['stairs'] == 'new-2'
    assert captured['fit'] is observations
    assert captured['incumbent'] == pytest.approx(.70)
    assert audit['global_incumbent'] == pytest.approx(.70)
    assert audit['pool_size'] == 3


@pytest.fixture
def completed(tmp_path):
    work = tmp_path
    conditions = dict(seeds=[313], reset_mode='fixed')
    entries = []
    for i in range(100):
        trial = work/f'trial_{i:04d}'
        candidate = trial/'candidate'
        candidate.mkdir(parents=True)
        report = candidate/'bo_evaluation.json'
        key = [.3, .3, 'knee', (ALLOWED + EXCLUDED)[i % 5]]
        proposal = dict(structure_key=key, design_id=f'tr_{i}', acquisition={'test': 1})
        row = dict(design_id=f'tr_{i}', structure_key=key, evaluation=str(report), valid_pairs=1,
                   joint_successes=1, complete_objectives=1, mean_objective=.8)
        for file, value in [(trial/'proposal.json', proposal), (trial/'observation.json', {**proposal, **row}),
                            (report, {}), (candidate/'manifest.json', {}),
                            (candidate/'tuning/manifest.json', {}), (candidate/'tuning/training_completed.json', {})]:
            file.parent.mkdir(exist_ok=True, parents=True)
            file.write_text(json.dumps(value))
        for task in bo.TASKS:
            receipt = candidate/'measurements'/f'313_{task}'/'accepted.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text('{}')
        entries.append(row)
    (work/'budget_0100.json').write_text(json.dumps(dict(trials=100, candidates=entries)))
    (work/'manifest.json').write_text(json.dumps(dict(
        coordinator_sha256=bo.sha(Path(bo.__file__)),
        surrogate_sha256=bo.sha(bo.REPO/'rough_bo/constrained_gp.py'))))
    return work, conditions


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_missing_historical_receipt_refuses_before_any_replay(completed, monkeypatch):
    work, conditions = completed
    (work/'trial_0050/candidate/tuning/training_completed.json').unlink()
    before = snapshot(work)
    monkeypatch.setattr(bo, 'search', lambda *args: pytest.fail("must not call native search"))
    with pytest.raises(ValueError, match='Historical receipt'):
        extension.search(bo, SimpleNamespace(trials=110), work, None, None, None, conditions)
    assert snapshot(work) == before


def test_resume_refuses_disallowed_future_proposal(completed, monkeypatch):
    work, conditions = completed
    path = work/'trial_0100/proposal.json'
    path.parent.mkdir()
    path.write_text(json.dumps(dict(design_id='new', structure_key=[.4, .3, 'knee', 'RS06'])))
    before = snapshot(work)
    monkeypatch.setattr(bo, 'search', lambda *args: pytest.fail("must not call native search"))
    with pytest.raises(ValueError, match='Frozen extension proposal'):
        extension.search(bo, SimpleNamespace(trials=110), work, None, None, None, conditions)
    assert snapshot(work) == before


def test_read_only_replay_and_new_proposals(completed, monkeypatch):
    work, conditions = completed
    request = work/'trial_0100/proposal.json'
    request.parent.mkdir()
    request.write_text(json.dumps(dict(design_id='new', structure_key=[.4, .3, 'knee', ALLOWED[0]])))
    before = snapshot(work)
    replayed = []

    def evaluate(a, where, *args):
        replayed.append(where)
        return {'report': 1}

    def native_search(a, where, source, plan, actor, conditions):
        assert a.trials == 110
        assert bo.evaluate(a, where/'trial_0050/candidate', source, plan, actor, 'tr_50', conditions)
        assert bo.evaluate(a, where/'trial_0100/candidate', source, plan, actor, 'new', conditions)

    monkeypatch.setattr(bo, 'evaluate', evaluate)
    monkeypatch.setattr(bo, 'search', native_search)
    extension.search(bo, SimpleNamespace(trials=110), work, None, None, None, conditions)
    assert replayed == [work/'trial_0050/candidate', work/'trial_0100/candidate']
    after = snapshot(work)
    assert all(after[k] == v for k, v in before.items())
    assert set(after) - set(before) == {'extension_0100_0110.json'}
    # An identical resume changes no evidence.
    extension.search(bo, SimpleNamespace(trials=110), work, None, None, None, conditions)
    assert snapshot(work) == after


def test_historical_replay_forbids_worker_even_if_receipt_check_races(completed, monkeypatch):
    work, conditions = completed
    monkeypatch.setattr(bo, 'evaluate', lambda *_: bo.run_worker())
    monkeypatch.setattr(bo, 'run_worker', lambda *args: pytest.fail("must not run historic worker"))

    def native_search(a, where, source, plan, actor, cond):
        bo.evaluate(a, where/'trial_0050/candidate', source, plan, actor, 'tr_50', cond)

    monkeypatch.setattr(bo, 'search', native_search)
    before = snapshot(work)
    with pytest.raises(RuntimeError, match='Historical BO replay'):
        extension.search(bo, SimpleNamespace(trials=110), work, None, None, None, conditions)
    after = snapshot(work)
    assert all(after[k] == v for k, v in before.items())


def test_refuses_wrong_frozen_coordinator(completed, monkeypatch):
    work, conditions = completed
    manifest = bo.read(work/'manifest.json')
    manifest['coordinator_sha256'] = 'another_source'
    (work/'manifest.json').write_text(json.dumps(manifest))
    before = snapshot(work)
    monkeypatch.setattr(bo, 'search', lambda *args: pytest.fail("must not call native search"))
    with pytest.raises(ValueError, match='frozen code'):
        extension.search(bo, SimpleNamespace(trials=110), work, None, None, None, conditions)
    assert snapshot(work) == before


def test_old_evidence_hashes_frozen_across_resume(completed, monkeypatch):
    work, conditions = completed
    monkeypatch.setattr(bo, 'search', lambda *args: None)
    args = SimpleNamespace(trials=110)
    extension.search(bo, args, work, None, None, None, conditions)
    path = work/'trial_0061/candidate/bo_evaluation.json'
    path.write_text('{"tampered": true}')
    before = snapshot(work)
    with pytest.raises(ValueError, match='Frozen BO identity differs'):
        extension.search(bo, args, work, None, None, None, conditions)
    assert snapshot(work) == before