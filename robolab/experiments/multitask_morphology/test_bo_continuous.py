"""Independent continuous BO; preserve 100 measured observations."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bo
import bo_continuous as extension

SCHEMA = dict(ranges_m=dict(thigh=[.2, .4], calf=[.25, .45]),
               knees=[f'k{i}' for i in range(5)], ankles=[f'a{i}' for i in range(5)])


def test_continuous_proposal_fits_all_history_and_retains_global_ei(monkeypatch):
    import numpy as np
    from rough_bo import constrained_gp

    schema = dict(SCHEMA, ankles=['DM-J4340P-2EC', 'DM-J4340P-2EC-V1.1-48V', 'DM-J6216P-2EC',
                                  'RS06', 'DM-J8009P-2EC-V1.1'])
    observations = [dict(params=dict(thigh_m=.2+i*.002, calf_m=.3, knee_motor='k0',
                                     ankle_motor=schema['ankles'][4 if i == 0 else i % 5]),
                         feasible=True, objective=.7513+i*.001) for i in range(100)]
    class Surrogate:
        def __init__(self, ranges, knees, ankles, **kwargs):
            assert (ranges, knees, ankles) == (schema['ranges_m'], schema['knees'], schema['ankles'])
            assert kwargs == {'seed': 42, 'objective_key': 'objective'}
            self.performance_count = self.feasibility_count = self.feasible_count = 100
            self.best = .7513
            self.regressor = SimpleNamespace(kernel_='test')
            self.classifier = None
            self.fit_warnings = []
        def fit(self, data):
            assert data is observations
            self.X = self.encode([r['params'] for r in data])
        def decode(self, X):
            return [dict(thigh_m=.2 + x[0]*.2, calf_m=.25 + x[1]*.2,
                         knee_motor=schema['knees'][int(x[2])], ankle_motor=schema['ankles'][int(x[3])])
                    for x in np.asarray(X)]
        def encode(self, params):
            return np.asarray([[(r['thigh_m']-.2)/.2, (r['calf_m']-.25)/.2,
                                schema['knees'].index(r['knee_motor']), schema['ankles'].index(r['ankle_motor'])]
                               for r in params])
        def acquisition(self, X):
            X = np.asarray(X)
            assert np.all(X[:, 3] < 3)
            return 10 - (X[:, 0] - .999)**2 - (X[:, 1] - .397801214)**2 - (X[:,2]-4)**2 - (X[:,3]-2)**2
        def predict(self, X):
            return np.full(len(X), .76), np.full(len(X), .03), np.full(len(X), .95)
    monkeypatch.setattr(constrained_gp, 'ConstrainedSurrogate', Surrogate)
    k, audit = extension.propose(schema, observations, {extension.key(o['params'], schema) for o in observations}, 42)
    assert k[:2] == pytest.approx((.3998, .3295602428), abs=1e-6)
    assert k[2:] == ('k4', 'DM-J6216P-2EC')
    assert audit['global_best_observed_composite_cost'] == .7513
    assert audit['predicted_composite_cost'] == .76
    assert 'predicted_mae_mps' not in audit
    assert audit['motor_pairs'] == 15
    assert audit['fit_performance_rows'] == 100

def test_canonical_rounded_duplicate_is_rejected():
    p = dict(thigh_m=.4, calf_m=.32956024281234, knee_motor='k4', ankle_motor='a4')
    k = extension.key(p, SCHEMA)
    assert k == (.4, .3295602428, 'k4', 'a4')
    with pytest.raises(ValueError, match='already evaluated'):
        extension.checked_key(k, SCHEMA, {k})
    with pytest.raises(ValueError):
        extension.checked_key((.41, .32956, 'k4', 'a4'), SCHEMA, set())
    with pytest.raises(ValueError):
        extension.checked_key((.4, .32956, 'k4', 'unknown'), SCHEMA, set())


def test_history_source_separate(tmp_path):
    historical = tmp_path/'history'
    output = tmp_path/'separate_continuous'
    extension.check_separation(historical, output)
    for invalid in (historical, historical/'new', tmp_path):
        with pytest.raises(ValueError, match='separate'):
            extension.check_separation(historical, invalid)


def test_trial_materialization_exact_key_reuses_only_matching_bank(tmp_path):
    grid = {(.4, .3, 'k0', 'a0'): dict(design_ids={'stairs':'exact', 'backpack':'other1', 'run':'other2'})}
    donor = tmp_path/'donor'
    donor_plan = {'schema':SCHEMA}
    def prepare(*args):
        assert tuple(args[-1]) == (.4, .3295602428, 'k0', 'a0')
        return tmp_path/'new_plan', {'assets':1}, 'generated'
    assert extension.materialize(None, None, tmp_path, donor, donor_plan, tuple(grid)[0], grid, prepare) == (
        donor, donor_plan, 'exact')
    assert extension.materialize(None, None, tmp_path, donor, donor_plan, (.4, .3295602428, 'k0', 'a0'),
                                 grid, prepare) == (tmp_path/'new_plan', {'assets':1}, 'generated')


def test_verify_old_trial_is_read_only_and_checks_native_reports(tmp_path, monkeypatch):
    root = tmp_path/'trial_0050'/'candidate'
    root.mkdir(parents=True)
    benchmark = dict(source_plan_sha256='source', initial_actor_sha256='actor', updates=100)
    report = dict(candidate=dict(design_ids=dict(stairs='id', backpack='b',run='r'),
                                 structure_key=[.4, .3, 'k0', 'a0']))
    file = root/'bo_evaluation.json'
    file.write_text(json.dumps(report))
    before = file.read_bytes()
    calls = []
    def initial(path, evidence, bench):
        assert path == file and evidence['sha256'] == bo.sha(file) and bench == benchmark
        return report
    monkeypatch.setattr(bo, 'initial_report', initial)
    def check(folder, value, bench, conditions, tuning):
        calls.append((folder, value, bench, conditions, tuning))
        return report
    monkeypatch.setattr(bo, 'checked_candidate_measurements', check)
    monkeypatch.setattr(bo, 'evaluate', lambda *args: pytest.fail('history must never evaluate'))
    got = extension.verify_candidate(bo, root, benchmark, dict(test='200Hz'))
    assert got == report
    assert calls == [(root,report,benchmark,dict(test='200Hz'),root/'tuning')]
    assert file.read_bytes() == before


def test_offgrid_native_usd_evidence_from_actual_completed_session(tmp_path):
    tuning = tmp_path/'tuning'
    checkpoint = tuning/'checkpoints/update_000100.json'
    checkpoint.parent.mkdir(parents=True)
    bo.freeze(checkpoint, dict(training_reward={'session_id':'actual_session'}))
    ids = dict(stairs='s', backpack='b', run='r')
    for task, identity in ids.items():
        session = tuning/'attempts/actual_session'/task
        asset = session/'native_assets'/task/identity
        asset.mkdir(parents=True)
        (asset/'.asset_hash').write_text('exact')
        (asset/'config.yaml').write_text('cfg')
        usd = asset/('robot_with_backpack.usd' if task=='backpack' else 'rpo.usd')
        usd.write_text('usd')
        bo.freeze(session/'asset_cache_audit.json', [dict(design=identity, cache=str(asset),
                                                         exact_hash='exact', reused=False)])
    evidence = extension.native_usd_evidence(bo, tmp_path, 100, ids)
    assert evidence['training_session'] == 'actual_session'
    assert set(evidence['tasks']) == set(ids)
    assert evidence['tasks']['backpack']['design_id'] == 'b'
    (tuning/'attempts/actual_session/run/native_assets/run/r/.asset_hash').write_text('wrong')
    with pytest.raises(ValueError, match='Native USD'):
        extension.native_usd_evidence(bo, tmp_path, 100, ids)