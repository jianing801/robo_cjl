"""Unified continuous BO from 50 initial or more committed observations."""
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


@pytest.mark.parametrize('subset, expected_pairs',[(None, 25), (['DM-J4340P-2EC','DM-J4340P-2EC-V1.1-48V','DM-J6216P-2EC'], 15)])
def test_continuous_proposal_fits_all_history_and_retains_global_ei(monkeypatch, subset, expected_pairs):
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
            assert np.all(X[:, 3] < (3 if subset is not None else 5))
            return 10 - (X[:, 0] - .999)**2 - (X[:, 1] - .397801214)**2 - (X[:,2]-4)**2 - (X[:,3]-(2 if subset is not None else 4))**2
        def predict(self, X):
            return np.full(len(X), .76), np.full(len(X), .03), np.full(len(X), .95)
    monkeypatch.setattr(constrained_gp, 'ConstrainedSurrogate', Surrogate)
    k, audit = extension.propose(schema, observations, {extension.key(o['params'], schema) for o in observations}, 42, allowed_ankles=subset)
    assert k[:2] == pytest.approx((.3998, .3295602428), abs=1e-6)
    assert k[2:] == ('k4', 'DM-J6216P-2EC' if subset is not None else 'DM-J8009P-2EC-V1.1')
    assert audit['global_best_observed_composite_cost'] == .7513
    assert audit['predicted_composite_cost'] == .76
    assert 'predicted_mae_mps' not in audit
    assert audit['motor_pairs'] == expected_pairs
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

def synthetic_initial(tmp_path, monkeypatch):
    panel = tmp_path/'50panel'; panel.mkdir()
    source = tmp_path/'plan.json'; source.write_text('{}')
    actor = tmp_path/'actor.pt'; actor.write_text('actor')
    scoring = tmp_path/'scoring.json'; scoring.write_text('{}')
    a = SimpleNamespace(history=None, history_budget=None, scoring=scoring, updates=100, seed=42,
                        trials=51, allowed_ankles=None, resume_paused=False)
    keys = [(round(.2+i*.003,10),.3,'k0','a0') for i in range(50)]
    grid = {k:dict(key=list(k),design_ids={t:f'{t}-{i}' for t in bo.TASKS}) for i,k in enumerate(keys)}
    plan = {'schema':SCHEMA}
    conditions = {'fixed':'test'}
    reports = {}
    evidence = []
    for i,k in enumerate(keys):
        p = panel/f'report-{i}.json'
        r = dict(candidate=dict(structure_key=list(k),design_ids=grid[k]['design_ids']),ppo_updates=100,
                 objective=1+i*.001)
        p.write_text(json.dumps(r)); reports[p] = r
        evidence.append(dict(report=p.name,sha256=bo.sha(p)))
    benchmark = dict(source_plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(actor),updates=100,
                     reports=evidence)
    monkeypatch.setattr(bo,'read_search_benchmark',lambda *args:(panel,benchmark))
    monkeypatch.setattr(bo,'calibration_keys',lambda:keys)
    monkeypatch.setattr(bo,'initial_report',lambda p,e,b:reports[p])
    monkeypatch.setattr(bo,'required_panel_pairs',lambda c:{(0,0)})
    def score(bo_,report,scoring_,conditions_):
        cost=report['objective']
        return dict(observations=[dict(objective=cost,feasible=True)],valid_pairs=1,joint_successes=1,
                    complete_objectives=1,mean_objective=cost)
    monkeypatch.setattr(extension,'score_report',score)
    monkeypatch.setattr(extension,'scoring_protocol',lambda *args:None)
    monkeypatch.setattr(extension,'grid_lookup',lambda *args:grid)
    return a,source,plan,actor,conditions,grid,reports


def test_initial50_only_and_new_trial_51_then_resume(tmp_path,monkeypatch):
    a,source,plan,actor,conditions,grid,reports=synthetic_initial(tmp_path,monkeypatch)
    out=tmp_path/'search';out.mkdir()
    k=(.4,.3295602428,'k4','a4'); calls=[]
    native_source=tmp_path/'candidate_plan.json';native_source.write_text('{}')
    native_plan={'schema':SCHEMA,'teachers':{task:{'design_ids':[task]} for task in bo.TASKS}}
    def propose(schema,obs,seen,seed,allowed_ankles=None):
        assert len(obs)==50 and len(seen)==50 and allowed_ankles==SCHEMA['ankles']
        calls.append('propose'); return k,{'continuous':True}
    def prepare(*args):
        calls.append('prepare');assert args[-1]==k
        return native_source,native_plan,'stairs'
    candidate_report=dict(candidate=dict(structure_key=list(k),design_ids={t:t for t in bo.TASKS}),
                          ppo_updates=100,objective=.8)
    def evaluate(a,folder,*args):
        calls.append('evaluate');folder.mkdir();bo.freeze(folder/'bo_evaluation.json',candidate_report)
        return candidate_report
    monkeypatch.setattr(extension,'propose',propose)
    monkeypatch.setattr(bo,'load_plan',lambda _:native_plan)
    monkeypatch.setattr(bo,'evaluate',evaluate)
    monkeypatch.setattr(extension,'native_usd_evidence',lambda *args:{'valid':True})
    monkeypatch.setattr(extension,'verify_materialized',lambda *args:(native_source,native_plan,'stairs'))
    monkeypatch.setattr(extension,'verify_candidate',lambda *args:candidate_report)
    extension.search(bo,a,out,source,plan,actor,conditions,prepare=prepare)
    assert calls==['propose','prepare','evaluate']
    assert bo.read(out/'trial_0050/proposal.json')['trial_number']==51
    budget=bo.read(out/'budget_0051.json')
    assert budget['trials']==51 and budget['best_observed']['mean_objective']==.8
    assert bo.read(out/'manifest.json')['historical_trials']==50
    extension.search(bo,a,out,source,plan,actor,conditions,prepare=prepare)
    assert calls==['propose','prepare','evaluate']


def test_completed_history_import_same_protocol_read_only(tmp_path,monkeypatch):
    a,source,plan,actor,conditions,grid,reports=synthetic_initial(tmp_path,monkeypatch)
    old=tmp_path/'old';old.mkdir();a.history=old;a.history_budget=51;a.trials=52
    a0=SimpleNamespace(**{**vars(a),'history':None,'history_budget':None})
    results,_,_=extension.load_history(bo,a0,source,plan,actor,conditions,{},grid)
    manifest=extension.history_identity(bo,a,source,actor,conditions)
    manifest['coordinator_sha256']='a'*64  # historical coordinator, not rewritten current version
    bo.freeze(old/'manifest.json',manifest)
    trial=old/'trial_0050';candidate=trial/'candidate';candidate.mkdir(parents=True)
    k=(.4,.3295602428,'k4','a4'); report=dict(candidate=dict(structure_key=list(k),design_ids={t:t for t in bo.TASKS}),ppo_updates=100,objective=.8)
    proposal=dict(kind='continuous_bo_proposal_v1',trial_number=51,structure_key=list(k),existing_grid_design_id=None,acquisition={})
    bo.freeze(trial/'proposal.json',proposal)
    bo.freeze(candidate/'bo_evaluation.json',report)
    row=dict(proposal,design_id='stairs',evaluation=str(candidate/'bo_evaluation.json'),**extension.score_report(bo,report,{},conditions))
    bo.freeze(trial/'observation.json',row)
    bo.freeze(candidate/'native_asset_evidence.json',{'valid':True})
    results.append(row)
    bo.freeze(old/'budget_0051.json',dict(trials=51,best_observed=row,candidates=[extension.summary(r) for r in results]))
    before={p:bo.sha(p) for p in old.rglob('*.json')}
    native_source=tmp_path/'candidate_plan.json';native_source.write_text('{}')
    native_plan={'schema':SCHEMA,'teachers':{t:{'design_ids':[t]} for t in bo.TASKS}}
    monkeypatch.setattr(extension,'verify_materialized',lambda *args:(native_source,native_plan,'stairs'))
    monkeypatch.setattr(extension,'verify_candidate',lambda *args:report)
    monkeypatch.setattr(extension,'native_usd_evidence',lambda *args:{'valid':True})
    monkeypatch.setattr(bo,'evaluate',lambda *args:pytest.fail('historical replay must not evaluate'))
    got,obs,evidence=extension.load_history(bo,a,source,plan,actor,conditions,{},grid)
    assert len(got)==len(obs)==51 and got[-1]==row
    assert evidence['budget_sha256']==bo.sha(old/'budget_0051.json')
    assert {p:bo.sha(p) for p in old.rglob('*.json')}==before
    manifest['scoring_sha256']='wrong';(old/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='scoring'):
        extension.load_history(bo,a,source,plan,actor,conditions,{},grid)


def test_import_inherited_offgrid_budget_keeps_earlier_paths(tmp_path,monkeypatch):
    a,source,plan,actor,conditions,grid,reports=synthetic_initial(tmp_path,monkeypatch)
    old=tmp_path/'old';old.mkdir()
    initial,_,initial_evidence=extension.load_history(bo,a,source,plan,actor,conditions,{},grid)
    native_source=tmp_path/'native_plan.json';native_source.write_text('{}')
    native_plan={'schema':SCHEMA,'teachers':{t:{'design_ids':[t]} for t in bo.TASKS}}
    k1=(.4,.3295602428,'k4','a4'); k2=(.39,.3271234567,'k4','a1')
    reports_by_path={}

    def committed(root,i,k,cost):
        trial=root/f'trial_{i:04d}';candidate=trial/'candidate';candidate.mkdir(parents=True)
        r=dict(candidate=dict(structure_key=list(k),design_ids={t:t+str(i) for t in bo.TASKS}),
               ppo_updates=100,objective=cost)
        # Stub IDs by trial; actual source/asset checks have separate tests.
        reports_by_path[candidate]=(native_source,{'schema':SCHEMA,'teachers':{t:{'design_ids':[t+str(i)]} for t in bo.TASKS}},'stairs'+str(i),r)
        proposal=dict(kind='continuous_bo_proposal_v1',trial_number=i+1,structure_key=list(k),existing_grid_design_id=None,acquisition={})
        bo.freeze(trial/'proposal.json',proposal);bo.freeze(candidate/'bo_evaluation.json',r)
        row=dict(proposal,design_id='stairs'+str(i),evaluation=str(candidate/'bo_evaluation.json'),
                 **extension.score_report(bo,r,{},conditions))
        bo.freeze(trial/'observation.json',row);bo.freeze(candidate/'native_asset_evidence.json',{'valid':True})
        return row
    def materialized(bo_,trial,*args):
        return reports_by_path[trial/'candidate'][:3]
    monkeypatch.setattr(extension,'verify_materialized',materialized)
    monkeypatch.setattr(extension,'verify_candidate',lambda bo_,folder,*args:reports_by_path[folder][3])
    monkeypatch.setattr(extension,'native_usd_evidence',lambda *args:{'valid':True})
    first=committed(old,50,k1,.8)
    proto=extension.history_identity(bo,a,source,actor,conditions)
    bo.freeze(old/'manifest.json',dict(original_protocol=proto,kind='rpo_shared_continuous_bo_search_v2',
          historical_trials=50, history_sha256='placeholder'))
    bo.freeze(old/'verified_history_0050.json',initial_evidence)
    m=bo.read(old/'manifest.json');m['history_sha256']=bo.sha(old/'verified_history_0050.json')
    (old/'manifest.json').write_text(json.dumps(m))
    bo.freeze(old/'budget_0051.json',dict(trials=51,best_observed=first,candidates=[extension.summary(r) for r in initial+[first]]))
    a.history=old;a.history_budget=None;a.trials=52
    history,_,evidence=extension.load_history(bo,a,source,plan,actor,conditions,{},grid)
    extended=tmp_path/'extended';extended.mkdir()
    bo.freeze(extended/'verified_history_0051.json',evidence)
    bo.freeze(extended/'manifest.json',dict(kind='rpo_shared_continuous_bo_extension_v1', original_protocol=proto,
           historical_trials=51,history_sha256=bo.sha(extended/'verified_history_0051.json')))
    second=committed(extended,51,k2,.9)
    bo.freeze(extended/'budget_0052.json',dict(trials=52,best_observed=first,candidates=[extension.summary(r) for r in history+[second]]))
    a.history=extended;a.trials=53;a.allowed_ankles=['a1']
    history,observations,new_evidence=extension.load_history(bo,a,source,plan,actor,conditions,{},grid)
    assert len(history)==len(observations)==52
    assert history[-2:]==[first,second]
    assert new_evidence['records'][-2]['report']==first['evaluation']
    assert observations[-2]['params']['ankle_motor']=='a4'  # excluded for future proposals only
    p=old/'trial_0050/observation.json';v=bo.read(p);v['mean_objective']=.7;p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='identity changed'):
        extension.load_history(bo,a,source,plan,actor,conditions,{},grid)


def test_old_discrete_output_rejected_before_any_new_receipt(tmp_path, monkeypatch):
    a,source,plan,actor,conditions,grid,reports=synthetic_initial(tmp_path,monkeypatch)
    old=tmp_path/'old';old.mkdir()
    bo.freeze(old/'manifest.json',extension.history_identity(bo,a,source,actor,conditions))
    bo.freeze(old/'budget_0100.json',{'unchanged':True})
    snapshot={p:bo.sha(p) for p in old.iterdir()}
    monkeypatch.setattr(extension,'load_history',lambda *args:pytest.fail('must reject old output before evaluation history reads'))
    with pytest.raises(ValueError,match='existing output'):
        extension.search(bo,a,old,source,plan,actor,conditions)
    assert {p:bo.sha(p) for p in old.iterdir()}==snapshot


def test_resume_new_manifest_requires_existing_matching_receipt(tmp_path,monkeypatch):
    a,source,plan,actor,conditions,grid,reports=synthetic_initial(tmp_path,monkeypatch)
    out=tmp_path/'continuous';out.mkdir()
    bo.freeze(out/'manifest.json',{'kind':'rpo_shared_continuous_bo_search_v2','historical_trials':100})
    bo.freeze(out/'verified_history_0100.json',{'records':'original'})
    before={p:bo.sha(p) for p in out.iterdir()}
    with pytest.raises(ValueError,match='Frozen BO history'):
        extension.search(bo,a,out,source,plan,actor,conditions)
    assert {p:bo.sha(p) for p in out.iterdir()}==before
