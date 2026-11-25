"""Extend a completed 100-observation search with ten continuous-leg BO designs."""
import argparse
import math
from pathlib import Path

from bo_compact_ankles import ALLOWED_ANKLES, frozen_coordinator


HERE = Path(__file__).resolve().parent
HISTORY_SIZE = 100
INITIAL_PANEL_SIZE = 50
TOTAL_SIZE = 110
FIELDS = ('thigh_m', 'calf_m', 'knee_motor', 'ankle_motor')
SUMMARY_KEYS = ('design_id', 'structure_key', 'evaluation', 'valid_pairs',
                'joint_successes', 'complete_objectives', 'mean_objective')


def key(params, schema):
    """Canonical physical identity, using the native shared-bank precision."""
    lengths = [float(params[f]) for f in FIELDS[:2]]
    for segment, value in zip(('thigh', 'calf'), lengths, strict=True):
        lower, upper = schema['ranges_m'][segment]
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'{segment} outside continuous search bounds')
    knee, ankle = params['knee_motor'], params['ankle_motor']
    if knee not in schema['knees'] or ankle not in schema['ankles']:
        raise ValueError('Unknown BO motor choice')
    result = (round(lengths[0], 10), round(lengths[1], 10), knee, ankle)
    for segment, value in zip(('thigh', 'calf'), result[:2], strict=True):
        lo, hi = schema['ranges_m'][segment]
        if not lo <= value <= hi:
            raise ValueError('Canonical length outside bounds')
    return result


def parameters(k):
    return dict(zip(FIELDS, k, strict=True))


def checked_key(value, schema, seen):
    canonical = key(parameters(value), schema)
    if tuple(value) != canonical:
        raise ValueError('Saved proposal is not at the canonical 10-decimal precision')
    if canonical in seen:
        raise ValueError('Continuous BO structure already evaluated')
    return canonical


def propose(schema, observations, seen, seed):
    """Original constrained EI, fitted on every observation, 15 new motor pairs."""
    from itertools import product

    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import qmc
    from rough_bo.constrained_gp import ConstrainedSurrogate

    if (len(schema['knees']) != 5 or len(set(schema['knees'])) != 5
            or len(schema['ankles']) != 5 or len(set(schema['ankles'])) != 5
            or not ALLOWED_ANKLES <= set(schema['ankles'])):
        raise ValueError('Expected the original five knee and ankle motor choices')
    model = ConstrainedSurrogate(schema['ranges_m'], schema['knees'], schema['ankles'],
                                 seed=seed, objective_key='objective')
    # Do not narrow the surrogate's schema or discard excluded-motor evidence.
    # Its EI threshold is the same unrestricted global feasible incumbent.
    model.fit(observations)
    proposal_seed = seed + len(observations)
    points = qmc.Sobol(2, scramble=True, seed=proposal_seed).random_base2(8)
    points = np.vstack((points, [[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]]))
    pairs = list(product(range(len(schema['knees'])), [
        i for i, motor in enumerate(schema['ankles']) if motor in ALLOWED_ANKLES]))
    raw = np.vstack([np.column_stack((points, np.tile(pair, (len(points), 1)))) for pair in pairs])
    existing = set(seen)

    def unseen_rows(X, used):
        rows = []
        for x in X:
            if not np.isfinite(x).all():
                continue
            k = key(model.decode([x])[0], schema)
            if k in used:
                continue
            used.add(k)
            rows.append(k)
        return rows

    proposal_keys = unseen_rows(raw, existing)
    if not proposal_keys:
        raise ValueError('No unevaluated continuous candidate in acquisition domain')
    pool = model.encode([parameters(k) for k in proposal_keys])
    if model.regressor is not None:
        extras = []
        scores = model.acquisition(pool)
        for pair in pairs:
            ix = np.where((pool[:, 2:] == pair).all(axis=1))[0]
            if not len(ix):
                continue
            start = pool[ix[np.argmax(scores[ix])], :2]
            result = minimize(lambda u: -float(model.acquisition([[*u, *pair]])[0]),
                              start, bounds=[(0, 1), (0, 1)], method='L-BFGS-B',
                              options={'maxiter': 25})
            extras.append(np.array([*result.x, *pair]))
        extra_keys = unseen_rows(extras, existing)
        if extra_keys:
            proposal_keys += extra_keys
            pool = np.vstack((pool, model.encode([parameters(k) for k in extra_keys])))
    values = model.acquisition(pool)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite continuous constrained EI')
    method = 'EI_times_probability_feasible'
    if model.regressor is None or values.max() <= 1e-16:
        distance = np.min(np.sum((pool[:, None, :2] - model.X[None, :, :2])**2, axis=-1)
                          + np.sum(pool[:, None, 2:] != model.X[None, :, 2:], axis=-1), axis=1)
        _, std, probability = model.predict(pool)
        values = probability * (1 + distance) if model.regressor is None else probability * (std + 1e-12)
        method = 'feasibility_exploration' if model.regressor is None else 'EI_underflow_uncertainty_fallback'
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite continuous acquisition fallback')
    index = int(np.argmax(values))
    mean, std, probability = model.predict(pool[index:index+1])
    result = checked_key(proposal_keys[index], schema, seen)
    return result, dict(
        method=method, seed=proposal_seed, candidate_pool_size=len(pool),
        global_best_observed_composite_cost=model.best,
        predicted_composite_cost=float(mean[0]) if model.regressor is not None else None,
        predicted_composite_cost_std=float(std[0]) if model.regressor is not None else None,
        predicted_feasibility=float(probability[0]), acquisition_value=float(values[index]),
        performance_kernel=str(model.regressor.kernel_) if model.regressor is not None else None,
        feasibility_kernel=str(model.classifier.kernel_) if model.classifier is not None else None,
        classifier='logistic_GPC_Laplace' if model.classifier is not None else 'one_class_Beta_prior',
        fit_warnings=model.fit_warnings, search_space='continuous_thigh_calf_exact_motor_categories',
        allowed_ankles=sorted(ALLOWED_ANKLES), motor_pairs=len(pairs),
        sobol_points_per_motor_pair=256, bounded_local_refinement=True,
        canonical_length_decimals=10, fit_performance_rows=model.performance_count,
        fit_feasibility_rows=model.feasibility_count)

def check_separation(history, output):
    old, new = Path(history).resolve(), Path(output).resolve()
    if old == new or old.is_relative_to(new) or new.is_relative_to(old):
        raise ValueError('Continuous search output must be separate from its immutable history')


def grid_lookup(bo, plan):
    keys = {tuple(k) for k in plan['selected_structure_keys']}
    rows = bo.read(plan['shared_bank'])['structures']
    grid = {tuple(row['key']): row for row in rows if tuple(row['key']) in keys}
    if len(grid) != len(keys):
        raise ValueError('Original common-bank structure map is incomplete')
    return grid


def scoring_protocol(bo, scoring, conditions):
    bo.validate_scoring(scoring, dict(domain='simulation',
                        conditions_sha256=bo.digest(conditions), conditions=conditions))
    if (scoring['method'] != bo.ORIGINAL_RPO_SCORING_METHOD
            or conditions.get('measurement_dt_s') != bo.PHYSICS_DT
            or conditions.get('reset_mode') != 'fixed'
            or bo.scoring_cot_basis(scoring) != bo.ABSOLUTE_COT_BASIS
            or scoring['task_weights'] != bo.FORMAL_TASK_WEIGHTS
            or len(bo.required_panel_pairs(conditions)) != 1):
        raise ValueError('Continuous extension needs the original fixed 200 Hz reference-ratio BO protocol')


def score_report(bo, report, scoring, conditions):
    bo.check_panel_pairs(report['pairs'], bo.required_panel_pairs(conditions))
    scored = bo.objective(report, scoring)
    if scored['complete_objectives'] != scored['joint_successes']:
        raise ValueError('Successful BO trial has incomplete performance measurements')
    return scored


def verify_candidate(bo, folder, benchmark, conditions):
    """Native evidence checks only: never launch or repair an old worker."""
    path = folder/'bo_evaluation.json'
    report = bo.initial_report(path, dict(sha256=bo.sha(path)), benchmark)
    return bo.checked_candidate_measurements(folder, report, benchmark, conditions, folder/'tuning')


def summary(result):
    return {k: result[k] for k in SUMMARY_KEYS}


def history_identity(bo, a, source, actor, conditions):
    return dict(kind='rpo_shared_bo_search_v1', source_plan_sha256=bo.sha(source),
                initial_actor_sha256=bo.sha(actor), conditions_sha256=bo.digest(conditions),
                scoring_sha256=bo.sha(a.scoring), coordinator_sha256=bo.sha(Path(bo.__file__)),
                surrogate_sha256=bo.sha(bo.REPO/'rough_bo/constrained_gp.py'),
                updates=a.updates, seed=a.seed)


def append_observations(observations, result):
    observations.extend(dict(o, params=parameters(result['structure_key']))
                        for o in result['observations'])


def load_history(bo, a, source, plan, actor, conditions, scoring, grid):
    """Read and verify all 100 committed measurements without old-run writes."""
    root = a.history.resolve()
    identity = history_identity(bo, a, source, actor, conditions)
    manifest_path = root/'manifest.json'
    if bo.read(manifest_path) != identity:
        raise ValueError('Original search must use the same frozen coordinator, actor, plan, scoring and seed')
    budget_path = root/'budget_0100.json'
    budget = bo.read(budget_path)
    entries = budget['candidates']
    if budget['trials'] != HISTORY_SIZE or len(entries) != HISTORY_SIZE:
        raise ValueError('A completed, exact 100-observation original budget is required')
    # This also validates the original reference and 50 initial raw traces,
    # checkpoint/actor hashes, frozen scale, and existing panel score receipt.
    panel, benchmark = bo.read_search_benchmark(a.scoring, scoring, conditions)
    keys = bo.calibration_keys()
    if (len(keys) != INITIAL_PANEL_SIZE or len(benchmark['reports']) != INITIAL_PANEL_SIZE
            or any(benchmark[k] != identity[k] for k in
                   ('source_plan_sha256', 'initial_actor_sha256', 'updates'))):
        raise ValueError('Original reference-ratio calibration does not match the search')
    results, observations, evidence, seen = [], [], [], set()
    for i, row in enumerate(entries):
        k = checked_key(row['structure_key'], plan['schema'], seen)
        if k not in grid or grid[k]['design_ids']['stairs'] != row['design_id']:
            raise ValueError('Original budget contains an invalid paired-bank design')
        design_ids = grid[k]['design_ids']
        if i < INITIAL_PANEL_SIZE:
            receipt = benchmark['reports'][i]
            if tuple(keys[i]) != k:
                raise ValueError('Initial structure order changed')
            path = panel/receipt['report']
            report = bo.initial_report(path, receipt, benchmark)
            proposal = dict(design_id=row['design_id'], structure_key=list(k))
            obs_hash = prop_hash = None
        else:
            trial = root/f'trial_{i:04d}'
            path = trial/'candidate/bo_evaluation.json'
            proposal_path, obs_path = trial/'proposal.json', trial/'observation.json'
            proposal = bo.read(proposal_path)
            if proposal['design_id'] != row['design_id'] or tuple(proposal['structure_key']) != k:
                raise ValueError('Historical trial proposal identity changed')
            report = verify_candidate(bo, path.parent, benchmark, conditions)
            obs_hash, prop_hash = bo.sha(obs_path), bo.sha(proposal_path)
        if (path.resolve() != Path(row['evaluation']).resolve()
                or report['candidate']['design_ids'] != design_ids
                or tuple(report['candidate']['structure_key']) != k
                or report['ppo_updates'] != a.updates):
            raise ValueError('Historical measured candidate differs from its budget entry')
        result = dict(proposal, evaluation=str(path), **score_report(bo, report, scoring, conditions))
        if summary(result) != row or (i >= INITIAL_PANEL_SIZE and bo.read(obs_path) != result):
            raise ValueError('Historical score differs from the frozen BO observation')
        results.append(result)
        append_observations(observations, result)
        evidence.append(dict(design_id=row['design_id'], structure_key=list(k),
                             report=str(path), report_sha256=bo.sha(path),
                             observation_sha256=obs_hash, proposal_sha256=prop_hash))
        seen.add(k)
    if len({r['design_id'] for r in results}) != HISTORY_SIZE:
        raise ValueError('Duplicate design identities in historical budget')
    return results, observations, dict(history=str(root), budget_sha256=bo.sha(budget_path),
        original_manifest_sha256=bo.sha(manifest_path), scoring_sha256=bo.sha(a.scoring),
        records=evidence)


def materialize(bo, a, trial, donor_source, donor_plan, k, grid, prepare):
    if k in grid:
        return donor_source, donor_plan, grid[k]['design_ids']['stairs']
    return prepare(bo, a, trial/'assets', donor_source, donor_plan, k)


def native_usd_evidence(bo, candidate_root, updates, design_ids):
    """Verify the exact native USD used by the final committed PPO session."""
    tuning = candidate_root/'tuning'
    checkpoint = tuning/'checkpoints'/f'update_{updates:06d}.json'
    session_id = bo.read(checkpoint)['training_reward']['session_id']
    attempts = (tuning/'attempts').resolve()
    session = (attempts/session_id).resolve()
    if session.parent != attempts or not session.is_dir():
        raise ValueError('Invalid Native USD evidence training session')
    tasks = {}
    for task, design_id in design_ids.items():
        audit_path = session/task/'asset_cache_audit.json'
        entries = bo.read(audit_path)
        if len(entries) != 1 or entries[0]['design'] != design_id:
            raise ValueError(f'Native USD asset audit not bound to exact candidate: {task}')
        row = entries[0]
        cache = Path(row['cache']).resolve()
        usd = cache/('robot_with_backpack.usd' if task == 'backpack' else 'rpo.usd')
        asset_hash = cache/'.asset_hash'
        if (not asset_hash.is_file() or not (cache/'config.yaml').is_file() or not usd.is_file()
                or usd.stat().st_size <= 0 or not row['exact_hash']
                or asset_hash.read_text(encoding='utf8').strip() != row['exact_hash']):
            raise ValueError(f'Native USD cache does not match converter audit: {task}')
        tasks[task] = dict(design_id=design_id, converter_hash=row['exact_hash'], usd=str(usd),
                           audit_sha256=bo.sha(audit_path), reused=row['reused'])
    return dict(training_session=session_id, checkpoint_receipt_sha256=bo.sha(checkpoint), tasks=tasks)

def search(bo, a, work, donor_source, donor_plan, actor, conditions, *, prepare):
    """Native frozen evaluator is used only for the new ten candidate dirs."""
    check_separation(a.history, work)
    if a.trials != TOTAL_SIZE or a.updates != bo.DEFAULT_BO_UPDATES:
        raise ValueError('Expected 100 original + 10 new structures, each using 100 PPO updates')
    scoring = bo.read(a.scoring)
    scoring_protocol(bo, scoring, conditions)
    grid = grid_lookup(bo, donor_plan)
    schema = donor_plan['schema']
    if (len(schema['knees']) != 5 or len(schema['ankles']) != 5
            or schema['ranges_m'] != {'thigh':[.2,.4], 'calf':[.25,.45]}):
        raise ValueError('Original morphology search schema changed')
    results, observations, evidence = load_history(
        bo, a, donor_source, donor_plan, actor, conditions, scoring, grid)
    recorded_history = work/'verified_history_0100.json'
    bo.freeze(recorded_history, evidence)
    manifest = dict(kind='rpo_shared_continuous_bo_extension_v1',
                    history_sha256=bo.sha(recorded_history), historical_trials=HISTORY_SIZE,
                    total_trials=TOTAL_SIZE, original_protocol=history_identity(
                        bo, a, donor_source, actor, conditions),
                    search_schema={field:schema[field] for field in ('ranges_m','knees','ankles')},
                    allowed_ankles=sorted(ALLOWED_ANKLES),
                    acquisition='restricted_sobol_local_refined_global_incumbent_constrained_EI',
                    coordinator_sha256=bo.sha(Path(__file__)),
                    asset_builder_sha256=bo.sha(HERE/'bo_continuous_assets.py'))
    bo.freeze(work/'manifest.json', manifest)
    bo.resume_pause(work, a.resume_paused)
    seen = {tuple(r['structure_key']) for r in results}
    print('BO_CONTINUOUS_HISTORY_VERIFIED', len(results), '/', HISTORY_SIZE, flush=True)

    for i in range(HISTORY_SIZE, TOTAL_SIZE):
        trial = work/f'trial_{i:04d}'
        trial.mkdir(exist_ok=True)
        request, observation = trial/'proposal.json', trial/'observation.json'
        if request.is_file():
            proposal = bo.read(request)
            k = checked_key(proposal['structure_key'], schema, seen)
        else:
            if any(trial.iterdir()):
                raise ValueError('Unexpected uncommitted candidate without a frozen proposal')
            k, audit = propose(schema, observations, seen, a.seed)
            proposal = dict(kind='continuous_bo_proposal_v1', trial_number=i+1,
                            structure_key=list(k), acquisition=audit,
                            existing_grid_design_id=grid[k]['design_ids']['stairs'] if k in grid else None)
            bo.freeze(request, proposal)
        if (k[3] not in ALLOWED_ANKLES or proposal['kind'] != 'continuous_bo_proposal_v1'
                or proposal['trial_number'] != i+1
                or proposal['existing_grid_design_id'] != (grid[k]['design_ids']['stairs'] if k in grid else None)):
            raise ValueError('Frozen continuous proposal identity changed')
        if (work/'USER_PAUSED').exists():
            raise KeyboardInterrupt
        print('BO_TRIAL', i+1, '/', TOTAL_SIZE, list(k), flush=True)
        source, native_plan, design_id = materialize(bo, a, trial, donor_source, donor_plan, k, grid, prepare)
        if native_plan['schema'] != schema or bo.load_plan(source) != native_plan:
            raise ValueError('Candidate preparation changed the original policy encoding')
        target = trial/'candidate'
        benchmark = dict(source_plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(actor),
                         updates=a.updates)
        if observation.is_file():
            report = verify_candidate(bo, target, benchmark, conditions)
        else:
            report = bo.evaluate(a, target, source, native_plan, actor, design_id, conditions)
        design_ids = grid[k]['design_ids'] if k in grid else {
            task: spec['design_ids'][0] for task, spec in native_plan['teachers'].items()}
        if (tuple(report['candidate']['structure_key']) != k or report['candidate']['design_ids'] != design_ids
                or design_id != design_ids['stairs'] or report['ppo_updates'] != a.updates):
            raise ValueError('Measured candidate differs from the frozen exact proposal')
        if k not in grid:
            bo.freeze(target/'native_asset_evidence.json', native_usd_evidence(bo, target, a.updates, design_ids))
        result = dict(proposal, design_id=design_id, evaluation=str(target/'bo_evaluation.json'),
                      **score_report(bo, report, scoring, conditions))
        bo.freeze(observation, result)
        results.append(result)
        append_observations(observations, result)
        seen.add(k)
    complete = [r for r in results if r['mean_objective'] is not None
                and r['complete_objectives'] == len(bo.required_panel_pairs(conditions))]
    best = min(complete, key=lambda r:r['mean_objective']) if complete else None
    bo.freeze(work/'budget_0110.json', dict(trials=len(results), best_observed=best,
                                         candidates=[summary(r) for r in results]))
    print('BO_CONTINUOUS_COMPLETE', work/'budget_0110.json', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('history','output','root','shared-continuation','scoring','frozen-code-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--conditions',type=Path)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--trials',type=int,default=110)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--resume-paused',action='store_true')
    parser.add_argument('--reference-python',type=Path)
    parser.add_argument('--reference-device',default='cuda:0')
    parser.add_argument('--warp-cache',type=Path)
    parser.add_argument('--template',type=Path)
    parser.add_argument('--mesh-dir',type=Path)
    a = parser.parse_args(argv)
    a.history, a.output, a.root, a.scoring = (p.resolve() for p in
                                              (a.history,a.output,a.root,a.scoring))
    check_separation(a.history, a.output)
    if a.updates != 100 or a.trials != TOTAL_SIZE or a.gpu < 0 or a.timeout <= 0:
        parser.error('Use the original 100-update protocol and a total of 110 designs')
    # Both new entry modules have only stdlib imports at import time. The
    # frozen coordinator then puts its code roots ahead of the current checkout.
    from bo_continuous_assets import prepare_candidate
    bo = frozen_coordinator(a.frozen_code_root)
    a.conditions = (a.conditions or bo.PHYSICS_CONDITIONS).resolve()
    conditions = bo.read_conditions(a.conditions)
    if bo.official_conditions_path(bo.digest(conditions)) != bo.PHYSICS_CONDITIONS:
        parser.error('Expected original 200 Hz physics-step evaluation')
    with bo.run_lock(a.output):
        a.output.mkdir(exist_ok=True,parents=True)
        try:
            if (a.output/'USER_PAUSED').exists() and not a.resume_paused:
                raise RuntimeError('USER_PAUSED; pass --resume-paused to continue this run')
            donor, plan, actor = bo.shared_input(a.root, formal=True,
                require_horizontal_selection=True, continuation=a.shared_continuation.resolve())
            search(bo, a, a.output, donor, plan, actor, conditions, prepare=prepare_candidate)
        except KeyboardInterrupt:
            (a.output/'USER_PAUSED').touch()
            raise SystemExit(130)


if __name__ == '__main__':
    main()
