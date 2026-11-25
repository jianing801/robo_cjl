"""Continuous three-task BO from the frozen 50-design calibration or completed history."""
import re
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
INITIAL_PANEL_SIZE = 50
# Exact LF and CRLF coordinator identities from pre-quota main b694a15.
PRE_QUOTA_COORDINATOR_SHA256 = {
    '7753035c3154838158f70c8ac45bd66a1ce7e1edc5807af879b4d6f5bafeff27',
    '53ffe6fa8c65f150bf2a7f39c10a7b1824914706d60146320a94f89450e422be',
}
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


def allowed_choices(schema, allowed_ankles=None):
    """Default to the full trained pool; restrictions apply only to new proposals."""
    if (len(schema['knees']) != 5 or len(set(schema['knees'])) != 5
            or len(schema['ankles']) != 5 or len(set(schema['ankles'])) != 5):
        raise ValueError('Expected the original five knee and ankle motor choices')
    requested = schema['ankles'] if allowed_ankles is None or allowed_ankles == ['all'] else allowed_ankles
    if (not requested or len(set(requested)) != len(requested)
            or not set(requested) <= set(schema['ankles'])):
        raise ValueError('Allowed ankle motor IDs must be a nonempty, unique subset of the original pool')
    return [motor for motor in schema['ankles'] if motor in requested]


def ankle_quota_policy(start, total, allowed, *, imported):
    """Only a fresh formal 50+50 BO run receives the fixed ankle quotas."""
    if imported or start != INITIAL_PANEL_SIZE or total != 100:
        return None
    minimum, maximum, new_trials = 5, 20, total-start
    coverage_trials = minimum*len(allowed)
    if not coverage_trials <= new_trials <= maximum*len(allowed):
        raise ValueError('Ankle quota cannot fit 50 new trials with these allowed motors')
    return dict(kind='ankle_coverage_then_capped_cei_v1', scope='new_committed_observations',
                minimum_per_ankle=minimum, maximum_per_ankle=maximum,
                coverage_trials=coverage_trials, new_trials=new_trials)


def quota_gate(policy, allowed, counts, ordinal):
    """Compute the eligible acquisition domain from completed new trials only."""
    minimum, maximum = policy['minimum_per_ankle'], policy['maximum_per_ankle']
    if (set(counts) != set(allowed) or type(ordinal) is not int
            or not 0 <= ordinal < policy['new_trials']
            or any(type(v) is not int or not 0 <= v <= maximum for v in counts.values())
            or sum(counts.values()) != ordinal):
        raise ValueError('Invalid completed ankle quota state')
    if ordinal < policy['coverage_trials']:
        phase, threshold = 'minimum_coverage', minimum
        if any(v > minimum for v in counts.values()):
            raise ValueError('Ankle quota coverage phase exceeded its minimum target')
    else:
        phase, threshold = 'capped_improvement', maximum
        if any(v < minimum for v in counts.values()):
            raise ValueError('Ankle quota coverage phase did not reach every minimum')
    eligible = [motor for motor in allowed if counts[motor] < threshold]
    if not eligible:
        raise ValueError('Ankle quota has no eligible motor for the next proposal')
    return dict(phase=phase, new_completed_before=ordinal,
                completed_counts=dict(counts), eligible_ankles=eligible)


def verify_quota_prefix(bo, work, start, total, policy, allowed):
    """Read-only preflight: no gap or inconsistent frozen proposal may launch work."""
    counts = dict.fromkeys(allowed,0)
    ended = False
    for i in range(start,total):
        trial = work/f'trial_{i:04d}'
        request, observation = trial/'proposal.json', trial/'observation.json'
        if not request.is_file():
            if observation.exists() or (trial.is_dir() and any(trial.iterdir())):
                raise ValueError('Ankle quota trial has artifacts without its frozen proposal')
            ended = True
            continue
        if ended:
            raise ValueError('Ankle quota has a proposal after an unfinished trial')
        proposal = bo.read(request)
        gate = quota_gate(policy,allowed,counts,i-start)
        if (proposal.get('trial_number') != i+1 or proposal.get('ankle_quota') != gate
                or proposal.get('acquisition',{}).get('allowed_ankles') != gate['eligible_ankles']
                or len(proposal.get('structure_key',[])) != 4
                or proposal['structure_key'][3] not in gate['eligible_ankles']):
            raise ValueError(f'Frozen ankle quota proposal is inconsistent: {request}')
        if observation.is_file():
            committed = bo.read(observation)
            if any(committed.get(k) != v for k,v in proposal.items()):
                raise ValueError(f'Ankle quota proposal and observation differ: {observation}')
            counts[proposal['structure_key'][3]]+=1
        else:
            ended = True
    return counts

def propose(schema, observations, seen, seed, allowed_ankles=None):
    """Constrained EI, fitted on every observation, over requested motor pairs."""
    from itertools import product

    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import qmc
    from rough_bo.constrained_gp import ConstrainedSurrogate

    allowed_ankles = allowed_choices(schema, allowed_ankles)
    model = ConstrainedSurrogate(schema['ranges_m'], schema['knees'], schema['ankles'],
                                 seed=seed, objective_key='objective')
    # Do not narrow the surrogate's schema or discard excluded-motor evidence.
    # Its EI threshold is the same unrestricted global feasible incumbent.
    model.fit(observations)
    proposal_seed = seed + len(observations)
    points = qmc.Sobol(2, scramble=True, seed=proposal_seed).random_base2(8)
    points = np.vstack((points, [[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]]))
    pairs = list(product(range(len(schema['knees'])), [
        i for i, motor in enumerate(schema['ankles']) if motor in allowed_ankles]))
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
        allowed_ankles=allowed_ankles, motor_pairs=len(pairs),
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
    if (scoring['method'] not in bo.ORIGINAL_RPO_SCORING_METHODS
            or conditions.get('measurement_dt_s') != bo.PHYSICS_DT
            or conditions.get('reset_mode') != 'fixed'
            or bo.scoring_cot_basis(scoring) != bo.ABSOLUTE_COT_BASIS
            or scoring['task_weights'] != bo.FORMAL_TASK_WEIGHTS
            or len(bo.required_panel_pairs(conditions)) != 1):
        raise ValueError('Continuous extension needs the original fixed 200 Hz reference-ratio BO protocol')


def score_report(bo, report, scoring, conditions):
    bo.check_panel_pairs(report['pairs'], bo.required_panel_pairs(conditions))
    scored = bo.objective(report, scoring)
    if (scored['valid_pairs'] != len(bo.required_panel_pairs(conditions))
            or scored['complete_objectives'] != scored['joint_successes']):
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


def load_initial(bo, a, source, plan, actor, conditions, scoring, grid):
    """The first 50 measurements are fixed; do not run calibration workers."""
    protocol = history_identity(bo, a, source, actor, conditions)
    # Verifies frozen original reference, initial panel traces, policy hashes,
    # fixed conditions and all scoring/initial observation receipts.
    panel, benchmark = bo.read_search_benchmark(a.scoring, scoring, conditions)
    keys = bo.calibration_keys()
    if (len(keys) != INITIAL_PANEL_SIZE or len(benchmark['reports']) != INITIAL_PANEL_SIZE
            or any(benchmark[k] != protocol[k] for k in
                   ('source_plan_sha256', 'initial_actor_sha256', 'updates'))):
        raise ValueError('Original reference-ratio calibration does not match the search')
    results, observations, evidence, seen = [], [], [], set()
    for k, receipt in zip(keys, benchmark['reports'], strict=True):
        k = checked_key(k, plan['schema'], seen)
        row = grid[k]
        path = panel/receipt['report']
        report = bo.initial_report(path, receipt, benchmark)
        if (report['candidate']['design_ids'] != row['design_ids']
                or tuple(report['candidate']['structure_key']) != k
                or report['ppo_updates'] != a.updates):
            raise ValueError('Initial BO observation is attached to another candidate or budget')
        result = dict(design_id=row['design_ids']['stairs'], structure_key=list(k), evaluation=str(path),
                      **score_report(bo, report, scoring, conditions))
        results.append(result)
        append_observations(observations, result)
        evidence.append(dict(design_id=result['design_id'], structure_key=list(k), report=str(path),
            report_sha256=bo.sha(path), observation_sha256=None, proposal_sha256=None))
        seen.add(k)
    return results, observations, evidence


def history_budget_path(root, requested):
    if requested is not None:
        if requested < INITIAL_PANEL_SIZE:
            raise ValueError('Historical BO budget must include the first 50 structures')
        candidate = root/f'budget_{requested:04d}.json'
        if not candidate.is_file():
            raise ValueError(f'Completed historical budget not found: {candidate}')
        return candidate, requested
    choices = [(int(m.group(1)), p) for p in root.glob('budget_*.json')
               if (m := re.fullmatch(r'budget_([0-9]{4,})\.json',p.name))]
    if not choices:
        raise ValueError(f'No completed historical budget in {root}')
    count, path = max(choices)
    return path, count


def load_history(bo, a, source, plan, actor, conditions, scoring, grid):
    """Read original panel, optionally append verified completed BO budgets without old writes."""
    results, observations, evidence = load_initial(bo, a, source, plan, actor, conditions, scoring, grid)
    if getattr(a, 'history', None) is None:
        if getattr(a, 'history_budget', None) is not None:
            raise ValueError('--history-budget requires --history')
        return results, observations, dict(history=None, scoring_sha256=bo.sha(a.scoring), records=evidence)
    root = Path(a.history).resolve()
    identity = history_identity(bo, a, source, actor, conditions)
    manifest_path = root/'manifest.json'
    manifest = bo.read(manifest_path)
    kinds = ('rpo_shared_bo_search_v1', 'rpo_shared_continuous_bo_extension_v1', 'rpo_shared_continuous_bo_search_v2')
    if manifest.get('kind') not in kinds:
        raise ValueError('Unknown historical BO protocol')
    saved = manifest.get('original_protocol',manifest)
    # Older coordinators are deliberately not rerun/rewritten; their recorded
    # sources stay frozen in the source manifest. Independently check every
    # measurement/checkpoint and the exact scientific protocol/score identity.
    if any(saved.get(k) != v for k,v in identity.items() if k != 'coordinator_sha256'):
        raise ValueError('Historical search must use the same actor, plan, scoring, surrogate, updates and seed')
    if not re.fullmatch(r'[0-9a-f]{64}',saved.get('coordinator_sha256','')):
        raise ValueError('Historical coordinator identity missing')
    budget_path, count = history_budget_path(root, getattr(a,'history_budget',None))
    budget = bo.read(budget_path)
    entries = budget['candidates']
    if (count < INITIAL_PANEL_SIZE or budget['trials'] != count or len(entries) != count
            or [summary(r) for r in results] != entries[:INITIAL_PANEL_SIZE]):
        raise ValueError('Historical BO budget does not match the frozen first 50 structures')
    inherited = manifest.get('historical_trials', INITIAL_PANEL_SIZE)
    prior_records = []
    if inherited != INITIAL_PANEL_SIZE or manifest.get('history_sha256') is not None:
        receipt_path = root/f'verified_history_{inherited:04d}.json'
        if (not INITIAL_PANEL_SIZE <= inherited <= count or bo.sha(receipt_path) != manifest['history_sha256']):
            raise ValueError('Frozen inherited BO history receipt changed')
        prior_records = bo.read(receipt_path)['records']
        if len(prior_records) != inherited or prior_records[:INITIAL_PANEL_SIZE] != evidence:
            raise ValueError('Inherited BO history does not match the initial calibration')
    seen = {tuple(r['structure_key']) for r in results}
    for i,row in enumerate(entries[INITIAL_PANEL_SIZE:],INITIAL_PANEL_SIZE):
        k = checked_key(row['structure_key'], plan['schema'], seen)
        path = Path(row['evaluation'])
        if i < inherited:
            prior = prior_records[i]
            if str(path) != prior['report'] or bo.sha(path) != prior['report_sha256']:
                raise ValueError('Inherited historical report identity changed')
            trial = path.parent.parent
        else:
            prior = None
            trial = root/f'trial_{i:04d}'
        if path.resolve() != (trial/'candidate/bo_evaluation.json').resolve():
            raise ValueError('Historical BO measurement is outside its exact committed trial')
        proposal_path, obs_path = trial/'proposal.json', trial/'observation.json'
        proposal = bo.read(proposal_path)
        if (tuple(proposal['structure_key']) != k or proposal.get('design_id',row['design_id']) != row['design_id']
                or proposal.get('trial_number',i+1) != i+1
                or (prior and (prior['proposal_sha256'] != bo.sha(proposal_path)
                              or prior['observation_sha256'] != bo.sha(obs_path)))):
            raise ValueError('Historical trial proposal identity changed')
        candidate_source, candidate_plan, design_id = verify_materialized(bo, trial, source, plan, k, grid)
        benchmark = dict(source_plan_sha256=bo.sha(candidate_source), initial_actor_sha256=bo.sha(actor),
                         updates=a.updates)
        report = verify_candidate(bo, path.parent, benchmark, conditions)
        design_ids = candidate_ids(k, grid, candidate_plan)
        if (report['candidate']['design_ids'] != design_ids or tuple(report['candidate']['structure_key']) != k
                or report['ppo_updates'] != a.updates or design_id != row['design_id']):
            raise ValueError('Historical measured candidate differs from its budget entry')
        if k not in grid and bo.read(path.parent/'native_asset_evidence.json') != native_usd_evidence(bo,path.parent,a.updates,design_ids):
            raise ValueError('Historical native USD evidence changed')
        result = dict(proposal,design_id=design_id, evaluation=str(path), **score_report(bo, report, scoring, conditions))
        if summary(result) != row or bo.read(obs_path) != result:
            raise ValueError('Historical score differs from the frozen BO observation')
        results.append(result)
        append_observations(observations, result)
        evidence.append(dict(design_id=row['design_id'], structure_key=list(k),
                             report=str(path), report_sha256=bo.sha(path),
                             observation_sha256=bo.sha(obs_path), proposal_sha256=bo.sha(proposal_path)))
        seen.add(k)
    if len({r['design_id'] for r in results}) != count:
        raise ValueError('Duplicate design identities in historical budget')
    return results, observations, dict(history=str(root), budget_sha256=bo.sha(budget_path),
        original_manifest_sha256=bo.sha(manifest_path), scoring_sha256=bo.sha(a.scoring), records=evidence)

def materialize(bo, a, trial, donor_source, donor_plan, k, grid, prepare):
    if k in grid:
        return donor_source, donor_plan, grid[k]['design_ids']['stairs']
    return prepare(bo, a, trial/'assets', donor_source, donor_plan, k)


def candidate_ids(k, grid, native_plan):
    return grid[k]['design_ids'] if k in grid else {
        task: spec['design_ids'][0] for task, spec in native_plan['teachers'].items()}


def verify_materialized(bo, trial, donor_source, donor_plan, k, grid):
    """Verify accepted exact native assets without re-running a generator or worker."""
    if k in grid:
        return donor_source, donor_plan, grid[k]['design_ids']['stairs']
    from bo_continuous_assets import BANKS, singleton_config, _audit_common
    from amp_engine.amp_references import validate_library
    from contract import reference_context

    work = trial/'assets'
    identity = bo.read(work/'native_identity.json')
    source = work/'candidate_plan.json'
    plan = bo.load_plan(source)
    schema = donor_plan['schema']
    if (identity['donor_plan_sha256'] != bo.sha(donor_source) or identity['key'] != list(k)
            or identity['donor_banks'] != {t:donor_plan['teachers'][t]['bank_sha256'] for t in BANKS}
            or plan['schema'] != schema or plan['reference_structure_key'] != list(k)
            or plan['selected_structure_keys'] != [list(k)]):
        raise ValueError('Historical off-grid candidate does not share original donor identity')
    config = singleton_config(list(k),schema)
    if bo.read(work/'singleton_config.json') != config:
        raise ValueError('Historical singleton config changed')
    donor = {t:bo.read(donor_plan['teachers'][t]['bank']) for t in BANKS}
    paths = {t:Path(plan['teachers'][t]['bank']).resolve() for t in BANKS}
    for task,label in BANKS.items():
        stage = work/'native_banks'/label
        receipt = bo.read(stage/'completed.json')
        if (paths[task] != (stage/receipt['attempt']/'bank.json').resolve()
                or bo.sha(paths[task]) != receipt['evidence']['bank_sha256']
                or plan['teachers'][task]['design_ids'] != [receipt['evidence']['design_id']]):
            raise ValueError('Historical native singleton bank receipt changed')
    common = _audit_common(bo,paths,donor,config,schema,list(k))
    if bo.read(work/'bank_audit.json') != common:
        raise ValueError('Historical exact common morphology audit changed')
    if bo.read(plan['shared_bank'])['structures'] != [dict(key=list(k),design_ids=common['design_ids'])]:
        raise ValueError('Historical candidate common bank changed')
    context = bo.read(work/'actor_context.json')
    for task in BANKS:
        bank = bo.read(paths[task]); (row,) = bank['designs']
        if (context != dict(schema=schema['context_schema'],key=list(k),
                            normalization=schema['context_normalization'],values=reference_context(bank,row,schema,list(k)))):
            raise ValueError('Historical candidate actor context changed')
    ref = Path(plan['teachers']['run']['reference_library']).resolve()
    stage = work/'native_references'; receipt = bo.read(stage/'completed.json')
    if (ref != (stage/receipt['attempt']/'library.json').resolve()
            or not bo.sha(ref) == receipt['evidence']['library_sha256'] == plan['teachers']['run']['reference_library_sha256']
            or receipt['evidence']['urdf_sha256'] != common['actual_urdf_hashes']['run']):
        raise ValueError('Historical AMP reference identity changed')
    validate_library(bo.read(paths['run']),paths['run'],ref)
    return source,plan,common['design_ids']['stairs']

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

def search(bo, a, work, donor_source, donor_plan, actor, conditions, *, prepare=None):
    """Run continuous BO; imported and already completed evidence stays read-only."""
    if prepare is None:
        from bo_continuous_assets import prepare_candidate as prepare
    work = Path(work).resolve()
    a.search_root = work
    manifest_path = work/'manifest.json'
    existing_manifest = bo.read(manifest_path) if manifest_path.is_file() else None
    if existing_manifest is not None and existing_manifest.get('kind') != 'rpo_shared_continuous_bo_search_v2':
        raise ValueError('The existing output belongs to a different BO protocol; choose a new output and --history')
    if (work/'USER_PAUSED').exists() and not a.resume_paused:
        raise RuntimeError('USER_PAUSED; pass --resume-paused to continue this run')
    if getattr(a,'history',None) is not None:
        check_separation(a.history,work)
    if a.trials < INITIAL_PANEL_SIZE or a.updates != bo.DEFAULT_BO_UPDATES:
        raise ValueError('BO requires >=50 total structures with the original 100-update PPO protocol')
    scoring = bo.read(a.scoring)
    scoring_protocol(bo, scoring, conditions)
    grid = grid_lookup(bo, donor_plan)
    schema = donor_plan['schema']
    if schema['ranges_m'] != {'thigh':[.2,.4], 'calf':[.25,.45]}:
        raise ValueError('Original morphology search schema changed')
    allowed_ankles = allowed_choices(schema,getattr(a,'allowed_ankles',None))
    results, observations, evidence = load_history(
        bo, a, donor_source, donor_plan, actor, conditions, scoring, grid)
    start = len(results)
    if a.trials < start:
        raise ValueError('Total BO trial budget cannot be below imported completed history count')
    legacy_no_quota = (existing_manifest is not None
                       and existing_manifest.get('coordinator_sha256') in PRE_QUOTA_COORDINATOR_SHA256
                       and 'ankle_quota' not in existing_manifest)
    quota = None if legacy_no_quota else ankle_quota_policy(start, a.trials, allowed_ankles,
                                     imported=getattr(a,'history',None) is not None)
    recorded_history = work/f'verified_history_{start:04d}.json'
    # An incompatible resume must not create even a new history receipt in an
    # existing run. If the coordinator was interrupted just before publishing
    # its first manifest, an identical standalone history receipt is reusable.
    if existing_manifest is not None:
        if (existing_manifest.get('historical_trials') != start
                or not recorded_history.is_file()
                or existing_manifest.get('history_sha256') != bo.sha(recorded_history)
                or bo.read(recorded_history) != evidence):
            raise ValueError('Frozen BO history does not match the existing output')
    else:
        if any(p != recorded_history and p.name != 'USER_PAUSED' for p in work.iterdir()):
            raise ValueError('The existing output contains untracked artifacts; use a separate BO output')
        bo.freeze(recorded_history,evidence)
    manifest = dict(kind='rpo_shared_continuous_bo_search_v2',
                    history_sha256=bo.sha(recorded_history), historical_trials=start,
                    total_trials=a.trials, original_protocol=history_identity(
                        bo, a, donor_source, actor, conditions),
                    search_schema={field:schema[field] for field in ('ranges_m','knees','ankles')},
                    allowed_ankles=allowed_ankles,
                    acquisition='sobol_local_refined_global_incumbent_constrained_EI',
                    coordinator_sha256=bo.sha(Path(__file__)),
                    asset_builder_sha256=bo.sha(HERE/'bo_continuous_assets.py'))
    if quota is not None:
        manifest['ankle_quota'] = quota
        verify_quota_prefix(bo,work,start,a.trials,quota,allowed_ankles)
    if legacy_no_quota:
        # Accept only the exact old coordinator with every other manifest field
        # unchanged; never silently introduce quotas into a pre-quota run.
        manifest['coordinator_sha256'] = existing_manifest['coordinator_sha256']
    bo.freeze(manifest_path, manifest)
    bo.resume_pause(work, a.resume_paused)
    counts = dict.fromkeys(allowed_ankles,0)
    seen = {tuple(r['structure_key']) for r in results}
    print('BO_CONTINUOUS_HISTORY_VERIFIED', start, '/', start, flush=True)

    for i in range(start, a.trials):
        trial = work/f'trial_{i:04d}'
        trial.mkdir(exist_ok=True)
        request, observation = trial/'proposal.json', trial/'observation.json'
        gate = quota_gate(quota,allowed_ankles,counts,i-start) if quota is not None else None
        eligible = gate['eligible_ankles'] if gate is not None else allowed_ankles
        existing_request = request.is_file()
        if existing_request:
            proposal = bo.read(request)
            k = checked_key(proposal['structure_key'], schema, seen)
        else:
            if any(trial.iterdir()):
                raise ValueError('Unexpected uncommitted candidate without a frozen proposal')
            k, audit = propose(schema, observations, seen, a.seed, allowed_ankles=eligible)
            proposal = dict(kind='continuous_bo_proposal_v1', trial_number=i+1,
                            structure_key=list(k), acquisition=audit,
                            existing_grid_design_id=grid[k]['design_ids']['stairs'] if k in grid else None)
            if gate is not None:
                proposal['ankle_quota'] = gate
        if (k[3] not in eligible or proposal['kind'] != 'continuous_bo_proposal_v1'
                or proposal['trial_number'] != i+1
                or proposal['existing_grid_design_id'] != (grid[k]['design_ids']['stairs'] if k in grid else None)):
            raise ValueError('Frozen continuous proposal identity or ankle quota eligibility changed')
        if gate is not None and (proposal.get('ankle_quota') != gate
                or proposal.get('acquisition',{}).get('allowed_ankles') != eligible):
            raise ValueError('Frozen continuous proposal ankle quota state changed')
        if not existing_request:
            bo.freeze(request, proposal)
        if (work/'USER_PAUSED').exists():
            raise KeyboardInterrupt
        target = trial/'candidate'
        print('BO_TRIAL', i+1, '/', a.trials, list(k), flush=True)
        if observation.is_file():
            source, native_plan, design_id = verify_materialized(bo,trial,donor_source,donor_plan,k,grid)
        else:
            source, native_plan, design_id = materialize(bo, a, trial, donor_source, donor_plan, k, grid, prepare)
        if native_plan['schema'] != schema or bo.load_plan(source) != native_plan:
            raise ValueError('Candidate preparation changed the original policy encoding')
        benchmark = dict(source_plan_sha256=bo.sha(source), initial_actor_sha256=bo.sha(actor),
                         updates=a.updates)
        if observation.is_file():
            report = verify_candidate(bo, target, benchmark, conditions)
        else:
            report = bo.evaluate(a, target, source, native_plan, actor, design_id, conditions)
        design_ids = candidate_ids(k, grid, native_plan)
        if (tuple(report['candidate']['structure_key']) != k or report['candidate']['design_ids'] != design_ids
                or design_id != design_ids['stairs'] or report['ppo_updates'] != a.updates):
            raise ValueError('Measured candidate differs from the frozen exact proposal')
        if k not in grid:
            assets = target/'native_asset_evidence.json'
            checked = native_usd_evidence(bo,target,a.updates,design_ids)
            if observation.is_file():
                if bo.read(assets) != checked:
                    raise ValueError('Committed native USD evidence changed')
            else:
                bo.freeze(assets,checked)
        result = dict(proposal, design_id=design_id, evaluation=str(target/'bo_evaluation.json'),
                      **score_report(bo, report, scoring, conditions))
        bo.freeze(observation, result)
        results.append(result)
        append_observations(observations, result)
        seen.add(k)
        if quota is not None:
            counts[k[3]]+=1
    complete = [r for r in results if r['mean_objective'] is not None
                and r['complete_objectives'] == len(bo.required_panel_pairs(conditions))]
    best = min(complete, key=lambda r:r['mean_objective']) if complete else None
    output = work/f'budget_{a.trials:04d}.json'
    budget = dict(trials=len(results), best_observed=best, candidates=[summary(r) for r in results])
    if quota is not None:
        if (sum(counts.values()) != quota['new_trials'] or any(
                not quota['minimum_per_ankle'] <= n <= quota['maximum_per_ankle'] for n in counts.values())):
            raise ValueError('Completed BO ankle quota not satisfied')
        budget['ankle_quota'] = dict(policy=quota, completed_counts=counts)
    bo.freeze(output,budget)
    print('BO_CONTINUOUS_COMPLETE', output, flush=True)
