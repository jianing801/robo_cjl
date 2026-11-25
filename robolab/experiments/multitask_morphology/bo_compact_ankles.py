"""Continue a completed 100-trial BO search for ten compact-ankle candidates.

The native coordinator, PPO implementation, scoring, and measurement protocols
are loaded from the exact frozen checkout used by the original search.
"""
import argparse
from importlib import import_module
from pathlib import Path
import sys


ALLOWED_ANKLES = frozenset({
    'DM-J4340P-2EC',
    'DM-J4340P-2EC-V1.1-48V',
    'DM-J6216P-2EC',
})
INITIAL_PANEL_SIZE = 50
HISTORY_SIZE = 100
TOTAL_SIZE = 110
SUMMARY_KEYS = ('design_id', 'structure_key', 'evaluation', 'valid_pairs',
                'joint_successes', 'complete_objectives', 'mean_objective')


def propose(bo, pool, observations, evaluated, schema, seed, *, native_propose=None):
    """Keep the original global EI and restrict only previously unseen designs."""
    # Evaluated IDs are kept in the pool so the native coordinator retains its
    # exact historical distance and surrogate behavior. Every observation,
    # including those with excluded motors, participates in the original fit.
    filtered = [r for r in pool if r['design_ids']['stairs'] in evaluated
                or r['key'][3] in ALLOWED_ANKLES]
    row, audit = (native_propose or bo.propose)(filtered, observations, evaluated, schema, seed)
    if row['key'][3] not in ALLOWED_ANKLES or row['design_ids']['stairs'] in evaluated:
        raise RuntimeError('Native BO proposed an excluded or already evaluated design')
    incumbent = min((o['objective'] for o in observations
                     if o['feasible'] and o.get('objective') is not None), default=None)
    return row, dict(audit, global_incumbent=incumbent, allowed_ankles=sorted(ALLOWED_ANKLES))

def require_receipt(path):
    if not path.is_file():
        raise ValueError(f'Historical receipt missing; refusing to recreate old BO evidence: {path}')
    return path


def check_history(bo, work, conditions):
    """Pin the old budget and all existing commits before native replay."""
    budget_path = require_receipt(work/'budget_0100.json')
    manifest_path = require_receipt(work/'manifest.json')
    manifest = bo.read(manifest_path)
    if (manifest['coordinator_sha256'] != bo.sha(Path(bo.__file__))
            or manifest['surrogate_sha256'] != bo.sha(bo.REPO/'rough_bo/constrained_gp.py')):
        raise ValueError('Use the original frozen code checkout for the completed 100-trial search')
    budget = bo.read(budget_path)
    entries = budget['candidates']
    if budget['trials'] != HISTORY_SIZE or len(entries) != HISTORY_SIZE:
        raise ValueError('Expected a completed 100-observation BO budget')
    ids = {row['design_id'] for row in entries}
    if len(ids) != HISTORY_SIZE:
        raise ValueError('Historical BO budget has duplicate designs')
    paths = [budget_path, manifest_path]
    historic = {}
    for i, row in enumerate(entries):
        report = require_receipt(Path(row['evaluation']))
        if i < INITIAL_PANEL_SIZE:
            # Native search additionally checks the calibration panel and its
            # frozen scoring/measurement hashes; no initial trial is rerun.
            paths.append(report)
            continue
        trial = work/f'trial_{i:04d}'
        candidate = trial/'candidate'
        proposal_path = require_receipt(trial/'proposal.json')
        observation_path = require_receipt(trial/'observation.json')
        proposal = bo.read(proposal_path)
        observation = bo.read(observation_path)
        if (report.resolve() != (candidate/'bo_evaluation.json').resolve()
                or any(observation.get(k) != row[k] for k in SUMMARY_KEYS)
                or proposal['design_id'] != row['design_id']
                or proposal['structure_key'] != row['structure_key']):
            raise ValueError(f'Historical BO budget does not match committed trial {i+1}')
        if (candidate/'USER_PAUSED').exists():
            raise RuntimeError(f'Historical candidate is paused: {candidate}')
        tuning = candidate/'tuning'
        commits = [proposal_path, observation_path, report,
                   candidate/'manifest.json', tuning/'manifest.json',
                   bo.actor_receipt(tuning)]
        commits.extend(candidate/'measurements'/f'{seed}_{task}'/'accepted.json'
                       for seed in conditions['seeds'] for task in bo.TASKS)
        paths.extend(require_receipt(path) for path in commits)
        historic[candidate.resolve()] = row['design_id']
    snapshot = {str(path.relative_to(work) if path.is_relative_to(work) else path): bo.sha(path)
                for path in paths}
    return ids, historic, snapshot


def check_future(bo, work, seen):
    """Reject out-of-scope or repeated frozen proposals before any computation."""
    seen = set(seen)
    for i in range(HISTORY_SIZE, TOTAL_SIZE):
        trial = work/f'trial_{i:04d}'
        request = trial/'proposal.json'
        if not request.is_file():
            if (trial/'candidate').exists():
                raise ValueError(f'Frozen extension proposal missing for existing candidate: {trial}')
            continue
        proposal = bo.read(request)
        if (proposal['structure_key'][3] not in ALLOWED_ANKLES
                or proposal['design_id'] in seen):
            raise ValueError(f'Frozen extension proposal is disallowed or repeated: {request}')
        seen.add(proposal['design_id'])


def search(bo, a, work, source, plan, actor, conditions, *, native_search=None):
    """Run only inside the original coordinator's process-lifetime lock."""
    if a.trials != TOTAL_SIZE:
        raise ValueError('Compact-ankle extension has a fixed total budget of 110')
    work = work.resolve()
    seen, historic, snapshot = check_history(bo, work, conditions)
    check_future(bo, work, seen)
    bo.freeze(work/'extension_0100_0110.json', dict(
        kind='compact_ankle_bo_extension_v1', previous_budget=HISTORY_SIZE,
        total_budget=TOTAL_SIZE, allowed_ankles=sorted(ALLOWED_ANKLES),
        acquisition_incumbent='global_best_feasible',
        historical_commits_sha256=snapshot, wrapper_sha256=bo.sha(Path(__file__))))
    native_search = native_search or bo.search
    native_propose, native_evaluate = bo.propose, bo.evaluate
    native_worker = bo.run_worker

    def forbid_worker(*args, **kwargs):
        raise RuntimeError('Historical BO replay is read-only; refusing to run a worker')

    def evaluate_guarded(a, folder, source, plan, actor, design_id, conditions):
        if folder.resolve() not in historic:
            # The native search verifies proposal integrity and uniqueness;
            # check the new driver's compact-ankle constraint as well.
            if folder.resolve() not in {(work/f'trial_{i:04d}/candidate').resolve()
                                        for i in range(HISTORY_SIZE, TOTAL_SIZE)}:
                raise RuntimeError(f'Unexpected BO evaluation directory: {folder}')
            proposal = bo.read(folder.parent/'proposal.json')
            if (proposal['structure_key'][3] not in ALLOWED_ANKLES
                    or proposal['design_id'] != design_id):
                raise ValueError(f'Frozen extension proposal is disallowed: {folder.parent}')
            return native_evaluate(a, folder, source, plan, actor, design_id, conditions)
        if historic[folder.resolve()] != design_id or (folder/'USER_PAUSED').exists():
            raise ValueError('Historical BO candidate identity or pause marker changed')
        bo.run_worker = forbid_worker
        try:
            # The unmodified native evaluator verifies frozen identities,
            # checkpoint receipts and raw-trace hashes, reusing every result.
            return native_evaluate(a, folder, source, plan, actor, design_id, conditions)
        finally:
            bo.run_worker = native_worker

    bo.propose = lambda *args: propose(bo, *args, native_propose=native_propose)
    bo.evaluate = evaluate_guarded
    try:
        return native_search(a, work, source, plan, actor, conditions)
    finally:
        bo.propose, bo.evaluate, bo.run_worker = native_propose, native_evaluate, native_worker


def frozen_coordinator(root):
    root = root.resolve()
    entry = root/'robolab/experiments/multitask_morphology/bo.py'
    if not entry.is_file():
        raise ValueError(f'Frozen native BO coordinator not found: {entry}')
    # This is deliberately done before importing any repository modules.
    sys.path.insert(0, str(entry.parent))
    bo = import_module('bo')
    if Path(bo.__file__).resolve() != entry.resolve() or Path(bo.REPO).resolve() != root:
        raise RuntimeError('BO coordinator did not load from the requested frozen checkout')
    return bo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--frozen-code-root', type=Path, required=True)
    args, native_args = parser.parse_known_args(argv)
    bo = frozen_coordinator(args.frozen_code_root)
    native_search = bo.search
    # Native main owns CLI validation and the exclusive run lock. Our search
    # wrapper is entered only after that lock has been acquired.
    bo.search = lambda *args: search(bo, *args, native_search=native_search)
    try:
        return bo.main(['search', *native_args])
    finally:
        bo.search = native_search


if __name__ == '__main__':
    main()
