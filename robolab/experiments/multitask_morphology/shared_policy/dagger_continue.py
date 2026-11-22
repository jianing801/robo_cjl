"""Append one audited DAgger round to a completed experiment."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from contract import design_ids_for_keys, load_plan, sha, write_new
from dagger_comparison import summarize_comparison
from dagger_parent import load_parent
from persistence import resume_pause
from pipeline import (collect_round, fit_round, jobs_for, run_selection_suite)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from task_timer import start_task_timer

KIND = 'shared_dagger_continuation_v1'
RESULT = 'DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING'
# The initial continuation used 12 mmap slots even after adding more archives.
PRE_CACHE_FIX_SOURCES = {
    '361280a2887272284db59072aa3c961bb62371b59865146013798844eb884822',
    'ea271341b9f48f0bfd3f33e35b6411517f2fbd8672e4a2ccae422e99b0106845',
}
# All changed source identities must match one exact pre-long-replay runtime.
PRE_LONG_REPLAY_SOURCES = (
    {
        'dagger_continue.py': '361280a2887272284db59072aa3c961bb62371b59865146013798844eb884822',
        'dagger_parent.py': 'bc68b3e9e5a3021ca7f83d16aa774ef9e3537c8be5f1d988f072182d4afb5c4a',
        'pipeline.py': 'a5c2160468e01a8399fd2884f94f4f5c98daa1054a363819e52d5741b90c5000',
    },
    {
        'dagger_continue.py': 'ea271341b9f48f0bfd3f33e35b6411517f2fbd8672e4a2ccae422e99b0106845',
        'dagger_parent.py': 'b4e14457a26478a9c231b867933ac23160ad2daea9d7c532e24269b1f0266a6c',
        'pipeline.py': 'd86aef6775f2677da2c409149748e8dc09a38ae58f4388fda2f6c9b327a7eb7f',
    },
    {
        'dagger_continue.py': 'e4c4584eed80dcd77daee90ce4ca339969d675035e723a22900003e93ce6dcf3',
        'dagger_parent.py': 'bc68b3e9e5a3021ca7f83d16aa774ef9e3537c8be5f1d988f072182d4afb5c4a',
        'pipeline.py': 'a5c2160468e01a8399fd2884f94f4f5c98daa1054a363819e52d5741b90c5000',
    },
    {
        'dagger_continue.py': '83c4a9394678e271db9b7bd8a27b7d87861a54a7db743afdf21963e0bfde643a',
        'dagger_parent.py': 'b4e14457a26478a9c231b867933ac23160ad2daea9d7c532e24269b1f0266a6c',
        'pipeline.py': 'd86aef6775f2677da2c409149748e8dc09a38ae58f4388fda2f6c9b327a7eb7f',
    },
)


def read(path):
    return json.loads(Path(path).read_bytes())


def long_replay_spec():
    return dict(steps=1000, stride=5, tasks=['stairs', 'backpack', 'run'], priority_probability=1/3,
                sources={name: sha(HERE / name) for name in
                         ('dagger_long_collect.py', 'dagger_long_fit.py', 'stair_course.py')})


def same_continuation_manifest(saved, current):
    if saved == current:
        return True
    if 'long_replay' in current or 'long_replay' in saved:
        return False
    saved_source = saved.get('sources', {}).get('dagger_continue.py')
    old_sources = PRE_LONG_REPLAY_SOURCES
    if saved_source in PRE_CACHE_FIX_SOURCES:
        old_sources = (*old_sources, {'dagger_continue.py': saved_source})
    return any(saved == dict(current, sources={**current['sources'], **prior}) for prior in old_sources)


def freeze(path, value):
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.is_file():
        if read(path) != value:
            raise ValueError(f'Continuation evidence changed: {path}')
    else:
        write_new(path, value)


@contextmanager
def lease_environment(a):
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    choices = candidates(query_gpus(), os.environ, min_free_mib=a.gpu_min_free_gb * 1024,
                         max_utilization=a.gpu_max_utilization, requested=a.device)
    with gpu_lease(choices, HERE.parents[2] / 'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        yield child_environment(gpu, os.environ)


def compare_round(source, ids, teacher, previous, candidate, incumbent):
    protocol = source['manifest']['protocol']
    return summarize_comparison(round_index=candidate['round_index'], previous_report=previous,
        candidate_report=candidate, incumbent_report=incumbent, teacher=teacher,
        structure_keys=protocol['selection_keys'], training_keys=protocol['collection_scope']['structure_keys'],
        design_ids=ids)


def verify_selection_history(a, source, plan, ids, jobs, seeds, criteria):
    profile = a.selection_profile
    strength = source['manifest']['protocol']['selection']['strength']

    def archived_suite(root, student, suite):
        return run_selection_suite(plan, Path(a.plan), root, jobs, student, profile, strength,
            seeds, suite, None, a.timeout_seconds, criteria, reuse_only=True)

    teacher = archived_suite(source['base'], None, 'selection/teacher')['task_metrics']
    if teacher != source['selection']['teacher']:
        raise ValueError('Original teacher evaluation differs from the frozen selection')
    previous = incumbent = None
    for ancestor in source['lineage']:
        index = ancestor['round_index']
        suite = archived_suite(ancestor['root'], ancestor['student'], f'selection/round_{index}')
        measured = {**ancestor, **suite}
        if previous is None:
            recorded = next(row for row in source['selection']['candidates'] if row['round_index'] == index)
            if any(suite[field] != recorded[field] for field in ('task_metrics', 'by_seed', 'evaluations')):
                raise ValueError('Original round evaluation differs from the frozen selection')
            incumbent = measured
        else:
            recomputed = compare_round(source, ids, teacher, previous, measured, incumbent)
            if read(ancestor['root'] / 'comparison.json') != recomputed:
                raise ValueError(f'Committed comparison differs for round {index}')
            if recomputed['strictly_improves_incumbent']:
                incumbent = measured
        previous = measured
    return teacher, previous, incumbent


def execute_continuation(args):
    started_wall, started_monotonic = time.time(), time.monotonic()
    root = Path(args.output).resolve()
    parent = Path(args.continue_from).resolve()
    if root == parent or parent in root.parents or root in parent.parents:
        raise ValueError('Continuation requires a separate output directory')
    path = Path(args.plan).resolve()
    plan = load_plan(path)
    source = load_parent(parent, path, engine=HERE)
    base = source['base']
    if base in root.parents or root in base.parents:
        raise ValueError('Continuation requires a separate output directory from the original experiment')

    inherited = dict(source['manifest']['arguments'], plan=str(path), output=str(root),
                     continue_from=str(parent), resume=bool(args.resume or args.resume_paused),
                     resume_paused=args.resume_paused)
    if getattr(args, 'device', None) is not None:
        inherited['device'] = args.device
    a = SimpleNamespace(**inherited)
    index = source['round_index'] + 1
    protocol = source['manifest']['protocol']
    identity = dict(kind=KIND, round_index=index, parent=str(parent), base=str(base), plan_sha256=sha(path),
        parent_provenance=source['provenance'], collection_protocol=source['collection_protocol'],
        selection_protocol=protocol['selection'],
        sources={name: sha(HERE / name) for name in
                 ('dagger_continue.py', 'dagger_parent.py', 'dagger_comparison.py', 'pipeline.py')})
    long_replay = long_replay_spec() if getattr(args, 'long_replay', False) else None
    if long_replay is not None:
        identity['long_replay'] = long_replay
    manifest = root / 'manifest.json'
    if a.resume:
        saved = read(manifest) if manifest.is_file() else {}
        if not same_continuation_manifest(saved, identity):
            raise ValueError('Continuation source, protocol or code differs from its frozen identity')
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_new(manifest, identity)

    resume_pause(root, a.resume_paused)
    if (root / 'completed.json').is_file():
        verified = load_parent(root, path, engine=HERE)
        if verified['round_index'] != index:
            raise ValueError('Continuation completion belongs to another round')
        print('DAGGER_CONTINUATION_ALREADY_COMPLETE', index, verified['student'], flush=True)
        return

    start_task_timer(path.parent.parent, 'distill', root, None, started_wall, started_monotonic,
                     start_source='distill.py --continue-from')
    collection_keys = list(map(tuple, protocol['collection_scope']['structure_keys']))
    collection_ids = design_ids_for_keys(plan, collection_keys)
    collection_jobs = jobs_for(plan, 400, collection_ids)
    replay_options = {}
    if long_replay is not None:
        if [task for task, _ in collection_jobs] != long_replay['tasks']:
            raise ValueError('Expected one ordinary collection batch per task before long replay')
        collection_jobs = collection_jobs + collection_jobs
        replay_options['long_replay'] = long_replay
    selection_keys = list(map(tuple, protocol['selection_keys']))
    selection_ids = design_ids_for_keys(plan, selection_keys)
    selection_jobs = jobs_for(plan, len(selection_keys), selection_ids)
    seeds = protocol['selection_seeds']
    criteria = dict(duration_s=20., time_tolerance_s=.1, max_missing_tracking_steps=1,
                    min_command_progress_ratio=a.min_command_progress_ratio)
    teacher, previous, incumbent = verify_selection_history(a, source, plan, selection_ids,
                                                           selection_jobs, seeds, criteria)
    start_student = source['student']
    print('DAGGER_CONTINUATION_START', index, 'strength', 1.0, 'parent', parent, flush=True)

    with lease_environment(a) as environment:
        collected = collect_round(a, root, source['collector'], source['collection_metadata'],
                                  collection_jobs, index, 1., start_student, environment, **replay_options)
        replay = [*source['data_paths'], *collected]
        # Retain each verified archive's read-only mapping across balanced batches.
        a.replay_cache_shards = max(a.replay_cache_shards, len(replay))
        print('DAGGER_REPLAY_READY', len(replay), 'archives', a.replay_cache_shards, 'cache slots', flush=True)
        report = fit_round(a, root, source['collector'], index, start_student, replay, environment, **replay_options)
        assessment = run_selection_suite(plan, path, root, selection_jobs, report['student'],
            a.selection_profile, protocol['selection']['strength'], seeds, f'selection/round_{index}',
            environment, a.timeout_seconds, criteria)
        comparison = compare_round(source, selection_ids, teacher, previous, {**report, **assessment}, incumbent)
    if (root / 'USER_PAUSED').exists():
        raise KeyboardInterrupt('USER_PAUSED')

    path_comparison = root / 'comparison.json'
    freeze(path_comparison, comparison)
    completion = dict(kind=KIND, result=RESULT, round_index=index, parent=str(parent), base=str(base),
        plan_sha256=sha(path), manifest_sha256=sha(manifest), student=report['student'],
        student_sha256=report['student_sha256'], previous_student_sha256=sha(start_student),
        fit_summary=report['fit_summary'], fit_summary_sha256=sha(report['fit_summary']),
        data_list_sha256=sha(root / f'data_round_{index}.json'),
        comparison=str(path_comparison), comparison_sha256=sha(path_comparison))
    freeze(root / 'completed.json', completion)
    print('DAGGER_CONTINUATION_COMPLETE', index, 'selected_round', comparison['selected_round'],
          path_comparison, flush=True)
