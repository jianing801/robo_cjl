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


def read(path):
    return json.loads(Path(path).read_bytes())


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
    manifest = root / 'manifest.json'
    if a.resume:
        if not manifest.is_file() or read(manifest) != identity:
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
                                  collection_jobs, index, 1., start_student, environment)
        report = fit_round(a, root, source['collector'], index, start_student,
                           [*source['data_paths'], *collected], environment)
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
