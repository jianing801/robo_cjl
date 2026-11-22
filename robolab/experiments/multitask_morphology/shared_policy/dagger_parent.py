"""Validate completed DAgger round receipts without modifying their runs."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import runpy
import subprocess

from collection_grid import SELECTION_VERSION
from contract import sha
from evaluation import COMPARISON_VERSION, EVALUATION_VELOCITY_FRAME
from shared_bo_metrics import TASKS, digest
from stair_course import COURSE
from stair_resume import EVALUATION_FIELDS, PREVIOUS_REVISION, SELECTION_MANIFEST_FIELDS


CONTINUATION_KIND = 'shared_dagger_continuation_v1'
RESULT = 'DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING'
BASE_ROUND = 3
EPOCHS = 10
CURRICULUM = (.25, .5, .75, 1.)
ANCHOR_SEEDS = [20261117, 20261119, 20261120]
SHARED_ENGINE = Path('robolab/experiments/multitask_morphology/shared_policy')
ROLLOUT_SOURCES = {'rollout.py', 'rollout_amp.py', 'scenario_sampling.py', 'physical_audit.py'}
PRE_CONTINUATION_REVISION = '58fd4476d5694decbce288d61a7584a305d7cbee'
PRE_TASK_SUCCESS_SELECTION_REVISION = '103bbd0be31ef7903d714cbd29326c6412653db5'
LONG_REPLAY_SOURCES = {'dagger_long_collect.py', 'dagger_long_fit.py', 'stair_course.py'}


def _read(path):
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'Expected a regular receipt at {path}')
    return json.loads(path.read_bytes())


def _object(path):
    obj = _read(path)
    if not isinstance(obj, dict):
        raise ValueError(f'Expected a JSON object at {path}')
    return obj


def _identity(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _verified(path, expected):
    if not _identity(expected) or path.resolve() != path or not path.is_file() or sha(path) != expected:
        raise ValueError(f'DAGger evidence changed or is missing: {path}')
    return expected


def _path(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f'Expected an absolute, resolved DAgger path: {path}')
    return path


def _protocol(manifest):
    protocol = manifest['protocol']
    if not isinstance(protocol, dict) or manifest.get('protocol_sha256') != digest(protocol):
        raise ValueError('DAgger manifest protocol digest changed')
    return protocol


def _verify_selection_coordinator(engine, expected):
    path = engine/'pipeline.py'
    if not _identity(expected) or path.resolve() != path or not path.is_file():
        raise ValueError(f'Invalid DAgger coordinator source: {path}')
    if sha(path) == expected:
        return
    for revision in (PRE_CONTINUATION_REVISION, PRE_TASK_SUCCESS_SELECTION_REVISION):
        previous = subprocess.check_output([
            'git', '--no-optional-locks', '-C', str(engine), 'show',
            f'{revision}:{(SHARED_ENGINE/"pipeline.py").as_posix()}',
        ])
        if expected in (sha256(previous).hexdigest(), sha256(previous.replace(b'\n', b'\r\n')).hexdigest()):
            return
    raise ValueError('The DAgger selection coordinator does not match its recorded source')


def _runtime(engine, protocol, purpose, *, archived=False):
    rollout_sources = protocol['rollout_implementation']
    expected = ROLLOUT_SOURCES if archived else ROLLOUT_SOURCES | {'stair_course.py'}
    if set(rollout_sources) != expected:
        raise ValueError('Incomplete DAgger rollout source identities')
    sources = {
        **rollout_sources,
        'pipeline.py': protocol['implementation_sha256'],
        'evaluation.py': protocol['evaluation_implementation_sha256'],
        'fit.py': protocol['fit_implementation_sha256'],
        'collection_grid.py': protocol['collection_scope']['selector_sha256'],
    }
    if 'selection_implementation_sha256' in protocol:
        sources['dagger_comparison.py'] = protocol['selection_implementation_sha256']
    for name, expected_hash in sources.items():
        if name == 'pipeline.py' and not archived:
            _verify_selection_coordinator(engine, expected_hash)
        else:
            _verified(engine/name, expected_hash)
    randomization = protocol[purpose]
    source_names = expected - {'physical_audit.py'} | {'randomization.py', 'student_observations.py'}
    if set(randomization['source_sha256']) != source_names:
        raise ValueError(f'Incomplete DAgger {purpose} protocol source identities')
    for name, expected_hash in randomization['source_sha256'].items():
        _verified(engine/name, expected_hash)
    # Executing the hash-verified metadata source directly avoids bytecode writes
    # to the completed run's frozen worktree.
    metadata = runpy.run_path(str(engine/'randomization.py'))['protocol_metadata']
    if metadata('robust', 1.) != randomization:
        raise ValueError(f'DAgger {purpose} protocol does not match its runtime')
    return metadata, {**sources, **randomization['source_sha256']}


def _collection_runtime(parent, engine, manifest):
    history = parent/'stair_evaluation_upgrade'
    archive = history/'original_manifest.json'
    if not archive.exists() and not history.exists():
        metadata, sources = _runtime(engine, manifest['protocol'], 'collection')
        return engine, metadata, sources, {}
    original = _object(archive)
    original_protocol = _protocol(original)
    current_protocol = manifest['protocol']
    ignored = EVALUATION_FIELDS | {'stair_evaluation'}
    if ('stair_evaluation' in original_protocol or
            {k: v for k, v in original_protocol.items() if k not in ignored} !=
            {k: v for k, v in current_protocol.items() if k not in ignored} or
            {k: v for k, v in original.items() if k not in SELECTION_MANIFEST_FIELDS} !=
            {k: v for k, v in manifest.items() if k not in SELECTION_MANIFEST_FIELDS}):
        raise ValueError('Archived DAgger manifest changed outside the stair evaluation upgrade')
    checkout = _path(history/'collection_checkout')

    def git(*args):
        return subprocess.check_output(
            ['git', '--no-optional-locks', '-C', str(checkout), *args], text=True,
        ).strip()

    if (Path(git('rev-parse', '--show-toplevel')).resolve() != checkout
            or git('rev-parse', 'HEAD') != PREVIOUS_REVISION
            or git('status', '--porcelain', '--untracked-files=normal')):
        raise ValueError('Frozen DAgger training checkout changed')
    collector = checkout/SHARED_ENGINE
    metadata, sources = _runtime(collector, original_protocol, 'collection', archived=True)
    # The archived selection was also tied to this implementation.
    if metadata('robust', 1.) != original_protocol['selection']:
        raise ValueError('Archived DAgger selection protocol does not match the frozen checkout')
    return collector, metadata, sources, {
        'original_manifest_sha256': sha(archive), 'collection_checkout_revision': PREVIOUS_REVISION,
    }


def _long_replay_spec(manifest):
    if 'long_replay' not in manifest:
        return None
    spec = manifest['long_replay']
    settings = dict(steps=1000, stride=5, tasks=list(TASKS), priority_probability=1/3)
    if (not isinstance(spec, dict) or set(spec) != set(settings) | {'sources'}
            or any(spec.get(key) != value for key, value in settings.items())):
        raise ValueError('Invalid DAgger long replay settings')
    sources = spec['sources']
    if (not isinstance(sources, dict) or set(sources) != LONG_REPLAY_SOURCES
            or any(not _identity(value) for value in sources.values())):
        raise ValueError('Incomplete DAgger long replay source identities')
    return spec


def _long_collection(folder, commit, task, seed, design_ids, spec):
    variant = dict(kind='long_replay', steps=spec['steps'], seed=seed, sources=spec['sources'])
    if commit.get('variant') != variant:
        raise ValueError(f'DAgger long collection variant changed at {folder}')
    summary_path = folder/'summary.json'
    summary_sha = _verified(summary_path, commit['summary_sha256'])
    summary = _object(summary_path)
    result = 'REAL_AMP_ROLLOUT_COMPLETE' if task == 'run' else 'REAL_ROLLOUT_COMPLETE'
    expected = dict(result=result, task=task, steps=spec['steps'], seed=seed, design_ids=design_ids,
                    student=commit['execution_student_sha256'], collection_protocol=commit['protocol'],
                    teacher_updates=0)
    if (any(summary.get(key) != value for key, value in expected.items())
            or ('data' in summary and _path(summary['data']) != folder/'data.pt')):
        raise ValueError(f'DAgger long collection summary differs at {folder}')
    audit_path = folder/'long_replay_collection.json'
    _verified(audit_path, commit['collection_audit_sha256'])
    audit = _object(audit_path)
    expected = dict(kind='dagger_long_collection_v1', task=task, steps=spec['steps'], stride=spec['stride'],
                    seed=seed, terrain='continuous_bottom_start' if task == 'stairs' else 'native',
                    summary_sha256=summary_sha)
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f'DAgger long collection audit differs at {folder}')


def _replay(root, index, inherited, previous_actor, collection_protocol, *, long_replay=None, design_ids=None):
    directory = root/'data_commits'/f'round_{index}'
    tasks = (*TASKS, *long_replay['tasks']) if long_replay else TASKS
    jobs = [f'{job:04d}_{task}.json' for job, task in enumerate(tasks)]
    if sorted(path.name for path in directory.glob('*.json')) != jobs:
        raise ValueError(f'Collection commits differ from the declared jobs for DAgger round {index}')
    paths = inherited.copy()
    commits = []
    for job_index, (task, filename) in enumerate(zip(tasks, jobs)):
        commit_path = directory/filename
        commit = _object(commit_path)
        folder = _path(commit['folder'])
        job = commit_path.stem
        task_output = root/'data'/f'round_{index}'/job
        if (folder.parent != task_output or not folder.name.startswith('attempt_')
                or (folder/'failure.json').exists()
                or type(commit.get('round_index')) is not int or commit['round_index'] != index
                or commit.get('protocol') != collection_protocol
                or 'execution_student_sha256' not in commit or commit['execution_student_sha256'] != previous_actor):
            raise ValueError(f'DAgger collection commit does not match round {index}, task {job}')
        data = folder/'data.pt'
        _verified(data, commit['sha256'])
        if job_index >= len(TASKS):
            _long_collection(folder, commit, task, 42 + index*10000 + job_index,
                             design_ids[task], long_replay)
        elif 'variant' in commit:
            raise ValueError(f'Unexpected DAgger collection variant in round {index}, job {job}')
        paths.append(str(data))
        commits.append(sha(commit_path))
    data_list = root/f'data_round_{index}.json'
    if _read(data_list) != paths:
        raise ValueError(f'DAgger round {index} replay list is missing data or out of order')
    return paths, sha(data_list), commits


def _long_fit_identity(root, index, paths, spec):
    priority_data = root/f'long_replay_round_{index}.json'
    if _read(priority_data) != paths[-len(spec['tasks']):]:
        raise ValueError(f'DAgger round {index} long priority replay list differs from its long collection jobs')
    return dict(priority_data_sha256=sha(priority_data), sources=spec['sources'])


def _long_sampling(summary_path, receipt, spec):
    audit_path = summary_path.with_name('long_replay_sampling.json')
    _verified(audit_path, receipt['sampling_audit_sha256'])
    audit = _object(audit_path)
    tasks = audit.get('tasks')
    if (audit.get('kind') != 'dagger_long_sampling_v1'
            or audit.get('priority_data_sha256') != receipt['identity']['long_replay']['priority_data_sha256']
            or audit.get('priority_probability') != spec['priority_probability']
            or not isinstance(tasks, dict) or set(tasks) != set(spec['tasks'])):
        raise ValueError(f'DAgger long sampling audit differs at {audit_path}')
    for counts in tasks.values():
        if (not isinstance(counts, dict) or type(counts.get('priority_rows')) is not int
                or type(counts.get('total_rows')) is not int
                or not 0 < counts['priority_rows'] < counts['total_rows']):
            raise ValueError(f'DAgger long sampling audit is missing priority or ordinary rows at {audit_path}')


def _fit(root, index, previous_actor, data_list_sha, implementation, *, long_replay=None, paths=None):
    path = root/f'fit_round_{index}.json'
    receipt = _object(path)
    identity = dict(data_list_sha256=data_list_sha, init_student_sha256=previous_actor,
                    implementation_sha256=implementation)
    if long_replay:
        identity['long_replay'] = _long_fit_identity(root, index, paths, long_replay)
    summary_path = _path(receipt['summary'])
    if (receipt.get('identity') != identity or summary_path.name != 'summary.json'
            or summary_path.parent.parent != root/f'bc_round_{index}'
            or not summary_path.parent.name.startswith('attempt_')):
        raise ValueError(f'DAgger round {index} fit inputs or runtime differ')
    _verified(summary_path, receipt['summary_sha256'])
    if long_replay:
        _long_sampling(summary_path, receipt, long_replay)
    summary = _object(summary_path)
    checkpoints = summary['checkpoints']
    if (summary.get('result') != 'BC_COMPLETE_NOT_GAIT_VALIDATED'
            or summary.get('closed_loop_required') is not True
            or sorted(row['epoch'] for row in checkpoints) != list(range(EPOCHS))
            or any(type(row['epoch']) is not int for row in checkpoints)):
        raise ValueError(f'DAgger round {index} is missing its fit checkpoints')
    last = next(row for row in checkpoints if row['epoch'] == EPOCHS - 1)
    actor = summary_path.parent/f'student_epoch_{EPOCHS - 1:04d}.pt'
    if (Path(last['student']) != actor or Path(summary['student']) != actor
            or summary['student_sha256'] != last['student_sha256']):
        raise ValueError(f'DAgger round {index} final actor differs from its fit summary')
    actor_sha = _verified(actor, last['student_sha256'])
    report = dict(round_index=index, epoch=EPOCHS - 1, student=str(actor), student_sha256=actor_sha,
                  fit_summary=str(summary_path))
    return report, sha(path), receipt['summary_sha256']


def _base_selection(parent, manifest, completion, rounds):
    path = parent/'selection.json'
    if Path(completion['selection']) != path:
        raise ValueError('The completed DAgger run refers to a different selection')
    _verified(path, completion['selection_sha256'])
    selection = _object(path)
    protocol = manifest['protocol']
    candidates = selection['candidates']
    if (selection.get('rule') != manifest['selection_rule'] or selection.get('profile') != 'robust'
            or selection.get('seeds') != protocol['selection_seeds']
            or selection.get('comparison_version') != protocol['comparison_version']
            or len(candidates) != BASE_ROUND):
        raise ValueError('The completed DAgger selection protocol differs')
    for candidate, expected in zip(candidates, rounds[1:]):
        rank = candidate.get('rank')
        if (any(candidate.get(k) != v for k, v in expected.items())
                or not isinstance(rank, list) or not rank
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in rank)):
            raise ValueError('A DAgger selection candidate differs from the verified final epoch')
    winner = max(candidates, key=lambda candidate: tuple(candidate['rank']))
    if (selection.get('selected_round') != winner['round_index']
            or completion.get('selected_round') != winner['round_index']
            or selection.get('selected_student') != winner['student']
            or completion.get('student') != winner['student']
            or completion.get('student_sha256') != winner['student_sha256']):
        raise ValueError('DAgger continuation requires the verified base selection winner')
    return selection


def _base(parent, plan_sha, engine, manifest, completion, provenance):
    protocol = _protocol(manifest)
    arguments = manifest['arguments']
    scope = protocol['collection_scope']
    if ('long_replay' in manifest or manifest.get('plan_sha256') != plan_sha or completion.get('protocol') != protocol
            or completion.get('protocol_sha256') != manifest['protocol_sha256']
            or any(completion.get(field) is not value for field, value in {
                'full_bank_evaluated': False, 'critic_trained': False,
                'requires_candidate_finetuning': True, 'requires_post_finetuning_evaluation': True,
            }.items())
            or completion.get('teacher_updates') != 0 or completion.get('ppo_updates') != 0
            or protocol.get('dagger_rounds') != BASE_ROUND or arguments.get('dagger_rounds') != BASE_ROUND
            or protocol.get('initial_strength') != CURRICULUM[0]
            or arguments.get('initial_randomization_strength') != CURRICULUM[0]
            or arguments.get('randomization_strength') != CURRICULUM[-1]
            or arguments.get('collection_profile') != 'robust' or arguments.get('epochs') != EPOCHS
            or scope.get('batches_per_round') != len(TASKS)
            or set(scope['design_ids_by_task']) != set(TASKS)
            or protocol.get('selection_version') != SELECTION_VERSION
            or protocol.get('selection_seeds') != ANCHOR_SEEDS
            or protocol.get('stair_evaluation') != COURSE
            or protocol.get('evaluation_velocity_frame') != EVALUATION_VELOCITY_FRAME
            or protocol.get('comparison_version') != COMPARISON_VERSION):
        raise ValueError('Expected the completed robust three-round DAgger plan and selection protocol')

    _, selection_sources = _runtime(engine, protocol, 'selection')
    collector, metadata, collector_sources, archive = _collection_runtime(parent, engine, manifest)
    paths, rounds, data_commits, fit_commits = [], [], [], []
    previous = None
    for index, strength in enumerate(CURRICULUM):
        paths, data_list_sha, commits = _replay(parent, index, paths, previous, metadata('robust', strength))
        # The original coordinator recorded its own fit hash. The archived
        # protocol above additionally verifies the actual frozen worker.
        fit, fit_commit_sha, fit_summary_sha = _fit(
            parent, index, previous, data_list_sha, protocol['fit_implementation_sha256'],
        )
        rounds.append(fit)
        data_commits.extend(commits)
        fit_commits.append(fit_commit_sha)
        previous = fit['student_sha256']
    if completion.get('rounds') != rounds:
        raise ValueError('Completed DAgger rounds differ from their replay or fit receipts')
    selection = _base_selection(parent, manifest, completion, rounds)
    last = rounds[-1]
    return dict(
        manifest=manifest, completion=completion, selection=selection, base=parent, round_index=BASE_ROUND,
        lineage=[dict(round_index=BASE_ROUND, root=parent, student=last['student'], student_sha256=previous)],
        data_paths=paths, student=last['student'], collector=collector,
        collection_protocol=protocol['collection'], collection_metadata=metadata,
        provenance=dict(provenance, **archive, selection_sha256=completion['selection_sha256'],
                        student_sha256=previous, data_list_sha256=data_list_sha, data_commits_sha256=data_commits,
                        fit_commit_sha256=fit_commit_sha, fit_commits_sha256=fit_commits,
                        fit_summary_sha256=fit_summary_sha, collection_protocol_sha256=digest(protocol['collection']),
                        collector_sources=collector_sources, selection_sources=selection_sources),
    )


def _extension(parent, ancestor, manifest, completion, provenance):
    index = ancestor['round_index'] + 1
    previous = ancestor['lineage'][-1]
    collection = ancestor['collection_protocol']
    identity = dict(kind=CONTINUATION_KIND, round_index=index, parent=str(previous['root']),
                    base=str(ancestor['base']), plan_sha256=provenance['plan_sha256'])
    sources = manifest.get('sources')
    if (any(manifest.get(k) != v or completion.get(k) != v for k, v in identity.items())
            or completion.get('manifest_sha256') != provenance['manifest_sha256']
            or completion.get('previous_student_sha256') != previous['student_sha256']
            or manifest.get('collection_protocol') != collection
            or manifest.get('selection_protocol') != ancestor['manifest']['protocol']['selection']
            or not isinstance(sources, dict) or not sources or any(not _identity(v) for v in sources.values())):
        raise ValueError(f'Stale or invalid DAgger continuation manifest at {parent}')
    if manifest.get('parent_provenance') != ancestor['provenance']:
        raise ValueError(f'DAgger parent provenance changed at {parent}')

    long_replay = _long_replay_spec(manifest)
    design_ids = ancestor['manifest']['protocol']['collection_scope']['design_ids_by_task']
    paths, data_list_sha, commits = _replay(
        parent, index, ancestor['data_paths'], previous['student_sha256'], collection,
        long_replay=long_replay, design_ids=design_ids,
    )
    inherited = ancestor['provenance']
    fit, fit_commit_sha, fit_summary_sha = _fit(
        parent, index, previous['student_sha256'], data_list_sha, inherited['collector_sources']['fit.py'],
        long_replay=long_replay, paths=paths,
    )
    comparison = parent/'comparison.json'
    if (completion.get('data_list_sha256') != data_list_sha
            or completion.get('fit_summary_sha256') != fit_summary_sha
            or any(completion.get(k) != fit[k] for k in ('student', 'student_sha256', 'fit_summary'))
            or completion.get('comparison') != str(comparison)):
        raise ValueError(f'DAgger round {index} completion differs from its replay, actor or comparison')
    _verified(comparison, completion['comparison_sha256'])
    assessment = _object(comparison)
    return dict(
        ancestor, completion=completion, round_index=index, data_paths=paths, student=fit['student'],
        previous_assessment=assessment,
        lineage=[*ancestor['lineage'], dict(round_index=index, root=parent, student=fit['student'],
                                           student_sha256=fit['student_sha256'])],
        provenance=dict(inherited, **provenance, student_sha256=fit['student_sha256'],
                        data_list_sha256=data_list_sha, data_commits_sha256=inherited['data_commits_sha256'] + commits,
                        fit_commit_sha256=fit_commit_sha, fit_commits_sha256=[*inherited['fit_commits_sha256'], fit_commit_sha],
                        fit_summary_sha256=fit_summary_sha, comparison_sha256=completion['comparison_sha256'],
                        parent_completion_sha256=inherited['completion_sha256']),
    )


def load_parent(parent: Path, plan_path: Path, *, engine: Path) -> dict:
    """Return the verified immediate parent, original selection, and ordered cumulative replay."""
    parent = Path(parent).resolve()
    try:
        plan_sha = sha(plan_path)
        engine = Path(engine).resolve()
        extensions = []
        visited = set()
        while True:
            if parent in visited:
                raise ValueError(f'Cycle in DAgger continuation lineage at {parent}')
            visited.add(parent)
            manifest_path, completion_path = parent/'manifest.json', parent/'completed.json'
            manifest, completion = _object(manifest_path), _object(completion_path)
            if completion.get('result') != RESULT:
                raise ValueError(f'DAgger run is missing a completed evaluation at {parent}')
            provenance = dict(manifest_sha256=sha(manifest_path), completion_sha256=sha(completion_path),
                              plan_sha256=plan_sha)
            if completion.get('kind') is None:
                verified = _base(parent, plan_sha, engine, manifest, completion, provenance)
                break
            if completion['kind'] != CONTINUATION_KIND:
                raise ValueError(f'Unknown DAgger continuation receipt at {parent}')
            extensions.append((parent, manifest, completion, provenance))
            parent = _path(completion['parent'])
        for root, manifest, completion, provenance in reversed(extensions):
            verified = _extension(root, verified, manifest, completion, provenance)
        return verified
    except (AttributeError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as exc:
        raise ValueError(f'Incomplete or invalid DAgger evidence at {parent}: {exc}') from exc
