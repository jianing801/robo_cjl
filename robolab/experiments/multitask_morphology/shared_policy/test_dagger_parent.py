"""Read-only DAgger parent and replay lineage checks."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collection_grid import SELECTION_VERSION
from contract import sha
from evaluation import COMPARISON_VERSION, EVALUATION_VELOCITY_FRAME
from randomization import protocol_metadata
from shared_bo_metrics import TASKS, digest
from stair_course import COURSE


RESULT = 'DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING'
KIND = 'shared_dagger_continuation_v1'
SEEDS = [20261117, 20261119, 20261120]


def load_parent(root, plan):
    return importlib.import_module('dagger_parent').load_parent(root, plan, engine=HERE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')
    return path


def read(path):
    return json.loads(path.read_bytes())


def collect(root, index, protocol, previous_actor, *, long_replay=None):
    data_paths = []
    tasks = TASKS * (2 if long_replay else 1)
    for job, task in enumerate(tasks):
        name = f'{job:04d}_{task}'
        folder = root/'data'/f'round_{index}'/name/'attempt_fixture'
        folder.mkdir(parents=True)
        blob = folder/'data.pt'
        blob.write_bytes(f'{index}-{task}'.encode())
        data_paths.append(str(blob))
        commit = dict(folder=str(folder), sha256=sha(blob), protocol=protocol,
                      round_index=index, execution_student_sha256=previous_actor)
        if job >= len(TASKS):
            seed = 42 + index*10000 + job
            result = 'REAL_AMP_ROLLOUT_COMPLETE' if task == 'run' else 'REAL_ROLLOUT_COMPLETE'
            summary = write(folder/'summary.json', dict(result=result, task=task, steps=1000,
                            seed=seed, collection_protocol=protocol, student=previous_actor,
                            teacher_updates=0, design_ids=[task]))
            audit = write(folder/'long_replay_collection.json',
                          dict(kind='dagger_long_collection_v1', task=task, steps=1000, stride=5, seed=seed,
                               terrain='continuous_bottom_start' if task == 'stairs' else 'native',
                               summary_sha256=sha(summary)))
            commit.update(variant=dict(kind='long_replay', steps=1000, seed=seed,
                                       sources=long_replay['sources']),
                          summary_sha256=sha(summary), collection_audit_sha256=sha(audit))
        write(root/'data_commits'/f'round_{index}'/f'{name}.json', commit)
    return data_paths


def fit(root, index, paths, previous_actor, implementation, *, long_replay=None):
    data_list = write(root/f'data_round_{index}.json', paths)
    output = root/f'bc_round_{index}'/'attempt_fixture'
    output.mkdir(parents=True)
    checkpoints = []
    for epoch in range(10):
        actor = output/f'student_epoch_{epoch:04d}.pt'
        actor.write_bytes(f'{index}-{epoch}'.encode())
        checkpoints.append(dict(epoch=epoch, student=str(actor), student_sha256=sha(actor)))
    summary = write(output/'summary.json',
                    dict(result='BC_COMPLETE_NOT_GAIT_VALIDATED', checkpoints=checkpoints,
                         closed_loop_required=True, student=str(actor), student_sha256=sha(actor)))
    receipt = dict(summary=str(summary), summary_sha256=sha(summary),
                   identity=dict(data_list_sha256=sha(data_list), init_student_sha256=previous_actor,
                                 implementation_sha256=implementation))
    if long_replay:
        priority = write(root/f'long_replay_round_{index}.json', paths[-3:])
        receipt['identity']['long_replay'] = dict(priority_data_sha256=sha(priority),
                                                 sources=long_replay['sources'])
        audit = write(output/'long_replay_sampling.json',
                      dict(kind='dagger_long_sampling_v1', priority_data_sha256=sha(priority),
                           priority_probability=1/3,
                           tasks={task: dict(priority_rows=1500, total_rows=2000) for task in TASKS}))
        receipt['sampling_audit_sha256'] = sha(audit)
    write(root/f'fit_round_{index}.json', receipt)
    return dict(round_index=index, epoch=9, student=str(actor), student_sha256=sha(actor),
                fit_summary=str(summary))


@pytest.fixture
def parent(tmp_path):
    root = tmp_path.resolve()/'shared_policy'
    plan = write(tmp_path/'plan.json', {'plan': 'fixture'})
    fit_sha = sha(HERE/'fit.py')
    base_protocol = protocol_metadata('robust', 1.)
    protocol = dict(
        collection=base_protocol, selection=base_protocol, dagger_rounds=3, initial_strength=.25,
        collection_scope=dict(batches_per_round=3, design_ids_by_task={task: [task] for task in TASKS},
                              selector_sha256=sha(HERE/'collection_grid.py')),
        selection_seeds=SEEDS, selection_version=SELECTION_VERSION, comparison_version=COMPARISON_VERSION,
        selection_budget={}, stair_evaluation=COURSE, evaluation_velocity_frame=EVALUATION_VELOCITY_FRAME,
        implementation_sha256=sha(HERE/'pipeline.py'), fit_implementation_sha256=fit_sha,
        evaluation_implementation_sha256=sha(HERE/'evaluation.py'),
        rollout_implementation={name: sha(HERE/name) for name in
                                ('rollout.py', 'rollout_amp.py', 'scenario_sampling.py',
                                 'physical_audit.py', 'stair_course.py')})
    write(root/'manifest.json',
          dict(plan_sha256=sha(plan), arguments=dict(epochs=10, dagger_rounds=3, collection_profile='robust',
               randomization_strength=1., initial_randomization_strength=.25),
               protocol=protocol, protocol_sha256=digest(protocol), selection_rule='fixture'))
    paths = []
    rounds = []
    previous = None
    for index, strength in enumerate((.25, .5, .75, 1.)):
        paths.extend(collect(root, index, dict(base_protocol, strength=strength), previous))
        rounds.append(fit(root, index, paths, previous, fit_sha))
        previous = rounds[-1]['student_sha256']
    selected = rounds[-1]
    selection = write(root/'selection.json',
                      dict(rule='fixture', profile='robust', seeds=SEEDS, selected_round=3,
                           selected_student=selected['student'], comparison_version=COMPARISON_VERSION,
                           candidates=[dict(row, rank=[i]) for i, row in enumerate(rounds[1:])]))
    write(root/'completed.json',
          dict(result=RESULT, rounds=rounds, selection=str(selection), selection_sha256=sha(selection),
               protocol=protocol, protocol_sha256=digest(protocol), student=selected['student'],
               student_sha256=selected['student_sha256'], selected_round=3, full_bank_evaluated=False,
               teacher_updates=0, ppo_updates=0, critic_trained=False,
               requires_candidate_finetuning=True, requires_post_finetuning_evaluation=True))
    return root, plan


def test_loads_epoch_nine_and_ordered_full_replay_from_same_source(parent):
    root, plan = parent
    previous = load_parent(root, plan)
    assert previous['round_index'] == 3
    assert previous['base'] == root
    assert previous['collector'] == HERE
    assert Path(previous['student']).name == 'student_epoch_0009.pt'
    assert previous['data_paths'] == read(root/'data_round_3.json')
    assert len(previous['data_paths']) == 12
    assert previous['collection_metadata']('robust', .5) == protocol_metadata('robust', .5)
    assert previous['provenance']['selection_sha256'] == sha(root/'selection.json')


@pytest.mark.parametrize('coordinator_sha', [
    'fffd518a08c6beeeb4348816d7baab3ba208f05572a302c86bf564da10351777',
    '4a6c0b5589d8faa30b402445db5f59dfcdab934a729b5fe04ea41e9350f36c20',
])
def test_loads_parent_with_known_pre_continuation_coordinator(parent, coordinator_sha):
    root, plan = parent
    for name in ('manifest.json', 'completed.json'):
        path = root/name
        receipt = read(path)
        receipt['protocol']['implementation_sha256'] = coordinator_sha
        receipt['protocol_sha256'] = digest(receipt['protocol'])
        write(path, receipt)
    verified = load_parent(root, plan)
    assert verified['provenance']['selection_sources']['pipeline.py'] == coordinator_sha


def test_resolves_the_pinned_frozen_checkout_without_writing_to_it(parent):
    root, plan = parent
    repository = HERE.parents[3]
    relative_engine = HERE.relative_to(repository)
    history = root/'stair_evaluation_upgrade'
    checkout = history/'collection_checkout'
    history.mkdir()
    subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', str(repository), str(checkout)], check=True)

    def git(*args):
        subprocess.run(['git', '-C', str(checkout), *args], check=True, capture_output=True)

    checkout_eol = 'true' if b'\r\n' in (HERE/'fit.py').read_bytes() else 'false'
    git('config', 'core.autocrlf', checkout_eol)
    if sys.platform == 'win32':
        git('config', 'core.longpaths', 'true')
    git('sparse-checkout', 'set', relative_engine.as_posix())
    git('checkout', '--detach', 'cd348ef7436b5f6e254eb3e34aaedd281a0b804a')
    collector = checkout/relative_engine
    original_metadata = runpy.run_path(str(collector/'randomization.py'))['protocol_metadata']

    manifest_path = root/'manifest.json'
    current = read(manifest_path)
    current['protocol']['collection'] = original_metadata('robust', 1.)
    current['protocol_sha256'] = digest(current['protocol'])
    write(manifest_path, current)
    completion_path = root/'completed.json'
    write(completion_path, dict(read(completion_path), protocol=current['protocol'],
                                protocol_sha256=current['protocol_sha256']))
    original = deepcopy(current)
    old = original['protocol']
    del old['stair_evaluation']
    del old['evaluation_velocity_frame']
    del old['rollout_implementation']['stair_course.py']
    old['rollout_implementation'] = {name: sha(collector/name) for name in old['rollout_implementation']}
    old.update(selection=old['collection'], implementation_sha256=sha(collector/'pipeline.py'),
               evaluation_implementation_sha256=sha(collector/'evaluation.py'))
    original['protocol_sha256'] = digest(old)
    archive = write(history/'original_manifest.json', original)
    for index, strength in enumerate((.25, .5, .75, 1.)):
        for commit in (root/'data_commits'/f'round_{index}').glob('*.json'):
            write(commit, dict(read(commit), protocol=original_metadata('robust', strength)))

    verified = load_parent(root, plan)
    assert verified['collector'] == collector
    assert verified['collection_protocol'] != current['protocol']['selection']
    assert verified['provenance']['original_manifest_sha256'] == sha(archive)
    assert not (collector/'__pycache__').exists()


@pytest.mark.parametrize('corruption', ['reordered_replay', 'changed_shard', 'wrong_execution_actor',
                                       'wrong_strength', 'winner', 'plan', 'protocol', 'archived_manifest',
                                       'unknown_coordinator'])
def test_rejects_changed_base_evidence(parent, corruption):
    root, plan = parent
    if corruption == 'reordered_replay':
        path = root/'data_round_3.json'
        write(path, list(reversed(read(path))))
    elif corruption == 'changed_shard':
        Path(read(root/'data_round_3.json')[0]).write_bytes(b'changed')
    elif corruption == 'wrong_execution_actor':
        path = root/'data_commits/round_2/0001_backpack.json'
        write(path, dict(read(path), execution_student_sha256='0' * 64))
    elif corruption == 'wrong_strength':
        path = root/'data_commits/round_1/0001_backpack.json'
        record = read(path)
        record['protocol']['strength'] = 1.
        write(path, record)
    elif corruption == 'winner':
        path = root/'selection.json'
        selection = read(path)
        selection['candidates'][0]['rank'] = [100]
        write(path, selection)
        completed = root/'completed.json'
        write(completed, dict(read(completed), selection_sha256=sha(path)))
    elif corruption == 'plan':
        write(plan, {'different': 'plan'})
    elif corruption == 'protocol':
        path = root/'completed.json'
        completed = read(path)
        completed['protocol']['selection_seeds'].reverse()
        write(path, completed)
    elif corruption == 'unknown_coordinator':
        for name in ('manifest.json', 'completed.json'):
            path = root/name
            receipt = read(path)
            receipt['protocol']['implementation_sha256'] = '9' * 64
            receipt['protocol_sha256'] = digest(receipt['protocol'])
            write(path, receipt)
    else:
        prior = deepcopy(read(root/'manifest.json'))
        del prior['protocol']['stair_evaluation']
        prior['protocol']['initial_strength'] = .5
        prior['protocol_sha256'] = digest(prior['protocol'])
        write(root/'stair_evaluation_upgrade/original_manifest.json', prior)
    with pytest.raises(ValueError):
        load_parent(root, plan)


def extension(root, plan, previous, *, long_replay=None):
    index = previous['round_index'] + 1
    protocol = previous['collection_protocol']
    parent_actor = previous['lineage'][-1]['student_sha256']
    paths = previous['data_paths'] + collect(root, index, protocol, parent_actor, long_replay=long_replay)
    candidate = fit(root, index, paths, parent_actor, sha(previous['collector']/'fit.py'), long_replay=long_replay)
    identity = dict(kind=KIND, round_index=index, parent=str(previous['lineage'][-1]['root']),
                    base=str(previous['base']), plan_sha256=sha(plan))
    fields = dict(identity, parent_provenance=previous['provenance'], collection_protocol=protocol,
                  selection_protocol=previous['manifest']['protocol']['selection'],
                  sources={'dagger_parent.py': '1' * 64})
    if long_replay:
        fields['long_replay'] = long_replay
    manifest = write(root/'manifest.json', fields)
    data_list = root/f'data_round_{index}.json'
    comparison = write(root/'comparison.json', {'round_index': index})
    return write(root/'completed.json',
                 dict(identity, result=RESULT, student=candidate['student'], student_sha256=candidate['student_sha256'],
                      fit_summary=candidate['fit_summary'], fit_summary_sha256=sha(candidate['fit_summary']),
                      previous_student_sha256=parent_actor, manifest_sha256=sha(manifest),
                      data_list_sha256=sha(data_list), comparison=str(comparison), comparison_sha256=sha(comparison)))


def test_later_rounds_use_immediate_actor_and_keep_original_selection(parent):
    base, plan = parent
    r3 = load_parent(base, plan)
    r4_root = base.parent/'round_four'
    extension(r4_root, plan, r3)
    r4 = load_parent(r4_root, plan)
    r5_root = base.parent/'round_five'
    extension(r5_root, plan, r4)
    r5 = load_parent(r5_root, plan)
    assert r5['selection'] == r3['selection']
    assert len(r5['data_paths']) == 18
    assert r5['data_paths'][:12] == r3['data_paths']
    assert [row['root'] for row in r5['lineage']] == [base, r4_root, r5_root]
    assert r5['student'] == r5['lineage'][-1]['student']


def test_extension_manifest_cannot_rebind_previous_round(parent):
    base, plan = parent
    root = base.parent/'extension'
    receipt = extension(root, plan, load_parent(base, plan))
    manifest = root/'manifest.json'
    write(manifest, dict(read(manifest), parent_provenance={}))
    write(receipt, dict(read(receipt), manifest_sha256=sha(manifest)))
    with pytest.raises(ValueError, match='parent provenance'):
        load_parent(root, plan)


@pytest.fixture
def long_round(parent):
    base, plan = parent
    r4_root = base.parent/'round_four'
    extension(r4_root, plan, load_parent(base, plan))
    r4 = load_parent(r4_root, plan)
    r5_root = base.parent/'round_five'
    extension(r5_root, plan, r4, long_replay=dict(
        steps=1000, stride=5, tasks=['stairs', 'backpack', 'run'], priority_probability=1/3,
        sources={'dagger_long_collect.py': '2' * 64, 'dagger_long_fit.py': '3' * 64,
                 'stair_course.py': '4' * 64},
    ))
    return r4, r5_root, plan


def test_next_round_inherits_all_six_collection_batches_and_epoch_nine(long_round):
    r4, r5_root, plan = long_round
    r5 = load_parent(r5_root, plan)
    assert r5['data_paths'][:15] == r4['data_paths']
    assert [Path(path).parents[1].name for path in r5['data_paths'][15:]] == [
        '0000_stairs', '0001_backpack', '0002_run', '0003_stairs', '0004_backpack', '0005_run',
    ]
    assert len(r5['provenance']['data_commits_sha256']) == 21
    assert r5['provenance']['fit_commit_sha256'] == sha(r5_root/'fit_round_5.json')
    r6_root = r5_root.parent/'round_six'
    extension(r6_root, plan, r5)
    r6 = load_parent(r6_root, plan)
    assert r6['data_paths'][:21] == r5['data_paths']
    assert r6['completion']['previous_student_sha256'] == r5['completion']['student_sha256']
    assert r6['selection'] == r4['selection']


@pytest.mark.parametrize('field,value', [('steps', 250), ('student', '0' * 64), ('design_ids', ['wrong'])])
def test_long_collection_rejects_inconsistent_native_summary(long_round, field, value):
    _, root, plan = long_round
    commit_path = root/'data_commits/round_5/0003_stairs.json'
    commit = read(commit_path)
    folder = Path(commit['folder'])
    summary = folder/'summary.json'
    audit = folder/'long_replay_collection.json'
    write(summary, dict(read(summary), **{field: value}))
    write(audit, dict(read(audit), summary_sha256=sha(summary)))
    write(commit_path, dict(commit, summary_sha256=sha(summary), collection_audit_sha256=sha(audit)))
    with pytest.raises(ValueError, match='long collection summary'):
        load_parent(root, plan)


def test_long_collection_rejects_native_terrain_for_stairs(long_round):
    _, root, plan = long_round
    commit_path = root/'data_commits/round_5/0003_stairs.json'
    commit = read(commit_path)
    audit = Path(commit['folder'])/'long_replay_collection.json'
    write(audit, dict(read(audit), terrain='native'))
    write(commit_path, dict(commit, collection_audit_sha256=sha(audit)))
    with pytest.raises(ValueError, match='long collection audit'):
        load_parent(root, plan)


def test_long_fit_rejects_priority_paths_from_ordinary_replay_even_if_hash_matches(long_round):
    _, root, plan = long_round
    priority = write(root/'long_replay_round_5.json', read(root/'data_round_5.json')[15:18])
    path = root/'fit_round_5.json'
    fit_receipt = read(path)
    fit_receipt['identity']['long_replay']['priority_data_sha256'] = sha(priority)
    write(path, fit_receipt)
    with pytest.raises(ValueError, match='long priority replay list'):
        load_parent(root, plan)


def test_long_fit_rejects_missing_priority_rows_even_if_hash_matches(long_round):
    _, root, plan = long_round
    path = root/'fit_round_5.json'
    fit_receipt = read(path)
    audit = Path(fit_receipt['summary']).with_name('long_replay_sampling.json')
    record = read(audit)
    record['tasks']['stairs']['priority_rows'] = 0
    write(audit, record)
    write(path, dict(fit_receipt, sampling_audit_sha256=sha(audit)))
    with pytest.raises(ValueError, match='long sampling audit'):
        load_parent(root, plan)
