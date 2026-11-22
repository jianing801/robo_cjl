"""Verify the completed continuation handoff to a new BO run."""
from pathlib import Path

import pytest

from dagger_comparison import TASK_SUCCESS_SELECTION_RULE
from evaluation import SELECTION_RULE, candidate_rank, evaluation_criteria
from test_dagger_comparison import DESIGN_IDS, SEEDS, STRUCTURE_KEYS, _report
from test_dagger_parent import extension, load_parent, parent, read, sha, write
from shared_bo_metrics import digest


@pytest.fixture(params=['legacy', 'mixed'])
def chain(parent, monkeypatch, request):
    import dagger_continue

    base, plan_path = parent
    teachers = {}
    for task, ids in DESIGN_IDS.items():
        designs = [dict(id=design, thigh_m=thigh, calf_m=calf, knee_motor=knee, ankle_motor=ankle)
                   for design, (thigh, calf, knee, ankle) in zip(ids, STRUCTURE_KEYS, strict=True)]
        bank = write(base.parent/f'{task}_bank.json', {'designs': designs})
        teachers[task] = dict(bank=str(bank), design_ids=ids)
    plan = dict(teachers=teachers)
    samples = {0: ((.5, 20), (.5, 20)), 1: ((.06, 10), (.06, 10)),
               2: ((.12, 20), (.12, 20)), 3: ((.4, 20), (.08, 20)),
               4: ((.5, 20), (.5, 20)), 5: ((.6, 20), (.6, 20))}
    scores = {}
    for ratio in (.2, .5):
        for index, sample in samples.items():
            report = _report(index, [sample] * len(SEEDS), progress_ratio=ratio)
            scores[ratio, index] = {field: report[field] for field in ('task_metrics', 'by_seed', 'evaluations')}
    assessments = {index: scores[.5, index] for index in samples}
    selection_path = base/'selection.json'
    selection = read(selection_path)
    selection['rule'] = SELECTION_RULE
    selection['teacher'] = assessments[0]['task_metrics']
    for candidate in selection['candidates']:
        assessment = assessments[candidate['round_index']]
        candidate.update(assessment, rank=candidate_rank(assessment['task_metrics'], selection['teacher']))
    write(selection_path, selection)
    manifest_path = base/'manifest.json'
    manifest = read(manifest_path)
    manifest['selection_rule'] = SELECTION_RULE
    manifest['arguments'].update(selection_profile='robust', timeout_seconds=90, min_command_progress_ratio=.5)
    manifest['protocol']['selection_keys'] = STRUCTURE_KEYS
    manifest['protocol']['collection_scope']['structure_keys'] = STRUCTURE_KEYS[:1]
    manifest['protocol']['evaluation_criteria'] = evaluation_criteria()
    manifest['protocol_sha256'] = digest(manifest['protocol'])
    write(manifest_path, manifest)
    completion = base/'completed.json'
    write(completion, dict(read(completion), selection_sha256=sha(selection_path),
                           protocol=manifest['protocol'], protocol_sha256=manifest['protocol_sha256']))

    def archived_suite(plan_arg, path, root, jobs, student, profile, strength, seeds, suite,
                       environment, timeout, criteria, *, reuse_only, rescore=None):
        assert (plan_arg, path, profile, strength, seeds) == (plan, plan_path, 'robust', 1., list(SEEDS))
        assert jobs == list(DESIGN_IDS.items())
        assert reuse_only and environment is None and timeout == 90
        index = 0 if suite == 'selection/teacher' else int(suite.rsplit('_', 1)[1])
        assert criteria['min_command_progress_ratio'] == (.2 if index == 5 and request.param == 'mixed' else .5)
        ratio = (rescore or criteria)['min_command_progress_ratio']
        if suite == 'selection/teacher':
            assert root == base and student is None
            return scores[ratio, 0]
        return scores[ratio, index]

    monkeypatch.setattr(dagger_continue, 'run_selection_suite', archived_suite)

    source = load_parent(base, plan_path)
    base_actor = Path(source['student'])
    previous = incumbent = {**source['lineage'][-1], **assessments[3]}
    for index in (4, 5):
        root = base.parent/f'round_{index}'
        receipt = extension(root, plan_path, source)
        if index == 5 and request.param == 'mixed':
            manifest_path = root/'manifest.json'
            write(manifest_path, dict(read(manifest_path), selection_criteria=evaluation_criteria(
                min_command_progress_ratio=.2)))
            write(receipt, dict(read(receipt), manifest_sha256=sha(manifest_path)))
        source = load_parent(root, plan_path)
        candidate = {**source['lineage'][-1], **assessments[index]}
        if index == 5 and request.param == 'mixed':
            comparison = dagger_continue.compare_round(source, DESIGN_IDS, scores[.2, 0]['task_metrics'],
                {**previous, **scores[.2, 4]}, {**candidate, **scores[.2, 5]},
                {**selection['candidates'][1], **scores[.2, 2]}, rule=TASK_SUCCESS_SELECTION_RULE)
        else:
            comparison = dagger_continue.compare_round(source, DESIGN_IDS, assessments[0]['task_metrics'],
                                                       previous, candidate, incumbent, rule=SELECTION_RULE)
        comparison_path = write(root/'comparison.json', comparison)
        write(receipt, dict(read(receipt), comparison_sha256=sha(comparison_path)))
        source = load_parent(root, plan_path)
        if comparison['strictly_improves_incumbent']:
            incumbent = candidate
        previous = candidate
    return base, plan_path, plan, base_actor


@pytest.mark.parametrize('index', [4, 5])
def test_bo_uses_twenty_percent_success_across_all_rounds_including_the_base(chain, index):
    from dagger_bo_input import continuation_actor

    base, plan_path, plan, base_actor = chain
    latest = base.parent/f'round_{index}'
    archived = read(latest/'comparison.json')
    assert archived['selected_round'] == (4 if archived['rule'] == SELECTION_RULE else 2)
    winner = base/'bc_round_2'/'attempt_fixture'/'student_epoch_0009.pt'
    assert continuation_actor(latest, base, plan_path, plan, base_actor) == winner


@pytest.mark.parametrize('mismatch', ['base_directory', 'base_actor', 'base_as_extension', 'unfinished'])
def test_bo_rejects_incompatible_or_incomplete_continuation(chain, mismatch):
    from dagger_bo_input import continuation_actor

    base, plan_path, plan, base_actor = chain
    latest = base.parent/'round_5'
    if mismatch == 'base_directory':
        base = base.parent/'unrelated'
    elif mismatch == 'base_actor':
        base_actor = Path(read(latest/'completed.json')['student'])
    elif mismatch == 'base_as_extension':
        latest = base
    else:
        (latest/'completed.json').unlink()
    with pytest.raises(ValueError):
        continuation_actor(latest, base, plan_path, plan, base_actor)


def test_bo_rejects_comparison_disagreeing_with_archived_evaluation(chain):
    from dagger_bo_input import continuation_actor

    base, plan_path, plan, base_actor = chain
    latest = base.parent/'round_5'
    comparison_path = latest/'comparison.json'
    write(comparison_path, dict(read(comparison_path), selected_round=5))
    completed = latest/'completed.json'
    write(completed, dict(read(completed), comparison_sha256=sha(comparison_path)))
    with pytest.raises(ValueError, match='Committed comparison differs'):
        continuation_actor(latest, base, plan_path, plan, base_actor)
