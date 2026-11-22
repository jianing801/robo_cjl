"""Report observed gains and preserve the incumbent on a paired rank tie."""

from evaluation import (
    COMPARISON_VERSION, EVALUATION_VELOCITY_FRAME, SELECTION_RULE,
    RANK_DECIMAL_PLACES, aggregate_evaluations, candidate_rank, teacher_comparison,
)


TASK_SUCCESS_PROGRESS_RATIO = .2
TASK_SUCCESS_SELECTION_RULE = ("Maximize student task success with equal tasks and equal structures per task. "
    "A successful episode survives the evaluation and reaches the declared forward progress. "
    "Ties retain the earlier checkpoint.")
METRICS = ('success_rate', 'survival_rate', 'mean_command_aligned_speed_m_s')


def selection_rank(student, teacher, rule):
    if rule == SELECTION_RULE:
        return candidate_rank(student, teacher)
    if rule != TASK_SUCCESS_SELECTION_RULE:
        raise ValueError('Unknown DAgger selection rule')
    comparison = teacher_comparison(student, teacher)
    return (round(comparison['mean_student_success_rate'], RANK_DECIMAL_PLACES),)


def _paired_metrics(rows, design_count):
    result = dict(design_count=design_count, episodes_per_policy=rows['candidate']['environments'])
    result.update({name: {metric: row[metric] for metric in METRICS} for name, row in rows.items()})
    candidate = result['candidate']
    for baseline in ('previous', 'incumbent'):
        reference = result[baseline]
        result[f'delta_vs_{baseline}'] = {
            metric: candidate[metric] - reference[metric]
            if candidate[metric] is not None and reference[metric] is not None else None
            for metric in METRICS
        }
    return result


def _subset_records(records, design_ids):
    """Present the recorded design statistics as jobs in the original aggregation."""
    return [
        dict(metric, task=row['task'], seed=row['seed'], evaluation_protocol=row['evaluation_protocol'],
             design_metrics={design: metric})
        for row in records
        for design, metric in row['design_metrics'].items()
        if design in design_ids[row['task']]
    ]


def summarize_comparison(*, round_index, previous_report, candidate_report, incumbent_report,
                         teacher, structure_keys, training_keys, design_ids, rule=TASK_SUCCESS_SELECTION_RULE):
    """Compare verified selection suites on the same panel, seeds and criteria.

    Reported rates pool episodes and mean speed pools measured horizontal
    distance and time. Ranking uses the declared rule for this comparison.
    """
    if (candidate_report['round_index'] != round_index
            or previous_report['round_index'] != round_index - 1
            or incumbent_report['round_index'] > previous_report['round_index']):
        raise ValueError('Invalid consecutive round or incumbent identity')

    tasks = list(design_ids)
    keys = list(map(tuple, structure_keys))
    seen = set(map(tuple, training_keys))
    splits = ['seen' if key in seen else 'held_out' for key in keys]
    if (not tasks or len(set(keys)) != len(keys) or set(splits) != {'seen', 'held_out'}
            or any(len(ids) != len(keys) or len(set(ids)) != len(keys) for ids in design_ids.values())):
        raise ValueError('Comparison requires paired design IDs and nonempty seen/held-out splits')
    subset_ids = {
        scope: {task: {design for design, split in zip(ids, splits) if split == scope}
                for task, ids in design_ids.items()}
        for scope in ('seen', 'held_out')
    }

    reports = dict(previous=previous_report, candidate=candidate_report, incumbent=incumbent_report)
    ranks, summaries = {}, {}
    for name, report in reports.items():
        records = report['evaluations']
        if any(row.get('velocity_frame') != EVALUATION_VELOCITY_FRAME for row in records):
            raise ValueError('Comparison requires the horizontal evaluation velocity frame')
        observed = aggregate_evaluations(records, tasks)
        if observed != report['task_metrics']:
            raise ValueError('Compared task metrics differ from the recorded episodes')
        if any(set(observed[task]['design_metrics']) != set(design_ids[task]) for task in tasks):
            raise ValueError('Compared design IDs differ from the paired panel')
        ranks[name] = selection_rank(observed, teacher, rule)
        summaries[name] = {'all': observed, **{
            scope: aggregate_evaluations(_subset_records(records, ids), tasks)
            for scope, ids in subset_ids.items()
        }}

    improves_incumbent = ranks['candidate'] > ranks['incumbent']
    selected = candidate_report if improves_incumbent else incumbent_report
    result = dict(
        round_index=round_index, selected_round=selected['round_index'], selected_student=selected['student'],
        selected_student_sha256=selected['student_sha256'], strictly_improves_incumbent=improves_incumbent,
        strictly_improves_previous=ranks['candidate'] > ranks['previous'],
        velocity_frame=EVALUATION_VELOCITY_FRAME, comparison_version=COMPARISON_VERSION,
        rule=rule, ranks={name: list(rank) for name, rank in ranks.items()},
        checkpoints={name: {field: report[field] for field in ('round_index', 'student', 'student_sha256')}
                     for name, report in reports.items()},
        tasks={},
    )
    if rule == TASK_SUCCESS_SELECTION_RULE:
        result['evaluation_criteria'] = next(iter(teacher.values()))['criteria']
    for task in tasks:
        comparison = {
            scope: _paired_metrics(
                {name: scopes[scope][task] for name, scopes in summaries.items()},
                len(keys) if scope == 'all' else len(subset_ids[scope][task]),
            )
            for scope in ('all', 'seen', 'held_out')
        }
        designs = {}
        for key, split, design in zip(keys, splits, design_ids[task]):
            rows = {name: scopes['all'][task]['design_metrics'][design] for name, scopes in summaries.items()}
            designs[design] = dict(_paired_metrics(rows, 1), structure_key=list(key), split=split,
                                   exposure_coverage=rows['candidate']['exposure_coverage'])
        comparison['designs'] = designs
        result['tasks'][task] = comparison
    return result
