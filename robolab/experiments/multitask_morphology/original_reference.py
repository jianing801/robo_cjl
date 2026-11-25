"""Portable, read-only validation of the archived original-RPO V4 milestone.

The reference was measured at committed update 100 of a 1000-update run.
Its original remote paths are provenance, not required filesystem locations.
No synthetic short-run manifest or simulation is needed to verify its ratios.
"""
import json
from math import isclose, isfinite
from pathlib import Path, PurePosixPath
import re

import numpy as np

from contract import sha
from finetune_settings import (MIXED_BO_LAYOUT, PPO_SCHEDULE, NATIVE_PPO_ALGORITHM,
                               NATIVE_PPO_SETTINGS, NATIVE_PPO_STATE_KIND)
from shared_bo_metrics import (ABSOLUTE_COT_BASIS, DT, FORMAL_TASK_WEIGHTS, FOUR_METRICS,
                               PHYSICS_DT, REPORT_VERSION, TASKS, TRIAL_VERSION, digest,
                               paired_trials, positive, read_conditions, score_trial, trial_metric,
                               validate_four_metric_weights)
from stair_course import COURSE, evaluation_flight, outside_flight


ARCHIVE_KIND = 'rpo_original_expert_prior_100_v1'
NORMALIZATION_KIND = 'rpo_original_expert_prior_100_normalization_v1'
DESIGN_KEY = [.25, .30, 'DM-J10010L-2EC', 'DM-J4340P-2EC']
REFERENCE_UPDATES = 100
SOURCE_RUN_UPDATES = 1000
REFERENCE_ROOT = 'docs/reference/rpo_original_configuration'
# Approved weights recorded in this archive release; future scoring may
# choose another explicit valid four-metric profile without changing the
# reference denominators or retroactively editing this measurement record.
ARCHIVED_WEIGHTS = {task: dict(zip(FOUR_METRICS[task], weights, strict=True))
                    for task, weights in {'stairs': (.25,.25,.40,.10),
                                          'backpack': (.40,.25,.20,.15),
                                          'run': (.25,.40,.15,.20)}.items()}

REQUIRED_FILES = {
    'conditions.json', 'reference_protocol.json', 'reference_result.json', 'normalization.json',
    'source_training_manifest.json', 'source_convergence_manifest.json', 'source_checkpoint_receipt.json',
    'source_candidate_plan.json', 'source_legacy_manifest.json', 'evaluation_manifest.json',
    'bo_evaluation.json', 'mean_reward.csv', 'training_reward.csv', 'sw_gp_report.json',
    *[f'{task}/{name}' for task in TASKS for name in ('bo_samples.npz', 'bo_trials.json')],
}


def _read(path):
    return json.loads(path.read_text(encoding='utf8'))


def _require(test, message):
    if not test:
        raise ValueError(message)


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _matches(row, expected):
    return all(row.get(field) == value for field, value in expected.items())


def _equal_measurement(a, b):
    """Compare immutable measurements with only floating-point roundoff tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal_measurement(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal_measurement(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return isfinite(a) and isfinite(b) and isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return type(a) is type(b) and a == b


def _check_files(root, archive):
    hashes = archive.get('files_sha256', {})
    _require(isinstance(hashes, dict) and REQUIRED_FILES <= hashes.keys(),
             'Incomplete V4 reference archive: missing required artifact hashes')
    actual = set()
    for path in root.rglob('*'):
        _require(not path.is_symlink(), f'V4 reference archive contains a symlink: {path.name}')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in ('manifest.json', '.gitattributes'):
                actual.add(name)
    _require(actual == set(hashes) - {'.gitattributes'}, 'V4 reference archive file inventory differs')
    for name, expected in hashes.items():
        parts = PurePosixPath(name)
        _require(not parts.is_absolute() and '..' not in parts.parts and '\\' not in name
                 and name == parts.as_posix() and _valid_sha(expected), 'Invalid archive artifact/hash identity')
        file = root/name
        _require(file.is_file() and file.resolve().is_relative_to(root) and sha(file) == expected,
                 f'V4 reference archive hash differs: {name}')
    return hashes


def _recompute_trial(trace, native, task, conditions, *, joint_names):
    """Use the same mechanical calculation as TrialCapture.publish, without simulation."""
    count = round(conditions['duration_s']/PHYSICS_DT)
    with np.load(trace, allow_pickle=False) as raw:
        values = {key: raw[key] for key in raw.files}
    shapes = dict(position=(count,1,3), vx=(count,1), omega=(count,1,2),
                  torque=(count,1,23), joint_vel=(count,1,23), active=(count,1),
                  initial_position=(1,3), origins=(1,3), total_mass_kg=(1,), fell=(1,),
                  timeout=(1,), dt_s=(), control_dt_s=())
    _require(set(values) == set(shapes) and all(values[k].shape == shape and np.isfinite(values[k]).all()
                    for k, shape in shapes.items()), f'Malformed 200 Hz raw trace: {task}')
    _require(all(values[k].dtype == np.bool_ for k in ('active','fell','timeout'))
             and values['active'].all() and not values['fell'].any() and not values['timeout'].any()
             and float(values['dt_s']) == PHYSICS_DT and float(values['control_dt_s']) == DT,
             f'Incomplete or terminated reference raw trace: {task}')
    motors = native['motors']
    _require([row['joint'] for row in motors] == joint_names,
             f'Reference raw trace motor mapping/joint order differs: {task}')
    motor_ids = {row['joint']: row['motor_id'] for row in motors}
    _require(all(motor_ids[f'{side}_knee_joint'] == DESIGN_KEY[2]
                 and all(motor_ids[f'{side}_ankle_{axis}_joint'] == DESIGN_KEY[3] for axis in ('pitch','roll'))
                 for side in ('left','right')), f'Reference motor identity differs: {task}')
    _require(len(native['trials']) == 1 and native['trials'][0]['replica'] == 0,
             f'Reference must contain exactly one native trial: {task}')
    measured = native['trials'][0]
    spec = conditions['tasks'][task]
    active = values['active'][:,0]
    positions = values['position'][active,0]
    forward = float(positions[-1,0]-values['initial_position'][0,0])
    mass = float(values['total_mass_kg'][0])
    _require(positive(mass), f'Invalid physical mass in raw trace: {task}')
    if task == 'stairs':
        _require(spec.get('course') == COURSE, 'Expected the full timed stair course')
        flight = evaluation_flight(conditions['duration_s'], spec['speed_mps'], spec['step_depth_m'],
                                   spec['travel_margin_factor'])
        relative = positions-values['origins'][0]
        outside = bool(outside_flight(relative[:,0],relative[:,1],flight['origin_x_m'],flight['length_m']).any())
        route_completed = None
    else:
        outside = False
        route_completed = bool((positions[:,0]-values['initial_position'][0,0] >= spec['route_distance_m']).any())
    payload = measured['payload']
    if task == 'backpack':
        _require(isinstance(payload,dict) and payload.get('mass_kg') == spec['payload_kg'],
                 'Reference backpack payload does not match 10 kg conditions')
    else:
        _require(payload is None, f'Unexpected reference payload: {task}')
    recomputed = dict(replica=0, **score_trial(task, speed=spec['speed_mps'], dt=PHYSICS_DT,
        expected_steps=count, mass=mass, forward=forward, motors=motors,
        fell=values['fell'][0], timeout=values['timeout'][0], reached_top=None,
        route_completed=route_completed, payload=payload, outside_stairs=outside,
        min_forward_progress_ratio=conditions.get('min_forward_progress_ratio'),
        **{name:values[name][active,0] for name in ('vx','omega','torque','joint_vel')}))
    _require(recomputed['status'] == 'success' and _equal_measurement(recomputed, measured),
             f'Native metric recomputation differs from actual raw trace: {task}')
    return measured


def _check_normalization(normalization, pair, report_sha):
    _require(_matches(normalization, dict(kind=NORMALIZATION_KIND, reference_updates=REFERENCE_UPDATES,
        ppo_algorithm=NATIVE_PPO_ALGORITHM, cot_basis=ABSOLUTE_COT_BASIS,
        task_weights=FORMAL_TASK_WEIGHTS, reference_task_objectives=dict.fromkeys(TASKS,1.),
        reference_objective=1., raw_source_report_sha256=report_sha)), 'V4 reference normalization identity differs')
    metrics = normalization['metrics']
    weights = validate_four_metric_weights({task:{name:entry['weight'] for name,entry in rows.items()}
                                           for task,rows in metrics.items()})
    _require(weights == ARCHIVED_WEIGHTS, 'Archived normalization weights differ from the approved four-metric release')
    for task,names in FOUR_METRICS.items():
        for name in names:
            baseline = trial_metric(pair['tasks'][task],name,ABSOLUTE_COT_BASIS,metric_profile='four')
            _require(positive(baseline) and _equal_measurement(metrics[task][name],
                dict(baseline=baseline, multiplier=1./baseline, weight=weights[task][name],
                     task_weighted_coefficient=FORMAL_TASK_WEIGHTS[task]*weights[task][name]/baseline)),
                f'Archived normalization differs from raw reference denominator: {task}/{name}')


def check_v4_archive(root, benchmark, conditions):
    """Return the original byte-proven native source report after independent verification."""
    root = Path(root).resolve()
    archive_path = root/'manifest.json'
    archive = _read(archive_path)
    _require(_matches(archive, dict(kind=ARCHIVE_KIND, design_key=DESIGN_KEY,
        ppo_algorithm=NATIVE_PPO_ALGORITHM, reference_updates=REFERENCE_UPDATES,
        source_run_updates=SOURCE_RUN_UPDATES)), 'Canonical V4 original RPO archive identity differs')
    hashes = _check_files(root,archive)
    protocol = _read(root/'reference_protocol.json')
    cond_hash = digest(conditions)
    expected = dict(source_plan_sha256=benchmark['source_plan_sha256'],
        initial_actor_sha256=benchmark['initial_actor_sha256'], updates=REFERENCE_UPDATES,
        conditions_sha256=cond_hash, ppo_algorithm=NATIVE_PPO_ALGORITHM, native_ppo_settings=NATIVE_PPO_SETTINGS)
    _require(protocol == expected and _matches(benchmark, {k:v for k,v in expected.items() if k != 'conditions_sha256'})
             and benchmark.get('conditions_sha256',cond_hash) == cond_hash
             and _matches(archive,dict(donor_sha256=protocol['source_plan_sha256'],
                 initial_actor_sha256=protocol['initial_actor_sha256']))
             and _valid_sha(protocol['source_plan_sha256']) and _valid_sha(protocol['initial_actor_sha256']),
             'Canonical V4 original RPO reference protocol differs from the measured panel')
    _require(read_conditions(root/'conditions.json') == conditions and conditions['duration_s'] == 20.
             and conditions['measurement_dt_s'] == PHYSICS_DT and conditions['copies'] == 1
             and conditions['reset_mode'] == 'fixed' and len(conditions['seeds']) == 1,
             'Canonical V4 original RPO conditions differ')

    training = _read(root/'source_training_manifest.json')
    convergence = _read(root/'source_convergence_manifest.json')
    checkpoint = _read(root/'source_checkpoint_receipt.json')
    plan = _read(root/'source_candidate_plan.json')
    legacy = _read(root/'source_legacy_manifest.json')
    evaluation = _read(root/'evaluation_manifest.json')
    report = _read(root/'bo_evaluation.json')
    candidate = report['candidate']
    actor_sha = archive['final_actor_sha256']
    singleton_sha = archive['source_plan_sha256']
    plan_sha = hashes['source_candidate_plan.json']
    training_sha = hashes['source_training_manifest.json']
    teachers = training['teacher_checkpoint_sha256']
    _require(candidate['structure_key'] == DESIGN_KEY and set(candidate['design_ids']) == set(TASKS)
             and _valid_sha(actor_sha) and _valid_sha(singleton_sha)
             and set(teachers) == set(TASKS) and all(_valid_sha(x) for x in teachers.values())
             and _matches(archive,dict(evaluation_plan_sha256=plan_sha,
                  checkpoint_receipt_sha256=hashes['source_checkpoint_receipt.json'],
                  raw_source_report_sha256=hashes['bo_evaluation.json'])),
             'Canonical V4 reference physical/model artifact identity differs')

    _require(_matches(convergence,dict(kind='native_joint_ppo_convergence',runtime_commit=archive['runtime_git_commit'],
        ppo_algorithm=NATIVE_PPO_ALGORITHM,candidate=candidate,source_plan_sha256=singleton_sha,
        initial_actor_sha256=protocol['initial_actor_sha256'],conditions_sha256=cond_hash,
        requested_updates=SOURCE_RUN_UPDATES,legacy_manifest_sha256=hashes['source_legacy_manifest.json']))
        and REFERENCE_UPDATES in convergence['evaluation_updates']
        and _matches(legacy,dict(kind='original_rpo_convergence_ppo1000',source_plan_sha256=singleton_sha,
            donor_sha256=protocol['source_plan_sha256'],initial_actor_sha256=protocol['initial_actor_sha256'],
            candidate=candidate,conditions_sha256=cond_hash,updates=SOURCE_RUN_UPDATES,seed=convergence['seed'])),
        'V4 reference source-run provenance does not match the committed convergence run')
    _require(_matches(training,dict(kind='shared_candidate_finetuning_mixed', plan_sha256=singleton_sha,
        initial_actor_sha256=protocol['initial_actor_sha256'], candidate=candidate, updates=SOURCE_RUN_UPDATES,
        seed=convergence['seed'],ppo_algorithm=NATIVE_PPO_ALGORITHM,native_ppo_settings=NATIVE_PPO_SETTINGS,
        ppo_schedule=PPO_SCHEDULE,init_at_random_ep_len=True,envs_per_task=MIXED_BO_LAYOUT,steps_per_task=24))
        and _matches(training['collection_protocol'],dict(profile='robust',strength=1.)),
        'V4 reference source training manifest has a different PPO/physical collection protocol')
    _require(_matches(checkpoint,dict(completed_updates=REFERENCE_UPDATES,manifest_sha256=training_sha,
        ppo_algorithm=NATIVE_PPO_ALGORITHM,state_kind=NATIVE_PPO_STATE_KIND,teacher_checkpoint_sha256=teachers,
        warmup_completed=NATIVE_PPO_SETTINGS['critic_warmup_updates'],sha256=archive['checkpoint_sha256']))
        and _valid_sha(checkpoint['sha256']) and re.fullmatch(r'update_000100_[0-9a-f]{32}\.pt',checkpoint['path'])
        and checkpoint['metrics']['samples'] == sum(MIXED_BO_LAYOUT.values())*24
        and checkpoint['metrics']['critic_only'] is False,
        'V4 reference committed checkpoint is not formal update 100')
    _require(_matches(evaluation,dict(kind='concurrent_native_milestone_measurement',candidate=candidate,
        ppo_updates=REFERENCE_UPDATES,ppo_algorithm=NATIVE_PPO_ALGORITHM,actor_sha256=actor_sha,
        checkpoint_sha256=checkpoint['sha256'],training_manifest_sha256=training_sha,
        conditions_sha256=cond_hash,plan_sha256=plan_sha))
        and _matches(report,dict(kind=REPORT_VERSION,domain='simulation',candidate=candidate,
            ppo_updates=REFERENCE_UPDATES,ppo_algorithm=NATIVE_PPO_ALGORITHM,student_sha256=actor_sha,
            conditions=conditions,conditions_sha256=cond_hash,evaluation_manifest_sha256=hashes['evaluation_manifest.json'])),
        'V4 original reference evaluation and checkpoint identity differ')
    _require(_matches(plan,dict(scope='exact_candidate',selected_structure_keys=[DESIGN_KEY],
        reference_structure_key=DESIGN_KEY)) and set(plan['teachers']) == set(TASKS)
        and set(archive['native_banks']) == set(TASKS), 'V4 original reference candidate plan differs')
    joint_names = plan['schema']['semantic_contract']['joint_names']
    _require(len(joint_names) == len(set(joint_names)) == 23, 'V4 reference native joint list differs')

    natives = []
    _require(set(archive['tasks']) == set(TASKS), 'V4 reference archive task identity differs')
    for task in TASKS:
        native_name, trace_name = f'{task}/bo_trials.json', f'{task}/bo_samples.npz'
        evidence = archive['tasks'][task]
        spec = plan['teachers'][task]
        bank_sha = archive['native_banks'][task]['bank_sha256']
        _require(_matches(spec,dict(design_ids=[candidate['design_ids'][task]],bank_sha256=bank_sha,
            checkpoint_sha256=teachers[task],full_body_torque=True)) and _valid_sha(bank_sha)
            and _matches(evidence,dict(native_report_sha256=hashes[native_name],trace_sha256=hashes[trace_name],artifact=native_name)),
            f'V4 reference native physical bank or archive receipt differs: {task}')
        native = _read(root/native_name)
        _require(_matches(native,dict(kind=TRIAL_VERSION,task=task,seed=conditions['seeds'][0],
            conditions_sha256=cond_hash,design_id=candidate['design_ids'][task],bank_sha256=bank_sha,
            plan_sha256=plan_sha,actor_sha256=actor_sha,trace_sha256=hashes[trace_name]))
            and isinstance(native['trace'],str) and native['trace'].replace('\\','/').endswith('/bo_samples.npz'),
            f'V4 reference native task source report differs: {task}')
        _recompute_trial(root/trace_name,native,task,conditions,joint_names=joint_names)
        natives.append(native)
    report_hashes = [row['sha256'] for row in report['measurements']]
    _require(sorted(report_hashes) == sorted(hashes[f'{task}/bo_trials.json'] for task in TASKS),
             'V4 reference evaluation lost a raw native source report')
    pairs = paired_trials(natives,conditions)
    _require(len(pairs) == 1 and pairs[0]['feasible'] is True and pairs == report['pairs'],
             'V4 reference paired evaluation differs from native measurements')

    summary = _read(root/'reference_result.json')
    pair = pairs[0]
    _require(_matches(summary,dict(candidate=candidate,seed=pair['seed'],jointly_feasible=True,
        ppo_updates=REFERENCE_UPDATES,ppo_algorithm=NATIVE_PPO_ALGORITHM,conditions_sha256=cond_hash,
        student_sha256=actor_sha,report_sha256=hashes['bo_evaluation.json'])) and set(summary['tasks']) == set(TASKS),
        'V4 reference summary does not describe the exact measured milestone')
    for task,row in pair['tasks'].items():
        expected = {name:row['observed'][name] for name in ('absolute_mechanical_cot','cot','forward_rmse_mps',
            'roll_pitch_rms_rad_s','knee_ankle_max_rated_ratio')}
        expected.update({name:row[name] for name in ('status','reason','steps','forward_m','mean_forward_mps')})
        expected['rate_hz'] = 1./row['dt_s']
        _require(_matches(summary['tasks'][task],expected), f'V4 reference summary raw metrics differ: {task}')
    _check_normalization(_read(root/'normalization.json'),pair,hashes['bo_evaluation.json'])
    return report, dict(source_type='repository_v4_milestone',root=REFERENCE_ROOT,
        archive_manifest_sha256=sha(archive_path),report_sha256=hashes['bo_evaluation.json'],
        protocol_sha256=hashes['reference_protocol.json'],normalization_sha256=hashes['normalization.json'])
