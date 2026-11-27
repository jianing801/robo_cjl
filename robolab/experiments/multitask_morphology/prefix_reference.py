"""Validate an actual 200-update prefix of a longer, identically configured run.

Do not rewrite the source manifest's budget or the checkpoint's identity. The
normalization uses the measured Actor at update 200, not the final Actor.
"""
from pathlib import Path


KIND = 'original_rpo_round5_update200_prefix_reference'


def checked_prefix_reference(bo, root, benchmark, conditions):
    from native_ppo_convergence import verify_blob
    from student import load_student
    import torch

    root = Path(root).resolve()
    identity_path = root/'reference_identity.json'
    identity = bo.read(identity_path)
    report_path = root/'bo_evaluation.json'
    norm_path = root/'normalization.json'
    normalizer = bo.read(norm_path)
    report = bo.read(report_path)
    actor = root/'shared_actor_update200.pt'
    expected = dict(kind=KIND, updates=bo.DEFAULT_BO_UPDATES,
                    initial_actor_sha256=benchmark['initial_actor_sha256'],
                    ppo_algorithm=bo.CURRENT_PPO_ALGORITHM,
                    native_ppo_settings=bo.CURRENT_PPO_SETTINGS,
                    conditions_sha256=bo.digest(conditions))
    if (benchmark.get('updates') != expected['updates']
            or benchmark.get('ppo_algorithm') != expected['ppo_algorithm']
            or benchmark.get('native_ppo_settings') != expected['native_ppo_settings']
            or any(identity.get(k) != v for k,v in expected.items())
            or normalizer.get('kind') != 'rpo_original_round5_ppo200_four_metrics_normalization'
            or normalizer.get('identity_sha256') != bo.sha(identity_path)
            or normalizer.get('report_sha256') != bo.sha(report_path)
            or normalizer.get('conditions_sha256') != expected['conditions_sha256']):
        raise ValueError('Prefix reference PPO or evidence protocol differs')
    t = Path(identity['source_training_root'])
    manifest_path = t/'manifest.json'
    manifest = bo.read(manifest_path)
    receipt_path = t/'checkpoints'/f"update_{expected['updates']:06d}.json"
    receipt = bo.read(receipt_path)
    source_settings = dict(manifest['native_ppo_settings'])
    checkpoint_interval = source_settings.pop('checkpoint_interval', None)
    candidate = identity['candidate']
    source_plan = t.parent/'plan.json'
    evaluation_plan = t/'plan.json'
    if (t.resolve() != t or bo.sha(manifest_path) != identity['source_manifest_sha256']
            or bo.sha(receipt_path) != identity['source_checkpoint_receipt_sha256']
            or receipt['sha256'] != identity['source_checkpoint_sha256']
            or receipt['completed_updates'] != expected['updates']
            or receipt['warmup_completed'] != bo.CURRENT_PPO_SETTINGS['critic_warmup_updates']
            or receipt['teacher_checkpoint_sha256'] != manifest['teacher_checkpoint_sha256']
            or receipt['ppo_algorithm'] != expected['ppo_algorithm']
            or manifest['ppo_algorithm'] != expected['ppo_algorithm']
            or manifest['initial_actor_sha256'] != expected['initial_actor_sha256']
            or manifest['updates'] != identity['source_run_budget'] or manifest['updates'] < expected['updates']
            or source_settings != expected['native_ppo_settings']
            or checkpoint_interval != identity['source_checkpoint_interval']
            or manifest['envs_per_task'] != bo.MIXED_BO_LAYOUT
            or manifest['ppo_schedule'] != bo.PPO_SCHEDULE
            or manifest['training_conditions'] != 'native_expert'
            or manifest['init_at_random_ep_len'] is not True
            or manifest['candidate'] != candidate
            or candidate['structure_key'] != list(bo.RPO_REFERENCE_KEY)
            or bo.sha(source_plan) != manifest['plan_sha256']
            or manifest['plan_sha256'] != identity['source_plan_sha256']
            or bo.sha(evaluation_plan) != identity['evaluation_plan_sha256']):
        raise ValueError('Prefix reference source checkpoint or training plan changed')
    plan = bo.load_plan(evaluation_plan)
    if (plan['selected_structure_keys'] != [list(bo.RPO_REFERENCE_KEY)]
            or {task: spec['checkpoint_sha256'] for task,spec in plan['teachers'].items()}
               != manifest['teacher_checkpoint_sha256']):
        raise ValueError('Prefix reference physical singleton or expert priors differ')
    state = verify_blob(bo, t, receipt, identity['source_manifest_sha256'])
    model, metadata = load_student(actor, 'cpu')
    if (actor.resolve() != actor or bo.sha(actor) != identity['actor_sha256']
            or state['ppo_algorithm'] != expected['ppo_algorithm']
            or state['warmup_completed'] != receipt['warmup_completed']
            or state['teacher_checkpoint_sha256'] != manifest['teacher_checkpoint_sha256']
            or state['schema'] != model.schema or state['architecture'] != model.architecture
            or metadata.get('ppo_updates') != expected['updates']
            or metadata.get('initial_actor_sha256') != expected['initial_actor_sha256']
            or metadata.get('manifest_sha256') != identity['source_manifest_sha256']
            or metadata.get('source_checkpoint_sha256') != receipt['sha256']
            or metadata.get('source_checkpoint_receipt_sha256') != identity['source_checkpoint_receipt_sha256']
            or set(state['actor']) != set(model.state_dict())
            or any(not torch.equal(v, model.state_dict()[k]) for k,v in state['actor'].items())):
        raise ValueError('Prefix reference Actor is not the committed update-200 Actor')
    if (report['candidate'] != candidate or report['conditions'] != conditions
            or report['conditions_sha256'] != expected['conditions_sha256']
            or report['ppo_updates'] != expected['updates'] or report['ppo_algorithm'] != expected['ppo_algorithm']
            or Path(report['student']) != actor or report['student_sha256'] != identity['actor_sha256']):
        raise ValueError('Prefix reference report identity differs')
    measured = {}
    expected_trials = {(seed, task) for seed in conditions['seeds'] for task in bo.TASKS}
    for entry in report['measurements']:
        p = Path(entry['report'])
        native = bo.read(p)
        key = (native['seed'], native['task'])
        stage = root/'evaluation'/'measurements'/f'{key[0]}_{key[1]}'
        if (key not in expected_trials or key in measured or p.resolve() != p
                or p.name != 'bo_trials.json' or p.parents[2] != stage
                or bo.sha(p) != entry['sha256'] or bo.read(stage/'accepted.json') != entry
                or Path(native['trace']) != p.with_name('bo_samples.npz')):
            raise ValueError('Prefix reference accepted trial identity differs')
        spec = plan['teachers'][key[1]]
        design_id = candidate['design_ids'][key[1]]
        if spec['design_ids'] != [design_id]:
            raise ValueError('Prefix reference native design differs')
        bo.check_native_measurement(native, p, conditions, task=key[1], seed=key[0], design_id=design_id,
            actor_sha=identity['actor_sha256'], bank_sha=spec['bank_sha256'], plan_sha=identity['evaluation_plan_sha256'])
        measured[key] = native
    if set(measured) != expected_trials or bo.paired_trials(measured.values(), conditions) != report['pairs']:
        raise ValueError('Prefix reference measurements incomplete or changed')
    bo.check_panel_pairs(report['pairs'], bo.required_panel_pairs(conditions))
    if len(report['pairs']) != 1 or report['pairs'][0]['feasible'] is not True:
        raise ValueError('Prefix reference has failed tasks')
    # Re-derive the denominators from accepted native data, never from the
    # saved display values. Formal scoring may intentionally select new weights.
    for task, names in bo.FOUR_METRICS.items():
        for name in names:
            raw = bo.trial_metric(report['pairs'][0]['tasks'][task], name, bo.ABSOLUTE_COT_BASIS, metric_profile='four')
            row = normalizer['metrics'][task][name]
            if row['baseline'] != raw or row['inverse'] != 1.0/raw:
                raise ValueError('Prefix reference normalization disagrees with measured data')
    return report, dict(root=str(root), source_type='round5_update200_prefix',
                        report_sha256=bo.sha(report_path), identity_sha256=bo.sha(identity_path),
                        normalization_sha256=bo.sha(norm_path), source_checkpoint_sha256=receipt['sha256'])
