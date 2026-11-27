"""Use a migrated, measured round-five Actor without transferring training replay.

The normal path still verifies the complete DAgger lineage. This explicit
reference-anchored path validates the final round receipt, frozen collection
plan, exact Actor bytes, expert identities, and committed original-RPO PPO200
checkpoint and measurements. It does not claim to re-read the original replay.
"""
from pathlib import Path


def measured_round5_input(bo, root, continuation, reference):
    from collection_grid import select_collection_grid
    from student import load_student
    root, continuation, reference = (Path(p).resolve() for p in (root, continuation, reference))
    path = root/'shared_plan/plan.json'
    plan = bo.load_plan(path)
    manifest_path = continuation/'manifest.json'
    complete = bo.read(continuation/'completed.json')
    manifest = bo.read(manifest_path)
    identity = dict(kind='shared_dagger_continuation_v1', round_index=5, plan_sha256=bo.sha(path))
    if (any(complete.get(k) != v or manifest.get(k) != v for k,v in identity.items())
            or complete.get('result') != 'DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING'
            or complete.get('manifest_sha256') != bo.sha(manifest_path)
            or complete.get('parent') != manifest.get('parent')
            or complete.get('base') != manifest.get('base')
            or complete.get('previous_student_sha256') != manifest['parent_provenance']['student_sha256']):
        raise ValueError('Migrated round-five completion or plan identity differs')
    base = Path(manifest['base']).resolve()
    if base != (root/'shared_policy').resolve():
        raise ValueError('Migrated round-five base differs from the frozen plan')
    present = bo.read(base/'manifest.json')
    original = base/'stair_evaluation_upgrade/original_manifest.json'
    archive_hash = manifest['parent_provenance'].get('original_manifest_sha256')
    if archive_hash:
        if bo.sha(original) != archive_hash:
            raise ValueError('Migrated original DAgger collection manifest differs')
        archived = bo.read(original)
    else:
        archived = present
    scope = archived['protocol']['collection_scope']
    if (scope != present['protocol']['collection_scope'] or scope['design_count'] != 400
            or set(map(tuple,scope['structure_keys'])) != set(select_collection_grid(plan['selected_structure_keys']))
            or Path(archived['arguments']['plan']).resolve() != path.resolve()):
        raise ValueError('Migrated round-five structure collection protocol differs')
    actor = Path(complete['student'])
    summary = Path(complete['fit_summary'])
    comparison = Path(complete['comparison'])
    if (actor.resolve() != actor or actor.name != 'student_epoch_0009.pt'
            or actor.parent.parent != continuation/'bc_round_5'
            or summary != actor.parent/'summary.json' or comparison != continuation/'comparison.json'
            or bo.sha(actor) != complete['student_sha256']
            or bo.sha(summary) != complete['fit_summary_sha256']
            or bo.sha(comparison) != complete['comparison_sha256']):
        raise ValueError('Migrated round-five Actor, fit summary or comparison changed')
    model, metadata = load_student(actor, 'cpu')
    if metadata['ppo_updates'] != 0 or model.schema != plan['schema'] or tuple(plan['schema']['tasks']) != bo.TASKS:
        raise ValueError('The BO starting Actor is not the un-finetuned three-task student')
    measured = bo.read(reference/'reference_identity.json')
    benchmark = dict(initial_actor_sha256=complete['student_sha256'], source_plan_sha256=bo.sha(path),
                     updates=bo.DEFAULT_BO_UPDATES, ppo_algorithm=bo.CURRENT_PPO_ALGORITHM,
                     native_ppo_settings=bo.CURRENT_PPO_SETTINGS)
    conditions = bo.read_conditions(bo.PHYSICS_CONDITIONS)
    from prefix_reference import checked_prefix_reference
    _, provenance = checked_prefix_reference(bo, reference, benchmark, conditions)
    source_manifest = bo.read(Path(measured['source_training_root'])/'manifest.json')
    if source_manifest['teacher_checkpoint_sha256'] != {task:spec['checkpoint_sha256'] for task,spec in plan['teachers'].items()}:
        raise ValueError('Original-RPO reference and BO donor experts differ')
    print('BO_ROUND5_REFERENCE_ANCHORED', actor, complete['student_sha256'], provenance['identity_sha256'], flush=True)
    return path, plan, actor
