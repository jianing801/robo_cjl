"""Resolve the verified selection winner from a completed DAgger continuation."""
from pathlib import Path
from types import SimpleNamespace

from contract import design_ids_for_keys
from dagger_continue import verify_selection_history
from dagger_parent import CONTINUATION_KIND, load_parent
from pipeline import jobs_for


HERE = Path(__file__).resolve().parent


def continuation_actor(extension: Path, base: Path, plan_path: Path, plan: dict, base_actor: Path) -> Path:
    """Return the best paired-selection checkpoint through the requested round."""
    plan_path = plan_path.resolve()
    source = load_parent(extension, plan_path, engine=HERE)
    if source['completion'].get('kind') != CONTINUATION_KIND:
        raise ValueError('BO requires a completed DAgger continuation')
    if source['base'] != base.resolve() or Path(source['lineage'][0]['student']) != base_actor.resolve():
        raise ValueError('DAgger continuation does not descend from the verified BO initialization')

    arguments = source['manifest']['arguments']
    protocol = source['manifest']['protocol']
    settings = SimpleNamespace(plan=str(plan_path), selection_profile=arguments['selection_profile'],
                               timeout_seconds=arguments['timeout_seconds'])
    selection_keys = list(map(tuple, protocol['selection_keys']))
    ids = design_ids_for_keys(plan, selection_keys)
    jobs = jobs_for(plan, len(selection_keys), ids)
    criteria = dict(duration_s=20., time_tolerance_s=.1, max_missing_tracking_steps=1,
                    min_command_progress_ratio=arguments['min_command_progress_ratio'])
    _, _, incumbent = verify_selection_history(settings, source, plan, ids, jobs,
                                               protocol['selection_seeds'], criteria)
    return Path(incumbent['student'])
