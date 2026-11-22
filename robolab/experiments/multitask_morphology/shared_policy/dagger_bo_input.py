"""Resolve the verified selection winner from a completed DAgger continuation."""
from pathlib import Path
from types import SimpleNamespace

from contract import design_ids_for_keys
from dagger_continue import selection_parameters, verify_selection_history
from dagger_parent import CONTINUATION_KIND, load_parent
from pipeline import jobs_for


HERE = Path(__file__).resolve().parent


def continuation_actor(extension: Path, base: Path, plan_path: Path, plan: dict, base_actor: Path) -> Path:
    """Return the best paired-selection checkpoint through the requested round."""
    plan_path = plan_path.resolve()
    source = load_parent(extension, plan_path, engine=HERE)
    if source['completion'].get('kind') != CONTINUATION_KIND:
        raise ValueError('BO requires a completed DAgger continuation')
    if source['base'] != base.resolve() or Path(source['selection']['selected_student']) != base_actor.resolve():
        raise ValueError('DAgger continuation does not descend from the verified BO initialization')

    arguments = source['manifest']['arguments']
    protocol = source['manifest']['protocol']
    settings = SimpleNamespace(plan=str(plan_path), selection_profile=arguments['selection_profile'],
                               timeout_seconds=arguments['timeout_seconds'])
    selection_keys = list(map(tuple, protocol['selection_keys']))
    ids = design_ids_for_keys(plan, selection_keys)
    jobs = jobs_for(plan, len(selection_keys), ids)
    recorded_criteria = source['previous_assessment'].get('evaluation_criteria')
    criteria = selection_parameters(recorded_criteria) if recorded_criteria is not None else None
    _, _, incumbent = verify_selection_history(settings, source, plan, ids, jobs,
                                               protocol['selection_seeds'], criteria)
    return Path(incumbent['student'])
