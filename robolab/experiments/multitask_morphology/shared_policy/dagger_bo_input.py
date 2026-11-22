"""Resolve the verified selection winner from a completed DAgger continuation."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from contract import design_ids_for_keys
from dagger_continue import selection_parameters, verify_selection_history
from dagger_parent import CONTINUATION_KIND, ROLLOUT_SOURCES, SHARED_ENGINE, load_parent
from pipeline import jobs_for


HERE = Path(__file__).resolve().parent
PRE_MIXED_REVISION = '1b9cc2d15882817823cf899807de62046274240b'
PRE_MIXED_CHANGES = ('rollout.py', 'rollout_amp.py', 'scenario_sampling.py')


def continuation_actor(extension: Path, base: Path, plan_path: Path, plan: dict, base_actor: Path) -> Path:
    """Return the best paired-selection checkpoint through the requested round."""
    try:
        return _continuation_actor(extension, base, plan_path, plan, base_actor, HERE)
    except ValueError as error:
        if str(error) not in {f'DAGger evidence changed or is missing: {HERE/name}' for name in PRE_MIXED_CHANGES}:
            raise
    sources = ROLLOUT_SOURCES | {'stair_course.py', 'pipeline.py', 'evaluation.py', 'fit.py', 'collection_grid.py',
                                 'dagger_comparison.py', 'randomization.py', 'student_observations.py'}
    archived = {name: subprocess.check_output(['git', '-C', str(HERE), 'show',
                                               f'{PRE_MIXED_REVISION}:{(SHARED_ENGINE/name).as_posix()}'])
                for name in sorted(sources)}
    expected = json.loads((base/'manifest.json').read_bytes())['protocol']['rollout_implementation']['rollout.py']
    original = archived['rollout.py']
    if expected == sha256(original.replace(b'\n', b'\r\n')).hexdigest():
        archived = {name: data.replace(b'\n', b'\r\n') for name, data in archived.items()}
    # The fallback changes the source directory, while the original evidence
    # and every recorded source digest still go through the full validator.
    with TemporaryDirectory(prefix='.bo-frozen-', dir=HERE) as folder:
        engine = Path(folder)
        for name, data in archived.items():
            (engine/name).write_bytes(data)
        return _continuation_actor(extension, base, plan_path, plan, base_actor, engine)


def _continuation_actor(extension, base, plan_path, plan, base_actor, engine):
    plan_path = plan_path.resolve()
    source = load_parent(extension, plan_path, engine=engine)
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
