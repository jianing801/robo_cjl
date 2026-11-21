"""Resume the inspected pre-flight run with its original collectors and fit implementation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

from contract import sha, write_new
from evaluation import EVALUATION_VELOCITY_FRAME
from persistence import atomic_publish
from shared_bo_metrics import digest
from stair_course import COURSE

PREVIOUS_REVISION = 'cd348ef7436b5f6e254eb3e34aaedd281a0b804a'
# Byte hash of pipeline.py in 802ff294f2de9a93dfde374258ede6e085b1daa1.
BOTTOM_APPROACH_PIPELINE_SHA256 = '2e146ef15d585613ed9c1e18e1913358c75a082cfc29078da329487b334eaa70'
EVALUATION_FIELDS = {
    'implementation_sha256', 'rollout_implementation', 'selection', 'evaluation_velocity_frame',
    'evaluation_criteria', 'evaluation_version', 'evaluation_implementation_sha256', 'comparison_version',
}
SELECTION_MANIFEST_FIELDS = {'protocol', 'protocol_sha256', 'selection_rule'}


def frozen_engine(root, here, previous, explicit):
    history = root / 'stair_evaluation_upgrade'
    marker = history / 'original_manifest.json'
    if 'stair_evaluation' in previous['protocol'] and not marker.exists():
        return None
    if not marker.exists() and (not explicit or not (root/'USER_PAUSED').is_file()):
        raise ValueError('Pause the original run and resume explicitly to upgrade its stair evaluations')
    if (root/'selection.json').exists() or (root/'completed.json').exists():
        if not marker.exists():
            raise ValueError('The original selection is final; preserve its evidence')
    original = json.loads(marker.read_bytes()) if marker.exists() else previous
    protocol = original['protocol']
    if (digest(protocol) != original['protocol_sha256'] or protocol.get('selection_version') != 'last_epoch_dagger_paired_200'
            or 'stair_evaluation' in protocol):
        raise ValueError('Unrecognized source protocol for stair evaluation upgrade')
    repository = here.parents[3]
    checkout = history/'collection_checkout'
    history.mkdir(exist_ok=True)
    if not checkout.exists():
        subprocess.run(['git','-C',str(repository),'worktree','add','--detach',str(checkout),PREVIOUS_REVISION],check=True)
    def git(*args):
        return subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()
    if git('rev-parse','HEAD') != PREVIOUS_REVISION or git('status','--porcelain','--untracked-files=normal'):
        raise ValueError('Frozen training checkout changed')
    engine = checkout/here.relative_to(repository)
    for name, value in {**protocol['collection']['source_sha256'], **protocol['rollout_implementation'],
            'pipeline.py': protocol['implementation_sha256'], 'fit.py': protocol['fit_implementation_sha256'],
            'evaluation.py': protocol['evaluation_implementation_sha256']}.items():
        if sha(engine/name) != value:
            raise ValueError(f'Original runtime does not match the run manifest: {name}')
    module = importlib.util.spec_from_file_location('frozen_stair_collection', engine/'randomization.py')
    rules = importlib.util.module_from_spec(module)
    module.loader.exec_module(rules)
    return engine, rules.protocol_metadata, original


def upgrade_manifest(root, previous, current):
    """Only evaluation code/identity may differ. Preserve the original receipt before atomic replacement."""
    _validate_evaluation_only_change(previous, current, EVALUATION_FIELDS | {'stair_evaluation'})
    backup = root/'stair_evaluation_upgrade'/'original_manifest.json'
    if not backup.exists():
        write_new(backup,previous)
    elif json.loads(backup.read_bytes()) != previous:
        raise ValueError('Archived original manifest changed')
    _replace_file(root/'manifest.json', json.dumps(current, ensure_ascii=False, indent=2).encode())


def upgrade_tracking_manifest(root, previous, current, *, explicit):
    """Archive the pending run before reevaluating all tasks with horizontal speed."""
    if not explicit or not (root/'USER_PAUSED').is_file():
        raise ValueError('Pause the existing run and resume explicitly to update evaluation velocity')
    if (root/'selection.json').exists() or (root/'completed.json').exists():
        raise ValueError('Selection already finalized; preserve the original evidence')
    history = root/'stair_evaluation_upgrade'
    original = json.loads((history/'original_manifest.json').read_bytes())
    old, new = previous['protocol'], current['protocol']
    if (digest(original['protocol']) != original['protocol_sha256']
            or 'stair_evaluation' in original['protocol']
            or digest(old) != previous['protocol_sha256']
            or old.get('implementation_sha256') != BOTTOM_APPROACH_PIPELINE_SHA256
            or old.get('stair_evaluation') != COURSE
            or 'evaluation_velocity_frame' in old
            or digest(new) != current['protocol_sha256']
            or new.get('evaluation_velocity_frame') != EVALUATION_VELOCITY_FRAME):
        raise ValueError('Unrecognized source or destination protocol for evaluation velocity update')
    _validate_evaluation_only_change(previous, current, EVALUATION_FIELDS)
    path = root/'manifest.json'
    original_bytes = path.read_bytes()
    if json.loads(original_bytes) != previous:
        raise ValueError('Pending selection manifest changed')
    backup = history/f"tracking_previous_{previous['protocol_sha256']}.json"
    if not backup.exists():
        atomic_publish(backup, lambda stream: stream.write(original_bytes))
    elif backup.read_bytes() != original_bytes:
        raise ValueError('Archived previous tracking manifest changed')
    _replace_file(path, json.dumps(current, ensure_ascii=False, indent=2).encode())


def _validate_evaluation_only_change(previous, current, ignored_protocol_fields):
    def protected(manifest):
        return ({k: v for k, v in manifest['protocol'].items() if k not in ignored_protocol_fields},
                {k: v for k, v in manifest.items() if k not in SELECTION_MANIFEST_FIELDS})

    if protected(previous) != protected(current):
        raise ValueError('Training, selection budget, arguments or source identity changed beyond the stair upgrade')


def _replace_file(path, contents):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode='wb') as tmp:
        tmp.write(contents)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
