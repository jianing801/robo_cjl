"""Resume the inspected pre-flight run with its original collectors and fit implementation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

from contract import sha, write_new
from shared_bo_metrics import digest

PREVIOUS_REVISION = 'cd348ef7436b5f6e254eb3e34aaedd281a0b804a'


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
    old, new = previous['protocol'].copy(), current['protocol'].copy()
    for field in ('implementation_sha256','rollout_implementation','selection','stair_evaluation'):
        old.pop(field,None)
        new.pop(field,None)
    if (old != new or {k:v for k,v in previous.items() if k not in ('protocol','protocol_sha256')}
            != {k:v for k,v in current.items() if k not in ('protocol','protocol_sha256')}):
        raise ValueError('Training, selection budget, arguments or source identity changed beyond the stair upgrade')
    backup = root/'stair_evaluation_upgrade'/'original_manifest.json'
    if not backup.exists():
        write_new(backup,previous)
    elif json.loads(backup.read_bytes()) != previous:
        raise ValueError('Archived original manifest changed')
    with tempfile.NamedTemporaryFile(dir=root, delete=False, mode='w',encoding='utf8') as tmp:
        json.dump(current,tmp,ensure_ascii=False,indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name,root/'manifest.json')
