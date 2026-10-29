"""Bind AMP to the exact Rough morphology assets, without touching Rough files."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROUGH = HERE.parent/'rough_hybrid'
BINDING_KIND = 'amp_rough_exact_morphology_v1'
SHARED_FIELDS = ('configuration', 'snapshot_root', 'num_envs', 'knees', 'ankles',
    'context_width', 'ranges_m', 'designs', 'motor_pairs', 'original_template',
    'original_template_sha256', 'reference_contract_source',
    'reference_contract_sha256', 'canonical_semantic_contract')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rough_module(name):
    """Unique module names; never import a cached sibling train/train_stage."""
    path = ROUGH/(name+'.py')
    spec = importlib.util.spec_from_file_location('_amp_shared_'+name, path)
    module = importlib.util.module_from_spec(spec)
    # Rough pure helpers import sibling hybrid_* modules, not AMP modules.
    original = list(sys.path)
    try:
        sys.path.insert(0, str(ROUGH))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original
    return module


def validate_alignment(amp, rough, rough_path):
    binding = amp.get('rough_morphology_binding', {})
    if binding.get('kind') != BINDING_KIND:
        raise ValueError('AMP bank is not bound to exact Rough morphology')
    if binding.get('rough_bank_sha256') != sha(rough_path):
        raise ValueError('bound Rough bank changed')
    if rough.get('kind') != 'hybrid_rough_grid_bank_v1' or rough.get('task_key') != 'stairs':
        raise ValueError('expected a genuine Rough grid bank')
    for name in SHARED_FIELDS:
        if amp[name] != rough[name]:
            raise ValueError('AMP/Rough morphology differs: '+name)
    return dict(verified=True, designs=len(rough['designs']),
        context_width=rough['context_width'], assets_reused_without_conversion=True,
        rough_bank_sha256=binding['rough_bank_sha256'])


def adopt_rough_bank(rough_path, grid):
    """Only task identity changes. All nominal physical asset fields stay exact."""
    from amp_bank import KIND
    from amp_support import RECIPE, source_identity
    path = Path(rough_path).resolve()
    rough = rough_module('hybrid_bank').load_configurable_bank(path)
    if rough['configuration'] != grid:
        raise ValueError('AMP grid must exactly match the supplied Rough bank')
    bank = deepcopy(rough)
    bank.update(kind=KIND, task_key='run')
    bank['training'] = dict(recipe=RECIPE, authorized_tasks=['run'],
        source_hashes=source_identity('run'), retargeted_references=True)
    bank['rough_morphology_binding'] = dict(kind=BINDING_KIND,
        rough_bank=str(path), rough_bank_sha256=sha(path),
        read_only_asset_reuse=True)
    validate_alignment(bank, rough, path)
    return bank


def validate_bound_bank(bank):
    binding = bank.get('rough_morphology_binding')
    if binding is None:
        return dict(verified=False, reason='independently generated legacy AMP bank')
    path = Path(binding['rough_bank'])
    rough = json.loads(path.read_text(encoding='utf8'))
    return validate_alignment(bank, rough, path)


def shared_runtime_sources():
    """Include external helper dependencies in the strict experiment identity."""
    return {name: sha(ROUGH/name) for name in
        ('hybrid_importer.py', 'hybrid_resources.py')}
