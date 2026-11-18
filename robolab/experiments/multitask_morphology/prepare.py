"""Prepare three exact common structure banks, generating base URDFs only once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
TASKS = ('rough', 'backpack', 'amp')

def select_existing_designs(bank, config):
    """Project the frozen old grid onto the new grid using exact physical keys."""
    from hybrid_bank import select_grid
    from robolab.morphology_context import SCHEMA

    grid = config['grid']
    old_grid = bank['configuration']
    if (bank['schema_version'] != 2 or config.get('context_schema') != SCHEMA
            or {k: v for k, v in old_grid.items() if k not in ('ankle_pool', 'ankle_motor_count')}
               != {k: v for k, v in grid.items() if k not in ('ankle_pool', 'ankle_motor_count')}
            or set(old_grid['ankle_pool']) - set(grid['ankle_pool']) not in (set(), {'DM-J8006-2EC-V1.1'})):
        raise ValueError('Source must use the current physical grid with at most the removed 8006 motor')
    planned = select_grid(grid, grid['knee_pool'], grid['ankle_pool'])
    old = {row['id']: row for row in bank['designs']}
    if len(old) != len(bank['designs']):
        raise ValueError('Duplicate source design')
    selected = []
    for target in planned['designs']:
        row = old.get(target['id'])
        if row is None or any(row.get(k) != v for k, v in target.items()):
            raise ValueError(f"Exact legacy morphology missing: {target['id']}")
        selected.append(row.copy())
    pairs = {row['pair_id'] for row in selected}
    return dict(bank, configuration=grid, knees=planned['knees'], ankles=planned['ankles'],
                num_envs=planned['num_envs'], designs=selected,
                motor_pairs={key: bank['motor_pairs'][key] for key in sorted(pairs)})


def checked_bytes(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Source asset changed: {path}')
    return data


def publish_bytes(path, data):
    """Publish a resumable copy. An existing file must be byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'Migration output must not be a symbolic link: {path}')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing migration output differs: {path}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='migration_', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_json(path, value):
    publish_bytes(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf8'))


def legacy_path(path, source):
    return path.absolute().is_relative_to(source) or path.resolve().is_relative_to(source)


def stage_urdf(path, expected, target, source, root, mesh_paths):
    original = checked_bytes(path, expected)
    tree = ET.fromstring(original)
    changed = False
    for element in tree.iter():
        filename = element.get('filename')
        if filename is None:
            continue
        dependency = Path(filename.removeprefix('file://'))
        absolute = dependency.is_absolute()
        if not absolute:
            if '://' in filename:
                raise ValueError(f'Unsupported URDF asset URI: {filename}')
            dependency = Path(path).parent/dependency
        if dependency not in mesh_paths:
            if legacy_path(dependency, source):
                data = dependency.read_bytes()
                path_hash = hashlib.sha256(data).hexdigest()
                destination = root/'assets/meshes'/path_hash/dependency.name
                publish_bytes(destination, data)
                mesh_paths[dependency] = str(destination)
            else:
                if not dependency.is_file():
                    raise FileNotFoundError(dependency)
                mesh_paths[dependency] = None
        migrated = mesh_paths[dependency]
        if migrated is not None or not absolute:
            element.set('filename', migrated or str(dependency.resolve()))
            changed = True
    data = ET.tostring(tree, encoding='utf8', xml_declaration=True) if changed else original
    publish_bytes(target, data)
    return str(target), hashlib.sha256(data).hexdigest()


def reject_legacy_paths(value, source):
    """Check path-valued manifest metadata without probing old directories."""
    if isinstance(value, dict):
        for item in value.values():
            reject_legacy_paths(item, source)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_legacy_paths(item, source)
    elif isinstance(value, str):
        path = Path(value.removeprefix('file://'))
        if path.is_absolute() and path.is_relative_to(source):
            raise ValueError(f'Migrated asset still references the legacy experiment: {value}')


def relocate_assets(banks, source, root):
    """Move the retained geometry to the new tree and rebuild its path bindings."""
    from robolab.physical_snapshot import verify_snapshot

    mesh_paths = {}
    templates = {}
    contracts = {}
    snapshots = {}
    for bank in banks.values():
        template = Path(bank['original_template'])
        expected = bank['original_template_sha256']
        if legacy_path(template, source):
            if (template, expected) not in templates:
                target = root/'assets/templates'/expected/template.name
                templates[template, expected] = stage_urdf(template, expected, target, source, root, mesh_paths)
            bank['original_template'], bank['original_template_sha256'] = templates[template, expected]
        contract = Path(bank['reference_contract_source'])
        expected = bank['reference_contract_sha256']
        if legacy_path(contract, source):
            if (contract, expected) not in contracts:
                target = root/'assets/contracts'/expected/contract.name
                publish_bytes(target, checked_bytes(contract, expected))
                contracts[contract, expected] = str(target)
            bank['reference_contract_source'] = contracts[contract, expected]
        snapshot = Path(bank['snapshot_root'])
        if legacy_path(snapshot, source):
            if snapshot not in snapshots:
                identity = verify_snapshot(snapshot)
                target = root/'assets/physical_snapshot'
                for name, digest in identity['verified_file_sha256'].items():
                    publish_bytes(target/name, checked_bytes(snapshot/name, digest))
                publish_bytes(target/'snapshot_manifest.json', (snapshot/'snapshot_manifest.json').read_bytes())
                snapshots[snapshot] = str(target)
            bank['snapshot_root'] = snapshots[snapshot]
            bank['motor_pairs'] = {key: dict(value, snapshot_root=snapshots[snapshot])
                                   for key, value in bank['motor_pairs'].items()}
        bank.pop('asset_source_bank', None)

    rough, backpack, amp = (banks[task] for task in TASKS)
    for base, load, run in zip(rough['designs'], backpack['designs'], amp['designs'], strict=True):
        rough_source = base['urdf']
        rough_hash = base['urdf_sha256']
        if (run['urdf_sha256'] != rough_hash or load['base_urdf_sha256'] != rough_hash
                or load['report_sha256'] != base['report_sha256'] or run['report_sha256'] != base['report_sha256']):
            raise ValueError(f'Legacy task geometry differs: {base["id"]}')
        for other in (run['urdf'], load['base_urdf']):
            if other != rough_source:
                checked_bytes(other, rough_hash)
        for other in (load['report'], run['report']):
            if other != base['report']:
                checked_bytes(other, base['report_sha256'])
        target = root/'banks/rough'/base['id']
        base['urdf'], base['urdf_sha256'] = stage_urdf(
            rough_source, rough_hash, target/'rpo.urdf', source, root, mesh_paths)
        report = checked_bytes(base['report'], base['report_sha256'])
        report_json = json.loads(report)
        template = report_json.get('template')
        if template and legacy_path(Path(template), source):
            report_json['template'] = rough['original_template']
            report = json.dumps(report_json, indent=2, ensure_ascii=False, allow_nan=False).encode('utf8')
        reject_legacy_paths(report_json, source)
        publish_bytes(target/'rpo.report.json', report)
        base.update(report=str(target/'rpo.report.json'), report_sha256=hashlib.sha256(report).hexdigest())
        load['base_urdf'], load['base_urdf_sha256'] = base['urdf'], base['urdf_sha256']
        pack_target = root/'banks/backpack'/load['id']/'robot_with_backpack.urdf'
        load['urdf'], load['urdf_sha256'] = stage_urdf(
            load['urdf'], load['urdf_sha256'], pack_target, source, root, mesh_paths)
        for name in ('urdf', 'urdf_sha256'):
            run[name] = base[name]
        for name in ('report', 'report_sha256'):
            load[name] = run[name] = base[name]


def relocate_references(library, bank, bank_path, source, root):
    rows = {row['id']: row for row in bank['designs']}
    completed = {}
    for key, row in library['designs'].items():
        geometry = row['geometry_key']
        directory = root/'references'/geometry
        old = Path(row['directory'])
        stamp = (old, row['clips'])
        if completed.get(geometry) != stamp:
            for name, digest in row['clips'].items():
                publish_bytes(directory/(name+'.pkl'), checked_bytes(old/(name+'.pkl'), digest))
            completed[geometry] = stamp
        row.update(directory=str(directory), urdf_sha256=rows[key]['urdf_sha256'])
    library['unique_geometry_count'] = len(completed)
    library['bank_sha256'] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    execution = library.get('reference_execution', {})
    warp_cache = execution.get('warp_cache')
    if warp_cache and legacy_path(Path(warp_cache), source):
        del execution['warp_cache']
    reject_legacy_paths(library, source)
    target = root/'references/library.json'
    publish_json(target, library)
    print('AMP_REFERENCE_MOTIONS_REUSED', target, flush=True)


def reuse_banks(source, root, settings):
    """Publish independently usable bank assets as a one-time migration."""
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError('Old and new experiment roots must be separate')
    sys.path[:0] = [str(HERE/'engine'), str(HERE/'amp_engine'), str(HERE.parents[1])]
    from hybrid_bank import load_configurable_bank as load_rough
    from amp_bank import load_configurable_bank as load_amp
    from amp_references import validate_library
    from robolab.morphology_context import bind_physical_context
    from robolab.rough_geometry import file_sha

    paths = {task: source/'banks'/task/'bank.json' for task in TASKS}
    original = {task: path.read_bytes() for task, path in paths.items()}
    source_hashes = {task: hashlib.sha256(data).hexdigest() for task, data in original.items()}
    originals = {task: json.loads(data) for task, data in original.items()}
    banks = {}
    for task, config in zip(TASKS, settings, strict=True):
        selected = select_existing_designs(originals[task], config)
        banks[task] = selected
        print('MORPHOLOGY_SUBSET_SELECTED', task, len(selected['designs']), '/', len(originals[task]['designs']), flush=True)

    references = source/'references/library.json'
    library = None
    if references.is_file():
        selected = [row['id'] for row in banks['amp']['designs']]
        try:
            library = validate_library(originals['amp'], paths['amp'], references, selected)
        except (ValueError, KeyError, OSError) as error:
            print('AMP_EXISTING_REFERENCES_REJECTED', error, flush=True)
        if library is not None:
            library['designs'] = {key: library['designs'][key] for key in selected}
    targets = {task: root/'banks'/task/'bank.json' for task in TASKS}
    relocate_assets(banks, source, root)
    for bank in banks.values():
        bind_physical_context(bank)
    if any(file_sha(path) != source_hashes[task] for task, path in paths.items()):
        raise ValueError('Source banks changed during context migration')

    for task, bank in banks.items():
        target = targets[task]
        if task == 'backpack':
            bank['base_morphology_binding'].update(bank=str(targets['rough']), sha256=file_sha(targets['rough']))
        elif task == 'amp':
            bank['rough_morphology_binding'].update(rough_bank=str(targets['rough']),
                rough_bank_sha256=file_sha(targets['rough']))
        reject_legacy_paths(bank, source)
        publish_json(target, bank)
        (load_amp if task == 'amp' else load_rough)(target)
    if library is not None:
        relocate_references(library, banks['amp'], targets['amp'], source, root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--template')
    p.add_argument('--mesh-dir')
    p.add_argument('--reuse-banks-from', type=Path, help='Select the physical 5+5 subset of a matching legacy experiment')
    p.add_argument('--generate-assets', action='store_true', help='Otherwise CPU preview only')
    a = p.parse_args()
    root = Path(a.output_root).resolve()
    configs = {t: HERE/'configs'/f'{t}.json' for t in TASKS}
    settings = [json.loads(p.read_text(encoding='utf8')) for p in configs.values()]
    grids = [config['grid'] for config in settings]
    if not grids[0] == grids[1] == grids[2]:
        raise ValueError('Three task morphology grids differ')
    if a.generate_assets and root.exists() and not a.reuse_banks_from:
        raise ValueError('Use a NEW output root; existing banks are never overwritten')
    sys.path.insert(0, str(HERE/'engine'))
    from hybrid_bank import select_grid
    from hybrid_schedule import load_schedule
    grid = grids[0]
    selected = select_grid(grid, grid['knee_pool'], grid['ankle_pool'])
    bank_preview = dict(configuration=grid, designs=selected['designs'])
    schedules = [load_schedule(bank_preview, config)[0] for config in settings]
    if not schedules[0] == schedules[1] == schedules[2]:
        raise ValueError('Three task formal schedules differ')
    if a.reuse_banks_from:
        if not a.generate_assets or a.template or a.mesh_dir:
            p.error('--reuse-banks-from requires --generate-assets and a new output root')
        reuse_banks(a.reuse_banks_from.resolve(), root, settings)
    else:
        if a.generate_assets and not (a.template and a.mesh_dir):
            p.error('fresh asset generation requires --template and --mesh-dir')
        common = ['--generate-assets'] if a.generate_assets else []
        template = ['--template', a.template, '--mesh-dir', a.mesh_dir] if a.template and a.mesh_dir else []
        commands = [
            [sys.executable, str(HERE/'engine/prepare_bank.py'), '--config', str(configs['rough']),
             *template, '--output-dir', str(root/'banks/rough'), *common],
            [sys.executable, str(HERE/'engine/prepare_bank.py'), '--task-key', 'backpack',
             '--config', str(configs['backpack']), '--rough-bank', str(root/'banks/rough/bank.json'),
             '--output-dir', str(root/'banks/backpack'), *common],
            [sys.executable, str(HERE/'amp_engine/prepare_bank.py'), '--config', str(configs['amp']),
             '--rough-bank', str(root/'banks/rough/bank.json'), '--output-dir', str(root/'banks/amp'), *common]]
        for cmd in commands:
            subprocess.run(cmd, check=True)
    if a.generate_assets:
        from audit_banks import audit
        report = audit(root/'banks/rough/bank.json', root/'banks/amp/bank.json', root/'banks/backpack/bank.json')
        publish_json(root/'bank_audit.json', report)
        print('THREE_TASK_BANKS_VALIDATED', root, flush=True)


if __name__ == '__main__':
    main()
