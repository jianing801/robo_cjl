"""Import and audit relocatable native USD caches without starting Isaac Sim."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from shutil import copy2
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from hybrid_assets import mesh_sources, progress, sha, urdf_mesh_paths, validate_asset, validate_asset_resume
from hybrid_native_cache import validate_training_cache
from native_cache_coordinator import publish_asset_proof

RESULT = 'NATIVE_ASSET_CACHE_COMPLETE'
TASKS = {'rough': 'stairs', 'backpack': 'backpack', 'amp': 'run'}


def read(path):
    return json.loads(path.read_bytes())


def runtime_environment(environment):
    """Expose Isaac's bundled CPU USD libraries to an isolated subprocess."""
    env = dict(environment)
    if importlib.util.find_spec('pxr') is not None:
        return env
    spec = importlib.util.find_spec('isaacsim')
    roots = list(spec.submodule_search_locations or ()) if spec else []
    if env.get('ISAAC_PATH'):
        roots.append(env['ISAAC_PATH'])
    for root in map(Path, roots):
        packages = sorted(root.glob('extscache/omni.usd.libs-*'), reverse=True)
        packages.extend(sorted(root.glob('exts/omni.usd.libs*'), reverse=True))
        for pkg in packages:
            if not (pkg/'pxr').is_dir():
                continue
            native = [str(x) for x in (pkg/'bin', pkg/'lib', root/'kit', root/'kit/kernel/plugins') if x.is_dir()]
            env['PORTABLE_USD_LIBRARY_PATH'] = os.pathsep.join(native)
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(pkg), env.get('PYTHONPATH'))))
            env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (*native, env.get('LD_LIBRARY_PATH'))))
            return env
    return env


def usd_reader():
    handles = []
    if os.name == 'nt':
        handles = [os.add_dll_directory(folder) for folder in os.environ.get('PORTABLE_USD_LIBRARY_PATH', '').split(os.pathsep)
                   if folder]
    from pxr import Sdf, Tf, UsdUtils

    def read_references(path):
        try:
            layer = Sdf.Layer.FindOrOpen(path)
            if layer is None:
                raise ValueError(f'Cannot open USD layer: {path}')
            return UsdUtils.ExtractExternalReferences(path)
        except Tf.ErrorException as error:
            raise ValueError(f'Cannot read USD dependencies: {path}') from error

    return read_references, handles


def check_layers(directory, names, read_references):
    paths = {(directory/name).resolve() for name in names}
    for path in paths:
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f'Asset layer escapes its folder or is missing: {path}')
        if path.suffix.lower() not in ('.usd', '.usda', '.usdc'):
            continue
        for refs in read_references(str(path)):
            for dependency in refs:
                if not dependency or dependency == 'OmniPBR.mdl':
                    continue  # Built-in Isaac material, resolved by Kit.
                relative = Path(dependency)
                if (relative.is_absolute() or relative.drive or '://' in dependency
                        or (path.parent/relative).resolve() not in paths):
                    raise ValueError(f'USD has an external or untracked dependency: {path} → {dependency}')


def cache_proofs(source, task):
    roots = [source/'资产准备'/task, source/'experts'/task,
             *sorted((source/'experts').glob(task+'_*'), reverse=True)]
    index = source/'启动记录'/f'{task}_asset_cache.json'
    seen = set()
    if index.is_file():
        try:
            record = read(index)
            path = (Path(record['cache'])/'completed.json').resolve()
            if path.is_relative_to(source):
                seen.add(path)
                if sha(path) == record['completed_sha256']:
                    yield path
        except (ValueError, KeyError, OSError, TypeError):
            pass
    for pattern in ('native_cache_*/completed.json', 'attempts/cache_*/train/result/batch_*/summary.json',
                    'attempts/cache_*/train/result/batch_*/assets/*.json'):
        for folder in roots:
            for path in sorted(folder.glob(pattern), reverse=True):
                path = path.resolve()
                if path not in seen and path.is_relative_to(source):
                    seen.add(path)
                    yield path


def receipt(row):
    usd = Path(row['usd']).resolve()
    directory = usd.parent
    signatures = {}
    for name in row['files_sha256']:
        if Path(name).is_absolute():
            raise ValueError(f'Unexpected absolute file in cache proof: {name}')
        item = (directory/name).resolve()
        if not item.is_relative_to(directory) or not item.is_file():
            raise ValueError(f'Missing or escaping layer: {name}')
        stat = item.stat()
        if stat.st_size <= 0:
            raise ValueError(f'Empty layer: {name}')
        signatures[name] = [stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino]
    return directory.parent/('.resume_'+directory.name+'.json'), dict(row=row, files=signatures)


def import_asset(row, design, incoming, staging_root, read_references):
    import yaml

    source = Path(row['usd']).resolve().parent
    old_receipt, binding = receipt(row)
    if old_receipt.is_symlink() or not old_receipt.is_file() or read(old_receipt) != binding:
        validate_asset(row)
    with TemporaryDirectory(prefix='.asset_', dir=staging_root) as temp:
        directory = Path(temp)
        config_file = source/'config.yaml'
        try:
            config = yaml.safe_load(config_file.read_text(encoding='utf8'))
        except yaml.YAMLError as error:
            raise ValueError('Invalid source converter configuration') from error
        if not isinstance(config, dict):
            raise ValueError('Invalid source converter configuration')
        # Isaac Lab excludes these three path-only fields from the native hash.
        fields = {k:v for k,v in config.items() if k not in ('asset_path','usd_dir','usd_file_name')}
        code = hashlib.md5(json.dumps(fields).encode() + Path(design['urdf']).read_bytes()).hexdigest()
        if code != row['converter_hash']:
            raise ValueError('Converter hash does not match the relocated URDF and configuration')
        config['asset_path'] = design['urdf']
        config['usd_dir'] = str(incoming)
        paths = dict(row['files_sha256'])
        for name in paths:
            target = directory/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == 'config.yaml':
                target.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
            elif target.suffix.lower() in ('.usd', '.usda', '.usdc'):
                try:
                    os.link(source/name, target)
                except OSError:
                    copy2(source/name, target)
            else:
                copy2(source/name, target)
        paths['config.yaml'] = sha(directory/'config.yaml')
        check_layers(directory, paths, read_references)
        if incoming.exists():
            incoming.rename(incoming.with_name(f'.unpublished_{incoming.name}_{uuid4().hex[:8]}'))
        # The temporary-directory manager can clean a path that has been moved.
        directory.rename(incoming)
    imported = dict(row, usd=str(incoming/Path(row['usd']).name), files_sha256=paths)
    new_receipt, binding = receipt(imported)
    if not new_receipt.exists():
        publish_asset_proof(new_receipt, binding)
    return imported


def import_task(source, root, task, read_references):
    old_bank = source/'banks'/task/'bank.json'
    new_bank = root/'banks'/task/'bank.json'
    source_sha, bank_sha = sha(old_bank), sha(new_bank)
    bank = read(new_bank)
    expected = {row['id']: row for row in bank['designs']}
    old = {row['id']: row for row in read(old_bank)['designs']}
    work = root/'资产准备'/task
    cache = work/'native_cache_portable'
    result = work/'attempts/cache_portable/train/result'
    folder = result/'batch_portable/assets'
    if (cache/'completed.json').is_file():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    meshes = mesh_sources(bank)
    known = {}
    for record in sorted(folder.glob('*.json')):
        value = read(record)
        row, = value['assets']
        key = row['design']
        if value['bank_sha256'] != bank_sha or expected[key]['urdf_sha256'] != row['urdf_sha256']:
            raise ValueError(f'Temporary migration binding changed: {record}')
        validate_asset_resume(row)
        known[key] = row
    eligible = {key for key, design in expected.items() if key not in known and key in old
                and old[key]['urdf_sha256'] == design['urdf_sha256']}
    rejected = 0
    progress('PORTABLE_USD_PROGRESS',len(known),len(expected))
    for path in cache_proofs(source, task):
        if not eligible:
            break
        try:
            proof = read(path)
        except (ValueError, OSError):
            continue
        if (proof.get('result') != RESULT or proof.get('cache_schema_version') != 2
                or proof.get('bank_sha256') != source_sha or proof.get('task', TASKS[task]) != TASKS[task]):
            continue
        for row in proof.get('assets', ()):
            key = row['design']
            if key not in eligible:
                continue
            design = expected[key]
            required = {name: meshes[name] for name in urdf_mesh_paths(design['urdf'])}
            original = Path(row['usd']).resolve()
            if (row['urdf_sha256'] != design['urdf_sha256'] or original.parent.name != key
                    or not original.is_relative_to(source)
                    or any(proof.get('mesh_sources',{}).get(name) != digest for name,digest in required.items())):
                continue
            try:
                migrated = import_asset(row, design, cache/key, cache, read_references)
            except (ValueError, OSError, TypeError, KeyError, RuntimeError) as error:
                rejected += 1
                if rejected <= 3:
                    print('PORTABLE_USD_REBUILD_REQUIRED', task, key, error, flush=True)
                continue
            publish_asset_proof(folder/f'{key}.json', dict(result=RESULT, bank_sha256=bank_sha,
                cache_schema_version=2, mesh_sources=required, task=TASKS[task], designs=1, assets=[migrated]))
            known[key] = migrated
            eligible.remove(key)
            progress('PORTABLE_USD_PROGRESS',len(known),len(expected))
    print('PORTABLE_USD_IMPORTED',task,len(known),'/',len(expected),'pending',len(expected)-len(known),flush=True)
    if len(known) == len(expected):
        publish_asset_proof(cache/'completed.json', dict(result=RESULT, cache_schema_version=2,
            bank_sha256=bank_sha, task=TASKS[task], mesh_sources=meshes, designs=len(known),
            assets=[known[d['id']] for d in bank['designs']]))
        return cache
    return None


def verify(root, source, read_references):
    import yaml
    def local(path):
        return Path(path).resolve().is_relative_to(root)

    def visit(value):
        if isinstance(value, str):
            if str(source).casefold().replace('\\','/') in value.casefold().replace('\\','/'):
                raise ValueError(f'Old experiment still referenced: {value}')
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key);visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for task in TASKS:
        manifest = root/'banks'/task/'bank.json'
        bank = read(manifest)
        visit(bank)
        records = root/'启动记录'/f'{task}_asset_cache.json'
        index = read(records)
        cache = Path(index['cache']).resolve()
        if not local(cache):
            raise ValueError(f'{task}: USD cache is outside the migration destination')
        if index['bank_sha256'] != sha(manifest) or index['completed_sha256'] != sha(cache/'completed.json'):
            raise ValueError(f'{task}: USD cache index changed during migration')
        for row in bank['designs']:
            if not local(row['urdf']) or not local(row['report']):
                raise ValueError(f'{task}: retained robot files are outside the migration destination')
            for mesh in urdf_mesh_paths(row['urdf']):
                visit(mesh)
        proof = validate_training_cache(cache, manifest, [row['id'] for row in bank['designs']])
        visit(proof)
        for number,row in enumerate(proof['assets'],1):
            folder = Path(row['usd']).resolve().parent
            visit(yaml.safe_load((folder/'config.yaml').read_text(encoding='utf8')))
            check_layers(folder, row['files_sha256'], read_references)
            progress('PORTABLE_USD_AUDIT_PROGRESS',number,len(proof['assets']))
    library = root/'references/library.json'
    if library.is_file():
        source_clips = read(library)
        visit(source_clips)
        if not all(local(row['directory']) for row in source_clips['designs'].values()):
            raise ValueError('Retargeted AMP clips must be inside the migration destination')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['import','verify'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--task',choices=TASKS)
    a=parser.parse_args()
    try:
        reader, libraries = usd_reader()
    except ImportError as error:
        raise RuntimeError('请在已安装 Isaac Sim CPU USD Python 库的训练环境运行迁移。') from error
    if a.action == 'import':
        if not a.task:parser.error('--task is required for import')
        import_task(a.source.resolve(),a.root.resolve(),a.task,reader)
    else:
        verify(a.root.resolve(),a.source.resolve(),reader)


if __name__=='__main__':
    main()
