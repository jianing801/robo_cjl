"""Controller-side native cache preparation before persistent scene lifetimes."""
from datetime import datetime
import hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_cache(cache,bank_path,design_ids=None):
    cache=Path(cache);bank=json.loads(Path(bank_path).read_text(encoding='utf8'))
    proof=json.loads((cache/'completed.json').read_text(encoding='utf8'))
    if proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE' or proof.get('bank_sha256')!=sha(bank_path):
        raise ValueError('native cache belongs to another bank')
    if proof.get('cache_schema_version')!=2:raise ValueError('native cache requires geometry-hash schema 2; rebuild cache')
    from hybrid_assets import mesh_sources,validate_asset
    if proof.get('mesh_sources')!=mesh_sources(bank):raise ValueError('source mesh dependencies changed')
    rows=proof['assets'];expected={d['id']:d for d in bank['designs']}
    if len(rows)!=len(expected) or {r['design'] for r in rows}!=set(expected):raise ValueError('incomplete native cache')
    selected=set(expected) if design_ids is None else set(design_ids)
    if not selected.issubset(expected):raise ValueError('unknown selected cache design')
    for row in rows:
        if row['design'] not in selected:continue
        usd=Path(row['usd']).resolve();directory=usd.parent
        if not usd.is_relative_to(cache.resolve()) or directory.name!=row['design'] or row['urdf_sha256']!=expected[row['design']]['urdf_sha256']:
            raise ValueError('native cache asset identity mismatch')
        if (directory/'.asset_hash').read_text().strip()!=row['converter_hash']:
            raise ValueError('native converter hash changed')
        if not (directory/'config.yaml').is_file() or sha(usd)!=row['usd_sha256']:
            raise ValueError('native converted asset is incomplete/changed')
        if sha(expected[row['design']]['urdf'])!=row['urdf_sha256']:raise ValueError('source URDF changed')
        validate_asset(row)
    return proof

def cache_directories(cache,bank_path,design_ids=None,require_complete=False):
    """Use only completed proof entries; never pick abandoned partial batches."""
    cache=Path(cache)
    if (cache/'completed.json').is_file():
        proof=validate_cache(cache,bank_path,design_ids)
        return [Path(row['usd']).parent for row in proof['assets'] if design_ids is None or row['design'] in design_ids]
    if require_complete:raise ValueError('full-body persistent training requires a completed native cache proof')
    return [p.parent for p in cache.rglob('.asset_hash')
            if (p.parent/'config.yaml').is_file() and any(p.parent.glob('*.usd'))]

def training_attempt_started(root):
    """Cache conversion is not PPO; unknown/malformed launches fail closed."""
    for path in (Path(root)/'attempts').glob('*/train/launch.json'):
        try:command=json.loads(path.read_text(encoding='utf8'))['command']
        except (ValueError,KeyError,OSError):return True
        scripts=[Path(arg).name for arg in command if isinstance(arg,str) and arg.endswith('.py')]
        if scripts!=['prepare_native_cache.py']:return True
    return False

def ensure_cache(root,bank_path,python,kit,environment,run_child):
    root=Path(root)
    for path in sorted(root.glob('native_cache_*/completed.json'),reverse=True):
        try:validate_cache(path.parent,bank_path)
        except (ValueError,OSError,KeyError):continue
        return path.parent
    tag=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cache=root/f'native_cache_{tag}';attempt=root/'attempts'/f'cache_{tag}';attempt.mkdir()
    script=Path(__file__).with_name('prepare_native_cache.py')
    command=[python,'-u',str(script),'--bank',str(bank_path),'--cache-root',str(cache),
        '--output-dir',str(attempt/'train/result'),'--headless','--device','cuda:0','--kit_args',kit,
        '--pause-root',str(root)]
    for previous in sorted((root/'attempts').glob('cache_*/train/result')):
        if previous!=attempt/'train/result':command+=['--reuse-proof-root',str(previous)]
    run_child(command,attempt/'train',environment)
    validate_cache(cache,bank_path)
    return cache
