"""Controller-side native cache preparation before persistent scene lifetimes."""
from datetime import datetime
import hashlib,json
from pathlib import Path
import functools,threading,time

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validation_heartbeat(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        stop=threading.Event();started=time.monotonic()
        def report():
            while not stop.wait(10):
                print('CACHE_VALIDATE_WAIT',f'{time.monotonic()-started:.0f}s','校验仍在进行，当前读取位置见最近的BEGIN输出',flush=True)
        thread=threading.Thread(target=report,daemon=True);thread.start()
        try:return function(*args,**kwargs)
        finally:stop.set();thread.join(timeout=1)
    return wrapped

@validation_heartbeat
def validate_cache(cache,bank_path,design_ids=None):
    cache=Path(cache);bank=json.loads(Path(bank_path).read_text(encoding='utf8'))
    proof=json.loads((cache/'completed.json').read_text(encoding='utf8'))
    if proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE' or proof.get('bank_sha256')!=sha(bank_path):
        raise ValueError('native cache belongs to another bank')
    if proof.get('cache_schema_version')!=2:raise ValueError('native cache requires geometry-hash schema 2; rebuild cache')
    from hybrid_assets import mesh_sources,validate_asset
    rows=proof['assets'];expected={d['id']:d for d in bank['designs']}
    if len(rows)!=len(expected) or {r['design'] for r in rows}!=set(expected):raise ValueError('incomplete native cache')
    selected=set(expected) if design_ids is None else set(design_ids)
    if not selected.issubset(expected):raise ValueError('unknown selected cache design')
    print('CACHE_VALIDATE_SOURCE_MESHES',len(selected),flush=True)
    actual_meshes=mesh_sources({'designs':[d for d in bank['designs'] if d['id'] in selected]})
    recorded=proof.get('mesh_sources',{})
    if any(recorded.get(path)!=digest for path,digest in actual_meshes.items()):
        raise ValueError('source mesh dependencies changed')
    if design_ids is None and recorded!=actual_meshes:raise ValueError('source mesh dependencies changed')
    done=0
    print('CACHE_VALIDATE_BEGIN',len(selected),str(cache),flush=True)
    for row in rows:
        if row['design'] not in selected:continue
        usd=Path(row['usd']).resolve();directory=usd.parent
        if not usd.is_relative_to(cache.resolve()) or directory.name!=row['design'] or row['urdf_sha256']!=expected[row['design']]['urdf_sha256']:
            raise ValueError('native cache asset identity mismatch')
        if not (directory/'config.yaml').is_file():raise ValueError('native converted asset is incomplete/changed')
        if sha(expected[row['design']]['urdf'])!=row['urdf_sha256']:raise ValueError('source URDF changed')
        print('CACHE_VALIDATE_ASSET_BEGIN',row['design'],str(usd),flush=True)
        validate_asset(row)
        done+=1
        print('CACHE_VALIDATE_PROGRESS',done,'/',len(selected),flush=True)
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
    # 子进程已逐资产校验；只交叉确认两份原子完成报告，不再次全量读取USD。
    print('CACHE_FINALIZING_REPORTS',str(cache),flush=True)
    proof=json.loads((cache/'completed.json').read_text(encoding='utf8'))
    summary=json.loads((attempt/'train/result/summary.json').read_text(encoding='utf8'))
    bank=json.loads(Path(bank_path).read_text(encoding='utf8'))
    expected={d['id'] for d in bank['designs']}
    rows=proof.get('assets',[])
    if (proof!=summary or proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE'
            or proof.get('cache_schema_version')!=2 or proof.get('bank_sha256')!=sha(bank_path)
            or len(rows)!=len(expected) or {r['design'] for r in rows}!=expected):
        raise ValueError('native coordinator completion reports disagree')
    print('CACHE_FINALIZED',len(rows),'/',len(expected),flush=True)
    return cache
