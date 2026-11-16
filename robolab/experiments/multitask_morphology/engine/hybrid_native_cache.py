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
                print('CACHE_VALIDATE_WAIT',f'{time.monotonic()-started:.0f}s','校验进行中；阶段与进度见最近一条进度输出',flush=True)
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
    from hybrid_assets import mesh_sources,validate_asset,progress
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
    progress('CACHE_VALIDATE_PROGRESS',0,len(selected))
    for row in rows:
        if row['design'] not in selected:continue
        usd=Path(row['usd']).resolve();directory=usd.parent
        if not usd.is_relative_to(cache.resolve()) or directory.name!=row['design'] or row['urdf_sha256']!=expected[row['design']]['urdf_sha256']:
            raise ValueError('native cache asset identity mismatch')
        if not (directory/'config.yaml').is_file():raise ValueError('native converted asset is incomplete/changed')
        if sha(expected[row['design']]['urdf'])!=row['urdf_sha256']:raise ValueError('source URDF changed')
        validate_asset(row)
        done+=1
        progress('CACHE_VALIDATE_PROGRESS',done,len(selected))
    return proof

@validation_heartbeat
def validate_training_cache(cache,bank_path,design_ids):
    """正式训练快检：信任已有完成证明，不重读USD内容；后续检查元数据变化。"""
    import os,stat
    from hybrid_assets import progress
    cache=Path(cache).resolve();bank_path=Path(bank_path)
    bank_bytes=bank_path.read_bytes();bank=json.loads(bank_bytes)
    proof_bytes=(cache/'completed.json').read_bytes();proof=json.loads(proof_bytes)
    bank_sha=hashlib.sha256(bank_bytes).hexdigest();proof_sha=hashlib.sha256(proof_bytes).hexdigest()
    if (proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE' or proof.get('cache_schema_version')!=2
            or proof.get('bank_sha256')!=bank_sha):raise ValueError('training cache belongs to another bank or lacks completion proof')
    expected={d['id']:d for d in bank['designs']};rows=proof['assets']
    if len(rows)!=len(expected) or {r['design'] for r in rows}!=set(expected):raise ValueError('incomplete native cache')
    selected=set(design_ids)
    if not selected or not selected.issubset(expected):raise ValueError('unknown/empty selected training designs')
    key=hashlib.sha256(json.dumps(sorted(selected)).encode()).hexdigest()
    receipt_dir=cache/'.training_fast_checks';receipt_path=receipt_dir/(key+'.json')
    if receipt_dir.is_symlink() or receipt_path.is_symlink():raise ValueError('training receipt must stay inside cache')
    previous=json.loads(receipt_path.read_text(encoding='utf8')) if receipt_path.is_file() else None
    binding=dict(version=1,mode='trusted_completed_proof_and_metadata',bank_sha256=bank_sha,proof_sha256=proof_sha)
    if previous is not None and any(previous.get(k)!=v for k,v in binding.items()):
        raise ValueError('training fast-check receipt identity changed; perform full cache audit')
    signatures={}
    def inspect(path):
        path=Path(path);value=path.stat()
        if not stat.S_ISREG(value.st_mode) or value.st_size<=0:raise ValueError('missing/empty training asset: '+str(path))
        # 增加硬链接会改变ctime和nlink，不改变内容；不能据此误报资产变化。
        signatures[str(path)]=[value.st_size,value.st_mtime_ns,value.st_dev,value.st_ino]
    progress('TRAIN_CACHE_FAST_PROGRESS',0,len(selected))
    done=0
    for row in rows:
        if row['design'] not in selected:continue
        usd=Path(row['usd']).resolve();directory=usd.parent
        if not usd.is_relative_to(cache) or directory.name!=row['design'] or row['urdf_sha256']!=expected[row['design']]['urdf_sha256']:
            raise ValueError('training cache asset identity mismatch')
        files=row.get('files_sha256',{})
        if not {usd.name,'.asset_hash','config.yaml'}.issubset(files) or files[usd.name]!=row['usd_sha256']:
            raise ValueError('training cache lacks consistent layer proof')
        # 小型URDF和配置仍核对内容；大型USD只读文件元数据。
        urdf=Path(expected[row['design']]['urdf'])
        if sha(urdf)!=row['urdf_sha256']:raise ValueError('source URDF changed: '+str(urdf))
        inspect(urdf)
        for name in files:
            path=(directory/name).resolve()
            if not path.is_relative_to(directory):raise ValueError('training cache dependency escapes asset directory')
            inspect(path)
            if path.suffix.lower()!='.usd' and sha(path)!=files[name]:
                raise ValueError('training cache metadata changed: '+str(path))
        if (directory/'.asset_hash').read_text().strip()!=row['converter_hash']:
            raise ValueError('native converter hash changed')
        done+=1;progress('TRAIN_CACHE_FAST_PROGRESS',done,len(selected))
    meshes=proof.get('mesh_sources')
    if not isinstance(meshes,dict) or not meshes:raise ValueError('missing source mesh identity')
    for path in meshes:inspect(path)
    if previous is not None and previous.get('files')!=signatures:
        raise ValueError('training cache file size/mtime/inode changed; perform full cache audit')
    # 只在成功快检后记录基线。旧缓存首次以已有完成证明为信任起点，不冒称全量复验。
    if previous is None:
        receipt_dir.mkdir(exist_ok=True)
        temporary=receipt_dir/(key+f'.{os.getpid()}.{time.time_ns()}.tmp')
        temporary.write_text(json.dumps(dict(binding,files=signatures),sort_keys=True),encoding='utf8')
        os.replace(temporary,receipt_path)
    return proof


def cache_directories(cache,bank_path,design_ids=None,require_complete=False,training_fast=False):
    """Use only completed proof entries; never pick abandoned partial batches."""
    cache=Path(cache)
    if (cache/'completed.json').is_file():
        proof=(validate_training_cache(cache,bank_path,design_ids) if training_fast
               else validate_cache(cache,bank_path,design_ids))
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
