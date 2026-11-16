"""Continuous native conversion with durable per-asset proofs and a progress watchdog."""
import json,os,subprocess,sys,time
from pathlib import Path

def exception_text(error):
    # 不读取源码行，避免共享存储故障时traceback自身的linecache也被阻塞。
    lines=['Traceback (most recent call last):']
    tb=error.__traceback__
    while tb is not None:
        code=tb.tb_frame.f_code
        lines.append(f'  File "{code.co_filename}", line {tb.tb_lineno}, in {code.co_name}')
        tb=tb.tb_next
    lines.append(type(error).__name__+': '+str(error))
    return '\n'.join(lines)

def reuse_verified_batches(roots,bank,bank_sha,cache):
    """Copy only completed, hash-verified native batches; keep failures intact."""
    import shutil
    from hybrid_assets import validate_asset,mesh_sources
    expected={d['id']:d for d in bank['designs']};rows={}
    for root in roots:
        for proof_path in [*sorted(Path(root).glob('batch_*/summary.json')), *sorted(Path(root).glob('batch_*/assets/*.json'))]:
            print('CACHE_REUSE_PROOF',str(proof_path),flush=True)
            proof=json.loads(proof_path.read_text(encoding='utf8'))
            if proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE' or proof.get('cache_schema_version')!=2 or proof.get('bank_sha256')!=bank_sha:continue
            keys=[row['design'] for row in proof['assets']]
            if keys and all(key in rows for key in keys):continue
            subset={'designs':[expected[row['design']] for row in proof['assets'] if row['design'] in expected]}
            if proof.get('mesh_sources')!=mesh_sources(subset):raise ValueError('native batch source meshes changed')
            for row in proof['assets']:
                key=row['design'];source=Path(row['usd'])
                if key not in expected or key in rows:continue
                print('CACHE_REUSE_ASSET_BEGIN',key,str(source),flush=True)
                validate_asset(row)
                # validate_asset已经核对USD哈希，不再重复读取大文件。
                if row['urdf_sha256']!=expected[key]['urdf_sha256']:
                    raise ValueError('changed native batch; do not reuse')
                if (source.parent/'.asset_hash').read_text().strip()!=row['converter_hash'] or not (source.parent/'config.yaml').is_file():
                    raise ValueError('incomplete native batch metadata')
                target=cache/'reused'/key;target.parent.mkdir(exist_ok=True)
                def copy_file(src,dst):
                    if Path(src).suffix.lower()=='.usd':
                        try:os.link(src,dst);return dst
                        except OSError:pass
                    return shutil.copy2(src,dst)
                shutil.copytree(source.parent,target,copy_function=copy_file)
                rows[key]=dict(row,usd=str(target/source.name))
                print('CACHE_REUSE_PROGRESS',len(rows),'/',len(expected),flush=True)
    return list(rows.values())

def stop_tree(child):
    import psutil
    try:
        parent=psutil.Process(child.pid);children=parent.children(recursive=True)
        parent.kill()
        for process in children:
            try:process.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs([parent,*children],timeout=10)
    except psutil.NoSuchProcess:pass

def publish_asset_proof(path,value):
    """Atomic publication makes completed assets recoverable before worker exit."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_suffix('.json.tmp')
    with temporary.open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def read_asset_proofs(folder,cache,bank,bank_sha,seen):
    from hybrid_assets import validate_asset,mesh_sources
    expected={d['id']:d for d in bank['designs']};rows=[]
    for path in sorted((Path(folder)/'assets').glob('*.json')):
        if path.stem in seen:continue
        proof=json.loads(path.read_text(encoding='utf8'))
        if (proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE' or proof.get('cache_schema_version')!=2
                or proof.get('bank_sha256')!=bank_sha or len(proof.get('assets',[]))!=1):
            raise ValueError('invalid per-asset conversion proof')
        row=proof['assets'][0];key=row['design']
        if key!=path.stem or key not in expected or key in seen:
            raise ValueError('unexpected per-asset design')
        if not Path(row['usd']).resolve().is_relative_to(Path(cache).resolve()):
            raise ValueError('conversion proof escapes this worker cache')
        if row['urdf_sha256']!=expected[key]['urdf_sha256']:
            raise ValueError('changed URDF in per-asset proof')
        if proof.get('mesh_sources')!=mesh_sources({'designs':[expected[key]]}):
            raise ValueError('changed mesh in per-asset proof')
        validate_asset(row);seen.add(key);rows.append(row)
    return rows


def coordinate(args,argv):
    from hybrid_support import write_new
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    print('CACHE_PHASE VALIDATING_BANK',str(args.bank),flush=True)
    try:
        return _coordinate(args,argv)
    except BaseException as error:
        text=exception_text(error)
        print(text,file=sys.stderr,flush=True)
        if not (output/'failure.json').exists():
            write_new(output/'failure.json',dict(traceback=text))
        raise


def _coordinate(args,argv):
    from hybrid_support import write_new
    from hybrid_contract import sha
    bank_path=Path(args.bank).resolve();task=json.loads(bank_path.read_text(encoding='utf8')).get('task_key','stairs')
    if task=='run':from amp_bank import load_configurable_bank
    else:from hybrid_bank import load_configurable_bank
    bank=load_configurable_bank(bank_path);bank_sha=sha(bank_path)
    from hybrid_assets import mesh_sources
    print('CACHE_PHASE HASHING_SOURCE_MESHES',len(bank['designs']),flush=True)
    meshes=mesh_sources(bank)
    print('CACHE_PHASE SOURCE_MESHES_VERIFIED',len(meshes),flush=True)
    size=args.batch_size
    if not 0<=size<=len(bank['designs']):raise ValueError('invalid conversion recycle interval')
    startup=getattr(args,'startup_timeout',300);stall=getattr(args,'stall_timeout',30)
    if startup<=0 or stall<=0:raise ValueError('watchdog timeouts must be positive')
    output=Path(args.output_dir).resolve()
    cache=Path(args.cache_root).resolve();cache.mkdir(parents=True,exist_ok=False)
    ids=[d['id'] for d in bank['designs']]
    print('CACHE_PHASE REUSING_PREVIOUS_ASSETS',len(args.reuse_proof_root),flush=True)
    rows=reuse_verified_batches(args.reuse_proof_root,bank,bank_sha,cache)
    seen={row['design'] for row in rows};failures=0;number=0
    if rows:print('NATIVE_CACHE_REUSED',len(rows),'/',len(ids),flush=True)
    try:
        while len(seen)<len(ids):
            if args.pause_root and (Path(args.pause_root)/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
            pending=[key for key in ids if key not in seen];chunk=pending[:size] if size else pending
            tag=f'batch_{number:04d}_attempt_1';number+=1
            batch_output=output/tag;batch_cache=cache/tag
            selected=output/f'{tag}.designs.json';write_new(selected,chunk)
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('prepare_native_cache.py')),*argv,
                 '--batch-worker','--design-ids-file',str(selected),'--expected-bank-sha',bank_sha,
                 '--cache-root',str(batch_cache),'--output-dir',str(batch_output)]
            kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            print('CACHE_SESSION_START',number,'pending',len(chunk),flush=True)
            before=len(seen);timeout=False
            with (output/f'{tag}.stdout.log').open('x',encoding='utf8') as out,(output/f'{tag}.stderr.log').open('x',encoding='utf8') as err:
                child=subprocess.Popen(cmd,stdout=out,stderr=err,**kwargs)
                last_progress=time.monotonic();progressed=False
                try:
                    while True:
                        if args.pause_root and (Path(args.pause_root)/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
                        new=read_asset_proofs(batch_output,batch_cache,bank,bank_sha,seen)
                        if new:
                            rows.extend(new);last_progress=time.monotonic();progressed=True
                            print('NATIVE_CACHE_PROGRESS',len(rows),'/',len(ids),flush=True)
                        if child.poll() is not None:break
                        limit=stall if progressed else startup
                        if time.monotonic()-last_progress>limit:
                            timeout=True;stop_tree(child);break
                        time.sleep(.5)
                    rows.extend(read_asset_proofs(batch_output,batch_cache,bank,bank_sha,seen))
                except BaseException:
                    stop_tree(child);raise
            failure=batch_output/'failure.json'
            if failure.exists():
                raise RuntimeError('Native converter recorded failure: '+failure.read_text(encoding='utf8'))
            if len(seen)==before:failures+=1
            else:failures=0
            if set(chunk).issubset(seen) and child.returncode==0 and not timeout:
                continue
            tail=(output/f'{tag}.stderr.log').read_text(encoding='utf8',errors='replace')[-2000:]
            print('CACHE_SESSION_INTERRUPTED','timeout' if timeout else child.returncode,
                  'verified',len(seen)-before,'preserved evidence',tag,tail,flush=True)
            if failures>=3:raise RuntimeError('Native converter failed three consecutive sessions without progress')
            # The next process only receives designs with no verified asset proof.
        if meshes!=mesh_sources(bank):raise ValueError('source meshes changed during native conversion')
        by_id={row['design']:row for row in rows}
        if set(by_id)!=set(ids) or len(rows)!=len(ids):raise ValueError('cache coverage mismatch')
        proof=dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,mesh_sources=meshes,
            bank_sha256=bank_sha,task=task,designs=len(rows),assets=[by_id[key] for key in ids],
            conversion_protocol='continuous_asset_proofs_v1',recycle_interval=size,
            startup_timeout_seconds=startup,stall_timeout_seconds=stall,sessions=number)
        write_new(cache/'completed.json',proof);write_new(output/'summary.json',proof)
    except BaseException as error:
        write_new(output/'failure.json',dict(traceback=exception_text(error)));raise
