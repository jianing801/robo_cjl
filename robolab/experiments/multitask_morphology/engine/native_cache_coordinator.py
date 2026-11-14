"""Bounded fresh-process batches isolate native URDF exporter lifetime faults."""
import json,os,subprocess,sys,time
from pathlib import Path

def reuse_verified_batches(roots,bank,bank_sha,cache):
    """Copy only completed, hash-verified native batches; keep failures intact."""
    import shutil
    from hybrid_native_cache import sha
    from hybrid_assets import validate_asset,mesh_sources
    expected={d['id']:d for d in bank['designs']};rows={}
    for root in roots:
        for proof_path in sorted(Path(root).glob('batch_*/summary.json')):
            proof=json.loads(proof_path.read_text(encoding='utf8'))
            if proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE' or proof.get('cache_schema_version')!=2 or proof.get('bank_sha256')!=bank_sha:continue
            subset={'designs':[expected[row['design']] for row in proof['assets'] if row['design'] in expected]}
            if proof.get('mesh_sources')!=mesh_sources(subset):raise ValueError('native batch source meshes changed')
            for row in proof['assets']:
                key=row['design'];source=Path(row['usd'])
                if key not in expected or key in rows:continue
                validate_asset(row)
                if row['urdf_sha256']!=expected[key]['urdf_sha256'] or sha(source)!=row['usd_sha256']:
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

def coordinate(args,argv):
    from hybrid_support import write_new
    from hybrid_contract import sha
    bank_path=Path(args.bank).resolve();task=json.loads(bank_path.read_text(encoding='utf8')).get('task_key','stairs')
    if task=='run':from amp_bank import load_configurable_bank
    else:from hybrid_bank import load_configurable_bank
    bank=load_configurable_bank(bank_path);bank_sha=sha(bank_path)
    from hybrid_assets import mesh_sources,validate_asset
    meshes=mesh_sources(bank)
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    cache=Path(args.cache_root).resolve();cache.mkdir(parents=True,exist_ok=False)
    ids=[d['id'] for d in bank['designs']]
    rows=reuse_verified_batches(args.reuse_proof_root,bank,bank_sha,cache)
    seen={row['design'] for row in rows};pending=[key for key in ids if key not in seen]
    if rows:print('NATIVE_CACHE_REUSED',len(rows),'/',len(ids),flush=True)
    try:
        for offset in range(0,len(pending),4):
            chunk=pending[offset:offset+4];number=offset//4
            for attempt in range(3):
                if args.pause_root and (Path(args.pause_root)/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
                tag=f'batch_{number:04d}_attempt_{attempt+1}'
                batch_output=output/tag;batch_cache=cache/tag
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('prepare_native_cache.py')),*argv,
                    '--batch-worker','--design-ids',','.join(chunk),'--expected-bank-sha',bank_sha,
                    '--cache-root',str(batch_cache),'--output-dir',str(batch_output)]
                kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
                print('CACHE_BATCH_START',number,attempt+1,'designs',len(chunk),flush=True)
                with (output/f'{tag}.stdout.log').open('x',encoding='utf8') as out,(output/f'{tag}.stderr.log').open('x',encoding='utf8') as err:
                    child=subprocess.Popen(cmd,stdout=out,stderr=err,**kwargs);started=time.monotonic();timeout=False
                    try:
                        while child.poll() is None:
                            if args.pause_root and (Path(args.pause_root)/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
                            if time.monotonic()-started>180:
                                timeout=True;stop_tree(child);break
                            time.sleep(.5)
                    except BaseException:
                        stop_tree(child);raise
                summary=batch_output/'summary.json';failure=batch_output/'failure.json'
                if failure.exists():raise RuntimeError(f'Native cache batch failed: {failure.read_text(encoding="utf8")}')
                if summary.exists():
                    proof=json.loads(summary.read_text(encoding='utf8'))
                    if (proof.get('result')!='NATIVE_ASSET_CACHE_COMPLETE' or proof.get('cache_schema_version')!=2
                        or proof.get('bank_sha256')!=bank_sha or len(proof['assets'])!=len(chunk)
                        or {r['design'] for r in proof['assets']}!=set(chunk)):
                        raise ValueError('native cache batch proof mismatch')
                    for row in proof['assets']:validate_asset(row)
                    if proof.get('mesh_sources')!=mesh_sources({'designs':[bank_design for bank_design in bank['designs'] if bank_design['id'] in chunk]}):
                        raise ValueError('native cache batch mesh identity mismatch')
                    if child.returncode not in (0,None) and not timeout:raise RuntimeError('Cache batch exited abnormally')
                    rows.extend(proof['assets'])
                    print('NATIVE_CACHE_PROGRESS',len(rows),'/',len(ids),flush=True);break
                if not timeout:raise RuntimeError(f'Native cache worker exited {child.returncode}; inspect {tag} logs')
                # Preserve all partial assets and logs; retry in a new process/directory.
                tail=(output/f'{tag}.stderr.log').read_text(encoding='utf8',errors='replace')[-2000:]
                print('CACHE_BATCH_TIMEOUT; preserved evidence',tag,tail,flush=True)
                if attempt==2:raise TimeoutError('Native conversion failed three bounded attempts')
        if meshes!=mesh_sources(bank):raise ValueError('source meshes changed during native conversion')
        proof=dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,mesh_sources=meshes,bank_sha256=bank_sha,task=task,
            designs=len(rows),assets=rows,batch_size=4,per_batch_timeout_seconds=180)
        write_new(cache/'completed.json',proof);write_new(output/'summary.json',proof)
    except BaseException:
        import traceback
        write_new(output/'failure.json',dict(traceback=traceback.format_exc()));raise
