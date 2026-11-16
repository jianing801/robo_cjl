"""Convert all required assets before persistent physics scene rebuilding."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT.parent/'rsl_rl'),str(HERE.parent/'amp_engine')]

class CoordinatorImportGuard:
    """协调进程禁止加载仿真原生组件；CPU模型依赖仍正常导入。"""
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'isaaclab','isaaclab_tasks','isaacsim','carb','omni','pxr'}:
            raise ImportError('USD coordinator must not load simulator modules: '+fullname)
        return None

def main():
    print('CACHE_PHASE ENTRY_READY',flush=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',required=True);parser.add_argument('--cache-root',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--pause-root')
    parser.add_argument('--batch-size',type=int,default=0,help='0: continuous conversion; positive: optional planned process recycle interval')
    parser.add_argument('--reuse-proof-root',action='append',default=[])
    parser.add_argument('--stall-timeout',type=float,default=30,help='Seconds without a completed asset after startup')
    parser.add_argument('--startup-timeout',type=float,default=300)
    parser.add_argument('--batch-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--design-ids',help=argparse.SUPPRESS)
    parser.add_argument('--design-ids-file',help=argparse.SUPPRESS)
    parser.add_argument('--expected-bank-sha',help=argparse.SUPPRESS)
    if '--batch-worker' not in sys.argv[1:]:
        # 协调器只解析并转发原生启动参数，绝不导入AppLauncher/Carb/Omni。
        # 真正的导出子进程仍使用原生解析器和原样argv，仿真配置不变。
        for option in ('headless','enable_cameras','xr','cpu','verbose','info','anim_recording_enabled'):
            parser.add_argument('--'+option,action='store_true')
        for option in ('device','experience','rendering_mode','kit_args'):
            parser.add_argument('--'+option)
        parser.add_argument('--livestream',type=int)
        parser.add_argument('--anim_recording_start_time',type=float)
        parser.add_argument('--anim_recording_stop_time',type=float)
        args=parser.parse_args()
        sys.meta_path.insert(0,CoordinatorImportGuard())
        print('CACHE_PHASE COORDINATOR_PYTHON_ONLY',flush=True)
        from native_cache_coordinator import coordinate
        coordinate(args,sys.argv[1:]);return
    print('CACHE_PHASE IMPORTING_ISAAC_WORKER_ONLY',flush=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    from hybrid_support import write_new
    from hybrid_contract import sha
    bank_path=Path(args.bank).resolve();task=json.loads(bank_path.read_text(encoding='utf8')).get('task_key','stairs')
    if sha(bank_path)!=args.expected_bank_sha:raise ValueError('bank changed during cache conversion')
    bank=json.loads(bank_path.read_text(encoding='utf8'))
    selected=json.loads(Path(args.design_ids_file).read_text(encoding='utf8')) if args.design_ids_file else args.design_ids.split(',');lookup={d['id']:d for d in bank['designs']}
    if len(set(selected))!=len(selected) or not set(selected).issubset(lookup):raise ValueError('unknown cache design')
    bank['designs']=[lookup[key] for key in selected]
    from hybrid_assets import mesh_sources
    meshes=None  # 共享源已由协调器校验；避免启动前遍历全部待转换结构。
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    cache=Path(args.cache_root).resolve();cache.mkdir(parents=True,exist_ok=False)
    args.kit_args=(args.kit_args or '')+f' --portable-root {output/"portable"} --portable'
    app=AppLauncher(args).app;failed=True
    try:
        import isaaclab_tasks,robolab.tasks
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        from hybrid_importer import install_local_importer_adapter
        write_new(output/'importer_audit.json',install_local_importer_adapter())
        from isaaclab.sim.converters import UrdfConverter
        import omni.kit.undo
        import gc
        from hybrid_assets import asset_files
        native={'stairs':'RPO-Rough','backpack':'RPO-Flat-Backpack-Compliant','run':'RPO-AMP'}[task]
        cfg=load_cfg_from_registry(native,'env_cfg_entry_point');rows=[]
        for index,design in enumerate(bank['designs']):
            if args.pause_root and (Path(args.pause_root)/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
            if sha(design['urdf'])!=design['urdf_sha256']:raise ValueError('changed URDF before conversion')
            spawn=cfg.scene.robot.spawn.copy()
            spawn.asset_path=design['urdf'];spawn.usd_dir=str(cache/design['id'])
            if task=='backpack':spawn.merge_fixed_joints=False
            spawn.force_usd_conversion=False
            # No GUI undo references should retain old conversion robot models.
            with omni.kit.undo.disabled():
                converter=UrdfConverter(spawn)
            key=AssetConverterBase._config_to_hash(spawn)
            if (Path(spawn.usd_dir)/'.asset_hash').read_text().strip()!=key or not Path(converter.usd_path).is_file():
                raise ValueError('native cache conversion did not complete')
            row=dict(design=design['id'],converter_hash=key,urdf_sha256=sha(design['urdf']),usd=converter.usd_path,usd_sha256=sha(converter.usd_path))
            del converter
            gc.collect()
            for _ in range(3):app.update()
            row['files_sha256']=asset_files(spawn.usd_dir);rows.append(row)
            # Publish only a fully hashed asset; a killed writer leaves only .tmp.
            from native_cache_coordinator import publish_asset_proof
            publish_asset_proof(output/'assets'/f'{design["id"]}.json',dict(
                result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,
                mesh_sources=mesh_sources({'designs':[design]}),bank_sha256=sha(bank_path),
                task=task,designs=1,assets=[row]))
            print('NATIVE_ASSET_READY',index+1,'/',len(bank['designs']),design['id'],flush=True)
        meshes=mesh_sources(bank)
        summary=dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,mesh_sources=meshes,bank_sha256=sha(bank_path),task=task,
            designs=len(rows),assets=rows)
        write_new(output/'summary.json',summary);failed=False
    except BaseException as error:
        from native_cache_coordinator import exception_text
        text=exception_text(error)
        write_new(output/'failure.json',dict(traceback=text))
        print(text,file=sys.stderr,flush=True)
        raise
    finally:
        if failed:
            import atexit,os
            atexit.register(lambda:os._exit(1))
        app.close()

if __name__=='__main__':
    import faulthandler
    faulthandler.enable()
    # 健康协调器以心跳报告状态，不再每分钟打印容易误解的Timeout栈。
    # 原生导出worker仍保留周期线程栈和崩溃诊断。
    if '--batch-worker' in sys.argv[1:]:
        faulthandler.dump_traceback_later(60,repeat=True)
    try:
        main()
    except SystemExit:
        raise
    except BaseException as error:
        from native_cache_coordinator import exception_text
        text=exception_text(error)
        print(text,file=sys.stderr,flush=True)
        outputs=[sys.argv[i+1] for i,arg in enumerate(sys.argv[:-1]) if arg=='--output-dir']
        if outputs:
            try:
                folder=Path(outputs[-1]);folder.mkdir(parents=True,exist_ok=True)
                target=folder/'failure.json'
                if not target.exists():target.write_text(json.dumps(dict(traceback=text)),encoding='utf8')
            except OSError as recording_error:
                print('CACHE_FAILURE_RECORD_WRITE_FAILED',repr(recording_error),file=sys.stderr,flush=True)
        raise SystemExit(130 if isinstance(error,KeyboardInterrupt) else 1)
    finally:
        faulthandler.cancel_dump_traceback_later()
