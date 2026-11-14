"""Convert all required assets before persistent physics scene rebuilding."""
import argparse,json,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT.parent/'rsl_rl'),str(HERE.parent/'amp_hybrid')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',required=True);parser.add_argument('--cache-root',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--pause-root')
    parser.add_argument('--reuse-proof-root',action='append',default=[])
    parser.add_argument('--batch-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--design-ids',help=argparse.SUPPRESS)
    parser.add_argument('--expected-bank-sha',help=argparse.SUPPRESS)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if not args.batch_worker:
        from native_cache_coordinator import coordinate
        coordinate(args,sys.argv[1:]);return
    from hybrid_support import write_new
    from hybrid_contract import sha
    bank_path=Path(args.bank).resolve();task=json.loads(bank_path.read_text(encoding='utf8')).get('task_key','stairs')
    if sha(bank_path)!=args.expected_bank_sha:raise ValueError('bank changed during cache conversion')
    bank=json.loads(bank_path.read_text(encoding='utf8'))
    selected=args.design_ids.split(',');lookup={d['id']:d for d in bank['designs']}
    if len(set(selected))!=len(selected) or not set(selected).issubset(lookup):raise ValueError('unknown cache design')
    bank['designs']=[lookup[key] for key in selected]
    from hybrid_assets import mesh_sources
    meshes=mesh_sources(bank)
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
            print('NATIVE_ASSET_READY',index+1,'/',len(bank['designs']),design['id'],flush=True)
        if meshes!=mesh_sources(bank):raise ValueError('source meshes changed during conversion')
        summary=dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,mesh_sources=meshes,bank_sha256=sha(bank_path),task=task,
            designs=len(rows),assets=rows)
        write_new(output/'summary.json',summary);failed=False
    except BaseException:
        write_new(output/'failure.json',dict(traceback=traceback.format_exc()));traceback.print_exc();raise
    finally:
        if failed:
            import atexit,os
            atexit.register(lambda:os._exit(1))
        app.close()

if __name__=='__main__':main()
