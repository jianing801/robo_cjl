"""Convert a bounded number of native assets without creating a simulation scene."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
from contract import load_plan,write_new

def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--task',choices=['stairs','backpack','run'],default='stairs')
    p.add_argument('--design-ids')
    p.add_argument('--output',required=True)
    p.add_argument('--cache-root',action='append',default=[])
    p.add_argument('--max-conversions',type=int,default=8)
    AppLauncher.add_app_launcher_args(p)
    a=p.parse_args()
    plan=load_plan(a.plan);spec=plan['teachers'][a.task];code=Path(spec['code_root'])
    sys.path[:0]=[str(code/'scripts/tools'),str(Path(spec['runtime_dir'])),str(code),str(code.parent/'rsl_rl')]
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    a.kit_args=(a.kit_args or '')+f' --vulkan --/renderer/multiGpu/enabled=false --portable-root {output/"portable"} --portable'
    app=AppLauncher(a).app
    try:
        import isaaclab_tasks,robolab.tasks
        s=importlib.util.spec_from_file_location('shared_importer',HERE.parent/'engine/hybrid_importer.py')
        adapter=importlib.util.module_from_spec(s);s.loader.exec_module(adapter)
        write_new(output/'importer_audit.json',adapter.install_local_importer_adapter())
        if a.task=='run':
            from amp_bank import load_configurable_bank
            from amp_mixed_runtime import configure_spatial
            from amp_runtime import configure_amp
            configurator=configure_amp
        else:
            from hybrid_bank import load_configurable_bank
            from hybrid_runtime import configure_spatial
            configurator=None
            if a.task=='backpack':
                from hybrid_backpack import configure_backpack
                configurator=configure_backpack
        from isaaclab.sim.converters import UrdfConverter
        from rollout import reuse_assets
        bank=load_configurable_bank(spec['bank']);by_id={d['id']:d for d in bank['designs']}
        selected=a.design_ids.split(',') if a.design_ids else spec['design_ids']
        if not set(selected)<=set(spec['design_ids']):raise ValueError('Unknown cache preparation design')
        designs=[by_id[k] for k in selected]
        options=SimpleNamespace(num_envs=len(designs),device=a.device,asset_cache=None,
            allow_cache_misses=True,torque_soft_constraint=None,reference_library=spec.get('reference_library'),full_body_torque=spec['full_body_torque'])
        kwargs={'configurator':configurator} if configurator else {}
        cfg,_,_=configure_spatial(options,bank,designs,output,**kwargs)
        audit=reuse_assets(cfg,[*a.cache_root,*spec['asset_cache']],output,a.task,designs)
        converted=[]
        missing=sum(not row['reused'] for row in audit)
        for asset,row in zip(cfg.scene.robot.spawn.assets_cfg,audit):
            if row['reused']:continue
            print('CONVERT_BEGIN',row['design'],flush=True)
            converter=UrdfConverter(asset)
            if not Path(converter.usd_path).is_file():raise RuntimeError('Native converter produced no USD')
            converted.append(row['design'])
            print('CONVERT_COMPLETE',row['design'],flush=True)
            if len(converted)>=a.max_conversions:break
        write_new(output/'summary.json',dict(converted=converted,remaining=missing-len(converted),assets=audit))
    except BaseException:
        import traceback
        write_new(output/'failure.json',dict(traceback=traceback.format_exc()))
        traceback.print_exc();os._exit(1)
    # USDs and converter metadata are persisted before exiting this isolated importer.
    sys.stdout.flush();os._exit(0)

if __name__=='__main__':main()
