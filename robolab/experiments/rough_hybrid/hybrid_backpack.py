"""Native backpack task on the same rotating morphology engine as Rough."""
from copy import deepcopy
from pathlib import Path
import torch
from isaaclab.terrains import TerrainImporter
import isaaclab.sim as sim_utils
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from robolab.physical_snapshot import configure_robot
from robolab.tasks.direct.backpack_rpo_flat.env import BackpackRPOFlatEnv
from robolab.tasks.direct.backpack_walk.backpack_urdf import sample_backpack_properties,apply_backpack_properties
from hybrid_runtime import SpatialRoughEnv
from hybrid_support import write_new,serializable
from hybrid_payload import payload_range,payload_segments

class OfflinePayloadPlane(TerrainImporter):
    def import_ground_plane(self,name,size=(2.e6,2.e6)):
        path=self.cfg.prim_path+'/'+name
        if path in self.terrain_prim_paths:raise ValueError('duplicate terrain')
        self.terrain_prim_paths.append(path)
        asset=Path(__file__).resolve().parents[2]/'data/native_probe_ground.usda'
        cfg=sim_utils.GroundPlaneCfg(usd_path=str(asset),physics_material=self.cfg.physics_material,size=size,color=None)
        cfg.func(path,cfg)

class SpatialBackpackEnv(SpatialRoughEnv,BackpackRPOFlatEnv):
    pass

def configure_backpack(options,bank,design,output):
    global _pristine
    if '_pristine' not in globals():
        _pristine=deepcopy(load_cfg_from_registry('RPO-Flat-Backpack-Compliant','env_cfg_entry_point'))
    cfg=deepcopy(_pristine)
    source=load_cfg_from_registry('RPO-Flat-Backpack-Compliant','rsl_rl_cfg_entry_point').to_dict()
    cfg.scene.num_envs=cfg.scene_context.num_envs=options.num_envs
    cfg.seed=source['seed'];cfg.sim.device=options.device or 'cuda:0'
    robot=cfg.scene.robot.copy()
    robot.spawn.asset_path=design['urdf'];robot.spawn.usd_dir=str(output/'usd')
    robot.spawn.merge_fixed_joints=False
    configure_robot(robot,bank['motor_pairs'][design['pair_id']])
    robot.init_state.pos=(*robot.init_state.pos[:2],robot.init_state.pos[2]+design['root_height_compensation_m'])
    cfg.scene.robot=cfg.scene_context.robot=robot
    cfg.scene.terrain.class_type=OfflinePayloadPlane
    cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
    cfg.commands.debug_vis=False
    for name in ('left_feet_scanner','right_feet_scanner','height_scanner'):
        if getattr(cfg.scene,name,None) is not None:getattr(cfg.scene,name).debug_vis=False
    return cfg,source,dict(task='backpack',source_task='RPO-Flat-Backpack-Compliant',
        original_source=serializable(_pristine.to_dict()),payload_schedule='absolute 0/1000/3000; original range',
        preserved_native_rewards_and_observations=True)

@torch.inference_mode()
def apply_payload(raw,output,update):
    values=sample_backpack_properties(raw.num_envs,seed=raw.cfg.seed+update,mass_range_kg=payload_range(update))
    raw.reset()
    audit=apply_backpack_properties(raw.robot,values)
    raw.reset()
    write_new(output/f'payload_{update:06d}.json',dict(update=update,mass_range_kg=payload_range(update),properties=values,audit=audit))

def learn_payload(runner,raw,output,stop):
    for first,end,bounds in payload_segments(runner.current_learning_iteration,stop):
        apply_payload(raw,output,first)
        runner.current_learning_iteration=first
        # Keep PPO optimization outside inference_mode.
        runner.learn(end-first)
