"""Real native teacher/DAgger sampling or closed-loop evaluation, one scene per process."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import traceback

HERE=Path(__file__).resolve().parent
from contract import load_plan,sha,write_new,graph_features,unified_context,input_width


def reuse_assets(cfg,roots,output,task,designs):
    """Reuse ONLY exact native converter hashes, keeping old assets unchanged."""
    from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
    available={}
    for root in roots:
        if not Path(root).exists():continue
        for entry in Path(root).rglob('.asset_hash'):
            if (entry.parent/'config.yaml').is_file() and any(entry.parent.glob('*.usd')):
                available.setdefault(entry.read_text().strip(),entry.parent)
    audit=[]
    for asset,design in zip(cfg.scene.robot.spawn.assets_cfg,designs):
        key=AssetConverterBase._config_to_hash(asset)
        if key in available:
            directory=available[key]
            asset.usd_dir=str(directory)
            asset.force_usd_conversion=False
            audit.append(dict(design=design['id'],cache=str(directory),exact_hash=key,reused=True))
        else:
            directory=Path(output)/'native_assets'/task/design['id']
            asset.usd_dir=str(directory)
            audit.append(dict(design=design['id'],cache=str(directory),exact_hash=key,reused=False))
    return audit


def canonical_input(base,task,indices,static_context,task_count=2):
    import torch
    ids=torch.zeros(len(base),task_count,device=base.device,dtype=base.dtype)
    ids[:,task]=1
    return torch.cat((base,ids,static_context[indices]),-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--task',choices=['stairs','backpack'],required=True)
    p.add_argument('--design-ids',required=True);p.add_argument('--output',required=True)
    p.add_argument('--asset-cache-root');p.add_argument('--student')
    p.add_argument('--mode',choices=['collect','evaluate'],default='collect')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--ppo-channel')
    p.add_argument('--full-body-torque',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--payload-kg',type=float,default=None,help='Explicit fixed backpack mass for evaluation')
    p.add_argument('--trace-stride',type=int,default=0,help='Optional diagnostic snapshots; 0 disables, does not affect actions')
    p.add_argument('--reset-yaw-range',type=float,default=.05,
        help='Explicit initial yaw coverage for real demonstrations; nominal evaluation remains .05 rad')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    if args.video:args.enable_cameras=True
    if min(args.steps,args.stride)<1 or not 0<=args.reset_yaw_range<=.5:raise ValueError('Positive finite sampling budget and bounded reset distribution required')
    plan=load_plan(args.plan);spec=plan['teachers'][args.task]
    code=Path(spec['code_root'])
    sys.path[:0]=[str(HERE),str(code/'scripts/tools'),str(code/'experiments/rough_hybrid'),
                  str(code),str(code.parent/'rsl_rl')]
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    if args.video:
        from shared_video import reuse_shader_cache
        reuse_shader_cache(output)
    if os.name=='nt' and len(str(output/'native_assets/backpack/tr_0123456789ab/configuration/right_ankle_pitch_link.tmp.usd'))>=250:
        raise ValueError('Native USD exporter path exceeds safe Windows length; use a short output root')
    if os.name=='nt' and '--vulkan' not in (args.kit_args or '').split():
        args.kit_args=(args.kit_args or '')+' --vulkan'
    args.kit_args=(args.kit_args or '')+' --/renderer/multiGpu/enabled=false'
    args.kit_args=(args.kit_args or '')+f' --portable-root {output/"portable"} --portable'
    app=AppLauncher(args).app
    raw=None;exit_code=1
    try:
        import torch
        import isaaclab_tasks,robolab.tasks
        from types import SimpleNamespace
        from student import load_student
        # Process-local importer compatibility; installed dependencies are not changed.
        importer_path=HERE.parent/'rough_hybrid/hybrid_importer.py'
        s=importlib.util.spec_from_file_location('shared_student_importer',importer_path)
        adapter=importlib.util.module_from_spec(s);s.loader.exec_module(adapter)
        write_new(output/'importer_audit.json',adapter.install_local_importer_adapter())
        if args.task=='stairs':
            from hybrid_bank import load_configurable_bank
            from hybrid_runtime import configure_spatial,SpatialRoughEnv as Env,SpatialVecEnv
            from hybrid_policy import SpatialActorCritic
            configurator=None
        else:
            from robolab.configurable_morphology import load_configurable_bank
            from spatial_rough_runtime import configure_spatial,SpatialVecEnv
            from spatial_task_runtime import configure_task,SpatialBackpackEnv as Env
            from robolab.spatial_policy import SpatialActorCritic
            configurator=configure_task
        bank=load_configurable_bank(spec['bank'])
        by_id={d['id']:d for d in bank['designs']}
        selected=args.design_ids.split(',')
        if len(set(selected))!=len(selected) or not set(selected)<=set(spec['design_ids']):raise ValueError('Unknown/duplicate selected design')
        designs=[by_id[k] for k in selected];count=len(designs);copies=spec['copies'];n=count*copies
        for design in designs:
            if sha(design['urdf'])!=design['urdf_sha256']:raise ValueError('Physical URDF changed')
        options=SimpleNamespace(num_envs=n,device=args.device,asset_cache=None,allow_cache_misses=True,
                                torque_soft_constraint=None)
        kwargs={'configurator':configurator} if configurator else {}
        cfg,agent,source_contract=configure_spatial(options,bank,designs,output,**kwargs)
        cfg.seed=args.seed;cfg.noise.add_noise=False;cfg.capture_terminal_state=True
        for name,event in list(vars(cfg.events).items()):
            if getattr(event,'mode',None) in ('startup','interval'):setattr(cfg.events,name,None)
        cfg.events.reset_base.params['pose_range']={k:(-.01,.01) if k in ('x','y') else (-args.reset_yaw_range,args.reset_yaw_range) if k=='yaw' else (0.,0.)
            for k in cfg.events.reset_base.params['pose_range']}
        cfg.events.reset_base.params['velocity_range']={k:(0.,0.) for k in cfg.events.reset_base.params['velocity_range']}
        cfg.events.reset_robot_joints.params.update(position_range=(-.01,.01),velocity_range=(0.,0.))
        for motor in cfg.scene.robot.actuators.values():motor.min_delay=0;motor.max_delay=0
        cfg.commands.rel_standing_envs=0.;cfg.commands.rel_heading_envs=1.
        cfg.commands.resampling_time_range=(1000.,1000.)
        cfg.commands.ranges.lin_vel_x=(.5,.5);cfg.commands.ranges.lin_vel_y=(0.,0.)
        cfg.commands.ranges.ang_vel_z=(-.4,.4);cfg.commands.ranges.heading=(0.,0.)
        if args.task=='stairs':
            import isaaclab.terrains as terrains
            from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
            generator=TerrainGeneratorCfg(seed=args.seed,curriculum=True,size=(40.,40.),border_width=2.,
                num_rows=1,num_cols=1,horizontal_scale=.05,vertical_scale=.005,difficulty_range=(.5,.5),use_cache=False,
                sub_terrains={'stairs':terrains.MeshInvertedPyramidStairsTerrainCfg(proportion=1.,
                    step_height_range=(.1,.1),step_width=.25,platform_width=2.,border_width=1.,holes=False)})
            for terrain in (cfg.scene.terrain,cfg.scene_context):
                terrain.terrain_type='generator';terrain.terrain_generator=generator;terrain.max_init_terrain_level=0
        cfg.sim.render_interval=1000000
        cache_roots=[*spec['asset_cache']]
        if args.asset_cache_root:cache_roots.insert(0,args.asset_cache_root)
        if args.full_body_torque:
            import importlib.util as _importlib
            _spec=_importlib.spec_from_file_location('_smoke_full_body',HERE.parent/'rough_hybrid/full_body_torque.py')
            _full=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_full)
            from spatial_rough_runtime import SpatialMotorCfg as FullMotorCfg
            write_new(output/'full_body_configuration.json',_full.activate_for_config(cfg,FullMotorCfg,count))
        cache=reuse_assets(cfg,cache_roots,output,args.task,designs)
        write_new(output/'asset_cache_audit.json',cache)
        class Capture(Env):
            def update_terrain_levels(self,env_ids):return {}
        if args.video:cfg.viewer.resolution=(960,540)
        raw=Capture(cfg,render_mode='rgb_array' if args.video else None)
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        class SamplingVecEnv(SpatialVecEnv):
            def step(self,actions):
                # Native physics, rewards, resets and observations are unchanged.
                # Skip only training-console per-design GPU metric reductions.
                obs,rewards,dones,extras=RslRlVecEnvWrapper.step(self,actions)
                return self._extend(obs),rewards,dones,extras
        env=SamplingVecEnv(raw,None,agent.get('clip_actions'))
        names=bank['canonical_semantic_contract']['joint_names']
        if list(raw.robot.joint_names)!=names:raise ValueError('Actual action order mismatch')
        if abs(raw.step_dt-.02)>1e-10 or abs(raw.action_scale-.25)>1e-10:raise ValueError('Control timing/action scale mismatch')
        expected=torch.tensor(bank['canonical_semantic_contract']['default_joint_pos'],device=raw.device)
        if not torch.allclose(raw.robot.data.default_joint_pos,expected.expand(n,-1),atol=1e-6):raise ValueError('Actual default pose mismatch')
        import omni.usd
        import xml.etree.ElementTree as ET
        stage=omni.usd.get_context().get_stage()
        actual_ids=[stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot').GetAttribute('spatial:design_id').Get() for i in range(n)]
        if actual_ids!=[designs[i%count]['id'] for i in range(n)]:raise ValueError('Physical design/context routing mismatch')
        geometry=[]
        masses=raw.robot.root_physx_view.get_masses().sum(-1)
        for i,design in enumerate(designs):
            if abs(float(masses[i])-design['total_mass_kg'])>2e-4:raise ValueError('Actual base mass differs from audited bank')
            xml=ET.parse(design['urdf']).getroot()
            for joint in ('left_knee_joint','right_knee_joint','left_ankle_pitch_joint','right_ankle_pitch_joint'):
                node=next(j for j in xml.findall('joint') if j.get('name')==joint)
                parent=raw.robot.body_names.index(node.find('parent').get('link'))
                child=raw.robot.body_names.index(node.find('child').get('link'))
                length=sum(float(v)**2 for v in node.find('origin').get('xyz').split())**.5
                error=abs(float((raw.robot.data.body_pos_w[i,parent]-raw.robot.data.body_pos_w[i,child]).norm())-length)
                if error>2e-4:raise ValueError('Actual leg geometry mismatch')
                geometry.append(dict(design=design['id'],joint=joint,error_m=error))
        write_new(output/'geometry_audit.json',dict(verified=True,design_routing_verified=True,
            template_mass_verified=True,joints=geometry))
        obs=env.get_observations()
        state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        policy_cfg=dict(agent['policy']);policy_cfg.pop('class_name',None)
        policy_cfg['native_config']=state['model_state_dict']['_extra_state']['config']
        teacher=SpatialActorCritic(obs,{'policy':['policy'],'critic':['critic']},raw.num_actions,**policy_cfg).to(raw.device)
        teacher.load_state_dict(state['model_state_dict'],strict=True);teacher.eval();teacher.requires_grad_(False)
        student=load_student(args.student,raw.device)[0] if args.student else None
        if student and student.schema!=plan['schema']:raise ValueError('Student schema differs')
        groups=torch.arange(n,device=raw.device)%count
        copy_index=torch.arange(n,device=raw.device)//count
        if args.payload_kg is not None and (args.task!='backpack' or not 0<args.payload_kg<=15):raise ValueError('Fixed payload requires backpack and mass in (0,15]kg')
        payload=[args.payload_kg]*4 if args.payload_kg is not None else [.3,5.,10.,15.]
        # Constructing an extra student consumes RNG. Reseed AFTER both actors
        # exist, so paired teacher/student evaluation starts from the same states.
        import random
        import numpy as np
        random.seed(args.seed+999);np.random.seed(args.seed+999)
        torch.manual_seed(args.seed+999);torch.cuda.manual_seed_all(args.seed+999)
        with torch.inference_mode():
            if args.task=='backpack':
                from robolab.spatial_expert_tasks import payload_module
                module=payload_module();properties=[]
                for i in range(n):
                    properties+=module.sample_backpack_properties(1,seed=args.seed+i,
                        mass_range_kg=(payload[int(copy_index[i])//2],)*2,
                        inertia_size_scale_range=(1.,1.),mount_surface_offset_max_m=(0.,0.),com_offset_max_m=(0.,0.,0.))
                raw.reset();audit=module.apply_backpack_properties(raw.robot,properties);raw.reset()
                write_new(output/'payload_audit.json',dict(properties=properties,audit=audit))
                validation=(copy_index%2)==1
            else:
                validation=copy_index==copies-1
            obs=env.get_observations()
            if args.mode=='evaluate':
                with (output/'initial_state.pt').open('xb') as stream:
                    torch.save(dict(root_pos_w=raw.robot.data.root_pos_w.cpu().clone(),
                        root_quat_w=raw.robot.data.root_quat_w.cpu().clone(),
                        joint_pos=raw.robot.data.joint_pos.cpu().clone(),
                        joint_vel=raw.robot.data.joint_vel.cpu().clone()),stream)
            static=[]
            for design in designs:
                graph=graph_features(design['urdf'],names,bank['canonical_semantic_contract']['default_joint_pos'])
                static.append([v for row in graph for v in row]+unified_context(bank,design,plan['schema']))
            static=torch.tensor(static,device=raw.device,dtype=torch.float32)
            io=dict(joint_names=names,default_joint_pos=expected.cpu().tolist(),action_scale=float(raw.action_scale),
                step_dt=raw.step_dt,observation_scales={k:float(v) for k,v in vars(raw.obs_scales).items() if isinstance(v,(int,float))},
                history_layout='frame_major_oldest_to_newest',history_width=780)
            write_new(output/'io_contract.json',io)
            data={k:[] for k in ('base_obs','teacher_actions','design_index','environment_ids','validation')}
            alive=torch.ones(n,device=raw.device,dtype=torch.bool)
            elapsed=torch.zeros(n,device=raw.device);forward=elapsed.clone();side=elapsed.clone();ascent=elapsed.clone()
            fell=torch.zeros(n,device=raw.device,dtype=torch.bool)
            action_clip=float(raw.clip_actions)
            if action_clip!=100. or float(raw.clip_obs)!=100.:raise ValueError('Unexpected native clipping contract')
            clipped=torch.zeros(n,device=raw.device)
            if args.ppo_channel:
                from ppo_channel import serve
                serve(args.ppo_channel,args.task,env,raw,obs,
                    lambda o,r:canonical_input(o['policy'][:,:780],plan['task_ids'][args.task],groups,static,len(plan['schema']['tasks'])),action_clip)
                exit_code=0
                return
            trace=[]
            recorder=None
            if args.video:
                from shared_video import Recorder
                recorder=Recorder(raw,output/'evaluation.mp4')
            for step in range(args.steps):
                native=obs['policy']
                if native.shape[1]!=780+spec['context_width']:raise ValueError('Native teacher observation mismatch')
                base=native[:,:780]
                label=teacher.act_inference(obs).detach()
                if not torch.isfinite(label).all():raise ValueError('Nonfinite teacher action')
                # The native PD and actor-observation paths already clip at 100.
                # Canonicalize executed targets to that SAME limit for DAgger;
                # do not discard precisely the off-policy states needing labels.
                label=label.clamp(-action_clip,action_clip)
                if args.mode=='collect' and step%args.stride==0:
                    for k,v in dict(base_obs=base,teacher_actions=label,design_index=groups,
                                    environment_ids=torch.arange(n,device=raw.device),validation=validation).items():
                        data[k].append(v.detach().cpu().clone())
                action=student(canonical_input(base,plan['task_ids'][args.task],groups,static,len(plan['schema']['tasks']))) if student else label
                if not torch.isfinite(action).all():raise ValueError('Nonfinite executing actor action')
                if args.trace_stride>0 and step%args.trace_stride==0:
                    trace.append(dict(step=step,root_pos_w=raw.robot.data.root_pos_w.cpu().clone(),
                        root_quat_w=raw.robot.data.root_quat_w.cpu().clone(),
                        root_lin_vel_b=raw.robot.data.root_lin_vel_b.cpu().clone(),
                        root_ang_vel_b=raw.robot.data.root_ang_vel_b.cpu().clone(),
                        current_frame=base[:,-78:].cpu().clone(),action=action.cpu().clone(),
                        teacher_action=label.cpu().clone(),alive=alive.cpu().clone()))
                clipped+=(action.abs()>action_clip).any(-1).float()
                if student and (action.abs()>action_clip).any() and not (output/'first_clipped_state.pt').exists():
                    with (output/'first_clipped_state.pt').open('xb') as stream:
                        torch.save(dict(input=canonical_input(base,plan['task_ids'][args.task],groups,static,len(plan['schema']['tasks'])).cpu(),
                            raw_student_actions=action.cpu(),teacher_actions=label.cpu(),step=step),stream)
                action=action.clamp(-action_clip,action_clip)
                before=raw.robot.data.root_pos_w.clone()
                obs,rewards,dones,extras=env.step(action)
                after=raw.robot.data.root_pos_w.clone()
                if recorder:recorder.capture(step,dones)
                # Terminal resets may move roots; exclude that step's displacement.
                active=alive&~dones.bool()
                delta=after-before
                forward[active]+=delta[active,0];side[active]+=delta[active,1];ascent[active]+=delta[active,2]
                elapsed[alive]+=raw.step_dt
                fell |= alive&raw.reset_terminated.bool()
                alive &= ~dones.bool()
                if step%100==0:print('ROLLOUT',args.task,step,'/',args.steps,'designs',count,'student',bool(student),flush=True)
            rows=[]
            if recorder:recorder.close()
            for i in range(n):
                rows.append(dict(design=designs[int(groups[i])]['id'],environment=i,
                    payload_kg=payload[int(copy_index[i])//2] if args.task=='backpack' else None,
                    seconds=float(elapsed[i]),fell=bool(fell[i]),forward_m=float(forward[i]),
                    lateral_m=float(side[i]),ascent_m=float(ascent[i]),
                    native_action_clipping_fraction=float(clipped[i]/args.steps),
                    mean_forward_speed=float(forward[i]/elapsed[i].clamp(min=.02))))
            if args.mode=='collect':
                value={k:torch.cat(v) for k,v in data.items()}
                value.update(kind='real_rough_backpack_teacher_data_v1',task=args.task,
                    plan_sha256=sha(args.plan),teacher_sha256=spec['checkpoint_sha256'],bank_sha256=spec['bank_sha256'],
                    student_sha256=sha(args.student) if args.student else None,
                    schema=plan['schema'],io_contract=io,design_ids=selected,static_context=static.cpu(),
                    source='real_teacher_execution' if not student else 'real_student_DAgger_with_frozen_native_teacher_labels',
                    seed=args.seed,reset_yaw_range_rad=args.reset_yaw_range,
                    physical_profile='startup/interval DR and noise off; fixed command; explicitly recorded independent reset offsets',
                    actual_payload_not_in_actor=True)
                with (output/'data.pt').open('xb') as stream:torch.save(value,stream)
            if trace:
                with (output/'diagnostic_trace.pt').open('xb') as stream:
                    torch.save(dict(design_ids=selected,groups=groups.cpu(),trace=trace,
                        student_sha256=sha(args.student) if args.student else None,
                        teacher_sha256=spec['checkpoint_sha256'],step_dt=raw.step_dt),stream)
            write_new(output/'summary.json',dict(result='REAL_ROLLOUT_COMPLETE',task=args.task,design_ids=selected,
                teacher=spec['checkpoint_sha256'],student=sha(args.student) if args.student else None,
                teacher_updates=0,steps=args.steps,reset_yaw_range_rad=args.reset_yaw_range,
                rows=rows,data=str(output/'data.pt') if args.mode=='collect' else None))
            exit_code=0
    except BaseException:
        write_new(output/'failure.json',dict(traceback=traceback.format_exc()))
        traceback.print_exc();exit_code=1
    finally:
        timer=threading.Timer(120,lambda:os._exit(1));timer.daemon=True;timer.start()
        try:
            if raw is not None:raw.close()
            if exit_code:
                # Kit close can terminate the interpreter with status 0. On a
                # recorded failure exit explicitly before that shutdown path.
                sys.stdout.flush();sys.stderr.flush();os._exit(1)
            app.close()
        except SystemExit as error:
            if error.code not in (None,0):exit_code=1
        except BaseException:
            traceback.print_exc();exit_code=1
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        timer.cancel()
    sys.stdout.flush();sys.stderr.flush()
    return exit_code


if __name__=='__main__':os._exit(main())
