"""Real native teacher/DAgger sampling or closed-loop evaluation, one scene per process."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import traceback

HERE=Path(__file__).resolve().parent
from contract import load_plan,sha,write_new,graph_features,unified_context,input_width
from randomization import configure_profile, rollout_metadata, assert_protocol_unchanged
from scenario_sampling import rollout_env_count, verify_collection_layout, verify_stair_origins


def load_rollout_bank(spec, task, selected_ids, identity, native_loader):
    """Use the frozen expert's full-bank identity to verify just the live assets."""
    path = Path(spec['bank'])
    if (not identity or not spec.get('full_body_torque') or spec.get('original_bank') != spec['bank']
            or identity.get('bank_sha256') != spec['bank_sha256']
            or identity.get('full_bank_context_is_stable') is not True or not identity.get('mesh_sources')):
        return native_loader(spec['bank'])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != spec['bank_sha256']:
        raise ValueError('Rollout bank differs from the frozen checkpoint')
    bank = json.loads(content)
    if bank['schema_version'] != 3:
        return native_loader(spec['bank'])
    from robolab.morphology_context import FIELDS, SCHEMA, verified_raw_context
    from robolab.physical_snapshot import generator, validate_physical_model
    from robolab.gp_extrapolation import design_extrapolation, validate_authorization
    validate_authorization(bank['configuration'], bank['training'])
    gp = generator(bank['snapshot_root']) if bank['configuration'].get('gp_extrapolation') else None
    if task == 'run':
        from amp_support import source_identity
        from amp_rough_alignment import rough_module, validate_bound_bank
        validate_bound_bank(bank)
        mesh_sources = rough_module('hybrid_assets').mesh_sources
    else:
        from hybrid_support import source_identity
        from hybrid_assets import mesh_sources
    if (task != bank['task_key'] or task != identity['task']
            or source_identity(task) != bank['training']['source_hashes']
            or identity['native_sources'] != bank['training']['source_hashes']):
        raise ValueError('Native source or task differs from the checkpoint-bound bank')
    for field, digest_field in (('original_template', 'original_template_sha256'),
            ('reference_contract_source', 'reference_contract_sha256')):
        if sha(bank[field]) != bank[digest_field]:
            raise ValueError('Changed bank dependency: ' + field)
    if task == 'backpack':
        binding = bank['base_morphology_binding']
        if sha(binding['bank']) != binding['sha256']:
            raise ValueError('Bound Rough bank changed')
    selected = set(selected_ids)
    active = [row for row in bank['designs'] if row['id'] in selected]
    if not active or len(active) != len(selected) or len(selected) != len(selected_ids):
        raise ValueError('Unknown or duplicate rollout design')
    norm = bank['context_normalization']
    if (bank['context_schema'] != SCHEMA or bank['context_fields'] != list(FIELDS)
            or bank['context_width'] != len(FIELDS) or norm['kind'] != 'training_bank_minmax_v1'):
        raise ValueError('Unexpected checkpoint-bound morphology context')
    for pair in {row['pair_id'] for row in active}:
        validate_physical_model(bank['motor_pairs'][pair])
    for row in active:
        if sha(row['report']) != row['report_sha256']:
            raise ValueError('Active design report changed: ' + row['id'])
        if task == 'backpack' and sha(row['base_urdf']) != row['base_urdf_sha256']:
            raise ValueError('Active backpack base URDF changed: ' + row['id'])
        physical = bank['motor_pairs'][row['pair_id']]
        if gp is not None and design_extrapolation(gp, physical, row) != row.get('gp_extrapolation'):
            raise ValueError('Active design GP extrapolation markers changed: ' + row['id'])
        values = verified_raw_context(row, physical)
        normalized = [0. if hi-lo <= 1e-12 else 2.*(value-lo)/(hi-lo)-1.
            for value, lo, hi in zip(values, norm['minimum'], norm['maximum'], strict=True)]
        if normalized != row['context']:
            raise ValueError('Active physical context differs from frozen full-bank normalization: ' + row['id'])
    for name, digest in mesh_sources({'designs':active}, report=False).items():
        if identity['mesh_sources'].get(name) != digest:
            raise ValueError('Active source mesh differs from the checkpoint: ' + name)
    return bank


def indexed_rollout_cache(spec, task, designs):
    """Resolve the standard launcher's bank-bound cache proof without scanning run history."""
    bank = Path(spec['bank']).resolve()
    task_dir = {'stairs':'rough', 'backpack':'backpack', 'run':'amp'}[task]
    if bank.parent.name != task_dir or bank.parent.parent.name != 'banks':
        return None
    index = bank.parents[2]/'启动记录'/f'{task_dir}_asset_cache.json'
    if not index.is_file():
        return None
    record = json.loads(index.read_text(encoding='utf8'))
    cache = Path(record['cache']).resolve()
    content = (cache/'completed.json').read_bytes()
    if (record['bank_sha256'] != spec['bank_sha256']
            or record['completed_sha256'] != hashlib.sha256(content).hexdigest()):
        raise ValueError('Native USD cache index or completion proof changed')
    proof = json.loads(content)
    if (proof['result'] != 'NATIVE_ASSET_CACHE_COMPLETE' or proof.get('cache_schema_version') != 2
            or proof['bank_sha256'] != spec['bank_sha256']):
        raise ValueError('Native USD cache proof belongs to another bank or schema')
    rows = {row['design']:row for row in proof['assets']}
    if len(rows) != len(proof['assets']):
        raise ValueError('Duplicate design in native USD cache proof')
    directories = []
    for design in designs:
        row = rows[design['id']]
        usd = Path(row['usd']).resolve()
        if (not usd.is_relative_to(cache) or usd.parent.name != design['id'] or not usd.is_file()
                or row['urdf_sha256'] != design['urdf_sha256'] or not (usd.parent/'config.yaml').is_file()
                or (usd.parent/'.asset_hash').read_text().strip() != row['converter_hash']):
            raise ValueError('Invalid indexed native USD for ' + design['id'])
        directories.append(usd.parent)
    return directories


def reuse_assets(cfg,roots,output,task,designs):
    """Reuse ONLY exact native converter hashes, keeping old assets unchanged."""
    from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
    available={}
    for root in roots:
        if not Path(root).exists():continue
        direct=Path(root)/'.asset_hash'
        for entry in ((direct,) if direct.is_file() else Path(root).rglob('.asset_hash')):
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
    p.add_argument('--num-envs',type=int,help='Exact physical environment count for collection')
    p.add_argument('--profile',choices=['nominal','robust'],default='nominal')
    p.add_argument('--randomization-strength',type=float,default=1.)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--ppo-channel')
    p.add_argument('--full-body-torque',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--bo-conditions', help='Collect independent fixed-condition BO measurements')
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
    selected=args.design_ids.split(',')
    if len(set(selected))!=len(selected) or not set(selected)<=set(spec['design_ids']):raise ValueError('Unknown/duplicate selected design')
    from bo_capture import CaptureReset, TrialCapture, configure_conditions, motor_records, parse_conditions
    bo_conditions=parse_conditions(args, selected)
    count=len(selected)
    copies=bo_conditions['copies'] if bo_conditions else spec['copies']
    if bo_conditions and args.task=='backpack':
        args.payload_kg=bo_conditions['tasks']['backpack']['payload_kg']
    paired_stairs=args.task=='stairs' and args.profile=='robust' and args.mode=='collect' and not args.ppo_channel
    if args.task=='stairs' and args.profile=='robust':copies=max(copies,12 if paired_stairs else 6)
    n=rollout_env_count(args,count,copies,training_strata=6 if paired_stairs else 1)
    code=Path(spec['code_root'])
    sys.path[:0]=[str(HERE),str(code/'scripts/tools'),str(Path(spec['runtime_dir'])),
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
        from student_observations import StudentHistory, boundary_contract, STUDENT_ACTION_CLIP, EpisodeTracker, TrackingError, speed_bins
        from scenario_sampling import configure_stairs, route_stairs, payload_properties, payload_bins, scenario_table
        from physical_audit import audit_physics
        # Process-local importer compatibility; installed dependencies are not changed.
        importer_path=HERE.parent/'engine/hybrid_importer.py'
        s=importlib.util.spec_from_file_location('shared_student_importer',importer_path)
        adapter=importlib.util.module_from_spec(s);s.loader.exec_module(adapter)
        write_new(output/'importer_audit.json',adapter.install_local_importer_adapter())
        if args.task=='stairs':
            from hybrid_bank import load_configurable_bank
            from hybrid_runtime import configure_spatial,SpatialRoughEnv as Env,SpatialVecEnv
            from hybrid_policy import SpatialActorCritic
            configurator=None
        else:
            from hybrid_bank import load_configurable_bank
            from hybrid_runtime import configure_spatial,SpatialVecEnv
            from hybrid_backpack import configure_backpack,SpatialBackpackEnv as Env
            from hybrid_policy import SpatialActorCritic
            configurator=configure_backpack
        state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        identity=None if bo_conditions or args.ppo_channel else state.get('hybrid_identity')
        bank=load_rollout_bank(spec,args.task,selected,identity,load_configurable_bank)
        by_id={d['id']:d for d in bank['designs']}
        designs=[by_id[k] for k in selected]
        for design in designs:
            if sha(design['urdf'])!=design['urdf_sha256']:raise ValueError('Physical URDF changed')
        options=SimpleNamespace(num_envs=n,device=args.device,asset_cache=None,allow_cache_misses=True,
                                torque_soft_constraint=None,full_body_torque=spec['full_body_torque'])
        kwargs={'configurator':configurator} if configurator else {}
        cfg,agent,source_contract=configure_spatial(options,bank,designs,output,**kwargs)
        cfg,randomization_audit=configure_profile(cfg,args.task,profile=args.profile,
            strength=args.randomization_strength,seed=args.seed,reset_yaw_range=args.reset_yaw_range)
        cfg.capture_terminal_state=True
        if args.task=='stairs' and not bo_conditions:
            randomization_audit['stairs']=configure_stairs(cfg,profile=args.profile,seed=args.seed)
        if bo_conditions:
            configure_conditions(cfg, args.task, bo_conditions, args.seed, randomization_audit)
        metadata=rollout_metadata(randomization_audit,dict(reset_yaw_range_rad=args.reset_yaw_range))
        write_new(output/'randomization_audit.json',randomization_audit)
        cfg.sim.render_interval=1000000
        cache_roots=[*spec['asset_cache']]
        if args.asset_cache_root:cache_roots.insert(0,args.asset_cache_root)
        if identity:
            indexed=indexed_rollout_cache(spec,args.task,designs)
            if indexed is not None:cache_roots=indexed
        if args.full_body_torque and not spec['full_body_torque']:
            raise ValueError('Cannot override frozen teacher physics')
        cache=reuse_assets(cfg,cache_roots,output,args.task,designs)
        write_new(output/'asset_cache_audit.json',cache)
        class Capture(CaptureReset, Env):
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
        if agent.get('clip_actions') not in (None,STUDENT_ACTION_CLIP):
            raise ValueError('Native wrapper action boundary differs from the shared action contract')
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
        if args.num_envs is not None:
            verify_collection_layout(raw,designs,n)
        geometry=[]
        write_new(output/'physics_audit.json',audit_physics(raw,cfg,designs,profile=args.profile))
        for i,design in enumerate(designs):
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
        policy_cfg=dict(agent['policy']);policy_cfg.pop('class_name',None)
        policy_cfg['native_config']=state['model_state_dict']['_extra_state']['config']
        teacher=SpatialActorCritic(obs,{'policy':['policy'],'critic':['critic']},raw.num_actions,**policy_cfg).to(raw.device)
        teacher.load_state_dict(state['model_state_dict'],strict=True);teacher.eval();teacher.requires_grad_(False)
        student=load_student(args.student,raw.device)[0] if args.student else None
        if student and student.schema!=plan['schema']:raise ValueError('Student schema differs')
        groups=torch.arange(n,device=raw.device)%count
        copy_index=torch.arange(n,device=raw.device)//count
        if args.payload_kg is not None and (args.task!='backpack' or not 0<args.payload_kg<=15):raise ValueError('Fixed payload requires backpack and mass in (0,15]kg')
        properties=None
        # Constructing an extra student consumes RNG. Reseed AFTER both actors
        # exist, so paired teacher/student evaluation starts from the same states.
        import random
        import numpy as np
        random.seed(args.seed+999);np.random.seed((args.seed+999) % 2**32)
        torch.manual_seed(args.seed+999);torch.cuda.manual_seed_all(args.seed+999)
        with torch.inference_mode():
            collection_columns=None
            if args.task=='stairs':
                validation=route_stairs(raw.scene,count,profile=args.profile,paired_validation=paired_stairs)
                if args.num_envs is not None:
                    collection_columns=raw.scene.terrain.terrain_types.clone()
                raw.reset()
                if collection_columns is not None:
                    verify_stair_origins(raw.scene,collection_columns)
            elif args.task=='backpack':
                from robolab.tasks.direct.backpack_walk import backpack_urdf as module
                properties=payload_properties(module,n,count,seed=args.seed,profile=args.profile,
                    strength=args.randomization_strength,fixed_mass=args.payload_kg)
                raw.reset();audit=module.apply_backpack_properties(raw.robot,properties);raw.reset()
                write_new(output/'payload_audit.json',dict(properties=properties,audit=audit))
                validation=(copy_index%2)==1
            terrain_indices=raw.scene.terrain.terrain_types if args.task=='stairs' else None
            scenarios=scenario_table(args.task,n,selected,terrain_indices=terrain_indices,
                profile=args.profile,properties=properties)
            if bo_conditions:
                fixed = bo_conditions['tasks'][args.task]
                for scenario in scenarios:
                    scenario['fixed_bo_conditions'] = fixed
                    if args.task == 'stairs':
                        scenario.update(step_height_m=fixed['step_height_m'], step_width_m=fixed['step_depth_m'])
            write_new(output/'scenarios.json',scenarios)
            condition_bins=(terrain_indices.to(raw.device) if args.task=='stairs' else
                payload_bins(raw.robot.backpack_mass_kg))
            obs=env.get_observations()
            bo = TrialCapture(raw, args.task, bo_conditions, args.seed, motor_records(raw, bank, designs[0]),
                              properties) if bo_conditions else None
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
                history_layout='frame_major_oldest_to_newest',history_width=780,**boundary_contract())
            write_new(output/'io_contract.json',io)
            data={k:[] for k in ('base_obs','teacher_actions','design_index','environment_ids','validation',
                'scenario_bin','episode_id','actual_command')}
            alive=torch.ones(n,device=raw.device,dtype=torch.bool)
            elapsed=torch.zeros(n,device=raw.device);forward=elapsed.clone();side=elapsed.clone();ascent=elapsed.clone()
            fell=torch.zeros(n,device=raw.device,dtype=torch.bool)
            action_clip=float(raw.clip_actions)
            if action_clip!=100. or float(raw.clip_obs)!=100.:raise ValueError('Unexpected native clipping contract')
            native_boundaries=dict(action_clip=action_clip,observation_clip=float(raw.clip_obs))
            action_clip=STUDENT_ACTION_CLIP
            history=StudentHistory(n,raw.device,args.task,profile=args.profile,
                strength=args.randomization_strength,seed=args.seed+700001)
            reset=None;episodes=EpisodeTracker(n,raw.device);tracking=TrackingError(n,raw.device)
            def student_base(o,r):
                # Match AMP ordering: fresh clean physical frame, student noise,
                # then the shared numerical clip. Native teacher history stays intact.
                clean,_=raw.compute_current_observations()
                return history.append(clean,r)
            clipped=torch.zeros(n,device=raw.device)
            if args.ppo_channel:
                from ppo_channel import serve
                serve(args.ppo_channel,args.task,env,raw,obs,
                    lambda o,r:canonical_input(student_base(o,r),plan['task_ids'][args.task],groups,static,len(plan['schema']['tasks'])),action_clip,
                    protocol=metadata['collection_protocol'])
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
                command=raw.command_generator.command.clone()
                base=student_base(obs,reset)
                label=teacher.act_inference(obs).detach()
                if not torch.isfinite(label).all():raise ValueError('Nonfinite teacher action')
                # The native PD and actor-observation paths already clip at 100.
                # Canonicalize executed targets to that SAME limit for DAgger;
                # do not discard precisely the off-policy states needing labels.
                label=label.clamp(-action_clip,action_clip)
                if args.mode=='collect' and step%args.stride==0:
                    if collection_columns is not None:
                        verify_stair_origins(raw.scene,collection_columns)
                    for k,v in dict(base_obs=base,teacher_actions=label,design_index=groups,
                                    environment_ids=torch.arange(n,device=raw.device),validation=validation,
                                    scenario_bin=condition_bins*3+speed_bins(command,args.task),
                                    episode_id=episodes.episode,actual_command=command).items():
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
                if bo:
                    bo.capture(command, dones)
                after=raw.robot.data.root_pos_w.clone()
                if recorder:recorder.capture(step,dones)
                # Terminal resets may move roots; exclude that step's displacement.
                active=alive&~dones.bool()
                tracking.update(command[:,0],raw.robot.data.root_lin_vel_b[:,0],active,
                    dt=raw.step_dt,command_valid=alive)
                reset=dones.bool();episodes.advance(reset)
                delta=after-before
                forward[active]+=delta[active,0];side[active]+=delta[active,1];ascent[active]+=delta[active,2]
                elapsed[alive]+=raw.step_dt
                fell |= alive&raw.reset_terminated.bool()
                alive &= ~dones.bool()
                if step%100==0:print('ROLLOUT',args.task,step,'/',args.steps,'designs',count,'student',bool(student),flush=True)
                if bo and not bo.active.any():
                    break
            rows=[]
            if recorder:recorder.close()
            tracking_rows=tracking.rows()
            for i in range(n):
                rows.append(dict(design=designs[int(groups[i])]['id'],environment=i,
                    payload_kg=float(raw.robot.backpack_mass_kg[i]) if args.task=='backpack' else None,
                    seconds=float(elapsed[i]),fell=bool(fell[i]),forward_m=float(forward[i]),
                    lateral_m=float(side[i]),ascent_m=float(ascent[i]),
                    native_action_clipping_fraction=float(clipped[i]/args.steps),
                    mean_forward_speed=float(forward[i]/elapsed[i].clamp(min=.02)),
                    **tracking_rows[i]))
            assert_protocol_unchanged(metadata['collection_protocol'])
            if args.mode=='collect':
                value={k:torch.cat(v) for k,v in data.items()}
                value.update(kind='real_rough_backpack_teacher_data_v1',task=args.task,
                    plan_sha256=sha(args.plan),teacher_sha256=spec['checkpoint_sha256'],bank_sha256=spec['bank_sha256'],
                    student_sha256=sha(args.student) if args.student else None,
                    schema=plan['schema'],io_contract=io,design_ids=selected,static_context=static.cpu(),
                    source='real_teacher_execution' if not student else 'real_student_DAgger_with_frozen_native_teacher_labels',
                    seed=args.seed,**metadata,
                    native_boundaries=native_boundaries,scenarios=scenarios,
                    physical_profile=args.profile,run_arguments=vars(args),
                    actual_payload_not_in_actor=True)
                with (output/'data.pt').open('xb') as stream:torch.save(value,stream)
            if trace:
                with (output/'diagnostic_trace.pt').open('xb') as stream:
                    torch.save(dict(design_ids=selected,groups=groups.cpu(),trace=trace,
                        student_sha256=sha(args.student) if args.student else None,
                        teacher_sha256=spec['checkpoint_sha256'],step_dt=raw.step_dt),stream)
            if bo:
                bo.publish(output, dict(design_id=selected[0], bank_sha256=spec['bank_sha256'],
                    actor_sha256=sha(args.student) if args.student else spec['checkpoint_sha256'],
                    plan_sha256=sha(args.plan)))
            write_new(output/'summary.json',dict(result='REAL_ROLLOUT_COMPLETE',task=args.task,design_ids=selected,
                teacher=spec['checkpoint_sha256'],student=sha(args.student) if args.student else None,
                teacher_updates=0,steps=args.steps,seed=args.seed,
                rows=rows,**metadata,
                data=str(output/'data.pt') if args.mode=='collect' else None))
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
