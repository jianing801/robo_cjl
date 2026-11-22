"""Frozen native AMP teacher sampling or shared-student evaluation, no PPO."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
from contract import load_plan,sha,write_new,graph_features,unified_context
from evaluation import EVALUATION_VELOCITY_FRAME, FIXED_COURSE_VELOCITY_FRAME
from randomization import configure_profile, rollout_metadata, assert_protocol_unchanged
from scenario_sampling import last_design_copy, rollout_env_count, verify_collection_layout

def prepare_rollout(argv=None, *, mixed_count=None):
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','design-ids','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--student');p.add_argument('--mode',choices=['collect','evaluate'],default='collect')
    p.add_argument('--num-envs',type=int,help='Exact physical environment count for collection')
    p.add_argument('--profile',choices=['nominal','robust'],default='nominal')
    p.add_argument('--randomization-strength',type=float,default=1.)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--stride',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261030);p.add_argument('--speed',type=float,default=2.)
    p.add_argument('--reset-yaw-range',type=float,default=.05)
    p.add_argument('--ppo-channel')
    p.add_argument('--full-body-torque',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--bo-conditions', help='Collect independent fixed-condition BO measurements')
    AppLauncher.add_app_launcher_args(p);a=p.parse_args(argv)
    if a.video:a.enable_cameras=True
    if min(a.steps,a.stride)<1 or not 0<=a.reset_yaw_range<=.5:raise ValueError('Positive bounded rollout/reset distribution required')
    plan=load_plan(a.plan);spec=plan['teachers']['run'];code=Path(spec['code_root'])
    ids=a.design_ids.split(',')
    if len(ids)!=len(set(ids)) or not set(ids)<=set(spec['design_ids']):raise ValueError('Unknown or duplicate AMP design')
    from bo_capture import CaptureReset, TrialCapture, configure_conditions, motor_records, parse_conditions
    bo_conditions = parse_conditions(a, ids)
    velocity_frame=None
    if a.mode=='evaluate' and not a.ppo_channel:
        velocity_frame=FIXED_COURSE_VELOCITY_FRAME if bo_conditions else EVALUATION_VELOCITY_FRAME
    if bo_conditions:
        a.speed = bo_conditions['tasks']['run']['speed_mps']
    n=mixed_count if mixed_count is not None else rollout_env_count(a,len(ids),bo_conditions['copies'] if bo_conditions else spec['copies'])
    sys.path[:0]=[str(Path(spec['runtime_dir'])),str(code),str(code.parent/'rsl_rl')]
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if a.video:
        from shared_video import reuse_shader_cache
        reuse_shader_cache(out)
    a.kit_args=(a.kit_args or '')+f' --vulkan --/renderer/multiGpu/enabled=false --portable-root {out/"portable"} --portable'
    app=AppLauncher(a).app if mixed_count is None else None
    raw=None;status=1
    try:
        import torch
        torch.set_num_threads(2)
        import isaaclab_tasks,robolab.tasks
        from amp_bank import load_configurable_bank
        from amp_references import validate_library
        from amp_rough_alignment import rough_module
        from amp_mixed_runtime import configure_spatial,SpatialVecEnv
        from amp_runtime import SpatialAmpEnv,configure_amp
        from amp_policy import SpatialActorCritic
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rollout import reuse_assets,canonical_input,load_rollout_bank,indexed_rollout_cache
        from amp_adapter import decode_amp_frames
        from student import load_student
        write_new(out/'importer_audit.json',rough_module('hybrid_importer').install_local_importer_adapter())
        state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        identity=None if bo_conditions or a.ppo_channel else state.get('hybrid_identity')
        bank=load_rollout_bank(spec,'run',ids,identity,load_configurable_bank)
        by_id={d['id']:d for d in bank['designs']};designs=[by_id[i] for i in ids]
        if sha(spec['reference_library'])!=spec['reference_library_sha256']:raise ValueError('Reference manifest changed')
        library=validate_library(bank,spec['bank'],spec['reference_library'],ids)
        options=SimpleNamespace(num_envs=n,device=a.device,asset_cache=None,allow_cache_misses=True,
            reference_library=spec['reference_library'],full_body_torque=spec['full_body_torque'])
        cfg,agent,source_contract=configure_spatial(options,bank,designs,out,configurator=configure_amp)
        cfg,randomization_audit=configure_profile(cfg,'run',profile=a.profile,strength=a.randomization_strength,
            seed=a.seed,reset_yaw_range=a.reset_yaw_range,speed=a.speed)
        if bo_conditions:
            configure_conditions(cfg, 'run', bo_conditions, a.seed, randomization_audit)
        if velocity_frame:
            randomization_audit['velocity_frame']=velocity_frame
        metadata=rollout_metadata(randomization_audit,dict(reset_yaw_range_rad=a.reset_yaw_range,speed_command=a.speed))
        write_new(out/'randomization_audit.json',randomization_audit)
        expected_terms=['base_ang_vel','projected_gravity','velocity_commands','joint_pos','joint_vel','actions']
        for name in expected_terms:
            term=getattr(cfg.observations.policy,name)
            if term.scale not in (None,1.):raise ValueError('AMP physical observation scale differs')
            if term.clip is not None:raise ValueError('AMP observation clipping needs explicit reconciliation')
        if cfg.observations.policy.history_length!=3:raise ValueError('AMP native history changed')
        cfg.sim.render_interval=1000000
        if a.full_body_torque and not spec['full_body_torque']:
            raise ValueError('Cannot override frozen teacher physics')
        indexed=indexed_rollout_cache(spec,'run',designs) if identity else None
        cache=reuse_assets(cfg,indexed if indexed is not None else spec['asset_cache'],out,'run',designs)
        write_new(out/'asset_cache_audit.json',cache)
        if a.video:cfg.viewer.resolution=(960,540)
        class CaptureAmp(CaptureReset, SpatialAmpEnv):
            pass
        if mixed_count is None:
            raw=CaptureAmp(cfg,render_mode='rgb_array' if a.video else None)
        else:
            raw=yield CaptureAmp,cfg
        class SamplingVecEnv(SpatialVecEnv):
            def step(self,actions):
                obs,rewards,dones,extras=RslRlVecEnvWrapper.step(self,actions)
                return self._extend(obs),rewards,dones,extras
        env=SamplingVecEnv(raw,None,agent.get('clip_actions'))
        names=bank['canonical_semantic_contract']['joint_names'];q0=bank['canonical_semantic_contract']['default_joint_pos']
        if list(raw.robot.joint_names)!=names or raw.step_dt!=.02:raise ValueError('AMP joint/control contract differs')
        if list(raw.observation_manager.active_terms['policy'])!=expected_terms:raise ValueError('AMP term order differs')
        action_term=raw.action_manager.get_term('joint_pos')
        if not torch.as_tensor(action_term._scale).eq(.25).all():raise ValueError('AMP action scale differs')
        if not torch.allclose(raw.robot.data.default_joint_pos,torch.tensor(q0,device=raw.device).expand(n,-1),atol=1e-6):
            raise ValueError('AMP default pose differs')
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        routed=[stage.GetPrimAtPath(f'{path}/Robot').GetAttribute('spatial:design_id').Get()
                for path in raw.scene.env_prim_paths]
        if routed!=[ids[i%len(ids)] for i in range(n)]:raise ValueError('AMP physical/context routing differs')
        if a.num_envs is not None or mixed_count is not None:
            verify_collection_layout(raw,designs,n)
        motion=raw.motion_data_manager.get_term('motion_dataset')
        actual_libraries=[motion.terms[i].cfg.motion_data_dir for i in motion.env_library.cpu().tolist()]
        if actual_libraries!=[library['designs'][i]['directory'] for i in routed]:raise ValueError('AMP reference routing differs')
        from amp_fast_reference import install_verified
        routing=install_verified(motion,raw.animation_manager.get_term('animation'))
        write_new(out/'reference_cache_parity.json',dict(native_outputs_exact=True,
            packed_native_lookup_verified=routing.merged_verified,packed_fallback_reason=routing.merge_fallback_reason,
            changed='Native reference values/interpolation retained; deterministic table lookup or membership cache; sampling order/RNG unchanged'))
        from physical_audit import audit_physics
        write_new(out/'physics_audit.json',audit_physics(raw,cfg,designs,profile=a.profile))
        import xml.etree.ElementTree as ET
        geometry=[]
        for i,d in enumerate(designs):
            tree=ET.parse(d['urdf']).getroot()
            for name in ('left_knee_joint','right_knee_joint','left_ankle_pitch_joint','right_ankle_pitch_joint'):
                joint=tree.find(f"joint[@name='{name}']")
                parent=raw.robot.body_names.index(joint.find('parent').get('link'))
                child=raw.robot.body_names.index(joint.find('child').get('link'))
                length=sum(float(v)**2 for v in joint.find('origin').get('xyz').split())**.5
                error=abs(float((raw.robot.data.body_pos_w[i,parent]-raw.robot.data.body_pos_w[i,child]).norm())-length)
                if error>2e-4:raise ValueError('AMP real geometry differs from URDF')
                geometry.append(dict(design=d['id'],joint=name,error_m=error))
        write_new(out/'geometry_audit.json',dict(verified=True,joints=geometry))
        write_new(out/'reference_routing_audit.json',dict(verified=True,design_ids=routed,per_env_libraries=actual_libraries))
        obs=env.get_observations()
        config=dict(agent['policy']);config.pop('class_name',None)
        config['native_config']=state['model_state_dict']['_extra_state']['config']
        teacher=SpatialActorCritic(obs,{'policy':['policy'],'critic':['critic']},env.num_actions,**config).to(raw.device)
        teacher.load_state_dict(state['model_state_dict'],strict=True);teacher.eval();teacher.requires_grad_(False)
        student=load_student(a.student,raw.device)[0] if a.student else None
        if student and student.schema!=plan['schema']:raise ValueError('Student schema differs')
        normalization_before={k:v.clone() for k,v in teacher.actor_obs_normalizer.state_dict().items()}
        import random,numpy as np
        random.seed(a.seed+999);np.random.seed((a.seed+999) % 2**32);torch.manual_seed(a.seed+999);torch.cuda.manual_seed_all(a.seed+999)
        with torch.inference_mode():
            from student_observations import StudentHistory, boundary_contract, STUDENT_ACTION_CLIP, EpisodeTracker, TrackingError, speed_bins
            from student_observations import forward_velocity
            from scenario_sampling import scenario_table
            obs=env.get_observations();environments=torch.arange(n,device=raw.device)
            groups=environments%len(ids);validation=last_design_copy(environments,len(ids))
            bo = TrialCapture(raw, 'run', bo_conditions, a.seed, motor_records(raw, bank, designs[0])) if bo_conditions else None
            static=torch.tensor([[v for row in graph_features(d['urdf'],names,q0) for v in row]+
                unified_context(bank,d,plan['schema']) for d in designs],device=raw.device)
            io=dict(joint_names=names,default_joint_pos=q0,action_scale=.25,step_dt=.02,
                observation_scales={k:1. for k in ('lin_vel','ang_vel','projected_gravity','commands','joint_pos','joint_vel','actions','height_scan')},
                history_layout='frame_major_oldest_to_newest',history_width=780,**boundary_contract())
            write_new(out/'io_contract.json',io)
            if a.mode=='evaluate':
                with (out/'initial_state.pt').open('xb') as stream:torch.save(dict(root_pos_w=raw.robot.data.root_pos_w.cpu().clone(),
                    root_quat_w=raw.robot.data.root_quat_w.cpu().clone(),joint_pos=raw.robot.data.joint_pos.cpu().clone(),
                    joint_vel=raw.robot.data.joint_vel.cpu().clone()),stream)
            history=StudentHistory(n,raw.device,'run',profile=a.profile,
                strength=a.randomization_strength,seed=a.seed+700001);reset=None
            episodes=EpisodeTracker(n,raw.device);tracking=TrackingError(n,raw.device)
            scenarios=scenario_table('run',n,ids,profile=a.profile)
            if bo_conditions:
                for scenario in scenarios:
                    scenario['fixed_bo_conditions'] = bo_conditions['tasks']['run']
            write_new(out/'scenarios.json',scenarios)
            data={k:[] for k in ('base_obs','teacher_actions','design_index','environment_ids','validation',
                'scenario_bin','episode_id','actual_command')}
            alive=torch.ones(n,device=raw.device,dtype=torch.bool);elapsed=torch.zeros(n,device=raw.device)
            forward=elapsed.clone();side=elapsed.clone();fell=~alive.clone();native_clip=agent.get('clip_actions')
            if native_clip not in (None,STUDENT_ACTION_CLIP):
                raise ValueError('Native action boundary is incompatible with the shared action contract')
            native_boundaries=dict(action_clip=native_clip,observation_clip=None)
            if a.ppo_channel:
                from ppo_channel import serve
                from rsl_rl.modules.amp import AMPDiscriminator,LossType
                amp=agent['algorithm']['amp_cfg'];disc_cfg=amp['amp_discriminator']
                discriminator=AMPDiscriminator(67,3,agent['obs_groups'],
                    loss_type=LossType[amp['loss_type']],device=str(raw.device),**disc_cfg).to(raw.device)
                discriminator.load_state_dict(state['amp_discriminator_state_dict'],strict=True)
                discriminator.disc_obs_normalizer.load_state_dict(state['amp_discriminator_normalizer_state_dict'],strict=True)
                discriminator.eval();discriminator.requires_grad_(False)
                def canonical(o,r):
                    latest=decode_amp_frames(o['policy'][:,:234])[:,-1]
                    return canonical_input(history.append(latest,r),plan['task_ids']['run'],groups,static,len(plan['schema']['tasks']))
                def amp_reward(o,reward):
                    style,_=discriminator.predict_style_reward(discriminator.get_disc_obs(o),dt=raw.step_dt)
                    return discriminator.lerp_reward(task_reward=reward,style_reward=style)
                if mixed_count is not None:
                    return env,raw,obs,canonical,STUDENT_ACTION_CLIP,amp_reward,metadata['collection_protocol']
                serve(a.ppo_channel,'run',env,raw,obs,canonical,STUDENT_ACTION_CLIP,amp_reward,
                    protocol=metadata['collection_protocol'])
                status=0
                return status
            recorder=None
            if a.video:
                from shared_video import Recorder
                recorder=Recorder(raw,out/'evaluation.mp4')
            for step in range(a.steps):
                native=obs['policy']
                if native.shape!=(n,234+spec['context_width']):raise ValueError('AMP native actor width differs')
                latest=decode_amp_frames(native[:,:234])[:,-1]
                expected=torch.cat((raw.robot.data.root_ang_vel_b,raw.robot.data.projected_gravity_b,
                    raw.command_manager.get_command('base_velocity'),raw.robot.data.joint_pos-raw.robot.data.default_joint_pos,
                    raw.robot.data.joint_vel-raw.robot.data.default_joint_vel,raw.action_manager.action),-1)
                if not torch.allclose(latest,expected,atol=2e-5,rtol=1e-5):raise ValueError('AMP history decoder differs from actual physical observations')
                command=raw.command_manager.get_command('base_velocity').clone()
                base=history.append(latest,reset);label=teacher.act_inference(obs).detach()
                if not torch.isfinite(label).all():raise ValueError('Nonfinite AMP teacher labels')
                label=label.clamp(-STUDENT_ACTION_CLIP,STUDENT_ACTION_CLIP)
                if a.mode=='collect' and step%a.stride==0:
                    for k,v in dict(base_obs=base,teacher_actions=label,design_index=groups,
                        environment_ids=environments,validation=validation,
                        scenario_bin=speed_bins(command,'run'),episode_id=episodes.episode,
                        actual_command=command).items():data[k].append(v.cpu().clone())
                action=student(canonical_input(base,plan['task_ids']['run'],groups,static,len(plan['schema']['tasks']))) if student else label
                if not torch.isfinite(action).all():raise ValueError('Nonfinite AMP executed action')
                action=action.clamp(-STUDENT_ACTION_CLIP,STUDENT_ACTION_CLIP)
                before=raw.robot.data.root_pos_w.clone();obs,reward,done,extra=env.step(action)
                if bo:
                    bo.capture(command, done)
                if recorder:recorder.capture(step,done)
                delta=raw.robot.data.root_pos_w-before;active=alive&~done.bool()
                tracking.update(command[:,0],forward_velocity(raw.robot.data,velocity_frame),active,
                    dt=raw.step_dt,command_valid=alive)
                forward[active]+=delta[active,0];side[active]+=delta[active,1];elapsed[alive]+=raw.step_dt
                fell|=alive&raw.reset_terminated.bool();alive&=~done.bool();reset=done.bool()
                episodes.advance(done.bool())
                if step%100==0:print('AMP_ROLLOUT',step,'/',a.steps,'designs',len(ids),'student',bool(student),flush=True)
                if bo and not bo.active.any():
                    break
            if recorder:recorder.close()
            if any(not torch.equal(normalization_before[k],v) for k,v in teacher.actor_obs_normalizer.state_dict().items()):
                raise ValueError('Frozen AMP actor normalizer changed')
            tracking_rows=tracking.rows(horizontal=velocity_frame is not None)
            rows=[dict(design=ids[i%len(ids)],environment=i,payload_kg=None,seconds=float(elapsed[i]),fell=bool(fell[i]),
                forward_m=float(forward[i]),lateral_m=float(side[i]),mean_forward_speed=float(forward[i]/elapsed[i].clamp(min=.02)),
                **tracking_rows[i]) for i in range(n)]
            assert_protocol_unchanged(metadata['collection_protocol'])
            if a.mode=='collect':
                value={k:torch.cat(v) for k,v in data.items()}
                value.update(kind='real_rough_backpack_teacher_data_v1',task='run',plan_sha256=sha(a.plan),
                    teacher_sha256=spec['checkpoint_sha256'],bank_sha256=spec['bank_sha256'],schema=plan['schema'],io_contract=io,
                    design_ids=ids,static_context=static.cpu(),actual_payload_not_in_actor=True,
                    student_sha256=sha(a.student) if a.student else None,
                    source='real_teacher_execution' if not student else 'real_student_DAgger_with_frozen_native_teacher_labels',
                    native_history='three term-major frames; original trained normalizer restored and frozen',
                    seed=a.seed,**metadata,
                    canonical_history='ten real streaming frames; reset-aware first-frame fill only',
                    reference_library_sha256=spec['reference_library_sha256'],history_adapter_live_verified=True,
                    native_boundaries=native_boundaries,scenarios=scenarios,run_arguments=vars(a))
                with (out/'data.pt').open('xb') as stream:torch.save(value,stream)
            if bo:
                bo.publish(out, dict(design_id=ids[0], bank_sha256=spec['bank_sha256'],
                    actor_sha256=sha(a.student) if a.student else spec['checkpoint_sha256'],
                    plan_sha256=sha(a.plan)))
            write_new(out/'summary.json',dict(result='REAL_AMP_ROLLOUT_COMPLETE',task='run',rows=rows,design_ids=ids,
                teacher=spec['checkpoint_sha256'],student=sha(a.student) if a.student else None,
                steps=a.steps,seed=a.seed,**metadata,
                **({'velocity_frame': velocity_frame} if velocity_frame else {}),
                actor_normalizer_unchanged=True,native_clip_actions=native_clip,
                teacher_updates=0,reference_routing_verified=True,history_adapter_live_verified=True))
            write_new(out/'reference_cache_stats.json',dict(hits=routing.hits,misses=routing.misses))
            status=0
    except BaseException:
        write_new(out/'failure.json',dict(traceback=traceback.format_exc()))
        if mixed_count is not None:
            raise
        traceback.print_exc()
    finally:
        if app is not None:
            timer=threading.Timer(120,lambda:os._exit(1));timer.daemon=True;timer.start()
            if status:sys.stdout.flush();sys.stderr.flush();os._exit(1)
            try:
                if raw is not None:raw.close()
                app.close()
            except SystemExit as error:
                if error.code not in (None,0):status=1
            except BaseException:traceback.print_exc();status=1
            timer.cancel()
    sys.stdout.flush();sys.stderr.flush();return status

def main():
    try:
        next(prepare_rollout())
    except StopIteration as result:
        return result.value
    raise RuntimeError('Unexpected deferred AMP rollout')


if __name__=='__main__':os._exit(main())
