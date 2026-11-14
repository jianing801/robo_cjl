"""Full-body candidate motor curves and native, dimensionless torque penalty."""
import torch

WEIGHT = -12.0
LIMIT_RATIO = 0.8
JOINT_COUNT = 23
GROUP_MOTORS = {'legs': 'DM-J10010L-2EC', 'shoulders': 'DM-J4340P-2EC',
                'arms': 'DM-J4340P-2EC', 'knees': 'design_knee', 'feet': 'design_ankle'}

def source_identity():
    """Bind the actual catalog and curve files, including cross-experiment use."""
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    paths=[Path(__file__).resolve(),root/'robolab/assets/rpo_motor_catalog.py']
    paths+=sorted((root/'robolab/assets/motor_data/torque_speed_curves').glob('*.csv'))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def install_fixed_motor_curves(robot, spatial_motor_cfg, design_count):
    """Replace fixed motor groups using the existing candidate catalog curves."""
    from copy import deepcopy
    import importlib.util,sys
    from pathlib import Path
    path=Path(__file__).resolve().parents[3]/'robolab/assets/rpo_motor_catalog.py'
    spec=importlib.util.spec_from_file_location('_full_body_motor_catalog',path)
    catalog=importlib.util.module_from_spec(spec);sys.modules[spec.name]=catalog;spec.loader.exec_module(catalog)
    get_motor_spec=catalog.get_motor_spec;load_torque_speed_envelope=catalog.load_torque_speed_envelope
    require_motor_actuator_ready=catalog.require_motor_actuator_ready
    if design_count < 1:raise ValueError('At least one design required')
    audit={}
    for group in ('legs','shoulders','arms'):
        motor=get_motor_spec(GROUP_MOTORS[group]);require_motor_actuator_ready(motor)
        curve=load_torque_speed_envelope(motor)
        if not curve:raise ValueError(f'{group}: candidate motor curve is missing')
        original=robot.actuators[group];values=deepcopy(vars(original));values.pop('class_type',None)
        values.update(effort_limit=motor.peak_torque_nm,effort_limit_sim=motor.peak_torque_nm,
            velocity_limit=motor.max_speed_rad_s,velocity_limit_sim=motor.max_speed_rad_s,
            armature=motor.armature_kgm2,torque_speed_lookup=curve)
        params=dict(curve=curve,armature=motor.armature_kgm2,effort_limit=motor.peak_torque_nm,
            effort_limit_sim=motor.peak_torque_nm,velocity_limit=motor.max_speed_rad_s,velocity_limit_sim=motor.max_speed_rad_s)
        robot.actuators[group]=spatial_motor_cfg(**values,spatial_parameters=[deepcopy(params) for _ in range(design_count)])
        audit[group]=dict(motor=motor.motor_id,peak_nm=motor.peak_torque_nm,curve_points=len(curve))
    return audit

def full_body_cost(applied, available, peak, threshold=LIMIT_RATIO):
    if applied.ndim != 2 or applied.shape != available.shape or applied.shape != peak.shape or applied.shape[-1] != JOINT_COUNT:
        raise ValueError('Expected matching tensors with exactly 23 controlled joints')
    if not 0 < threshold <= 1:
        raise ValueError('threshold must be in (0,1]')
    valid=available>peak*1e-6
    safe=torch.where(valid,available,torch.ones_like(available))
    utilization=torch.where(valid,applied.abs()/safe,torch.zeros_like(applied))
    cost=torch.where(valid,(utilization-threshold).clamp(min=0).square(),torch.zeros_like(applied)).sum(-1)
    finite=(torch.isfinite(applied)&torch.isfinite(available)&torch.isfinite(peak)&(available>=0)&(peak>0)).all(-1)
    cost=torch.where(finite,cost,torch.full_like(cost,float('nan')))
    return cost,utilization,~valid

def live_full_body_cost(env, threshold=LIMIT_RATIO):
    """Read the same dynamic ceilings used by each live actuator's hard clip."""
    ids=[];limits=[];peaks=[]
    for group in GROUP_MOTORS:
        actuator=env.robot.actuators[group]
        if not hasattr(actuator,'dynamic_effort_limit'):
            raise RuntimeError(f'{group}: missing live torque-speed envelope; refuse partial constraint')
        indices=list(actuator.joint_indices)
        ids.extend(indices)
        limits.append(actuator.dynamic_effort_limit)
        peak=torch.as_tensor(actuator.effort_limit,device=env.device,dtype=env.robot.data.applied_torque.dtype)
        peaks.append(torch.broadcast_to(peak,(env.num_envs,len(indices))))
    if len(ids)!=JOINT_COUNT or len(set(ids))!=JOINT_COUNT:
        raise ValueError('Full-body torque constraint requires exactly 23 distinct controlled joints')
    applied=env.robot.data.applied_torque[:,ids];available=torch.cat(limits,-1)
    cost,utilization,unavailable=full_body_cost(applied,available,torch.cat(peaks,-1),threshold)
    env._full_body_torque_diagnostics={'utilization':utilization.detach(),'unavailable':unavailable.detach(),
        'cost':cost.detach(),'hard_clip_excess_nm':(applied.abs()-available).clamp(min=0).detach()}
    return cost

def install_full_body_reward(cfg):
    """Install once; fail on old six-joint term to prevent double counting."""
    from isaaclab.managers import RewardTermCfg
    rewards=getattr(cfg,'reward',None)
    if rewards is None:rewards=getattr(cfg,'rewards',None)
    if rewards is None:raise ValueError('No supported native reward manager configuration')
    if getattr(rewards,'design_motor_torque_limits',None) is not None:
        raise ValueError('Disable old six-joint penalty before installing full-body penalty')
    rewards.full_body_torque_limits=RewardTermCfg(func=live_full_body_cost,weight=WEIGHT,params={'threshold':LIMIT_RATIO})

def activate_for_config(cfg, spatial_motor_cfg, design_count):
    """Shared integration for native Rough, backpack and AMP configurations."""
    robot=cfg.scene.robot
    audit=install_fixed_motor_curves(robot,spatial_motor_cfg,design_count)
    for group in GROUP_MOTORS:
        cls=robot.actuators[group].class_type
        if not getattr(cls._clip_effort,'_exports_full_body_limit',False):
            original=cls._clip_effort
            def clip(self,effort,_original=original):
                self.dynamic_effort_limit=_original(self,torch.full_like(effort,1.e20)).clone()
                return _original(self,effort)
            clip._exports_full_body_limit=True
            cls._clip_effort=clip;cls._exports_full_body_limit=True
    if hasattr(cfg,'scene_context'):cfg.scene_context.robot=robot
    install_full_body_reward(cfg)
    return audit
