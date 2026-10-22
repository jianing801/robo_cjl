"""Evaluation-only fixed conditions for the source-native direct Rough task."""


def configure_nominal_rough(cfg, *, seed=42, speed=.5, duration=20.):
    disabled = []
    for name in cfg.events.to_dict():
        term = getattr(cfg.events, name)
        if term is not None and name not in ("reset_base", "reset_robot_joints"):
            disabled.append(name)
            setattr(cfg.events, name, None)
    reset = cfg.events.reset_base
    if reset.func.__name__ != "reset_root_state_uniform":
        raise ValueError("unknown root reset")
    reset.params["pose_range"] = {key: (0., 0.) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    reset.params["velocity_range"] = {key: (0., 0.) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    joints = cfg.events.reset_robot_joints
    if joints.func.__name__ not in ("reset_joints_by_offset", "reset_joints_by_scale"):
        raise ValueError("unknown joint reset")
    value = 1. if joints.func.__name__ == "reset_joints_by_scale" else 0.
    joints.params.update(position_range=(value, value), velocity_range=(0., 0.))
    cfg.noise.add_noise = False
    delays = {}
    for name, actuator in cfg.scene.robot.actuators.items():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = actuator.max_delay = 0
            delays[name] = 0
    commands = cfg.commands
    commands.debug_vis = False
    commands.heading_command = True
    commands.rel_heading_envs, commands.rel_standing_envs = 1., 0.
    commands.heading_control_stiffness = 2.
    commands.ranges.lin_vel_x = (speed, speed)
    commands.ranges.lin_vel_y = (0., 0.)
    commands.ranges.heading = (0., 0.)
    commands.ranges.ang_vel_z = (-.4, .4)
    commands.resampling_time_range = (duration + 10., duration + 10.)
    cfg.seed = seed
    cfg.episode_length_s = duration + .1
    if cfg.scene_context.height_scanner.enable_height_scan_actor:
        raise ValueError("actor must not gain terrain scan observations")
    return dict(profile="fixed_nominal_native_rough_v1", seed=seed, speed_m_s=speed,
        duration_s=duration, disabled_events=disabled, observation_noise=False,
        physical_domain_randomization=False, pushes=False, initial_pose_jitter=False,
        action_delays_steps=delays, heading_hold=True, privileged_height_scan_critic_only=True,
        identical_parallel_replicas_not_independent_randomized_trials=True)
