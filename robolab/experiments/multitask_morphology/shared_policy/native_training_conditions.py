"""PPO-only operating conditions inherited from frozen native task recipes.

Demonstration/DAgger collection and fixed-condition measurement retain their
existing profiles. Native observation corruption is replaced by the existing
shared fresh-frame sensor noise, never applied twice.
"""
from copy import deepcopy
from randomization import configure_profile

NATIVE_EXPERT_CONDITIONS = 'native_expert'
LEGACY_CONDITIONS = 'continuous_locomotion'
TRAINING_CONDITIONS = (NATIVE_EXPERT_CONDITIONS, LEGACY_CONDITIONS)
MATURE_PAYLOAD_UPDATE = 12000


def validate_training_conditions(mode, profile, strength):
    if mode not in TRAINING_CONDITIONS:
        raise ValueError('Unknown PPO operating-condition protocol')
    if mode == NATIVE_EXPERT_CONDITIONS and (profile != 'robust' or strength != 1.):
        raise ValueError('Native expert PPO conditions require full native randomization')
    return mode


def configure_native_training(cfg, task, *, seed, profile='robust', strength=1.):
    validate_training_conditions(NATIVE_EXPERT_CONDITIONS, profile, strength)
    # Reuse the shared observation adapter and audit, restoring the source's
    # exact command, events, reset distribution and actuator-delay configuration.
    result, audit = configure_profile(cfg, task, profile=profile, strength=strength, seed=seed)
    result.commands = deepcopy(cfg.commands)
    result.events = deepcopy(cfg.events)
    for name, motor in result.scene.robot.actuators.items():
        source = cfg.scene.robot.actuators[name]
        if hasattr(source, 'min_delay'):
            motor.min_delay, motor.max_delay = source.min_delay, source.max_delay
    command = result.commands.base_velocity if task == 'run' else result.commands
    audit['commands'] = dict(ranges={key: tuple(getattr(command.ranges, key)) for key in
        ('lin_vel_x', 'lin_vel_y', 'ang_vel_z', 'heading')},
        resampling_time_range=tuple(command.resampling_time_range),
        rel_standing_envs=command.rel_standing_envs, rel_heading_envs=command.rel_heading_envs,
        heading_command=command.heading_command,
        standing_commands=command.rel_standing_envs > 0., source=NATIVE_EXPERT_CONDITIONS)
    if hasattr(command, 'heading_control_stiffness'):
        audit['commands']['heading_control_stiffness'] = command.heading_control_stiffness
    audit['reset'] = dict(base=deepcopy(result.events.reset_base.params),
        joint_position=tuple(result.events.reset_robot_joints.params['position_range']),
        joint_velocity=tuple(result.events.reset_robot_joints.params['velocity_range']))
    audit['ppo_operating_conditions'] = dict(kind=NATIVE_EXPERT_CONDITIONS,
        native_terrain=True, native_reset_events=True, full_native_randomization=True,
        payload_phase_update=MATURE_PAYLOAD_UPDATE if task == 'backpack' else None,
        observation_noise='shared_fresh_frame_only')
    return result, audit


def native_payload_properties(module, num_envs, *, seed, payload_range):
    bounds = tuple(payload_range(MATURE_PAYLOAD_UPDATE))
    rows = module.sample_backpack_properties(num_envs, seed=seed, mass_range_kg=bounds)
    for i, row in enumerate(rows):
        row['env_id'] = i
        row['sampling_mass_range_kg'] = bounds
    return rows


def native_terrain_scenarios(num_envs, design_ids, terrain_columns, terrain_levels):
    if not design_ids or len(terrain_columns) != num_envs or len(terrain_levels) != num_envs:
        raise ValueError('Native terrain layout does not cover the live environments')
    return [dict(environment_id=i, design_id=design_ids[i % len(design_ids)], task='stairs',
        terrain_column=int(terrain_columns[i]), terrain_level=int(terrain_levels[i])) for i in range(num_envs)]
