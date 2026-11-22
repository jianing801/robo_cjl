"""Fixed BO courses and pre-reset samples from the native simulator."""
from pathlib import Path

import numpy as np

from shared_bo_metrics import DT, TRIAL_VERSION, digest, read_conditions, score_trial
from contract import sha, write_new
from stair_course import COURSE, WIDTH_M, evaluation_flight, flight_tile, outside_flight


def straight_stairs(_difficulty, cfg):
    """Parallel treads with explicitly located bottom and top platforms."""
    import trimesh
    length, width = cfg.size
    height = cfg.step_height_range[0]
    start = 4.
    first = start + cfg.approach_m
    depth = cfg.step_width
    top = first + cfg.step_count * depth
    meshes = [trimesh.creation.box((length, width, .2), transform=trimesh.transformations.translation_matrix(
        (length / 2, width / 2, -.1)))]
    for i in range(cfg.step_count):
        left = first + i * depth
        right = top if i == cfg.step_count - 1 else left + depth
        elevation = (i + 1) * height
        meshes.append(trimesh.creation.box((right - left, width, elevation),
            transform=trimesh.transformations.translation_matrix(((left + right) / 2, width / 2, elevation / 2))))
    elevation = cfg.step_count * height
    meshes.append(trimesh.creation.box((length - top, width, elevation),
        transform=trimesh.transformations.translation_matrix(((top + length) / 2, width / 2, elevation / 2))))
    return meshes, np.array((start, width / 2, 0.))


def configure_conditions(cfg, task, conditions, seed, audit=None):
    spec = conditions['tasks'][task]
    command = cfg.commands.base_velocity if task == 'run' else cfg.commands
    command.rel_standing_envs = 0.
    command.rel_heading_envs = 0.
    if hasattr(command, 'heading_command'):
        command.heading_command = False
    command.resampling_time_range = (1000., 1000.)
    command.ranges.lin_vel_x = (spec['speed_mps'],) * 2
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    command.ranges.heading = (0., 0.)
    # Isaac's native horizon must not interrupt the declared measurement.
    cfg.episode_length_s = max(cfg.episode_length_s, conditions['duration_s'] + DT * 2)
    flight = None
    if task != 'run':
        import isaaclab.terrains as terrains

        size = (40., 40.)
        if task == 'stairs' and spec.get('course') == COURSE:
            flight = evaluation_flight(conditions['duration_s'], spec['speed_mps'], spec['step_depth_m'],
                                       spec['travel_margin_factor'])
            tile = flight_tile(spec['step_height_m'], spec['step_depth_m'], 1., flight['rear_steps'])
            size = (flight['length_m'], WIDTH_M)
        elif task == 'stairs':
            from isaaclab.utils import configclass

            @configclass
            class StraightStairsCfg(terrains.MeshInvertedPyramidStairsTerrainCfg):
                function = straight_stairs
                step_count: int = 12
                approach_m: float = 1.

            tile = StraightStairsCfg(proportion=1., step_height_range=(spec['step_height_m'],) * 2,
                step_width=spec['step_depth_m'], platform_width=2., border_width=0., holes=False,
                step_count=spec['step_count'], approach_m=spec['approach_m'])
        else:
            # Use a mesh plane so the native ground-filtered fall sensors keep their paths.
            tile = terrains.MeshPlaneTerrainCfg(proportion=1.)
        generator = terrains.TerrainGeneratorCfg(seed=seed, size=size, num_rows=1, num_cols=1,
            border_width=0., curriculum=True, use_cache=False, sub_terrains={'bo_course': tile})
        for terrain in (cfg.scene.terrain, cfg.scene_context):
            terrain.terrain_type = 'generator'
            terrain.terrain_generator = generator
            terrain.max_init_terrain_level = 0
    elif cfg.scene.terrain.terrain_type != 'plane':
        raise ValueError(f'Fixed BO {task} requires the native flat scene')
    if conditions.get('reset_mode') == 'fixed' or flight is not None:
        base = cfg.events.reset_base.params
        for key in ('pose_range', 'velocity_range'):
            base[key] = {axis: (0., 0.) for axis in base[key]}
        joint_position = (1., 1.) if task == 'run' else (0., 0.)
        cfg.events.reset_robot_joints.params.update(position_range=joint_position, velocity_range=(0., 0.))
        if audit is not None:
            audit['reset'].update(base=dict(base), joint_position=joint_position, joint_velocity=(0., 0.))
    if audit is not None:
        audit['commands'].update(
            ranges={k: tuple(getattr(command.ranges, k))
                    for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z', 'heading')},
            resampling_time_range=command.resampling_time_range, rel_standing_envs=0.,
            rel_heading_envs=0., source=TRIAL_VERSION)
        if task == 'stairs':
            audit['stairs'] = dict(terrain='straight_stairs_bo', **spec)
            if flight:
                audit['stairs'].update(flight)


def parse_conditions(args, ids):
    if args.bo_conditions is None:
        return None
    conditions = read_conditions(args.bo_conditions)
    if (len(ids) != 1 or args.mode != 'evaluate' or args.ppo_channel or args.profile != 'nominal'
            or args.randomization_strength != 0. or args.reset_yaw_range != 0.
            or args.seed not in conditions['seeds'] or args.steps != round(conditions['duration_s'] / DT)):
        raise ValueError('BO requires one design and the declared nominal seed, horizon and reset profile')
    return conditions


class CaptureReset:
    def _reset_idx(self, env_ids):
        recorder = getattr(self, 'bo_recorder', None)
        if recorder is not None:
            recorder.before_reset(env_ids)
        return super()._reset_idx(env_ids)


def motor_records(raw, bank, design):
    from robolab.physical_snapshot import catalog
    try:
        from full_body_torque import GROUP_MOTORS
    except ModuleNotFoundError:
        from amp_rough_alignment import rough_module
        GROUP_MOTORS = rough_module('full_body_torque').GROUP_MOTORS
    snapshot = catalog(bank['snapshot_root'])
    motor_ids = dict(GROUP_MOTORS, knees=design['knee_motor'], feet=design['ankle_motor'])
    records = {}
    for group, motor_id in motor_ids.items():
        spec = snapshot.get_motor_spec(motor_id)
        manufacturer = spec.manufacturer_parameters or {}
        source = manufacturer.get('source_page') or spec.source_url
        actuator = raw.robot.actuators[group]
        if not hasattr(actuator, 'dynamic_effort_limit'):
            # Before the first physics action the attribute may not yet be published.
            if not getattr(type(actuator), '_full_body_limit', False):
                raise ValueError('BO requires verified full-body explicit torque curves')
        for joint in actuator.joint_names:
            if joint in records:
                raise ValueError('A physical joint maps to two motors')
            records[joint] = dict(joint=joint, motor_id=motor_id, rated_torque_nm=spec.rated_torque_nm,
                nominal_peak_torque_nm=spec.peak_torque_nm, source=source,
                rated_conditions=manufacturer, catalog_note=spec.data_notes, torque_side='joint_output')
    names = raw.robot.joint_names
    if set(names) != set(records):
        raise ValueError('Missing motor mapping for the whole robot')
    return [records[joint] for joint in names]


class TrialCapture:
    """Capture the first attempt per replica, including the final pre-reset sample."""
    def __init__(self, raw, task, conditions, seed, motors, properties=None):
        self.raw, self.task, self.conditions, self.seed = raw, task, conditions, seed
        self.motors, self.properties = motors, properties
        self.spec = conditions['tasks'][task]
        self.count = raw.num_envs
        self.steps = round(conditions['duration_s'] / DT)
        if self.count != conditions['copies'] or abs(raw.step_dt - DT) > 1e-10:
            raise ValueError('Unexpected BO environment count/control step')
        self.initial = raw.robot.data.root_pos_w.detach().cpu().numpy().copy()
        self.origins = raw.scene.env_origins.detach().cpu().numpy().copy()
        self.mass = raw.robot.root_physx_view.get_masses().sum(dim=1).cpu().numpy().copy()
        if not np.isfinite(self.mass).all() or (self.mass <= 0.).any():
            raise ValueError('Invalid effective total mass')
        self.fields = ('position', 'vx', 'omega', 'torque', 'joint_vel')
        self.samples = []
        self.active = np.ones(self.count, dtype=bool)
        self.fell = np.zeros(self.count, dtype=bool)
        self.timeout = self.fell.copy()
        self.terminal = None
        if task == 'backpack':
            self.payload = raw.robot.backpack_mass_kg.cpu().numpy().copy()
            if not np.allclose(self.payload, self.spec['payload_kg'], atol=1e-5, rtol=0):
                raise ValueError('Actual applied payload differs from the fixed BO conditions')
        raw.bo_recorder = self

    def snapshot(self, ids=slice(None)):
        data = self.raw.robot.data
        tensors = (data.root_pos_w[ids], data.root_lin_vel_w[ids, 0], data.root_ang_vel_b[ids, :2],
                   data.applied_torque[ids], data.joint_vel[ids])
        return [x.detach().cpu().numpy().copy() for x in tensors]

    def before_reset(self, ids):
        if ids is None:
            ids = slice(None)
            indices = np.arange(self.count)
        else:
            indices = np.asarray(ids.cpu() if hasattr(ids, 'cpu') else ids, dtype=int)
        self.terminal = (indices, self.snapshot(ids))

    def capture(self, command, dones):
        if not np.allclose(command.detach().cpu().numpy(),
                           (self.spec['speed_mps'], 0., 0.), atol=1e-6, rtol=0):
            raise ValueError('Actual BO command differs from the fixed forward command')
        done = dones.detach().cpu().numpy().reshape(-1).astype(bool)
        state = self.snapshot()
        reset = self.terminal
        self.terminal = None
        if (done & self.active).any():
            if reset is None or not set(np.flatnonzero(done & self.active)).issubset(reset[0]):
                raise ValueError('BO terminal data must precede the simulator reset')
            for combined, terminal in zip(state, reset[1], strict=True):
                combined[reset[0]] = terminal
        terminated = self.raw.reset_terminated.detach().cpu().numpy().reshape(-1).astype(bool)
        self.fell |= self.active & done & terminated
        self.timeout |= self.active & done & ~terminated
        self.samples.append((*state, self.active.copy()))
        self.active &= ~done

    def publish(self, output, identity):
        self.raw.bo_recorder = None
        values = {name: np.stack(items) for name, items in zip(
            (*self.fields, 'active'), zip(*self.samples, strict=True), strict=True)}
        trace = Path(output) / 'bo_samples.npz'
        with trace.open('xb') as stream:
            np.savez_compressed(stream, **values, initial_position=self.initial, origins=self.origins,
                                total_mass_kg=self.mass, fell=self.fell, timeout=self.timeout)
        rows = []
        for i in range(self.count):
            active = values['active'][:, i]
            samples = {name: values[name][active, i] for name in self.fields}
            end = samples.pop('position')[-1]
            forward = float(end[0] - self.initial[i, 0])
            reached_top = None
            outside = False
            if self.task == 'stairs' and self.spec.get('course') == COURSE:
                flight = evaluation_flight(self.conditions['duration_s'], self.spec['speed_mps'],
                                           self.spec['step_depth_m'], self.spec['travel_margin_factor'])
                positions = values['position'][active, i] - self.origins[i]
                outside = bool(outside_flight(positions[:, 0], positions[:, 1],
                                              flight['origin_x_m'], flight['length_m']).any())
                route_completed = None
            elif self.task == 'stairs':
                top = self.spec['approach_m'] + self.spec['step_count'] * self.spec['step_depth_m']
                height = self.spec['step_count'] * self.spec['step_height_m']
                positions = values['position'][active, i] - self.origins[i]
                reached_top = bool(((positions[:, 0] >= top) &
                    (positions[:, 2] >= self.initial[i, 2] - self.origins[i, 2] + height - .15)).any())
                route_completed = reached_top
            else:
                route_completed = bool((values['position'][active, i, 0] - self.initial[i, 0]
                                        >= self.spec['route_distance_m']).any())
            payload = None
            if self.properties is not None:
                payload = dict(mass_kg=float(self.payload[i]),
                    mount_position_torso_m=list(self.properties[i]['mount_position_torso_m']),
                    center_of_mass_torso_m=list(self.properties[i]['com_position_torso_m']))
            rows.append(dict(replica=i, **score_trial(self.task, speed=self.spec['speed_mps'], dt=DT,
                expected_steps=self.steps, mass=float(self.mass[i]), forward=forward,
                motors=self.motors, fell=self.fell[i], timeout=self.timeout[i], reached_top=reached_top,
                route_completed=route_completed, payload=payload, outside_stairs=outside, **samples)))
        write_new(Path(output)/'bo_trials.json', dict(kind=TRIAL_VERSION, task=self.task, seed=self.seed,
            conditions_sha256=digest(self.conditions), trace=str(trace), trace_sha256=sha(trace),
            motors=self.motors, trials=rows, **identity))
