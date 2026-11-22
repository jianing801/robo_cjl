"""Run the frozen collection workers with a 20-second DAgger horizon."""

import argparse
import math
from pathlib import Path
import runpy
import sys

import stair_course


TASKS = ('stairs', 'backpack', 'run')
STEPS = 1000
STRIDE = 5
ENVIRONMENTS = 4096
DESIGNS = 400
CONTROL_DT = .02
TIMEOUT_STEPS = STEPS + 2
RECEIPT = 'long_replay_collection.json'


def configure_long_stairs(cfg, spec, audit):
    generator = cfg.scene.terrain.terrain_generator
    strata = spec['strata']
    count = len(strata)
    if (count != 6 or generator.num_rows != 1 or generator.num_cols != count
            or cfg.scene_context.terrain_generator is not generator):
        raise ValueError('Expected the native six-stratum stair generator')

    duration = STEPS * CONTROL_DT
    speed = max(cfg.commands.ranges.lin_vel_x)
    flights = [stair_course.evaluation_flight(duration, speed, row['step_width_m']) for row in strata]
    tile_length = max(flight['length_m'] for flight in flights)
    generator.size = (tile_length, stair_course.WIDTH_M)
    generator.border_width = 0.
    generator.sub_terrains = {
        name: stair_course.flight_tile(row['step_height_m'], row['step_width_m'], 1. / count,
                                       flight['rear_steps'])
        for name, row, flight in zip(generator.sub_terrains, strata, flights, strict=True)
    }
    cfg.events.reset_base.params['pose_range']['x'] = (0., 0.)
    audit['reset']['base']['pose_range']['x'] = (0., 0.)
    spec.update(terrain=stair_course.COURSE, duration_s=duration, tile_length_m=tile_length,
                origin_x_by_stratum_m=[flight['origin_x_m'] for flight in flights])
    return dict(course=stair_course.COURSE, stratum_count=count, tile_length_m=tile_length,
                origin_x_by_stratum_m=spec['origin_x_by_stratum_m'], max_command_mps=speed)


def verify_stair_start(raw, stairs):
    import torch

    terrain = raw.scene.terrain
    generator = terrain.cfg.terrain_generator
    count, length = stairs['stratum_count'], stairs['tile_length_m']
    if (terrain.cfg.terrain_type != 'generator' or tuple(generator.size) != (length, stair_course.WIDTH_M)
            or not terrain.terrain_prim_paths or len(generator.sub_terrains) != count
            or any(tile.function is not stair_course.flight_mesh for tile in generator.sub_terrains.values())):
        raise ValueError('Live stair terrain differs from the continuous flight geometry')
    expected = terrain.terrain_origins.new_tensor([
        (x - length / 2, (i + .5 - count / 2) * stair_course.WIDTH_M, 0.)
        for i, x in enumerate(stairs['origin_x_by_stratum_m'])
    ])
    if not torch.allclose(terrain.terrain_origins[0], expected, rtol=0., atol=2e-4):
        raise ValueError('Live stair origins differ from the six bottom platforms')
    distance = stair_course.APPROACH_M - (raw.robot.data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0])
    if not torch.allclose(distance, torch.full_like(distance, stair_course.APPROACH_M), rtol=0., atol=2e-4):
        raise ValueError('Stair reset failed to start at the base of a continuous flight')
    return dict(initial_start_environments=distance.numel(), first_riser_min_m=float(distance.min()),
                first_riser_max_m=float(distance.max()))


def stair_split(scene, validation, design_count):
    import torch

    columns = scene.terrain.terrain_types
    count = 6
    indices = torch.arange(columns.numel(), device=columns.device) % design_count
    train, held_out = [torch.bincount(indices[mask] * count + columns[mask], minlength=design_count * count)
                       .reshape(design_count, count) for mask in (~validation, validation)]
    if (train == 0).any() or (held_out.sum(dim=1) == 0).any():
        raise ValueError('Long stair collection must retain every native training stratum and held-out copy')
    return dict(training_copies_by_stratum=train.sum(dim=0).tolist(),
                validation_copies_by_stratum=held_out.sum(dim=0).tolist())


def native_arguments(native):
    # Read the variant's invariants and forward all approved options verbatim.
    parser = argparse.ArgumentParser(add_help=False)
    for key in ('plan', 'design-ids', 'output', 'student', 'profile'):
        parser.add_argument('--' + key, required=True)
    for key in ('steps', 'stride', 'num-envs', 'seed'):
        parser.add_argument('--' + key, type=int, required=True)
    parser.add_argument('--randomization-strength', type=float, required=True)
    parser.add_argument('--mode', default='collect')
    for key in ('task', 'ppo-channel', 'bo-conditions', 'payload-kg'):
        parser.add_argument('--' + key)
    args, _ = parser.parse_known_args(native)
    if (any(value is not None for value in (args.task, args.ppo_channel, args.bo_conditions, args.payload_kg))
            or (args.steps, args.stride, args.num_envs) != (STEPS, STRIDE, ENVIRONMENTS)
            or (args.mode, args.profile, args.randomization_strength) != ('collect', 'robust', 1.)):
        parser.error('Long DAgger requires robust strength 1 collection: 1000 steps, stride 5, and 4096 environments')
    args.design_ids = args.design_ids.split(',')
    if len(args.design_ids) != DESIGNS or len(set(args.design_ids)) != DESIGNS or '' in args.design_ids:
        parser.error('Long DAgger requires 400 unique training designs')
    return args


def run_collector(collector, task, native, args):
    # This entry point runs in a dedicated worker process. Keep the original
    # source files and import precedence, including for the AMP helper imports.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(collector))
    import collection_grid
    import contract
    import randomization
    import scenario_sampling

    output = Path(args.output).resolve()
    evidence = dict(kind='dagger_long_collection_v1', task=task, steps=STEPS, stride=STRIDE, seed=args.seed,
                    terrain='continuous_bottom_start' if task == 'stairs' else 'native',
                    duration_s=STEPS * CONTROL_DT)
    configure_profile = randomization.configure_profile
    verify_collection_layout = scenario_sampling.verify_collection_layout
    raw_env = None
    initial_control_step = None
    profile_audit = None

    def configure_profile_with_timeout(cfg, native_task, **options):
        nonlocal profile_audit
        cfg, profile_audit = configure_profile(cfg, native_task, **options)
        cfg.episode_length_s = max(cfg.episode_length_s, TIMEOUT_STEPS * CONTROL_DT)
        return cfg, profile_audit

    def verify_layout_with_timeout(raw, designs, count):
        nonlocal raw_env, initial_control_step
        verify_collection_layout(raw, designs, count)
        limit = int(raw.max_episode_length if task == 'run' else raw.episode_length)
        seconds = float(raw.cfg.episode_length_s)
        dt = float(raw.step_dt)
        if (count != ENVIRONMENTS or [design['id'] for design in designs] != args.design_ids
                or not math.isclose(dt, CONTROL_DT, rel_tol=0., abs_tol=1e-10)
                or not math.isfinite(seconds) or seconds < TIMEOUT_STEPS * CONTROL_DT or limit < TIMEOUT_STEPS):
            raise ValueError('Live environment layout or native episode horizon differs from long replay')
        raw_env = raw
        initial_control_step = int(raw.common_step_counter)
        evidence.update(environment_count=count, design_count=len(designs), control_dt_s=dt,
                        configured_episode_length_s=seconds, native_episode_limit_steps=limit)

    randomization.configure_profile = configure_profile_with_timeout
    scenario_sampling.verify_collection_layout = verify_layout_with_timeout
    if task == 'stairs':
        configure_stairs = scenario_sampling.configure_stairs
        route_stairs = scenario_sampling.route_stairs
        verify_stair_origins = scenario_sampling.verify_stair_origins

        def configure_stair_flights(cfg, *, profile, seed, evaluation_seconds=None):
            if evaluation_seconds is not None:
                raise ValueError('Long replay requires native stairs in collection mode')
            spec = configure_stairs(cfg, profile=profile, seed=seed)
            evidence['stairs'] = configure_long_stairs(cfg, spec, profile_audit)
            evidence['stairs']['stair_course_sha256'] = contract.sha(stair_course.__file__)
            return spec

        def route_stair_flights(scene, design_count, *, profile, paired_validation=False):
            if profile != 'robust' or not paired_validation or design_count != DESIGNS:
                raise ValueError('Expected the original paired robust stair collection layout')
            validation = route_stairs(scene, design_count, profile=profile, paired_validation=paired_validation)
            evidence['stairs'].update(stair_split(scene, validation, design_count))
            return validation

        def verify_stair_flights(scene, columns):
            verify_stair_origins(scene, columns)
            stairs = evidence['stairs']
            if 'initial_start_environments' not in stairs:
                stairs.update(verify_stair_start(raw_env, stairs))

        scenario_sampling.configure_stairs = configure_stair_flights
        scenario_sampling.route_stairs = route_stair_flights
        scenario_sampling.verify_stair_origins = verify_stair_flights

    worker = collector / ('rollout_amp.py' if task == 'run' else 'rollout.py')
    sys.argv = [str(worker), *native]
    if task != 'run':
        sys.argv.extend(('--task', task))
    # Use the original main and bypass its CLI os._exit shim so the adapter
    # can attach to native publication before Kit takes down the interpreter.
    main = runpy.run_path(str(worker), run_name='_dagger_native_collector')['main']
    globals_ = main.__globals__
    load_plan, write_new = globals_['load_plan'], globals_['write_new']
    student_sha = contract.sha(args.student)

    def verify_training_grid(path):
        plan = load_plan(path)
        grid = collection_grid.select_collection_grid(plan['selected_structure_keys'])
        if contract.design_ids_for_keys(plan, grid)[task] != args.design_ids:
            raise ValueError('Long replay requires the original training grid in native order')
        return plan

    def publish_with_evidence(path, value):
        if path != output/'summary.json':
            write_new(path, value)
            return
        result = 'REAL_AMP_ROLLOUT_COMPLETE' if task == 'run' else 'REAL_ROLLOUT_COMPLETE'
        expected = dict(result=result, task=task, steps=STEPS, seed=args.seed, student=student_sha,
                        teacher_updates=0, design_ids=args.design_ids)
        if (any(value.get(k) != expected[k] for k in expected) or raw_env is None
                or len(value['rows']) != ENVIRONMENTS or not (output/'data.pt').is_file()
                or (task == 'stairs' and evidence['stairs'].get('initial_start_environments') != ENVIRONMENTS)):
            raise ValueError('Frozen collector did not publish the verified long replay')
        executed = int(raw_env.common_step_counter) - initial_control_step
        if executed != STEPS:
            raise ValueError(f'Frozen collector did not execute {STEPS} native control steps')
        if task == 'stairs' and evidence['stairs']['stair_course_sha256'] != contract.sha(stair_course.__file__):
            raise ValueError('Stair flight source changed during collection')
        write_new(path, value)
        evidence['executed_control_steps'] = executed
        write_new(output/RECEIPT, dict(evidence, summary_sha256=contract.sha(path)))

    globals_.update(load_plan=verify_training_grid, write_new=publish_with_evidence)
    return main()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--collector', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('native', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.native[:1] != ['--']:
        parser.error('Separate the native collection arguments with --')
    native = options.native[1:]
    return run_collector(options.collector.resolve(), options.task, native, native_arguments(native))


if __name__ == '__main__':
    sys.exit(main())
