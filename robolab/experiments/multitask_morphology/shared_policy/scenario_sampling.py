"""Recorded, deterministic operating conditions for shared policy collection."""
from randomization import stairs_spec, payload_spec


def rollout_env_count(args, design_count, default_copies, *, training_strata=1):
    """Keep the native copy budget unless an exact collection size was requested."""
    if args.num_envs is None:
        return design_count * default_copies
    if args.mode != 'collect' or args.ppo_channel or args.bo_conditions:
        raise ValueError('--num-envs is only supported for demonstration or DAgger collection')
    if design_count < 1 or args.num_envs < design_count * (training_strata + 1):
        raise ValueError('Exact environment count must cover each training stratum and a held-out copy per design')
    return args.num_envs


def last_design_copy(environment_ids, design_count):
    """The final round-robin window contains the last copy of every design."""
    return environment_ids >= len(environment_ids) - design_count


def verify_collection_layout(raw, designs, num_envs):
    """Verify that tensor rows, USD environment indices and native contexts agree."""
    import torch
    spawn = raw.cfg.scene.robot.spawn
    if (spawn.random_choice or list(spawn.spatial_design_ids) != [d['id'] for d in designs]
            or [asset.asset_path for asset in spawn.assets_cfg] != [d['urdf'] for d in designs]):
        raise ValueError('Native URDF spawn order differs from the round-robin design assignment')
    paths = raw.robot.root_physx_view.prim_paths
    if raw.num_envs != num_envs or len(paths) != num_envs:
        raise ValueError('Physical articulation count differs from requested collection size')
    for index, path in enumerate(paths):
        robot_path = f'/World/envs/env_{index}/Robot'
        if path != robot_path and not path.startswith(robot_path + '/'):
            raise ValueError(f'Physical articulation row {index} points to {path}, expected {robot_path}')
    groups = torch.arange(num_envs, device=raw.spatial_groups.device) % len(designs)
    contexts = raw.spatial_context.new_tensor([design['context'] for design in designs])
    if not torch.equal(groups, raw.spatial_groups) or not torch.equal(contexts[groups], raw.spatial_context):
        raise ValueError('Native teacher context differs from the round-robin design assignment')


def configure_stairs(cfg, *, profile, seed):
    # Lazy imports keep protocol and sampling tests independent of Isaac Lab.
    import isaaclab.terrains as terrains
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
    spec = stairs_spec(profile)
    strata = spec['strata']
    generator = TerrainGeneratorCfg(seed=seed, curriculum=True, size=(40., 40.), border_width=2.,
        num_rows=1, num_cols=len(strata), horizontal_scale=.05, vertical_scale=.005,
        difficulty_range=(.5, .5), use_cache=False,
        sub_terrains={f'stairs_{i}': terrains.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1. / len(strata), step_height_range=(row['step_height_m'],) * 2,
            step_width=row['step_width_m'], platform_width=2., border_width=1., holes=False)
            for i, row in enumerate(strata)})
    for terrain in (cfg.scene.terrain, cfg.scene_context):
        terrain.terrain_type = 'generator'
        terrain.terrain_generator = generator
        terrain.max_init_terrain_level = 0
    return spec


def route_stairs(scene, design_count, *, profile, paired_validation=False):
    """Route physical reset origins, keeping each morphology's strata balanced."""
    import torch
    terrain = scene.terrain
    count = len(stairs_spec(profile)['strata'])
    n = terrain.terrain_types.numel()
    if design_count < 1 or n < design_count or terrain.terrain_origins.shape != (1, count, 3):
        raise ValueError('Terrain origins or design count differ from the declared stair layout')
    copies = n // design_count
    environments = torch.arange(n, device=terrain.terrain_types.device)
    copy = environments // design_count
    if paired_validation and copies < 2 * count:
        if copies <= count:
            raise ValueError('Stair collection requires all training strata and a held-out copy per design')
        # Keep all six training columns. Rotate the remaining four or five
        # held-out columns across designs when the scene cannot fit six pairs.
        validation = copy >= count
        column = (copy + environments % design_count) % count
    elif paired_validation:
        column = (copy // 2) % count
        validation = copy.remainder(2).bool()
    else:
        column = copy % count
        validation = last_design_copy(environments, design_count)
    terrain.terrain_types[:] = column
    terrain.terrain_levels[:] = 0
    origins = terrain.terrain_origins[terrain.terrain_levels, column]
    terrain.env_origins[:] = origins
    scene.env_origins[:] = origins
    # Caller must reset after routing so actual robot states use these origins.
    return validation


def verify_stair_origins(scene, columns):
    """Check live terrain and reset origins against the labels assigned at startup."""
    import torch
    terrain = scene.terrain
    if not torch.equal(terrain.terrain_types, columns) or torch.any(terrain.terrain_levels != 0):
        raise ValueError('Live stair column or level changed since physical environments were routed')
    origins = terrain.terrain_origins[0, columns]
    if not torch.equal(scene.env_origins, origins) or not torch.equal(terrain.env_origins, origins):
        raise ValueError('Live stair environment origin differs from the assigned mesh column')


def payload_properties(module, num_envs, design_count, *, seed, profile, strength, fixed_mass=None):
    if design_count < 1 or num_envs < design_count:
        raise ValueError('Payload sampling requires a physical environment for every design')
    if fixed_mass is not None and not 0 < fixed_mass <= 15:
        raise ValueError('Fixed payload must be in (0, 15] kg')
    spec = payload_spec(profile, strength=strength)
    result = []
    for i in range(num_envs):
        pair = (i // design_count) // 2
        if fixed_mass is not None:
            bounds = (fixed_mass, fixed_mass)
        elif profile == 'nominal':
            points = spec['nominal_mass_points_kg']
            bounds = (points[pair % len(points)],) * 2
        else:
            # Adjacent training/validation copies share a stratum, not a draw.
            bounds = ((.1, 5.), (5., 10.), (10., 15.))[(pair + seed % 3) % 3]
        row = module.sample_backpack_properties(1, seed=seed + i,
            mass_range_kg=bounds, inertia_size_scale_range=spec['inertia_size_scale_range'],
            mount_surface_offset_max_m=spec['mount_surface_offset_max_m'],
            com_offset_max_m=spec['com_offset_max_m'])[0]
        row['env_id'] = i
        row['sampling_mass_range_kg'] = bounds
        result.append(row)
    return result


def payload_bins(masses):
    import torch
    return torch.bucketize(masses, masses.new_tensor([5., 10.]))


def scenario_table(task, num_envs, design_ids, *, terrain_indices=None, profile='nominal', properties=None):
    strata = stairs_spec(profile)['strata'] if task == 'stairs' else None
    result = []
    for i in range(num_envs):
        row = dict(environment_id=i, design_id=design_ids[i % len(design_ids)], task=task)
        if strata is not None:
            index = int(terrain_indices[i])
            if not 0 <= index < len(strata):
                raise ValueError('Actual terrain column is outside the declared stair strata')
            row.update(terrain_stratum=index, **strata[index])
        if properties is not None:
            row['payload'] = properties[i]
        result.append(row)
    return result
