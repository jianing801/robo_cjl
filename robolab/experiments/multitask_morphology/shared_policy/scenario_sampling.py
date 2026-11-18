"""Recorded, deterministic operating conditions for shared policy collection."""
from randomization import stairs_spec, payload_spec


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
    if n % design_count:
        raise ValueError('Terrain routing requires complete design copies')
    copies = n // design_count
    if paired_validation and copies < 2 * count:
        raise ValueError('Paired stair collection requires two copies per stratum')
    copy = torch.arange(n, device=terrain.terrain_types.device) // design_count
    column = (copy // 2 if paired_validation else copy) % count
    terrain.terrain_types[:] = column
    terrain.terrain_levels[:] = 0
    origins = terrain.terrain_origins[terrain.terrain_levels, column]
    terrain.env_origins[:] = origins
    scene.env_origins[:] = origins
    # Caller must reset after routing so actual robot states use these origins.
    return copy.remainder(2).bool() if paired_validation else copy == copies - 1


def payload_properties(module, num_envs, design_count, *, seed, profile, strength, fixed_mass=None):
    if design_count < 1 or num_envs % design_count:
        raise ValueError('Payload sampling requires complete physical design copies')
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
