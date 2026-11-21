from types import SimpleNamespace
import random
import pytest
import torch
import scenario_sampling
from scenario_sampling import payload_properties, payload_bins, scenario_table, route_stairs


def sampler(count, *, seed, mass_range_kg, **kwargs):
    return [dict(env_id=0, mass_kg=random.Random(seed).uniform(*mass_range_kg), **kwargs)]


def test_payload_nominal_matches_old_four_points_and_records_real_env_ids():
    rows = payload_properties(SimpleNamespace(sample_backpack_properties=sampler), 16, 2,
                              seed=42, profile='nominal', strength=1.)
    assert [r['env_id'] for r in rows] == list(range(16))
    assert [r['mass_kg'] for r in rows] == [.3] * 4 + [5.] * 4 + [10.] * 4 + [15.] * 4
    assert all(r['inertia_size_scale_range'] == (1., 1.) for r in rows)


def test_payload_strata_cover_each_design_and_train_validation_use_distinct_draws():
    module = SimpleNamespace(sample_backpack_properties=sampler)
    rows = payload_properties(module, 16, 2, seed=42, profile='robust', strength=.5)
    again = payload_properties(module, 16, 2, seed=42, profile='robust', strength=.5)
    assert rows == again
    for design in range(2):
        training = torch.tensor([rows[i]['mass_kg'] for i in range(design, 16, 4)])
        assert set(payload_bins(training).tolist()) == {0, 1, 2}
    assert rows[0]['sampling_mass_range_kg'] == rows[2]['sampling_mass_range_kg']
    assert rows[0]['mass_kg'] != rows[2]['mass_kg']


def test_scenario_records_actual_terrain_route():
    rows = scenario_table('stairs', 6, ['a', 'b'], terrain_indices=[0, 1, 2, 3, 4, 5], profile='robust')
    assert [r['design_id'] for r in rows] == ['a', 'b'] * 3
    assert {(r['step_height_m'], r['step_width_m']) for r in rows} == {
        (h, w) for h in (.05, .075, .1) for w in (.25, .35)}


def test_stair_routing_moves_real_origins_and_covers_both_splits():
    design_count = 3
    n = design_count * 12
    terrain = SimpleNamespace(terrain_types=torch.zeros(n, dtype=torch.long),
        terrain_levels=torch.zeros(n, dtype=torch.long),
        terrain_origins=torch.arange(18.).reshape(1, 6, 3), env_origins=torch.zeros(n, 3))
    scene = SimpleNamespace(terrain=terrain, env_origins=torch.zeros(n, 3))
    validation = route_stairs(scene, design_count, profile='robust', paired_validation=True)
    for design in range(design_count):
        for split in (validation, ~validation):
            selected = split & (torch.arange(n) % design_count == design)
            assert set(terrain.terrain_types[selected].tolist()) == set(range(6))
    assert torch.equal(scene.env_origins, terrain.terrain_origins[0, terrain.terrain_types])
    assert torch.equal(terrain.env_origins, scene.env_origins)


def test_stair_evaluation_covers_all_strata_for_each_design():
    n = 24
    terrain = SimpleNamespace(terrain_types=torch.zeros(n, dtype=torch.long),
        terrain_levels=torch.zeros(n, dtype=torch.long),
        terrain_origins=torch.arange(18.).reshape(1, 6, 3), env_origins=torch.zeros(n, 3))
    scene = SimpleNamespace(terrain=terrain, env_origins=torch.zeros(n, 3))
    route_stairs(scene, 3, profile='robust')
    assert all(set(terrain.terrain_types[d::3].tolist()) == set(range(6)) for d in range(3))


def test_exact_collection_count_and_last_run_copy():
    args = SimpleNamespace(num_envs=4096, mode='collect', ppo_channel=None, bo_conditions=None)
    assert scenario_sampling.rollout_env_count(args, 400, 8) == 4096
    environments = torch.arange(4096)
    held_out = scenario_sampling.last_design_copy(environments, 400)
    assert held_out.nonzero().flatten().tolist() == list(range(3696, 4096))
    assert torch.bincount(environments[held_out] % 400).tolist() == [1] * 400
    assert torch.bincount(environments[~held_out] % 400).tolist() == [10] * 96 + [9] * 304
    args.num_envs = None
    assert scenario_sampling.rollout_env_count(args, 400, 8) == 3200


@pytest.mark.parametrize('override', [dict(num_envs=799), dict(mode='evaluate'),
    dict(ppo_channel='ppo'), dict(bo_conditions='bo')])
def test_exact_collection_count_rejects_invalid_or_unrelated_modes(override):
    values = dict(num_envs=4096, mode='collect', ppo_channel=None, bo_conditions=None)
    with pytest.raises(ValueError):
        scenario_sampling.rollout_env_count(SimpleNamespace(**(values | override)), 400, 8)


def test_partial_stair_copies_cover_all_training_strata_with_independent_validation():
    n = 4096
    origins = torch.arange(18.).reshape(1, 6, 3)
    terrain = SimpleNamespace(terrain_types=torch.zeros(n, dtype=torch.long),
        terrain_levels=torch.zeros(n, dtype=torch.long), terrain_origins=origins,
        env_origins=torch.zeros(n, 3))
    scene = SimpleNamespace(terrain=terrain, env_origins=torch.zeros(n, 3))
    held_out = route_stairs(scene, 400, profile='robust', paired_validation=True)
    assert torch.equal(scene.env_origins, origins[0, terrain.terrain_types])
    assert torch.equal(terrain.env_origins, scene.env_origins)
    for design in range(400):
        mask = held_out[design::400]
        columns = terrain.terrain_types[design::400]
        assert sorted(columns[~mask].tolist()) == list(range(6))
        distinct = columns[mask].unique()
        assert len(distinct) == (5 if design < 96 else 4)
    coverage = torch.bincount(terrain.terrain_types[held_out])
    assert len(coverage) == 6 and int(coverage.max() - coverage.min()) <= 2


def test_payload_partial_final_copy_keeps_its_design_and_physical_conditions():
    module = SimpleNamespace(sample_backpack_properties=sampler)
    rows = payload_properties(module, 13, 2, seed=42, profile='robust', strength=1.)
    scenarios = scenario_table('backpack', 13, ['a', 'b'], properties=rows)
    assert scenarios[-1]['design_id'] == 'a'
    assert scenarios[-1]['payload']['env_id'] == 12
    for design in range(2):
        for split in range(2):
            bins = {row['sampling_mass_range_kg'] for row in rows[design + 2 * split::4]}
            assert bins == {(.1, 5.), (5., 10.), (10., 15.)}


def test_verify_native_layout_checks_actual_articulation_order_and_teacher_context():
    designs = [dict(id='a', urdf='a.urdf', context=[.2, .3]), dict(id='b', urdf='b.urdf', context=[.4, .5])]
    spawn = SimpleNamespace(random_choice=False, spatial_design_ids=['a', 'b'],
        assets_cfg=[SimpleNamespace(asset_path='a.urdf'), SimpleNamespace(asset_path='b.urdf')])
    paths = [f'/World/envs/env_{i}/Robot/torso' for i in range(5)]
    view = SimpleNamespace(prim_paths=paths)
    raw = SimpleNamespace(num_envs=5, robot=SimpleNamespace(root_physx_view=view),
        cfg=SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn))),
        spatial_groups=torch.tensor([0, 1, 0, 1, 0]),
        spatial_context=torch.tensor([[.2,.3],[.4,.5],[.2,.3],[.4,.5],[.2,.3]]))
    scenario_sampling.verify_collection_layout(raw, designs, 5)
    paths[2], paths[4] = paths[4], paths[2]
    with pytest.raises(ValueError, match='articulation'):
        scenario_sampling.verify_collection_layout(raw, designs, 5)
    paths[2], paths[4] = paths[4], paths[2]
    raw.spatial_context[-1] = raw.spatial_context[1]
    with pytest.raises(ValueError, match='context'):
        scenario_sampling.verify_collection_layout(raw, designs, 5)
    raw.spatial_context[-1] = raw.spatial_context[0]
    spawn.assets_cfg.reverse()
    with pytest.raises(ValueError, match='URDF'):
        scenario_sampling.verify_collection_layout(raw, designs, 5)


def test_changed_stair_origin_after_reset_is_detected_before_labelling():
    columns = torch.tensor([0, 1, 2, 3, 4, 5, 0])
    origins = torch.arange(18.).reshape(1, 6, 3)
    terrain = SimpleNamespace(terrain_types=columns.clone(), terrain_levels=torch.zeros_like(columns),
        terrain_origins=origins, env_origins=origins[0, columns])
    scene = SimpleNamespace(terrain=terrain, env_origins=terrain.env_origins.clone())
    scenario_sampling.verify_stair_origins(scene, columns)
    scene.env_origins[-1] = scene.env_origins[1]
    with pytest.raises(ValueError, match='origin'):
        scenario_sampling.verify_stair_origins(scene, columns)
    scene.env_origins[-1] = scene.env_origins[0]
    terrain.terrain_types[-1] = 1
    with pytest.raises(ValueError, match='column'):
        scenario_sampling.verify_stair_origins(scene, columns)
