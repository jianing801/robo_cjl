from types import SimpleNamespace
import random
import torch
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
