"""Check the actual mesh and scored travel window without launching the simulator."""
from types import SimpleNamespace

import numpy as np
import pytest

from stair_course import evaluation_flight, flight_mesh, outside_flight


@pytest.mark.parametrize("duration,speed,expected", [(20, .5, 67), (10, .5, 34), (20, .6, 80)])
def test_starts_on_floor_then_climbs_until_past_target_distance(duration, speed, expected):
    depth, height = .3, .075
    flight = evaluation_flight(duration, speed, depth)
    assert flight['forward_steps'] == expected
    cfg = SimpleNamespace(size=(flight['length_m'], 40.), step_width=depth,
                          step_height_range=(height, height), rear_steps=expected)
    meshes, origin = flight_mesh(0., cfg)
    bounds = np.asarray([mesh.bounds for mesh in meshes])
    def surface(x):
        cells = bounds[(bounds[:, 0, 0] < x) & (x < bounds[:, 1, 0])]
        return cells[:, 1, 2].max()
    assert surface(origin[0]) == pytest.approx(origin[2])
    assert origin[2] == pytest.approx(0.)
    assert origin[0] >= duration * speed * 2
    for x in (-.999, -.3, 0., .3, .999):
        assert surface(origin[0] + x) == pytest.approx(0.)
    for i in range(1, expected + 1):
        offset = 1. + (i - .5) * depth
        assert surface(origin[0] + offset) == pytest.approx(i * height)
        assert surface(origin[0] - offset) == pytest.approx(i * height)
    assert flight['length_m'] - origin[0] > duration * speed
    assert np.max(bounds[:, 1, 0]) == pytest.approx(flight['length_m'])


def test_boundary_checks_each_sample_including_lateral_and_return_to_flight():
    flight = evaluation_flight(20., .5, .3)
    x = np.array([0., 10., 21., 1., 0., -21., -3.])
    y = np.array([0., 0., 0., 20., 0., 0., 0.])
    assert outside_flight(x, y, flight['origin_x_m'], flight['length_m']).tolist() == [
        False, False, True, True, False, True, False]


def test_upgrade_only_changes_evaluation_preserves_original_and_refuses_budget_change(tmp_path):
    import json
    from stair_resume import upgrade_manifest
    from shared_bo_metrics import digest
    old = dict(collection='original', selection='old', implementation_sha256='original')
    new = dict(old, selection='new', stair_evaluation='timed_continuous_ascent')
    def manifest(proto):
        return dict(protocol=proto, protocol_sha256=digest(proto), plan_sha256='original_plan')
    history = tmp_path/'stair_evaluation_upgrade'
    history.mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(manifest(old)))
    with pytest.raises(ValueError, match='Training'):
        upgrade_manifest(tmp_path, manifest(old), manifest(dict(new,collection='different')))
    upgrade_manifest(tmp_path, manifest(old), manifest(new))
    assert json.loads((history/'original_manifest.json').read_bytes()) == manifest(old)
    assert json.loads((tmp_path/'manifest.json').read_bytes()) == manifest(new)
