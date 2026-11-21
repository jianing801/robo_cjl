from types import SimpleNamespace

import numpy as np
import pytest

from retry_stair_boundary import extended_flight, check_longitudinal_exit
from stair_course import evaluation_flight, flight_mesh, outside_flight


@pytest.mark.parametrize('depth', [.25, .35])
def test_extension_preserves_approach_and_original_tread_heights(depth):
    def surface(flight):
        cfg = SimpleNamespace(size=(flight['length_m'], 40.), step_width=depth,
                              step_height_range=(.075, .075), rear_steps=flight['rear_steps'])
        meshes, origin = flight_mesh(0., cfg)
        bounds = np.asarray([mesh.bounds for mesh in meshes])
        return lambda x: bounds[(bounds[:, 0, 0] < x + origin[0])
                                & (x + origin[0] < bounds[:, 1, 0]), 1, 2].max()
    original = evaluation_flight(20., .6, depth)
    extended = extended_flight(20., .6, depth)
    assert extended['origin_x_m'] - extended['length_m']/2 == pytest.approx(
        original['origin_x_m'] - original['length_m']/2)
    old, new = surface(original), surface(extended)
    for x in [-.9, .9, *[1. + (i + .5)*depth for i in range(original['forward_steps'] - 1)]]:
        assert new(x) == pytest.approx(old(x))
    assert not outside_flight(25., 3.6, extended['origin_x_m'], extended['length_m'])


def test_retry_rejects_lateral_or_missing_boundary_evidence():
    course = dict(origin_x_by_stratum_m=[25.], tile_length_m=50.3)
    strata = {424: 0}
    check_longitudinal_exit({'424': [904., 25., 3.6, 7.8]}, course, strata)
    with pytest.raises(ValueError, match='longitudinal'):
        check_longitudinal_exit({'424': [904., 0., 19.7, 0.]}, course, strata)
    with pytest.raises(ValueError, match='longitudinal'):
        check_longitudinal_exit({'424': [904., 5., 3.6, 0.]}, course, strata)
    with pytest.raises(ValueError, match='longitudinal'):
        check_longitudinal_exit({}, course, strata)
