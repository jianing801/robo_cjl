from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

KNEES = ("RS04", "DM-J10010L-2EC", "DM-J10010-2EC", "DM-J6248P-2EC", "DM-J10422P-2EC")
ANKLES = ("RS06", "DM-J4340P-2EC", "DM-J8009P-2EC-V1.1", "DM-J4340P-2EC-V1.1-48V", "DM-J6216P-2EC")
THIGHS = tuple(round(.20 + .20 * i / 9, 10) for i in range(10))
CALVES = tuple(round(.25 + .20 * i / 9, 10) for i in range(10))


def grid(thighs=THIGHS, calves=CALVES, knees=KNEES, ankles=ANKLES):
    return list(product(thighs, calves, knees, ankles))


def test_current_grid_uses_endpoints_and_exact_thirds():
    from collection_grid import select_collection_grid

    keys = grid()
    selected = select_collection_grid(list(key) for key in reversed(keys))
    expected = sorted(product(THIGHS[::3], CALVES[::3], KNEES, ANKLES))
    assert selected == expected
    assert len(selected) == 400
    assert set(selected) <= set(keys)
    assert select_collection_grid(keys) == selected


def test_default_selection_covers_all_100_leg_geometries_across_the_panel():
    from collection_grid import calibration_keys, select_collection_grid, selection_panel

    keys = grid()
    panel = selection_panel(keys, select_collection_grid(keys), calibration_keys())
    assert {row[:2] for row in panel} == set(product(THIGHS, CALVES))


def test_nonuniform_axes_use_nearest_existing_levels_with_stable_ties():
    from collection_grid import select_collection_grid

    thighs = (1., 2.5, 3.5, 4.5, 5.5, 7.)
    calves = (1., 2.3, 3.1, 5.2, 7.)
    selected = select_collection_grid(grid(thighs, calves))
    assert {key[0] for key in selected} == {1., 2.5, 4.5, 7.}
    assert {key[1] for key in selected} == {1., 3.1, 5.2, 7.}


@pytest.mark.parametrize("bad_keys", [
    [],
    grid()[1:],
    [*grid(), grid()[0]],
    grid(thighs=THIGHS[:3]),
    grid(calves=CALVES[:3]),
    grid(knees=KNEES[:4]),
    grid(ankles=ANKLES[:4]),
    [(float("nan"), *grid()[0][1:]), *grid()[1:]],
    [(THIGHS[0], float("inf"), KNEES[0], ANKLES[0]), *grid()[1:]],
    [(True, *grid()[0][1:]), *grid()[1:]],
    [(*grid()[0][:2], "", ANKLES[0]), *grid()[1:]],
    [grid()[0][:3], *grid()[1:]],
])
def test_rejects_incomplete_or_invalid_grids(bad_keys):
    from collection_grid import select_collection_grid

    with pytest.raises(ValueError):
        select_collection_grid(bad_keys)
