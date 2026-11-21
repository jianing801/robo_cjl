"""Select a balanced rollout grid from existing common morphology keys."""

from collections import Counter
import json
from pathlib import Path

from itertools import combinations, permutations, product
from math import isfinite, prod


SELECTION_VERSION = 'last_epoch_dagger_paired_200'
DESIGNS_PER_PAIR_PER_SPLIT = 4


def _four_levels(levels):
    low, high = levels[0], levels[-1]
    third = (high - low) / 3
    first_target, second_target = low + third, low + 2 * third
    first, second = min(
        combinations(levels[1:-1], 2),
        key=lambda pair: (abs(pair[0] - first_target) + abs(pair[1] - second_target), pair),
    )
    return low, first, second, high


def select_collection_grid(keys):
    """Return the sorted 4×4×5×5 subset of a complete existing Cartesian grid."""
    available = set()
    for key in keys:
        if not isinstance(key, (tuple, list)) or len(key) != 4:
            raise ValueError("Expected a structure key with two lengths and two motor IDs")
        thigh, calf, knee, ankle = key
        if not all(type(value) in (int, float) and isfinite(value) and value > 0 for value in (thigh, calf)):
            raise ValueError("Grid lengths must be positive finite numbers")
        if not all(isinstance(motor, str) and motor.strip() for motor in (knee, ankle)):
            raise ValueError("Grid motors must have nonempty IDs")
        key = (thigh, calf, knee, ankle)
        if key in available:
            raise ValueError(f"Duplicate structure key: {key}")
        available.add(key)

    if not available:
        raise ValueError("Empty common structure grid")
    thighs, calves, knees, ankles = [sorted(set(axis)) for axis in zip(*available)]
    if len(thighs) < 4 or len(calves) < 4 or len(knees) != 5 or len(ankles) != 5:
        raise ValueError("Expected at least four levels per length axis and five motors per joint")
    if len(available) != prod(map(len, (thighs, calves, knees, ankles))):
        raise ValueError("Common structure grid has missing Cartesian combinations")
    return list(product(_four_levels(thighs), _four_levels(calves), knees, ankles))


def calibration_keys():
    """Physical identities, in the original fixed group-21 scoring order."""
    source = Path(__file__).resolve().parents[4] / 'docs/reference/rough_amp_expert_20261121/rough/selection.json'
    rows = json.loads(source.read_text(encoding='utf8'))['structures']
    return [(round(row['thigh_m'], 10), round(row['calf_m'], 10), row['knee_motor'], row['ankle_motor'])
            for row in rows]


def selection_panel(keys, collection, reserved):
    """Select four training and four held-out leg geometries per knee/ankle pair."""
    available = set(map(tuple, keys))
    training = set(map(tuple, collection))
    excluded = set(map(tuple, reserved))
    pairs = sorted({key[2:] for key in available})
    if (len(pairs) != 25 or len(reserved) != 50 or len(excluded) != 50
            or not (excluded | training) <= available
            or Counter(key[2:] for key in excluded) != {pair: 2 for pair in pairs}):
        raise ValueError('The fixed 50 calibration structures must cover all 25 native motor pairs twice')
    spans = [max(k[i] for k in available) - min(k[i] for k in available) for i in (0, 1)]
    def distance(a, b):
        return sum(((a[i] - b[i]) / spans[i]) ** 2 for i in (0, 1))

    panel = []
    eligible = available - excluded
    held_out_levels = (Counter(), Counter())
    geometry_counts = Counter()
    for pair in pairs:
        pair_keys = {k for k in eligible if k[2:] == pair}
        seen = pair_keys & training
        thighs = sorted({k[0] for k in seen})
        calves = sorted({k[1] for k in seen})
        if len(thighs) != DESIGNS_PER_PAIR_PER_SPLIT or len(calves) != DESIGNS_PER_PAIR_PER_SPLIT:
            raise ValueError(f'Need four distinct training lengths on each axis for motor pair {pair}')
        # Only 4! assignments. Check all of them so that reserved BO points
        # cannot force a repeated length on the fourth draw.
        assignments = [[(t, c, *pair) for t, c in zip(thighs, order)] for order in permutations(calves)]
        valid = [rows for rows in assignments if all(row in seen for row in rows)]
        if not valid:
            raise ValueError(f'No four-way training length match for motor pair {pair}')

        def training_priority(rows):
            repeat = sum(geometry_counts[row[:2]] for row in rows)
            separation = min(distance(a, b) for a, b in combinations(rows, 2))
            return -repeat, separation

        pair_panel = max(valid, key=training_priority)
        geometry_counts.update(row[:2] for row in pair_panel)
        choices = sorted(pair_keys - training)
        if len(choices) < DESIGNS_PER_PAIR_PER_SPLIT:
            raise ValueError(f'Need four held-out morphologies for motor pair {pair}')
        local_levels = (set(), set())

        def held_out_priority(key):
            new_levels = sum(key[i] not in local_levels[i] for i in (0, 1))
            repeat = sum(held_out_levels[i][key[i]] for i in (0, 1))
            separation = min(distance(key, other) for other in pair_panel)
            return new_levels, -geometry_counts[key[:2]], -repeat, separation

        for _ in range(DESIGNS_PER_PAIR_PER_SPLIT):
            chosen = max(choices, key=held_out_priority)
            choices.remove(chosen)
            pair_panel.append(chosen)
            geometry_counts[chosen[:2]] += 1
            for i in (0, 1):
                local_levels[i].add(chosen[i])
                held_out_levels[i][chosen[i]] += 1
        panel.extend(pair_panel)
    return panel
