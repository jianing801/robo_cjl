"""Select a balanced rollout grid from existing common morphology keys."""

from collections import Counter
import json
from pathlib import Path

from itertools import combinations, product
from math import isfinite, prod


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
    """One training and one held-out morphology per pair, disjoint from BO calibration."""
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
    lengths = [Counter(), Counter()]
    for pair in pairs:
        for seen in (True, False):
            choices = sorted(k for k in available - excluded if k[2:] == pair and (k in training) == seen)
            if not choices:
                raise ValueError(f'Missing selection morphology for motor pair {pair}, seen={seen}')
            def priority(key):
                repeat = sum(lengths[i][key[i]] for i in (0, 1))
                separation = min((distance(key, other) for other in panel), default=0.)
                same_pair_distance = 0. if seen else distance(key, panel[-1])
                return -repeat, same_pair_distance, separation
            chosen = max(choices, key=priority)
            panel.append(chosen)
            for i in (0, 1):
                lengths[i][chosen[i]] += 1
    return panel
