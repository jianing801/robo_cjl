"""Exact coverage and checkpoint boundaries for the single-pass Rough baseline."""
from collections import Counter
import json

import pytest

from hybrid_bank import select_grid
from hybrid_contract import HERE, single_pass_order, stage_count, stage_spec


def test_official_single_pass_covers_the_frozen_bank_once():
    configs = HERE.parent / 'configs'
    serial = json.loads((configs / 'rough_single_pass.json').read_text(encoding='utf8'))
    rotating = json.loads((configs / 'rough.json').read_text(encoding='utf8'))
    assert {k: v for k, v in serial.items() if k != 'rotation'} == {
        k: v for k, v in rotating.items() if k != 'rotation'
    }
    grid = serial['grid']
    selected = select_grid(grid, grid['knee_pool'], grid['ankle_pool'])
    bank = dict(configuration=grid, designs=selected['designs'])
    order = single_pass_order(bank, serial)
    stages = [stage_spec(bank, serial, i, order) for i in range(stage_count(bank, serial))]
    counts = Counter(s['end_update'] - s['first_update'] for s in stages)
    assert counts == {5: 2000, 4: 500}
    assert {s['design_ids'][0] for s in stages} == {row['id'] for row in bank['designs']}
    assert stages[0]['first_update'] == 0
    assert stages[1999]['end_update'] == stages[2000]['first_update'] == 10000
    assert all(a['end_update'] == b['first_update'] for a, b in zip(stages, stages[1:]))
    assert stages[-1]['end_update'] == 12000
    with pytest.raises(ValueError, match='absolute total'):
        stage_spec(bank, serial, len(stages), order)
