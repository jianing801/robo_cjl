"""Exact coverage and checkpoint boundaries for the single-pass Rough baseline."""
from collections import Counter
import json
import os
import subprocess
import sys

import pytest

from hybrid_bank import select_grid
from hybrid_contract import HERE, single_pass_order, stage_count, stage_spec, validate_config


@pytest.mark.parametrize('task,task_key', [('rough', 'stairs'), ('backpack', 'backpack')])
def test_official_single_pass_covers_the_frozen_bank_once(task, task_key):
    configs = HERE.parent / 'configs'
    serial = json.loads((configs / f'{task}_single_pass.json').read_text(encoding='utf8'))
    rotating = json.loads((configs / f'{task}.json').read_text(encoding='utf8'))
    assert {k: v for k, v in serial.items() if k != 'rotation'} == {
        k: v for k, v in rotating.items() if k != 'rotation'
    }
    grid = serial['grid']
    selected = select_grid(grid, grid['knee_pool'], grid['ankle_pool'])
    bank = dict(configuration=grid, designs=selected['designs'], task_key=task_key,
                context_schema=serial['context_schema'])
    validate_config(serial, bank)
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


@pytest.mark.parametrize('task', ['backpack', 'amp'])
def test_serial_launcher_previews_existing_assets_without_creating_run(tmp_path, task):
    root = tmp_path
    bank = root / 'banks' / task / 'bank.json'
    bank.parent.mkdir(parents=True)
    bank.write_text('{}', encoding='utf8')
    cache = root / 'asset-cache' / task
    cache.mkdir(parents=True)
    proof = cache / 'completed.json'
    proof.write_text('{}', encoding='utf8')
    from hybrid_contract import sha
    records = root / '启动记录'
    records.mkdir()
    (records / f'{task}_asset_cache.json').write_text(
        json.dumps(dict(cache=str(cache), bank_sha256=sha(bank), completed_sha256=sha(proof))),
        encoding='utf8',
    )
    repo = HERE.parents[3]
    command = [sys.executable, str(repo / 'start.py'), f'{task}-serial', '--root', str(root), '--dry-run']
    library = root / 'references' / 'library.json'
    if task == 'amp':
        result = subprocess.run(command, capture_output=True, encoding='utf8', env=dict(os.environ, PYTHONUTF8='1'))
        assert result.returncode != 0
        assert 'AMP 参考动作' in result.stdout + result.stderr
        library.parent.mkdir()
        library.write_text('{}', encoding='utf8')
    result = subprocess.run(command, capture_output=True, encoding='utf8', env=dict(os.environ, PYTHONUTF8='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'{task}_single_pass.json' in result.stdout
    assert str(bank) in result.stdout and str(cache) in result.stdout
    if task == 'amp':
        assert '--reference-library' in result.stdout and str(library) in result.stdout
    assert not (root / 'experts').exists()
    assert not (records / f'serial_{task}.json').exists()
