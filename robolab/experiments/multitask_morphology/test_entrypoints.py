"""Unified layout and real CPU launch checks; no long training or Kit startup."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(path):
    spec = importlib.util.spec_from_file_location('unified_' + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize('task,key,folder', [
    ('rough','stairs','engine'), ('backpack','backpack','engine'), ('amp','run','amp_engine')])
def test_task_routes_to_owned_runtime_and_rejects_wrong_bank(tmp_path, task, key, folder):
    entry = module(HERE/'train.py')
    config = HERE/'configs'/f'{task}.json'
    settings = json.loads(config.read_text(encoding='utf8'))
    grid = settings['grid']
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps({'task_key': key, 'configuration': grid, 'context_width': 28}))
    with pytest.raises(ValueError, match='context schema differ'):
        entry.command(task, config, bank, [])
    schema = settings['context_schema']
    bank.write_text(json.dumps({'task_key': key, 'configuration': grid, 'context_schema': schema}))
    cmd = entry.command(task, config, bank, ['--execute'])
    assert Path(cmd[2]) == HERE/folder/'train.py'
    assert cmd[-1] == '--execute'
    bank.write_text(json.dumps({'task_key': 'wrong', 'configuration': grid}))
    with pytest.raises(ValueError):
        entry.command(task, config, bank, [])
    grid['ankle_pool'].append('DM-J8006-2EC-V1.1')
    grid['ankle_motor_count'] = 6
    bank.write_text(json.dumps({'task_key': key, 'configuration': grid}))
    with pytest.raises(ValueError, match='bank grid differ'):
        entry.command(task, config, bank, [])


def test_owned_source_roots_and_full_motor_curves():
    for folder, name in [('engine','hybrid_support.py'), ('amp_engine','amp_support.py')]:
        assert module(HERE/folder/name).ROOT == ROOT
    torque = module(HERE/'engine/full_body_torque.py')
    hashes = torque.source_identity()
    assert 'experiments/multitask_morphology/engine/full_body_torque.py' in hashes
    assert any(path.endswith('.csv') for path in hashes)
    alignment = module(HERE/'amp_engine/amp_rough_alignment.py')
    assert alignment.ROUGH == HERE/'engine'
    assert module(HERE/'amp_engine/setup_hhtools.py').verify_snapshot() > 500


@pytest.mark.parametrize('entry', ['train.py','prepare.py','prepare_shared.py','distill.py','finetune.py',
    'engine/train.py','amp_engine/train.py','validate_teacher.py'])
def test_real_command_help_without_simulation(entry):
    result = subprocess.run([sys.executable, str(HERE/entry), '--help'], cwd=ROOT,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_teacher_manifest_remains_json_when_native_validator_logs(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    validator = module(HERE/'validate_teacher.py')
    bank = tmp_path/'bank.json'
    (tmp_path/'completed.json').write_text(json.dumps(dict(checkpoint=str(tmp_path/'c.pt'), completed_updates=1)))
    (tmp_path/'plan.json').write_text(json.dumps(dict(bank=str(bank))))
    def validate(*_):
        print('CACHE_URDF_PROGRESS 0 / 2500')
        return dict(configuration=dict(rotation=dict(total_updates=1), full_body_torque=True), experiment={})
    monkeypatch.setitem(sys.modules, 'hybrid_contract', SimpleNamespace(load_plan=validate,
        validate_checkpoint=lambda *_:{'iter':0}, sha=lambda _:'digest'))
    monkeypatch.setitem(sys.modules, 'hybrid_bank', SimpleNamespace(load_configurable_bank=lambda _:
        dict(task_key='stairs', context_width=28, designs=[{}])))
    monkeypatch.setattr(sys, 'argv', ['validate_teacher.py','--task','stairs','--root',str(tmp_path)])
    validator.main()
    output = capsys.readouterr()
    assert json.loads(output.out)['bank'] == str(bank)
    assert 'CACHE_URDF_PROGRESS' in output.err


def test_no_legacy_runtime_paths_or_historical_258_width():
    for folder in ('engine','amp_engine','shared_policy'):
        for path in (HERE/folder).glob('*.py'):
            if path.name.startswith('test_'):
                continue
            ast.parse(path.read_text(encoding='utf8'))
            source = path.read_text(encoding='utf8')
            assert "code/'experiments/rough_hybrid'" not in source
            assert "code/'experiments/amp_hybrid'" not in source
            assert "HERE.parent/'rough_hybrid" not in source
    amp = (HERE/'shared_policy/rollout_amp.py').read_text()
    assert "234+spec['context_width']" in amp and '(n,258)' not in amp


def test_three_task_shared_routing_and_mlp():
    source = (HERE/'shared_policy/pipeline.py').read_text()
    assert "'rollout_amp.py' if task=='run'" in source
    assert "'--architecture','mlp'" in source
    assert "for index,(task,ids) in enumerate(collection_jobs):" in source
    assert "'--max-rows-per-design-shard'" in source
