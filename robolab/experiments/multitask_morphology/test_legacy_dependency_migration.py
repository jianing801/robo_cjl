"""Protect the remaining compatibility consumers after old expert directory removal."""
import ast
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def test_legacy_torque_bytes_and_tool_resolution():
    moved = REPO / 'robolab/scripts/tools/full_body_torque.py'
    archived = REPO / 'legacy/expert_sources_20261114/robolab/experiments/rough_hybrid/full_body_torque.py'
    assert hashlib.sha256(moved.read_bytes()).digest() == hashlib.sha256(archived.read_bytes()).digest()
    for name in ('train_backpack_distribution_stage.py', 'run_backpack_distribution_pilot.py'):
        text = (moved.parent / name).read_text(encoding='utf8')
        ast.parse(text)
        assert 'experiments/rough_hybrid' not in text
        assert 'from full_body_torque import' in text
    # Its unchanged relative asset lookup still resolves to the real catalog.
    assert (moved.parents[2] / 'robolab/assets/rpo_motor_catalog.py').is_file()


def test_rebind_command_does_not_upgrade_dependencies(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(HERE / 'amp_engine'))
    spec = importlib.util.spec_from_file_location('rebind_hhtools', HERE / 'amp_engine/rebind_hhtools.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.command()
    assert args[:3] == [sys.executable, '-m', 'pip']
    assert '--no-deps' in args and '--no-build-isolation' in args
    assert '--upgrade' not in args
    assert args[-1] == str(HERE / 'amp_engine/vendor/hhtools')
    assert module.verify_snapshot() > 0
