import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location("distribution", Path(__file__).resolve().parents[1] /
                                             "robolab/backpack_distribution.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_schedule_budget_and_rehearsal():
    rows = module.stages()
    assert sum(row["updates"] for row in rows) == 12000
    assert rows[0]["updates"] == 1000
    assert rows[-1]["updates"] == 9000
    assert rows[-1]["start"] == 3000
    assert all(row["updates"] == 500 for row in rows[1:-1])
    assert rows[-1]["mass_range_kg"] == [0.1, 15.0]
    assert all(row["mass_range_kg"][0] == 0.1 for row in rows)
    assert all(a["stop"] == b["start"] for a, b in zip(rows, rows[1:]))
    assert [row["mass_range_kg"][1] for row in rows] == sorted(row["mass_range_kg"][1] for row in rows)


@pytest.mark.parametrize("options", [dict(total=500), dict(ramps=3), dict(high=float("nan")), dict(low=0)])
def test_invalid_schedule(options):
    with pytest.raises(ValueError):
        module.stages(**options)


def test_checkpoint_identity_iteration_and_finiteness():
    saved = dict(iter=99, infos=dict(distribution_plan_sha256="abc"),
        model_state_dict=dict(actor=torch.ones(1)),
        optimizer_state_dict=dict(state={0: dict(exp_avg=torch.ones(1))}, param_groups=[dict(lr=1e-4)]))
    assert module.validate_checkpoint(saved, "abc", 100) == 100
    for identity, count in [("wrong", 100), ("abc", 101)]:
        with pytest.raises(ValueError):
            module.validate_checkpoint(saved, identity, count)
    saved["model_state_dict"]["actor"][0] = float("nan")
    with pytest.raises(ValueError):
        module.validate_checkpoint(saved, "abc", 100)


def test_controller_lock_is_exclusive_and_recoverable(tmp_path):
    first = module.acquire_controller_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            module.acquire_controller_lock(tmp_path)
    finally:
        first.close()
    module.acquire_controller_lock(tmp_path).close()
