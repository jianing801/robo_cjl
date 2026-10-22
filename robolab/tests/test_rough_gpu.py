from types import SimpleNamespace
import pytest
from robolab.rough_gpu import candidates, visible_gpus, child_environment, gpu_lease, reject_duplicate_bank


def inventory():
    return [dict(index=i, uuid=f"GPU-test-{i}", total_mib=24564, free_mib=free, utilization=util)
            for i, free, util in [(0, 14000, 90), (1, 5000, 0), (2, 16000, 24), (3, 18000, 10)]]


def test_pick_most_free_eligible_not_merely_no_process():
    rows = candidates(inventory(), {}, min_free_mib=12288, max_utilization=30)
    assert [r["index"] for r in rows] == [3, 2]
    assert not candidates(inventory(), {}, min_free_mib=20000, max_utilization=30)


def test_respects_visibility_and_explicit_logical_index():
    rows = inventory()
    assert [r["index"] for r in visible_gpus(rows, {"CUDA_VISIBLE_DEVICES": "2,0"})] == [2, 0]
    chosen = candidates(rows, {"CUDA_VISIBLE_DEVICES": "2,3"}, min_free_mib=12288, max_utilization=30, requested="cuda:0")
    assert chosen[0]["index"] == 2
    assert visible_gpus(rows, {"CUDA_VISIBLE_DEVICES": ""}) == []
    assert visible_gpus(rows, {"CUDA_VISIBLE_DEVICES": "-1"}) == []
    assert visible_gpus(rows, {"NVIDIA_VISIBLE_DEVICES": "none"}) == []
    with pytest.raises(ValueError, match="unambiguously"):
        visible_gpus(rows, {"CUDA_VISIBLE_DEVICES": "GPU-test"})
    assert visible_gpus(rows, {"CUDA_VISIBLE_DEVICES": "GPU-test-3"})[0]["index"] == 3


def test_uuid_child_environment_does_not_mutate_parent():
    parent = dict(CUDA_VISIBLE_DEVICES="2,3", OTHER="keep")
    child = child_environment(inventory()[3], parent)
    assert child["CUDA_VISIBLE_DEVICES"] == "GPU-test-3" and child["OTHER"] == "keep"
    assert parent["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_cooperative_lease_falls_back_and_releases_on_error(tmp_path):
    rows = inventory()[2:]
    with gpu_lease(rows, tmp_path) as first:
        assert first["index"] == 2
        with gpu_lease(rows, tmp_path) as second:
            assert second["index"] == 3
            with pytest.raises(RuntimeError, match="No GPU"):
                with gpu_lease(rows, tmp_path):
                    pass
    with pytest.raises(ValueError):
        with gpu_lease(rows, tmp_path):
            raise ValueError("interrupted")
    with gpu_lease(rows, tmp_path) as first:
        assert first["index"] == 2


def test_unrelated_python_allowed_same_bank_rejected(tmp_path):
    bank = tmp_path/"bank.json"
    unrelated = SimpleNamespace(pid=5, info=dict(cmdline=["python", "jupyter.py"], cwd=str(tmp_path)))
    another = SimpleNamespace(pid=6, info=dict(cmdline=["python", "train_spatial_rough.py", "--bank", "other.json"], cwd=str(tmp_path)))
    duplicate = SimpleNamespace(pid=7, info=dict(cmdline=["python", "train_spatial_rough.py", "--bank=bank.json"], cwd=str(tmp_path)))
    reject_duplicate_bank(bank, processes=[unrelated, another], own_tree={1})
    with pytest.raises(RuntimeError, match="same Rough bank"):
        reject_duplicate_bank(bank, processes=[duplicate], own_tree={1})
    reject_duplicate_bank(bank, processes=[duplicate], own_tree={7})
