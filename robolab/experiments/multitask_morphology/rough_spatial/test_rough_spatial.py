"""CPU checks for the isolated Rough spatial experiment."""
import hashlib
import importlib
import json
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
ENTRY = HERE.parent
REPO = ENTRY.parents[2]
sys.path[:0] = [str(ENTRY / "engine"), str(ENTRY), str(REPO / "robolab"), str(REPO)]


def experiment():
    return importlib.import_module("rough_spatial.train_spatial_rough")


@pytest.mark.parametrize("task", ["rough", "backpack"])
def test_plan_covers_all_2500_designs_in_one_stage(task):
    from hybrid_contract import stage_spec
    baseline = json.loads((ENTRY / "configs" / f"{task}.json").read_text())
    config = json.loads(experiment().config_path(task).read_text())
    assert config["grid"] == baseline["grid"]
    assert config["context_schema"] == baseline["context_schema"]
    assert config["full_body_torque"] == baseline["full_body_torque"]
    assert "group_schedule" not in config
    assert config["rotation"]["total_updates"] == baseline["rotation"]["total_updates"] == 12000
    rows = [dict(id=str(i), split="train", thigh_m=.3, calf_m=.35,
                 knee_motor="k", ankle_motor="a") for i in range(2500)]
    bank = dict(designs=rows, ranges_m={"thigh": [.2, .4], "calf": [.25, .45]})
    stage = stage_spec(bank, config, 0)
    assert len(set(stage["design_ids"])) == 2500
    assert (stage["first_update"], stage["end_update"]) == (0, 12000)
    with pytest.raises(ValueError, match="absolute total budget"):
        stage_spec(bank, config, 1)


@pytest.mark.parametrize("task,task_key", [("rough", "stairs"), ("backpack", "backpack")])
def test_spatial_entry_reuses_task_assets_and_stays_out_of_baseline_selection(tmp_path, monkeypatch, task, task_key):
    import start
    config = json.loads((ENTRY / "configs" / f"{task}.json").read_text())
    bank = tmp_path / "banks" / task / "bank.json"
    bank.parent.mkdir(parents=True)
    bank.write_text(json.dumps(dict(
        task_key=task_key, configuration=config["grid"], context_schema=config["context_schema"],
        designs=[dict(id=str(i), split="train") for i in range(2500)])), encoding="utf-8")
    cache = tmp_path / "assets" / task
    cache.mkdir(parents=True)
    proof = cache / "completed.json"
    proof.write_text("{}", encoding="utf-8")
    record = tmp_path / "\u542f\u52a8\u8bb0\u5f55" / f"{task}_asset_cache.json"
    record.parent.mkdir()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record.write_text(json.dumps(dict(cache=str(cache), bank_sha256=sha(bank),
                                      completed_sha256=sha(proof))), encoding="utf-8")
    seen = []
    monkeypatch.setattr(start, "run", lambda argv, *, preview: seen.extend(map(str, argv)))
    start.main([f"{task}-spatial", "--root", str(tmp_path), "--gpu", "2", "--dry-run"])
    assert seen[seen.index("--asset-cache") + 1] == str(cache)
    assert seen[seen.index("--device") + 1] == "cuda:2"
    assert "--reference-library" not in seen
    if task == "backpack":
        assert seen[seen.index("--task") + 1] == "backpack"
    assert Path(seen[seen.index("--output-root") + 1]).name.startswith(f"spatial_{task}_all2500_")
    out = Path(seen[seen.index("--output-root") + 1])
    out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(dict(bank=str(bank))), encoding="utf-8")
    assert start.select_run(tmp_path, task) == (tmp_path / "experts" / task, False)
    record.unlink()
    with pytest.raises(FileNotFoundError):
        start.main([f"{task}-spatial", "--root", str(tmp_path), "--dry-run"])


def test_worker_preserves_original_process_identity():
    module = experiment()
    native = str(module.ENGINE / "persistent_worker.py")
    command = [sys.executable, "-u", native, "--bank", "bank.json"]
    assert module.wrap_worker_command(command) == [
        sys.executable, "-u", str(Path(module.__file__)), "--worker", native, "--bank", "bank.json"]
    assert module.wrap_worker_command(command, "backpack") == [
        sys.executable, "-u", str(Path(module.__file__)), "--worker", "--task", "backpack",
        native, "--bank", "bank.json"]


def test_runtime_uses_native_timer_aggregate_logging_and_vectorized_batches():
    module = experiment()
    events = []
    obs = {"policy": torch.ones(2, 3)}
    rewards, dones = torch.arange(2), torch.zeros(2)
    health = {"Health/knee": torch.tensor(1.)}

    class Base:
        def step(self, actions):
            return obs, rewards, dones, {"log": {"Episode/reward": 2}}

    env = SimpleNamespace(step=lambda actions: None, _extend=lambda x: x,
                          unwrapped=SimpleNamespace(_motor_health_log=health))
    native_sampler = object()
    runtime = SimpleNamespace(balanced_indices=native_sampler, RslRlVecEnvWrapper=Base)
    runner = SimpleNamespace(env=env, current_learning_iteration=0, learn=lambda remaining, **kw: events.append((remaining, kw)))

    with ExitStack() as stack:
        module.install_runtime(stack, runner, runtime)
        runner.learn(12000)
        runner.current_learning_iteration = 1000
        runner.learn(500)
        assert events == [(12000, {"init_at_random_ep_len": True}), (500, {"init_at_random_ep_len": False})]
        got_obs, got_rewards, got_dones, extras = env.step(None)
        assert got_obs is obs and got_rewards is rewards and got_dones is dones
        assert extras["log"] == {"Episode/reward": 2, **health}
        groups = torch.arange(4096) % 2500
        batches = runtime.balanced_indices(groups, 24, 4)
        assert all(x.numel() == 24576 for x in batches)
        assert torch.equal(torch.cat(batches).sort().values, torch.arange(98304))
        expected = torch.bincount(groups) * 6
        assert all(torch.equal(torch.bincount(groups[x % 4096]), expected) for x in batches)
    assert runtime.balanced_indices is native_sampler
