import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rough_launcher_test", ROOT/"scripts/tools/launch_spatial_rough.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_source_budget_and_resource_resolution_independent_of_cwd(tmp_path, monkeypatch):
    assert launcher.original_iterations(ROOT) == 12000
    robot = tmp_path/"data/robots/roboparty/rpo"
    (robot/"urdf").mkdir(parents=True)
    (robot/"meshes").mkdir()
    (robot/"urdf/rpo.urdf").write_text("<robot/>")
    monkeypatch.chdir(tmp_path.parent)
    urdf, meshes = launcher.robot_assets(tmp_path)
    assert urdf == robot/"urdf/rpo.urdf" and meshes == robot/"meshes"
    with pytest.raises(FileNotFoundError, match="Original RPO assets"):
        launcher.robot_assets(tmp_path/"missing")


def fixture_args(tmp_path, monkeypatch):
    import robolab.configurable_morphology as cm
    import robolab.rough_gpu as gpu
    from contextlib import contextmanager
    row = dict(index=3, uuid="GPU-test", total_mib=24564, free_mib=18000, utilization=10)
    monkeypatch.setattr(gpu, "query_gpus", lambda: [row])
    @contextmanager
    def lease(rows, root):
        yield rows[0]
    monkeypatch.setattr(gpu, "gpu_lease", lease)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    config = dict(num_envs=17, length_count=1, thigh_range_m=[.25,.4], calf_range_m=[.3,.45],
                  knee_motor_count=1, ankle_motor_count=1, motor_seed=42)
    cfg = tmp_path/"config.json"; cfg.write_text(json.dumps(config))
    monkeypatch.setattr(cm, "eligible_pools", lambda root: (dict(knee=["k"], ankle=["a"]), {}))
    monkeypatch.setattr(launcher, "robot_assets", lambda **kw: (tmp_path/"rpo.urdf", tmp_path/"meshes"))
    out = tmp_path/"output"
    return ["--multi_morph", "--multi_morph_config", str(cfg), "--multi_morph_root", str(out)], out


def test_one_switch_orders_stages_and_explicit_budget(tmp_path, monkeypatch):
    argv, out = fixture_args(tmp_path, monkeypatch)
    calls = []
    def stage(name, command, output, **kwargs):
        calls.append((name, command))
        if name in ("preflight", "train"):
            assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
            assert command[command.index("--device")+1] == "cuda:0"
            assert "--/renderer/activeGpu=3" in command[command.index("--kit_args")+1]
            folder = output/name; folder.mkdir()
            value = dict(result=("SPATIAL_TWO_UPDATE_PREFLIGHT_PASSED_DISCARD_POLICY" if name == "preflight" else
                                "SPATIAL_FINITE_TRAINING_COMPLETE_NOT_GAIT_VALIDATED"), completed_updates=50)
            (folder/"summary.json").write_text(json.dumps(value))
    monkeypatch.setattr(launcher, "run_stage", stage)
    launcher.main(argv+["--headless", "--num_envs", "32", "--max_iterations", "50"])
    assert [name for name, _ in calls] == ["prepare", "preflight", "train"]
    assert "--preflight-only" in calls[1][1] and "--execute" in calls[2][1]
    assert calls[2][1][calls[2][1].index("--updates")+1] == "50"
    assert json.loads((out/"configuration.json").read_text())["num_envs"] == 32
    assert (out/"completed.json").is_file()


def test_failure_never_launches_next_stage(tmp_path, monkeypatch):
    argv, out = fixture_args(tmp_path, monkeypatch)
    calls = []
    def stage(name, command, output, **kwargs):
        calls.append(name)
        if name == "preflight":
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(launcher, "run_stage", stage)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main(argv)
    assert calls == ["prepare", "preflight"]
    assert not (out/"completed.json").exists()


def test_prepare_only_does_not_use_gpu_or_start_training(tmp_path, monkeypatch):
    argv, out = fixture_args(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(launcher, "run_stage", lambda name, command, output: calls.append(name))
    launcher.main(argv+["--multi_morph_prepare_only"])
    assert calls == ["prepare"]


def test_dispatch_is_opt_in_and_before_isaac_import():
    tree = ast.parse((ROOT/"scripts/rsl_rl/train.py").read_text(encoding="utf8"))
    dispatch = next(n for n in tree.body if isinstance(n, ast.If) and "--multi_morph" in ast.unparse(n.test))
    isaac = next(n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "isaaclab.app")
    assert dispatch.lineno < isaac.lineno
    assert "__main__" in ast.unparse(dispatch.test)


def test_resource_change_after_preflight_stops_train(tmp_path, monkeypatch):
    import robolab.rough_gpu as gpu
    argv, out = fixture_args(tmp_path, monkeypatch)
    inventory = [dict(index=3, uuid="GPU-test", total_mib=24564, free_mib=18000, utilization=10)]
    reads = iter([inventory, [dict(inventory[0], free_mib=100)]])
    monkeypatch.setattr(gpu, "query_gpus", lambda: next(reads))
    calls = []
    def stage(name, command, output, **kwargs):
        calls.append(name)
        if name == "preflight":
            folder = output/name; folder.mkdir()
            (folder/"summary.json").write_text(json.dumps(dict(result="SPATIAL_TWO_UPDATE_PREFLIGHT_PASSED_DISCARD_POLICY")))
    monkeypatch.setattr(launcher, "run_stage", stage)
    with pytest.raises(RuntimeError, match="no longer eligible"):
        launcher.main(argv)
    assert calls == ["prepare", "preflight"]
