"""CPU checks for BO capacity configuration; no simulator is launched."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bo_finetune_common import configure_bo_physx, BO_AGGREGATE_PAIRS_CAPACITY


def test_capacity_increases_without_changing_other_physics():
    physx = SimpleNamespace(gpu_total_aggregate_pairs_capacity=1024, gpu_max_rigid_patch_count=327680)
    cfg = SimpleNamespace(sim=SimpleNamespace(physx=physx, dt=.005))
    audit = configure_bo_physx(cfg)
    assert audit["previous_capacity"] == 1024
    assert physx.gpu_total_aggregate_pairs_capacity == BO_AGGREGATE_PAIRS_CAPACITY
    assert physx.gpu_max_rigid_patch_count == 327680 and cfg.sim.dt == .005
    configure_bo_physx(cfg)
    assert physx.gpu_total_aggregate_pairs_capacity == BO_AGGREGATE_PAIRS_CAPACITY


def test_larger_existing_capacity_preserved():
    cfg = SimpleNamespace(sim=SimpleNamespace(physx=SimpleNamespace(gpu_total_aggregate_pairs_capacity=2**25)))
    configure_bo_physx(cfg)
    assert cfg.sim.physx.gpu_total_aggregate_pairs_capacity == 2**25


def test_both_workers_configure_capacity_before_creating_scene():
    import ast
    root = Path(__file__).resolve().parents[1]
    for filename in ("bo_finetune_candidate.py", "bo_finetune_evaluate.py"):
        tree = ast.parse((root/filename).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        config = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "apply_fixed_stair_task")
        scene = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id in ("SpatialRoughEnv", "Capture"))
        assert config.lineno < scene.lineno
    common = ast.parse((root/"bo_finetune_common.py").read_text(encoding="utf-8"))
    task = next(n for n in common.body if isinstance(n, ast.FunctionDef) and n.name == "apply_fixed_stair_task")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "configure_bo_physx" for n in ast.walk(task))
