"""AMP-only provenance, immutable JSON and unchanged-source PPO adapter."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPE = "standalone_hybrid_amp_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if callable(value):
        return value.__module__+":"+value.__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def source_identity(task="run"):
    """Freeze only shared dependencies used by this independent AMP copy."""
    if task != "run":
        raise ValueError("hybrid training supports AMP only")
    paths = {"robolab/rough_geometry.py", "robolab/physical_snapshot.py",
             "robolab/rough_gpu.py", "robolab/spatial_spawn.py",
             "robolab/assets/robots/roboparty.py",
             "configs/rough_robot_contract.json", "robolab/amp_reference_serialization.py"}
    for folder in ("robolab/tasks/manager_based/amp", "robolab/actuators", "robolab/utils", "../rsl_rl/rsl_rl"):
        paths.update(folder+"/"+p.relative_to(ROOT/folder).as_posix()
                     for p in (ROOT/folder).rglob("*.py"))
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"data/motions/rpo_lab").glob("*.pkl"))
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)}

def interpolate(points, x):
    if x <= points[0][0]:
        return points[0][1]
    for (a, y), (b, z) in zip(points, points[1:]):
        if x <= b:
            return y+(z-y)*(x-a)/(b-a)
    return points[-1][1]


def extend_agent(source, *, bank_sha, context_width, task="run"):
    if task != "run":
        raise ValueError("hybrid training supports AMP only")
    agent = deepcopy(source)
    agent["policy"].update(class_name="amp_policy:SpatialActorCritic",
        native_config=dict(recipe=RECIPE, role=task, bank_sha256=bank_sha, context_width=context_width))
    algorithm = agent["algorithm"]
    for key, expected in (("optimizer", "adam"), ("share_cnn_encoders", False)):
        if key in algorithm:
            if algorithm[key] != expected:
                raise ValueError(f"unsupported source {key}; refusing a semantic change")
            algorithm.pop(key)
    if algorithm.get("symmetry_cfg") is not None:
        algorithm["symmetry_cfg"]["data_augmentation_func"] = "amp_policy:spatial_design_symmetry"
    algorithm["class_name"] = "amp_runtime:HybridPPOAMP"
    return agent
