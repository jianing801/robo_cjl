"""Deterministic payload curriculum; no performance-dependent promotion.

CPU-only planning. Each stage creates a fresh simulator and preserves PPO state.
Payload properties are fixed per environment within each stage, never changed
mid-episode. This is a single-design feasibility pilot, not a multimorph expert.
"""
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stages(total=12000, warmup=1000, expansion=2000, ramps=4,
           low=0.1, warm_high=0.5, high=15.0):
    if any(type(x) is not int or x <= 0 for x in (total, warmup, expansion, ramps)):
        raise ValueError("positive integer update counts required")
    if expansion % ramps or warmup + expansion >= total:
        raise ValueError("expansion must divide into ramps and leave full-range training")
    if not all(math.isfinite(x) for x in (low, warm_high, high)) or not 0 < low <= warm_high < high:
        raise ValueError("invalid payload bounds")
    rows = [("warmup", warmup, warm_high)]
    rows += [(f"expand_{i}", expansion // ramps,
              warm_high + (high - warm_high) * i / ramps) for i in range(1, ramps + 1)]
    rows += [("full_range", total - warmup - expansion, high)]
    result, start = [], 0
    for name, count, upper in rows:
        result.append(dict(name=name, start=start, updates=count, stop=start + count,
                           mass_range_kg=[low, upper]))
        start += count
    return result


def validate_plan(plan):
    if plan.get("kind") != "backpack_fixed_distribution_server_v1":
        raise ValueError("not the server backpack experiment")
    if plan.get("num_envs") != 4096 or plan.get("max_updates") != 12000:
        raise ValueError("authorized server budget is 4096 envs / 12000 updates")
    if plan.get("stages") != stages():
        raise ValueError("schedule differs from the authorized 1000/2000/9000 split")
    for path, digest in plan["source_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"frozen source changed: {path}")
    if sha(plan["design"]["urdf"]) != plan["design"]["urdf_sha256"]:
        raise ValueError("URDF identity changed")
    for path, digest in plan["mesh_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"mesh identity changed: {path}")


def write_new(path, data):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, default=str)


def acquire_controller_lock(root):
    """Nonblocking OS lock; file remains for audit, OS releases on crash."""
    stream = (Path(root) / "controller.lock").open("a+b")
    try:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("another controller owns this experiment")
    return stream


def validate_checkpoint(saved, plan_sha, expected_completed):
    import torch
    if saved.get("infos", {}).get("distribution_plan_sha256") != plan_sha:
        raise ValueError("checkpoint belongs to another experiment")
    if saved.get("iter") != expected_completed - 1:
        raise ValueError("incorrect absolute iteration")
    model, optimizer = saved.get("model_state_dict", {}), saved.get("optimizer_state_dict", {})
    if not model or not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("missing policy or Adam state")
    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(v) for v in value)
        return True
    if not finite((model, optimizer)):
        raise ValueError("nonfinite policy/optimizer")
    return expected_completed
