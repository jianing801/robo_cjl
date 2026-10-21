"""Shared training/evaluation profile selection (no Isaac Sim imports)."""

import json
from pathlib import Path


TASKS = {
    "baseline": "RPO-Flat",
    "adaptive": "RPO-Flat-Backpack-Adaptive",
    "bounded": "RPO-Flat-Backpack-Bounded",
    "compliant": "RPO-Flat-Backpack-Compliant",
}
EXPERIMENTS = {
    "baseline": "rpo_flat_backpack_baseline",
    "adaptive": "rpo_flat_backpack_adaptive",
    "bounded": "rpo_flat_backpack_bounded",
    "compliant": "rpo_flat_backpack_compliant",
}


def checkpoint_profile(checkpoint: str) -> str | None:
    design_path = Path(checkpoint).resolve().parent / "params" / "design.json"
    if not design_path.is_file():
        return None
    with design_path.open(encoding="utf-8") as source:
        design = json.load(source)
    profile = design.get("backpack_profile")
    if profile is None and design.get("task") == "RPO-Flat":
        profile = "baseline"  # Legacy unmodified-flat backpack runs.
    if profile not in TASKS:
        raise ValueError(f"Unknown backpack profile in {design_path}: {profile!r}")
    return profile


def evaluation_profile(checkpoint: str, requested: str = "auto") -> str:
    recorded = checkpoint_profile(checkpoint)
    if requested == "auto":
        if recorded is None:
            raise ValueError("Missing params/design.json; specify --profile baseline or adaptive explicitly")
        return recorded
    if requested not in TASKS:
        raise ValueError(f"Unknown backpack profile: {requested}")
    if recorded is not None and recorded != requested:
        raise ValueError(f"Checkpoint profile is {recorded}, not {requested}; do not evaluate with mismatched rules")
    return requested
