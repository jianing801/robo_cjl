"""Shared helpers for adaptation-aware Rough morphology BO.

This module intentionally lives outside the checkpoint-identified training tree.
It does not edit or shadow any source used to produce the pretrained policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


EXPECTED_CHECKPOINT_SHA256 = "a0cd01d59c47588eb23a52b440beac41d626f7b4a85473fb7135743930face26"
EXPECTED_BANK_SHA256 = "25c00be6b64257ef5a049f34d2046895982b253c77495b0dfb0e4652116af2ac"
EXPECTED_CHECKPOINT_ITER = 11999
EXPECTED_COMPLETED_UPDATES = 12000
EXPECTED_CONTEXT_WIDTH = 24
DEFAULT_NUM_ENVS = 4096
DEFAULT_FINETUNE_UPDATES = 200
DEFAULT_TRAINING_LR = 5.0625000000000024e-05
DISABLE_EXTENSION_FS_WATCHER = "--/app/extensions/fsWatcherEnabled=false"
_FS_WATCHER_SETTING = "--/app/extensions/fsWatcherEnabled"


def disable_extension_fs_watcher(kit_args: str | None) -> str:
    """Disable Kit's extension file watcher without overriding an explicit choice.

    The Linux inotify watch limit is easily exhausted by the isaacsim extension
    cache; BO workers never use extension hot reload. Kept local because this
    module is imported before the checkpoint-matched RoboLab tree is on sys.path.
    """
    tokens = (kit_args or "").split()
    if any(token == _FS_WATCHER_SETTING or token.startswith(_FS_WATCHER_SETTING + "=")
           for token in tokens):
        return " ".join(tokens)
    tokens.append(DISABLE_EXTENSION_FS_WATCHER)
    return " ".join(tokens)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_source_import_paths(code_root: str | Path) -> Path:
    """Expose the recovered, checkpoint-matched source tree to this process."""
    root = Path(code_root).resolve()
    tools = root / "scripts" / "tools"
    hybrid = root / "experiments" / "rough_hybrid"
    for item in (root, hybrid, tools, root.parent / "rsl_rl"):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


def load_and_validate_assets(checkpoint_path: str | Path, bank_path: str | Path,
                             code_root: str | Path, *,
                             expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
                             expected_bank_sha256: str = EXPECTED_BANK_SHA256,
                             allow_hash_override: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the audited 12k checkpoint and source-compatible replay bank; fail closed by default."""
    import torch

    root = add_source_import_paths(code_root)
    from hybrid_bank import load_configurable_bank
    from pretrained_rough_source_audit import audit_source_files

    checkpoint = Path(checkpoint_path).resolve()
    bank_file = Path(bank_path).resolve()
    if not checkpoint.is_file() or not bank_file.is_file():
        raise FileNotFoundError("checkpoint and bank must both exist")
    checkpoint_sha = sha256_file(checkpoint)
    bank_sha = sha256_file(bank_file)
    if not allow_hash_override:
        if checkpoint_sha != expected_checkpoint_sha256:
            raise ValueError(f"unexpected base checkpoint hash: {checkpoint_sha}")
        if bank_sha != expected_bank_sha256:
            raise ValueError(f"unexpected BO replay bank hash: {bank_sha}")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("iter") != EXPECTED_CHECKPOINT_ITER:
        raise ValueError(f"expected checkpoint iter {EXPECTED_CHECKPOINT_ITER}; found {saved.get('iter')!r}")
    if not isinstance(saved.get("optimizer_state_dict"), dict):
        raise ValueError("base checkpoint has no PPO optimizer state")
    if not isinstance(saved.get("model_state_dict"), dict):
        raise ValueError("base checkpoint has no policy state")
    if not isinstance(saved.get("learning_rate"), (float, int)) or saved["learning_rate"] <= 0:
        raise ValueError("base checkpoint has no valid learning rate")

    identity = saved.get("hybrid_identity", {})
    grid = identity.get("configuration", {}).get("grid", {})
    rotations = identity.get("configuration", {}).get("rotation", {})
    native_config = saved["model_state_dict"].get("_extra_state", {}).get("config", {})
    if (identity.get("task") != "stairs"
            or rotations.get("total_updates") != EXPECTED_COMPLETED_UPDATES
            or native_config.get("context_width") != EXPECTED_CONTEXT_WIDTH
            or native_config.get("role") != "stairs"):
        raise ValueError("checkpoint is not the expected 12k spatial-context Rough stairs policy")
    if any(not torch.isfinite(value).all() for value in saved["model_state_dict"].values()
           if isinstance(value, torch.Tensor)):
        raise ValueError("base policy contains non-finite tensors")

    runtime_audit = audit_source_files(root, identity)
    if runtime_audit.get("mismatch_count") != 0:
        raise ValueError(f"checkpoint source audit failed ({runtime_audit.get('mismatch_count')} mismatches): "
                         + ", ".join(runtime_audit.get("mismatches", [])))

    bank = load_configurable_bank(bank_file)
    if (bank.get("context_width") != EXPECTED_CONTEXT_WIDTH
            or set(bank.get("knees", [])) != set(grid.get("knee_pool", []))
            or set(bank.get("ankles", [])) != set(grid.get("ankle_pool", []))
            or bank.get("ranges_m", {}).get("thigh") != grid.get("thigh_range_m")
            or bank.get("ranges_m", {}).get("calf") != grid.get("calf_range_m")):
        raise ValueError("BO bank morphology/motor/context scope differs from the checkpoint")
    checkpoint_sources = identity.get("native_sources", {})
    bank_sources = bank.get("training", {}).get("source_hashes", {})
    mismatches = sorted(key for key in set(checkpoint_sources) | set(bank_sources)
                        if checkpoint_sources.get(key) != bank_sources.get(key))
    if mismatches:
        raise ValueError(f"checkpoint and bank native-source hashes differ: {mismatches[:8]}")

    info = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_iter": int(saved["iter"]),
        "completed_updates": EXPECTED_COMPLETED_UPDATES,
        "checkpoint_learning_rate": float(saved["learning_rate"]),
        "bank": str(bank_file),
        "bank_sha256": bank_sha,
        "code_root": str(root),
        "runtime_source_audit": runtime_audit,
        "source_mismatch_count": len(mismatches),
    }
    return saved, bank, info


def make_candidate_design(request: dict[str, Any], bank: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Generate the real candidate URDF and its frozen 24-D morphology context."""
    from hybrid_bank import context
    from robolab.physical_snapshot import generator
    from robolab.rough_geometry import audit_candidate_urdf, file_sha, neutral_foot_compensation

    params = {
        "thigh_m": float(request["thigh_m"]),
        "calf_m": float(request["calf_m"]),
        "knee_motor": str(request["knee_motor"]),
        "ankle_motor": str(request["ankle_motor"]),
    }
    if params["knee_motor"] not in bank["knees"] or params["ankle_motor"] not in bank["ankles"]:
        raise ValueError("candidate motor is outside the pretrained 3x3 motor pool")
    for name in ("thigh", "calf"):
        value = params[f"{name}_m"]
        lo, hi = bank["ranges_m"][name]
        if not lo <= value <= hi:
            raise ValueError(f"candidate {name} length {value} outside [{lo}, {hi}]")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    trial = int(request["trial_number"])
    design_id = f"bo_trial_{trial:05d}"
    pair_id = f"{params['knee_motor']}__{params['ankle_motor']}"
    physical = bank["motor_pairs"][pair_id]
    design = {"id": design_id, "split": "bo_finetune", **params, "pair_id": pair_id}
    template = Path(bank["original_template"])
    mesh_dir = template.parent.parent / "meshes"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"RPO mesh directory missing: {mesh_dir}")
    asset_dir = output / "candidate_asset"
    urdf = Path(generator(physical["snapshot_root"]).generate_urdf(
        params["thigh_m"], params["calf_m"], asset_dir, template,
        mesh_dir=mesh_dir, knee_motor=params["knee_motor"], ankle_motor=params["ankle_motor"]))
    report = urdf.with_suffix(".report.json")
    geometry = audit_candidate_urdf(template, urdf)
    contract = bank["canonical_semantic_contract"]
    placement = neutral_foot_compensation(template, urdf, contract["joint_names"],
                                            contract["default_joint_pos"], max_height_shift_m=.31)
    design.update(
        urdf=str(urdf), urdf_sha256=file_sha(urdf), report=str(report), report_sha256=file_sha(report),
        graph={"joint_names": contract["joint_names"]},
        context=context(design, physical, bank["knees"], bank["ankles"], bank["ranges_m"]),
        root_height_compensation_m=placement["root_height_compensation_m"],
        total_mass_kg=geometry["total_mass_kg"]["candidate"],
        visuals_are_original_meshes_not_parameterized_CAD=True,
    )
    if len(design["context"]) != EXPECTED_CONTEXT_WIDTH:
        raise ValueError(f"candidate context width {len(design['context'])} != {EXPECTED_CONTEXT_WIDTH}")
    return design


def apply_fixed_stair_task(cfg: Any, *, seed: int = 42) -> None:
    """Pin the adaptation/evaluation task to the existing 10 cm stair test."""
    import isaaclab.terrains as terrains
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

    stairs = TerrainGeneratorCfg(
        seed=seed, curriculum=True, size=(40.0, 40.0), border_width=2.0,
        num_rows=1, num_cols=1, horizontal_scale=.05, vertical_scale=.005,
        difficulty_range=(.5, .5), use_cache=False,
        sub_terrains={"stairs": terrains.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(.10, .10), step_width=.25,
            platform_width=2.0, border_width=1.0, holes=False)},
    )
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = stairs
    if hasattr(cfg, "scene_context"):
        cfg.scene_context.terrain_type = "generator"
        cfg.scene_context.terrain_generator = stairs
        cfg.scene_context.max_init_terrain_level = 0
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.commands.rel_standing_envs = 0.0
    cfg.commands.rel_heading_envs = 1.0
    cfg.commands.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.ranges.lin_vel_x = (.5, .5)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.ranges.heading = (0.0, 0.0)


def atomic_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
