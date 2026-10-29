"""Fine-tune one fixed morphology for exactly 200 PPO updates.

Run this with the Isaac Lab Python from the checkpoint-matched source tree.  A
fresh process is started for every BO candidate by ``bo_finetune_controller``.
The base policy weights/normalizers are restored; Adam moments are deliberately
fresh for every candidate, initialized at the learning rate saved in the base
checkpoint unless explicitly overridden.  The base checkpoint is never edited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import traceback

import numpy as np

from bo_finetune_common import (
    DEFAULT_FINETUNE_UPDATES,
    DEFAULT_NUM_ENVS,
    add_source_import_paths,
    apply_fixed_stair_task,
    atomic_json,
    load_and_validate_assets,
    make_candidate_design,
    sha256_file,
)


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--updates", type=int, default=DEFAULT_FINETUNE_UPDATES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Default: base-checkpoint LR (fresh Adam moments).")
    parser.add_argument(
        "--allow-source-matched-asset-hash-override",
        action="store_true",
        help="Permit differing raw checkpoint/bank hashes while retaining semantic and source audits.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    if args.updates != DEFAULT_FINETUNE_UPDATES:
        parser.error("the BO protocol fixes every candidate at exactly 200 PPO updates")
    if args.num_envs != DEFAULT_NUM_ENVS:
        parser.error(f"the BO protocol fixes every candidate at {DEFAULT_NUM_ENVS} environments")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if not args.headless:
        parser.error("BO fine-tuning must run headless")
    return args


def _verify_spawn(raw, design, bank):
    import torch
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    actual_ids = [
        stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        .GetAttribute("spatial:design_id").Get()
        for i in range(raw.num_envs)
    ]
    if actual_ids != [design["id"]] * raw.num_envs:
        raise ValueError("one-design fine-tune scene contains a wrong morphology assignment")
    contract = bank["canonical_semantic_contract"]
    if list(raw.robot.joint_names) != contract["joint_names"]:
        raise ValueError("candidate joint order differs from the checkpoint contract")
    expected_pose = torch.tensor(contract["default_joint_pos"], device=raw.device)
    if not torch.allclose(raw.robot.data.default_joint_pos,
                          expected_pose.expand_as(raw.robot.data.default_joint_pos),
                          atol=1e-6, rtol=0):
        raise ValueError("candidate reset pose differs from the pretrained Rough contract")

    import xml.etree.ElementTree as ET
    xml = ET.parse(design["urdf"]).getroot()
    positions = raw.robot.data.body_pos_w
    geometry = {}
    for joint_name in ("left_knee_joint", "right_knee_joint",
                       "left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        node = xml.find(f"joint[@name='{joint_name}']")
        parent = raw.robot.body_names.index(node.find("parent").get("link"))
        child = raw.robot.body_names.index(node.find("child").get("link"))
        expected = sum(float(v) ** 2 for v in node.find("origin").get("xyz").split()) ** .5
        actual = (positions[:, parent] - positions[:, child]).norm(dim=-1)
        error = float((actual - expected).abs().max().item())
        if error > 2e-4:
            raise ValueError(f"spawned geometry differs from candidate URDF at {joint_name}: {error}m")
        geometry[joint_name] = {
            "expected_m": expected,
            "actual_min_m": float(actual.min().item()),
            "actual_max_m": float(actual.max().item()),
            "max_error_m": error,
        }
    return actual_ids, geometry


def run(args, app) -> int:
    root = add_source_import_paths(args.code_root)
    import torch
    import isaaclab_tasks  # noqa: F401
    import robolab.tasks  # noqa: F401

    checkpoint = Path(args.checkpoint).resolve()
    bank_path = Path(args.bank).resolve()
    request_path = Path(args.request).resolve()
    output = Path(args.output).resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"controller must create the candidate output directory first: {output}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    saved, bank, provenance = load_and_validate_assets(
        checkpoint, bank_path, root,
        allow_hash_override=args.allow_source_matched_asset_hash_override,
    )
    if sha256_file(checkpoint) != request.get("checkpoint_sha256"):
        raise ValueError("base checkpoint changed after the controller created the candidate request")
    if provenance["bank_sha256"] != request.get("bank_sha256"):
        raise ValueError("replay bank changed after the controller created the candidate request")

    from hybrid_importer import install_local_importer_adapter
    from hybrid_runtime import configure_spatial, SpatialRoughEnv, SpatialVecEnv, SpatialRunner
    from hybrid_support import extend_agent, serializable
    from rsl_rl.utils import resolve_callable

    importer_audit = install_local_importer_adapter()
    atomic_json(output / "importer_audit.json", importer_audit)
    design = make_candidate_design(request, bank, output / "design_build")
    atomic_json(output / "design.json", design)

    # Same single-task stair command and terrain for every candidate. The native
    # Rough training noise/events remain enabled; only target command/terrain
    # are pinned so the 200 updates adapt to the BO task being measured.
    options = argparse.Namespace(
        num_envs=args.num_envs,
        device=args.device,
        asset_cache=None,
        allow_cache_misses=True,
        torque_soft_constraint=None,
    )
    cfg, source, native_contract = configure_spatial(options, bank, [design], output)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    apply_fixed_stair_task(cfg, seed=42)
    cfg.capture_terminal_state = True

    env = raw = runner = None
    failed = False
    try:
        torch.set_num_threads(2)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        raw = SpatialRoughEnv(cfg, render_mode=None)
        actual_ids, geometry = _verify_spawn(raw, design, bank)
        symmetry = source["algorithm"].get("symmetry_cfg")
        symmetry_fn = resolve_callable(symmetry["data_augmentation_func"]) if symmetry else None
        env = SpatialVecEnv(raw, symmetry_fn, source.get("clip_actions"))

        agent = extend_agent(source, bank_sha=provenance["bank_sha256"],
                             context_width=bank["context_width"], task="stairs")
        agent["policy"]["class_name"] = "hybrid_policy:SpatialActorCritic"
        if agent["algorithm"].get("symmetry_cfg") is not None:
            agent["algorithm"]["symmetry_cfg"]["data_augmentation_func"] = \
                "hybrid_policy:spatial_design_symmetry"
        base_policy_config = saved["model_state_dict"]["_extra_state"]["config"]
        agent["policy"]["native_config"] = base_policy_config
        agent["seed"] = args.seed
        agent.update(device=cfg.sim.device, logger="tensorboard")

        runner = SpatialRunner(env, agent, log_dir=str(output / "models"), device=cfg.sim.device)
        runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
        # A new, identical Adam state per candidate avoids carrying momentum
        # accumulated across the mixed morphology distribution into one design.
        if runner.alg.optimizer.state:
            raise RuntimeError("fresh fine-tune Adam unexpectedly has non-empty state")
        learning_rate = float(saved["learning_rate"] if args.learning_rate is None
                              else args.learning_rate)
        runner.alg.learning_rate = learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = learning_rate
        runner.current_learning_iteration = 0

        protocol_identity = {
            "result_kind": "rough_candidate_200_update_finetune_v1",
            "base_checkpoint_sha256": provenance["checkpoint_sha256"],
            "bank_sha256": provenance["bank_sha256"],
            "candidate": design,
            "updates": DEFAULT_FINETUNE_UPDATES,
            "optimizer": "fresh_adam",
            "initial_learning_rate": learning_rate,
            "seed": args.seed,
        }
        runner.hybrid_identity = protocol_identity
        runner.hybrid_stage = {
            "stage_index": int(request["trial_number"]),
            "first_update": 0,
            "end_update_exclusive": DEFAULT_FINETUNE_UPDATES,
            "design_id": design["id"],
        }
        atomic_json(output / "geometry_audit.json", geometry)
        atomic_json(output / "fine_tune_plan.json", {
            **provenance,
            "result_kind": "ROUGH_MORPHOLOGY_CANDIDATE_FINETUNE",
            "candidate_request": request,
            "candidate_design": design,
            "candidate_urdf_sha256": design["urdf_sha256"],
            "updates_requested": DEFAULT_FINETUNE_UPDATES,
            "updates_completed_expected": DEFAULT_FINETUNE_UPDATES,
            "environment_count": args.num_envs,
            "training_seed": args.seed,
            "training_task": "fixed 10 cm stairs, fixed 0.5 m/s forward command",
            "training_noise_events": "source Rough settings retained and identical across candidates",
            "optimizer_policy": "fresh Adam moments for every design; initialized at base checkpoint LR",
            "optimizer_state_restored_from_base": False,
            "base_policy_weights_and_normalization_restored": True,
            "source_agent": serializable(source),
            "native_contract": serializable(native_contract),
            "spawned_design_id_count": len(actual_ids),
            "spawned_design_ids": sorted(set(actual_ids)),
        })

        # Seed after simulator/policy construction so every candidate's PPO
        # sampling stream starts from the same point despite different startup.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        runner.learn(DEFAULT_FINETUNE_UPDATES)
        if runner.current_learning_iteration != DEFAULT_FINETUNE_UPDATES - 1:
            raise RuntimeError(
                f"expected 200 updates (final local iter 199); got {runner.current_learning_iteration}"
            )
        fine_checkpoint = output / "models" / f"model_{DEFAULT_FINETUNE_UPDATES - 1}.pt"
        if not fine_checkpoint.is_file() or fine_checkpoint.stat().st_size == 0:
            raise RuntimeError("200-update fine-tune did not produce its final checkpoint")
        candidate_sha = sha256_file(fine_checkpoint)
        final_state = torch.load(fine_checkpoint, map_location="cpu", weights_only=False)
        if final_state.get("iter") != DEFAULT_FINETUNE_UPDATES - 1:
            raise RuntimeError("fine-tuned checkpoint iteration does not prove 200 updates")
        if not final_state.get("optimizer_state_dict", {}).get("state"):
            raise RuntimeError("fine-tune Adam state is empty after 200 PPO updates")
        changed_keys = []
        for name, value in saved["model_state_dict"].items():
            updated = final_state["model_state_dict"].get(name)
            if isinstance(value, torch.Tensor) and isinstance(updated, torch.Tensor):
                if not torch.equal(value.detach().cpu(), updated.detach().cpu()):
                    changed_keys.append(name)
        if not changed_keys:
            raise RuntimeError("policy and normalization state are unchanged after the requested fine-tuning")
        if any(not torch.isfinite(value).all() for value in final_state["model_state_dict"].values()
               if isinstance(value, torch.Tensor)):
            raise RuntimeError("fine-tuned checkpoint contains non-finite policy tensors")
        atomic_json(output / "finetune_complete.json", {
            "result": "ROUGH_CANDIDATE_200_UPDATE_FINETUNE_COMPLETE",
            "trial_number": request["trial_number"],
            "design_id": design["id"],
            "base_checkpoint_sha256": provenance["checkpoint_sha256"],
            "base_bank_sha256": provenance["bank_sha256"],
            "candidate_urdf_sha256": design["urdf_sha256"],
            "checkpoint": str(fine_checkpoint),
            "checkpoint_sha256": candidate_sha,
            "fine_tune_updates": DEFAULT_FINETUNE_UPDATES,
            "final_local_iteration": int(final_state["iter"]),
            "environment_count": args.num_envs,
            "optimizer_mode": "fresh_adam",
            "optimizer_initial_learning_rate": learning_rate,
            "changed_state_tensor_count": len(changed_keys),
            "changed_state_tensor_examples": changed_keys[:12],
            "training_seed": args.seed,
            "requires_separate_gait_evaluation": True,
        })
        print(f"BO_FINETUNE_COMPLETE trial={request['trial_number']} updates=200 "
              f"checkpoint={fine_checkpoint}", flush=True)
        return 0
    except BaseException:
        failed = True
        atomic_json(output / "failure.json", {"traceback": traceback.format_exc()})
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            from hybrid_importer import close_without_masking
            close_without_masking(env, failed)
        elif raw is not None and hasattr(raw, "_is_closed"):
            from hybrid_importer import close_without_masking
            close_without_masking(raw, failed)
        app.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    args.kit_args = (args.kit_args or "") + f" --portable-root {output / 'portable'} --portable"
    application = AppLauncher(args).app
    raise SystemExit(run(args, application))
