"""Evaluate one independently fine-tuned Rough candidate on fixed stairs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

from bo_finetune_common import (
    DEFAULT_FINETUNE_UPDATES,
    add_source_import_paths,
    apply_fixed_stair_task,
    atomic_json,
    load_and_validate_assets,
    sha256_file,
)


COMMAND_MPS = 0.5
HORIZON_S = 20.0
EVAL_SEED = 42


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="The candidate's model_199.pt")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--candidate-dir", required=True, help="Successful 200-update fine-tune directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument(
        "--allow-source-matched-asset-hash-override",
        action="store_true",
        help="Permit differing raw checkpoint/bank hashes while retaining semantic and source audits.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    if not args.headless:
        parser.error("BO evaluations must run headless")
    return args


def main(args, app) -> int:
    root = add_source_import_paths(args.code_root)
    import torch
    import isaaclab_tasks  # noqa: F401
    import robolab.tasks  # noqa: F401

    fine_checkpoint = Path(args.checkpoint).resolve()
    base_checkpoint = Path(args.base_checkpoint).resolve()
    bank_path = Path(args.bank).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    saved_base, bank, provenance = load_and_validate_assets(
        base_checkpoint, bank_path, root,
        allow_hash_override=args.allow_source_matched_asset_hash_override,
    )
    fine_meta_path = candidate_dir / "finetune_complete.json"
    design_path = candidate_dir / "design.json"
    if not fine_meta_path.is_file() or not design_path.is_file():
        raise ValueError("candidate has no successful fine-tune manifest/design")
    fine_meta = json.loads(fine_meta_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if fine_meta.get("result") != "ROUGH_CANDIDATE_200_UPDATE_FINETUNE_COMPLETE":
        raise ValueError("candidate did not complete the fixed 200-update adaptation")
    if fine_meta.get("fine_tune_updates") != DEFAULT_FINETUNE_UPDATES:
        raise ValueError("fine-tune update count differs from the BO objective")
    if fine_meta.get("base_checkpoint_sha256") != provenance["checkpoint_sha256"]:
        raise ValueError("candidate was not initialized from the specified base model")
    if fine_meta.get("base_bank_sha256") != provenance["bank_sha256"]:
        raise ValueError("candidate was fine-tuned with a different replay bank")
    if sha256_file(fine_checkpoint) != fine_meta.get("checkpoint_sha256"):
        raise ValueError("fine-tuned model hash differs from its completion record")
    if not Path(design["urdf"]).is_file() or sha256_file(design["urdf"]) != design["urdf_sha256"]:
        raise ValueError("candidate URDF is missing or changed after fine-tuning")

    from pretrained_rough_source_audit import audit_loaded_modules, audit_source_files
    from hybrid_bank import load_configurable_bank
    from hybrid_importer import install_local_importer_adapter
    from hybrid_runtime import configure_spatial, SpatialRoughEnv, SpatialVecEnv
    from hybrid_policy import SpatialActorCritic
    from robolab.rough_geometry import file_sha
    from robolab.physical_snapshot import generator as _physical_snapshot_generator  # noqa: F401

    bank = load_configurable_bank(bank_path)
    runtime_audit = audit_source_files(root, saved_base["hybrid_identity"])
    if runtime_audit["mismatch_count"]:
        raise ValueError("checkpoint source audit failed during evaluation")
    fine = torch.load(fine_checkpoint, map_location="cpu", weights_only=False)
    if fine.get("iter") != DEFAULT_FINETUNE_UPDATES - 1:
        raise ValueError("candidate checkpoint does not contain exactly 200 update iterations")
    base_extra = saved_base["model_state_dict"]["_extra_state"]["config"]
    fine_extra = fine["model_state_dict"]["_extra_state"]["config"]
    if fine_extra != base_extra:
        raise ValueError("fine-tune checkpoint changed the pretrained policy's context schema")

    candidate_request = {
        "trial_number": fine_meta["trial_number"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        **{key: design[key] for key in ("thigh_m", "calf_m", "knee_motor", "ankle_motor")},
    }
    options = argparse.Namespace(
        num_envs=1, device=args.device, asset_cache=None,
        allow_cache_misses=True, torque_soft_constraint=None,
    )
    cfg, source, _contract = configure_spatial(options, bank, [design], output)
    apply_fixed_stair_task(cfg, seed=args.seed)
    cfg.noise.add_noise = False
    cfg.capture_terminal_state = True
    for name, event in list(vars(cfg.events).items()):
        if getattr(event, "mode", None) in ("startup", "interval"):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {
        key: (0.0, 0.0) for key in cfg.events.reset_base.params["pose_range"]
    }
    cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in cfg.events.reset_base.params["velocity_range"]
    }
    cfg.events.reset_robot_joints.params.update(position_range=(0.0, 0.0), velocity_range=(0.0, 0.0))
    for motor in cfg.scene.robot.actuators.values():
        motor.min_delay = 0
        motor.max_delay = 0
    cfg.sim.device = args.device
    install_audit = install_local_importer_adapter()
    atomic_json(output / "importer_audit.json", install_audit)

    class Capture(SpatialRoughEnv):
        suppress_resets = False

        def update_terrain_levels(self, env_ids):
            return {}

        def _reset_idx(self, env_ids):
            if not self.suppress_resets:
                super()._reset_idx(env_ids)

    raw = env = None
    failed = False
    try:
        torch.set_num_threads(2)
        cfg.seed = args.seed
        raw = Capture(cfg, render_mode=None)
        env = SpatialVecEnv(raw, None, source.get("clip_actions"))
        raw.suppress_resets = True

        import omni.usd
        stage = omni.usd.get_context().get_stage()
        actual_id = stage.GetPrimAtPath("/World/envs/env_0/Robot").GetAttribute("spatial:design_id").Get()
        if actual_id != design["id"]:
            raise ValueError(f"spawned morphology mismatch: {actual_id!r} != {design['id']!r}")
        if list(raw.robot.joint_names) != bank["canonical_semantic_contract"]["joint_names"]:
            raise ValueError("evaluation joint order differs from the pretrained policy contract")

        import xml.etree.ElementTree as ET
        xml = ET.parse(design["urdf"]).getroot()
        geometry = []
        for joint_name in ("left_knee_joint", "right_knee_joint",
                           "left_ankle_pitch_joint", "right_ankle_pitch_joint"):
            node = xml.find(f"joint[@name='{joint_name}']")
            parent = raw.robot.body_names.index(node.find("parent").get("link"))
            child = raw.robot.body_names.index(node.find("child").get("link"))
            expected = sum(float(v) ** 2 for v in node.find("origin").get("xyz").split()) ** .5
            actual = float((raw.robot.data.body_pos_w[0, parent] -
                            raw.robot.data.body_pos_w[0, child]).norm())
            error = abs(actual - expected)
            if error > 2e-4:
                raise ValueError(f"evaluation geometry mismatch at {joint_name}: {error}m")
            geometry.append({"joint": joint_name, "expected_m": expected,
                             "actual_m": actual, "error_m": error})
        atomic_json(output / "geometry_audit.json", geometry)

        policy_cfg = dict(source["policy"])
        policy_cfg.pop("class_name", None)
        policy_cfg["native_config"] = base_extra
        obs = env.get_observations()
        policy_width = int(saved_base["model_state_dict"]["actor.0.weight"].shape[1])
        if int(obs["policy"].shape[-1]) != policy_width:
            raise ValueError("evaluation policy observation width differs from the base checkpoint")
        policy = SpatialActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]},
                                    raw.num_actions, **policy_cfg).to(raw.device)
        policy.load_state_dict(fine["model_state_dict"], strict=True)
        policy.eval()

        max_steps = round(HORIZON_S / raw.step_dt)
        actual_abs_error_sum = 0.0
        actual_steps = 0
        elapsed_s = 0.0
        episode_return = 0.0
        max_action = 0.0
        fell = False
        timed_out = False
        clip_sum = 0.0
        obs = env.get_observations()
        for _step in range(max_steps):
            with torch.inference_mode():
                actions = policy.act_inference(obs)
            if not torch.isfinite(actions).all():
                raise ValueError("fine-tuned policy produced a non-finite action")
            max_action = max(max_action, float(actions.abs().max()))
            obs, rewards, dones, _extras = env.step(actions)
            episode_return += float(rewards[0])
            velocity = float(raw.robot.data.root_lin_vel_b[0, 0])
            actual_abs_error_sum += abs(velocity - COMMAND_MPS)
            actual_steps += 1
            elapsed_s += float(raw.step_dt)
            actuator = raw.robot.actuators["feet"]
            clip_sum += float((actuator.computed_effort[0] - actuator.applied_effort[0])
                              .abs().gt(1e-4).float().mean())
            tilt = torch.acos((-raw.robot.data.projected_gravity_b[0, 2]).clamp(-1.0, 1.0)) > torch.pi / 3
            fell = bool(raw.reset_terminated[0]) or bool(tilt)
            timed_out = bool(raw.reset_time_outs[0])
            if fell or timed_out or bool(dones[0]):
                break

        # Fixed-horizon metric: after an early fall, count the unobserved tail
        # as standing still (vx=0), so early termination cannot improve MAE by
        # shortening the measurement window.
        remaining_steps = max(0, max_steps - actual_steps)
        total_abs_error = actual_abs_error_sum + remaining_steps * abs(COMMAND_MPS) \
            * float(raw.step_dt)
        tracking_mae = total_abs_error / HORIZON_S
        observed_mae = actual_abs_error_sum / max(actual_steps, 1)
        feasible = not fell and elapsed_s >= HORIZON_S - float(raw.step_dt) - 1e-6
        module_audit = audit_loaded_modules(root, saved_base["hybrid_identity"], [
            "hybrid_bank", "hybrid_importer", "hybrid_policy", "hybrid_runtime",
            "robolab.physical_snapshot", "robolab.rough_geometry",
        ])
        if module_audit["mismatch_count"]:
            raise ValueError("loaded evaluation modules do not match the pretrained source hashes")
        summary = {
            "result": "ROUGH_CANDIDATE_200_UPDATE_SPEED_TRACKING_EVALUATED",
            "trial_number": int(fine_meta["trial_number"]),
            "params": {key: design[key] for key in
                       ("thigh_m", "calf_m", "knee_motor", "ankle_motor")},
            "base_checkpoint_sha256": provenance["checkpoint_sha256"],
            "base_checkpoint_iter": provenance["checkpoint_iter"],
            "fine_tuned_checkpoint": str(fine_checkpoint),
            "fine_tuned_checkpoint_sha256": sha256_file(fine_checkpoint),
            "fine_tune_updates": DEFAULT_FINETUNE_UPDATES,
            "fine_tune_optimizer_mode": "fresh_adam",
            "fine_tune_initial_learning_rate": fine_meta["optimizer_initial_learning_rate"],
            "candidate_urdf_sha256": design["urdf_sha256"],
            "bank": str(bank_path),
            "bank_sha256": provenance["bank_sha256"],
            "runtime_source_audit": runtime_audit,
            "loaded_module_audit": module_audit,
            "task": "RPO-Rough fixed 10cm stairs",
            "command_mps": COMMAND_MPS,
            "horizon_s": HORIZON_S,
            "eval_seed": args.seed,
            "num_envs": 1,
            "domain_randomization": False,
            "observation_noise": False,
            "initial_perturbation": False,
            "motor_delay": 0,
            "elapsed_s": elapsed_s,
            "fell": fell,
            "timed_out": timed_out,
            "feasible": feasible,
            "tracking_mae_observed_mps": observed_mae,
            "tracking_mae_fixed_horizon_mps": tracking_mae,
            "objective_value": tracking_mae,
            "objective_direction": "minimize",
            "failure_tail_rule": "after early fall, remaining horizon contributes |0 m/s - command|",
            "episode_return_diagnostic": episode_return,
            "ankle_clipping_fraction": clip_sum / max(actual_steps, 1),
            "max_abs_action": max_action,
            "asset_identity_boundary":
                "URDF generated locally from the checkpoint-matched motor SW-GP snapshot; source identity audited, server URDF bytes not claimed",
        }
        atomic_json(output / "summary.json", summary)
        print(f"BO_FINETUNE_EVAL_COMPLETE trial={fine_meta['trial_number']} "
              f"mae={tracking_mae:.6f} fell={fell} feasible={feasible}", flush=True)
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
        raise FileExistsError(f"evaluation output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    args.kit_args = (args.kit_args or "") + f" --portable-root {output / 'portable'} --portable"
    application = AppLauncher(args).app
    raise SystemExit(main(args, application))
