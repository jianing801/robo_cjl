"""One authorized single-task, single-physical-design block of a shared expert.

Use run_multimorph_experts.py for the finite curriculum. Import/help never train.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "rsl_rl")]


def main():
    from isaaclab.app import AppLauncher
    from robolab.morph_experiments import load_bank, write_new, execution_budget
    from robolab.mxt_design_transfer import file_sha
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--task", choices=("run", "stairs"), required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-only", action="store_true", help="Two physics steps, zero PPO updates")
    parser.add_argument("--smoke-design-id", help="Smoke-only: inspect any prepared design without training it")
    parser.add_argument("--smoke-num-envs", type=int, default=2)
    parser.add_argument("--smoke-physics-profile", choices=("source_native",), default="source_native",
                        help="Native smoke keeps source startup/reset randomization; no nominal evaluation overrides")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.execute and not args.smoke_only:
        parser.error("execution is opt-in; use curriculum dry-run to inspect commands")
    bank_path = Path(args.bank).resolve()
    bank = load_bank(bank_path)
    from robolab.native_expert_contract import validate_native_bank, extend_agent, RECIPE, serializable
    validate_native_bank(bank)
    if args.task not in bank["training"].get("authorized_tasks", ("run", "stairs")):
        parser.error("this experiment does not authorize that task")
    limits, schedules, budget_identity = execution_budget(bank, bank_path)
    schedule = schedules[args.task]
    if not 0 <= args.block < len(schedule):
        parser.error("block out of frozen curriculum")
    block = schedule[args.block]
    if args.smoke_design_id and not args.smoke_only:
        parser.error("design override is for zero-update physics checks only")
    if args.smoke_only and (args.resume_checkpoint or args.smoke_num_envs < 2 or args.smoke_num_envs % 2):
        parser.error("smoke requires an even small batch and never loads a checkpoint")
    design = next((r for r in bank["designs"] if r["id"] == (args.smoke_design_id or block["design_id"])), None)
    if design is None:
        parser.error("unknown physical design")
    if not args.smoke_only and (design["split"] != "train" or bool(args.resume_checkpoint) != bool(block["first_update"])):
        parser.error("holdout training / missing or unexpected full-state continuation")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.num_envs, args.seed = bank["num_envs"], bank["seed"] + args.block + 10000 * (args.task == "stairs")
    if args.smoke_only:
        args.num_envs = args.smoke_num_envs
    args.physics_profile = "source_native"
    # Evaluation values must never flow into native training commands/resets.
    args.kit_args = (args.kit_args or "") + f" --portable-root {output / 'portable'} --portable"
    sys.argv = [sys.argv[0]]
    app = AppLauncher(args).app
    env = None
    started = time.monotonic()
    try:
        import torch
        import isaaclab_tasks  # noqa
        import robolab.tasks  # noqa
        import gymnasium as gym
        from native_multimorph_runtime import configure_native, NativeMorphVecEnv, actual_physics
        from robolab.morph_expert_runner import NativeMorphRunRunner, NativeMorphStairsRunner, module_digest
        from robolab.physical_snapshot import live_actuator_audit
        torch.set_num_threads(2)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cfg, source_agent, contract = configure_native(args, bank, design, output)
        if args.task == "run" and not args.smoke_only:
            from robolab.native_expert_contract import require_normalizer_coverage
            contract["normalizer_coverage"] = require_normalizer_coverage(
                bank, schedule, source_agent, cfg.observations.disc.history_length)
        # Retain a reference before __init__: Isaac's partially-created scene
        # destructor can shut down Kit while unwinding a failed gym.make(),
        # hiding the actual asset-loading exception and returning exit code 0.
        creator = gym.envs.registration.load_env_creator(gym.spec(contract["source_task"]).entry_point)
        raw = creator.__new__(creator)
        creator.__init__(raw, cfg=cfg)
        raw.routing_contract = contract
        robot = raw.scene["robot"]
        if list(robot.joint_names) != bank["canonical_semantic_contract"]["joint_names"]:
            raise ValueError("actual native articulation joint order differs from prepared physical bank")
        symmetry = source_agent["algorithm"].get("symmetry_cfg")
        source_symmetry = symmetry["data_augmentation_func"] if symmetry else None
        if isinstance(source_symmetry, str):
            from rsl_rl.utils import resolve_callable
            source_symmetry = resolve_callable(source_symmetry)
        env = NativeMorphVecEnv(raw, context=design["context"], source_symmetry=source_symmetry,
                               clip_actions=source_agent.get("clip_actions"))
        agent = extend_agent(source_agent, task=args.task, bank_sha=file_sha(bank_path))
        # Operational settings only, never optimizer/reward/exploration overrides.
        agent.update(device=cfg.sim.device, logger="tensorboard")
        write_new(output / "source_agent.json", serializable(source_agent))
        write_new(output / "resolved_agent.json", serializable(agent))
        write_new(output / "native_contract.json", contract)
        runner_type = NativeMorphRunRunner if args.task == "run" else NativeMorphStairsRunner
        runner = runner_type(env, agent, log_dir=str(output / "models"), device=cfg.sim.device)
        if runner.alg.rnd is not None:
            raise ValueError("v1 full checkpoint protocol does not support RND")
        runner.bank, runner.bank_sha, runner.expert_task, runner.design_id = bank, file_sha(bank_path), args.task, design["id"]
        runner.execution_budget = budget_identity
        if args.resume_checkpoint:
            runner.restore_expert(args.resume_checkpoint, completed=block["first_update"])
        elif len(runner.alg.optimizer.state):
            raise RuntimeError("first expert block must start fresh")
        before = {"actor": module_digest(runner.alg.policy.actor), "critic": module_digest(runner.alg.policy.critic),
                  "optimizer_entries": len(runner.alg.optimizer.state), "learning_rate": runner.alg.learning_rate}
        if args.task == "run":
            before.update(discriminator=module_digest(runner.alg.amp_discriminator),
                discriminator_normalizer=module_digest(runner.alg.amp_discriminator.disc_obs_normalizer),
                discriminator_optimizer_entries=len(runner.alg.disc_optimizer.state))
        manifest = {"bank": str(bank_path), "bank_sha256": file_sha(bank_path), "task": args.task, "block": block,
            "design_id": design["id"], "urdf_sha256": design["urdf_sha256"], "pid": os.getpid(),
            "gp_extrapolation": design["gp_extrapolation"], "smoke_only": args.smoke_only,
            "resume_checkpoint": str(Path(args.resume_checkpoint).resolve()) if args.resume_checkpoint else None,
            "resume_sha256": file_sha(args.resume_checkpoint) if args.resume_checkpoint else None,
            "worker_sha256": file_sha(__file__), "training_recipe": RECIPE,
            "actor_sensors": "source task proprioception+known design; no task ID or mixed actor",
            "no_privileged_motion_in_actor": True, "single_actual_URDF_per_block": True,
            "same_expert_parameters_optimizer_disc_across_designs": True,
            "AMP_rollout_replay_and_physics_reset_at_switch": True}
        manifest.update(num_envs=args.num_envs, physics_profile=args.physics_profile,
                        execution_budget=budget_identity, target_updates=limits[args.task],
                        actual_design_split=design["split"], expected_nominal_total_mass_kg=design["total_mass_kg"],
                        model_before=before)
        write_new(output / "manifest.json", manifest)
        write_new(output / "live_config.json", json.loads(json.dumps(cfg.to_dict(), default=str)))
        if args.smoke_only:
            from robolab.mxt_pilot import actor_digest
            initial_actor = actor_digest(runner.alg.policy)
            initial_pose = {"root_position_relative_origin": (robot.data.root_pos_w - raw.scene.env_origins).cpu().tolist(),
                "joint_names": list(robot.joint_names), "joint_position": robot.data.joint_pos.cpu().tolist(),
                "body_names": list(robot.body_names), "body_position_w": robot.data.body_pos_w.cpu().tolist()}
            write_new(output / "initial_physical_pose.json", initial_pose)
            with torch.inference_mode():
                obs = env.get_observations()
                for _ in range(2):
                    obs, _, _, _ = env.step(runner.alg.policy.act_inference(obs))
            if actor_digest(runner.alg.policy) != initial_actor or len(runner.alg.optimizer.state):
                raise RuntimeError("zero-update smoke modified expert parameters/optimizer")
            final, completed = None, 0
        else:
            runner.learn(num_learning_iterations=block["updates"])
            completed = runner.current_learning_iteration + 1
            if completed != block["first_update"] + block["updates"]:
                raise RuntimeError("frozen block update budget mismatch")
            final = output / "models" / f"model_{completed - 1}.pt"
        write_new(output / "live_actuators.json", live_actuator_audit(robot, bank["motor_pairs"][design["pair_id"]]))
        live = actual_physics(raw)
        write_new(output / "live_physics.json", live)
        after = {"actor": module_digest(runner.alg.policy.actor), "critic": module_digest(runner.alg.policy.critic),
                 "optimizer_entries": len(runner.alg.optimizer.state), "learning_rate": runner.alg.learning_rate}
        if args.task == "run":
            after.update(discriminator=module_digest(runner.alg.amp_discriminator),
                discriminator_normalizer=module_digest(runner.alg.amp_discriminator.disc_obs_normalizer),
                discriminator_optimizer_entries=len(runner.alg.disc_optimizer.state))
        write_new(output / "summary.json", {**manifest, "result": "PHYSICS_SMOKE_ONLY" if args.smoke_only else "EXPERT_BLOCK_COMPLETE_NOT_GAIT_VALIDATED",
            "completed_updates": completed, "final_checkpoint": str(final) if final else None,
            "final_sha256": file_sha(final) if final else None, "wall_seconds": time.monotonic() - started,
            "model_after": after, "actor_changed": before["actor"] != after["actor"],
            "cuda_tensor_peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,
            "cuda_tensor_peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_available() else None,
            "cuda_tensor_memory_excludes_non_torch_renderer_and_PhysX": True})
    except BaseException as error:
        import traceback
        diagnostic = traceback.format_exc()
        print(diagnostic, file=sys.stderr, flush=True)
        write_new(output / "failure.json", {"type": type(error).__name__, "error": str(error), "traceback": diagnostic})
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
