"""Paper-inspired spatial Rough, isolated from all existing serial experiments.

Default: CPU plan only. --smoke-only: physical verification, zero PPO updates.
--execute --updates N: explicit finite training with configurable environment count.
"""
import argparse
from pathlib import Path
import sys
import json
import os
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "rsl_rl")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--design-ids", nargs="+")
    p.add_argument("--num-envs", type=int, help="Default: bank configuration (2048 for legacy banks)")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="Configured envs, all train designs, exactly two PPO updates; discard policy")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--updates", type=int)
    p.add_argument("--resume-checkpoint")
    p.add_argument("--preflight-report", help="Matching all-design two-update preflight summary; required for long training")
    p.add_argument("--asset-cache", help="Optional read-only root of previously converted USDs; copy only exact converter-hash matches")
    # Planning needs no Isaac/Kit imports.
    if any(flag in sys.argv for flag in ("--smoke-only", "--preflight-only", "--execute")):
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    from robolab.morph_experiments import load_bank, write_new
    from robolab.mxt_design_transfer import file_sha
    from robolab.native_expert_contract import validate_native_bank, source_identity
    from robolab.spatial_morphology import make_layout, identity
    bank_path = Path(a.bank).resolve()
    from robolab.configurable_morphology import KIND, load_configurable_bank
    configurable = json.loads(bank_path.read_text(encoding="utf8")).get("kind") == KIND
    if configurable:
        bank = load_configurable_bank(bank_path)
    else:
        bank = load_bank(bank_path)
        validate_native_bank(bank)
    if a.num_envs is None:
        a.num_envs = bank["num_envs"]
    if sum((a.smoke_only, a.preflight_only, a.execute)) > 1:
        p.error("choose one execution mode")
    if a.smoke_only and (a.updates or a.resume_checkpoint):
        p.error("smoke is zero-update and cannot load policies")
    if a.preflight_only:
        if a.design_ids or a.updates or a.resume_checkpoint:
            p.error("preflight is all train designs / fresh / two updates")
        a.updates = 2
    if a.execute and (a.updates is None or a.updates < 1 or a.design_ids):
        p.error("formal training requires every train design and explicit finite updates")
    if a.resume_checkpoint and not a.execute:
        p.error("resume requires explicit execute")
    layout = make_layout(bank, a.num_envs, a.design_ids)
    designs = [next(d for d in bank["designs"] if d["id"] == key) for key in layout["design_ids"]]
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT / name for name in ("robolab/spatial_morphology.py", "scripts/tools/spatial_rough_runtime.py",
        "scripts/tools/train_spatial_rough.py", "robolab/configurable_morphology.py", "robolab/spatial_policy.py")]
    experiment = dict(kind="spatial_native_rough_v1", bank_sha256=file_sha(bank_path), layout=layout,
        source_identity=source_identity(), spatial_sources={str(f.relative_to(ROOT)): file_sha(f) for f in source_files},
        task="stairs", discount="unchanged source", pd="unchanged source",
        reward_commands_randomization_mirror="unchanged native Rough; existing 5-10cm stairs override retained",
        separate_from_serial=True)
    if a.execute:
        if not a.preflight_report:
            p.error("formal execution requires a successful matching preflight report")
        proof = json.loads(Path(a.preflight_report).read_text())
        if (proof.get("result") != "SPATIAL_TWO_UPDATE_PREFLIGHT_PASSED_DISCARD_POLICY"
                or proof.get("spatial_identity_sha256") != identity(experiment)
                or proof.get("layout") != layout):
            raise ValueError("preflight does not match current source/assets/layout")
    write_new(output / "plan.json", dict(**experiment, plan_sha256=identity(experiment),
        requested_updates=a.updates, smoke_only=a.smoke_only, preflight_only=a.preflight_only, pid=os.getpid()))
    if not a.smoke_only and not a.execute and not a.preflight_only:
        print(json.dumps({k: v for k, v in layout.items() if k != "env_design_index"}, indent=2))
        return
    # Windows WDDM nvidia-smi lists desktop graphics clients as compute apps.
    # Check actual Python/Isaac workers rather than treating Explorer as a trainer.
    import psutil
    own_tree = {os.getpid(), *[v.pid for v in psutil.Process().parents()]}
    workflow_names = {"train_multimorph_expert_block.py", "evaluate_native_rough_case.py", "run_multimorph_experts.py",
                      "run_native_expert_pair.py", "diagnose_short_rough.py", "auto_evaluate_completed_rough.py"}
    for process in psutil.process_iter(["name", "cmdline"]):
        if process.pid in own_tree:
            continue
        name = (process.info["name"] or "").lower()
        if "python" in name or name in ("kit.exe", "isaac-sim.exe"):
            raise RuntimeError(f"other Python/Isaac worker {process.pid}; inspect before launching")
        if any(Path(v).name in workflow_names for v in (process.info["cmdline"] or [])):
            raise RuntimeError("another training/evaluation workflow exists")
    a.task = "stairs"
    a.kit_args = (a.kit_args or "") + f" --portable-root {output / 'portable'} --portable"
    app = AppLauncher(a).app
    env = raw = None
    try:
        import torch
        import isaaclab_tasks
        import robolab.tasks
        import omni.usd
        from spatial_rough_runtime import configure_spatial, SpatialRoughEnv, SpatialVecEnv, SpatialRunner
        from robolab.native_expert_contract import extend_agent, serializable
        from rsl_rl.utils import resolve_callable
        torch.set_num_threads(2)
        cfg, source, contract = configure_spatial(a, bank, designs, output)
        raw = SpatialRoughEnv.__new__(SpatialRoughEnv)
        SpatialRoughEnv.__init__(raw, cfg)
        stage = omni.usd.get_context().get_stage()
        # Never assume regex traversal and tensor environment ordering coincide.
        actual_ids = [stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot").GetAttribute("spatial:design_id").Get()
                      for i in range(a.num_envs)]
        expected_ids = [layout["design_ids"][i] for i in layout["env_design_index"]]
        if actual_ids != expected_ids:
            raise ValueError("spawn order != tensor order: refusing wrong context/motor association")
        if list(raw.robot.joint_names) != bank["canonical_semantic_contract"]["joint_names"]:
            raise ValueError("physical joint order differs from canonical bank")
        # Tags are not evidence of real differing geometry: independently check
        # parent/child link distances against each corresponding URDF origin.
        import xml.etree.ElementTree as ET
        lengths = {}
        positions = raw.robot.data.body_pos_w
        for group, design in enumerate(designs):
            ids = raw.spatial_groups == group
            xml = ET.parse(design["urdf"]).getroot()
            rows = {}
            for joint_name in ("left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint"):
                joint = xml.find(f"joint[@name='{joint_name}']")
                parent = raw.robot.body_names.index(joint.find("parent").get("link"))
                child = raw.robot.body_names.index(joint.find("child").get("link"))
                xyz = [float(x) for x in joint.find("origin").get("xyz").split()]
                expected = sum(x*x for x in xyz)**.5
                actual = (positions[ids, parent] - positions[ids, child]).norm(dim=-1)
                error = (actual-expected).abs().max().item()
                rows[joint_name] = dict(expected=expected, actual_min=actual.min().item(), actual_max=actual.max().item(), max_error=error)
                if error > 2e-4:
                    write_new(output / "geometry_failure.json", dict(design=design["id"], joint=joint_name, **rows[joint_name]))
                    raise ValueError("real physical link geometry differs from assigned URDF; not valid spatial training")
            lengths[design["id"]] = rows
        write_new(output / "geometry_audit.json", lengths)
        symmetry = source["algorithm"].get("symmetry_cfg")
        fn = resolve_callable(symmetry["data_augmentation_func"]) if symmetry else None
        env = SpatialVecEnv(raw, fn, source.get("clip_actions"))
        agent = extend_agent(source, task="stairs", bank_sha=file_sha(bank_path))
        if configurable:
            agent["policy"]["class_name"] = "robolab.spatial_policy:SpatialActorCritic"
            agent["policy"]["native_config"]["context_width"] = layout["context_width"]
            if agent["algorithm"].get("symmetry_cfg"):
                agent["algorithm"]["symmetry_cfg"]["data_augmentation_func"] = "robolab.spatial_policy:spatial_design_symmetry"
        agent["policy"]["native_config"]["spatial_identity"] = identity(experiment)
        agent.update(device=cfg.sim.device, logger="tensorboard")
        runner = SpatialRunner(env, agent, log_dir=str(output / "models"), device=cfg.sim.device)
        runner.spatial_identity = experiment
        if a.resume_checkpoint:
            runner.restore_spatial(a.resume_checkpoint)
        remaining = 0 if a.smoke_only else a.updates - runner.current_learning_iteration
        if not a.smoke_only and remaining <= 0:
            raise ValueError("absolute update target already reached; do not add another budget")
        write_new(output / "manifest.json", dict(**experiment, actual_design_ids=actual_ids,
            source_agent=serializable(source), native_contract=contract,
            initial_root_height=raw.robot.data.default_root_state[:, 2].cpu().tolist(),
            before_body_masses=raw.robot.root_physx_view.get_masses().tolist(),
            before_body_positions=raw.robot.data.body_pos_w.cpu().tolist(),
            body_names=list(raw.robot.body_names),
            actor_observation_width=runner.alg.storage.observations["policy"].shape[-1]))
        if a.smoke_only:
            with torch.inference_mode():
                obs = env.get_observations()
                for _ in range(8):
                    obs, _, _, _ = env.step(runner.alg.policy.act_inference(obs))
            if len(runner.alg.optimizer.state):
                raise RuntimeError("zero-update smoke changed optimizer")
        else:
            runner.learn(remaining)
        audit = {}
        for name in ("knees", "feet"):
            actuator = raw.robot.actuators[name]
            error = (actuator._clip_effort(actuator.computed_effort) - actuator.applied_effort).abs().max().item()
            if error > 1e-4 or not torch.isfinite(actuator.applied_effort).all():
                raise ValueError("live grouped motor clipping mismatch")
            audit[name] = dict(clipping_max_error=error, armature=actuator.armature.cpu().tolist(),
                motor_groups=actuator.groups.cpu().tolist())
        result = ("SPATIAL_PHYSICS_SMOKE_PASSED" if a.smoke_only else
                  "SPATIAL_TWO_UPDATE_PREFLIGHT_PASSED_DISCARD_POLICY" if a.preflight_only else
                  "SPATIAL_FINITE_TRAINING_COMPLETE_NOT_GAIT_VALIDATED")
        write_new(output / "summary.json", dict(result=result,
            spatial_identity_sha256=identity(experiment),
            optimizer_updates=0 if a.smoke_only else remaining,
            completed_updates=0 if a.smoke_only else runner.current_learning_iteration + 1,
            absolute_target_updates=a.updates, layout=layout, motor_audit=audit,
            no_serial_checkpoint_loaded=True, requires_gait_evaluation=not a.smoke_only))
        print("SPATIAL_COMPLETE", output, flush=True)
    except BaseException:
        import traceback
        write_new(output / "failure.json", dict(traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
        app.close()


if __name__ == "__main__":
    main()
