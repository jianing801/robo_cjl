"""Independent Rough-only rotating-stage worker, copied without changing original entries.

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


def attach_global_logger(runner, target):
    """Native metrics every PPO update, with the full experiment denominator."""
    native_log = runner.logger.log
    def log_global_update(**values):
        values["total_it"] = target
        native_log(**values)
    runner.logger.log = log_global_update


def reset_group_logger(logger):
    """A new physical group has no completed episodes from the previous group."""
    import torch
    with torch.inference_mode():
        logger.cur_reward_sum.zero_()
        logger.cur_episode_length.zero_()
    logger.ep_extras.clear()
    logger.rewbuffer.clear()
    logger.lenbuffer.clear()


def capture_learning_rng(session):
    import torch
    if session is not None:
        session['learning_rng']=(torch.get_rng_state(),torch.cuda.get_rng_state_all())


def restore_learning_rng(session):
    import torch
    cpu,cuda=session['learning_rng']
    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)


def main(argv=None, session=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--design-ids", nargs="+")
    p.add_argument("--hybrid-plan", required=True)
    p.add_argument("--stage-index", required=True, type=int)
    p.add_argument("--num-envs", type=int, help="Default: bank configuration (2048 for legacy banks)")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--updates", type=int)
    p.add_argument("--resume-checkpoint")
    p.add_argument("--asset-cache", help="Optional read-only root of previously converted USDs; copy only exact converter-hash matches")
    p.add_argument("--allow-cache-misses", action="store_true",
                   help="Internal experiment cache: convert previously unseen designs normally")
    # Planning needs no Isaac/Kit imports.
    if any(flag in sys.argv for flag in ("--smoke-only", "--execute")):
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(argv)
    from hybrid_support import write_new, source_identity
    from robolab.rough_geometry import file_sha
    from hybrid_layout import make_layout, identity
    bank_path = Path(a.bank).resolve()
    from hybrid_bank import KIND, load_configurable_bank
    if json.loads(bank_path.read_text(encoding="utf8")).get("kind") != KIND:
        raise ValueError("hybrid worker requires its independently generated hybrid bank; old banks remain untouched")
    bank = load_configurable_bank(bank_path)
    task = bank.get("task_key", "stairs")
    if task != "stairs":
        raise ValueError("this independent copy supports Rough only")
    from hybrid_contract import load_plan, stage_spec
    frozen = load_plan(a.hybrid_plan, bank_path)
    a.torque_soft_constraint = frozen['configuration'].get('torque_soft_constraint')
    stage = stage_spec(bank, frozen["configuration"], a.stage_index)
    if a.design_ids is not None and a.design_ids != stage["design_ids"]:
        raise ValueError("CLI design IDs differ from frozen schedule")
    a.design_ids = stage["design_ids"]
    if a.num_envs is None:
        a.num_envs = frozen["configuration"]["grid"]["num_envs"]
    if a.num_envs != frozen["configuration"]["grid"]["num_envs"]:
        raise ValueError("environment count differs from frozen plan")
    if sum((a.smoke_only, a.execute)) > 1:
        p.error("choose one execution mode")
    if a.smoke_only and (a.updates or a.resume_checkpoint):
        p.error("smoke is zero-update and cannot load policies")
    if a.execute and (a.updates != stage["end_update"]):
        p.error("formal stage target must exactly equal the frozen schedule boundary")
    if a.resume_checkpoint and not a.execute:
        p.error("resume requires explicit execute")
    layout = make_layout(bank, a.num_envs, a.design_ids)
    designs = [next(d for d in bank["designs"] if d["id"] == key) for key in layout["design_ids"]]
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    experiment = frozen["experiment"]
    write_new(output / "plan.json", dict(**experiment, plan_sha256=identity(experiment),
        stage=stage, requested_updates=a.updates, smoke_only=a.smoke_only,
        execution_protocol="direct_stage_v2", pid=os.getpid()))
    if not a.smoke_only and not a.execute:
        print(json.dumps({k: v for k, v in layout.items() if k != "env_design_index"}, indent=2))
        return
    from hybrid_processes import reject_duplicate_bank_safely as reject_duplicate_bank
    reject_duplicate_bank(bank_path)
    a.task = task
    a.kit_args = (a.kit_args or "") + f" --portable-root {output / 'portable'} --portable"
    app = session.get('app') if session is not None else None
    if app is None:
        app = AppLauncher(a).app
        if session is not None: session['app'] = app
    env = raw = None
    try:
        import torch
        from hybrid_resources import prepare_scene, snapshot
        prepare_scene(output / 'scene_memory.jsonl', torch)
        import isaaclab_tasks
        import robolab.tasks
        import omni.usd
        from hybrid_importer import install_local_importer_adapter
        write_new(output / 'importer_audit.json', install_local_importer_adapter())
        from hybrid_runtime import configure_spatial, SpatialRoughEnv, SpatialVecEnv, SpatialRunner
        from hybrid_support import extend_agent, serializable
        from rsl_rl.utils import resolve_callable
        torch.set_num_threads(2)
        env_type, runner_type, configurator = SpatialRoughEnv, SpatialRunner, None
        cfg, source, contract = configure_spatial(a, bank, designs, output, configurator=configurator)
        if session is not None:
            def check_cycles(obj, path='', ancestors=None):
                ancestors = {} if ancestors is None else ancestors
                if type(obj).__name__ == 'MeshConverterCfg': return
                if isinstance(obj, (list, tuple)): fields = enumerate(obj)
                elif isinstance(obj, dict): fields = obj.items()
                elif hasattr(obj, '__dict__'): fields = vars(obj).items()
                else: return
                if id(obj) in ancestors:
                    raise ValueError(f'configuration reference cycle: {path} -> {ancestors[id(obj)]}')
                chain = dict(ancestors); chain[id(obj)] = path
                for key, value in fields:
                    if not str(key).startswith('__'): check_cycles(value, f'{path}.{key}', chain)
            check_cycles(cfg)
        raw = env_type.__new__(env_type)
        env_type.__init__(raw, cfg)
        snapshot(output / 'scene_memory.jsonl', 'after_scene_initialization', torch)
        scene_stage = omni.usd.get_context().get_stage()
        # Never assume regex traversal and tensor environment ordering coincide.
        actual_ids = [scene_stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot").GetAttribute("spatial:design_id").Get()
                      for i in range(a.num_envs)]
        expected_ids = [layout["design_ids"][i] for i in layout["env_design_index"]]
        if actual_ids != expected_ids:
            raise ValueError("spawn order != tensor order: refusing wrong context/motor association")
        if list(raw.robot.joint_names) != bank["canonical_semantic_contract"]["joint_names"]:
            raise ValueError("physical joint order differs from canonical bank")
        expected_pose = torch.tensor(bank["canonical_semantic_contract"]["default_joint_pos"], device=raw.device)
        if not torch.allclose(raw.robot.data.default_joint_pos, expected_pose.expand_as(raw.robot.data.default_joint_pos), atol=1e-6, rtol=0):
            raise ValueError("source Rough default posture differs from bank; do not silently retarget reset pose")
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
        agent = extend_agent(source, bank_sha=file_sha(bank_path), context_width=layout["context_width"], task=task)
        agent["policy"]["class_name"] = "hybrid_policy:SpatialActorCritic"
        if agent["algorithm"].get("symmetry_cfg") is not None:
            agent["algorithm"]["symmetry_cfg"]["data_augmentation_func"] = "hybrid_policy:spatial_design_symmetry"
        agent["policy"]["native_config"]["spatial_identity"] = identity(experiment)
        agent.update(device=cfg.sim.device, logger="tensorboard")
        continuing = session is not None and session.get('runner') is not None
        if continuing:
            runner = session['runner']
            if runner.current_learning_iteration != stage['first_update']:
                raise ValueError('persistent runner iteration differs from next group')
            obs = env.get_observations()
            if any(obs[k].shape != runner.alg.storage.observations[k].shape[1:]
                   for k in obs.keys()):
                raise ValueError('persistent group changed storage observation shapes')
            runner.env = env
            runner.alg.storage.clear()
            runner.alg.storage.spatial_groups = raw.spatial_groups
            if runner.alg.symmetry:
                runner.alg.symmetry['_env'] = env
            # Logging from different physical episodes must not be combined.
            reset_group_logger(runner.logger)
        else:
            runner = runner_type(env, agent, log_dir=str(output / "models"), device=cfg.sim.device)
            if session is not None: session['runner'] = runner
        runner.hybrid_identity = experiment
        runner.hybrid_stage = stage
        # Keep the native per-update metrics, but display the absolute target
        # instead of misleadingly showing only this group's end boundary.
        if not continuing:
            attach_global_logger(runner, frozen["configuration"]["rotation"]["total_updates"])
        if a.execute and bool(a.resume_checkpoint) != (stage["first_update"] > 0):
            raise ValueError("exact previous completed-stage checkpoint required; no scratch restart")
        if a.resume_checkpoint and not continuing:
            runner.restore_spatial(a.resume_checkpoint)
            saved=torch.load(a.resume_checkpoint,map_location=raw.device,weights_only=False)
            def same(left,right):
                if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left.detach().cpu(),right.detach().cpu())
                if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(same(v,right[k]) for k,v in left.items())
                if isinstance(left,(list,tuple)):return len(left)==len(right) and all(same(x,y) for x,y in zip(left,right))
                return left==right
            policy_equal=same(saved['model_state_dict'],runner.alg.policy.state_dict())
            optimizer_equal=same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
            if not policy_equal or not optimizer_equal:raise ValueError('restore altered policy/normalization/optimizer')
            write_new(output/'restore_audit.json',dict(checkpoint=str(a.resume_checkpoint),next_update=runner.current_learning_iteration,
                policy_and_normalization_exact=policy_equal,adam_exact=optimizer_equal,learning_rate=runner.alg.learning_rate))
        remaining = 0 if a.smoke_only else a.updates - runner.current_learning_iteration
        if continuing:
            # Rebuilding the environment seeds and consumes Torch RNG. Restore
            # the learning stream after scene creation, as checkpoint resume did.
            restore_learning_rng(session)
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
        capture_learning_rng(session)
        audit = {}
        for name in ("knees", "feet"):
            actuator = raw.robot.actuators[name]
            error = (actuator._clip_effort(actuator.computed_effort) - actuator.applied_effort).abs().max().item()
            if error > 1e-4 or not torch.isfinite(actuator.applied_effort).all():
                raise ValueError("live grouped motor clipping mismatch")
            audit[name] = dict(clipping_max_error=error, armature=actuator.armature.cpu().tolist(),
                motor_groups=actuator.groups.cpu().tolist())
        result = ("HYBRID_PHYSICS_SMOKE_PASSED" if a.smoke_only else
                  "HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED")
        write_new(output / "summary.json", dict(result=result,
            spatial_identity_sha256=identity(experiment),
            optimizer_updates=0 if a.smoke_only else remaining,
            completed_updates=0 if a.smoke_only else runner.current_learning_iteration + 1,
            absolute_target_updates=a.updates, stage=stage, layout=layout, motor_audit=audit,
            no_serial_checkpoint_loaded=True, requires_gait_evaluation=not a.smoke_only))
        print("SPATIAL_COMPLETE", output, flush=True)
    except BaseException:
        import traceback
        write_new(output / "failure.json", dict(traceback=traceback.format_exc()))
        if session is not None and session.get('failure_signal'):
            write_new(session['failure_signal'], dict(group_output=str(output),
                traceback=traceback.format_exc()))
        traceback.print_exc()
        sys.stderr.flush()
        from hybrid_resources import arm_failed_cleanup_timeout
        arm_failed_cleanup_timeout()
        raise
    finally:
        # SimulationApp.close may terminate the interpreter with status zero.
        # Flush the authoritative traceback first and force a failing status,
        # even when Kit exits during cleanup.
        failed = sys.exc_info()[0] is not None
        if not failed:
            snapshot(output / 'scene_memory.jsonl', 'before_scene_close', torch)
        if failed:
            import atexit
            atexit.register(lambda: os._exit(1))
        if env is not None:
            from hybrid_importer import close_without_masking
            close_without_masking(env, failed)
        elif raw is not None and hasattr(raw, '_is_closed'):
            from hybrid_importer import close_without_masking
            close_without_masking(raw, failed)
        if session is None:
            app.close()
        else:
            if session.get('runner') is not None:
                session['runner'].env = None
                if session['runner'].alg.symmetry:
                    session['runner'].alg.symmetry['_env'] = None
            # Keep Kit, policy, Adam and rollout allocations; remove old USD
            # physics objects before the next real morphology group is built.
            import omni.usd
            from isaaclab.sensors import RayCaster
            # RayCaster.meshes is a class-level Warp mesh cache. It references
            # the old terrain/device runtime and is traversed by cfg.validate.
            # New terrain MUST rebuild it; otherwise the next scene can also
            # raycast the previous group's terrain.
            RayCaster.meshes.clear()
            omni.usd.get_context().new_stage()
            if not failed:
                snapshot(output / 'scene_memory.jsonl', 'after_scene_close', torch)


if __name__ == "__main__":
    main()
