"""Isolated new-physics backpack feasibility worker. No old chain is resumed."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "rsl_rl")]
from robolab.backpack_distribution import sha, write_new, validate_checkpoint, validate_plan

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", required=True)
parser.add_argument("--stage", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--resume-checkpoint")
parser.add_argument("--smoke-only", action="store_true")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
plan_sha = sha(args.plan)
validate_plan(plan)
design, stage = plan["design"], plan["stages"][args.stage]
if sha(design["urdf"]) != design["urdf_sha256"]:
    raise ValueError("URDF identity changed")
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
args.kit_args = (args.kit_args or "") + f" --portable-root {output / 'portable'} --portable"
app = AppLauncher(args).app

import torch
import gymnasium as gym
import isaaclab_tasks
import robolab.tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from robolab.physical_snapshot import configure_robot, live_actuator_audit
from robolab.tasks.direct.backpack_walk.backpack_urdf import (
    sample_backpack_properties, write_backpack_urdf, apply_backpack_properties)
from ppo_resume_state import restore_learning_rate


class Runner(OnPolicyRunner):
    def save(self, path, infos=None):
        if Path(path).exists():
            return
        metadata = dict(infos or {}, distribution_plan_sha256=plan_sha,
                        stage=args.stage, design_id=design["id"],
                        torch_rng_state=torch.get_rng_state(),
                        cuda_rng_state=torch.cuda.get_rng_state_all())
        temporary = path + ".pending"
        super().save(temporary, metadata)
        os.rename(temporary, path)


def main():
    torch.set_num_threads(2)
    task = "RPO-Flat-Backpack-Compliant"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    # Freeze the backpack baseline locally to this worker. The repository HEAD
    # also contains co-design reward/DR edits; do not change AMP/Rough globally.
    cfg.reward.track_lin_vel_xy_exp.weight = 1.0
    cfg.reward.track_ang_vel_z_exp.weight = 1.0
    if hasattr(cfg.reward, "underspeed_lin_vel"):
        cfg.reward.underspeed_lin_vel = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
    cfg.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
    cfg.events.scale_joint_parameters.params["armature_distribution_params"] = (0.8, 1.2)
    from isaaclab.envs import mdp
    cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
    cfg.scene_context.num_envs = cfg.scene.num_envs = plan["num_envs"]
    cfg.seed = agent.seed = plan["seed"]
    cfg.sim.device = agent.device = args.device or "cuda:0"
    robot_cfg = cfg.scene_context.robot.copy()
    pack_urdf = output / "robot_with_backpack.urdf"
    write_backpack_urdf(design["urdf"], pack_urdf, sum(stage["mass_range_kg"]) / 2)
    robot_cfg.spawn = robot_cfg.spawn.replace(asset_path=str(pack_urdf),
        usd_dir=str(output / "usd"), merge_fixed_joints=False, activate_contact_sensors=True)
    configure_robot(robot_cfg, plan["physical_model"])
    robot_cfg.init_state.pos = (*robot_cfg.init_state.pos[:2],
        robot_cfg.init_state.pos[2] + design["root_height_compensation_m"])
    cfg.scene_context.robot = cfg.scene.robot = robot_cfg
    cfg.scene.replicate_physics = True
    cfg.commands.debug_vis = False
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    for name in ("left_feet_scanner", "right_feet_scanner", "height_scanner"):
        sensor = getattr(cfg.scene, name, None)
        if sensor is not None:
            sensor.debug_vis = False
    cfg.log_dir = str(output)
    agent.policy.class_name = "rsl_rl.modules.actor_critic:ActorCritic"
    agent.algorithm.class_name = "rsl_rl.algorithms:PPO"
    for name in ("optimizer", "share_cnn_encoders"):
        if hasattr(agent.algorithm, name):
            delattr(agent.algorithm, name)
    agent.logger, agent.save_interval = "tensorboard", 100
    write_new(output / "resolved_env.json", cfg.to_dict())
    write_new(output / "resolved_agent.json", agent.to_dict())
    properties = sample_backpack_properties(plan["num_envs"], seed=plan["seed"] + args.stage,
                                           mass_range_kg=tuple(stage["mass_range_kg"]))
    raw = gym.make(task, cfg=cfg)
    try:
        raw.reset()
        audit = apply_backpack_properties(raw.unwrapped.scene["robot"], properties)
        raw.reset()
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        runner = Runner(env, agent.to_dict(), log_dir=str(output / "models"), device=agent.device)
        first = stage["start"]
        if args.resume_checkpoint:
            saved = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
            first = saved["iter"] + 1
            validate_checkpoint(saved, plan_sha, first)
            if not stage["start"] <= first <= stage["stop"]:
                raise ValueError("checkpoint outside this stage")
            expected_stage = args.stage - 1 if first == stage["start"] else args.stage
            if saved["infos"].get("stage") != expected_stage:
                raise ValueError("incorrect predecessor stage")
            runner.load(args.resume_checkpoint, map_location=agent.device)
            restore_learning_rate(runner.alg)
        elif stage["start"]:
            raise ValueError("later stages require full PPO checkpoint")
        runner.current_learning_iteration = first
        with torch.inference_mode():
            for _ in range(2):
                obs, reward, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=agent.device))
                if not torch.isfinite(reward).all() or any(not torch.isfinite(v).all() for v in obs.values()):
                    raise ValueError("nonfinite physics smoke")
        write_new(output / "manifest.json", dict(plan_sha256=plan_sha, pid=os.getpid(), stage=stage,
            payload_audit=audit, num_envs=env.num_envs, resume=args.resume_checkpoint,
            smoke_only=args.smoke_only, actor_payload_privileged_input=False,
            actuators=live_actuator_audit(raw.unwrapped.scene["robot"], plan["physical_model"])))
        write_new(output / "payload_samples.json", properties)
        if args.smoke_only:
            write_new(output / "summary.json", dict(result="PHYSICS_ONLY", updates=0))
            return
        if args.resume_checkpoint:
            torch.set_rng_state(saved["infos"]["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["infos"]["cuda_rng_state"]])
        if first < stage["stop"]:
            runner.learn(stage["stop"] - first, init_at_random_ep_len=True)
        else:
            runner.current_learning_iteration = first - 1
            runner.save(str(output / "models" / f"model_{first - 1}.pt"))
        checkpoint = output / "models" / f"model_{stage['stop'] - 1}.pt"
        validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False), plan_sha, stage["stop"])
        write_new(output / "summary.json", dict(result="STAGE_COMPLETE_NOT_GAIT_VALIDATED",
            completed_updates=stage["stop"], checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
    finally:
        raw.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
