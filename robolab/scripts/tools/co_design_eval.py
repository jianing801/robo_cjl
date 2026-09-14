"""Deterministic, clean evaluation for the RPO leg co-design study.

The evaluator supports one or more semicolon-separated velocity commands and
repeats each command for ``--num-episodes`` complete rollouts. It measures the
pre-reset terminal state, so transport distance, mechanical power, CoT, and
optional recordings never include the environment's automatic reset pose.
The structural performance metric is the time-averaged squared command
tracking error; episode return is retained only as a training diagnostic.
"""

import argparse
import json
import math
import os
import sys
import tempfile

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--urdf-model", choices=["sw", "legacy"], default="sw",
                    help="Must match the URDF model used during training.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--thigh", type=float, required=True)
parser.add_argument("--calf", type=float, required=True)
parser.add_argument("--task", type=str, default="RPO-Flat")
parser.add_argument("--num-episodes", type=int, default=1,
                    help="Complete rollouts to average for each command.")
parser.add_argument("--num-envs", type=int, default=1,
                    help="Only one environment is supported for scalar CoT reporting.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--command", type=str, default="1.0,0,0",
                    help="One or more ';'-separated commands, each formatted as vx,vy,wz.")
parser.add_argument("--record", type=str, default=None,
                    help="Export one command and one episode of robot state to this CSV path.")
parser.add_argument("--no-heading", action="store_true", default=False,
                    help="Disable heading tracking (A/B comparison only; normal evaluation keeps it on).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    parser.error("--num-envs must be 1: scalar return, power, mass, and CoT are reported for one robot.")
if args_cli.num_episodes < 1:
    parser.error("--num-episodes must be at least 1.")
if args_cli.record and (args_cli.num_episodes != 1 or ";" in args_cli.command):
    parser.error("--record supports exactly one command and one episode to keep the CSV unambiguous.")


def _parse_commands(spec: str) -> list[tuple[float, float, float]]:
    commands = []
    for command_text in spec.split(";"):
        fields = [field.strip() for field in command_text.split(",") if field.strip()]
        if len(fields) != 3:
            parser.error(f"Invalid command '{command_text}': expected vx,vy,wz.")
        try:
            commands.append(tuple(float(field) for field in fields))
        except ValueError:
            parser.error(f"Invalid numeric command '{command_text}'.")
    if not commands:
        parser.error("At least one velocity command is required.")
    return commands


commands = _parse_commands(args_cli.command)
print(f"[eval] {len(commands)} command(s): {commands}", flush=True)

# Clear command-line arguments before Hydra/RSL-RL imports inspect sys.argv.
sys.argv = [sys.argv[0]]

# Compatibility workaround for the local rsl-rl installation.
import distutils.core
import setuptools

setuptools.setup = lambda *a, **kw: print("[Bypassed] setuptools.setup skipped.")
distutils.core.setup = lambda *a, **kw: print("[Bypassed] distutils.core.setup skipped.")

import rsl_rl.runners.on_policy_runner as opr

if hasattr(opr, "resolve_callable"):
    _orig_resolve = opr.resolve_callable

    def _smart_resolve(value):
        mapping = {
            "ActorCritic": "rsl_rl.modules.actor_critic:ActorCritic",
            "PPO": "rsl_rl.algorithms:PPO",
        }
        return _orig_resolve(mapping.get(value, value))

    opr.resolve_callable = _smart_resolve


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from packaging import version
import importlib.metadata as metadata

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
import robolab.tasks  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from robolab.scripts.tools.generate_urdf import generate_urdf
if args_cli.urdf_model == "sw":
    from robolab.scripts.tools.generate_urdf_sw import generate_urdf
from robolab.assets.robots import RPO_CFG
from robolab.tasks.direct.base.scene_cfg import SceneCfg


installed_version = metadata.version("rsl-rl-lib")
tmp_dir = tempfile.mkdtemp(prefix="rpo_eval_")
urdf_path = generate_urdf(args_cli.thigh, args_cli.calf, output_dir=tmp_dir)
base_z = 0.75 + (args_cli.thigh + args_cli.calf - 0.55)


def _command_key(command: tuple[float, float, float]) -> str:
    return ",".join(f"{value:g}" for value in command)


def _make_eval_env(command: tuple[float, float, float]):
    """Create a fresh nominal environment with a fixed command distribution."""
    lx, ly, az = command
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    if hasattr(env_cfg, "scene_context"):
        env_cfg.scene_context.num_envs = 1

    env_cfg.noise.add_noise = False
    for event_name in (
        "push_robot", "physics_material", "add_base_mass", "randomize_rigid_body_com",
        "scale_link_mass", "scale_actuator_gains", "scale_joint_parameters",
        "reset_base", "reset_robot_joints",
    ):
        if hasattr(env_cfg.events, event_name) and getattr(env_cfg.events, event_name) is not None:
            setattr(env_cfg.events, event_name, None)

    env_cfg.episode_length_s = 20.0
    env_cfg.capture_terminal_state = True
    env_cfg.commands.rel_standing_envs = 0.0
    # Degenerate ranges ensure both the initial reset and every automatic reset
    # use the requested command. This is essential for multi-episode runs.
    env_cfg.commands.ranges.lin_vel_x = (lx, lx)
    env_cfg.commands.ranges.lin_vel_y = (ly, ly)
    env_cfg.commands.ranges.ang_vel_z = (az, az)
    env_cfg.commands.resampling_time_range = (100.0, 100.0)

    # Heading stabilization and an explicit yaw-rate command are mutually
    # exclusive in UniformVelocityCommand. Keep heading for translation, but
    # automatically disable it for a requested turn so multi-command reports
    # do not silently turn ``az`` into zero.
    heading_tracking = not args_cli.no_heading and az == 0.0
    if not heading_tracking:
        env_cfg.commands.heading_command = False
        env_cfg.commands.rel_heading_envs = 0.0
    else:
        env_cfg.commands.heading_command = True
        env_cfg.commands.rel_heading_envs = 1.0
        # The nominal RPO reset yaw is zero. Fixing this target makes automatic
        # resets deterministic while retaining heading stabilization.
        env_cfg.commands.ranges.heading = (0.0, 0.0)

    if hasattr(env_cfg.scene, "terrain") and env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None

    custom_robot_cfg = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    custom_robot_cfg.spawn.asset_path = urdf_path
    custom_robot_cfg.init_state.pos = (0.0, 0.0, base_z)
    if hasattr(env_cfg, "scene_context"):
        env_cfg.scene_context.robot = custom_robot_cfg
        env_cfg.scene = SceneCfg(
            config=env_cfg.scene_context,
            physics_dt=env_cfg.sim.dt,
            step_dt=env_cfg.decimation * env_cfg.sim.dt,
        )
    else:
        env_cfg.scene.robot = custom_robot_cfg

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)
    agent_cfg.logger = "tensorboard"

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    return RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions), agent_cfg, heading_tracking


def _terminal_or_live_state(robot_asset, terminal_state):
    """Return env-0 state after a step, preferring the pre-reset snapshot."""
    if terminal_state is not None:
        env_ids = terminal_state["env_ids"].tolist()
        if env_ids != [0]:
            raise RuntimeError(f"Unexpected terminal snapshot env_ids={env_ids}; expected [0].")
        return {name: value[0] for name, value in terminal_state.items() if name != "env_ids"}

    data = robot_asset.data
    return {
        "root_pos_w": data.root_pos_w[0],
        "root_quat_w": data.root_quat_w[0],
        "root_lin_vel_w": data.root_lin_vel_w[0],
        "root_ang_vel_w": data.root_ang_vel_w[0],
        "projected_gravity_b": data.projected_gravity_b[0],
        "joint_pos": data.joint_pos[0],
        "joint_vel": data.joint_vel[0],
        "applied_torque": data.applied_torque[0],
    }


def _record_row(state, reward: float, step: int, step_dt: float, joint_names: list[str], done: bool) -> dict:
    row = {
        "step": step,
        "t_s": round((step + 1) * step_dt, 4),
        "reward": reward,
        "done": done,
        "base_x": state["root_pos_w"][0].item(),
        "base_y": state["root_pos_w"][1].item(),
        "base_z": state["root_pos_w"][2].item(),
        "quat_w": state["root_quat_w"][0].item(),
        "quat_x": state["root_quat_w"][1].item(),
        "quat_y": state["root_quat_w"][2].item(),
        "quat_z": state["root_quat_w"][3].item(),
        "lin_vel_x": state["root_lin_vel_w"][0].item(),
        "lin_vel_y": state["root_lin_vel_w"][1].item(),
        "lin_vel_z": state["root_lin_vel_w"][2].item(),
        "ang_vel_x": state["root_ang_vel_w"][0].item(),
        "ang_vel_y": state["root_ang_vel_w"][1].item(),
        "ang_vel_z": state["root_ang_vel_w"][2].item(),
        "grav_x": state["projected_gravity_b"][0].item(),
        "grav_y": state["projected_gravity_b"][1].item(),
        "grav_z": state["projected_gravity_b"][2].item(),
    }
    for index, name in enumerate(joint_names):
        row[f"q_{name}"] = state["joint_pos"][index].item()
        row[f"dq_{name}"] = state["joint_vel"][index].item()
        row[f"tau_{name}"] = state["applied_torque"][index].item()
    return row


def _run_command(env, policy, command: tuple[float, float, float], heading_tracking: bool):
    """Run complete episodes for one fixed command and return aggregate metrics."""
    robot_asset = env.unwrapped.scene["robot"]
    total_mass = float(robot_asset.data.default_mass[0].sum())
    step_dt = env.unwrapped.step_dt
    max_steps = int(env.unwrapped.episode_length)
    joint_names = list(robot_asset.data.joint_names)
    episode_metrics = []
    recorded_rows = []

    for _episode_index in range(args_cli.num_episodes):
        # The fixed config already controls resets. Assigning it explicitly
        # also documents the evaluated command in the live generator state.
        env.unwrapped.command_generator.command[:, :3] = torch.tensor(
            command, device=env.unwrapped.device, dtype=torch.float
        )
        if heading_tracking:
            env.unwrapped.command_generator.heading_target[:] = 0.0
            env.unwrapped.command_generator.is_heading_env[:] = True

        obs = env.get_observations()
        start_pos = robot_asset.data.root_pos_w[0].clone()
        ep_return = 0.0
        total_energy_j = 0.0
        linear_tracking_sse = 0.0
        yaw_tracking_sse = 0.0
        path_distance_m = 0.0
        end_pos = start_pos
        previous_pos = start_pos
        terminated_early = False

        for step in range(max_steps):
            evaluated_command = env.unwrapped.command_generator.command[0, :3].clone()
            with torch.no_grad():
                obs, rewards, dones, extras = env.step(policy(obs))
            reward = float(rewards[0])
            ep_return += reward
            done = bool(dones[0].item())
            terminal_state = extras.get("terminal_state") if done else None
            if done and terminal_state is None:
                raise RuntimeError("Missing terminal-state snapshot; evaluation requires capture_terminal_state=True.")
            state = _terminal_or_live_state(robot_asset, terminal_state)

            # Integrate mechanical energy from the physical state before a
            # reset. This fixes the previous final-step reset contamination.
            total_energy_j += float(torch.sum(torch.abs(
                state["applied_torque"] * state["joint_vel"]
            )).item()) * step_dt
            end_pos = state["root_pos_w"].clone()
            path_distance_m += float(torch.linalg.vector_norm(
                (end_pos - previous_pos)[:2]
            ).item())
            previous_pos = end_pos

            # The training reward compares planar velocity in the yaw frame
            # and yaw rate in the world frame. Use the same physical errors,
            # but report their raw squared values rather than shaped reward.
            root_quat = state["root_quat_w"].unsqueeze(0)
            root_lin_vel = state["root_lin_vel_w"].unsqueeze(0)
            velocity_yaw = math_utils.quat_apply_inverse(
                math_utils.yaw_quat(root_quat), root_lin_vel
            )[0]
            command_tensor = evaluated_command.to(device=velocity_yaw.device,
                                                   dtype=velocity_yaw.dtype)
            linear_tracking_sse += float(torch.sum(torch.square(
                velocity_yaw[:2] - command_tensor[:2]
            )).item())
            yaw_tracking_sse += float(torch.square(
                state["root_ang_vel_w"][2] - command_tensor[2]
            ).item())

            if args_cli.record:
                recorded_rows.append(_record_row(state, reward, step, step_dt, joint_names, done))

            if done:
                timed_out = bool(extras.get("time_outs", torch.zeros(1, device=env.unwrapped.device))[0].item())
                terminated_early = not timed_out
                break
        else:
            raise RuntimeError(f"Episode did not terminate within its configured {max_steps} control steps.")

        steps = step + 1
        elapsed_s = steps * step_dt
        displacement_m = float(torch.norm((end_pos - start_pos)[:2]).item())
        mean_power_w = total_energy_j / elapsed_s
        v_actual_m_s = path_distance_m / elapsed_s
        cot = total_energy_j / (total_mass * 9.81 * path_distance_m) if path_distance_m > 0.01 else float("nan")
        episode_metrics.append({
            "episode_return": ep_return,
            "elapsed_s": elapsed_s,
            "displacement_m": displacement_m,
            "distance_m": path_distance_m,
            "energy_j": total_energy_j,
            "mean_power_w": mean_power_w,
            "v_actual_m_s": v_actual_m_s,
            "cot": cot,
            "linear_tracking_mse": linear_tracking_sse / steps,
            "yaw_tracking_mse": yaw_tracking_sse / steps,
            "tracking_cost": (linear_tracking_sse + yaw_tracking_sse) / steps,
            "terminated_early": terminated_early,
        })

    total_time_s = sum(metric["elapsed_s"] for metric in episode_metrics)
    total_distance_m = sum(metric["distance_m"] for metric in episode_metrics)
    total_energy_j = sum(metric["energy_j"] for metric in episode_metrics)
    mean_power_w = total_energy_j / total_time_s
    v_actual_m_s = total_distance_m / total_time_s
    cot = mean_power_w / (total_mass * 9.81 * v_actual_m_s) if v_actual_m_s > 0.01 else float("nan")
    returns = [metric["episode_return"] for metric in episode_metrics]
    avg_return = sum(returns) / len(returns)
    total_steps = sum(round(metric["elapsed_s"] / step_dt) for metric in episode_metrics)
    linear_tracking_mse = sum(
        metric["linear_tracking_mse"] * round(metric["elapsed_s"] / step_dt)
        for metric in episode_metrics
    ) / total_steps
    yaw_tracking_mse = sum(
        metric["yaw_tracking_mse"] * round(metric["elapsed_s"] / step_dt)
        for metric in episode_metrics
    ) / total_steps
    aggregate = {
        "command": _command_key(command),
        "command_vector": list(command),
        "total_return": avg_return,
        "avg_episode_return": avg_return,
        "episode_returns": returns,
        "std_episode_return": math.sqrt(sum((value - avg_return) ** 2 for value in returns) / len(returns)),
        "tracking_cost": linear_tracking_mse + yaw_tracking_mse,
        "linear_tracking_mse": linear_tracking_mse,
        "linear_tracking_rmse_m_s": math.sqrt(linear_tracking_mse),
        "yaw_tracking_mse": yaw_tracking_mse,
        "yaw_tracking_rmse_rad_s": math.sqrt(yaw_tracking_mse),
        "cot": cot,
        "mass_kg": total_mass,
        "mean_power_w": mean_power_w,
        "v_cmd_m_s": math.hypot(command[0], command[1]),
        "v_actual_m_s": v_actual_m_s,
        "total_energy_j": total_energy_j,
        "total_distance_m": total_distance_m,
        "terminated_early_episodes": sum(metric["terminated_early"] for metric in episode_metrics),
    }
    return aggregate, recorded_rows


def _write_record(rows: list[dict], record_path: str, joint_names: list[str]) -> None:
    import csv
    import pandas as pd

    with open(record_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] recorded {len(rows)} terminal-correct steps -> {record_path}", flush=True)

    frame = pd.DataFrame(rows)
    output_dir = record_path.rsplit(".", 1)[0]
    os.makedirs(output_dir, exist_ok=True)
    frame[["t_s", "reward", "done"]].to_csv(f"{output_dir}/reward.csv", index=False)
    base_dir = os.path.join(output_dir, "base_imu")
    os.makedirs(base_dir, exist_ok=True)
    base_columns = [
        "t_s", "base_x", "base_y", "base_z", "quat_w", "quat_x", "quat_y", "quat_z",
        "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
        "grav_x", "grav_y", "grav_z",
    ]
    frame[base_columns].to_csv(f"{base_dir}/base_state.csv", index=False)
    for joint_name in joint_names:
        joint_dir = os.path.join(output_dir, joint_name)
        os.makedirs(joint_dir, exist_ok=True)
        pd.DataFrame({"t_s": frame["t_s"], "position_rad": frame[f"q_{joint_name}"]}).to_csv(
            f"{joint_dir}/position.csv", index=False
        )
        pd.DataFrame({"t_s": frame["t_s"], "velocity_rad_s": frame[f"dq_{joint_name}"]}).to_csv(
            f"{joint_dir}/velocity.csv", index=False
        )
        pd.DataFrame({"t_s": frame["t_s"], "torque_Nm": frame[f"tau_{joint_name}"]}).to_csv(
            f"{joint_dir}/torque.csv", index=False
        )


runner = None
policy = None
command_results = []
recorded_rows = []
record_joint_names = []

try:
    for command in commands:
        env, agent_cfg, heading_tracking = _make_eval_env(command)
        try:
            if runner is None:
                print(f"[eval] Loading checkpoint: {args_cli.checkpoint}", flush=True)
                runner = OnPolicyRunner(
                    env, agent_cfg.to_dict(), log_dir=tempfile.mkdtemp(prefix="eval_log_"), device=agent_cfg.device
                )
                runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                print("[eval] Policy ready", flush=True)

            command_result, rows = _run_command(env, policy, command, heading_tracking)
            command_result["heading_tracking"] = heading_tracking
            command_results.append(command_result)
            if args_cli.record:
                recorded_rows = rows
                record_joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
            print(
                f"[eval] cmd={command_result['command']} return={command_result['avg_episode_return']:.3f} "
                f"track={command_result['tracking_cost']:.4f} cot={command_result['cot']:.4f} "
                f"v_path={command_result['v_actual_m_s']:.3f} m/s "
                f"({args_cli.num_episodes} episode(s))",
                flush=True,
            )
        finally:
            env.close()

    if args_cli.record:
        _write_record(recorded_rows, args_cli.record, record_joint_names)

    avg_episode_return = sum(result["avg_episode_return"] for result in command_results) / len(command_results)
    # This aggregate has a physical interpretation only for translational
    # commands; per-command CoT remains the primary reported metric.
    total_energy_j = sum(result["total_energy_j"] for result in command_results)
    total_distance_m = sum(result["total_distance_m"] for result in command_results)
    total_mass = command_results[0]["mass_kg"]
    aggregate_cot = total_energy_j / (total_mass * 9.81 * total_distance_m) if total_distance_m > 0.01 else float("nan")
    tracking_cost = sum(result["tracking_cost"] for result in command_results) / len(command_results)
    linear_tracking_mse = sum(result["linear_tracking_mse"] for result in command_results) / len(command_results)
    yaw_tracking_mse = sum(result["yaw_tracking_mse"] for result in command_results) / len(command_results)
    result = {
        "avg_episode_return": avg_episode_return,
        "avg_return": avg_episode_return,  # legacy alias for batch tooling
        "episode_returns": command_results[0]["episode_returns"] if len(command_results) == 1 else [],
        "cot": aggregate_cot,
        "tracking_cost": tracking_cost,
        "linear_tracking_mse": linear_tracking_mse,
        "linear_tracking_rmse_m_s": math.sqrt(linear_tracking_mse),
        "yaw_tracking_mse": yaw_tracking_mse,
        "yaw_tracking_rmse_rad_s": math.sqrt(yaw_tracking_mse),
        "terminated_early_episodes": sum(result["terminated_early_episodes"] for result in command_results),
        "total_evaluated_episodes": len(command_results) * args_cli.num_episodes,
        "mass_kg": total_mass,
        "thigh": args_cli.thigh,
        "calf": args_cli.calf,
        "commands": command_results,
    }
    if len(command_results) == 1:
        result.update({
            key: command_results[0][key]
            for key in ("mean_power_w", "v_cmd_m_s", "v_actual_m_s", "total_energy_j", "total_distance_m")
        })
    print(f"RESULT: {json.dumps(result, allow_nan=True)}", flush=True)
finally:
    simulation_app.close()
