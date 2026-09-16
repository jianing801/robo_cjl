"""Parameterized training script for RPO co-design BO inner loop.

Accepts thigh/calf length parameters, generates a custom URDF, injects
it into the RPO_CFG, then runs full RSL-RL training.

STANDALONE — does NOT modify train.py or any existing source file.

Usage:
    python co_design_train.py --thigh 0.25 --calf 0.30 \
        --task RPO-Flat --max-iterations 12000 --num-envs 4096 --headless
"""

import argparse
import os
import sys
import tempfile
import re
import logging
from datetime import datetime

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from robolab.assets.rpo_motor_catalog import (  # noqa: E402
    ANKLE_MOTOR_CHOICES,
    DEFAULT_ANKLE_MOTOR,
    DEFAULT_KNEE_MOTOR,
    KNEE_MOTOR_CHOICES,
)

from isaaclab.app import AppLauncher

# local imports — add rsl_rl scripts dir to path for cli_args
_RSL_RL_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "rsl_rl")
sys.path.insert(0, os.path.abspath(_RSL_RL_SCRIPTS))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Co-design parameterized training.")
parser.add_argument("--urdf-model", choices=["sw", "legacy"], default="sw",
                    help="URDF parameter model; SW fits are the default.")
parser.add_argument("--thigh", type=float, required=True, help="Thigh length (m).")
parser.add_argument("--calf", type=float, required=True, help="Calf length (m).")
parser.add_argument("--knee-motor", choices=KNEE_MOTOR_CHOICES, default=DEFAULT_KNEE_MOTOR,
                    help="Knee motor ID; applied symmetrically to both knees.")
parser.add_argument("--ankle-motor", choices=ANKLE_MOTOR_CHOICES, default=DEFAULT_ANKLE_MOTOR,
                    help="Common motor ID for all four ankle pitch/roll joints.")
parser.add_argument("--knee-envelope-csv", type=str, default=None,
                    help="Optional MATLAB torque-speed CSV overriding the selected knee motor curve.")
parser.add_argument("--ankle-envelope-csv", type=str, default=None,
                    help="Optional MATLAB torque-speed CSV overriding the selected ankle motor curve.")
parser.add_argument("--task", type=str, default="RPO-Flat")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-iterations", type=int, default=12000,
                    help="Number of PPO updates; 12000 writes final model_11999.pt.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video-length", type=int, default=200)
parser.add_argument("--video-interval", type=int, default=2000)
parser.add_argument("--distributed", action="store_true", default=False, help="Multi-GPU training.")
parser.add_argument("--early-check", type=int, default=0, help="Abort training after this many iters if mean reward is below --early-threshold (0=disabled).")
parser.add_argument("--early-threshold", type=float, default=0.0, help="Fail threshold: runs below this mean reward at --early-check are considered diverged/crashed and aborted.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Auto-enable distributed for torchrun
if int(os.getenv("WORLD_SIZE", "1")) > 1:
    args_cli.distributed = True
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Version check ───────────────────────────────────────────────────────
import importlib.metadata as metadata
import platform
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(f"ERROR: rsl-rl-lib >= {RSL_RL_VERSION} required, got {installed_version}")
    exit(1)

# ── Imports after Isaac Sim startup ─────────────────────────────────────
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner, DistillationRunner, AMPRunner

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import robolab.tasks  # noqa: F401

# ── Parameterized URDF ──────────────────────────────────────────────────
from generate_urdf import generate_urdf
if args_cli.urdf_model == "sw":
    from generate_urdf_sw import generate_urdf
from robolab.assets.robots import RPO_CFG, configure_rpo_motors

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ── Helper: checkpoint path resolution (same logic as train.py) ────────
def _get_resume_checkpoint_path(log_path, run_dir, checkpoint):
    runs = [os.path.join(log_path, r.name)
            for r in os.scandir(log_path) if r.is_dir() and re.match(run_dir, r.name)]
    if not runs:
        raise ValueError(f"No runs in '{log_path}' matching '{run_dir}'.")
    runs = sorted(runs, key=os.path.getmtime)
    for run_path in reversed(runs):
        ckpts = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
        if ckpts:
            # model_11999.pt is newer than model_9000.pt; lexicographic
            # ordering gets this wrong once checkpoint numbers have different
            # digit widths.
            ckpts.sort(key=lambda name: int(re.search(r"model_(\d+)\.pt", name).group(1)))
            return os.path.join(run_path, ckpts[-1])
    raise ValueError(f"No checkpoint in '{log_path}' matching '{checkpoint}'.")


def _resolve_log_dir(log_root, resume, load_run, load_checkpoint, run_name):
    os.makedirs(log_root, exist_ok=True)
    resume_path = None
    if resume:
        resume_path = _get_resume_checkpoint_path(log_root, load_run, load_checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")

    log_run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        log_run_name += f"_{run_name}"
    print(f"Exact experiment name requested from command line: {log_run_name}")
    log_dir = os.path.join(log_root, log_run_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, resume_path


def _read_mean_reward(log_dir: str, n_last: int = 10) -> float:
    """Read mean training reward from tensorboard, average last N values."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for f in os.listdir(log_dir):
            if f.startswith("events.out"):
                ea = EventAccumulator(os.path.join(log_dir, f))
                ea.Reload()
                events = ea.Scalars("Train/mean_reward")
                if events and len(events) >= n_last:
                    vals = [e.value for e in events[-n_last:]]
                    return sum(vals) / len(vals)
                elif events:
                    vals = [e.value for e in events]
                    return sum(vals) / len(vals)
    except Exception:
        pass
    return float("-inf")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    """Train with parameterized URDF."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if hasattr(env_cfg, "scene_context"):
        env_cfg.scene_context.num_envs = env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = 2.5
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    # rsl-rl compat
    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.headless:
        env_cfg.commands.debug_vis = False

    # Force tensorboard logger to avoid wandb auth issues in headless co-design
    agent_cfg.logger = "tensorboard"
    # Always save at least the final checkpoint
    agent_cfg.save_interval = min(agent_cfg.save_interval, agent_cfg.max_iterations)

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    global_rank = getattr(app_launcher, "global_rank", 0)

    # ── Logging ─────────────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging to: {log_root_path}")

    should_resume = agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
    log_dir, resume_path = _resolve_log_dir(
        log_root_path, resume=should_resume,
        load_run=agent_cfg.load_run, load_checkpoint=agent_cfg.load_checkpoint,
        run_name=agent_cfg.run_name or None,
    )
    env_cfg.log_dir = log_dir

    # ── Generate parameterized URDF + inject ────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="rpo_train_")
    urdf_path = generate_urdf(args_cli.thigh, args_cli.calf, output_dir=tmp_dir)
    base_z = 0.75 + (args_cli.thigh + args_cli.calf - 0.55)

    custom_robot_cfg = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    custom_robot_cfg.spawn.asset_path = urdf_path
    custom_robot_cfg.spawn.usd_dir = os.path.join(tmp_dir, "usd")
    custom_robot_cfg.init_state.pos = (0.0, 0.0, base_z)
    motor_selection = configure_rpo_motors(
        custom_robot_cfg,
        knee_motor=args_cli.knee_motor,
        ankle_motor=args_cli.ankle_motor,
        knee_envelope_csv=args_cli.knee_envelope_csv,
        ankle_envelope_csv=args_cli.ankle_envelope_csv,
    )

    if hasattr(env_cfg, "scene_context"):
        env_cfg.scene_context.robot = custom_robot_cfg
    elif hasattr(env_cfg.scene, "robot"):
        env_cfg.scene.robot = custom_robot_cfg

    # Rebuild scene with custom robot cfg
    if hasattr(env_cfg, "scene") and hasattr(env_cfg, "scene_context"):
        from robolab.tasks.direct.base.scene_cfg import SceneCfg
        env_cfg.scene = SceneCfg(
            config=env_cfg.scene_context,
            physics_dt=env_cfg.sim.dt,
            step_dt=env_cfg.decimation * env_cfg.sim.dt,
        )

    print(f"[co_design_train] thigh={args_cli.thigh:.4f}m  calf={args_cli.calf:.4f}m  base_z={base_z:.3f}m")
    print(
        f"[co_design_train] motors: knee={args_cli.knee_motor} "
        f"({motor_selection['knee']['peak_torque_nm']:.3f} Nm, "
        f"{motor_selection['knee']['max_speed_rad_s']:.3f} rad/s, "
        f"dynamic_envelope={motor_selection['knee']['dynamic_envelope']}), "
        f"ankle={args_cli.ankle_motor} "
        f"({motor_selection['ankle']['peak_torque_nm']:.3f} Nm, "
        f"{motor_selection['ankle']['max_speed_rad_s']:.3f} rad/s, "
        f"dynamic_envelope={motor_selection['ankle']['dynamic_envelope']})"
    )
    print(f"[co_design_train] URDF: {urdf_path}")

    # ── Create env ──────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO] Loading model from: {resume_path}")
        runner.load(resume_path, map_location=agent_cfg.device)

    if not args_cli.distributed or global_rank == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        # Keep the selected motor IDs and the exact actuator-side limits next
        # to the checkpoint.  URDF mass/inertia are intentionally unchanged.
        dump_yaml(os.path.join(log_dir, "params", "motors.yaml"), motor_selection)

    # ── Train (with early stop) ───────────────────────────────────────
    early_check = args_cli.early_check
    early_threshold = args_cli.early_threshold

    if early_check > 0 and early_check < agent_cfg.max_iterations:
        # Phase 1: train early_check iterations
        runner.learn(num_learning_iterations=early_check, init_at_random_ep_len=True)
        # Check reward from tensorboard
        mean_r = _read_mean_reward(log_dir)
        print(f"[co_design_train] After {early_check} iters: mean_reward={mean_r:.2f}")
        # Abort diverged/crashed runs early to save GPU time. Promising runs
        # (mean_r >= threshold) MUST train to max_iterations: truncating them
        # biases the BO ranking against high-potential designs.
        if mean_r < early_threshold:
            print(f"[co_design_train] EARLY ABORT! reward={mean_r:.2f} < {early_threshold} (diverged/crashed)")
        else:
            remaining = agent_cfg.max_iterations - early_check
            print(f"[co_design_train] Continue to {agent_cfg.max_iterations} (reward={mean_r:.2f} >= {early_threshold})")
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()

    # ── Find final checkpoint for eval script ───────────────────────────
    ckpts = [f for f in os.listdir(log_dir) if re.match(r"model_.*\.pt", f)]
    if ckpts:
        # sort by iteration number (string sort would put model_9000 after model_11999)
        ckpts.sort(key=lambda m: int(re.search(r"model_(\d+)\.pt", m).group(1)))
        final_ckpt = os.path.join(log_dir, ckpts[-1])
    else:
        # fallback: scan parent for newest run's checkpoint
        final_ckpt = _get_resume_checkpoint_path(os.path.dirname(log_dir), ".*", r"model_.*\.pt")
    # Print special marker line for BO script to parse
    print(f"CO_DESIGN_CKPT: {final_ckpt}")
    print(f"CO_DESIGN_LOG_DIR: {log_dir}")


if __name__ == "__main__":
    # Monkey-patch setuptools (same as train.py)
    import setuptools
    import distutils.core
    setuptools.setup = lambda *a, **kw: print("[Bypassed] setuptools.setup skipped.")
    distutils.core.setup = lambda *a, **kw: print("[Bypassed] distutils.core.setup skipped.")

    # Auto-fix ActorCritic/PPO path resolution
    import rsl_rl.runners.on_policy_runner as opr
    if hasattr(opr, "resolve_callable"):
        orig_resolve = opr.resolve_callable
        def smart_resolve(x):
            mapping = {
                "ActorCritic": "rsl_rl.modules.actor_critic:ActorCritic",
                "PPO": "rsl_rl.algorithms:PPO"
            }
            return orig_resolve(mapping.get(x, x))
        opr.resolve_callable = smart_resolve

    main()
    simulation_app.close()
