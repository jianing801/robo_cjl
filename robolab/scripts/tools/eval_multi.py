"""Multi-command evaluation for a co-design checkpoint."""
import argparse, json, os, sys, tempfile
import torch
import gymnasium as gym

_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from robolab.assets.rpo_motor_catalog import (
    ANKLE_MOTOR_CHOICES,
    DEFAULT_ANKLE_MOTOR,
    DEFAULT_KNEE_MOTOR,
    KNEE_MOTOR_CHOICES,
)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--thigh", type=float, required=True)
parser.add_argument("--calf", type=float, required=True)
parser.add_argument("--urdf-model", choices=["sw", "legacy"], default="sw")
parser.add_argument("--knee-motor", choices=KNEE_MOTOR_CHOICES, default=DEFAULT_KNEE_MOTOR)
parser.add_argument("--ankle-motor", choices=ANKLE_MOTOR_CHOICES, default=DEFAULT_ANKLE_MOTOR)
parser.add_argument("--knee-envelope-csv", type=str, default=None)
parser.add_argument("--ankle-envelope-csv", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]

import setuptools, distutils.core
setuptools.setup = lambda *a, **kw: None
distutils.core.setup = lambda *a, **kw: None

import rsl_rl.runners.on_policy_runner as opr
if hasattr(opr, "resolve_callable"):
    _orig = opr.resolve_callable
    opr.resolve_callable = lambda x: _orig({"ActorCritic":"rsl_rl.modules.actor_critic:ActorCritic","PPO":"rsl_rl.algorithms:PPO"}.get(x,x))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks; import robolab.tasks
from generate_urdf import generate_urdf
if args_cli.urdf_model == "sw":
    from generate_urdf_sw import generate_urdf
from robolab.assets.robots import RPO_CFG, configure_rpo_motors
from robolab.tasks.direct.base.scene_cfg import SceneCfg
from packaging import version
iv = __import__('importlib').metadata.version("rsl-rl-lib")

env_cfg = load_cfg_from_registry("RPO-Flat", "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry("RPO-Flat", "rsl_rl_cfg_entry_point")
if version.parse(iv) < version.parse("5.0.0"):
    for k in ("optimizer", "share_cnn_encoders"):
        if hasattr(agent_cfg.algorithm, k): delattr(agent_cfg.algorithm, k)

tmp_dir = tempfile.mkdtemp(prefix="eval_multi_")
urdf_kwargs = {}
if args_cli.urdf_model == "sw":
    urdf_kwargs = {
        "knee_motor": args_cli.knee_motor,
        "ankle_motor": args_cli.ankle_motor,
    }
urdf_path = generate_urdf(
    args_cli.thigh, args_cli.calf, output_dir=tmp_dir, **urdf_kwargs
)
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
print(
    f"[eval] motors: knee={args_cli.knee_motor} "
    f"({motor_selection['knee']['peak_torque_nm']:.3f} Nm, "
    f"armature={motor_selection['knee']['armature_kgm2']:.5f} kg*m^2, "
    f"dynamic={motor_selection['knee']['dynamic_envelope']}), "
    f"ankle={args_cli.ankle_motor} "
    f"({motor_selection['ankle']['peak_torque_nm']:.3f} Nm, "
    f"armature={motor_selection['ankle']['armature_kgm2']:.5f} kg*m^2, "
    f"dynamic={motor_selection['ankle']['dynamic_envelope']})"
)

env_cfg.seed = args_cli.seed
env_cfg.scene.num_envs = 1
env_cfg.scene.env_spacing = 2.5
env_cfg.scene_context.num_envs = 1
env_cfg.noise.add_noise = False
env_cfg.events.push_robot = None
env_cfg.episode_length_s = 20.0
env_cfg.commands.heading_command = False
env_cfg.commands.rel_standing_envs = 0.0
if args_cli.headless:
    env_cfg.commands.debug_vis = False
env_cfg.scene_context.robot = custom_robot_cfg
env_cfg.scene = SceneCfg(config=env_cfg.scene_context, physics_dt=env_cfg.sim.dt, step_dt=env_cfg.decimation * env_cfg.sim.dt)

env = gym.make("RPO-Flat", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"[eval] Loading {args_cli.checkpoint}")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=tempfile.mkdtemp(prefix="eval_log_"), device=agent_cfg.device)
runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
policy = runner.get_inference_policy(device=env.unwrapped.device)

commands = [
    ("fwd_0.5", 0.5, 0.0, 0.0),
    ("fwd_1.0", 1.0, 0.0, 0.0),
    ("bwd_0.5", -0.5, 0.0, 0.0),
    ("turn_L", 0.0, 0.0, 1.0),
    ("turn_R", 0.0, 0.0, -1.0),
    ("side_L", 0.0, 0.5, 0.0),
]

results = []
for name, lx, ly, az in commands:
    env.unwrapped.command_generator.command[:, 0] = lx
    env.unwrapped.command_generator.command[:, 1] = ly
    env.unwrapped.command_generator.command[:, 2] = az
    obs = env.get_observations()
    ep_return, steps = 0.0, 0
    prev_energy, fell = None, False
    done_early = False
    while True:
        with torch.inference_mode():
            obs, rewards, dones, extras = env.step(policy(obs))
        ep_return += float(rewards[0]); steps += 1
        if dones[0]:
            time_out = extras.get("time_outs", torch.zeros(1))[0] if extras else False
            if not time_out:
                fell = True; done_early = True
            break
        log = env.unwrapped.extras.get("log", {})
        cur = log.get("Episode_Reward/energy")
        if cur is not None and prev_energy is not None and cur > prev_energy:
            break
        prev_energy = cur
    avg_r = ep_return / max(steps, 1)
    tag = "FELL" if fell else ""
    print(f"  {name:10s}: return={ep_return:8.1f}  steps={steps:4d}  avg={avg_r:.3f}  {tag}")
    results.append({"command": name, "lx": lx, "ly": ly, "az": az,
                    "total_return": ep_return, "steps": steps,
                    "avg_return": avg_r, "fell": fell})

env.close()
print(f"\n=== SUMMARY ===")
print(f"{'cmd':>10s}  {'rtn':>8s}  {'avg':>6s}  {'steps':>5s}  {'status'}")
for r in results:
    s = "FELL" if r["fell"] else "OK"
    print(f"{r['command']:>10s}  {r['total_return']:8.1f}  {r['avg_return']:6.3f}  {r['steps']:5d}  {s}")
print(f"\nOverall avg return: {sum(r['avg_return'] for r in results)/len(results):.3f}")
simulation_app.close()
