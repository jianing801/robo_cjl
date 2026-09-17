"""Bi-level co-design over leg lengths and knee/ankle motor choices.

Outer loop: Optuna Gaussian-process multi-objective minimization of command-tracking error
and mechanical cost of transport (CoT).
Inner loop: co_design_train.py (full training) + co_design_eval.py (deterministic eval).
Crashed, non-finite, or early-terminated trials receive dominated penalty
values so the surrogate learns the infeasible boundary.

STANDALONE — does NOT modify any existing source files.

Prerequisites:
    pip install "optuna>=4.4" scipy torch

Usage:
    python co_design.py --trials 30 --max-iterations 12000 --num-envs 4096
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(HERE, "co_design_train.py")
EVAL_SCRIPT  = os.path.join(HERE, "co_design_eval.py")
PACKAGE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
URDF_MODEL_FILE = os.path.join(
    PACKAGE_ROOT, "robolab", "assets", "motor_data", "motor_urdf_models.json"
)
URDF_GENERATOR_FILE = os.path.join(HERE, "generate_urdf_sw.py")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from robolab.assets.rpo_motor_catalog import (  # noqa: E402
    ANKLE_MOTOR_CANDIDATES,
    KNEE_MOTOR_CANDIDATES,
    get_motor_spec,
    load_torque_speed_envelope,
)

parser = argparse.ArgumentParser(description="Bi-level co-design BO over leg lengths and motor choices.")
parser.add_argument("--trials", type=int, default=30,
                    help="Target number of COMPLETE BO trials in the study.")
parser.add_argument("--max-iterations", type=int, default=12000,
                    help="PPO updates per trial; 12000 writes final model_11999.pt.")
parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs for training.")
parser.add_argument("--eval-episodes", type=int, default=1, help="Evaluation episodes per trial.")
parser.add_argument("--eval-commands", type=str, default="0.5,0,0",
                    help="Semicolon-separated velocity commands for the BO objective. "
                         "Default: 0.5 m/s forward. Pass e.g. "
                         "'0.5,0,0;1.0,0,0;-0.5,0,0' to enable multi-speed evaluation.")
parser.add_argument("--study-name", type=str, default="rpo_flat_track_cot_motor_urdf_v1")
parser.add_argument("--db", type=str, default=None,
                    help="Optuna DB path. Default: <HERE>/co_design_track_cot_motor_urdf_v1.db")
parser.add_argument("--knee-motors", type=str, default=",".join(KNEE_MOTOR_CANDIDATES),
                    help="Comma-separated knee motor IDs used as a categorical search space.")
parser.add_argument("--ankle-motors", type=str, default=",".join(ANKLE_MOTOR_CANDIDATES),
                    help="Comma-separated common ankle motor IDs used as a categorical search space.")
parser.add_argument("--knee-envelope-csv", type=str, default=None,
                    help="Optional MATLAB CSV override used for every selected knee motor.")
parser.add_argument("--ankle-envelope-csv", type=str, default=None,
                    help="Optional MATLAB CSV override used for every selected ankle motor.")
parser.add_argument("--n-startup-trials", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train-timeout", type=int, default=86400, help="Timeout per training run (seconds).")
parser.add_argument("--eval-timeout", type=int, default=7200, help="Timeout per eval run (seconds).")
parser.add_argument("--fail-tracking", type=float, default=1e6,
                    help="Tracking objective assigned to infeasible trials.")
parser.add_argument("--fail-cot", type=float, default=1e6,
                    help="CoT objective assigned to infeasible trials.")
args = parser.parse_args()


def _parse_motor_candidates(spec: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    # dict.fromkeys removes accidental duplicates without changing user order.
    candidates = tuple(dict.fromkeys(value.strip() for value in spec.split(",") if value.strip()))
    if not candidates:
        parser.error(f"{label} motor candidate list must not be empty.")
    try:
        for motor_id in candidates:
            get_motor_spec(motor_id, allowed)
    except ValueError as exc:
        parser.error(str(exc))
    return candidates


def _envelope_signature(motor_id: str, csv_override: str | None) -> str | None:
    """Fingerprint the actual curve values so an Optuna study cannot mix models."""

    points = load_torque_speed_envelope(get_motor_spec(motor_id), csv_override)
    if points is None:
        return None
    payload = json.dumps(points, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_signature(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


knee_motor_candidates = _parse_motor_candidates(
    args.knee_motors, KNEE_MOTOR_CANDIDATES, "Knee"
)
ankle_motor_candidates = _parse_motor_candidates(
    args.ankle_motors, ANKLE_MOTOR_CANDIDATES, "Ankle"
)
if args.knee_envelope_csv and len(knee_motor_candidates) != 1:
    parser.error("--knee-envelope-csv requires exactly one --knee-motors candidate.")
if args.ankle_envelope_csv and len(ankle_motor_candidates) != 1:
    parser.error("--ankle-envelope-csv requires exactly one --ankle-motors candidate.")
if min(args.trials, args.max_iterations, args.num_envs, args.eval_episodes,
       args.n_startup_trials, args.train_timeout, args.eval_timeout) < 1:
    parser.error("Trial, training, evaluation, startup, and timeout counts must be positive.")
if not all(math.isfinite(value) and value > 0.0
           for value in (args.fail_tracking, args.fail_cot)):
    parser.error("Infeasible-design objective values must be positive and finite.")

db_path = args.db or os.path.join(HERE, "co_design_track_cot_motor_urdf_v1.db")
db_parent = os.path.dirname(os.path.abspath(db_path))
os.makedirs(db_parent, exist_ok=True)
for command_text in args.eval_commands.split(";"):
    try:
        command = [float(value) for value in command_text.split(",")]
    except ValueError:
        parser.error(f"Invalid evaluation command: {command_text}")
    if len(command) != 3 or not all(math.isfinite(value) for value in command):
        parser.error(f"Evaluation command must be finite vx,vy,wz: {command_text}")
    if math.hypot(command[0], command[1]) <= 0.0:
        parser.error("Every CoT evaluation command must contain planar translation.")

import optuna


# ── Subprocess runners ──────────────────────────────────────────────────
def run_training(thigh: float, calf: float, knee_motor: str, ankle_motor: str) -> tuple[str, str]:
    """Run co_design_train.py in subprocess. Returns (log_dir, ckpt_path)."""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--urdf-model", "sw",
        "--thigh", str(thigh),
        "--calf", str(calf),
        "--knee-motor", knee_motor,
        "--ankle-motor", ankle_motor,
        "--task", "RPO-Flat",
        "--max-iterations", str(args.max_iterations),
        "--num-envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--headless",
    ]
    if args.knee_envelope_csv:
        cmd += ["--knee-envelope-csv", args.knee_envelope_csv]
    if args.ankle_envelope_csv:
        cmd += ["--ankle-envelope-csv", args.ankle_envelope_csv]
    print(f"[co_design] TRAIN start: thigh={thigh:.4f} calf={calf:.4f} "
          f"knee={knee_motor} ankle={ankle_motor}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.train_timeout)
    elapsed = time.time() - t0

    stdout = result.stdout + result.stderr

    # Parse checkpoint and log dir markers
    ckpt_match = re.search(r"CO_DESIGN_CKPT:\s*(.*)", stdout)
    log_match  = re.search(r"CO_DESIGN_LOG_DIR:\s*(.*)", stdout)

    if result.returncode != 0:
        print(f"[co_design] TRAIN FAILED (rc={result.returncode}) after {elapsed:.0f}s")
        print(f"[co_design] stdout tail: {stdout[-2000:]}")
        raise RuntimeError(f"Training failed for thigh={thigh} calf={calf}")

    ckpt = ckpt_match.group(1).strip() if ckpt_match else ""
    log_dir = log_match.group(1).strip() if log_match else ""
    print(f"[co_design] TRAIN done in {elapsed:.0f}s  ckpt={ckpt}")
    return log_dir, ckpt


def extract_train_reward(log_dir: str) -> float:
    """Extract final mean training reward from tensorboard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for f in os.listdir(log_dir):
            if f.startswith("events.out"):
                ea = EventAccumulator(os.path.join(log_dir, f))
                ea.Reload()
                events = ea.Scalars("Train/mean_reward")
                if events:
                    vals = [e.value for e in events[-10:]]
                    return sum(vals) / len(vals)
    except Exception:
        pass
    return float("nan")


def run_evaluation(ckpt_path: str, thigh: float, calf: float,
                   knee_motor: str, ankle_motor: str) -> dict:
    """Evaluate all fixed commands and return physical metrics."""
    cmd = [sys.executable, EVAL_SCRIPT,
           "--urdf-model", "sw", "--checkpoint", ckpt_path,
           "--thigh", str(thigh), "--calf", str(calf), "--task", "RPO-Flat",
           "--knee-motor", knee_motor, "--ankle-motor", ankle_motor,
           "--num-episodes", str(args.eval_episodes), "--num-envs", "1",
           "--seed", str(args.seed),
           f"--command={args.eval_commands}", "--headless"]
    if args.knee_envelope_csv:
        cmd += ["--knee-envelope-csv", args.knee_envelope_csv]
    if args.ankle_envelope_csv:
        cmd += ["--ankle-envelope-csv", args.ankle_envelope_csv]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.eval_timeout)
    elapsed = time.time() - t0
    stdout = result.stdout + result.stderr
    matches = re.findall(r"^RESULT:\s*(.*)$", stdout, re.MULTILINE)
    if result.returncode != 0 or not matches:
        print(f"[co_design] EVAL FAILED (rc={result.returncode})")
        print(f"[co_design] stdout tail: {stdout[-2000:]}")
        raise RuntimeError(f"Evaluation failed for thigh={thigh} calf={calf}")
    data = json.loads(matches[-1])
    print(f"[co_design] EVAL done in {elapsed:.0f}s  tracking={data['tracking_cost']:.6f} "
          f"CoT={data['cot']:.6f} return(diag)={data['avg_episode_return']:.3f}")
    return data


# ── Optuna objective ────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> tuple[float, float]:
    thigh = trial.suggest_float("thigh_length", 0.25, 0.40)
    calf  = trial.suggest_float("calf_length", 0.30, 0.45)
    knee_motor = trial.suggest_categorical("knee_motor", knee_motor_candidates)
    ankle_motor = trial.suggest_categorical("ankle_motor", ankle_motor_candidates)

    print(f"\n{'='*60}")
    print(f"[co_design] Trial {trial.number}: thigh={thigh:.4f} calf={calf:.4f} "
          f"knee={knee_motor} ankle={ankle_motor}")
    print(f"{'='*60}")

    try:
        log_dir, ckpt_path = run_training(thigh, calf, knee_motor, ankle_motor)
        evaluation = run_evaluation(ckpt_path, thigh, calf, knee_motor, ankle_motor)
        train_reward = extract_train_reward(log_dir)
    except Exception as e:
        # Infeasible design (training/eval crashed or timed out). Return the
        # penalty value instead of letting the trial FAIL with no value, so
        # the BO surrogate learns the infeasible boundary instead of wasting
        # the trial slot.
        print(f"[co_design] Trial {trial.number} INFEASIBLE: {type(e).__name__}: {e}")
        trial.set_user_attr("thigh_length", thigh)
        trial.set_user_attr("calf_length", calf)
        trial.set_user_attr("knee_motor", knee_motor)
        trial.set_user_attr("ankle_motor", ankle_motor)
        trial.set_user_attr("fail_reason", f"{type(e).__name__}: {str(e)[:300]}")
        return args.fail_tracking, args.fail_cot

    tracking = float(evaluation["tracking_cost"])
    cot = float(evaluation["cot"])
    terminated = int(evaluation["terminated_early_episodes"])
    feasible = math.isfinite(tracking) and math.isfinite(cot) and tracking >= 0.0 and cot >= 0.0 and terminated == 0
    if not feasible:
        tracking, cot = args.fail_tracking, args.fail_cot
    print(f"[co_design]  train_reward(diag)={train_reward:.3f} tracking={tracking:.6f} "
          f"CoT={cot:.6f} feasible={feasible}")

    trial.set_user_attr("thigh_length", thigh)
    trial.set_user_attr("calf_length", calf)
    trial.set_user_attr("knee_motor", knee_motor)
    trial.set_user_attr("ankle_motor", ankle_motor)
    trial.set_user_attr("ckpt_path", ckpt_path)
    trial.set_user_attr("log_dir", log_dir)
    trial.set_user_attr("evaluation_return_diagnostic", evaluation["avg_episode_return"])
    trial.set_user_attr("tracking_cost", evaluation["tracking_cost"])
    trial.set_user_attr("linear_tracking_rmse_m_s", evaluation["linear_tracking_rmse_m_s"])
    trial.set_user_attr("yaw_tracking_rmse_rad_s", evaluation["yaw_tracking_rmse_rad_s"])
    trial.set_user_attr("mechanical_cot", evaluation["cot"])
    trial.set_user_attr("terminated_early_episodes", terminated)
    trial.set_user_attr("feasible", feasible)
    trial.set_user_attr("train_reward", train_reward)

    return tracking, cot


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[co_design] Config:")
    print(f"  trials={args.trials}")
    print(f"  max-iterations={args.max_iterations}")
    print(f"  num-envs={args.num_envs}")
    print(f"  eval-episodes={args.eval_episodes}")
    print(f"  study={args.study_name}")
    print(f"  db={db_path}")
    print(f"  n-startup-trials={args.n_startup_trials}")
    print(f"  knee-motors={knee_motor_candidates}")
    print(f"  ankle-motors={ankle_motor_candidates}")
    print(f"  objectives=minimize tracking cost, minimize mechanical CoT")

    from packaging.version import Version
    if Version(optuna.__version__) < Version("4.4.0"):
        raise RuntimeError("Multi-objective GPSampler requires optuna>=4.4.0.")
    sampler = optuna.samplers.GPSampler(
        seed=args.seed, n_startup_trials=args.n_startup_trials
    )
    print("[co_design] Using multi-objective GP sampler (logEHVI)")

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        storage=f"sqlite:///{db_path}",
        study_name=args.study_name,
        load_if_exists=True,
    )

    model_protocol = {"urdf_model": "sw", "thigh_range": [0.25, 0.40],
                      "calf_range": [0.30, 0.45],
                      "knee_motors": list(knee_motor_candidates),
                      "ankle_motors": list(ankle_motor_candidates),
                      "motor_urdf_model_sha256": _file_signature(URDF_MODEL_FILE),
                      "motor_urdf_generator_sha256": _file_signature(URDF_GENERATOR_FILE),
                      "knee_urdf_models": {
                          motor_id: get_motor_spec(motor_id).urdf_model_id
                          for motor_id in knee_motor_candidates
                      },
                      "ankle_urdf_models": {
                          motor_id: get_motor_spec(motor_id).urdf_model_id
                          for motor_id in ankle_motor_candidates
                      },
                      "knee_armature_kgm2": {
                          motor_id: get_motor_spec(motor_id).armature_kgm2
                          for motor_id in knee_motor_candidates
                      },
                      "ankle_armature_kgm2": {
                          motor_id: get_motor_spec(motor_id).armature_kgm2
                          for motor_id in ankle_motor_candidates
                      },
                      "knee_envelope_sha256": {
                          motor_id: _envelope_signature(motor_id, args.knee_envelope_csv)
                          for motor_id in knee_motor_candidates
                      },
                      "ankle_envelope_sha256": {
                          motor_id: _envelope_signature(motor_id, args.ankle_envelope_csv)
                          for motor_id in ankle_motor_candidates
                      },
                      "objectives": ["tracking_cost_min", "mechanical_cot_min"],
                      "eval_commands": args.eval_commands,
                      "eval_episodes": args.eval_episodes,
                      "seed": args.seed}
    if study.trials and study.user_attrs.get("model_protocol") != model_protocol:
        raise ValueError(
            "URDF, motor armature, curve, or evaluation protocol differs or is unrecorded; "
            "use a new --db or --study-name."
        )
    study.set_user_attr("model_protocol", model_protocol)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    remaining = max(0, args.trials - len(completed))
    print(f"\n[co_design] Target COMPLETE trials: {args.trials}")
    print(f"[co_design] Existing trials: total={len(study.trials)} complete={len(completed)} "
          f"remaining={remaining}")

    try:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=None,
        )
    except KeyboardInterrupt:
        print("\n[co_design] Interrupted. Study saved to DB. Resume by re-running the same command.")

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[co_design] Optimization complete!")
    feasible_front = [trial for trial in study.best_trials if trial.user_attrs.get("feasible")]
    print(f"  Feasible Pareto designs: {len(feasible_front)}")
    print(f"  Study DB: {db_path}")
    print(f"{'='*60}")

    print("\nPareto front (both lower is better):")
    feasible_front.sort(key=lambda trial: trial.values[0])
    for trial in feasible_front:
        print(f"  trial={trial.number} thigh={trial.params['thigh_length']:.4f} "
              f"calf={trial.params['calf_length']:.4f} knee={trial.params['knee_motor']} "
              f"ankle={trial.params['ankle_motor']} tracking={trial.values[0]:.6f} "
              f"CoT={trial.values[1]:.6f}")

    import csv
    pareto_path = os.path.splitext(db_path)[0] + "_pareto.csv"
    with open(pareto_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "trial", "thigh_length_m", "calf_length_m", "knee_motor", "ankle_motor", "tracking_cost",
            "mechanical_cot", "linear_tracking_rmse_m_s", "yaw_tracking_rmse_rad_s",
            "checkpoint",
        ])
        writer.writeheader()
        for trial in feasible_front:
            writer.writerow({
                "trial": trial.number,
                "thigh_length_m": trial.params["thigh_length"],
                "calf_length_m": trial.params["calf_length"],
                "knee_motor": trial.params["knee_motor"],
                "ankle_motor": trial.params["ankle_motor"],
                "tracking_cost": trial.values[0], "mechanical_cot": trial.values[1],
                "linear_tracking_rmse_m_s": trial.user_attrs.get("linear_tracking_rmse_m_s"),
                "yaw_tracking_rmse_rad_s": trial.user_attrs.get("yaw_tracking_rmse_rad_s"),
                "checkpoint": trial.user_attrs.get("ckpt_path"),
            })
    print(f"  Pareto CSV: {pareto_path}")
