"""Bi-level co-design: Bayesian Optimization over (thigh, calf) with RL inner loop.

Outer loop: Optuna + BoTorch GP single-objective maximization of the averaged
multi-command episode return.
Inner loop: co_design_train.py (full training) + co_design_eval.py (deterministic eval).
Crashed/timed-out trials are penalized with --fail-value instead of failing
without a value, so the surrogate learns the infeasible boundary.

STANDALONE — does NOT modify any existing source files.

Prerequisites:
    pip install botorch gpytorch optuna

Usage:
    python co_design.py --trials 30 --max-iterations 12000 --num-envs 4096
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(HERE, "co_design_train.py")
EVAL_SCRIPT  = os.path.join(HERE, "co_design_eval.py")

parser = argparse.ArgumentParser(description="Bi-level co-design BO over leg lengths.")
parser.add_argument("--trials", type=int, default=30,
                    help="Target number of COMPLETE BO trials in the study.")
parser.add_argument("--max-iterations", type=int, default=12000,
                    help="PPO updates per trial; 12000 writes final model_11999.pt.")
parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs for training.")
parser.add_argument("--eval-episodes", type=int, default=1, help="Evaluation episodes per trial.")
parser.add_argument("--eval-commands", type=str, default="1.0,0,0",
                    help="Semicolon-separated velocity commands for the BO objective. "
                         "Default: single 1.0 m/s forward command. Pass e.g. "
                         "'0.5,0,0;1.0,0,0;-0.5,0,0' to enable multi-speed evaluation.")
parser.add_argument("--study-name", type=str, default="rpo_flat_leg_co_design_sw")
parser.add_argument("--db", type=str, default=None, help="Optuna DB path. Default: <HERE>/co_design_sw_study.db")
parser.add_argument("--n-startup-trials", type=int, default=5)
parser.add_argument("--train-timeout", type=int, default=86400, help="Timeout per training run (seconds).")
parser.add_argument("--eval-timeout", type=int, default=7200, help="Timeout per eval run (seconds).")
parser.add_argument("--fail-value", type=float, default=-200.0,
                    help="Objective value returned when a trial's training/eval crashes or times out "
                         "(infeasible design). -200 matches the fall-termination penalty floor.")
args = parser.parse_args()

db_path = args.db or os.path.join(HERE, "co_design_sw_study.db")

import optuna


# ── Subprocess runners ──────────────────────────────────────────────────
def run_training(thigh: float, calf: float) -> tuple[str, str]:
    """Run co_design_train.py in subprocess. Returns (log_dir, ckpt_path)."""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--urdf-model", "sw",
        "--thigh", str(thigh),
        "--calf", str(calf),
        "--task", "RPO-Flat",
        "--max-iterations", str(args.max_iterations),
        "--num-envs", str(args.num_envs),
        "--headless",
    ]
    print(f"[co_design] TRAIN start: thigh={thigh:.4f} calf={calf:.4f}")
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


def run_evaluation(ckpt_path: str, thigh: float, calf: float) -> float:
    """Run co_design_eval.py once per command, average across commands."""
    cmd_returns = {}
    for cmd_str in args.eval_commands.split(";"):
        cmd_parts = [sys.executable, EVAL_SCRIPT,
                     "--urdf-model", "sw",
                     "--checkpoint", ckpt_path,
                     "--thigh", str(thigh), "--calf", str(calf),
                     "--task", "RPO-Flat",
                     "--num-episodes", str(args.eval_episodes),
                     "--num-envs", "1",
                     f"--command={cmd_str.strip()}",
                     "--headless"]
        t0 = time.time()
        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=args.eval_timeout)
        elapsed = time.time() - t0
        stdout = result.stdout + result.stderr
        match = re.search(r"RESULT:\s*(.*)", stdout)
        if result.returncode != 0 or not match:
            print(f"[co_design] EVAL FAILED for cmd={cmd_str} (rc={result.returncode})")
            print(f"[co_design] stdout tail: {stdout[-2000:]}")
            raise RuntimeError(f"Evaluation failed for thigh={thigh} calf={calf} cmd={cmd_str}")
        data = json.loads(match.group(1))
        r = data.get("avg_episode_return", float("nan"))
        cmd_returns[cmd_str.strip()] = r
        print(f"[co_design]   cmd={cmd_str.strip()}  return={r:.3f}  ({elapsed:.0f}s)")

    returns = list(cmd_returns.values())
    avg_return = sum(returns) / len(returns)
    min_return = min(returns)
    print(f"[co_design] EVAL done  avg={avg_return:.3f}  min={min_return:.3f}  per_cmd={cmd_returns}")
    return avg_return


# ── Optuna objective ────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    thigh = trial.suggest_float("thigh_length", 0.25, 0.40)
    calf  = trial.suggest_float("calf_length", 0.30, 0.45)

    print(f"\n{'='*60}")
    print(f"[co_design] Trial {trial.number}: thigh={thigh:.4f}  calf={calf:.4f}")
    print(f"{'='*60}")

    try:
        log_dir, ckpt_path = run_training(thigh, calf)
        ep_return = run_evaluation(ckpt_path, thigh, calf)
        train_reward = extract_train_reward(log_dir)
    except Exception as e:
        # Infeasible design (training/eval crashed or timed out). Return the
        # penalty value instead of letting the trial FAIL with no value, so
        # the BO surrogate learns the infeasible boundary instead of wasting
        # the trial slot.
        print(f"[co_design] Trial {trial.number} INFEASIBLE: {type(e).__name__}: {e}")
        trial.set_user_attr("thigh_length", thigh)
        trial.set_user_attr("calf_length", calf)
        trial.set_user_attr("fail_reason", f"{type(e).__name__}: {str(e)[:300]}")
        return args.fail_value

    print(f"[co_design]  train_reward={train_reward:.3f}  eval_return={ep_return:.3f}")

    trial.set_user_attr("thigh_length", thigh)
    trial.set_user_attr("calf_length", calf)
    trial.set_user_attr("ckpt_path", ckpt_path)
    trial.set_user_attr("log_dir", log_dir)
    trial.set_user_attr("episode_return", ep_return)
    trial.set_user_attr("train_reward", train_reward)

    # Maximize episode return. Falling naturally gives low reward.
    return ep_return


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
    print(f"  fail-value={args.fail_value}")

    # Try GP sampler first, fallback to TPE
    try:
        sampler = optuna.samplers.GPSampler(n_startup_trials=args.n_startup_trials)
        print("[co_design] Using GP sampler")
    except (ImportError, Exception) as e:
        sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_startup_trials)
        print(f"[co_design] GP not available ({e}), using TPE sampler")

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{db_path}",
        study_name=args.study_name,
        load_if_exists=True,
    )

    model_protocol = {"urdf_model": "sw", "thigh_range": [0.25, 0.40],
                      "calf_range": [0.30, 0.45]}
    if study.trials and study.user_attrs.get("model_protocol") != model_protocol:
        raise ValueError("URDF model/ranges differ or are unrecorded; use a new --db or --study-name.")
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
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best value (avg episode return): {study.best_value:.6f}")
    print(f"  Best params: thigh={study.best_trial.params['thigh_length']:.4f}m"
          f"  calf={study.best_trial.params['calf_length']:.4f}m")
    if study.best_trial.user_attrs.get("ckpt_path"):
        print(f"  Best checkpoint: {study.best_trial.user_attrs['ckpt_path']}")
    print(f"  Study DB: {db_path}")
    print(f"{'='*60}")

    # Show top-5
    print("\nTop-5 trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    # Study direction is maximize: best trials first (highest value).
    completed.sort(key=lambda t: t.value, reverse=True)
    for i, t in enumerate(completed[:5]):
        _er = t.user_attrs.get("episode_return", float("nan"))
        _tr = t.user_attrs.get("train_reward", float("nan"))
        print(f"  #{i+1}  thigh={t.params['thigh_length']:.4f}  calf={t.params['calf_length']:.4f}"
              f"  value={t.value:.6f}  train_r={_tr:.3f}  eval_r={_er:.3f}")
