"""Re-evaluate all completed co-design trials with multi-episode averaging.

Loads checkpoints from the Optuna study DB, re-runs eval with multi-episode
averaging, and updates BOTH trial values (for GP model) and user_attrs (for
reporting) directly in the SQLite DB.

Usage:
    python re_eval_completed.py [--num-episodes 3]
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(HERE, "co_design_eval.py")

parser = argparse.ArgumentParser()
parser.add_argument("--num-episodes", type=int, default=3)
parser.add_argument("--db", type=str, default=None)
parser.add_argument("--study-name", type=str, default="rpo_flat_leg_co_design")
parser.add_argument("--eval-timeout", type=int, default=7200)
args = parser.parse_args()

db_path = os.path.abspath(args.db or os.path.join(HERE, "co_design_study.db"))

import optuna

db_url = f"sqlite:///{db_path}"
study = optuna.load_study(study_name=args.study_name, storage=db_url)
if len(study.directions) != 1:
    raise ValueError("This legacy utility only supports single-objective studies; it cannot rewrite tracking/CoT studies.")

# ── Collect completed trials with checkpoints ────────────────────────────
to_reeval = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    ckpt = t.user_attrs.get("ckpt_path", "") if t.user_attrs else ""
    if not ckpt or not os.path.exists(ckpt):
        print(f"T{t.number}: SKIP (no checkpoint at {ckpt})")
        continue
    to_reeval.append((t.number, t.params["thigh_length"], t.params["calf_length"], ckpt))

print(f"Re-evaluating {len(to_reeval)} completed trials with {args.num_episodes} episodes each...\n")

# ── Run evals ────────────────────────────────────────────────────────────
updates = []  # (trial_number, avg_return, ep_returns, old_return)

for tn, thigh, calf, ckpt in to_reeval:
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--checkpoint", ckpt,
        "--thigh", str(thigh),
        "--calf", str(calf),
        "--task", "RPO-Flat",
        "--num-episodes", str(args.num_episodes),
        "--num-envs", "1",
        "--headless",
    ]
    old_val = study.trials[tn].values[0] if study.trials[tn].values else float("nan")
    print(f"T{tn}: thigh={thigh:.4f} calf={calf:.4f}  old_rtn={old_val:.1f}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.eval_timeout)
        stdout = result.stdout + result.stderr
        match = re.search(r"RESULT:\s*(.*)", stdout)
        if result.returncode != 0 or not match:
            print(f"  FAILED (rc={result.returncode})")
            continue
        data = json.loads(match.group(1))
        avg_return = data["avg_episode_return"]
        ep_returns = data.get("episode_returns", [])
        print(f"  returns: {[f'{r:.1f}' for r in ep_returns]}  avg={avg_return:.3f}")
        updates.append((tn, avg_return, ep_returns, old_val))
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
    except Exception as e:
        print(f"  ERROR: {e}")

# ── Update SQLite DB atomically ─────────────────────────────────────────
if not updates:
    print("\nNo results to update.")
    sys.exit(0)

conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode=WAL")

# Get study_id
study_id = conn.execute(
    "SELECT study_id FROM studies WHERE study_name = ?", (args.study_name,)
).fetchone()
if not study_id:
    print(f"ERROR: study '{args.study_name}' not found in DB")
    conn.close()
    sys.exit(1)
study_id = study_id[0]

try:
    with conn:
        for tn, avg_return, ep_returns, old_val in updates:
            # Get trial_id for this trial number
            trial_row = conn.execute(
                "SELECT trial_id FROM trials WHERE study_id = ? AND number = ?",
                (study_id, tn)
            ).fetchone()
            if not trial_row:
                print(f"T{tn}: trial_id not found, skipping")
                continue
            trial_id = trial_row[0]

            # Update trial value (objective 0)
            conn.execute(
                "UPDATE trial_values SET value = ? WHERE trial_id = ? AND objective = 0",
                (avg_return, trial_id)
            )

            # Optuna's actual SQLite table name is trial_user_attributes.
            attrs = {
                "episode_return": avg_return,
                "eval_episodes": ep_returns,
                "num_eval_episodes": args.num_episodes,
            }
            for key, val in attrs.items():
                val_json = json.dumps(val)
                existing = conn.execute(
                    "SELECT 1 FROM trial_user_attributes WHERE trial_id = ? AND key = ?",
                    (trial_id, key)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE trial_user_attributes SET value_json = ? WHERE trial_id = ? AND key = ?",
                        (val_json, trial_id, key)
                    )
                else:
                    conn.execute(
                        "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, ?, ?)",
                        (trial_id, key, val_json)
                    )

            print(f"T{tn}: DB updated  {old_val:.1f} → {avg_return:.3f}")
finally:
    conn.close()

print(f"\nDone. Updated {len(updates)} trials in DB.")
print(f"Both trial values (GP model) and user_attrs (reporting) have been updated.")
