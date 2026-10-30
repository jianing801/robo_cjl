"""Run OCD-50 initialization and mixed-variable BO for the 200-update policy.

Each objective evaluation is a two-process transaction: fine-tune a fresh copy
of the same 12k Rough policy for exactly 200 PPO updates on one morphology,
then evaluate that candidate with the frozen policy on the fixed 10cm stairs.
The controller is sequential by design because each fine-tune uses 4096 envs.

Nothing is launched by importing this module. Run explicitly with Isaac Lab's
Python; see BO_FINETUNE_PROTOCOL.md for the invocation and artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState, create_trial

from bo_finetune_common import (
    DEFAULT_FINETUNE_UPDATES,
    DEFAULT_NUM_ENVS,
    EXPECTED_BANK_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    add_source_import_paths,
    atomic_json,
    load_and_validate_assets,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DESIGN_CSV = HERE / "ocd_mixed_design_50_full_cooling" / "ocd_design_50.csv"
DEFAULT_CODE_ROOT = Path(
    r"C:\Users\Admin\.codex\.chatgpt-projects\g-p-6a967dbd08f4819188bc02c144f1a591"
    r"\rough_training_source_recovery_20261029\exact_source_snapshot_raw_20261029\robolab"
)
DEFAULT_CHECKPOINT = Path(
    r"C:\Users\Admin\.codex\.chatgpt-projects\g-p-6a967dbd08f4819188bc02c144f1a591"
    r"\rough_final_import_20261028\model_11999.pt"
)
DEFAULT_BANK = Path(
    r"C:\Users\Admin\.codex\.chatgpt-projects\g-p-6a967dbd08f4819188bc02c144f1a591"
    r"\rough_training_source_recovery_20261029\exact_source_snapshot_raw_20261029"
    r"\replay_bank_regenerated\bank.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--bank", default=str(DEFAULT_BANK))
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--design-csv", default=str(DEFAULT_DESIGN_CSV))
    parser.add_argument("--output", required=True, help="New run directory; --resume to continue it")
    parser.add_argument("--python", default=r"C:\Roboparty\env_isaaclab\Scripts\python.exe",
                        help="Isaac Lab Python for this controller and child workers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS,
                        help="Fine-tune replicas per candidate; fixed at 4096 for this protocol")
    parser.add_argument("--updates", type=int, default=DEFAULT_FINETUNE_UPDATES,
                        help="Fine-tuning PPO update count; fixed at exactly 200")
    parser.add_argument("--seed", type=int, default=42, help="Common fine-tuning seed")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Fine-tune fresh Adam at this LR; default is the base checkpoint LR")
    parser.add_argument("--bo-trials", type=int, default=50,
                        help="Additional GP proposals after the 50 OCD design evaluations")
    parser.add_argument("--startup-trials", type=int, default=5)
    parser.add_argument("--fine-tune-timeout", type=int, default=21600)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--kit-args", default="--/renderer/multiGpu/enabled=false")
    parser.add_argument(
        "--allow-source-matched-asset-hash-override",
        action="store_true",
        help=("Allow checkpoint/bank file SHA256 values to differ from the frozen local replay pair. "
              "Checkpoint structure/iteration, bank semantics/native source hashes, and source-tree "
              "identity are still validated."),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def params_key(params: dict[str, Any]) -> tuple:
    return (round(float(params["thigh_m"]), 10), round(float(params["calf_m"]), 10),
            str(params["knee_motor"]), str(params["ankle_motor"]))


def read_designs(path: Path, bank: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50:
        raise ValueError(f"OCD initialization must contain exactly 50 rows; found {len(rows)}")
    result, seen = [], set()
    for index, row in enumerate(rows):
        params = {
            "thigh_m": float(row["thigh_mm"]) / 1000.0,
            "calf_m": float(row["calf_mm"]) / 1000.0,
            "knee_motor": str(row["knee_motor"]),
            "ankle_motor": str(row["ankle_motor"]),
        }
        key = params_key(params)
        if key in seen:
            raise ValueError(f"duplicate OCD design at row {index + 1}: {key}")
        seen.add(key)
        if params["knee_motor"] not in bank["knees"] or params["ankle_motor"] not in bank["ankles"]:
            raise ValueError(f"motor outside pretrained policy scope: {params}")
        for field in ("thigh", "calf"):
            lo, hi = bank["ranges_m"][field]
            if not lo <= params[f"{field}_m"] <= hi:
                raise ValueError(f"{field} outside trained interval: {params}")
        result.append({
            "design_id": str(row.get("design_id") or f"ocd50_{index + 1:02d}"),
            **params,
            "o_cd_row": index,
        })
    return result


def build_protocol(args, provenance: dict[str, Any], bank: dict[str, Any],
                   designs: list[dict[str, Any]]) -> dict:
    protocol = {
        "result_kind": "adaptation_aware_mixed_variable_rough_bo_v1",
        "objective": "minimize fixed-horizon forward-speed tracking MAE after candidate-specific PPO fine-tuning",
        "initial_design": {
            "method": "OCD mixed-factor 50-point design",
            "csv": str(Path(args.design_csv).resolve()),
            "rows": len(designs),
            "old_zero_update_pilot_imported": False,
        },
        "bo_implementation_sha256": {
            name: sha256_file(HERE / name)
            for name in (
                "bo_finetune_common.py",
                "bo_finetune_candidate.py",
                "bo_finetune_evaluate.py",
                "bo_finetune_controller.py",
                "ocd_mixed_design_50.py",
            )
        },
        "base_checkpoint": provenance["checkpoint"],
        "base_checkpoint_sha256": provenance["checkpoint_sha256"],
        "base_checkpoint_iter": provenance["checkpoint_iter"],
        "base_completed_updates": provenance["completed_updates"],
        "bank": provenance["bank"],
        "bank_sha256": provenance["bank_sha256"],
        "source_matched_asset_hash_override": bool(args.allow_source_matched_asset_hash_override),
        "code_root": provenance["code_root"],
        "source_mismatch_count": provenance["source_mismatch_count"],
        "design_csv_sha256": sha256_file(args.design_csv),
        "continuous_search_ranges_m": bank["ranges_m"],
        "categorical_levels": {"knee_motor": bank["knees"], "ankle_motor": bank["ankles"]},
        "device": args.device,
        "python": str(Path(args.python).resolve()),
        "kit_args": args.kit_args,
        "fine_tune_timeout_s": args.fine_tune_timeout,
        "evaluation_timeout_s": args.eval_timeout,
        "fine_tuning": {
            "updates_per_candidate": DEFAULT_FINETUNE_UPDATES,
            "environment_count": args.num_envs,
            "training_seed": args.seed,
            "terrain": "fixed 10cm stairs",
            "forward_command_mps": 0.5,
            "source_rough_noise_and_events": "retained consistently for all candidates",
            "policy_initialization": "identical model_11999 policy and observation-normalization weights",
            "optimizer": "fresh Adam for every candidate",
            "optimizer_initial_learning_rate": args.learning_rate or provenance["checkpoint_learning_rate"],
            "optimizer_state_restored": False,
        },
        "evaluation": {
            "terrain": "fixed 10cm stairs",
            "command_mps": 0.5,
            "horizon_s": 20.0,
            "seed": args.eval_seed,
            "num_envs": 1,
            "domain_randomization": False,
            "observation_noise": False,
            "initial_perturbation": False,
            "motor_delay": 0,
            "objective": "fixed-horizon mean absolute forward-speed tracking error in m/s",
            "early_fall_tail": "remaining horizon contributes abs(0 m/s - command); fall flag retained",
        },
        "adaptive_bo": {
            "sampler": "Optuna GPSampler mixed continuous/categorical",
            "startup_trials": args.startup_trials,
            "additional_gp_trials_requested": args.bo_trials,
            "direction": "minimize",
            "sequential_execution": True,
        },
        "runtime_source_audit": {
            "matched_source_count": provenance["runtime_source_audit"]["matched_source_count"],
            "native_source_count": provenance["runtime_source_audit"]["native_source_count"],
            "hybrid_source_count": provenance["runtime_source_audit"]["hybrid_source_count"],
            "mismatch_count": provenance["runtime_source_audit"]["mismatch_count"],
        },
        "asset_identity_boundary": "Local URDF regenerated from checkpoint-matched SW-GP/motor snapshot; no byte-identical server URDF claim.",
    }
    return protocol


def tail(path: Path, n=60) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def next_attempt_dir(output: Path, trial_number: int) -> Path:
    trial_root = output / "runs" / f"trial_{trial_number:05d}"
    trial_root.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = trial_root / f"attempt_{index:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        index += 1


def run_child(command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path,
              stderr_path: Path, timeout: int, label: str) -> None:
    """Run one Isaac child in the foreground, teeing live output to terminal and files."""
    started = time.monotonic()
    console_lock = threading.Lock()

    def pump(source, path: Path, console) -> None:
        with path.open("w", encoding="utf-8", errors="replace", buffering=1) as log:
            for line in iter(source.readline, ""):
                log.write(line)
                log.flush()
                with console_lock:
                    try:
                        console.write(line)
                        console.flush()
                    except (BrokenPipeError, OSError):
                        # Keep preserving per-attempt logs if the terminal disconnects.
                        pass
        source.close()

    process = subprocess.Popen(
        command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    stdout_thread = threading.Thread(
        target=pump, args=(process.stdout, stdout_path, sys.stdout), daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump, args=(process.stderr, stderr_path, sys.stderr), daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=timeout)
    except KeyboardInterrupt:
        # The foreground terminal sends SIGINT to the whole process group.
        # Reap the worker before the controller exits, including native hangs.
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stdout_thread.join()
        stderr_thread.join()
        raise
    except subprocess.TimeoutExpired as exc:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stdout_thread.join()
        stderr_thread.join()
        atomic_json(stderr_path.with_suffix(stderr_path.suffix + ".timeout.json"), {
            "label": label, "timeout_seconds": timeout, "elapsed_wall_s": time.monotonic() - started,
            "error": str(exc),
        })
        raise RuntimeError(f"{label} timed out after {timeout}s; logs: {stdout_path}, {stderr_path}") from exc
    stdout_thread.join()
    stderr_thread.join()
    if returncode != 0:
        raise RuntimeError(f"{label} failed with rc={returncode}\n{tail(stderr_path)}")


def run_candidate(args, output: Path, provenance: dict[str, Any], params: dict[str, Any],
                  trial_number: int, phase: str) -> dict[str, Any]:
    attempt = next_attempt_dir(output, trial_number)
    request = {
        "trial_number": int(trial_number),
        "phase": phase,
        "design_id": params.get("design_id", f"trial_{trial_number:05d}"),
        "thigh_m": float(params["thigh_m"]),
        "calf_m": float(params["calf_m"]),
        "knee_motor": str(params["knee_motor"]),
        "ankle_motor": str(params["ankle_motor"]),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "bank_sha256": provenance["bank_sha256"],
    }
    request_path = output / "requests" / f"trial_{trial_number:05d}_{attempt.name}.json"
    atomic_json(request_path, request)
    train_dir = attempt / "finetune"
    eval_dir = attempt / "evaluation"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    code_root = Path(args.code_root).resolve()
    python = str(Path(args.python).resolve())
    kit_args = args.kit_args.strip()

    train_cmd = [
        python, "-u", str(HERE / "bo_finetune_candidate.py"),
        "--checkpoint", provenance["checkpoint"],
        "--bank", provenance["bank"],
        "--code-root", str(code_root),
        "--request", str(request_path),
        "--output", str(train_dir),
        "--device", args.device,
        "--num-envs", str(args.num_envs),
        "--updates", str(DEFAULT_FINETUNE_UPDATES),
        "--seed", str(args.seed),
        "--headless",
    ]
    if args.learning_rate is not None:
        train_cmd += ["--learning-rate", str(args.learning_rate)]
    if args.allow_source_matched_asset_hash_override:
        train_cmd.append("--allow-source-matched-asset-hash-override")
    if kit_args:
        train_cmd += ["--kit-args", kit_args]
    atomic_json(attempt / "train_command.json", {"argv": train_cmd, "cwd": str(code_root.parent)})
    run_child(train_cmd, cwd=code_root.parent, env=env,
              stdout_path=attempt / "finetune.stdout.log", stderr_path=attempt / "finetune.stderr.log",
              timeout=args.fine_tune_timeout, label=f"trial {trial_number} fine-tune")
    fine_meta_path = train_dir / "finetune_complete.json"
    if not fine_meta_path.is_file():
        raise RuntimeError(f"fine-tune exited without success manifest: {fine_meta_path}")
    fine_meta = json.loads(fine_meta_path.read_text(encoding="utf-8"))
    if fine_meta.get("fine_tune_updates") != DEFAULT_FINETUNE_UPDATES:
        raise RuntimeError("fine-tune success manifest reports wrong update count")

    fine_checkpoint = Path(fine_meta["checkpoint"]).resolve()
    eval_cmd = [
        python, "-u", str(HERE / "bo_finetune_evaluate.py"),
        "--checkpoint", str(fine_checkpoint),
        "--base-checkpoint", provenance["checkpoint"],
        "--bank", provenance["bank"],
        "--code-root", str(code_root),
        "--candidate-dir", str(train_dir),
        "--output", str(eval_dir),
        "--device", args.device,
        "--seed", str(args.eval_seed),
        "--headless",
    ]
    if kit_args:
        eval_cmd += ["--kit-args", kit_args]
    if args.allow_source_matched_asset_hash_override:
        eval_cmd.append("--allow-source-matched-asset-hash-override")
    atomic_json(attempt / "evaluation_command.json", {"argv": eval_cmd, "cwd": str(code_root.parent)})
    run_child(eval_cmd, cwd=code_root.parent, env=env,
              stdout_path=attempt / "evaluation.stdout.log", stderr_path=attempt / "evaluation.stderr.log",
              timeout=args.eval_timeout, label=f"trial {trial_number} evaluation")
    summary_path = eval_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"evaluation exited without summary: {summary_path}")
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "result": "ROUGH_CANDIDATE_200_UPDATE_SPEED_TRACKING_EVALUATED",
        "trial_number": trial_number,
        "base_checkpoint_sha256": provenance["checkpoint_sha256"],
        "fine_tune_updates": DEFAULT_FINETUNE_UPDATES,
        "bank_sha256": provenance["bank_sha256"],
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"trial result identity mismatch for {key}: {metrics.get(key)!r}")
    if not math.isfinite(float(metrics.get("objective_value", float("nan")))):
        raise RuntimeError("candidate objective is not finite")
    metrics["attempt_dir"] = str(attempt.resolve())
    return metrics


def existing_success(output: Path, trial_number: int, params: dict[str, Any]) -> dict[str, Any] | None:
    trial_root = output / "runs" / f"trial_{trial_number:05d}"
    if not trial_root.is_dir():
        return None
    expected = params_key(params)
    for summary_path in sorted(trial_root.glob("attempt_*/evaluation/summary.json"), reverse=True):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (summary.get("result") == "ROUGH_CANDIDATE_200_UPDATE_SPEED_TRACKING_EVALUATED"
                and summary.get("fine_tune_updates") == DEFAULT_FINETUNE_UPDATES
                and params_key(summary["params"]) == expected):
            summary["attempt_dir"] = str(summary_path.parent.parent.resolve())
            return summary
    return None


def trial_by_key(study, key: tuple):
    for frozen in study.get_trials(deepcopy=False):
        if frozen.state != TrialState.COMPLETE:
            continue
        if frozen.user_attrs.get("design_key") == list(key):
            return frozen
    return None


def result_attrs(metrics: dict[str, Any], *, phase: str, design_id: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "design_id": design_id,
        "design_key": list(params_key(metrics["params"])),
        "attempt_dir": metrics.get("attempt_dir"),
        "fell": bool(metrics["fell"]),
        "feasible": bool(metrics["feasible"]),
        "elapsed_s": float(metrics["elapsed_s"]),
        "tracking_mae_observed_mps": float(metrics["tracking_mae_observed_mps"]),
        "tracking_mae_fixed_horizon_mps": float(metrics["tracking_mae_fixed_horizon_mps"]),
        "fine_tune_updates": int(metrics["fine_tune_updates"]),
        "fine_tuned_checkpoint_sha256": metrics["fine_tuned_checkpoint_sha256"],
    }


def find_metrics(output: Path, trial_number: int, params: dict[str, Any]) -> dict[str, Any] | None:
    return existing_success(output, trial_number, params)


def write_results(output: Path, study, provenance: dict[str, Any], protocol: dict[str, Any]) -> None:
    rows = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != TrialState.COMPLETE:
            continue
        attrs = trial.user_attrs
        rows.append({
            "trial": trial.number,
            **trial.params,
            "objective_tracking_mae_mps": trial.value,
            "phase": attrs.get("phase"),
            "design_id": attrs.get("design_id"),
            "fell": attrs.get("fell"),
            "feasible": attrs.get("feasible"),
            "elapsed_s": attrs.get("elapsed_s"),
            "fine_tune_updates": attrs.get("fine_tune_updates"),
            "fine_tuned_checkpoint_sha256": attrs.get("fine_tuned_checkpoint_sha256"),
            "attempt_dir": attrs.get("attempt_dir"),
        })
    with (output / "trials.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "trial", "thigh_m", "calf_m", "knee_motor", "ankle_motor",
            "objective_tracking_mae_mps", "phase", "design_id", "fell", "feasible",
            "elapsed_s", "fine_tune_updates", "fine_tuned_checkpoint_sha256", "attempt_dir",
        ])
        writer.writeheader()
        writer.writerows(rows)
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    best = study.best_trial if completed else None
    atomic_json(output / "summary.json", {
        "result": "ADAPTATION_AWARE_ROUGH_MIXED_BO_IN_PROGRESS",
        "completed_trials": len(completed),
        "completed_initial_ocd_trials": sum(t.user_attrs.get("phase") == "ocd50" for t in completed),
        "completed_adaptive_bo_trials": sum(t.user_attrs.get("phase") == "gp" for t in completed),
        "best_trial": None if best is None else {
            "number": best.number, "objective_tracking_mae_mps": best.value,
            "params": best.params, "phase": best.user_attrs.get("phase"),
        },
        "objective": protocol["objective"],
        "fine_tune_updates_per_candidate": DEFAULT_FINETUNE_UPDATES,
        "environment_count_per_candidate": protocol["fine_tuning"]["environment_count"],
        "base_checkpoint_sha256": provenance["checkpoint_sha256"],
        "bank_sha256": provenance["bank_sha256"],
        "study_db": str((output / "study.sqlite3").resolve()),
        "policy_frozen_during_evaluation": True,
        "candidate_policy_fine_tuning_performed": True,
        "old_zero_update_pilot_imported": False,
        "requires_finalist_replicates_and_gait_confirmation": True,
    })


def main() -> int:
    args = parse_args()
    if args.num_envs != DEFAULT_NUM_ENVS:
        raise ValueError("this objective fixes fine-tuning at 4096 environments per morphology")
    if args.updates != DEFAULT_FINETUNE_UPDATES:
        raise ValueError("this BO objective fixes candidate fine-tuning at exactly 200 updates")
    if args.bo_trials < 0 or args.startup_trials < 1:
        raise ValueError("bo-trials must be nonnegative and startup-trials positive")
    if args.fine_tune_timeout < 1 or args.eval_timeout < 1:
        raise ValueError("timeouts must be positive")
    if tuple(int(x) for x in optuna.__version__.split(".")[:2]) < (4, 4):
        raise RuntimeError("Optuna >=4.4 is required for GPSampler")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Isaac Lab Python not found: {args.python}")

    add_source_import_paths(args.code_root)
    saved, bank, provenance = load_and_validate_assets(
        args.checkpoint, args.bank, args.code_root,
        allow_hash_override=args.allow_source_matched_asset_hash_override,
    )
    if not args.allow_source_matched_asset_hash_override:
        if provenance["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("base model hash differs from the audited completed Rough checkpoint")
        if provenance["bank_sha256"] != EXPECTED_BANK_SHA256:
            raise ValueError("BO bank hash differs from the audited local replay bank")
    designs = read_designs(Path(args.design_csv).resolve(), bank)
    output = Path(args.output).resolve()
    plan_path = output / "plan.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"output exists; pass --resume to continue: {output}")
        if not plan_path.is_file():
            raise ValueError("resume directory has no plan.json")
    else:
        if args.resume:
            raise FileNotFoundError("--resume requested but output directory does not exist")
        output.mkdir(parents=True, exist_ok=False)
        for name in ("requests", "runs"):
            (output / name).mkdir()

    protocol = build_protocol(args, provenance, bank, designs)
    contract_path = output / "run_contract.json"
    if args.resume:
        if not plan_path.is_file() or not contract_path.is_file():
            raise ValueError("resume directory is missing its immutable plan/contract")
        original_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if original_contract != protocol:
            raise ValueError("resume configuration differs from the existing run contract")
    else:
        atomic_json(plan_path, protocol)
        atomic_json(contract_path, protocol)

    db_path = output / "study.sqlite3"
    storage = "sqlite:///" + db_path.as_posix()
    sampler = optuna.samplers.GPSampler(seed=args.seed, n_startup_trials=args.startup_trials)
    study = optuna.create_study(
        study_name="rough_morphology_after_200_update_finetune",
        direction="minimize", sampler=sampler, storage=storage, load_if_exists=args.resume,
    )
    study.set_user_attr("protocol", protocol)
    distributions = {
        "thigh_m": FloatDistribution(*bank["ranges_m"]["thigh"]),
        "calf_m": FloatDistribution(*bank["ranges_m"]["calf"]),
        "knee_motor": CategoricalDistribution(bank["knees"]),
        "ankle_motor": CategoricalDistribution(bank["ankles"]),
    }

    # First complete the 50 OCD design points.  Existing 0-update pilot data are
    # intentionally excluded because they measure a different objective.
    for index, row in enumerate(designs):
        params = {key: row[key] for key in ("thigh_m", "calf_m", "knee_motor", "ankle_motor")}
        key = params_key(params)
        if trial_by_key(study, key):
            continue
        trial_number = index
        metrics = find_metrics(output, trial_number, params)
        total_candidates = len(designs) + args.bo_trials
        completed_before = sum(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
        if metrics is None:
            print(f"\n[BO_PROGRESS] {completed_before + 1}/{total_candidates} START "
                  f"phase=OCD design={index + 1}/{len(designs)} trial={trial_number} params={params}",
                  flush=True)
            metrics = run_candidate(args, output, provenance, {**row, **params}, trial_number, "ocd50")
        else:
            print(f"\n[BO_PROGRESS] {completed_before + 1}/{total_candidates} REUSE "
                  f"phase=OCD design={index + 1}/{len(designs)} trial={trial_number}",
                  flush=True)
        attrs = result_attrs(metrics, phase="ocd50", design_id=row["design_id"])
        study.add_trial(create_trial(params=params, distributions=distributions,
                                     values=[float(metrics["objective_value"])], user_attrs=attrs))
        write_results(output, study, provenance, protocol)
        completed_after = sum(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
        print(f"[BO_PROGRESS] {completed_after}/{total_candidates} COMPLETE "
              f"phase=OCD trial={trial_number} MAE={metrics['objective_value']:.6f} "
              f"fell={metrics['fell']}", flush=True)

    # Recover any adaptive trial interrupted after ask() by rerunning the same
    # request or importing its already-written summary, never another model.
    for frozen in study.get_trials(deepcopy=False):
        if frozen.state != TrialState.RUNNING:
            continue
        if frozen.user_attrs.get("phase") != "gp":
            study.tell(frozen.number, state=TrialState.FAIL)
            continue
        if set(frozen.params) != set(distributions):
            study.tell(frozen.number, state=TrialState.FAIL)
            continue
        params = dict(frozen.params)
        metrics = find_metrics(output, frozen.number, params)
        if metrics is None:
            completed_before = sum(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
            total_candidates = len(designs) + args.bo_trials
            print(f"\n[BO_PROGRESS] {completed_before + 1}/{total_candidates} START "
                  f"phase=GP trial={frozen.number} (resume pending) params={params}", flush=True)
            metrics = run_candidate(args, output, provenance, params, frozen.number, "gp")
        for name, value in result_attrs(metrics, phase="gp", design_id=f"gp_{frozen.number:05d}").items():
            study._storage.set_trial_user_attr(frozen._trial_id, name, value)
        study.tell(frozen.number, float(metrics["objective_value"]))

    completed_gp = sum(t.state == TrialState.COMPLETE and t.user_attrs.get("phase") == "gp"
                       for t in study.get_trials(deepcopy=False))
    unique_keys = {params_key(t.params) for t in study.get_trials(deepcopy=False)
                   if t.state == TrialState.COMPLETE}
    attempts = 0
    max_attempts = max(args.bo_trials * 30, 30)
    while completed_gp < args.bo_trials and attempts < max_attempts:
        attempts += 1
        trial = study.ask()
        params = {
            "thigh_m": trial.suggest_float("thigh_m", *bank["ranges_m"]["thigh"]),
            "calf_m": trial.suggest_float("calf_m", *bank["ranges_m"]["calf"]),
            "knee_motor": trial.suggest_categorical("knee_motor", bank["knees"]),
            "ankle_motor": trial.suggest_categorical("ankle_motor", bank["ankles"]),
        }
        trial.set_user_attr("phase", "gp")
        key = params_key(params)
        if key in unique_keys:
            study.tell(trial, state=TrialState.PRUNED)
            continue
        total_candidates = len(designs) + args.bo_trials
        completed_before = sum(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
        print(f"\n[BO_PROGRESS] {completed_before + 1}/{total_candidates} START "
              f"phase=GP adaptive={completed_gp + 1}/{args.bo_trials} trial={trial.number} params={params}",
              flush=True)
        metrics = run_candidate(args, output, provenance, params, trial.number, "gp")
        attrs = result_attrs(metrics, phase="gp", design_id=f"gp_{trial.number:05d}")
        for name, value in attrs.items():
            trial.set_user_attr(name, value)
        study.tell(trial, float(metrics["objective_value"]))
        unique_keys.add(key)
        completed_gp += 1
        write_results(output, study, provenance, protocol)
        print(f"[BO_PROGRESS] {len(designs) + completed_gp}/{len(designs) + args.bo_trials} COMPLETE "
              f"phase=GP trial={trial.number} MAE={metrics['objective_value']:.6f} "
              f"fell={metrics['fell']}", flush=True)
    if completed_gp < args.bo_trials:
        raise RuntimeError(f"obtained {completed_gp}/{args.bo_trials} unique adaptive BO observations")

    write_results(output, study, provenance, protocol)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["result"] = "ADAPTATION_AWARE_ROUGH_MIXED_BO_COMPLETE_NOT_FINAL_DESIGN"
    atomic_json(summary_path, summary)
    print(f"[bo-finetune] complete: {summary_path}", flush=True)
    if study.best_trial:
        print(f"[bo-finetune] best MAE={study.best_value:.6f} params={study.best_params}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        traceback.print_exc()
        raise
