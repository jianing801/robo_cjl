"""Read-only conversion of legacy BO observations; never edit an old run/database.

Run with --run OLD --output NEW_OBSERVATIONS.json. No Isaac or training starts.
Successful full rollouts retain their observed MAE. Falls remain feasibility
observations with objective=None. The erroneous old score is kept for audit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from bo_finetune_common import atomic_json, sha256_file
from bo_metrics import METRICS_SCHEMA, score_rollout, validate_metrics

OBSERVATION_SCHEMA = "rough_bo_observations_v2"


def convert_summary(summary: dict, contract: dict) -> dict:
    for field in ("base_checkpoint_sha256", "bank_sha256"):
        if summary[field] != contract[field]:
            raise ValueError(f"legacy observation identity differs: {field}")
    expected = {
        "result": "ROUGH_CANDIDATE_200_UPDATE_SPEED_TRACKING_EVALUATED",
        "base_checkpoint_iter": 11999, "fine_tune_updates": 200,
        "fine_tune_optimizer_mode": "fresh_adam", "command_mps": .5,
        "horizon_s": 20.0, "num_envs": 1, "domain_randomization": False,
        "observation_noise": False, "initial_perturbation": False, "motor_delay": 0,
        "eval_seed": contract["evaluation"]["seed"],
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(f"legacy evaluation protocol differs: {name}")
    for name in ("runtime_source_audit", "loaded_module_audit"):
        if summary[name].get("mismatch_count") != 0:
            raise ValueError("legacy source audit failed")
    dt = float(summary.get("control_dt_s", .02))
    if dt != .02:
        raise ValueError("only the audited 50Hz legacy protocol can be converted")
    elapsed = float(summary["elapsed_s"])
    observed = float(summary["tracking_mae_observed_mps"])
    if not math.isfinite(elapsed) or not math.isfinite(observed):
        raise ValueError("nonfinite legacy observation")
    steps = round(elapsed / dt)
    if abs(steps * dt - elapsed) > 1e-6:
        raise ValueError("legacy duration is not an integral sample count")
    corrected = score_rollout(observed * steps, steps, dt, 20.0,
                               summary["fell"], summary["timed_out"])
    if corrected["feasible"] != summary["feasible"]:
        raise ValueError("legacy feasibility flag conflicts with completion")
    if summary.get("metrics_schema") == METRICS_SCHEMA:
        validate_metrics(summary)
    else:
        # Explicitly recognize the old implementation, rather than silently
        # accepting any score from an unknown protocol/version.
        old_score = (observed * steps + (1000 - steps) * .5 * dt) / 20.0
        if not math.isclose(float(summary["objective_value"]), old_score, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("legacy objective is not the recognized v1 formula")
    return {**summary, **corrected, "original_objective_value": summary["objective_value"]}


def build_observations(run: Path) -> dict:
    run = run.resolve()
    contract_path, summary_path = run / "run_contract.json", run / "summary.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    root_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (contract["fine_tuning"]["updates_per_candidate"] != 200
            or contract["fine_tuning"]["environment_count"] != 4096
            or contract["fine_tuning"]["training_seed"] != 42):
        raise ValueError("legacy adaptation budget or seed differs")
    if root_summary["base_checkpoint_sha256"] != contract["base_checkpoint_sha256"]:
        raise ValueError("legacy root summary/checkpoint identity mismatch")
    observations = []
    for trial_dir in sorted((run / "runs").glob("trial_*")):
        candidates = sorted(trial_dir.glob("attempt_*/evaluation/summary.json"))
        if not candidates:
            continue  # Infrastructure failures are NOT infeasible structures.
        if len(candidates) != 1:
            raise ValueError(f"ambiguous evaluation attempts: {trial_dir}; select an audited result explicitly")
        path = candidates[0]
        raw = json.loads(path.read_text(encoding="utf-8"))
        metrics = convert_summary(raw, contract)
        number = int(trial_dir.name.split("_")[-1])
        if metrics["trial_number"] != number:
            raise ValueError("legacy trial number differs from its directory")
        phase = "ocd50" if number < 50 else "historical_gp"
        observations.append({
            "original_trial_number": number, "phase": phase,
            "source_summary": str(path), "source_summary_sha256": sha256_file(path),
            "metrics": metrics,
        })
    numbers = [o["original_trial_number"] for o in observations]
    if numbers != list(range(len(numbers))) or len(numbers) != root_summary["completed_trials"]:
        raise ValueError("legacy observations do not match the completed run summary")
    return {
        "observation_schema": OBSERVATION_SCHEMA, "metrics_schema": METRICS_SCHEMA,
        "source_run": str(run), "source_contract_sha256": sha256_file(contract_path),
        "source_root_summary_sha256": sha256_file(summary_path),
        "base_checkpoint_sha256": contract["base_checkpoint_sha256"],
        "bank_sha256": contract["bank_sha256"], "seed": 42,
        "eval_seed": contract["evaluation"]["seed"],
        "optimizer_initial_learning_rate": contract["fine_tuning"]["optimizer_initial_learning_rate"],
        "observations": observations,
        "old_run_modified": False,
        "historical_gp_was_selected_with_legacy_objective": True,
        "not_a_retroactive_correction_of_the_search_path": True,
    }


def load_observations(path: Path, provenance: dict, *, seed: int, eval_seed: int,
                      learning_rate: float) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError("unknown warm-start observation format")
    for key in ("base_checkpoint_sha256", "bank_sha256"):
        expected = provenance["checkpoint_sha256" if key == "base_checkpoint_sha256" else key]
        if document[key] != expected:
            raise ValueError(f"warm-start {key} differs from current run")
    if (document["seed"] != seed or document["eval_seed"] != eval_seed
            or not math.isclose(document["optimizer_initial_learning_rate"], learning_rate, rel_tol=1e-12)):
        raise ValueError("warm-start seeds or fine-tuning learning rate differ")
    seen = set()
    for i, observation in enumerate(document["observations"]):
        metrics = validate_metrics(observation["metrics"])
        if observation["original_trial_number"] != i or metrics["trial_number"] != i:
            raise ValueError("warm-start observations must retain original trial order")
        if metrics["base_checkpoint_sha256"] != document["base_checkpoint_sha256"] or metrics["bank_sha256"] != document["bank_sha256"]:
            raise ValueError("warm-start entry identity differs from manifest")
        if metrics["fine_tune_updates"] != 200:
            raise ValueError("warm-start requires 200-update candidates")
        # The manifest is self-contained for transport; original SHA256 values
        # remain recorded even if the old run is on another computer.
        params = metrics["params"]
        key = (float(params["thigh_m"]), float(params["calf_m"]), params["knee_motor"], params["ankle_motor"])
        if key in seen:
            raise ValueError("duplicate warm-start design")
        seen.add(key)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target, source = args.output.resolve(), args.run.resolve()
    if target.exists() or target.is_relative_to(source):
        raise ValueError("write observations to a NEW file OUTSIDE the old run")
    document = build_observations(source)
    atomic_json(target, document)
    count = len(document["observations"])
    feasible = sum(o["metrics"]["feasible"] for o in document["observations"])
    print(f"Converted {count} observations: {feasible} feasible, {count-feasible} infeasible; old run unchanged")


if __name__ == "__main__":
    main()
