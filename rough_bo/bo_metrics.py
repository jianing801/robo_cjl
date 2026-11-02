"""Versioned, simulator-independent scoring for complete Rough evaluations.

An infeasible rollout is a feasibility observation, not a full-horizon MAE.
The finite Optuna ledger value below is NEVER a target for either GP.
"""
from __future__ import annotations

import math

METRICS_SCHEMA = "rough_complete_rollout_mae_v2"
INFEASIBLE_LEDGER_VALUE = 1e6


def score_rollout(error_sum: float, steps: int, dt: float, horizon: float,
                  fell: bool, timed_out: bool = False) -> dict:
    if (type(fell) is not bool or type(timed_out) is not bool
            or not math.isfinite(error_sum) or error_sum < 0 or steps < 1
            or isinstance(steps, bool) or int(steps) != steps
            or not math.isfinite(dt) or dt <= 0
            or not math.isfinite(horizon) or horizon <= 0):
        raise ValueError("invalid rollout error, sample count or timing")
    expected = round(horizon / dt)
    if expected < 1 or abs(expected * dt - horizon) > 1e-8 or steps > expected:
        raise ValueError("rollout horizon must contain an integral number of control steps")
    observed = float(error_sum / steps)
    feasible = not fell and steps == expected
    return {
        "metrics_schema": METRICS_SCHEMA,
        "control_dt_s": float(dt), "sample_count": int(steps),
        "expected_sample_count": expected,
        "tracking_mae_observed_mps": observed,
        "tracking_mae_fixed_horizon_mps": observed if feasible else None,
        "objective_value": observed if feasible else None,
        "feasible": feasible,
        "infeasible_reason": "fell" if fell else "incomplete_horizon" if not feasible else None,
    }


def validate_metrics(metrics: dict) -> dict:
    """Reject stale summaries and inconsistent feasibility before feeding models."""
    if metrics.get("metrics_schema") != METRICS_SCHEMA:
        raise ValueError("legacy BO metrics: convert observations into a NEW run, never resume v1")
    observed = float(metrics["tracking_mae_observed_mps"])
    checked = score_rollout(observed * metrics["sample_count"], metrics["sample_count"],
                            metrics["control_dt_s"], metrics["horizon_s"], metrics["fell"],
                            metrics.get("timed_out", False))
    elapsed = float(metrics["elapsed_s"])
    if not math.isfinite(elapsed) or abs(elapsed - checked["sample_count"] * checked["control_dt_s"]) > 1e-6:
        raise ValueError("evaluation elapsed time differs from sample count")
    for name in ("feasible", "objective_value", "tracking_mae_fixed_horizon_mps"):
        actual, expected = metrics.get(name), checked[name]
        if expected is None:
            if actual is not None:
                raise ValueError(f"infeasible rollout cannot have {name}")
        elif isinstance(expected, bool):
            if type(actual) is not bool or actual != expected:
                raise ValueError("feasibility flag differs from rollout completion")
        elif actual is None or not math.isfinite(float(actual)) or abs(float(actual) - expected) > 1e-10:
            raise ValueError("incorrect full-horizon MAE")
    return metrics


def ledger_value(metrics: dict) -> float:
    validate_metrics(metrics)
    return float(metrics["objective_value"]) if metrics["feasible"] else INFEASIBLE_LEDGER_VALUE


def metric_label(metrics: dict) -> str:
    return f"{metrics['objective_value']:.6f}" if metrics["feasible"] else "NA(infeasible)"
