"""Small, reproducible OCD pilot for 50 mixed leg/motor designs.

Implements the Optimal Coverage Design (OCD) objective and simulated-annealing
proposal scheme from Yi, Pan & Chen (2025), adapted to this study's two
continuous leg lengths and two nominal 3-level motor factors.

The expensive robot simulator is not called here. This script only chooses
initial BO design points and writes a CSV plus an audit JSON.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ocd_mixed_design_50_full_cooling"
N = 50
SEED = 20261029
N_PROBES = 4096  # paper recommends at least 4,000; LHS probes reduce MC noise
N_VALIDATION_PROBES = 64_000  # independent holdout to reduce coverage-score noise
INITIAL_TEMPERATURE = 10_000.0
COOLING = 0.95
FINAL_TEMPERATURE = 0.01
INNER_ITERATION_CAP = 1_000  # practical cap discussed in the article
IMPROVEMENT_TOLERANCE = 1e-4

THIGH_RANGE_M = (0.25, 0.40)
CALF_RANGE_M = (0.30, 0.45)
KNEE_MOTORS = ("RS04", "DM-J10010-2EC", "DM-J10010L-2EC")
ANKLE_MOTORS = ("DM-J6216P-2EC", "DM-J4340P-2EC", "RS06")
LEVELS = 3
COMBINATIONS = LEVELS * LEVELS


def midpoint_lhd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n midpoint levels in a randomly permuted 1-D Latin hypercube."""
    return rng.permutation(n).astype(np.int32)


def balanced_categories(n: int, levels: int, rng: np.random.Generator) -> np.ndarray:
    """Make a categorical column whose counts differ by at most one."""
    base, remainder = divmod(n, levels)
    values = np.repeat(np.arange(levels, dtype=np.int8), base)
    if remainder:
        extras = rng.choice(levels, size=remainder, replace=False)
        values = np.concatenate((values, extras.astype(np.int8)))
    return rng.permutation(values)


def lhs_probes(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform Latin-hypercube probes used to estimate covered volume."""
    probes = np.empty((n, dimensions), dtype=np.float64)
    for dim in range(dimensions):
        probes[:, dim] = (rng.permutation(n) + rng.random(n)) / n
    return probes


def make_site_coverage_masks(probes: np.ndarray, n: int, radius: float) -> np.ndarray:
    """Precompute packed probe-coverage bits for the n-by-n midpoint grid."""
    strata = (np.arange(n, dtype=np.float64) + 0.5) / n
    grid_x, grid_y = np.meshgrid(strata, strata, indexing="ij")
    sites = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    radius_sq = radius * radius
    masks = np.empty((len(sites), (len(probes) + 7) // 8), dtype=np.uint8)
    chunk = 64
    for start in range(0, len(sites), chunk):
        stop = min(start + chunk, len(sites))
        delta = sites[start:stop, None, :] - probes[None, :, :]
        covered = np.einsum("ijk,ijk->ij", delta, delta) <= radius_sq
        masks[start:stop] = np.packbits(covered, axis=1, bitorder="little")
    return masks


POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def covered_fraction(site_ids: np.ndarray, masks: np.ndarray, n_probes: int) -> float:
    if len(site_ids) == 0:
        return 0.0
    union = np.bitwise_or.reduce(masks[site_ids], axis=0)
    return float(POPCOUNT[union].sum(dtype=np.int64) / n_probes)


def score_design(
    x: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    masks: np.ndarray,
    n_probes: int,
) -> tuple[float, dict]:
    """Paper's strength-2 OCD criterion: global + sliced coverage, equally weighted."""
    site_ids = x[:, 0].astype(np.int32) * N + x[:, 1].astype(np.int32)
    global_coverage = covered_fraction(site_ids, masks, n_probes)

    pair_counts = np.zeros((LEVELS, LEVELS), dtype=np.int32)
    pair_coverages = np.zeros((LEVELS, LEVELS), dtype=np.float64)
    for k in range(LEVELS):
        for a in range(LEVELS):
            rows = np.flatnonzero((knee == k) & (ankle == a))
            pair_counts[k, a] = len(rows)
            pair_coverages[k, a] = covered_fraction(site_ids[rows], masks, n_probes)

    count_range_fraction = float((pair_counts.max() - pair_counts.min()) / N)
    synthesis_coverage = float(pair_coverages.mean() / (1.0 + count_range_fraction))
    objective = 0.5 * (global_coverage + synthesis_coverage)
    details = {
        "objective": objective,
        "global_leg_coverage": global_coverage,
        "mean_pair_leg_coverage": float(pair_coverages.mean()),
        "synthesis_coverage_after_count_penalty": synthesis_coverage,
        "pair_count_range": int(pair_counts.max() - pair_counts.min()),
        "pair_counts": pair_counts.tolist(),
        "pair_coverages": pair_coverages.tolist(),
        "knee_counts": np.bincount(knee, minlength=LEVELS).tolist(),
        "ankle_counts": np.bincount(ankle, minlength=LEVELS).tolist(),
    }
    return objective, details


def perturb(
    x: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper-style move: swap a continuous column or rebalance-preserving category move."""
    x2, knee2, ankle2 = x.copy(), knee.copy(), ankle.copy()
    column = int(rng.integers(4))
    if column < 2:
        i, j = rng.choice(N, size=2, replace=False)
        x2[i, column], x2[j, column] = x2[j, column], x2[i, column]
        return x2, knee2, ankle2

    values = knee2 if column == 2 else ankle2
    counts = np.bincount(values, minlength=LEVELS)
    base = N // LEVELS
    i1 = int(rng.integers(N))
    value_count = int(counts[values[i1]])
    if value_count == base:
        others = rng.choice(np.delete(np.arange(N), i1), size=2, replace=False)
        indices = np.array([i1, int(others[0]), int(others[1])])
        values[indices] = rng.permutation(values[indices])
    else:
        if rng.random() < 0.5:
            others = rng.choice(np.delete(np.arange(N), i1), size=2, replace=False)
            indices = np.array([i1, int(others[0]), int(others[1])])
            values[indices] = rng.permutation(values[indices])
        else:
            underrepresented = np.flatnonzero(counts == base)
            candidates = np.flatnonzero(values == underrepresented[0])
            i2 = int(rng.choice(candidates))
            values[i1], values[i2] = values[i2], values[i1]
    return x2, knee2, ankle2


def simulated_annealing(
    x: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    masks: np.ndarray,
    n_probes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    current_x, current_k, current_a = x.copy(), knee.copy(), ankle.copy()
    current_score, current_detail = score_design(current_x, current_k, current_a, masks, n_probes)
    best_x, best_k, best_a = current_x.copy(), current_k.copy(), current_a.copy()
    best_score, best_detail = current_score, current_detail
    temperature = INITIAL_TEMPERATURE
    proposals = accepted = best_updates = 0
    stalled_temperatures = 0
    temperature_rounds = 0

    while temperature > FINAL_TEMPERATURE:
        # Algorithm 2 restarts each temperature round from the best design so far.
        current_x, current_k, current_a = best_x.copy(), best_k.copy(), best_a.copy()
        current_score = best_score
        score_at_round_start = best_score
        for _ in range(INNER_ITERATION_CAP):
            try_x, try_k, try_a = perturb(current_x, current_k, current_a, rng)
            try_score, try_detail = score_design(try_x, try_k, try_a, masks, n_probes)
            delta = try_score - current_score
            proposals += 1
            if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
                current_x, current_k, current_a = try_x, try_k, try_a
                current_score, current_detail = try_score, try_detail
                accepted += 1
            if try_score > best_score + 1e-12:
                best_x, best_k, best_a = try_x.copy(), try_k.copy(), try_a.copy()
                best_score, best_detail = try_score, try_detail
                best_updates += 1

        temperature_rounds += 1
        if best_score - score_at_round_start > IMPROVEMENT_TOLERANCE:
            stalled_temperatures = 0
        else:
            stalled_temperatures += 1
        temperature *= COOLING

    meta = {
        "proposals": proposals,
        "accepted_moves": accepted,
        "best_updates": best_updates,
        "temperature_rounds": temperature_rounds,
        "terminal_consecutive_stalled_rounds": stalled_temperatures,
        "last_temperature": temperature / COOLING if temperature_rounds else INITIAL_TEMPERATURE,
        "initial_objective": float(score_design(x, knee, ankle, masks, n_probes)[0]),
        "final_objective": float(best_score),
        "initial_details": score_design(x, knee, ankle, masks, n_probes)[1],
        "final_details": best_detail,
    }
    return best_x, best_k, best_a, meta


def physical_value(normalized: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return bounds[0] + normalized * (bounds[1] - bounds[0])


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    # One midpoint LHD per continuous factor, then assign category columns.
    x = np.column_stack((midpoint_lhd(N, rng), midpoint_lhd(N, rng)))
    knee = balanced_categories(N, LEVELS, rng)
    ankle = balanced_categories(N, LEVELS, rng)

    # OCD radius from the paper's q=1 choice: sqrt(p)/(2*n^(1/p)); p=2,n=50 => 0.1.
    continuous_dimensions = 2
    radius = math.sqrt(continuous_dimensions) / (2.0 * N ** (1.0 / continuous_dimensions))
    probes = lhs_probes(N_PROBES, continuous_dimensions, rng)
    masks = make_site_coverage_masks(probes, N, radius)

    out_x, out_knee, out_ankle, sa = simulated_annealing(x, knee, ankle, masks, N_PROBES, rng)
    # Independent, larger LHS probe set checks that the SA did not merely fit
    # noise in its 4,096-point optimization probe set.
    validation_rng = np.random.default_rng(SEED + 1)
    validation_probes = lhs_probes(N_VALIDATION_PROBES, continuous_dimensions, validation_rng)
    validation_masks = make_site_coverage_masks(validation_probes, N, radius)
    initial_validation_score, initial_validation = score_design(
        x, knee, ankle, validation_masks, N_VALIDATION_PROBES
    )
    final_validation_score, final_validation = score_design(
        out_x, out_knee, out_ankle, validation_masks, N_VALIDATION_PROBES
    )
    thigh_norm = (out_x[:, 0] + 0.5) / N
    calf_norm = (out_x[:, 1] + 0.5) / N
    thigh_m = physical_value(thigh_norm, THIGH_RANGE_M)
    calf_m = physical_value(calf_norm, CALF_RANGE_M)

    csv_path = OUT_DIR / "ocd_design_50.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "design_id", "thigh_mm", "calf_mm", "knee_motor", "ankle_motor",
            "knee_level", "ankle_level", "normalized_thigh", "normalized_calf",
        ))
        writer.writeheader()
        for i in range(N):
            writer.writerow({
                "design_id": f"ocd50_{i + 1:02d}",
                "thigh_mm": round(float(thigh_m[i] * 1000), 3),
                "calf_mm": round(float(calf_m[i] * 1000), 3),
                "knee_motor": KNEE_MOTORS[int(out_knee[i])],
                "ankle_motor": ANKLE_MOTORS[int(out_ankle[i])],
                "knee_level": int(out_knee[i]) + 1,
                "ankle_level": int(out_ankle[i]) + 1,
                "normalized_thigh": round(float(thigh_norm[i]), 6),
                "normalized_calf": round(float(calf_norm[i]), 6),
            })

    summary = {
        "result": "OCD_50_POINT_MIXED_DESIGN_PILOT_COMPLETE",
        "method": "Yi, Pan & Chen (2025), Optimal Coverage Design, strength s=2",
        "paper_doi": "10.1080/00224065.2025.2551916",
        "sample_size": N,
        "seed": SEED,
        "factors": {
            "continuous": {"thigh_m": list(THIGH_RANGE_M), "calf_m": list(CALF_RANGE_M)},
            "nominal": {"knee_motor": list(KNEE_MOTORS), "ankle_motor": list(ANKLE_MOTORS)},
        },
        "coverage": {
            "distance": "Euclidean on normalized [0,1]^2 leg-length domain",
            "radius": radius,
            "radius_rule": "sqrt(p)/(2*n^(1/p)), paper numerical-study q=1; p=2,n=50",
            "probe_method": "fixed 2D Latin-hypercube Monte Carlo probes",
            "probe_count": N_PROBES,
            "independent_validation_probe_count": N_VALIDATION_PROBES,
            "objective": "0.5 * (global continuous volumetric coverage + strength-2 synthesis coverage); synthesis coverage averages pair-sliced coverage and divides by 1 + (max pair count - min pair count)/n",
            "note": "This short computational pilot uses 4,096 probes and the article's practical 1,000-proposal inner-loop cap. It is an approximate design-score run, not a simulator run or proof of a global optimum.",
        },
        "independent_validation": {
            "probe_seed": SEED + 1,
            "probe_count": N_VALIDATION_PROBES,
            "initial_objective": initial_validation_score,
            "final_objective": final_validation_score,
            "initial_details": initial_validation,
            "final_details": final_validation,
            "approximate_worst_pair_binomial_cov": max(
                math.sqrt((1.0 - c) / (N_VALIDATION_PROBES * c))
                for row in final_validation["pair_coverages"] for c in row if c > 0
            ),
            "note": "Independent LHS probes are used only to audit the chosen design's coverage estimate; they were not used by the annealing search.",
        },
        "annealing": {
            "initial_temperature": INITIAL_TEMPERATURE,
            "cooling_factor": COOLING,
            "final_temperature": FINAL_TEMPERATURE,
            "inner_iteration_cap": INNER_ITERATION_CAP,
            "early_stopping": False,
            "stalled_round_improvement_tolerance": IMPROVEMENT_TOLERANCE,
            **sa,
        },
        "ideal_pair_frequency_pattern_not_forced": "five cells with 6 runs and four cells with 5 runs; the published OCD penalizes imbalance but does not impose this as a hard quota",
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "design_csv": str(csv_path),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUT_DIR),
        "design_csv": str(csv_path),
        "summary": str(OUT_DIR / "summary.json"),
        "elapsed_seconds": summary["elapsed_seconds"],
        "initial_objective": sa["initial_objective"],
        "final_objective": sa["final_objective"],
        "pair_counts": sa["final_details"]["pair_counts"],
        "knee_counts": sa["final_details"]["knee_counts"],
        "ankle_counts": sa["final_details"]["ankle_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
