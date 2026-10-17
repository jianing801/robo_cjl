"""Evaluate the fitted RS04 thigh GP at arbitrary leg lengths.

The 301-row dense CSV is for plotting/reporting only. This module evaluates the
serialized GP parameters directly, so an optimizer can query any allowed length.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "output" / "gp_models.json"


def matern52(xa, xb, length_scale):
    r = np.abs(np.subtract.outer(xa, xb)) / length_scale
    sr = math.sqrt(5.0) * r
    return (1.0 + sr + (5.0 / 3.0) * r * r) * np.exp(-sr)


def predict(lengths_m):
    """Return GP mean/std for requested lengths (meters), without table lookup."""
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    lengths = np.atleast_1d(np.asarray(lengths_m, dtype=float))
    low, high = model["prediction_domain_m"]
    if not np.all(np.isfinite(lengths)) or np.any(lengths < low) or np.any(lengths > high):
        raise ValueError(f"Query lengths must be finite and within [{low}, {high}] m")

    result = []
    for length in lengths:
        row = {
            "model_id": model["model_id"],
            "length_m": float(length),
            "outside_measured_range": bool(length < model["measured_domain_m"][0] or
                                            length > model["measured_domain_m"][1]),
            "properties": {},
        }
        for name, fitted in model["outputs"].items():
            if fitted["constant"]:
                mean = float(fitted["constant_value"])
                std = 0.0
            else:
                x_train = np.asarray(fitted["x_train_m"], dtype=float)
                x_scale = float(fitted["x_scale"])
                x_mean = float(fitted["x_mean"])
                xs = (x_train - x_mean) / x_scale
                q = np.array([(float(length) - x_mean) / x_scale])
                ell = float(fitted["length_scale_standardized"])
                noise = float(fitted["noise_std_standardized"])
                k0 = matern52(xs, xs, ell)
                k = k0.copy()
                k.flat[::k.shape[0] + 1] += noise * noise + 1e-10
                chol = np.linalg.cholesky(k)
                k_star = matern52(q, xs, ell)
                alpha = np.asarray(fitted["alpha_standardized"], dtype=float)
                z_mean = float(fitted["beta_standardized"] + (k_star @ alpha).item())
                mean = float(fitted["y_mean"] + fitted["y_scale"] * z_mean)

                one = np.ones(len(xs))
                whitened_one = np.linalg.solve(chol, one)
                denom = float(whitened_one @ whitened_one)
                whitened_cross = np.linalg.solve(chol, k_star.T)
                latent_var = float(1.0 - np.sum(whitened_cross * whitened_cross, axis=0)[0])
                mean_correction = float((1.0 - whitened_one @ whitened_cross).item())
                latent_var = max(0.0, latent_var + mean_correction * mean_correction / denom)
                std = float(fitted["y_scale"] * math.sqrt(latent_var + noise * noise))
            row["properties"][name] = {"mean": mean, "std": std}

        p = row["properties"]
        row["urdf_inertia_kg_m2"] = {
            "ixx": p["Lxx"]["mean"], "ixy": -p["Lxy"]["mean"],
            "ixz": -p["Lxz"]["mean"], "iyy": p["Lyy"]["mean"],
            "iyz": -p["Lyz"]["mean"], "izz": p["Lzz"]["mean"],
        }
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length-m", type=float, nargs="+", required=True,
                        help="One or more thigh lengths in meters, within 0.25–0.40")
    args = parser.parse_args()
    print(json.dumps(predict(args.length_m), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
