"""Fit the DM-J10010L-2EC thigh SolidWorks scan with independent GPs.

Requires only NumPy. Writes machine-readable predictions, nested LOOCV metrics,
an inspectable GP parameter JSON, and vector SVG plots under output/.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR
MODEL_ID = "DM-J10010L-2EC_thigh_URDF_real_curve_GP"
MOTOR_CURVE_ID = "DM-J10010L-2EC"
INPUT_CSV = ROOT_DIR / "data" / "10010L_SW_scan_20261014_234815.csv"
TRAIN_CSV = ROOT_DIR / "data" / "DM10010L_thigh_train.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
X_MIN, X_MAX, GRID_COUNT = 0.25, 0.40, 301
MEASURED_MIN, MEASURED_MAX = 0.26, 0.40

PROPERTIES = [
    ("mass_kg", "质量", "kg"),
    ("com_x_m", "质心 X", "m"),
    ("com_y_m", "质心 Y", "m"),
    ("com_z_m", "质心 Z", "m"),
    ("Lxx", "惯量 Lxx", "kg·m²"),
    ("Lxy", "惯积 Lxy", "kg·m²"),
    ("Lxz", "惯积 Lxz", "kg·m²"),
    ("Lyy", "惯量 Lyy", "kg·m²"),
    ("Lyz", "惯积 Lyz", "kg·m²"),
    ("Lzz", "惯量 Lzz", "kg·m²"),
]


def read_and_validate():
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 15:
        raise ValueError(f"预期 15 个 SW 样本，实际读到 {len(rows)} 个")
    if any(r["record_type"] != "sample" for r in rows):
        raise ValueError("CSV 含非 sample 记录")
    if any(r["coordinate_system"] != "coordinate_system_1" for r in rows):
        raise ValueError("CSV 并非全部来自坐标系1")
    if any(r["part_rebuild_ok"].lower() != "true" or
           r["assembly_rebuild_ok"].lower() != "true" for r in rows):
        raise ValueError("存在零件或装配体重建失败样本")
    x = np.array([float(r["nominal_length_m"]) for r in rows])
    delta = np.array([float(r["delta_mm"]) for r in rows])
    actual_x = np.array([float(r["length_m"]) for r in rows])
    expected_delta = np.arange(10., 151., 10.)
    if not np.allclose(np.sort(delta), expected_delta, atol=1e-8, rtol=0):
        raise ValueError("扫描增量不是完整的 10…150 mm，每 10 mm 一点")
    if len(np.unique(x)) != 15 or not np.allclose(x, 0.25 + delta / 1000., atol=1e-8, rtol=0):
        raise ValueError("腿长与 deltl 映射不一致，应为 0.25 m + deltl")
    if not np.allclose(actual_x, x, atol=1e-8, rtol=0):
        raise ValueError("length_m 与 nominal_length_m 不一致")
    if not np.allclose([x.min(), x.max()], [MEASURED_MIN, MEASURED_MAX], atol=1e-8, rtol=0):
        raise ValueError("实测范围不是 0.26…0.40 m")
    y = {name: np.array([float(r[name]) for r in rows]) for name, _, _ in PROPERTIES}
    if any(not np.all(np.isfinite(v)) for v in y.values()):
        raise ValueError("拟合数据包含 NaN 或 Inf")
    return rows, x, actual_x, delta, y


def matern52(xa, xb, length_scale):
    r = np.abs(np.subtract.outer(xa, xb)) / length_scale
    sr = math.sqrt(5.0) * r
    return (1.0 + sr + (5.0 / 3.0) * r * r) * np.exp(-sr)


def factor_kernel(k0, sigma):
    k = k0.copy()
    k.flat[:: k.shape[0] + 1] += sigma * sigma + 1e-10
    chol = np.linalg.cholesky(k)
    eye = np.eye(k.shape[0])
    kinv = np.linalg.solve(chol.T, np.linalg.solve(chol, eye))
    return chol, kinv


def marginal_nll(k0, y, sigma):
    try:
        chol, kinv = factor_kernel(k0, sigma)
    except np.linalg.LinAlgError:
        return math.inf
    one = np.ones(len(y))
    kinv_y = kinv @ y
    kinv_one = kinv @ one
    denom = float(one @ kinv_one)
    if denom <= 0 or not np.isfinite(denom):
        return math.inf
    beta = float(one @ kinv_y) / denom
    residual = y - beta
    logdet_half = float(np.log(np.diag(chol)).sum())
    return 0.5 * float(residual @ kinv @ residual) + logdet_half + 0.5 * len(y) * math.log(2.0 * math.pi)


def optimize_hyperparameters(x_std, z):
    # Profiled exact marginal likelihood, with broad then locally refined grids.
    length_grid = np.geomspace(0.10, 25.0, 25)
    noise_grid = np.geomspace(1e-5, 0.5, 19)
    best = (math.inf, None, None)

    def search(lengths, noises, current):
        best_local = current
        for ell in lengths:
            k0 = matern52(x_std, x_std, float(ell))
            for sigma in noises:
                score = marginal_nll(k0, z, float(sigma))
                if score < best_local[0]:
                    best_local = (score, float(ell), float(sigma))
        return best_local

    best = search(length_grid, noise_grid, best)
    for fraction in (0.22, 0.06):
        ell = best[1]
        sig = best[2]
        lengths = np.geomspace(max(1e-3, ell * (1-fraction)), ell * (1+fraction), 9)
        noises = np.geomspace(max(1e-7, sig * (1-fraction)), max(sig * (1+fraction), 1.01e-7), 9)
        best = search(lengths, noises, best)
    return best[1], best[2], best[0]


def fit_gp(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y, ddof=0))
    const_tol = max(1e-15, 32.0 * np.spacing(max(1.0, float(np.max(np.abs(y))))))
    is_constant = float(np.max(y) - np.min(y)) <= const_tol
    if is_constant:
        return {"constant": True, "constant_value": float(y[0]), "y_mean": y_mean,
                "y_scale": 1.0, "x_mean": float(np.mean(x)),
                "x_scale": float(np.std(x, ddof=0)), "length_scale_std": None,
                "noise_std": 0.0, "beta_std": 0.0, "alpha_std": np.zeros(len(x)),
                "x_train": x.copy(), "nll": 0.0}
    x_mean = float(np.mean(x))
    x_scale = float(np.std(x, ddof=0))
    z = (y - y_mean) / y_scale
    xs = (x - x_mean) / x_scale
    ell, sigma, nll = optimize_hyperparameters(xs, z)
    k0 = matern52(xs, xs, ell)
    _, kinv = factor_kernel(k0, sigma)
    one = np.ones(len(z))
    kinv_one = kinv @ one
    beta = float(one @ kinv @ z) / float(one @ kinv_one)
    alpha = kinv @ (z - beta)
    return {"constant": False, "constant_value": None, "y_mean": y_mean,
            "y_scale": y_scale, "x_mean": x_mean, "x_scale": x_scale,
            "length_scale_std": ell, "noise_std": sigma, "beta_std": beta,
            "alpha_std": alpha, "x_train": x.copy(), "nll": nll}


def predict_gp(model, xq):
    xq = np.asarray(xq, dtype=float)
    if model["constant"]:
        return np.full(xq.shape, model["constant_value"]), np.zeros(xq.shape)
    xs = (model["x_train"] - model["x_mean"]) / model["x_scale"]
    q = (xq - model["x_mean"]) / model["x_scale"]
    ell = model["length_scale_std"]
    k_star = matern52(q, xs, ell)
    k_train = matern52(xs, xs, ell)
    _, kinv = factor_kernel(k_train, model["noise_std"])
    zmean = model["beta_std"] + k_star @ model["alpha_std"]
    one = np.ones(len(xs))
    kinv_one = kinv @ one
    denom = float(one @ kinv_one)
    kinv_k = kinv @ k_star.T
    latent_var = 1.0 - np.sum(k_star * kinv_k.T, axis=1)
    mean_correction = 1.0 - one @ kinv_k
    latent_var += mean_correction * mean_correction / denom
    # Roundoff can make a near-zero latent variance slightly negative at a training
    # point. Clip that component before adding the nonzero fitted observation noise.
    latent_var = np.maximum(latent_var, 0.0)
    # Include the fitted observation-noise term in the predictive interval.
    pred_std = model["y_scale"] * np.sqrt(latent_var + model["noise_std"]**2)
    return model["y_mean"] + model["y_scale"] * zmean, pred_std


def metric_row(name, label, unit, actual, predicted, is_constant):
    err = predicted - actual
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    sst = float(np.sum((actual - np.mean(actual))**2))
    r2 = 1.0 - float(np.sum(err**2)) / sst if sst > np.finfo(float).eps * max(1.0, float(np.max(np.abs(actual)))) else (1.0 if max_abs == 0 else float("nan"))
    peak = float(np.max(np.abs(actual)))
    nrmse = 100.0 * rmse / peak if peak > 0 else 0.0
    return {"property": name, "label_cn": label, "unit": unit, "point_count": len(actual),
            "loo_rmse": rmse, "loo_mae": mae, "loo_max_abs_error": max_abs,
            "loo_r2": r2, "loo_nrmse_percent_peak": nrmse, "constant_data": is_constant}


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def esc(text):
    return html.escape(str(text), quote=True)


def svg_plot(path, selected, x, values, grid, means, stds, metrics, title, ncols, nrows):
    width, height = 1560, 980 if nrows == 2 else 1200
    margin_x, margin_top, margin_bottom, gap_x, gap_y = 42, 78, 72, 32, 36
    panel_w = (width - 2*margin_x - gap_x) / 2
    panel_h = (height - margin_top - margin_bottom - gap_y*(nrows-1)) / nrows
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="{width/2}" y="40" text-anchor="middle" font-family="Microsoft YaHei,Arial,sans-serif" font-size="27" font-weight="700" fill="#14395b">{esc(title)}</text>')
    colors = {"curve":"#1769aa", "band":"#a8c9e5", "dot":"#172b3a", "grid":"#dfe7ee", "axis":"#526576", "extra":"#fff0df", "marker":"#c65d16"}

    for panel_index, (key, label, unit) in enumerate(selected):
        row, col = divmod(panel_index, ncols)
        px = margin_x + col*(panel_w+gap_x)
        py = margin_top + row*(panel_h+gap_y)
        left, right, top, bottom = px+84, px+panel_w-22, py+42, py+panel_h-60
        y_all = np.concatenate([values[key], means[key]-1.96*stds[key], means[key]+1.96*stds[key]])
        y_lo, y_hi = float(np.min(y_all)), float(np.max(y_all))
        if abs(y_hi-y_lo) < max(abs(y_hi),1e-12)*1e-8:
            pad = max(abs(y_hi)*0.08, 1e-10)
            y_lo, y_hi = y_lo-pad, y_hi+pad
        else:
            pad = (y_hi-y_lo)*0.10
            y_lo, y_hi = y_lo-pad, y_hi+pad
        def X(v): return left+(float(v)-X_MIN)/(X_MAX-X_MIN)*(right-left)
        def Y(v): return bottom-(float(v)-y_lo)/(y_hi-y_lo)*(bottom-top)

        parts.append(f'<text x="{left}" y="{py+24}" font-family="Microsoft YaHei,Arial,sans-serif" font-size="17" font-weight="700" fill="#183d5d">{esc(label)} <tspan font-size="12" font-weight="400">({esc(unit)})</tspan></text>')
        # Pale band marks the 0.25-0.26 m interval with no SW observations.
        parts.append(f'<rect x="{X(X_MIN):.2f}" y="{top:.2f}" width="{X(MEASURED_MIN)-X(X_MIN):.2f}" height="{bottom-top:.2f}" fill="{colors["extra"]}"/>')
        parts.append(f'<text x="{(X(X_MIN)+X(MEASURED_MIN))/2:.2f}" y="{top+15:.2f}" text-anchor="middle" font-family="Microsoft YaHei,Arial,sans-serif" font-size="10" fill="#9c4d12">外推</text>')
        for i in range(5):
            yv = y_lo + (y_hi-y_lo)*i/4
            yy = Y(yv)
            parts.append(f'<line x1="{left:.2f}" y1="{yy:.2f}" x2="{right:.2f}" y2="{yy:.2f}" stroke="{colors["grid"]}" stroke-width="1"/>')
            label_y = f'{yv:.2e}' if max(abs(y_lo),abs(y_hi)) < 1e-2 else f'{yv:.4g}'
            parts.append(f'<text x="{left-9:.2f}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="{colors["axis"]}">{label_y}</text>')
        for xv in np.linspace(X_MIN,X_MAX,6):
            xx = X(xv)
            parts.append(f'<line x1="{xx:.2f}" y1="{top:.2f}" x2="{xx:.2f}" y2="{bottom:.2f}" stroke="{colors["grid"]}" stroke-width="1"/>')
            parts.append(f'<text x="{xx:.2f}" y="{bottom+18:.2f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="{colors["axis"]}">{xv:.2f}</text>')
        parts.append(f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="{colors["axis"]}" stroke-width="1.2"/>')
        parts.append(f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="{colors["axis"]}" stroke-width="1.2"/>')
        parts.append(f'<line x1="{X(MEASURED_MIN):.2f}" y1="{top:.2f}" x2="{X(MEASURED_MIN):.2f}" y2="{bottom:.2f}" stroke="{colors["marker"]}" stroke-width="1.1" stroke-dasharray="5 4"/>')

        upper = means[key] + 1.96*stds[key]
        lower = means[key] - 1.96*stds[key]
        polygon_points = [(X(g),Y(v)) for g,v in zip(grid,upper)] + [(X(g),Y(v)) for g,v in zip(grid[::-1],lower[::-1])]
        poly = " ".join(f"{a:.2f},{b:.2f}" for a,b in polygon_points)
        parts.append(f'<polygon points="{poly}" fill="{colors["band"]}" fill-opacity="0.56" stroke="none"/>')
        line = " ".join(f"{X(g):.2f},{Y(v):.2f}" for g,v in zip(grid,means[key]))
        parts.append(f'<polyline points="{line}" fill="none" stroke="{colors["curve"]}" stroke-width="2.4" stroke-linejoin="round"/>')
        for xx,yy in zip(x,values[key]):
            parts.append(f'<circle cx="{X(xx):.2f}" cy="{Y(yy):.2f}" r="3.2" fill="{colors["dot"]}" stroke="#ffffff" stroke-width="1"/>')
        parts.append(f'<text x="{(left+right)/2:.2f}" y="{bottom+42:.2f}" text-anchor="middle" font-family="Microsoft YaHei,Arial,sans-serif" font-size="11" fill="{colors["axis"]}">大腿长度 Lₜ (m)</text>')
        metric = metrics[key]
        parts.append(f'<text x="{right:.2f}" y="{py+24}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#526576">LOOCV RMSE {metric["loo_rmse"]:.3g}</text>')

    legend_y = height-22
    parts.append(f'<circle cx="{width/2-180}" cy="{legend_y-4}" r="3.2" fill="{colors["dot"]}"/>')
    parts.append(f'<text x="{width/2-170}" y="{legend_y}" font-family="Microsoft YaHei,Arial,sans-serif" font-size="11" fill="#526576">SW 实测点</text>')
    parts.append(f'<line x1="{width/2-66}" y1="{legend_y-4}" x2="{width/2-38}" y2="{legend_y-4}" stroke="{colors["curve"]}" stroke-width="2.4"/>')
    parts.append(f'<text x="{width/2-31}" y="{legend_y}" font-family="Microsoft YaHei,Arial,sans-serif" font-size="11" fill="#526576">GP 均值</text>')
    parts.append(f'<rect x="{width/2+52}" y="{legend_y-11}" width="24" height="13" fill="{colors["band"]}" fill-opacity="0.65"/>')
    parts.append(f'<text x="{width/2+82}" y="{legend_y}" font-family="Microsoft YaHei,Arial,sans-serif" font-size="11" fill="#526576">95% 预测区间</text>')
    parts.append(f'<rect x="{width/2+205}" y="{legend_y-11}" width="14" height="13" fill="{colors["extra"]}"/>')
    parts.append(f'<text x="{width/2+225}" y="{legend_y}" font-family="Microsoft YaHei,Arial,sans-serif" font-size="11" fill="#526576">实测范围外（0.25–0.26 m）</text>')
    parts.append('<text x="50%" y="' + str(height-4) + '" text-anchor="middle" font-family="Microsoft YaHei,Arial,sans-serif" font-size="9" fill="#72808c">15 个实测点：0.26–0.40 m。橙色区域为外推段，未由 SW 样本验证。</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    rows, x, actual_x, delta, values = read_and_validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_fields = ["model_id", "length_m", "mass_kg", "com_x_m", "com_y_m", "com_z_m",
                    "sw_com_xx_kg_m2", "sw_com_xy_kg_m2", "sw_com_xz_kg_m2",
                    "sw_com_yy_kg_m2", "sw_com_yz_kg_m2", "sw_com_zz_kg_m2"]
    train_rows = []
    for row in rows:
        train_rows.append({
            "model_id": MODEL_ID,
            "length_m": row["length_m"], "mass_kg": row["mass_kg"],
            "com_x_m": row["com_x_m"], "com_y_m": row["com_y_m"], "com_z_m": row["com_z_m"],
            "sw_com_xx_kg_m2": row["Lxx"], "sw_com_xy_kg_m2": row["Lxy"],
            "sw_com_xz_kg_m2": row["Lxz"], "sw_com_yy_kg_m2": row["Lyy"],
            "sw_com_yz_kg_m2": row["Lyz"], "sw_com_zz_kg_m2": row["Lzz"],
        })
    write_csv(TRAIN_CSV, train_rows, train_fields)
    grid = np.linspace(X_MIN, X_MAX, GRID_COUNT)
    models, means, stds, metrics = {}, {}, {}, {}
    loo_records = []

    for key, label, unit in PROPERTIES:
        y = values[key]
        full_model = fit_gp(x, y)
        mu, sd = predict_gp(full_model, grid)
        models[key] = full_model
        means[key], stds[key] = mu, sd
        loo_pred = np.empty(len(x))
        for i in range(len(x)):
            mask = np.ones(len(x), dtype=bool)
            mask[i] = False
            fold = fit_gp(x[mask], y[mask])
            loo_pred[i] = predict_gp(fold, np.array([x[i]]))[0][0]
        metrics[key] = metric_row(key, label, unit, y, loo_pred, full_model["constant"])
        for i, row in enumerate(rows):
            loo_records.append({"model_id": MODEL_ID, "sample_index": i+1, "length_m": x[i], "delta_mm": delta[i],
                                "property": key, "actual": y[i], "loo_prediction": loo_pred[i],
                                "loo_residual": loo_pred[i]-y[i]})

    dense_rows = []
    for i, length in enumerate(grid):
        rec = {"model_id": MODEL_ID, "length_m": float(length), "outside_measured_range": bool(length < MEASURED_MIN)}
        for key, _, _ in PROPERTIES:
            rec[key+"_mean"] = float(means[key][i])
            rec[key+"_std"] = float(stds[key][i])
        rec["urdf_ixx_kg_m2"] = rec["Lxx_mean"]
        rec["urdf_ixy_kg_m2"] = -rec["Lxy_mean"]
        rec["urdf_ixz_kg_m2"] = -rec["Lxz_mean"]
        rec["urdf_iyy_kg_m2"] = rec["Lyy_mean"]
        rec["urdf_iyz_kg_m2"] = -rec["Lyz_mean"]
        rec["urdf_izz_kg_m2"] = rec["Lzz_mean"]
        J = np.array([[rec["Lxx_mean"], -rec["Lxy_mean"], -rec["Lxz_mean"]],
                      [-rec["Lxy_mean"], rec["Lyy_mean"], -rec["Lyz_mean"]],
                      [-rec["Lxz_mean"], -rec["Lyz_mean"], rec["Lzz_mean"]]])
        eig = np.linalg.eigvalsh(J)
        margin = float(eig[0] + eig[1] - eig[2])
        rec["minimum_inertia_eigenvalue_kg_m2"] = float(eig[0])
        rec["triangle_margin_kg_m2"] = margin
        rec["physical_valid"] = bool(np.all(eig > 0) and margin >= -1e-12 and rec["mass_kg_mean"] > 0)
        dense_rows.append(rec)

    metric_records = [{"model_id": MODEL_ID, **row} for row in metrics.values()]
    write_csv(OUTPUT_DIR/"gp_loo_cv_metrics.csv", metric_records)
    write_csv(OUTPUT_DIR/"gp_loo_predictions.csv", loo_records)
    write_csv(OUTPUT_DIR/"gp_dense_predictions_301.csv", dense_rows)
    metadata = []
    model_json = {"model_id": MODEL_ID,
                  "variant": "DM-J10010L-2EC thigh, independent GP URDF model package",
                  "motor_curve_id": MOTOR_CURVE_ID,
                  "motor_curve_file": "../motor_curve/DM-J10010L-2EC_gp_envelope.csv",
                  "sw_input_file": INPUT_CSV.name,
                  "sw_input_sha256": hashlib.sha256(INPUT_CSV.read_bytes()).hexdigest(),
                  "segment": "thigh", "model": "independent exact Gaussian processes, Matern 5/2",
                  "coordinate_system": "coordinate_system_1",
                  "input_length_unit": "m", "prediction_domain_m": [X_MIN, X_MAX],
                  "measured_domain_m": [MEASURED_MIN, MEASURED_MAX], "n_scans": len(x), "outputs": {}}
    for key, label, unit in PROPERTIES:
        m = models[key]
        metrics_row = metrics[key]
        metadata.append({"model_id": MODEL_ID, "property": key, "label_cn": label, "unit": unit, "model_type": "constant" if m["constant"] else "GPR Matern 5/2",
                         "point_count": len(x), "data_min_m": float(x.min()), "data_max_m": float(x.max()),
                         "kernel_length_scale_m": None if m["constant"] else m["length_scale_std"]*m["x_scale"],
                         "kernel_length_scale_standardized": m["length_scale_std"], "sigma_standardized": m["noise_std"],
                         "y_center": m["y_mean"], "y_scale": m["y_scale"], "loo_rmse": metrics_row["loo_rmse"],
                         "loo_nrmse_percent_peak": metrics_row["loo_nrmse_percent_peak"]})
        model_json["outputs"][key] = {"label_cn": label, "unit": unit, "constant": m["constant"],
            "constant_value": m["constant_value"], "x_mean": m["x_mean"], "x_scale": m["x_scale"],
            "y_mean": m["y_mean"], "y_scale": m["y_scale"], "length_scale_standardized": m["length_scale_std"],
            "noise_std_standardized": m["noise_std"], "beta_standardized": m["beta_std"],
            "x_train_m": m["x_train"].tolist(), "alpha_standardized": np.asarray(m["alpha_std"]).tolist()}
    write_csv(OUTPUT_DIR/"gp_model_metadata.csv", metadata)
    (OUTPUT_DIR/"gp_models.json").write_text(json.dumps(model_json, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_count = sum(r["physical_valid"] for r in dense_rows)
    physical = [{"model_id": MODEL_ID, "dense_points": len(dense_rows), "physically_valid_points": valid_count,
                 "minimum_eigenvalue_kg_m2": min(r["minimum_inertia_eigenvalue_kg_m2"] for r in dense_rows),
                 "minimum_triangle_margin_kg_m2": min(r["triangle_margin_kg_m2"] for r in dense_rows),
                 "all_valid": valid_count == len(dense_rows)}]
    write_csv(OUTPUT_DIR/"gp_physical_check.csv", physical)

    svg_plot(OUTPUT_DIR/"DMJ10010L_thigh_GP_mass_COM.svg", PROPERTIES[:4], x, values, grid, means, stds, metrics,
             "DM-J10010L 大腿质量与质心 GP 拟合", 2, 2)
    svg_plot(OUTPUT_DIR/"DMJ10010L_thigh_GP_inertia.svg", PROPERTIES[4:], x, values, grid, means, stds, metrics,
             "DM-J10010L 大腿质心惯性参数 GP 拟合", 2, 3)

    with (OUTPUT_DIR/"README_CN.md").open("w", encoding="utf-8") as f:
        f.write("# DM-J10010L-2EC 大腿装配体 URDF 参数 GP 拟合\n\n")
        f.write(f"模型唯一标识：`{MODEL_ID}`。这是 10010L 独立保存的 GP 结构代理模型，不读取多项式模型参数。\n\n")
        f.write(f"坐标系1；输入15个SW点，实测长度 {x.min():.2f}–{x.max():.2f} m。预测域为0.25–0.40 m，0.25–0.26 m为外推区，无扫描点验证。\n\n")
        f.write(f"SW扫描输入：`data/{INPUT_CSV.name}`；SHA-256：`{hashlib.sha256(INPUT_CSV.read_bytes()).hexdigest()}`。15个样本均参与最终拟合，没有独立留出测试集；交叉验证采用嵌套留一法。早期独立多项式结果已归档在 `../../../archive_legacy/DM10010L_URDF真实曲线版/`，不作为当前预测入口。\n\n")
        f.write(f"执行器包络标识：`{MOTOR_CURVE_ID}`，独立副本保存在 `motor_curve/`。包络当前为48 V、120 N·m峰值限幅、200 rpm空载转速锚点的参数模型，并非厂家实测的完整全速扭矩包络。\n\n")
        f.write("## 任意长度直接预测\n\n`output/gp_models.json` 保存可复现的GP参数；`predict_10010l_thigh_gp.py` 从这些参数直接计算任意输入大腿长度，不需要从301点密集表中查表。示例：`python predict_10010l_thigh_gp.py --length-m 0.25 0.325 0.40`。该命令行预测器尚未自动接入外部 BO 脚本。标准化训练CSV是原始15个SW样本的便于检查副本，不是优化期间的查表文件。\n\n")
        f.write("质量、质心 xyz 和质心处惯性张量的6个独立分量分别用 Matérn 5/2 高斯过程拟合。核长度尺度和噪声项在由粗到细的对数网格上，以精确边际似然最小值选择。各输出单独标准化。该实现仅依赖 NumPy。\n\n")
        f.write("精度采用嵌套留一交叉验证：每折留出1点，并在其余14点重新选择超参数。它不是独立外部测试集。拟合观测噪声是 GP 正则化参数，不代表 SolidWorks 或实物的测量噪声。图中的区间是条件于选定超参数的 GP 预测区间，不包含超参数估计的不确定性。\n\n")
        f.write("SolidWorks CSV 的 Lxy/Lxz/Lyz 按正张量记数法输出，URDF 输出列已对非对角项取负号。使用预测值前需确认坐标系1与 URDF 大腿 link 坐标轴对齐。主轴方向未直接拟合，URDF 使用的是 link 坐标系下的惯性张量。\n\n")
        f.write(f"物理检查：预测网格有效 {valid_count}/{len(dense_rows)} 点；最小特征值 {physical[0]['minimum_eigenvalue_kg_m2']:.8g} kg·m²；最小三角不等式裕量 {physical[0]['minimum_triangle_margin_kg_m2']:.8g} kg·m²；全部有效：{physical[0]['all_valid']}。\n\n")
        f.write("## 嵌套留一交叉验证\n\n| 参数 | 单位 | RMSE | MAE | 最大绝对误差 | R² | 峰值归一化 RMSE |\n|---|---|---:|---:|---:|---:|---:|\n")
        for m in metrics.values():
            r2 = "n/a" if not np.isfinite(m["loo_r2"]) else f"{m['loo_r2']:.6f}"
            f.write(f"| {m['label_cn']} | {m['unit']} | {m['loo_rmse']:.6g} | {m['loo_mae']:.6g} | {m['loo_max_abs_error']:.6g} | {r2} | {m['loo_nrmse_percent_peak']:.4f}% |\n")
        f.write("\n## 文件\n\n- `gp_dense_predictions_301.csv`：301点预测均值、标准差、URDF惯性分量和物理检查，仅供绘图/检查。\n- `gp_loo_cv_metrics.csv`、`gp_loo_predictions.csv`：嵌套留一交叉验证精度和逐点预测。\n- `gp_model_metadata.csv`、`gp_models.json`：核参数和可复现模型参数，包含唯一模型标识。\n- `DMJ10010L_thigh_GP_mass_COM.svg`、`DMJ10010L_thigh_GP_inertia.svg`：GP矢量拟合图。\n- `fit_10010l_thigh_gp.py`：独立运行的 NumPy 拟合实现。\n- `predict_10010l_thigh_gp.py`：从 GP 参数对新腿长直接计算预测。\n")

    print(f"Fitted {len(PROPERTIES)} properties from {len(x)} scans.")
    print(f"Measured domain: {x.min():.2f}..{x.max():.2f} m; prediction domain: {X_MIN:.2f}..{X_MAX:.2f} m")
    print(f"Physical validity: {valid_count}/{len(dense_rows)} dense points")
    for m in metrics.values():
        unit_ascii = m["unit"].replace("·", "*").replace("²", "^2")
        print(f"{m['property']:8s} RMSE={m['loo_rmse']:.6g} {unit_ascii}, R2={m['loo_r2']:.6f}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
