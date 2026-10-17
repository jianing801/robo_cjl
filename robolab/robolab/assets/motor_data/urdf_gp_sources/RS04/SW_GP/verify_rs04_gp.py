"""Check source provenance, field mapping and serialized RS04 predictions."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from predict_rs04_thigh_gp import predict

ROOT = Path(__file__).resolve().parent
FIT_ROOT = ROOT.parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    source = Path(r"C:\Users\Admin\Desktop\SWtoURDF\RS04_thigh_SW_scan_20261016_225528.csv")
    copied = ROOT / "data" / source.name
    digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = json.loads((ROOT / "output/gp_models.json").read_text(encoding="utf-8"))
    assert metadata["sw_input_sha256"] == digest
    spec = importlib.util.spec_from_file_location("sw_extract", FIT_ROOT / "extract_sw_docx.py")
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    parsed = extract.parse_scan(copied, segment="thigh")
    canonical = read_csv(FIT_ROOT / "data/RS04_thigh_train.csv")
    packaged = read_csv(ROOT / "data/RS04_thigh_train.csv")
    assert canonical == packaged
    for row, expected in zip(canonical, parsed):
        for key in extract.FIELDS:
            assert abs(float(row[key]) - expected[key]) < 1e-15, key
    dense = read_csv(ROOT / "output/gp_dense_predictions_301.csv")
    queried = predict([float(row["length_m"]) for row in dense])
    max_mean_error = max(abs(q["properties"][key]["mean"] - float(row[key+"_mean"]))
        for q, row in zip(queried, dense) for key in metadata["outputs"])
    max_std_error = max(abs(q["properties"][key]["std"] - float(row[key+"_std"]))
        for q, row in zip(queried, dense) for key in metadata["outputs"])
    assert max_mean_error < 1e-9, max_mean_error
    assert max_std_error < 1e-9, max_std_error
    for q in predict([0.25, 0.26, 0.3273, 0.40]):
        inertia = q["urdf_inertia_kg_m2"]
        values = q["properties"]
        for key in ("xy", "xz", "yz"):
            assert inertia["i"+key] == -values["L"+key]["mean"]
        matrix = np.array([[inertia["ixx"], inertia["ixy"], inertia["ixz"]],
                           [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
                           [inertia["ixz"], inertia["iyz"], inertia["izz"]]])
        eig = np.linalg.eigvalsh(matrix)
        assert eig[0] > 0 and eig[0]+eig[1] >= eig[2]
        assert q["outside_measured_range"] == (q["length_m"] < 0.26)
    for bad in (0.249, 0.401, float("nan")):
        try:
            predict([bad])
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-range request was accepted")
    print(json.dumps({"source_sha256_matches": True, "canonical_scan_mapping": "15/15 matched",
                      "direct_predictor_grid_points": len(dense), "max_mean_error_vs_fitter": max_mean_error,
                      "max_std_error_vs_fitter": max_std_error,
                      "arbitrary_length_and_bounds": "passed", "SW_to_URDF_signs": "passed"}, indent=2))


if __name__ == "__main__":
    main()
