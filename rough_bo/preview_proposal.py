"""Fit the two CPU GPs and preview ONE proposal without training or simulation.

This checks a transported observation manifest internally. Only the foreground
controller validates it against the actual checkpoint/bank before execution.
"""
import argparse
import json
from pathlib import Path

from bo_finetune_common import atomic_json, sha256_file
from constrained_gp import ConstrainedSurrogate
from repair_observations import load_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True,
                        help="Original bank JSON for search bounds/categories; SHA256 must match manifest")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.observations.read_text(encoding="utf-8"))
    if sha256_file(args.bank) != raw["bank_sha256"]:
        raise ValueError("preview bank differs from observations")
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    document = load_observations(args.observations,
        {"checkpoint_sha256":raw["base_checkpoint_sha256"], "bank_sha256":raw["bank_sha256"]},
        seed=raw["seed"], eval_seed=raw["eval_seed"],
        learning_rate=raw["optimizer_initial_learning_rate"])
    observations = [{"params":o["metrics"]["params"], "feasible":o["metrics"]["feasible"],
                     "mae":o["metrics"]["objective_value"]} for o in document["observations"]]
    print(f"Fitting performance GP on {sum(o['feasible'] for o in observations)} successes; "
          f"feasibility GPC on all {len(observations)} outcomes", flush=True)
    params, audit = ConstrainedSurrogate(bank["ranges_m"], bank["knees"], bank["ankles"],
                                       seed=raw["seed"]).fit(observations).propose()
    atomic_json(args.output, {"params":params, **audit, "preview_only":True,
        "observation_manifest_sha256":sha256_file(args.observations),
        "checkpoint_loaded":False, "optimization_updates":0,
        "requires_controller_identity_validation":True})
    print(json.dumps({"params":params, "predicted_feasibility":audit["predicted_feasibility"],
                      "method":audit["method"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
