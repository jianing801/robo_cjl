"""Validate design-specific reference changes without relaxing native rewards."""
from copy import deepcopy


def validate_continuation_contract(previous, current, *, bank, task, previous_design, current_design):
    binding = bank.get("training", {}).get("retargeted_references")
    if not binding:
        if previous != current:
            raise ValueError("native reward contract changed")
        return
    if task != "run":
        raise ValueError("retargeted AMP references are RUN-only")
    from robolab.amp_reference_retarget import approved_dataset
    stripped = []
    for contract, design_id in ((previous, previous_design), (current, current_design)):
        design = next(d for d in bank["designs"] if d["id"] == design_id and d["split"] == "train")
        record = approved_dataset(bank, design, {n: c["weight"] for n, c in contract["motion_files"].items()})
        expected = {"design_id": design_id, "urdf_sha256": design["urdf_sha256"],
                    "library": binding, "directory": record["directory"],
                    "review_sha256": record["review_sha256"], "dynamic_feasibility_validated": False}
        clips = {n: {"weight": record["weights"][n], "sha256": c["sha256"]}
                 for n, c in record["clips"].items()}
        if contract.get("retargeted_reference_binding") != expected or contract.get("motion_files") != clips:
            raise ValueError("checkpoint reference does not match its own frozen design library")
        invariant = deepcopy(contract)
        del invariant["retargeted_reference_binding"]
        del invariant["motion_files"]
        stripped.append(invariant)
    if stripped[0] != stripped[1]:
        raise ValueError("native rewards/algorithm contract changed beyond authorized reference switch")
