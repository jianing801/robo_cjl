"""Finite fixed-topology morphology bank, contracts and serial curriculum.

Pure preparation helpers: importing this module never launches Isaac/training.
"""
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

KNEES = ("DM-J10010L-2EC", "DM-J10010-2EC")
ANKLES = ("DM-J4340P-2EC", "DM-J8009P-2EC-V1.1")
THIGH_RANGE = (.25, .27)
CALF_RANGE = (.30, .32)
TRAIN_LENGTHS = ((.25, .30), (.25, .32), (.27, .30), (.27, .32), (.26, .31))
HOLDOUT_LENGTHS = ((.255, .305), (.255, .315), (.265, .305), (.265, .315))
CONTEXT_WIDTH = 22  # lengths2, motor one-hot4, torque/speed/armature6, envelope samples10
KIND = "rpo_two_multimorph_experts_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def morphology_rows(knees=KNEES, ankles=ANKLES, *, ranges=None):
    if len(knees) != 2 or len(set(knees)) != 2 or len(ankles) != 2 or len(set(ankles)) != 2:
        raise ValueError("exactly two distinct motors per joint group required")
    if ranges is None or ranges == {"thigh": list(THIGH_RANGE), "calf": list(CALF_RANGE)}:
        train, holdout = TRAIN_LENGTHS, HOLDOUT_LENGTHS
    else:
        if set(ranges) != {"thigh", "calf"}:
            raise ValueError("explicit thigh/calf bounds required")
        for key, bounds in ranges.items():
            allowed = (.25,.40) if key == "thigh" else (.30,.45)
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or not allowed[0] <= bounds[0] < bounds[1] <= allowed[1]:
                raise ValueError("bounds exceed pinned physical prediction range")
        t0,t1 = ranges["thigh"]; c0,c1 = ranges["calf"]
        train = ((t0,c0),(t0,c1),(t1,c0),(t1,c1),((t0+t1)/2,(c0+c1)/2))
        holdout = tuple((t0+(t1-t0)*u,c0+(c1-c0)*v) for u,v in itertools.product((.25,.75),repeat=2))
    rows = []
    for split, lengths in (("train", train), ("holdout", holdout)):
        for (thigh, calf), knee, ankle in itertools.product(lengths, knees, ankles):
            item = {"split": split, "thigh_m": thigh, "calf_m": calf,
                    "knee_motor": knee, "ankle_motor": ankle}
            item["id"] = ("tr_" if split == "train" else "ho_") + digest(item)[:12]
            rows.append(item)
    return rows


def _interpolate(points, x):
    if x <= points[0][0]:
        return points[0][1]
    for (a, y), (b, z) in zip(points, points[1:]):
        if x <= b:
            return y + (z - y) * (x - a) / (b - a)
    return points[-1][1]


def design_context(row, physical, *, knees=KNEES, ankles=ANKLES, ranges=None):
    """Known robot design, not a new sensor or privileged dynamic state."""
    from robolab.physical_snapshot import catalog
    ranges = ranges or {"thigh": THIGH_RANGE, "calf": CALF_RANGE}
    if not ranges["thigh"][0] <= row["thigh_m"] <= ranges["thigh"][1] or not ranges["calf"][0] <= row["calf_m"] <= ranges["calf"][1]:
        raise ValueError("leg length outside authorized box")
    if row["knee_motor"] != physical["knee_motor"] or row["ankle_motor"] != physical["ankle_motor"]:
        raise ValueError("context must identify the actual physical motor pair")
    if tuple(ranges["thigh"]) == THIGH_RANGE and tuple(ranges["calf"]) == CALF_RANGE:
        values = [(row["thigh_m"]-.26)/.01,(row["calf_m"]-.31)/.01]
    else:
        values = [(row[field]-(ranges[seg][0]+ranges[seg][1])/2)/((ranges[seg][1]-ranges[seg][0])/2)
                  for field,seg in (("thigh_m","thigh"),("calf_m","calf"))]
    values += [float(row["knee_motor"] == k) for k in knees]
    values += [float(row["ankle_motor"] == k) for k in ankles]
    module = catalog(physical["snapshot_root"])
    profiles = []
    for label in ("knee", "ankle"):
        selected = physical["selected"][label]
        values += [selected["peak_torque_nm"] / 150., selected["max_speed_rad_s"] / 25., selected["armature_kgm2"] / .1]
        curve = module.load_torque_speed_envelope(module.get_motor_spec(physical[label + "_motor"]))
        profiles += [_interpolate(curve, f * selected["max_speed_rad_s"]) / selected["peak_torque_nm"]
                     for f in (0., .25, .5, .75, 1.)]
    values += profiles
    if len(values) != CONTEXT_WIDTH or not all(math.isfinite(x) for x in values):
        raise ValueError("invalid design conditioning vector")
    return values


def semantic_contract(contract):
    # Geometry is explicitly carried in each observation, not overwritten into
    # a fixed checkpoint buffer. All control/sensor semantics remain strict.
    return {k: v for k, v in contract.items() if k not in ("urdf_sha256", "mtg")}


def block_schedule(rows, *, updates, block_updates, seed, initial_sweep_updates=None):
    if type(updates) is not int or type(block_updates) is not int or min(updates, block_updates) < 1:
        raise ValueError("positive integer update budgets required")
    if initial_sweep_updates is not None and (type(initial_sweep_updates) is not int or not 1 <= initial_sweep_updates <= block_updates):
        raise ValueError("initial sweep size must be between 1 and the normal block size")
    train = [r["id"] for r in rows if r["split"] == "train"]
    if len(set(train)) != len(train) or not train or updates < len(train):
        raise ValueError("each training morphology needs a visit; no duplicates")
    # Each sweep covers every training design before repeating; final partial
    # sweep is explicit. Holdouts never enter this sampler.
    remaining, schedule, rng, sweep = updates, [], random.Random(seed), 0
    while remaining:
        order = train.copy()
        rng.shuffle(order)
        for index, design in enumerate(order):
            # Guarantee at least one update for all still-unvisited designs in
            # the first sweep even if the requested budget is very short.
            reserve = len(order) - index - 1 if sweep == 0 else 0
            block_size = initial_sweep_updates if sweep == 0 and initial_sweep_updates is not None else block_updates
            count = min(block_size, remaining - reserve)
            if count <= 0:
                break
            schedule.append({"block": len(schedule), "design_id": design,
                             "first_update": updates - remaining, "updates": count, "sweep": sweep})
            remaining -= count
            if not remaining:
                break
        sweep += 1
    return schedule


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def execution_budget(bank, bank_path):
    """Optional immutable stop-only revision; never relabel a physical bank/checkpoint.

    Preserve the original curriculum prefix, including its shuffle order. The
    last block may be shortened (e.g. Rough's original 9001 updates).
    """
    from robolab.mxt_design_transfer import file_sha
    path = Path(bank_path).resolve().with_name("training_budget.json")
    limits = {task: bank["updates_per_expert"] for task in ("run", "stairs")}
    identity = None
    if path.exists():
        revision = json.loads(path.read_text(encoding="utf8"))
        if (revision.get("kind") != "rpo_multimorph_stop_budget_v1"
                or revision.get("bank_sha256") != file_sha(bank_path)
                or set(revision.get("task_updates", {})) != set(limits)):
            raise ValueError("stop-budget revision has wrong bank or task identity")
        limits = revision["task_updates"]
        minimum = sum(r["split"] == "train" for r in bank["designs"])
        if any(type(n) is not int or not minimum <= n <= bank["updates_per_expert"] for n in limits.values()):
            raise ValueError("stop-budget must cover training designs and cannot extend original budget")
        identity = {"path": str(path), "sha256": file_sha(path), "task_updates": limits}
    schedules = {}
    for task, limit in limits.items():
        schedules[task] = []
        for block in bank["schedules"][task]:
            if block["first_update"] >= limit:
                break
            schedules[task].append({**block, "updates": min(block["updates"], limit-block["first_update"])})
        if sum(b["updates"] for b in schedules[task]) != limit:
            raise ValueError("stop budget does not match original curriculum prefix")
        train = {r["id"] for r in bank["designs"] if r["split"] == "train"}
        if {b["design_id"] for b in schedules[task]} != train:
            raise ValueError("stop budget would omit training designs")
    return limits, schedules, identity


def load_bank(path, *, validate_assets=True):
    from robolab.mxt_design_transfer import file_sha
    from robolab.physical_snapshot import validate_physical_model
    value = json.loads(Path(path).read_text(encoding="utf8"))
    if value.get("kind") != KIND or value.get("schema_version") != 1:
        raise ValueError("wrong multi-morphology bank version")
    ids = [r["id"] for r in value["designs"]]
    if len(ids) != len(set(ids)) or value["designs"] == []:
        raise ValueError("duplicate or missing morphology")
    if [(r["id"], r["split"], r["thigh_m"], r["calf_m"], r["knee_motor"], r["ankle_motor"])
        for r in value["designs"]] != [(r["id"], r["split"], r["thigh_m"], r["calf_m"], r["knee_motor"], r["ankle_motor"])
                                     for r in morphology_rows(value["knees"], value["ankles"], ranges=value.get("ranges_m"))]:
        raise ValueError("morphology train/holdout partition changed")
    for task in ("run", "stairs"):
        if value["schedules"][task] != block_schedule(value["designs"], updates=value["updates_per_expert"],
            block_updates=value["block_updates"], seed=value["seed"] + (task == "stairs"),
            initial_sweep_updates=value.get("training", {}).get("initial_sweep_updates")):
            raise ValueError("frozen update budget / curriculum changed")
    if validate_assets:
        if file_sha(value["original_template"]) != value["original_template_sha256"]:
            raise ValueError("original template changed")
        for physical in value["motor_pairs"].values():
            validate_physical_model(physical)
        for row in value["designs"]:
            if file_sha(row["urdf"]) != row["urdf_sha256"] or file_sha(row["report"]) != row["report_sha256"]:
                raise ValueError("prepared URDF or report changed")
            expected = design_context(row, value["motor_pairs"][row["pair_id"]], knees=value["knees"], ankles=value["ankles"], ranges=value.get("ranges_m"))
            if row["context"] != expected:
                raise ValueError("physical design / context mismatch")
    return value


def checkpoint_identity(bank_sha, task, completed, *, bank, design_id):
    if task not in ("run", "stairs") or not any(r["id"] == design_id and r["split"] == "train" for r in bank["designs"]):
        raise ValueError("expert training must never use a holdout")
    if not 0 <= completed <= bank["updates_per_expert"]:
        raise ValueError("expert update budget exceeded")
    return {"kind": KIND, "bank_sha256": bank_sha, "task": task,
            "completed_updates": completed, "last_design_id": design_id,
            "environment_reset_on_switch": True, "physical_trajectory_not_resumed": True}
