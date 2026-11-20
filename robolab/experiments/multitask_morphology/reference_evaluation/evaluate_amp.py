"""Measure a frozen AMP expert on the fixed one-env-per-design BO protocol."""
import argparse
from collections import Counter, defaultdict
from copy import copy, deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

REPO = Path(__file__).resolve().parents[4]
PROJECT = Path(os.environ.get("ROBO_REFERENCE_PROJECT", REPO / "robolab/logs/multitask_full2500_physical_migrated")).resolve()
EXPERT = PROJECT / "experts/amp"
ENTRY = REPO / "robolab/experiments/multitask_morphology"
ENGINE = ENTRY / "engine"
SHARED = ENTRY / "shared_policy"
NATIVE = ENTRY / "amp_engine"
CACHE = PROJECT / "资产准备/amp/native_cache_portable"
EXPECTED_MODEL = "e2116d5d173955dba8ac446825110ada24f8f96d0a3244c7708f3b510b6995d0"
SEEDS = (20261118, 20261119, 20261120)
REFERENCE_SELECTION = REPO / "docs/reference/rough_amp_expert_20261121/amp/selection.json"

sys.path[:0] = [str(SHARED), str(ENGINE), str(REPO/"robolab"), str(REPO/"rsl_rl")]
from contract import load_plan, sha, structure_key
from shared_bo_metrics import METRICS, digest, read_conditions

FIELDS = ("id", "thigh_m", "calf_m", "knee_motor", "ankle_motor")
COSTS = METRICS["run"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, data):
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        name = stream.name
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(name, path)


def freeze(path, data):
    if path.is_file():
        if read(path) != data:
            raise ValueError(f"Frozen inputs differ: {path}")
    else:
        atomic(path, data)


def select_group(bank, configuration):
    from hybrid_schedule import load_schedule
    schedule, path = load_schedule(bank, configuration)
    structures = {d["id"]: d for d in bank["designs"]}
    assert len(structures) == 2500
    ranges = bank["ranges_m"]

    def coverage(rows):
        cells = defaultdict(Counter)
        for row in rows:
            t = min(2, int(3*(row["thigh_m"]-ranges["thigh"][0]) /
                           (ranges["thigh"][1]-ranges["thigh"][0])))
            c = min(2, int(3*(row["calf_m"]-ranges["calf"][0]) /
                           (ranges["calf"][1]-ranges["calf"][0])))
            for key in ("all", "knee/"+row["knee_motor"], "ankle/"+row["ankle_motor"]):
                cells[key][(t, c)] += 1
        return {k: [counts[(t, c)] for t in range(3) for c in range(3)]
                for k, counts in cells.items()}

    def tv(a, b):
        return sum(abs(x/sum(a)-y/sum(b)) for x, y in zip(a, b, strict=True))/2

    whole = coverage(structures.values())
    candidates = []
    for index, group in enumerate(schedule["groups"], 1):
        rows = [structures[design] for design in group]
        hist = coverage(rows)
        by_motor = [tv(hist[name], target) for name, target in whole.items() if name != "all"]
        cells = sum(x > 0 for x in hist["all"])
        leg_pairs = len({(row["thigh_m"], row["calf_m"]) for row in rows})
        rank = (-cells, round(tv(hist["all"], whole["all"]), 10),
                round(sum(by_motor)/len(by_motor), 10), round(max(by_motor), 10), -leg_pairs, index)
        candidates.append((rank, index, rows, hist))
    rank, index, chosen, hist = min(candidates, key=lambda item: item[0])
    return dict(
        group=index, schedule_sha256=sha(path), group_count=len(candidates),
        selection_order=[
            "maximize covered two-dimensional length regions", "minimize group total variation vs whole bank",
            "minimize mean then worst per-motor total variation",
            "maximize distinct length pairs", "earliest group on exact tie",
        ],
        rank=list(rank), selected_counts_3x3=hist, bank_counts_3x3=whole,
        structures=[{key: d[key] for key in FIELDS} for d in chosen],
    )


def setup(out):
    teacher = read(EXPERT/"cloud_import/evaluation_teacher.json")
    done = read(EXPERT/"completed.json")
    training_plan = read(EXPERT/"plan.json")
    assert done["completed_updates"] == 5000
    assert done["checkpoint"] == teacher["checkpoint"]
    assert sha(teacher["checkpoint"]) == teacher["checkpoint_sha256"] == EXPECTED_MODEL
    assert Path(teacher["bank"]).resolve() == PROJECT/"banks/amp/bank.json"
    assert sha(teacher["bank"]) == teacher["bank_sha256"]
    assert sha(teacher["reference_library"]) == teacher["reference_library_sha256"]
    assert teacher["full_body_torque"] and teacher["asset_cache"] == [str(CACHE)]
    bank = read(teacher["bank"])
    selection = select_group(bank, training_plan["configuration"])
    chosen = selection["structures"]
    frozen = read(REFERENCE_SELECTION)
    if selection["group"] != frozen["group"] or [structure_key(r) for r in chosen] != [structure_key(r) for r in frozen["structures"]]:
        raise ValueError("AMP selection no longer matches the published structure order")
    keys = [list(structure_key(row)) for row in chosen]
    bank_snapshot = out/"selection.json"
    freeze(bank_snapshot, selection)
    schema = {key: bank[key] for key in ("knees", "ankles", "context_schema", "context_width")}
    schema.update(tasks=["run"], shared_structure_keys=keys)
    teacher["design_ids"] = [row["id"] for row in chosen]
    teacher["copies"] = 1
    plan = dict(kind="rough_backpack_amp_distillation_plan_v3", structure_scope="teacher_intersection_only",
                selected_structure_keys=keys, task_ids={"run":0}, teachers={"run":teacher}, schema=schema,
                shared_bank=str(bank_snapshot), shared_bank_sha256=sha(bank_snapshot))
    plan_path = out/"evaluation_plan.json"
    freeze(plan_path, plan)
    load_plan(plan_path)
    cfg = read(ENTRY/"configs/bo_pilot_conditions.json")
    cfg.update(duration_s=20., seeds=list(SEEDS), copies=1)
    cfg["tasks"]["run"].update(speed_mps=1.5)
    cfg_path = out/"conditions.json"
    freeze(cfg_path, cfg)
    read_conditions(cfg_path)
    return plan_path, cfg_path, teacher, selection


def deterministic_profile(configure_profile):
    def configure(cfg, task, **kwargs):
        result, audit = configure_profile(cfg, task, **kwargs)
        assert task == "run" and kwargs["profile"] == "nominal" and kwargs["strength"] == 0
        assert kwargs["reset_yaw_range"] == 0
        base = result.events.reset_base.params
        for name in ("pose_range", "velocity_range"):
            ranges = {key:(0., 0.) for key in base[name]}
            base[name] = ranges
        joints = result.events.reset_robot_joints.params
        joints.update(position_range=(1.,1.), velocity_range=(0.,0.))
        result.events.base_external_force_torque = None
        active_reset = {name for name, event in vars(result.events).items()
                        if getattr(event, "mode", "") == "reset"}
        assert active_reset == {"reset_base", "reset_robot_joints"}, active_reset
        result.animation.animation.random_initialize = False
        audit["reset"] = dict(base=deepcopy(base), joint_position=(1.,1.), joint_velocity=(0.,0.))
        audit["fixed_reference"] = dict(clip_index_per_structure=0, start_seconds=0.)
        audit["events"]["base_external_force_torque"] = dict(enabled=False)
        audit["deterministic_evaluation"] = True
        return result, audit
    return configure


def worker(args):
    import bo_capture
    import rollout
    import rollout_amp

    spec = load_plan(args.plan)["teachers"]["run"]
    bank_path = Path(spec["bank"])
    bank = read(bank_path)
    ids = args.ids.split(",")
    assert len(ids) == len(set(ids)) and set(ids).issubset(spec["design_ids"])
    rollout_amp.configure_profile = deterministic_profile(rollout_amp.configure_profile)
    normal_parse = bo_capture.parse_conditions
    normal_capture = bo_capture.TrialCapture

    def parse_batch(cfg, selected):
        assert cfg.student is None and selected == ids
        conditions = normal_parse(cfg, selected[:1])
        assert conditions["copies"] == 1
        return conditions

    class BatchCapture(normal_capture):
        def __init__(self, raw, task, conditions, seed, motors):
            assert task == "run" and raw.num_envs == len(ids)
            routed = raw.cfg.spatial_reference_designs
            assert [x["id"] for x in routed] == ids
            super().__init__(raw, task, dict(conditions, copies=len(ids)), seed, motors)
            self.conditions = conditions
            self.design_motors = [bo_capture.motor_records(raw, bank, row) for row in routed]
            assert self.design_motors[0] == motors

        def publish(self, output, identity):
            self.raw.bo_recorder = None
            for index, design in enumerate(ids):
                child = output/design
                child.mkdir()
                single = copy(self)
                single.count = 1
                single.motors = self.design_motors[index]
                choice = slice(index, index+1)
                single.samples = [tuple(array[choice] for array in sample) for sample in self.samples]
                for name in ("initial", "origins", "mass", "active", "fell", "timeout"):
                    setattr(single, name, getattr(self, name)[choice])
                normal_capture.publish(single, child, dict(identity, design_id=design))

    def use_verified_assets(cfg, roots, output, task, designs):
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        from amp_rough_alignment import rough_module
        assert task == "run" and roots == [str(CACHE)] and [d["id"] for d in designs] == ids
        paths = rough_module("hybrid_native_cache").cache_directories(
            CACHE, bank_path, ids, require_complete=True, training_fast=True)
        directories = {path.name:path for path in paths}
        assert set(directories) == set(ids)
        records = []
        for asset, design in zip(cfg.scene.robot.spawn.assets_cfg, designs, strict=True):
            directory = directories[design["id"]]
            exact_hash = AssetConverterBase._config_to_hash(asset)
            if (directory/".asset_hash").read_text().strip() != exact_hash:
                raise ValueError(f"Cached USD converter identity differs for {design['id']}")
            asset.usd_dir = str(directory)
            asset.force_usd_conversion = False
            records.append(dict(design=design["id"], cache=str(directory), exact_hash=exact_hash, reused=True))
        return records

    # Import after AppLauncher initializes Kit. Reference routing is pinned only
    # in this evaluation process, with its own physical motion library per robot.
    configure = rollout_amp.configure_profile
    def fixed_reference(cfg, task, **kwargs):
        import torch
        from reference_router import ReferenceRouter
        def first_motion(router, env_ids):
            offsets = torch.as_tensor(router.offsets[:-1], device=router.device, dtype=torch.long)
            return offsets[router.env_library[env_ids]]
        ReferenceRouter.sample_motions_for_envs = first_motion
        return configure(cfg, task, **kwargs)

    rollout_amp.configure_profile = fixed_reference
    bo_capture.parse_conditions = parse_batch
    bo_capture.TrialCapture = BatchCapture
    rollout.reuse_assets = use_verified_assets
    sys.argv = [
        str(SHARED/"rollout_amp.py"), "--plan", str(args.plan), "--design-ids", args.ids,
        "--output", str(args.output), "--bo-conditions", str(args.conditions), "--mode", "evaluate",
        "--profile", "nominal", "--randomization-strength", "0", "--reset-yaw-range", "0",
        "--seed", str(args.seed), "--steps", "1000", "--headless", "--device", "cuda:0",
        f"--kit_args=--/renderer/activeGpu={args.gpu} --/physics/cudaDevice=0",
    ]
    return rollout_amp.main()


def accepted_trials(output, ids, seed, plan, cfg, teacher):
    summary = read(output/"summary.json")
    audit = read(output/"randomization_audit.json")
    assert summary["result"] == "REAL_AMP_ROLLOUT_COMPLETE"
    assert summary["design_ids"] == ids and summary["seed"] == seed and summary["steps"] == 1000
    assert summary["student"] is None and summary["teacher"] == EXPECTED_MODEL
    assert summary["actor_normalizer_unchanged"] and summary["teacher_updates"] == 0
    assert summary["reference_routing_verified"] and not (output/"failure.json").exists()
    assert audit["deterministic_evaluation"] and audit["native_teacher_noise"] is False
    assert all(event["enabled"] is False for event in audit["events"].values())
    assert all(list(pair) == [0.,0.] for pair in audit["reset"]["base"]["pose_range"].values())
    assert all(list(pair) == [0.,0.] for pair in audit["reset"]["base"]["velocity_range"].values())
    assert audit["reset"]["joint_position"] == [1.,1.] and audit["reset"]["joint_velocity"] == [0.,0.]
    assert audit["commands"]["ranges"]["lin_vel_x"] == [1.5,1.5]
    assert audit["commands"]["ranges"]["lin_vel_y"] == [0.,0.]
    assert audit["commands"]["ranges"]["ang_vel_z"] == [0.,0.]
    assert audit["fixed_reference"] == {"clip_index_per_structure":0, "start_seconds":0.}
    rows = []
    for design in ids:
        report = read(output/design/"bo_trials.json")
        assert report["design_id"] == design and report["task"] == "run" and report["seed"] == seed
        assert report["bank_sha256"] == teacher["bank_sha256"]
        assert report["actor_sha256"] == EXPECTED_MODEL
        assert report["conditions_sha256"] == digest(read(cfg)) and report["plan_sha256"] == sha(plan)
        assert sha(report["trace"]) == report["trace_sha256"]
        assert len(report["trials"]) == 1
        row = report["trials"][0]
        if row["status"] not in ("success", "failure"):
            raise RuntimeError(f"Invalid trace for {design}, {seed}: {row['reason']}")
        assert row["command_mps"] == 1.5 and row["dt_s"] == .02
        if row["status"] == "success":
            assert row["steps"] == 1000 and row["metrics"] is not None
            assert set(row["metrics"]) == set(COSTS)
        assert row["motors"] is None or len(row["motors"]) == 23
        rows.append(dict(design_id=design, seed=seed, report=str(output/design/"bo_trials.json"), **row))
    return rows


def trial_batch(root, label, ids, seed, plan, cfg, teacher, gpu):
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    batch = root/"batches"/label
    batch.mkdir(parents=True, exist_ok=True)
    accepted = batch/"accepted.json"
    if accepted.is_file():
        return accepted_trials(Path(read(accepted)["output"]), ids, seed, plan, cfg, teacher)
    for _ in range(2):
        attempt = batch/f"attempt_{len(list(batch.glob('attempt_*')))+1:02d}"
        attempt.mkdir()
        output = attempt/"native"
        cards = candidates(query_gpus(), os.environ, min_free_mib=12.5*1024,
                           max_utilization=15, requested=f"cuda:{gpu}")
        with gpu_lease(cards, REPO/"robolab/logs/rpo_rough_multimorph/gpu_locks") as device:
            command = [
                sys.executable, "-u", __file__, "worker", "--plan", str(plan), "--conditions", str(cfg),
                "--ids", ",".join(ids), "--seed", str(seed), "--output", str(output),
                "--gpu", str(device["index"]),
            ]
            atomic(attempt/"command.json", command)
            print(f"AMP_EVAL_START {label} seed={seed} designs={len(ids)} gpu={device['index']}", flush=True)
            env = child_environment(device, os.environ)
            with (attempt/"stdout.log").open("x", encoding="utf-8") as stdout, \
                 (attempt/"stderr.log").open("x", encoding="utf-8") as stderr:
                try:
                    result = subprocess.run(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, timeout=1800)
                    if result.returncode:
                        print(f"AMP_EVAL_RETRY {label} rc={result.returncode}", flush=True)
                        continue
                    rows = accepted_trials(output, ids, seed, plan, cfg, teacher)
                except (OSError, ValueError, KeyError, AssertionError, RuntimeError, subprocess.TimeoutExpired) as error:
                    print(f"AMP_EVAL_RETRY {label} {type(error).__name__}: {error}", flush=True)
                    continue
            atomic(accepted, {"output":str(output)})
            print(f"AMP_EVAL_ACCEPTED {label} rows={len(rows)}", flush=True)
            return rows
    raise RuntimeError(f"AMP batch {label} exhausted its two attempts; see {batch}")


def evaluate(root, gpu, pilot_only=False):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root/".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, cfg, teacher, selection = setup(root)
        ids = [d["id"] for d in selection["structures"]]
        print(f"AMP_EVAL_SELECTION group={selection['group']} designs={len(ids)}", flush=True)
        pilot = trial_batch(root, "pilot", ids[:2], SEEDS[0], plan, cfg, teacher, gpu)
        if pilot_only:
            atomic(root/"pilot_verified.json", dict(rows=len(pilot), results=pilot))
            return
        rows = pilot + trial_batch(root, "seed1_remaining", ids[2:], SEEDS[0], plan, cfg, teacher, gpu)
        for seed in SEEDS[1:]:
            rows += trial_batch(root, f"seed_{seed}", ids, seed, plan, cfg, teacher, gpu)
        summarize(root, rows, selection)


def csv_write(path, records):
    with NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as stream:
        name = stream.name
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(name, path)


def summarize(root, rows, selection):
    from statistics import fmean, median, stdev
    import numpy as np

    structures = {row["id"]:row for row in selection["structures"]}
    assert len(rows) == 150 and {(r["design_id"],r["seed"]) for r in rows} == {
        (design,seed) for design in structures for seed in SEEDS}
    table = []
    detail = []
    def risk_joint(row):
        value = (row["observed"] or {}).get(COSTS[2])
        if value is None:
            return None
        return next((name for name, motor in (row["motors"] or {}).items()
                     if motor["rated_ratio"] == value and name.split("_",1)[1] in
                     {"knee_joint","ankle_pitch_joint","ankle_roll_joint"}), None)

    for design, morphology in structures.items():
        trials = sorted([row for row in rows if row["design_id"]==design], key=lambda row:row["seed"])
        successful = [row for row in trials if row["status"]=="success"]
        base = dict(morphology, trials=len(trials), successes=len(successful),
                    falls=sum(row["fell"] for row in trials),
                    route_completions=sum(bool(row["route_completed"]) for row in trials))
        for field in COSTS:
            values = [row["metrics"][field] for row in successful if row["metrics"][field] is not None]
            base.update({field+"_n":len(values), field+"_mean":fmean(values) if values else None,
                         field+"_std":stdev(values) if len(values)>1 else None})
        joint = Counter(x for row in successful if (x:=risk_joint(row)) is not None)
        base["torque_bottleneck_joint"] = joint.most_common(1)[0][0] if joint else None
        table.append(base)
        for row in trials:
            observed = row["observed"] or {}
            detail.append(dict(morphology, seed=row["seed"], status=row["status"], reason=row["reason"],
                measured_s=row["steps"]*.02, fell=row["fell"], forward_m=row["forward_m"],
                mean_forward_mps=row["mean_forward_mps"], route_completed=row["route_completed"],
                **{key:observed.get(key) for key in COSTS}, torque_bottleneck_joint=risk_joint(row),
                report=row["report"]))
    def measures(field):
        values = [r[field+"_mean"] for r in table if r[field+"_mean"] is not None]
        repeats = [r[field+"_std"] for r in table if r[field+"_std"] is not None]
        return dict(valid_structures=len(values), median=median(values) if values else None,
                    p10=float(np.quantile(values,.1)) if values else None,
                    p90=float(np.quantile(values,.9)) if values else None,
                    median_repeat_std=median(repeats) if repeats else None)

    complete = [r for r in table if r["successes"]==3 and all(r[key+"_n"]==3 for key in COSTS)]
    corr = None
    if len(complete)>1:
        from scipy.stats import rankdata
        ranks = np.column_stack([rankdata([r[key+"_mean"] for r in complete]) for key in COSTS])
        corr = [[float(np.corrcoef(ranks[:,i],ranks[:,j])[0,1])
                 if np.std(ranks[:,i]) and np.std(ranks[:,j]) else None
                 for j in range(len(COSTS))] for i in range(len(COSTS))]
    csv_write(root/"amp_structure_metrics.csv", table)
    csv_write(root/"amp_trial_metrics.csv", detail)
    atomic(root/"amp_metric_summary.json", dict(
        structures=50, trials=150, seeds=list(SEEDS), commanded_speed_mps=1.5, horizon_s=20.,
        schedule_group=selection["group"], checkpoint_sha256=EXPECTED_MODEL,
        metric_order=list(COSTS), statistics={key:measures(key) for key in COSTS},
        success_counts=dict(Counter(str(row["successes"]) for row in table)),
        successful_trials=sum(r["status"]=="success" for r in rows),
        fully_valid_structures=len(complete), spearman=corr,
        failures=dict(Counter(r["reason"] for r in rows if r["status"]!="success")),
    ))
    print("AMP_EVAL_COMPLETE 50 structures, 150 validated trials", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="task", required=True)
    controller = sub.add_parser("evaluate")
    controller.add_argument("--root", type=Path, required=True)
    controller.add_argument("--gpu", type=int, required=True)
    controller.add_argument("--pilot-only", action="store_true")
    controller.add_argument("--check-only", action="store_true")
    child = sub.add_parser("worker")
    for name in ("plan","conditions","output"):
        child.add_argument("--"+name, type=Path, required=True)
    child.add_argument("--ids", required=True)
    child.add_argument("--gpu", type=int, required=True)
    child.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.task == "worker":
        return worker(args)
    if args.check_only:
        with TemporaryDirectory(prefix="amp-reference-check-") as tmp:
            _, _, teacher, _ = setup(Path(tmp))
        print(f"AMP_REFERENCE_INPUTS_OK {len(teacher['design_ids'])} {teacher['checkpoint_sha256']}")
        return 0
    evaluate(args.root.resolve(), args.gpu, args.pilot_only)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
