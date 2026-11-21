"""Measure the finished Rough expert on the AMP reference morphologies."""
import argparse
from collections import Counter
from copy import copy, deepcopy
import csv
import json
import os
from pathlib import Path
from statistics import fmean, stdev
from tempfile import TemporaryDirectory
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
PROJECT = Path(os.environ.get("ROBO_REFERENCE_PROJECT", REPO / "robolab/logs/multitask_full2500_physical_migrated")).resolve()
EXPERT = PROJECT/"experts/rough"
ENTRY = REPO/"robolab/experiments/multitask_morphology"
ENGINE = ENTRY/"engine"
SHARED = ENTRY/"shared_policy"
SOURCE_CHECKPOINT = "358247fdf142fddc2ac1874f9aa2463acc1e66039717bc13e02b83a8180be18f"
SERVER_CHECKPOINT = "dbbe631bf0b95d3484e9014ec98b5c9766ea080eee5fa968a3848eb40ba0275b"
AMP_SELECTION = REPO / "docs/reference/rough_amp_expert_20261121/amp/selection.json"
SEEDS = (20261118, 20261119, 20261120)

sys.path[:0] = [str(SHARED), str(ENGINE), str(REPO/"robolab"), str(REPO/"rsl_rl")]
from contract import load_plan, sha, structure_key
from shared_bo_metrics import METRICS, digest, read_conditions

COSTS = METRICS["stairs"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temp.replace(path)


def freeze(path, value):
    if path.is_file():
        if read(path) != value:
            raise ValueError(f"Frozen input changed: {path}")
    else:
        save(path, value)


def select_reference_structures(bank, configuration, source):
    """Match the published AMP cohort to the Rough schedule in AMP evaluation order."""
    from hybrid_schedule import load_schedule

    assert source["group"] == 21 and len(source["structures"]) == 50
    lookup = {structure_key(row): row for row in bank["designs"]}
    keys = [structure_key(row) for row in source["structures"]]
    chosen = [lookup[key] for key in keys]
    ids = {row["id"] for row in chosen}
    assert len(set(keys)) == len(ids) == 50
    schedule, _ = load_schedule(bank, configuration)
    assert ids == set(schedule["groups"][source["group"] - 1])
    return [
        dict(amp_id=amp["id"], id=row["id"], thigh_m=row["thigh_m"], calf_m=row["calf_m"],
             knee_motor=row["knee_motor"], ankle_motor=row["ankle_motor"])
        for amp, row in zip(source["structures"], chosen, strict=True)
    ]


def preparation(root):
    bank_path = PROJECT/"banks/rough/bank.json"
    bank = read(bank_path)
    assert bank["task_key"] == "stairs"
    old_plan = read(EXPERT/"plan.json")
    old_identity = old_plan["experiment"]
    assert sha(bank_path)==old_identity["bank_sha256"]
    assert sha(ENGINE/"hybrid_runtime.py")==old_identity["hybrid_sources"]["hybrid_runtime.py"]
    assert bank["training"]["stairs_height_range_m"] == [.05,.10]
    done = read(EXPERT/"completed.json")
    checkpoint = Path(done["checkpoint"])
    assert done["completed_updates"]==12000 and checkpoint.name=="model_11999.pt"
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha not in (SOURCE_CHECKPOINT, SERVER_CHECKPOINT):
        raise ValueError(f"Unknown Rough reference checkpoint: {checkpoint_sha}")
    from hybrid_contract import validate_checkpoint
    import torch
    torch.set_num_threads(1)
    validate_checkpoint(checkpoint,old_identity,12000)
    original = root/"amp_selection.json"
    source = read(AMP_SELECTION)
    freeze(original, source)
    structures = select_reference_structures(bank, old_plan["configuration"], source)
    ids = [row["id"] for row in structures]
    selection = dict(
        amp_selection_sha256=sha(AMP_SELECTION), schedule_group=21, motor_pairs=25,
        structures=structures,
    )
    frozen = root/"selection.json"
    freeze(frozen, selection)
    schema = {key:bank[key] for key in ("knees","ankles","context_schema","context_width")}
    schema.update(tasks=["stairs"],shared_structure_keys=[list(structure_key(row)) for row in structures])
    binding = EXPERT / "cloud_import/evaluation_teacher.json"
    if checkpoint_sha == SERVER_CHECKPOINT:
        imported = read(binding)
        assert imported["checkpoint_sha256"] == checkpoint_sha
        assert imported["bank_sha256"] == sha(bank_path)
        assert len(imported["asset_cache"]) == 1
        cache = Path(imported["asset_cache"][0])
    else:
        cache_paths = sorted(EXPERT.glob("native_cache_*/completed.json"))
        assert len(cache_paths) == 1
        cache = cache_paths[0].parent
    if not (cache / "completed.json").is_file():
        raise ValueError(f"Rough native cache is incomplete: {cache}")
    teacher = dict(bank=str(bank_path), bank_sha256=sha(bank_path),
                   checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                   asset_cache=[str(cache)], code_root=str(REPO/"robolab"),runtime_dir=str(ENGINE),
                   full_body_torque=True, context_width=bank["context_width"], copies=1, design_ids=ids)
    plan = dict(kind="rough_backpack_distillation_plan_v2", structure_scope="teacher_intersection_only",
                selected_structure_keys=schema["shared_structure_keys"],
                teachers={"stairs":teacher}, task_ids={"stairs":0},
                shared_bank=str(frozen), shared_bank_sha256=sha(frozen), schema=schema)
    plan_path = root/"evaluation_plan.json"
    freeze(plan_path,plan)
    load_plan(plan_path)
    config = read(ENTRY/"configs/bo_pilot_conditions.json")
    config.update(duration_s=20.,seeds=list(SEEDS),copies=1)
    config["tasks"]["stairs"].update(speed_mps=.5,step_height_m=.075,step_depth_m=.30)
    conditions_path = root/"conditions.json"
    freeze(conditions_path,config)
    read_conditions(conditions_path)
    freeze(root/"training_stair_geometry.json",dict(
        step_height_range_m=bank["training"]["stairs_height_range_m"],
        tread_depth_options_m=[.25,.35], training_runtime_sha256=sha(ENGINE/"hybrid_runtime.py"),
        code_revision=subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
        checkpoint_updates=12000, checkpoint_sha256=checkpoint_sha))
    return plan_path, conditions_path, teacher, selection


def worker(args):
    import bo_capture
    import rollout

    teacher=load_plan(args.plan)["teachers"]["stairs"]
    bank_path=Path(teacher["bank"])
    bank=read(bank_path)
    ids=args.ids.split(",")
    assert len(set(ids))==len(ids) and set(ids)<=set(teacher["design_ids"])
    bank_designs={row["id"]:row for row in bank["designs"]}
    native_profile=rollout.configure_profile
    native_parse=bo_capture.parse_conditions
    native_capture=bo_capture.TrialCapture

    def fixed_start(cfg, task, **kwargs):
        assert task=="stairs" and kwargs["profile"]=="nominal" and kwargs["strength"]==0.
        assert kwargs["reset_yaw_range"]==0.
        updated,audit=native_profile(cfg,task,**kwargs)
        base=updated.events.reset_base.params
        for key in ("pose_range","velocity_range"):
            base[key]={axis:(0.,0.) for axis in base[key]}
        updated.events.reset_robot_joints.params.update(position_range=(0.,0.),velocity_range=(0.,0.))
        reset_events={key for key,val in vars(updated.events).items() if getattr(val,"mode",None)=="reset"}
        assert reset_events=={"reset_base","reset_robot_joints"}, reset_events
        audit["reset"]=dict(base=deepcopy(base),joint_position=(0.,0.),joint_velocity=(0.,0.))
        audit["deterministic_evaluation"]=True
        return updated,audit

    def parse(cfg, selected):
        assert selected==ids and cfg.student is None
        conditions=native_parse(cfg,selected[:1])
        assert conditions["copies"]==1
        return conditions

    class CaptureBatch(native_capture):
        def __init__(self, raw, task, conditions, seed, motors, properties=None):
            assert task=="stairs" and raw.num_envs==len(ids) and properties is None
            assert list(raw.cfg.spatial_design_ids)==ids
            super().__init__(raw,task,dict(conditions,copies=len(ids)),seed,motors)
            self.conditions=conditions
            self.design_motors=[bo_capture.motor_records(raw,bank,bank_designs[i]) for i in ids]
            assert self.design_motors[0]==motors

        def publish(self, output, identity):
            for index, design in enumerate(ids):
                one=copy(self)
                one.count=1
                one.motors=self.design_motors[index]
                selected=slice(index,index+1)
                one.samples=[tuple(array[selected] for array in step) for step in self.samples]
                for name in ("initial","origins","mass","active","fell","timeout"):
                    setattr(one,name,getattr(self,name)[selected])
                child=output/design
                child.mkdir()
                native_capture.publish(one,child,dict(identity,design_id=design))

    def exact_usd(cfg, roots, output, task, designs):
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        from hybrid_native_cache import cache_directories
        assert task=="stairs" and roots==teacher["asset_cache"]
        assert [d["id"] for d in designs]==ids
        cache=Path(roots[0])
        known={p.name:p for p in cache_directories(cache,bank_path,ids,require_complete=True,training_fast=True)}
        assert set(known)==set(ids)
        evidence=[]
        for asset,design in zip(cfg.scene.robot.spawn.assets_cfg,ids,strict=True):
            directory=known[design]
            exact=AssetConverterBase._config_to_hash(asset)
            if (directory/".asset_hash").read_text().strip()!=exact:
                raise ValueError(f"Unexpected native USD converter identity for {design}")
            asset.usd_dir=str(directory)
            asset.force_usd_conversion=False
            evidence.append(dict(design=design,cache=str(directory),exact_hash=exact,reused=True))
        return evidence

    rollout.configure_profile=fixed_start
    rollout.reuse_assets=exact_usd
    bo_capture.parse_conditions=parse
    bo_capture.TrialCapture=CaptureBatch
    sys.argv=[
        str(SHARED/"rollout.py"), "--task","stairs","--plan",str(args.plan),
        "--design-ids",args.ids,"--output",str(args.output),
        "--bo-conditions",str(args.conditions),"--mode","evaluate","--profile","nominal",
        "--randomization-strength","0","--reset-yaw-range","0","--seed",str(args.seed),"--steps","1000",
        "--headless","--device","cuda:0",f"--kit_args=--vulkan --/renderer/activeGpu={args.renderer_gpu} --/physics/cudaDevice=0",
    ]
    return rollout.main()


def accepted_rows(output, ids, seed, plan, conditions, teacher):
    report=read(output/"summary.json")
    assert report["result"]=="REAL_ROLLOUT_COMPLETE" and report["task"]=="stairs"
    assert report["design_ids"]==ids and report["seed"]==seed and report["teacher"]==teacher["checkpoint_sha256"]
    assert report["steps"]==1000 and report["student"] is None and report["teacher_updates"]==0
    assert not (output/"failure.json").is_file()
    audit=read(output/"randomization_audit.json")
    assert audit["deterministic_evaluation"] and audit["native_teacher_noise"] is False
    assert all(event["enabled"] is False for event in audit["events"].values())
    assert all(bounds==[0.,0.] for key in ("pose_range","velocity_range")
               for bounds in audit["reset"]["base"][key].values())
    assert audit["reset"]["joint_position"]==[0.,0.] and audit["reset"]["joint_velocity"]==[0.,0.]
    for axis,bounds in (("lin_vel_x",[.5,.5]),("lin_vel_y",[0.,0.]),("ang_vel_z",[0.,0.])):
        assert audit["commands"]["ranges"][axis]==bounds
    fixed=read(conditions)["tasks"]["stairs"]
    assert all(audit["stairs"][key]==fixed[key] for key in fixed)
    assert all(asset["reused"] for asset in read(output/"asset_cache_audit.json"))
    found=[]
    for design in ids:
        path=output/design/"bo_trials.json"
        source=read(path)
        assert source["task"]=="stairs" and source["seed"]==seed and source["design_id"]==design
        assert source["actor_sha256"]==teacher["checkpoint_sha256"] and source["bank_sha256"]==teacher["bank_sha256"]
        assert source["conditions_sha256"]==digest(read(conditions)) and source["plan_sha256"]==sha(plan)
        assert sha(source["trace"])==source["trace_sha256"]
        assert len(source["trials"])==1
        case=source["trials"][0]
        if case["status"] not in ("success","failure"):
            raise ValueError(f"Invalid simulation evidence for {design}: {case['reason']}")
        assert case["command_mps"]==.5 and case["dt_s"]==.02
        if case["status"]=="success":
            assert case["steps"]==1000 and set(case["metrics"])==set(COSTS)
        found.append(dict(design_id=design,seed=seed,report=str(path),**case))
    return found


def batch(root, label, ids, seed, plan, conditions, teacher, gpu):
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    section=root/"batches"/label
    section.mkdir(parents=True,exist_ok=True)
    accepted=section/"accepted.json"
    if accepted.is_file():
        return accepted_rows(Path(read(accepted)["output"]),ids,seed,plan,conditions,teacher)
    for _ in range(2):
        numbers=[int(p.name.split("_")[-1]) for p in section.glob("attempt_*")]
        attempt=section/f"attempt_{max(numbers,default=0)+1:02d}"
        attempt.mkdir()
        output=attempt/"native"
        devices=candidates(query_gpus(),os.environ,min_free_mib=12.5*1024,max_utilization=15,requested=f"cuda:{gpu}")
        with gpu_lease(devices,REPO/"robolab/logs/rpo_rough_multimorph/gpu_locks") as device:
            command=[sys.executable,"-u",__file__,"worker","--plan",str(plan),"--conditions",str(conditions),
                     "--ids",",".join(ids),"--seed",str(seed),"--output",str(output),
                     "--renderer-gpu",str(device["index"])]
            save(attempt/"command.json",command)
            env=child_environment(device,os.environ)
            print(f"ROUGH_EVAL_START {label} seed={seed} designs={len(ids)}",flush=True)
            try:
                with (attempt/"stdout.log").open("x") as out,(attempt/"stderr.log").open("x") as err:
                    process=subprocess.run(command,cwd=REPO,env=env,stdout=out,stderr=err,timeout=2400)
                if process.returncode:
                    raise RuntimeError(f"native rollout returned {process.returncode}")
                rows=accepted_rows(output,ids,seed,plan,conditions,teacher)
            except (AssertionError,OSError,ValueError,KeyError,RuntimeError,subprocess.TimeoutExpired) as error:
                print(f"ROUGH_EVAL_RETRY {label} {type(error).__name__}: {error}",flush=True)
                continue
        save(accepted,dict(output=str(output)))
        distribution=Counter((p["status"],p["reached_top"]) for p in rows)
        print("ROUGH_EVAL_ACCEPTED",label,len(rows),dict(distribution),flush=True)
        return rows
    raise RuntimeError(f"Both attempts failed: {section}")


def csv_write(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, rows, selection, teacher):
    import numpy as np
    from math import isfinite

    chosen={row["id"]:row for row in selection["structures"]}
    assert len(rows)==150 and {(r["design_id"],r["seed"]) for r in rows}=={
        (design,seed) for design in chosen for seed in SEEDS
    }
    table, detail = [], []
    for design,info in chosen.items():
        trials=sorted((row for row in rows if row["design_id"]==design),key=lambda row:row["seed"])
        valid=[r for r in trials if r["status"]=="success"]
        full_climb=[r for r in valid if r["reached_top"]]
        base=dict(info,trials=3,successes=len(valid),falls=sum(r["fell"] for r in trials),
                  reached_top=sum(bool(r["reached_top"]) for r in trials),
                  successful_climbs=len(full_climb))
        for key in COSTS:
            use=[r["metrics"][key] for r in valid if r["metrics"][key] is not None]
            climbs=[r["metrics"][key] for r in full_climb if r["metrics"][key] is not None]
            base[key+"_mean"]=fmean(use) if use else None
            base[key+"_std"]=stdev(use) if len(use)>1 else None
            base[key+"_climb_mean"]=fmean(climbs) if len(climbs)==3 else None
        table.append(base)
        for row in trials:
            observed=row["observed"] or {}
            detail.append(dict(info,seed=row["seed"],status=row["status"],reason=row["reason"],
                measured_s=row["steps"]*.02,fell=row["fell"],reached_top=row["reached_top"],
                forward_m=row["forward_m"],mean_forward_mps=row["mean_forward_mps"],
                **{key:observed.get(key) for key in COSTS},report=row["report"]))

    scales={}
    for key in COSTS:
        values=[row[key+"_climb_mean"] for row in table if row[key+"_climb_mean"] is not None]
        assert all(isfinite(x) for x in values)
        if values:
            q10,mid,q90=(float(x) for x in np.quantile(values,[.10,.50,.90]))
            width=q90-q10
            scales[key]=dict(p10=q10,median=mid,p90=q90,scale=width,scale_multiplier=1/width if width>0 else None,
                             structures=len(values))
        else:
            scales[key]=dict(p10=None,median=None,p90=None,scale=None,scale_multiplier=None,structures=0)
        for row in table:
            value=row[key+"_climb_mean"]
            multiplier=scales[key]["scale_multiplier"]
            row[key+"_fixed_scale"]=value*multiplier if value is not None and multiplier is not None else None

    successful=sum(row["status"]=="success" for row in rows)
    atop=sum(row["status"]=="success" and bool(row["reached_top"]) for row in rows)
    save(root/"rough_normalization_p10_p90.json",dict(
        method="fixed_p90_minus_p10_multiplier",formula="raw * (1 / (P90-P10))",
        calibration_subset="three successful and top-reaching trials",
        baseline_design_count=50,checkpoint_sha256=teacher["checkpoint_sha256"],
        horizon_s=20.,speed_mps=.5,step_height_m=.075,step_depth_m=.30,seeds=list(SEEDS),metrics=scales))
    csv_write(root/"rough_structure_metrics.csv",table)
    csv_write(root/"rough_trial_metrics.csv",detail)
    save(root/"rough_metric_summary.json",dict(
        structures=50,trials=150,successful_trials=successful,successful_climbs=atop,
        any_top=sum(bool(row["reached_top"]) for row in rows),falls=sum(row["fell"] for row in rows),
        fully_successful_structures=sum(row["successes"]==3 for row in table),
        fully_climbing_structures=sum(row["successful_climbs"]==3 for row in table),
        failure_reasons=dict(Counter(r["reason"] for r in rows if r["status"]!="success")),
        calibration=scales,
    ))
    print(f"ROUGH_EVAL_COMPLETE structures=50 trials=150 successful_climbs={atop}",flush=True)


def evaluate(root, gpu):
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/".controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan,conditions,teacher,selection=preparation(root)
        ids=teacher["design_ids"]
        print("ROUGH_EVAL_INPUTS_OK same AMP set, 50 structures",flush=True)
        batch(root,"pilot",ids[:2],SEEDS[0],plan,conditions,teacher,gpu)
        rows=[]
        for seed in SEEDS:
            rows+=batch(root,f"seed_{seed}",ids,seed,plan,conditions,teacher,gpu)
        summarize(root,rows,selection,teacher)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="task",required=True)
    parent=sub.add_parser("evaluate")
    parent.add_argument("--root",type=Path,required=True)
    parent.add_argument("--gpu",type=int,required=True)
    parent.add_argument("--check-only",action="store_true")
    w=sub.add_parser("worker")
    for key in ("plan","conditions","output"):
        w.add_argument("--"+key,type=Path,required=True)
    w.add_argument("--ids",required=True)
    w.add_argument("--seed",required=True,type=int)
    w.add_argument("--renderer-gpu",required=True,type=int)
    args=parser.parse_args()
    if args.task=="worker":
        return worker(args)
    if args.check_only:
        with TemporaryDirectory(prefix="rough-reference-check-") as tmp:
            _,_,teacher,_=preparation(Path(tmp))
        print(f"ROUGH_REFERENCE_INPUTS_OK {len(teacher['design_ids'])} {teacher['checkpoint_sha256']}")
        return 0
    evaluate(args.root.resolve(),args.gpu)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
