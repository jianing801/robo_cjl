"""Offline-only plan/generate/assemble commands. Never starts training.

plan: CPU metadata only. generate: explicit HHTools CPU IK candidates.
assemble: requires reviewed, complete eight-clip datasets and identity controls.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def generate(plan, out, args):
    started = time.monotonic()
    import numpy as np
    from robolab.amp_reference_retarget import (sha, write_new, load_payload, validate_payload,
        encode_payload, geometry_metrics, quality_issues, blocking_issues)
    # Import only in the isolated HHTools environment, never in training Python.
    sys.path.insert(0, plan["hhtools_root"])
    from hhtools_amp_backend import Backend
    backend = Backend(plan, out)
    names = plan["joint_names"]
    clips = args.clips.split(",") if args.clips else list(plan["clips"])
    designs = args.design_ids.split(",") if args.design_ids else [d["id"] for d in plan["designs"]]
    if set(clips)-plan["clips"].keys() or set(designs)-{d["id"] for d in plan["designs"]}:
        raise ValueError("unknown clip or design ID")
    if args.max_frames is not None and args.max_frames < 3:
        raise ValueError("diagnostic requires at least three frames")
    target_models = {"identity": backend.model("identity", plan["source_urdf"])}
    signatures = {"identity": "identity"}
    geometric_models = {}
    for d in plan["designs"]:
        if d["id"] in designs:
            key = d["reference_kinematic_signature"]
            if key not in geometric_models:
                geometric_models[key] = backend.model(d["id"], d["urdf"])
            target_models[d["id"]] = geometric_models[key]
            signatures[d["id"]] = key
    for clip in clips:
        entry = plan["clips"][clip]
        source = load_payload(entry["path"], names, entry["sha256"])
        if args.max_frames:
            for key in ("root_pos", "root_rot", "dof_pos", "key_body_pos"):
                source[key] = source[key][:args.max_frames]
        q0 = validate_payload(source, names)
        source_fk, source_z = backend.fk_and_soles(backend.source, q0)
        stored_names = entry.get("stored_keypoint_body_names", plan["key_bodies"])
        stored_fk = source_fk if stored_names == plan["key_bodies"] else backend.fk_and_soles(
            backend.source, q0, key_bodies=stored_names)[0]
        source_error = float(np.linalg.norm(stored_fk-source["key_body_pos"], axis=-1).max())
        if source_error > .005:
            raise ValueError(f"source FK mismatch {source_error}m; do not retarget")
        fps = float(source["fps"])
        source_metrics = geometry_metrics(q0, source_z, fps, *backend.limits(backend.source))
        solved = {}
        for design_id, target in target_models.items():
            job = out / design_id / clip
            job.mkdir(parents=True, exist_ok=False)
            key = signatures[design_id]
            if key in solved:
                q1, target_fk, target_z, metrics, metadata, solved_design = solved[key]
                solve_seconds = 0.
            else:
                solve_started = time.monotonic()
                q1, metadata = backend.solve(target, q0, fps, clip)
                solve_seconds = time.monotonic()-solve_started
                target_fk, target_z = backend.fk_and_soles(target, q1)
                metrics = geometry_metrics(q1, target_z, fps, *backend.limits(target))
                solved_design = design_id
                solved[key] = (q1, target_fk, target_z, metrics, metadata, solved_design)
            if q1.shape != q0.shape:
                raise ValueError("IK changed frame count; comparison would be confounded")
            payload = encode_payload(q1, target_fk, source, names)
            delta = np.linalg.norm((target_fk-q1[:, None, :3])-(source_fk-q0[:, None, :3]), axis=-1)
            relative = {"mean_m": float(delta.mean()), "max_m": float(delta.max())}
            issues = quality_issues(source_metrics, metrics, identity=design_id == "identity", relative_fk_error=relative)
            file = job / (clip+".pkl")
            with file.open("xb") as stream:
                pickle.dump(payload, stream, protocol=4)
            validate_payload(load_payload(file, names, sha(file)), names)
            np.savez(job / "geometry.npz", source_q_xyzw=q0, target_q_xyzw=q1,
                     source_sole_z=source_z, target_sole_z=target_z,
                     source_keypoints_world=source_fk, target_keypoints_world=target_fk,
                     joint_names=np.asarray(names), key_body_names=np.asarray(plan["key_bodies"]), fps=np.asarray(fps))
            write_new(job / "result.json", {"kind": "HHTOOLS_KINEMATIC_CANDIDATE_NOT_TRAINING_APPROVED",
                "plan_sha256": sha(args.plan), "design_id": design_id,
                "urdf_sha256": plan["source_urdf_sha256"] if design_id == "identity" else
                    next(d["urdf_sha256"] for d in plan["designs"] if d["id"] == design_id),
                "clip": clip, "file": str(file), "sha256": sha(file), "weight": entry["weight"],
                "full_clip": len(q1) == entry["frames"] and args.max_frames is None,
                "frames": len(q1), "fps": fps, "source_sha256": entry["sha256"],
                "source_fk_max_error_m": source_error, "relative_keypoint_difference": relative,
                "source_stored_keypoint_body_names": stored_names,
                "target_keypoint_body_names": plan["key_bodies"],
                "source_stored_vs_canonical_keypoint_max_error_m": float(np.linalg.norm(source_fk-source["key_body_pos"],axis=-1).max()),
                "joint_angle_rms_difference_rad": float(np.sqrt(np.mean((q1[:, 7:]-q0[:, 7:])**2))),
                "source_geometry": source_metrics, "target_geometry": metrics, "quality_issues": issues,
                "blocking_issues": blocking_issues(issues, plan.get("acceptance_policy")),
                "acceptance_policy": plan.get("acceptance_policy"),
                "kinematic_cache_solved_design": solved_design, "kinematic_cache_key": key,
                "solver": metadata, "IK_wall_seconds": solve_seconds, "dynamic_feasibility_validated": False})
            print(design_id, clip, "offline alarms", issues, flush=True)
    write_new(out / "complete.json", {"status": "OFFLINE_CANDIDATES_ONLY_NO_TRAINING",
        "plan_sha256": sha(args.plan), "design_ids": designs, "clips": clips,
        "max_frames": args.max_frames, "all_require_explicit_review": True,
        "total_offline_wall_seconds_including_imports_and_audits": time.monotonic()-started})


def assemble(plan, out, args):
    from robolab.amp_reference_retarget import (sha, write_new, KIND, approved_dataset, load_payload,
        DIRECT_POLICY, DIRECT_INSTRUCTION, blocking_issues)
    from robolab.native_expert_contract import source_identity
    weights = {n:c["weight"] for n, c in plan["clips"].items()}
    candidates = Path(args.candidates).resolve()
    direct = plan.get("acceptance_policy") == DIRECT_POLICY and plan.get("user_instruction") == DIRECT_INSTRUCTION
    reviews = out / "acceptance_records" if direct else Path(args.reviews).resolve() if args.reviews else None
    if reviews is None:
        raise ValueError("strict plans require actual visual review records")
    if direct:
        reviews.mkdir()
    if not (candidates / "complete.json").is_file():
        raise ValueError("incomplete candidate generation")
    library = {"kind": KIND, "status": "USER_AUTHORIZED_HHTOOLS_REFERENCE_LIBRARY" if direct else "REVIEWED_KINEMATIC_REFERENCE_LIBRARY",
               "plan_sha256": sha(args.plan), "designs": {}, "dynamic_feasibility_validated": False}
    def result(design_id, name):
        value = json.loads((candidates / design_id / name / "result.json").read_text(encoding="utf8"))
        expected_urdf = plan["source_urdf_sha256"] if design_id == "identity" else next(
            d["urdf_sha256"] for d in plan["designs"] if d["id"] == design_id)
        blockers = blocking_issues(value["quality_issues"], plan.get("acceptance_policy"))
        if (value["plan_sha256"] != sha(args.plan) or not value["full_clip"] or blockers
                or value["design_id"] != design_id or value["urdf_sha256"] != expected_urdf
                or value["source_sha256"] != plan["clips"][name]["sha256"]
                or value["frames"] != plan["clips"][name]["frames"] or value["fps"] != plan["clips"][name]["fps"]):
            raise ValueError(f"incomplete/unsafe/wrong candidate: {design_id}/{name}")
        load_payload(value["file"], plan["joint_names"], value["sha256"])
        return value
    # Identity controls are required for ALL eight real training clips, not C3.
    for name in weights:
        result("identity", name)
    for design in plan["designs"]:
        design_id = design["id"]
        directory = out / "references" / design_id
        directory.mkdir(parents=True)
        records = {n: result(design_id, n) for n in weights}
        review = reviews / (design_id+".json")
        if direct:
            write_new(review, {"acceptance_basis": DIRECT_POLICY, "user_instruction": DIRECT_INSTRUCTION,
                "plan_sha256": sha(args.plan), "design_id": design_id, "urdf_sha256": design["urdf_sha256"],
                "clips": {n:r["sha256"] for n,r in records.items()}, "blocking_issues": [],
                "visual_review_performed": False, "dynamic_feasibility_validated": False,
                "observations": "Machine data-integrity audit only; user accepts HHTools references without contact/identity-fit gates. Not a visual or physical gait approval.",
                "diagnostics": {n:r["quality_issues"] for n,r in records.items()},
                "evidence": [{"path": str(candidates / design_id / n / "result.json"),
                              "sha256": sha(candidates / design_id / n / "result.json")} for n in weights]})
        # All files are hash bound below. No --force or approval bypass.
        review_data = json.loads(review.read_text(encoding="utf8"))
        if review_data.get("plan_sha256") != sha(args.plan):
            raise ValueError("review is for another plan")
        for name, value in records.items():
            with (directory / (name+".pkl")).open("xb") as stream:
                stream.write(Path(value["file"]).read_bytes())
        library["designs"][design_id] = {"urdf_sha256": design["urdf_sha256"],
            "directory": str(directory), "weights": weights,
            "clips": {n:{"sha256": r["sha256"]} for n, r in records.items()},
            "review": str(review), "review_sha256": sha(review)}
    library_path = out / "reference_library.json"
    write_new(library_path, library)
    bank = deepcopy(json.loads(Path(plan["baseline_bank"]).read_text(encoding="utf8")))
    bank["training"].update(source_hashes=source_identity(), initialization="scratch; HHTools retarget control",
        reference_retargeting=True, baseline_bank_sha256=plan["baseline_bank_sha256"],
        comparison="same curriculum/physics/AMP parameters; only per-design reference data changed",
        retargeted_references={"plan": str(Path(args.plan).resolve()), "plan_sha256": sha(args.plan),
                               "library": str(library_path), "library_sha256": sha(library_path)})
    for design in bank["designs"]:
        approved_dataset(bank, design, weights)
    write_new(out / "bank.json", bank)
    print("NEW_BANK_PREPARED_NOT_STARTED", out / "bank.json", flush=True)


def main():
    from robolab.amp_reference_retarget import make_plan, verify_plan, write_new
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--baseline-bank", required=True)
    p.add_argument("--hhtools-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--accept-direct-hhtools", action="store_true", help="Explicit user-authorized raw R2R control, no contact/identity-fit quality gate")
    p = commands.add_parser("generate")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--execute-offline", action="store_true", required=True)
    p.add_argument("--design-ids")
    p.add_argument("--clips")
    p.add_argument("--max-frames", type=int, help="diagnostic only, NEVER training eligible")
    p = commands.add_parser("assemble")
    p.add_argument("--plan", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--reviews", help="Required for strict plans, not for user-authorized direct-HHTools plans")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = make_plan(args.baseline_bank, args.hhtools_root, direct_hhtools=args.accept_direct_hhtools)
    else:
        plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
        verify_plan(plan)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.command == "plan":
        write_new(output / "plan.json", plan)
        print("PLAN_ONLY", output / "plan.json", flush=True)
    elif args.command == "generate":
        generate(plan, output, args)
    else:
        assemble(plan, output, args)


if __name__ == "__main__":
    main()
