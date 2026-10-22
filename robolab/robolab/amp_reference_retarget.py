"""Offline reference-library contracts. No Isaac, HHTools or training on import.

Motion candidates are deliberately unusable by the training loader until a
separate review binds their exact hashes. Never edit the source motion bank.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

KIND = "rpo_hhtools_per_design_references_v1"
DIRECT_POLICY = "user_authorized_hhtools_contact_and_identity_diagnostics_only_v1"
DIRECT_INSTRUCTION = "那你直接用hhtools重定向的作为参考动作，先不管离地了"
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link",
              "right_knee_link", "left_elbow_yaw_link", "right_elbow_yaw_link")
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def native_weights():
    """Read the actual source task, not the earlier C3 diagnostic clip."""
    tree = ast.parse((ROOT / "robolab/tasks/manager_based/amp/rpo_amp_env_cfg.py").read_text(encoding="utf8"))
    matches = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "motion_data_weights" for t in n.targets)]
    if len(matches) != 1:
        raise ValueError("ambiguous source AMP motion mixture")
    weights = ast.literal_eval(matches[0])
    if len(weights) != 8 or any(not isinstance(k, str) or not np.isfinite(v) or v <= 0
                                for k, v in weights.items()):
        raise ValueError("source AMP eight-clip contract changed; inspect explicitly")
    return weights


def validate_payload(data, joint_names):
    """Return canonical xyz/xyzw/joints without silently repairing input."""
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid reference frame rate")
    n = len(data["root_pos"])
    shapes = {"root_pos": (n, 3), "root_rot": (n, 4),
              "dof_pos": (n, len(joint_names)), "key_body_pos": (n, 6, 3)}
    if n < 3 or len(set(joint_names)) != 23:
        raise ValueError("need complete 23-joint motion with at least three frames")
    for name, shape in shapes.items():
        a = np.asarray(data[name])
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"invalid {name} shape/values")
    if not np.allclose(np.linalg.norm(data["root_rot"], axis=1), 1, atol=2e-4):
        raise ValueError("root quaternion is not unit length")
    return np.concatenate((data["root_pos"], np.asarray(data["root_rot"])[:, [1, 2, 3, 0]],
                           data["dof_pos"]), axis=1).astype(np.float64)


def load_payload(path, joint_names, expected_sha):
    # Only caller-identified, hash-pinned local RoboLab assets may be unpickled.
    if sha(path) != expected_sha:
        raise ValueError("reference file identity changed")
    try:
        with Path(path).open("rb") as stream:
            value = pickle.load(stream)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError):
        # Same supported serialization fallback as source MotionDataManager.
        import joblib
        value = joblib.load(path)
    validate_payload(value, joint_names)
    return value


def encode_payload(q_xyzw, keypoints_world, source, joint_names):
    q = np.asarray(q_xyzw)
    # Preserve loop semantics and any source metadata. No temporal resampling,
    # smoothing, truncation or invented velocity arrays in this comparison.
    allowed = {"fps", "root_pos", "root_rot", "dof_pos", "key_body_pos", "loop_mode"}
    derived = {"root_vel", "root_ang_vel", "dof_vel"}
    if set(source) - allowed - derived:
        raise ValueError(f"unknown source fields need explicit conversion: {set(source)-allowed-derived}")
    # Source loader ignores these stored derivatives and recomputes them. Do
    # not copy stale velocities into a new reference or change that loader.
    result = {k:deepcopy(v) for k,v in source.items() if k in allowed}
    result.update(root_pos=q[:, :3].astype(np.float32),
                  root_rot=q[:, [6, 3, 4, 5]].astype(np.float32),
                  dof_pos=q[:, 7:].astype(np.float32),
                  key_body_pos=np.asarray(keypoints_world, dtype=np.float32))
    validate_payload(result, joint_names)
    return result


def reorder_solver(q, solver_names, canonical_names):
    if len(set(solver_names)) != 23 or set(solver_names) != set(canonical_names):
        raise ValueError("Newton solver and Isaac joint names do not match")
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != 30:
        raise ValueError("invalid Newton coordinate layout")
    return np.concatenate((q[:, :7], q[:, 7:][:, [solver_names.index(n) for n in canonical_names]]), axis=1)


def longest_run(mask, fps):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    begin, end = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return float(max(end - begin, default=0) / fps)


def geometry_metrics(q, sole_z, fps, lower, upper):
    z = np.asarray(sole_z)
    return {
        "joint_limit_violation_rad": float(max(0, np.max(np.maximum(lower-q[:, 7:], q[:, 7:]-upper)))),
        "joint_speed_peak_rad_s": float(np.max(np.abs(np.diff(q[:, 7:], axis=0))*fps)),
        "lowest_sole_median_m": float(np.median(z.min(axis=1))),
        "sole_min_m": float(z.min()),
        "both_soles_above_2cm_fraction": float((z.min(axis=1) > .02).mean()),
        "longest_both_soles_above_2cm_s": longest_run(z.min(axis=1) > .02, fps),
        "mean_root_forward_speed_m_s_report_only": float((q[-1, 0]-q[0, 0])*fps/(len(q)-1)),
        "contact_measurement": "collision geometry proximity, NOT measured contact or dynamics",
    }


def quality_issues(source_metrics, target_metrics, *, identity=False, relative_fk_error=None):
    """Conservative offline regression alarms, NOT gait/speed pass thresholds."""
    issues = []
    if target_metrics["joint_limit_violation_rad"] > 1e-4:
        issues.append("joint_limit_violation")
    if target_metrics["sole_min_m"] < source_metrics["sole_min_m"] - .01:
        issues.append("extra_ground_penetration_over_1cm")
    if target_metrics["lowest_sole_median_m"] > source_metrics["lowest_sole_median_m"] + .02:
        issues.append("extra_median_sole_height_over_2cm")
    if target_metrics["longest_both_soles_above_2cm_s"] > source_metrics["longest_both_soles_above_2cm_s"] + .10:
        issues.append("extra_continuous_airtime_over_0p1s")
    if target_metrics["joint_speed_peak_rad_s"] > max(1, source_metrics["joint_speed_peak_rad_s"]*1.5):
        issues.append("new_joint_speed_spike_over_1p5x_source")
    if identity and (relative_fk_error is None or relative_fk_error["mean_m"] > .003
                     or relative_fk_error["max_m"] > .02):
        issues.append("identity_retarget_does_not_preserve_source_geometry")
    return issues


def blocking_issues(issues, policy=None):
    """Explicit raw-R2R experiment, not a declaration of gait quality.

    Known contact regressions and identity fitting residuals remain visible;
    the user's direct-HHTools control does not require optimizing them away.
    Numerical/URDF-limit and new speed spikes still block. Unknown alarms fail
    closed. Legacy plans retain their original strict behavior.
    """
    if policy != DIRECT_POLICY:
        return list(issues)
    warnings = {"extra_ground_penetration_over_1cm", "extra_median_sole_height_over_2cm",
                "extra_continuous_airtime_over_0p1s", "identity_retarget_does_not_preserve_source_geometry"}
    return [issue for issue in issues if issue not in warnings]


def kinematic_signature(path):
    """Exact geometric cache key; excludes masses and motor effort/speed only.

    Same lengths alone are insufficient: joint frames, axes, position limits,
    all visual/collision geometry and mesh content must also match.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    records = []
    def node(element):
        attrs = dict(element.attrib)
        if element.tag == "mesh":
            mesh = Path(attrs.pop("filename").removeprefix("file://"))
            if not mesh.is_absolute():
                mesh = (Path(path).parent / mesh).resolve()
            attrs["mesh_sha256"] = sha(mesh)
        if element.tag == "limit":
            attrs = {k:v for k,v in attrs.items() if k in ("lower", "upper")}
        children = [node(c) for c in element if c.tag not in ("inertial", "dynamics")]
        return [element.tag, attrs, children]
    for element in root:
        if element.tag in ("joint", "link"):
            records.append(node(element))
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def make_plan(bank_path, hhtools_root, *, direct_hhtools=False):
    from robolab.morph_experiments import load_bank
    from robolab.amp_reference_source_audit import stored_source_keypoints
    bank_path, hhtools_root = Path(bank_path).resolve(), Path(hhtools_root).resolve()
    bank = load_bank(bank_path)
    files = {n: {"path": str(ROOT / "data/motions/rpo_lab" / (n+".pkl")), "weight": w}
             for n, w in native_weights().items()}
    for name, record in files.items():
        record["sha256"] = sha(record["path"])
        payload = load_payload(record["path"], bank["canonical_semantic_contract"]["joint_names"], record["sha256"])
        record.update(frames=len(payload["root_pos"]), fps=float(payload["fps"]))
        record["stored_keypoint_body_names"] = list(stored_source_keypoints(name))
    sources = sorted((hhtools_root / "hhtools").rglob("*.py"))
    sources.append(hhtools_root / "configs/robots/_template/robot.yaml")
    if len(sources) < 10:
        raise ValueError("missing pinned local HHTools checkout")
    designs = deepcopy(bank["designs"])
    for design in designs:
        design["reference_kinematic_signature"] = kinematic_signature(design["urdf"])
    return {"kind": KIND, "status": "PLAN_ONLY_NO_TRAINING", "baseline_bank": str(bank_path),
            "baseline_bank_sha256": sha(bank_path), "source_urdf": bank["original_template"],
            "source_urdf_sha256": bank["original_template_sha256"],
            "joint_names": bank["canonical_semantic_contract"]["joint_names"], "key_bodies": list(KEY_BODIES),
            "clips": files, "designs": designs,
            "acceptance_policy": DIRECT_POLICY if direct_hhtools else "strict_visual_review_v1",
            "user_instruction": DIRECT_INSTRUCTION if direct_hhtools else None,
            "hhtools_root": str(hhtools_root),
            "hhtools_sources": {str(p.relative_to(hhtools_root)): sha(p) for p in sources},
            "adapter_sources": {str(p.relative_to(ROOT)): sha(p) for p in
                (Path(__file__), ROOT / "scripts/tools/hhtools_amp_backend.py", ROOT / "scripts/tools/prepare_amp_retarget_control.py",
                 ROOT / "robolab/amp_reference_source_audit.py")},
            "method": {"backend": "HHTools Newton R2R IK", "device": "cpu", "ik_iterations": 24,
                       "source_floor_z_m": 0., "no_global_minimum_floor_redefinition": True,
                       "no_synthetic_endpoint_objectives": True,
                       "human_ankle_root_offset_disabled": True,
                       "root_height_offset": "target minus source rest-pose collision sole depth; not clip minimum",
                       "generic_8rad_s_limiter_disabled_in_offline_IK_only": True,
                       "joint_midpoint_regularizer_disabled_in_offline_IK_only": True,
                       "no_change_to_physical_motor_limits": True,
                       "preserve_source_fps_frames_weights_loop": True,
                       "not_full_dynamic_retargeting": True},
            "comparison": {"num_envs": 2048, "updates": 5000, "initialization": "scratch",
                           "schedule": "identical design order and update allocation to baseline",
                           "policy_critic_discriminator_optimizers_persist_across_designs": True,
                           "discriminator_architecture_unchanged": True,
                           "only_intended_training_difference": "per-design AMP reference dataset",
                           "holdouts_never_used_for_training": True}}


def verify_plan(plan):
    if plan["kind"] != KIND or sha(plan["baseline_bank"]) != plan["baseline_bank_sha256"]:
        raise ValueError("retarget plan / baseline identity mismatch")
    checks = [(plan["source_urdf"], plan["source_urdf_sha256"])]
    checks += [(c["path"], c["sha256"]) for c in plan["clips"].values()]
    checks += [(Path(plan["hhtools_root"])/p, h) for p, h in plan["hhtools_sources"].items()]
    checks += [(ROOT/p, h) for p, h in plan["adapter_sources"].items()]
    checks += [(d["urdf"], d["urdf_sha256"]) for d in plan["designs"]]
    if any(sha(p) != h for p, h in checks):
        raise ValueError("frozen reference/URDF/HHTools/adapter changed")


def approved_dataset(bank, design, source_weights):
    """Fail-closed binding used before the AMP environment is constructed."""
    binding = bank.get("training", {}).get("retargeted_references")
    if not binding:
        return None
    if sha(binding["plan"]) != binding["plan_sha256"]:
        raise ValueError("retarget plan changed")
    plan = json.loads(Path(binding["plan"]).read_text(encoding="utf8"))
    verify_plan(plan)
    if sha(binding["library"]) != binding["library_sha256"]:
        raise ValueError("retarget library changed")
    library = json.loads(Path(binding["library"]).read_text(encoding="utf8"))
    direct = plan.get("acceptance_policy") == DIRECT_POLICY and plan.get("user_instruction") == DIRECT_INSTRUCTION
    expected_status = "USER_AUTHORIZED_HHTOOLS_REFERENCE_LIBRARY" if direct else "REVIEWED_KINEMATIC_REFERENCE_LIBRARY"
    if (library["kind"] != KIND or library["status"] != expected_status
            or library["plan_sha256"] != binding["plan_sha256"]):
        raise ValueError("unreviewed or mismatched reference library")
    record = library["designs"][design["id"]]
    if record["urdf_sha256"] != design["urdf_sha256"] or record["weights"] != source_weights:
        raise ValueError("reference design/weight mismatch")
    if set(record["clips"]) != set(source_weights):
        raise ValueError("incomplete eight-clip dataset")
    for name, clip in record["clips"].items():
        if sha(Path(record["directory"])/(name+".pkl")) != clip["sha256"]:
            raise ValueError("retargeted clip changed")
    if sha(record["review"]) != record["review_sha256"]:
        raise ValueError("reference review changed")
    review = json.loads(Path(record["review"]).read_text(encoding="utf8"))
    expected = {n: c["sha256"] for n, c in record["clips"].items()}
    accepted = (review.get("acceptance_basis") == DIRECT_POLICY
                and review.get("user_instruction") == DIRECT_INSTRUCTION
                and review.get("visual_review_performed") is False
                and review.get("blocking_issues") == []) if direct else review.get("approved") is True
    if (not accepted or review.get("clips") != expected
            or review.get("plan_sha256") != binding["plan_sha256"]
            or review.get("design_id") != design["id"] or review.get("urdf_sha256") != design["urdf_sha256"]
            or not review.get("observations") or not review.get("evidence")):
        raise ValueError("review does not bind actual clips/design/evidence")
    for evidence in review["evidence"]:
        if sha(evidence["path"]) != evidence["sha256"]:
            raise ValueError("review evidence changed")
    return record
