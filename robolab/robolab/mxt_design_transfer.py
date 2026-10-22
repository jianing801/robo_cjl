"""Explicit fixed-topology design transfer; never relax ordinary checkpoint loads.

Only learned actor tensors transfer. Geometry comes from the target robot;
critic, exploration std, AMP discriminator and ALL optimizers start fresh.
This is candidate adaptation, not full-state continuation of the source run.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from robolab.mxt_policy import fixed_rpo_graph

SEGMENT_LINKS = ("left_thigh_pitch_link", "right_thigh_pitch_link", "left_knee_link", "right_knee_link")
LENGTH_JOINTS = ("left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(element):
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple(_canonical(child) for child in element))


def _vec(element, key, default="0 0 0"):
    values = np.array([float(x) for x in element.get(key, default).split()])
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("finite three-dimensional URDF vector required")
    return values


def audit_candidate_urdf(original, candidate):
    """Permit only four segment inertias/boxes and four joint z offsets.

    Mesh filenames may become absolute, but basename/visual geometry must not
    change. These are original display meshes, NOT parameterized CAD exports.
    """
    roots = [ET.parse(path).getroot() for path in (original, candidate)]
    normalized = [deepcopy(root) for root in roots]
    rows, masses = {}, []
    for index, (root, norm) in enumerate(zip(roots, normalized)):
        links = {link.get("name"): link for link in root.findall("link")}
        total = 0.
        for name, link in links.items():
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            moment = inertial.find("inertia")
            inertia = np.array([[float(moment.get(k)) for k in keys] for keys in
                               (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))])
            eig = np.linalg.eigvalsh(inertia)
            if not math.isfinite(mass) or mass <= 0 or not np.isfinite(eig).all() or eig[0] <= 0 or eig[0] + eig[1] < eig[2] - 1e-10:
                raise ValueError(f"nonphysical mass / principal inertia: {name}")
            total += mass
            if name in SEGMENT_LINKS:
                rows.setdefault(name, {})[("original", "candidate")[index]] = {
                    "mass_kg": mass, "com_m": _vec(inertial.find("origin"), "xyz").tolist(),
                    "inertia_kg_m2": inertia.tolist(), "principal_inertia": eig.tolist()}
        masses.append(total)
        for link in norm.findall("link"):
            if link.get("name") in SEGMENT_LINKS:
                link.remove(link.find("inertial"))
                for collision in link.findall("collision"):
                    box, origin = collision.find("geometry/box"), collision.find("origin")
                    if box is None or origin is None:
                        raise ValueError("candidate segments require unrotated box collisions")
                    size, xyz = _vec(box, "size"), _vec(origin, "xyz")
                    if np.any(size <= 0) or not np.allclose(_vec(origin, "rpy"), 0, atol=1e-10):
                        raise ValueError("invalid segment collision")
                    # Keep transverse dimensions and position checked below.
                    box.set("size", " ".join(map(str, [*size[:2], 0.])))
                    origin.set("xyz", " ".join(map(str, [*xyz[:2], 0.])))
            for mesh in link.findall(".//mesh"):
                mesh.set("filename", mesh.get("filename").replace("\\", "/").split("/")[-1])
        for joint in norm.findall("joint"):
            if joint.get("name") in LENGTH_JOINTS:
                xyz = _vec(joint.find("origin"), "xyz")
                joint.find("origin").set("xyz", " ".join(map(str, [*xyz[:2], 0.])))
    if _canonical(normalized[0]) != _canonical(normalized[1]):
        raise ValueError("candidate changes topology, motor limits, visuals or non-leg physics")
    lengths = {}
    for root, label in zip(roots, ("original", "candidate")):
        joints = {j.get("name"): j for j in root.findall("joint")}
        lengths[label] = {name: -float(_vec(joints[name].find("origin"), "xyz")[2]) for name in LENGTH_JOINTS}
        if any(value <= 0 for value in lengths[label].values()):
            raise ValueError("positive leg lengths required")
    return {"original_urdf_sha256": file_sha(original), "candidate_urdf_sha256": file_sha(candidate),
            "total_mass_kg": dict(zip(("original", "candidate"), masses)), "segment_properties": rows,
            "joint_lengths_m": lengths, "topology_joint_limits_and_nonleg_physics_unchanged": True,
            "visual_meshes_not_parameterized": True, "cad_fit_accuracy_not_verified_by_this_audit": True,
            "simulation_started": False}


def candidate_contract(source_contract, candidate_urdf):
    """Build the planned graph; actual simulator contract must match it later."""
    target = deepcopy(source_contract)
    target["urdf_sha256"] = file_sha(candidate_urdf)
    target["mtg"] = fixed_rpo_graph(candidate_urdf, target["joint_names"], target["default_joint_pos"])
    return target


def neutral_foot_compensation(original, candidate, joint_names, default_joint_pos, *, max_height_shift_m=.1):
    """Exact URDF forward kinematics at q0, not a guessed sum-of-lengths shift."""
    positions = dict(zip(joint_names, default_joint_pos))

    def rotation(axis, angle):
        axis = np.asarray(axis, dtype=float)
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-10:
            raise ValueError("invalid joint rotation axis")
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross

    def feet(path):
        root = ET.parse(path).getroot()
        links = {link.get("name") for link in root.findall("link")}
        remaining = list(root.findall("joint"))
        roots = links - {joint.find("child").get("link") for joint in remaining}
        if len(roots) != 1:
            raise ValueError("one connected root required")
        transforms = {next(iter(roots)): (np.eye(3), np.zeros(3))}
        while remaining:
            advanced = False
            for joint in remaining[:]:
                parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
                if parent not in transforms:
                    continue
                rot, pos = transforms[parent]
                origin = joint.find("origin")
                xyz = _vec(origin, "xyz") if origin is not None else np.zeros(3)
                roll, pitch, yaw = _vec(origin, "rpy") if origin is not None else np.zeros(3)
                local = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
                if joint.get("type") != "fixed":
                    if joint.get("type") not in ("revolute", "continuous") or joint.get("name") not in positions:
                        raise ValueError("unknown default position / joint type")
                    axis = joint.find("axis")
                    local = local @ rotation(_vec(axis, "xyz", "1 0 0") if axis is not None else [1, 0, 0], positions[joint.get("name")])
                transforms[child] = (rot @ local, pos + rot @ xyz)
                remaining.remove(joint)
                advanced = True
            if not advanced:
                raise ValueError("disconnected or cyclic URDF")
        return {name: transforms[name][1] for name in ("left_ankle_roll_link", "right_ankle_roll_link")}

    before, after = feet(original), feet(candidate)
    offsets = [float(before[name][2] - after[name][2]) for name in before]
    if abs(offsets[0] - offsets[1]) > 1e-6:
        raise ValueError("asymmetric neutral foot heights require explicit reset design")
    shift = sum(offsets) / 2
    if not math.isfinite(max_height_shift_m) or not 0 < max_height_shift_m <= .31:
        raise ValueError("explicit bounded neutral-height correction required")
    if not math.isfinite(shift) or abs(shift) > max_height_shift_m:
        raise ValueError("first pilot requires a small neutral-height correction")
    return {"root_height_compensation_m": shift,
            "original_foot_origins_m": {n: p.tolist() for n, p in before.items()},
            "candidate_foot_origins_m": {n: p.tolist() for n, p in after.items()},
            "method": "URDF_forward_kinematics_at_unchanged_default_joint_angles",
            "preserves_neutral_foot_height_not_dynamic_stability_proof": True}


def transfer_config(source_config, target_contract):
    if "stairs_adapter" in source_config:
        raise ValueError("use the retained original shared policy, not a rejected adapter candidate")
    source = source_config["io_contract"]
    if source_config["graph"] != source["mtg"]:
        raise ValueError("source graph and IO metadata disagree")
    for name in set(source) | set(target_contract):
        if name not in ("urdf_sha256", "mtg") and source.get(name) != target_contract.get(name):
            raise ValueError(f"candidate changes observation/action semantics: {name}")
    before, after = source["mtg"], target_contract["mtg"]
    if before["urdf_sha256"] != source["urdf_sha256"] or after["urdf_sha256"] != target_contract["urdf_sha256"]:
        raise ValueError("graph and physical URDF identities disagree")
    for name in set(before) | set(after):
        if name not in ("urdf_sha256", "static_features") and before.get(name) != after.get(name):
            raise ValueError(f"candidate changes topology / node identity: {name}")
    a, b = np.asarray(before["static_features"]), np.asarray(after["static_features"])
    if a.shape != (25, 14) or b.shape != a.shape or not np.isfinite(b).all():
        raise ValueError("invalid candidate static graph")
    # Only position and child mass are design attributes; axes/limits/q0 stay fixed.
    keep = [3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    if not np.array_equal(a[:, keep], b[:, keep]) or not np.array_equal(a[[0, 24]], b[[0, 24]]):
        raise ValueError("candidate changes axes, joint limits, parent identity, defaults or root")
    return {**deepcopy(source_config), "graph": deepcopy(after), "io_contract": deepcopy(target_contract)}


def actor_only_state(source_checkpoint, fresh_target_state):
    source = source_checkpoint["model_state_dict"]
    extra, target_extra = source["_extra_state"], fresh_target_state["_extra_state"]
    if extra.get("kind") != "rpo_mxt" or target_extra.get("kind") != "rpo_mxt":
        raise ValueError("plain shared MxT policies required for design transfer")
    expected = transfer_config(extra["config"], target_extra["config"]["io_contract"])
    if expected != target_extra["config"]:
        raise ValueError("candidate architecture differs from common initialization")
    for state, extra_state in ((source, extra), (fresh_target_state, target_extra)):
        if not torch.equal(state["actor.static_features"].cpu(),
                           torch.tensor(extra_state["config"]["graph"]["static_features"], dtype=torch.float32)):
            raise ValueError("checkpoint static buffer disagrees with declared geometry")
    keys = {name for name in source if name.startswith("actor.") and name != "actor.static_features"}
    target_keys = {name for name in fresh_target_state if name.startswith("actor.") and name != "actor.static_features"}
    if keys != target_keys:
        raise ValueError("learned actor parameter names differ")
    migrated = deepcopy(fresh_target_state)
    for name in keys:
        a, b = source[name], fresh_target_state[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all():
            raise ValueError(f"incompatible learned actor tensor: {name}")
        migrated[name] = a.detach().clone()
    return migrated, {"mode": "actor_only_candidate_initialization_NOT_resume", "copied_actor_tensor_count": len(keys),
                      "source_iteration_not_inherited": True, "optimizer_state_not_inherited": True,
                      "critic_std_discriminator_normalizer_not_inherited": True,
                      "candidate_geometry_not_overwritten": True}


def benchmark_plan(*, source_sha, audit, num_envs=128, warmup_updates=25, actor_updates=100, seed=93117,
                   physical_model=None):
    if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("source SHA256 required")
    if num_envs < 8 or num_envs % 8 or not 1 <= warmup_updates <= 100 or not 1 <= actor_updates <= 500:
        raise ValueError("finite equal-budget pilot bounds required")
    if not isinstance(seed, int) or seed in (12345, 23456, 89341):
        raise ValueError("training seed must be separate from declared evaluation seeds")
    if physical_model is not None:
        from robolab.physical_snapshot import validate_physical_model
        validate_physical_model(physical_model)
    result = {"schema_version": 1, "kind": "mxt_design_transfer_pilot", "source_sha256": source_sha,
            "physical_audit": audit, "enable_motor_curve": physical_model is not None, "backpack": False,
            "num_envs": num_envs, "rollout_steps": 24, "warmup_updates": warmup_updates,
            "actor_updates": actor_updates, "train_seed": seed,
            "transitions_per_training_arm": num_envs * 24 * (warmup_updates + actor_updates),
            "arms": ["zero_shot_actor_transfer", "scratch", "actor_transfer"],
            "independent_arms_not_sequential_resume": True,
            "both_training_arms_fresh_critic_discriminator_normalizers_optimizers": True,
            "learning_rates": {"warmup": .001, "actor": .0001}, "fixed_action_std": .1,
            "evaluation": {"seeds": [12345, 23456], "final_holdout_seed": 89341,
                           "num_envs": 16, "duration_s": 20., "run_speed": 2.4, "stairs_speed": .5,
                           "step_height": .1, "step_width": .25, "heading_gain": 2., "yaw_rate_limit": .4,
                           "min_speed_ratio": .5, "max_lateral_ratio": .2,
                           "minimum_task_success_survival": .9,
                           "profiles": ["nominal", "domain_randomized_with_noise"]},
            "cost_fields_required": ["environment_transitions", "training_wall_seconds_including_startup",
                                     "evaluation_wall_seconds_including_startup", "source_pretraining",
                                     "expert_training_collection_distillation", "failed_attempts"],
            "result_is_single_design_single_seed_pilot_not_cost_saving_proof": True,
            "scratch_budget_not_convergence_claim": True,
            "simulation_started": False, "execution_requires_explicit_urdf_update_authorization": True}
    if physical_model is not None:
        result["physical_model"] = deepcopy(physical_model)
    return result
