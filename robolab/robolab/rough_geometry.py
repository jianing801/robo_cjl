"""Rough-only URDF inertia/topology audit and neutral-pose forward kinematics.

Geometry utilities only: no policy transfer, teacher, MTG or AMP dependency.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

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
