"""Versioned physical inputs for the fixed-topology RPO designs."""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "leg_motor_inertia_v1"
LINKS = ("left_thigh_pitch_link", "left_knee_link")
MOTOR_FIELDS = ("peak_torque_nm", "max_speed_rad_s", "armature_kgm2")
INERTIA_FIELDS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
LINK_FIELDS = ("mass_kg", "com_x_m", "com_y_m", "com_z_m", *INERTIA_FIELDS)
FIELDS = (
    "thigh_m", "calf_m",
    *(f"{role}_{field}" for role in ("knee", "ankle") for field in MOTOR_FIELDS),
    *(f"{segment}_{field}" for segment in ("thigh", "calf") for field in LINK_FIELDS),
)
WIDTH = len(FIELDS)


def link_values(root, name):
    inertial = root.find(f"./link[@name='{name}']/inertial")
    if inertial is None:
        raise ValueError(f"Missing segment inertia: {name}")
    origin = inertial.find("origin")
    center = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.] * 3
    roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split()) if origin is not None else (0., 0., 0.)
    if len(center) != 3:
        raise ValueError(f"Invalid segment center of mass: {name}")
    mass = float(inertial.find("mass").get("value"))
    xx, yy, zz, xy, xz, yz = (float(inertial.find("inertia").get(key)) for key in INERTIA_FIELDS)
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    if not math.isfinite(mass) or mass <= 0 or not np.isfinite(tensor).all():
        raise ValueError(f"Invalid segment mass or inertia: {name}")
    if any((roll, pitch, yaw)):
        # The tensor remains about the CoM. Rotate the axes, not its reference point.
        cr, cp, cy = map(math.cos, (roll, pitch, yaw))
        sr, sp, sy = map(math.sin, (roll, pitch, yaw))
        rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
        ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
        rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
        rotation = rz @ ry @ rx
        tensor = rotation @ tensor @ rotation.T
    eigenvalues = np.linalg.eigvalsh(tensor)
    if eigenvalues[0] <= 0 or eigenvalues[2] > eigenvalues[0] + eigenvalues[1] + 1e-10:
        raise ValueError(f"Nonphysical segment inertia: {name}")
    return [
        mass, *center,
        *[float(tensor[i, j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))],
    ]


def raw_context(design, physical, urdf_bytes=None):
    values = [design["thigh_m"], design["calf_m"]]
    for role in ("knee", "ankle"):
        if design[f"{role}_motor"] != physical[f"{role}_motor"]:
            raise ValueError("Context/physical motor mismatch")
        values.extend(physical["selected"][role][field] for field in MOTOR_FIELDS)
    root = ET.parse(design["urdf"]).getroot() if urdf_bytes is None else ET.fromstring(urdf_bytes)
    for name in LINKS:
        values.extend(link_values(root, name))
    if len(values) != WIDTH or not all(math.isfinite(value) for value in values):
        raise ValueError("Invalid physical context")
    return values


def verified_raw_context(design, physical):
    """Use the verified URDF bytes for both the identity check and extraction."""
    content = Path(design["urdf"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != design["urdf_sha256"]:
        raise ValueError("urdf asset changed")
    return raw_context(design, physical, content)


def _contexts(bank, samples=None):
    if samples is None:
        samples = [raw_context(design, bank["motor_pairs"][design["pair_id"]]) for design in bank["designs"]]
    columns = list(zip(*samples, strict=True))
    low, high = [min(col) for col in columns], [max(col) for col in columns]
    normalization = dict(kind="training_bank_minmax_v1", minimum=low, maximum=high)
    return normalization, [
        [0.0 if hi - lo <= 1e-12 else 2.0 * (value - lo) / (hi - lo) - 1.0
         for value, lo, hi in zip(sample, low, high, strict=True)]
        for sample in samples
    ]


def bind_physical_context(bank):
    normalization, contexts = _contexts(bank)
    bank.update(schema_version=3, context_schema=SCHEMA, context_width=WIDTH,
                context_fields=list(FIELDS), context_normalization=normalization)
    for design, vector in zip(bank["designs"], contexts, strict=True):
        design["context"] = vector


def validate_physical_context(bank, samples=None):
    normalization, contexts = _contexts(bank, samples)
    if (bank.get("context_schema") != SCHEMA or bank["context_width"] != WIDTH
            or bank.get("context_fields") != list(FIELDS) or bank.get("context_normalization") != normalization
            or any(row["context"] != expected for row, expected in zip(bank["designs"], contexts, strict=True))):
        raise ValueError("Frozen physical context does not match the URDFs and actuator parameters")


def validate_context_config(config, bank):
    schema = config.get("context_schema")
    if schema not in (None, SCHEMA) or schema != bank.get("context_schema"):
        raise ValueError("Training configuration and bank context schema differ")
