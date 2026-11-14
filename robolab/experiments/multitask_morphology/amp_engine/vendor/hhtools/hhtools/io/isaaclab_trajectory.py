"""Decode the documented IsaacLab training-convention robot NPZ arrays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ISAACLAB_TRAINING_KEYS = frozenset({"joint_names", "joint_pos", "base_pos_w", "base_quat_w"})


def decode_isaaclab_trajectory(data: Mapping[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return root xyz/xyzw plus named DoFs, without enabling pickle loading.

    Callers own archive opening and resource limits. This only recognizes the
    four documented arrays; it does not identify arbitrary IsaacLab exports.
    """
    if not ISAACLAB_TRAINING_KEYS.issubset(data):
        raise ValueError("missing IsaacLab training-convention arrays")
    names = np.asarray(data["joint_names"])
    joints = np.asarray(data["joint_pos"])
    positions = np.asarray(data["base_pos_w"])
    quaternions = np.asarray(data["base_quat_w"])
    if (
        names.ndim != 1
        or joints.ndim != 2
        or joints.shape[0] < 1
        or joints.shape[1] < 1
        or names.shape[0] != joints.shape[1]
        or positions.shape != (joints.shape[0], 3)
        or quaternions.shape != (joints.shape[0], 4)
    ):
        raise ValueError("invalid IsaacLab training-convention shapes")
    if names.dtype.kind not in {"S", "U"}:
        raise ValueError("IsaacLab joint_names must contain strings")
    dof_names = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in names.tolist()
    )
    if not all(name.strip() for name in dof_names) or len(set(dof_names)) != len(dof_names):
        raise ValueError("IsaacLab joint_names must be nonempty and unique")
    for array in (joints, positions, quaternions):
        if array.dtype.kind not in {"f", "i", "u"} or not np.isfinite(array).all():
            raise ValueError("IsaacLab trajectory arrays must contain finite real numbers")
    # The documented base_quat_w layout is wxyz; hhtools uses xyzw.
    joint_q = np.concatenate(
        (positions, quaternions[:, 1:], quaternions[:, :1], joints),
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(joint_q).all():
        raise ValueError("IsaacLab trajectory values exceed float32 range")
    if (np.linalg.norm(joint_q[:, 3:7], axis=1) <= 1e-8).any():
        raise ValueError("IsaacLab trajectory contains a zero root quaternion")
    return joint_q, dof_names
