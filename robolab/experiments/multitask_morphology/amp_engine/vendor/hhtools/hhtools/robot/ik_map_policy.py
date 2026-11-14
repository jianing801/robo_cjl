# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Motion-aware ik_map filtering (drop head targets for selected robots)."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "drop_head_targets_for_motion",
    "filter_ik_map_for_motion",
    "ik_map_canonicals_for_motion",
    "is_omnicontact_like_motion",
]

# Unitree G1 family — head IK pulls waist_pitch on OmniContact gait bob.
_DROP_HEAD_ON_OMNICONTACT = frozenset({
    "g1",
})


def is_omnicontact_like_motion(motion: Any) -> bool:
    """True for OmniContact clips (dataset tag or 2-spine Mixamo)."""
    meta = getattr(motion, "meta", None) or {}
    if str(meta.get("dataset") or "") == "omnicontact":
        return True
    try:
        from hhtools.retarget.newton_basic.human_aliases import (
            is_two_segment_mixamo_like,
        )
    except Exception:
        return False
    names = getattr(getattr(motion, "hierarchy", None), "bone_names", None)
    return bool(names) and is_two_segment_mixamo_like(names)


def drop_head_targets_for_motion(robot_name: str, motion: Any) -> bool:
    """OmniContact + G1: skip head/neck IK so waist pitch does not chase head bob."""
    if not is_omnicontact_like_motion(motion):
        return False
    name = str(robot_name or "").strip().lower()
    if name in _DROP_HEAD_ON_OMNICONTACT:
        return True
    # Bundled / user aliases like ``g1_29dof`` if ever registered.
    return name.startswith("g1_") or name.startswith("g1-")


def filter_ik_map_for_motion(
    ik_map: Mapping[str, Any] | None,
    robot_name: str,
    motion: Any,
) -> dict[str, Any]:
    """Return a copy of ``ik_map`` with head/neck removed when policy says so."""
    out = dict(ik_map or {})
    if not drop_head_targets_for_motion(robot_name, motion):
        return out
    out.pop("head", None)
    out.pop("neck", None)
    return out


def ik_map_canonicals_for_motion(
    robot_name: str,
    ik_map: Mapping[str, Any] | None,
    motion: Any,
) -> frozenset[str]:
    """Canonical keys for yellow-overlay / stature, after motion policy."""
    return frozenset(filter_ik_map_for_motion(ik_map, robot_name, motion).keys())
