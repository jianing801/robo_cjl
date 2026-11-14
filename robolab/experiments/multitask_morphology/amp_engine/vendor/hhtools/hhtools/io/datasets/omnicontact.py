"""OmniContact-Dataset adapter (human BVH + rigid object CSV).

Official release: https://huggingface.co/datasets/lightcone02/OmniContact-Dataset

hhtools consumes the **raw mocap** subset, not the already-retargeted G1
``npz/`` trajectories. Each capture directory looks like::

    raw_mocap/<domain>/<case>/<capture_id>/
        motion_actor.bvh          # actor skeleton (optical mocap, typically 90 Hz)
        motion_actor.csv          # actor timestamps
        prop_*.csv | object_pose_*.csv
        capture_meta.json

Drop the Hugging Face tree (or just ``raw_mocap/``) under a folder named
``OmniContact-Dataset`` — e.g. ``assets/motions/intermimic/OmniContact-Dataset/``.
Object meshes are resolved from a sibling ``assets/`` directory when present.

Object CSVs are paired to the BVH timeline by timestamp (or by normalized
index when timestamps are missing). Positions / rotations are converted from
the source Y-up mocap frame into hhtools Z-up, matching :func:`load_bvh`.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hhtools.core.math import quaternion as Q
from hhtools.core.math import rotation as R
from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject
from hhtools.io.bvh import load_bvh
from hhtools.io.datasets.base import DatasetAdapter, register_dataset

_SOURCE_URL = "https://huggingface.co/datasets/lightcone02/OmniContact-Dataset"
_ACTOR_BVH = "motion_actor.bvh"
_ACTOR_CSV = "motion_actor.csv"
_META_JSON = "capture_meta.json"
_SKIP_DIR_PARTS = frozenset({"npz", "npz_clips", "bvh_cache", "demo", "assets", "metadata"})

_OBJECT_EXTENTS_METRES: dict[str, tuple[float, float, float]] = {
    "box": (0.40, 0.30, 0.30),
    "largebox": (0.50, 0.40, 0.35),
    "smallbox": (0.30, 0.22, 0.20),
    "soccer": (0.22, 0.22, 0.22),
    "ball": (0.22, 0.22, 0.22),
    "soccerball": (0.22, 0.22, 0.22),
}
_DEFAULT_OBJECT_EXTENTS = (0.35, 0.28, 0.28)

_POS_ALIASES = {
    "x": 0, "y": 1, "z": 2,
    "px": 0, "py": 1, "pz": 2,
    "tx": 0, "ty": 1, "tz": 2,
    "pos_x": 0, "pos_y": 1, "pos_z": 2,
    "position_x": 0, "position_y": 1, "position_z": 2,
    "posx": 0, "posy": 1, "posz": 2,
}
_TIME_ALIASES = frozenset({"t", "time", "timestamp", "sec", "seconds", "frame_time"})
_FRAME_ALIASES = frozenset({"frame", "frames", "frame_index", "index", "id"})
_QUAT_ALIASES = {
    "qx": "x", "qy": "y", "qz": "z", "qw": "w",
    "q_x": "x", "q_y": "y", "q_z": "z", "q_w": "w",
    "quat_x": "x", "quat_y": "y", "quat_z": "z", "quat_w": "w",
}
_EULER_ALIASES = {
    "rx": 0, "ry": 1, "rz": 2,
    "roll": 0, "pitch": 1, "yaw": 2,
    "rot_x": 0, "rot_y": 1, "rot_z": 2,
    "rotation_x": 0, "rotation_y": 1, "rotation_z": 2,
}


def is_omnicontact_capture(path: str | Path) -> bool:
    """True when ``path`` is an OmniContact actor BVH or a capture directory."""
    path = Path(path)
    if path.is_dir():
        return (path / _ACTOR_BVH).is_file()
    return path.is_file() and path.name.lower() == _ACTOR_BVH


def _normalise_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _object_name_from_csv(path: Path) -> str:
    stem = path.stem
    lowered = stem.lower()
    for prefix in ("object_pose_", "object_poses_", "object_", "prop_pose_", "prop_"):
        if lowered.startswith(prefix):
            rest = stem[len(prefix) :]
            return rest or stem
    return stem


def _lookup_extents(object_name: str) -> NDArray:
    key = re.sub(r"[^a-z0-9]+", "", object_name.lower())
    for token, extents in _OBJECT_EXTENTS_METRES.items():
        if token in key:
            return np.asarray(extents, dtype=np.float32)
    return np.asarray(_DEFAULT_OBJECT_EXTENTS, dtype=np.float32)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    return []


def _load_capture_meta(capture_dir: Path) -> dict[str, Any]:
    meta_path = capture_dir / _META_JSON
    return _read_json(meta_path) if meta_path.is_file() else {}


def _object_entries(capture_dir: Path, meta: dict[str, Any]) -> list[tuple[Path, str]]:
    """Return ``(csv_path, object_name)`` pairs from ``capture_meta.json`` or globs."""
    entries: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def _add(path: Path, name: str) -> None:
        if not path.is_file():
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        clean = name or _object_name_from_csv(path)
        if clean.lower().startswith("prop_"):
            clean = clean[5:]
        entries.append((path, clean))

    declared_objs = meta.get("objects")
    if isinstance(declared_objs, list):
        for obj in declared_objs:
            if isinstance(obj, dict):
                rel = obj.get("published_file") or obj.get("source_file") or obj.get("file") or obj.get("csv")
                if rel:
                    _add(capture_dir / Path(str(rel)).name, str(obj.get("object_name") or ""))
            elif isinstance(obj, str):
                _add(capture_dir / Path(obj).name, "")
    for rel in _as_str_list(meta.get("object_files")):
        _add(capture_dir / Path(rel).name, "")

    if entries:
        return entries
    for pattern in ("prop_*.csv", "object_pose_*.csv", "object_poses_*.csv"):
        for p in sorted(capture_dir.glob(pattern)):
            if p.name.lower() == _ACTOR_CSV:
                continue
            _add(p, "")
    return entries


def _parse_numeric_table(path: Path) -> tuple[list[str], NDArray]:
    """Read a CSV/TSV into ``(headers, rows)``. Headers may be empty."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    rows: list[list[str]] = []
    reader = csv.reader(raw.splitlines(), dialect)
    for row in reader:
        if not row or all(not str(c).strip() for c in row):
            continue
        rows.append([str(c).strip() for c in row])
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float64)

    def _row_is_numeric(row: list[str]) -> bool:
        usable = 0
        for cell in row:
            if cell == "":
                continue
            try:
                float(cell)
                usable += 1
            except ValueError:
                return False
        return usable > 0

    header: list[str] = []
    data_rows = rows
    if not _row_is_numeric(rows[0]):
        header = [_normalise_header(c) for c in rows[0]]
        data_rows = rows[1:]
        # Qualisys-style multi-line headers: keep skipping until numbers appear.
        while data_rows and not _row_is_numeric(data_rows[0]):
            header = [_normalise_header(c) for c in data_rows[0]]
            data_rows = data_rows[1:]

    parsed: list[list[float]] = []
    width = max((len(r) for r in data_rows), default=0)
    for row in data_rows:
        if not _row_is_numeric(row):
            continue
        vals: list[float] = []
        for i in range(width):
            cell = row[i] if i < len(row) else ""
            if cell == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(cell))
                except ValueError:
                    vals.append(np.nan)
        parsed.append(vals)
    if not parsed:
        return header, np.zeros((0, width), dtype=np.float64)
    return header, np.asarray(parsed, dtype=np.float64)


def _extract_times(headers: list[str], table: NDArray, *, framerate: float) -> NDArray:
    n = int(table.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if headers:
        time_cols = [i for i, h in enumerate(headers) if h in _TIME_ALIASES]
        if time_cols:
            t = table[:, time_cols[0]]
            if np.isfinite(t).any():
                return np.where(np.isfinite(t), t, np.nan)
        frame_cols = [i for i, h in enumerate(headers) if h in _FRAME_ALIASES]
        if frame_cols:
            frames = table[:, frame_cols[0]]
            if np.isfinite(frames).any():
                return frames / max(float(framerate), 1e-6)
    if _first_column_is_time(headers, table):
        return table[:, 0]
    return np.arange(n, dtype=np.float64) / max(float(framerate), 1e-6)


def _first_column_is_time(headers: list[str], table: NDArray) -> bool:
    if headers and headers[0] in _TIME_ALIASES | _FRAME_ALIASES:
        return True
    # Headerless: only [t, x, y, z] or [t, x, y, z, qx, qy, qz, qw] start with time.
    # 7-col [x, y, z, qx, qy, qz, qw] must keep column 0 as X.
    if table.shape[1] not in {4, 8}:
        return False
    first = table[:, 0]
    finite = first[np.isfinite(first)]
    if finite.size < 2:
        return False
    diffs = np.diff(finite)
    if np.any(diffs < -1e-9):
        return False
    span = float(finite[-1] - finite[0])
    return span >= 0.0 and span < 1.0e5 and float(np.max(np.abs(finite))) < 1.0e6


def _extract_positions(headers: list[str], table: NDArray) -> NDArray | None:
    if table.size == 0:
        return None
    if headers:
        xyz = [-1, -1, -1]
        for i, h in enumerate(headers):
            if h in _POS_ALIASES:
                xyz[_POS_ALIASES[h]] = i
        if all(i >= 0 for i in xyz):
            return table[:, xyz]
    if table.shape[1] < 3:
        return None
    start = 1 if _first_column_is_time(headers, table) else 0
    if table.shape[1] < start + 3:
        return None
    return table[:, start : start + 3]


def _quat_xyzw_from_wxyz(qwxyz: NDArray) -> NDArray:
    q = np.asarray(qwxyz, dtype=np.float64)
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def _infer_w_index(quat: NDArray) -> int:
    """Return 0 if leading component is W, else 3 (xyzw)."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    finite = np.isfinite(q).all(axis=1)
    if not np.any(finite):
        return 3
    sample = q[finite][:32]
    # Identity-ish rows: the W component sits near ±1 with xyz near 0.
    lead = np.mean(np.abs(sample[:, 0]))
    trail = np.mean(np.abs(sample[:, 3]))
    lead_xyz = np.mean(np.linalg.norm(sample[:, 1:], axis=1))
    trail_xyz = np.mean(np.linalg.norm(sample[:, :3], axis=1))
    if lead > 0.7 and lead_xyz < 0.5 and lead >= trail:
        return 0
    if trail > 0.7 and trail_xyz < 0.5:
        return 3
    return 0 if lead > trail else 3


def _extract_quaternions(headers: list[str], table: NDArray) -> NDArray | None:
    n = int(table.shape[0])
    if n == 0:
        return None
    if headers:
        comps: dict[str, int] = {}
        for i, h in enumerate(headers):
            if h in _QUAT_ALIASES:
                comps[_QUAT_ALIASES[h]] = i
            elif h in {"x", "y", "z", "w"} and h not in _POS_ALIASES:
                # bare w is unambiguous; bare x/y/z already claimed by positions.
                if h == "w":
                    comps["w"] = i
        if {"x", "y", "z", "w"}.issubset(comps):
            q = table[:, [comps["x"], comps["y"], comps["z"], comps["w"]]]
            return q
        # rotation-matrix columns
        mat_idx = []
        for name in (
            "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
        ):
            if name in headers:
                mat_idx.append(headers.index(name))
        if len(mat_idx) == 9:
            mats = table[:, mat_idx].reshape(n, 3, 3)
            from scipy.spatial.transform import Rotation as SciR

            return SciR.from_matrix(mats).as_quat()  # xyzw
        eul = [-1, -1, -1]
        for i, h in enumerate(headers):
            if h in _EULER_ALIASES:
                eul[_EULER_ALIASES[h]] = i
        if all(i >= 0 for i in eul):
            from scipy.spatial.transform import Rotation as SciR

            angles = table[:, eul]
            degrees = float(np.nanmax(np.abs(angles))) > np.pi + 0.2
            return SciR.from_euler("xyz", angles, degrees=degrees).as_quat()

    start = 1 if _first_column_is_time(headers, table) else 0
    quat_start = start + 3
    if table.shape[1] >= quat_start + 4:
        quat = table[:, quat_start : quat_start + 4]
        w_at = _infer_w_index(quat)
        return quat if w_at == 3 else _quat_xyzw_from_wxyz(quat)
    return None


def _finite_interpolate(
    src_t: NDArray,
    src_y: NDArray,
    dst_t: NDArray,
    *,
    is_quat: bool = False,
) -> NDArray:
    """Interpolate ``src_y`` sampled at ``src_t`` onto ``dst_t``."""
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    dst_t = np.asarray(dst_t, dtype=np.float64).reshape(-1)
    src_y = np.asarray(src_y, dtype=np.float64)
    if src_y.ndim == 1:
        src_y = src_y.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    mask = np.isfinite(src_t) & np.isfinite(src_y).all(axis=1)
    if not np.any(mask):
        out = np.zeros((dst_t.shape[0], src_y.shape[1]), dtype=np.float64)
        return out[:, 0] if squeeze else out
    t = src_t[mask]
    y = src_y[mask]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    # Collapse duplicate timestamps (keep last).
    _, uniq = np.unique(t, return_index=True)
    t = t[np.sort(uniq)]
    y = y[np.sort(uniq)]
    if t.shape[0] == 1:
        out = np.repeat(y, dst_t.shape[0], axis=0)
        return out[:, 0] if squeeze else out

    if is_quat:
        from scipy.spatial.transform import Rotation as SciR
        from scipy.spatial.transform import Slerp

        # scipy Slerp is xyzw / scalar-last — same as us.
        rot = SciR.from_quat(y)
        sl = Slerp(t, rot)
        clipped = np.clip(dst_t, t[0], t[-1])
        out = sl(clipped).as_quat()
        return out.astype(np.float64)

    out = np.stack(
        [np.interp(dst_t, t, y[:, i]) for i in range(y.shape[1])],
        axis=1,
    )
    return out[:, 0] if squeeze else out


def _load_actor_times(capture_dir: Path, meta: dict[str, Any], *, num_frames: int, framerate: float) -> NDArray:
    stamp_name = str(meta.get("timestamps") or meta.get("timestamp_file") or _ACTOR_CSV)
    stamp_path = capture_dir / stamp_name
    if not stamp_path.is_file():
        stamp_path = capture_dir / _ACTOR_CSV
    if stamp_path.is_file():
        headers, table = _parse_numeric_table(stamp_path)
        if table.shape[0] > 0:
            times = _extract_times(headers, table, framerate=framerate)
            if times.shape[0] == num_frames:
                return times.astype(np.float64)
            if times.shape[0] > 0:
                src_idx = np.linspace(0.0, 1.0, times.shape[0])
                dst_idx = np.linspace(0.0, 1.0, num_frames)
                return np.interp(dst_idx, src_idx, times).astype(np.float64)
    return np.arange(num_frames, dtype=np.float64) / max(float(framerate), 1e-6)


def _apply_y_up_to_z_up(positions: NDArray, quaternions: NDArray) -> tuple[NDArray, NDArray]:
    rot_mat = R.up_axis_rotation("Y", "Z")
    positions = np.asarray(positions, dtype=np.float32) @ rot_mat.T
    rot_quat = Q.from_matrix(rot_mat)
    quats = Q.multiply(
        np.broadcast_to(rot_quat, quaternions.shape),
        np.asarray(quaternions, dtype=np.float32),
    )
    return positions.astype(np.float32), Q.normalize(quats).astype(np.float32)


def _maybe_rescale_object(
    positions: NDArray,
    human_positions: NDArray,
) -> NDArray:
    """Guess millimetre / centimetre object CSVs from the human metre-scale track."""
    pos = np.asarray(positions, dtype=np.float64)
    human = np.asarray(human_positions, dtype=np.float64)
    if pos.size == 0 or human.size == 0:
        return positions
    obj_span = float(np.nanmax(np.linalg.norm(pos - pos[0], axis=1)))
    human_span = float(np.nanmax(np.linalg.norm(human[:, 0] - human[0, 0], axis=1)))
    obj_abs = float(np.nanmedian(np.abs(pos)))
    if not np.isfinite(obj_span) or obj_span < 1e-8:
        obj_span = obj_abs
    if human_span < 1e-6:
        human_span = 1.0
    ratio = obj_span / human_span
    if ratio > 50.0:
        # cm (×100) or mm (×1000)
        scale = 0.001 if ratio > 400.0 or obj_abs > 200.0 else 0.01
        return (pos * scale).astype(np.float32)
    return positions.astype(np.float32)


def _find_object_mesh(start: Path, object_name: str) -> str:
    """Search the clip folder (OMOMO-style sidecar) then nearby ``assets/`` trees."""
    if not object_name:
        return ""
    tokens = [object_name.lower(), re.sub(r"[^a-z0-9]+", "", object_name.lower())]
    for ext in (".obj", ".stl", ".ply", ".glb"):
        sibling = start / f"{object_name}{ext}"
        if sibling.is_file():
            return str(sibling.resolve())
    for path in start.iterdir() if start.is_dir() else []:
        if path.suffix.lower() not in {".obj", ".stl", ".ply", ".glb"}:
            continue
        lowered = path.stem.lower()
        compact = re.sub(r"[^a-z0-9]", "", lowered)
        if any(tok and (tok in lowered or tok in compact) for tok in tokens if tok):
            return str(path.resolve())
    roots: list[Path] = []
    for parent in (start, *start.parents):
        for child in (parent / "assets", parent / "Assets"):
            if child.is_dir():
                roots.append(child)
        if re.sub(r"[^a-z0-9]", "", parent.name.lower()) in {
            "omnicontactdataset",
            "omnicontact",
        }:
            break
        if parent.name.lower() in {"raw_mocap", "intermimic", "motions", "assets"}:
            # keep walking one more level for the dataset root's assets/
            continue
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for ext in (".obj", ".stl", ".ply", ".glb"):
            for path in root.rglob(f"*{ext}"):
                lowered = path.stem.lower()
                compact = re.sub(r"[^a-z0-9]", "", lowered)
                if any(tok and (tok in lowered or tok in compact) for tok in tokens if tok):
                    return str(path.resolve())
    return ""


def _load_object_track(
    csv_path: Path,
    *,
    actor_times: NDArray,
    human_positions: NDArray,
    framerate: float,
    object_name: str | None = None,
    extents: NDArray | None = None,
    mesh_path: str = "",
    source_up_axis: str = "Y",
) -> SceneObject:
    headers, table = _parse_numeric_table(csv_path)
    name = object_name or _object_name_from_csv(csv_path)
    positions = _extract_positions(headers, table)
    if positions is None or positions.shape[0] == 0:
        raise ValueError(f"{csv_path}: no object positions found")
    quats = _extract_quaternions(headers, table)
    if quats is None:
        quats = np.zeros((positions.shape[0], 4), dtype=np.float64)
        quats[:, 3] = 1.0
    src_t = _extract_times(headers, table, framerate=framerate)
    if src_t.shape[0] != positions.shape[0]:
        src_t = np.arange(positions.shape[0], dtype=np.float64) / max(float(framerate), 1e-6)

    dst_t = np.asarray(actor_times, dtype=np.float64)
    if positions.shape[0] == dst_t.shape[0] and np.allclose(
        src_t[: min(3, src_t.size)], dst_t[: min(3, dst_t.size)], atol=1e-3
    ):
        aligned_p = positions
        aligned_q = quats
    else:
        aligned_p = _finite_interpolate(src_t, positions, dst_t, is_quat=False)
        aligned_q = _finite_interpolate(src_t, quats, dst_t, is_quat=True)

    aligned_p = _maybe_rescale_object(aligned_p, human_positions)
    aligned_q = Q.normalize(np.asarray(aligned_q, dtype=np.float32))
    if source_up_axis.upper() == "Y":
        aligned_p, aligned_q = _apply_y_up_to_z_up(aligned_p, aligned_q)

    resolved_mesh = mesh_path or _find_object_mesh(csv_path.parent, name)
    scale = _infer_mesh_scale(resolved_mesh)
    if extents is None:
        extents_arr = _mesh_extents_metres(resolved_mesh, scale)
        if extents_arr is None:
            extents_arr = _lookup_extents(name)
    else:
        extents_arr = np.asarray(extents, dtype=np.float32)

    return SceneObject(
        name=name,
        positions=np.asarray(aligned_p, dtype=np.float32),
        quaternions=np.asarray(aligned_q, dtype=np.float32),
        extents=extents_arr,
        mesh_path=resolved_mesh,
        scale=scale,
    )


def _infer_mesh_scale(mesh_path: str) -> float:
    """OmniContact object OBJs are authored in centimetres (see file header)."""
    if not mesh_path:
        return 1.0
    path = Path(mesh_path)
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2048].lower()
    except OSError:
        head = ""
    if "millimeter" in head:
        return 0.001
    if "centimeter" in head:
        return 0.01
    try:
        import trimesh

        mesh = trimesh.load(str(path), force="mesh", process=False)
        verts = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
        if verts.size:
            span = float(np.ptp(verts, axis=0).max())
            if span > 80.0:
                return 0.001
            if span > 3.0:
                return 0.01
    except Exception:
        pass
    return 1.0


def _mesh_extents_metres(mesh_path: str, scale: float) -> NDArray | None:
    if not mesh_path:
        return None
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        verts = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
        if verts.size == 0:
            return None
        span = np.ptp(verts, axis=0) * float(scale)
        return np.maximum(span, 1e-3).astype(np.float32)
    except Exception:
        return None


def _bvh_unit_from_meta(meta: dict[str, Any]) -> str:
    unit = str(meta.get("unit") or meta.get("bvh_unit") or meta.get("length_unit") or "cm")
    return unit.strip().lower() or "cm"


def _capture_label(bvh_path: Path, dataset_root: Path) -> str:
    capture_id = bvh_path.parent.name
    try:
        rel = bvh_path.parent.relative_to(dataset_root)
        parts = [p for p in rel.parts if p.lower() != "raw_mocap"]
        if parts:
            return "__".join(parts)
    except ValueError:
        pass
    return capture_id


@register_dataset
class OmniContactAdapter(DatasetAdapter):
    """Adapter for OmniContact-Dataset ``raw_mocap`` captures."""

    name = "omnicontact"
    display_name = "OmniContact-Dataset"
    requires = "bvh"
    file_patterns = ("**/motion_actor.bvh",)

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob(_ACTOR_BVH)):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root)
            if any(part.lower() in _SKIP_DIR_PARTS for part in rel.parts):
                continue
            yield str(rel.as_posix())

    def _resolve_bvh(self, sequence_id: str) -> Path:
        raw = Path(sequence_id)
        candidates = [
            (self.root / raw).resolve(),
            (self.root / raw / _ACTOR_BVH).resolve(),
            (self.root / _ACTOR_BVH).resolve(),
        ]
        if raw.suffix.lower() != ".bvh" and raw.name:
            candidates.append((self.root / raw.name / _ACTOR_BVH).resolve())
            matches = list(self.root.rglob(f"{raw.name}/{_ACTOR_BVH}"))
            candidates.extend(m.resolve() for m in matches)
        if raw.name.lower() in {_ACTOR_BVH, f"{self.root.name}.bvh"}:
            candidates.append((self.root / _ACTOR_BVH).resolve())
        seen: set[Path] = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if cand.is_file() and cand.suffix.lower() == ".bvh":
                return cand
        raise FileNotFoundError(
            f"OmniContact sequence not found: {self.root / sequence_id}"
        )

    def load_motion(
        self,
        sequence_id: str,
        *,
        unit: str | None = None,
        progress_callback=None,
        **_kwargs: Any,
    ) -> Motion:
        bvh_path = self._resolve_bvh(sequence_id)
        capture_dir = bvh_path.parent
        meta = _load_capture_meta(capture_dir)
        bvh_unit = unit or _bvh_unit_from_meta(meta)
        motion = load_bvh(
            bvh_path,
            unit=bvh_unit,
            target_up_axis="Z",
            progress_callback=progress_callback,
        )

        actor_times = _load_actor_times(
            capture_dir,
            meta,
            num_frames=motion.num_frames,
            framerate=motion.framerate,
        )
        source_up = str(meta.get("up_axis") or meta.get("object_up_axis") or "Y")

        objects: list[SceneObject] = []
        for csv_path, object_name in _object_entries(capture_dir, meta):
            try:
                objects.append(
                    _load_object_track(
                        csv_path,
                        actor_times=actor_times,
                        human_positions=motion.positions,
                        framerate=motion.framerate,
                        object_name=object_name,
                        source_up_axis=source_up,
                    )
                )
            except Exception as exc:
                motion.meta.setdefault("object_load_errors", []).append(
                    f"{csv_path.name}: {exc}"
                )

        domain = str(meta.get("domain") or "")
        case = str(meta.get("normalized_case") or "")
        if not domain or not case:
            rel_parts = []
            try:
                rel_parts = list(capture_dir.relative_to(self.root).parts)
            except ValueError:
                rel_parts = capture_dir.parts[-4:]
            cleaned = [p for p in rel_parts if p.lower() != "raw_mocap"]
            if len(cleaned) >= 3:
                domain, case = domain or cleaned[0], case or cleaned[1]
            elif len(cleaned) == 2:
                domain, case = domain or cleaned[0], case or cleaned[1]

        motion.name = _capture_label(bvh_path, self.root)
        motion.meta.update(
            {
                "dataset": self.name,
                "sequence_id": sequence_id,
                "capture_id": capture_dir.name,
                "domain": domain,
                "case": case,
                "object_name": objects[0].name if objects else "",
                "object_names": [o.name for o in objects],
                "source_repo": _SOURCE_URL,
                "notes": (
                    "OmniContact raw_mocap: BVH actor + object-pose CSV. "
                    "G1 npz/ trajectories are not consumed by this adapter."
                ),
            }
        )
        motion.objects = objects
        return motion


__all__ = [
    "OmniContactAdapter",
    "is_omnicontact_capture",
    "_object_entries",
    "_extract_positions",
    "_extract_quaternions",
    "_load_object_track",
    "_object_name_from_csv",
]
