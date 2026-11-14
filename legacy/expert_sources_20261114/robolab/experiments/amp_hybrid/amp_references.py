"""Pure per-morphology AMP reference identity/serialization; no old bank dependency."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
KEY_BODIES=("left_ankle_roll_link","right_ankle_roll_link","left_knee_link","right_knee_link","left_elbow_yaw_link","right_elbow_yaw_link")


def validate_library(bank, bank_path, manifest_path, design_ids=None):
    """Exact design -> URDF -> six key bodies -> eight native clips binding."""
    library = json.loads(Path(manifest_path).read_text(encoding="utf8"))
    if library.get("kind") != "spatial_hhtools_library_v1" or library.get("bank_sha256") != sha(bank_path):
        raise ValueError("AMP reference library belongs to another bank")
    if library.get("joint_names") != bank["canonical_semantic_contract"]["joint_names"] or library.get("key_bodies") != list(KEY_BODIES):
        raise ValueError("AMP reference joint/keypoint semantics mismatch")
    if library.get("weights") != native_weights() or set(library["designs"]) != {d["id"] for d in bank["designs"]}:
        raise ValueError("AMP reference coverage/mixture mismatch")
    if set(library.get('source_clips', {})) != set(library['weights']):
        raise ValueError('missing native source clip identity')
    for name, record in library['source_clips'].items():
        source=ROOT/'data/motions/rpo_lab'/(name+'.pkl')
        if Path(record['path']).resolve()!=source.resolve() or record['sha256']!=sha(source):
            raise ValueError('native AMP reference source changed')
    active=set(design_ids) if design_ids is not None else set(library['designs'])
    if not active or not active<=set(library['designs']):
        raise ValueError('unknown or empty active reference subset')
    checked = set()
    for design in bank["designs"]:
        if design['id'] not in active:
            continue
        row = library["designs"][design["id"]]
        if row["urdf_sha256"] != design["urdf_sha256"] or row["geometry_key"] != kinematic_signature(design["urdf"]):
            raise ValueError("AMP reference bound to wrong physical morphology")
        if row["directory"] not in checked:
            for name in library["weights"]:
                path = Path(row["directory"])/(name+".pkl")
                load_payload(path, library["joint_names"], row["clips"][name])
            checked.add(row["directory"])
    return library

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
