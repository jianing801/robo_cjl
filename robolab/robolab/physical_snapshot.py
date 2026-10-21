"""Pinned upstream physical assets, isolated from the old experiment checkout.

Verify original Git blob hashes before import. Only the catalog import name in
the two upstream modules is remapped in memory to the pinned catalog; no GP,
actuator formula, or old global module is replaced. Isaac imports occur only
when configure_robot is called AFTER AppLauncher.
"""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import types

COMMIT = "1450a51c46a8b26e95a5aaee4bc850bba9e35115"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "vendor/physical_1450a51"
CATALOG = "robolab/robolab/assets/rpo_motor_catalog.py"
GENERATOR = "robolab/scripts/tools/generate_urdf_sw.py"
ROBOT = "robolab/robolab/assets/robots/roboparty.py"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_snapshot(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf8"))
    if manifest.get("repository") != "jianing801/robo_cjl" or manifest.get("commit") != COMMIT:
        raise ValueError("unexpected physical repository / revision")
    seen, hashes = set(), {}
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or row["path"] in seen:
            raise ValueError("duplicate / escaping snapshot path")
        seen.add(row["path"])
        data = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != row["git_blob_sha"]:
            raise ValueError(f"upstream physical asset modified: {row['path']}")
        hashes[row["path"]] = hashlib.sha256(data).hexdigest()
    if not {CATALOG, GENERATOR, ROBOT}.issubset(seen):
        raise ValueError("incomplete physical snapshot")
    return {"snapshot_root": str(root), "upstream_commit": COMMIT, "verified_file_sha256": hashes,
            "catalog_import_only_remapped_in_memory": True, "upstream_source_files_modified": False}


@lru_cache(maxsize=None)
def _module(root, relative, catalog_name=None):
    name = "_rpo_physical_" + hashlib.sha256((root + relative).encode()).hexdigest()[:16]
    path = Path(root) / relative
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    remapped = 0
    if catalog_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "robolab.assets.rpo_motor_catalog":
                node.module = catalog_name
                remapped += 1
        if remapped != 1:
            raise ValueError("unexpected upstream catalog-import interface")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module  # dataclasses resolve their defining module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def catalog(root=DEFAULT_ROOT):
    verify_snapshot(root)
    return _module(str(Path(root).resolve()), CATALOG)


def generator(root=DEFAULT_ROOT):
    source = catalog(root)
    return _module(str(Path(root).resolve()), GENERATOR, source.__name__)


def physical_model(*, root=DEFAULT_ROOT, knee_motor="DM-J10010L-2EC", ankle_motor="DM-J4340P-2EC",
                   enable_motor_curve=True):
    if not enable_motor_curve:
        raise ValueError("this new-physics baseline explicitly requires the selected motor envelopes")
    provenance = verify_snapshot(root)
    module = catalog(root)
    selected = {}
    for name, motor_id, choices in (("knee", knee_motor, module.KNEE_MOTOR_BO_CANDIDATES),
                                    ("ankle", ankle_motor, module.ANKLE_MOTOR_BO_CANDIDATES)):
        spec = module.get_motor_spec(motor_id, choices)
        module.require_motor_ready(spec)
        curve = module.load_torque_speed_envelope(spec)
        if curve is None:
            raise ValueError("explicit new-physics motors need a real registered curve")
        selected[name] = {"motor_id": motor_id, "urdf_model_id": spec.urdf_model_id,
                          "peak_torque_nm": spec.peak_torque_nm, "max_speed_rad_s": spec.max_speed_rad_s,
                          "armature_kgm2": spec.armature_kgm2,
                          "curve_sha256": _sha(Path(module.__file__).parent / spec.torque_speed_curve_csv),
                          "curve_points": len(curve), "model_kind": spec.model_kind}
    return {**provenance, "kind": "pinned_SW_GP_and_selected_motor_envelopes_v1",
            "knee_motor": knee_motor, "ankle_motor": ankle_motor, "selected": selected,
            "enable_motor_curve": True, "curves_are_registered_models_not_new_bench_measurements": True}


def validate_physical_model(record):
    if record != physical_model(root=record["snapshot_root"], knee_motor=record["knee_motor"],
                                ankle_motor=record["ankle_motor"], enable_motor_curve=record["enable_motor_curve"]):
        raise ValueError("physical model / curve / armature identity changed")


def configure_robot(robot_cfg, record):
    validate_physical_model(record)
    module = catalog(record["snapshot_root"])
    upstream = _module(str(Path(record["snapshot_root"]).resolve()), ROBOT, module.__name__)
    selection = upstream.configure_rpo_motors(robot_cfg, knee_motor=record["knee_motor"],
        ankle_motor=record["ankle_motor"], use_torque_speed_curve=True)
    for name, cfg_name in (("knee", "knees"), ("ankle", "feet")):
        actual, expected = selection[name], record["selected"][name]
        if (not actual["dynamic_envelope"] or actual["peak_torque_nm"] != expected["peak_torque_nm"]
                or actual["armature_kgm2"] != expected["armature_kgm2"]):
            raise ValueError("selected actuator differs from plan")
        if robot_cfg.actuators[cfg_name].class_type.__name__ != "TorqueSpeedEnvelopeActuator":
            raise ValueError("dynamic envelope actuator not installed")
    return selection


def live_actuator_audit(robot, record):
    """Actual runtime clipping check after real physics steps, not config alone."""
    import torch
    result = {}
    for label, name in (("knee", "knees"), ("ankle", "feet")):
        actuator = robot.actuators[name]
        expected = record["selected"][label]
        if type(actuator).__name__ != "TorqueSpeedEnvelopeActuator":
            raise ValueError("live robot is not using its planned dynamic actuator")
        lookup = torch.as_tensor(actuator.cfg.torque_speed_lookup, device=actuator.applied_effort.device)
        if lookup.shape[0] != expected["curve_points"]:
            raise ValueError("live curve table differs")
        clipped = actuator._clip_effort(actuator.computed_effort)
        error = (clipped - actuator.applied_effort).abs().max().item()
        if error > 1e-4 or not torch.isfinite(actuator.applied_effort).all():
            raise ValueError("actual motor output does not match speed-dependent clipping")
        armature_error = (actuator.armature - expected["armature_kgm2"]).abs().max().item()
        if armature_error > 1e-6:
            raise ValueError("live motor armature differs")
        result[label] = {"class": type(actuator).__name__, "joint_names": list(actuator.joint_names),
                         "curve_points": len(lookup), "armature_max_error": armature_error,
                         "actual_vs_recomputed_clipped_torque_max_error": error,
                         "max_abs_computed_torque_nm": actuator.computed_effort.abs().max().item(),
                         "max_abs_applied_torque_nm": actuator.applied_effort.abs().max().item()}
    return result
