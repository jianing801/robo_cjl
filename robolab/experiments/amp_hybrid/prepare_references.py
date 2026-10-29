"""CPU HHTools R2R for a new spatial AMP bank; no contact repair or PPO."""
import argparse
import importlib.util
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent/"rsl_rl")]


def main():
    from amp_bank import load_configurable_bank
    from amp_references import (sha, write_new, native_weights, load_payload,
        validate_payload, encode_payload, kinematic_signature, KEY_BODIES, validate_library)
    from robolab.amp_reference_serialization import dump_portable
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True); p.add_argument("--output", required=True)
    p.add_argument("--hhtools-root")
    a = p.parse_args()
    bank = load_configurable_bank(a.bank)
    if bank.get("task_key") != "run":
        raise ValueError("AMP reference generation requires a RUN bank")
    if a.hhtools_root:
        hh = Path(a.hhtools_root).resolve()
    else:
        spec = importlib.util.find_spec("hhtools")
        if spec is None:
            raise RuntimeError("HHTools is not installed in this Python; select its environment with --hhtools_python. No original-reference fallback is allowed.")
        locations = [Path(spec.origin).parent, Path(spec.origin).parent.parent]
        hh = next((v for v in locations if (v/"configs/robots/_template/robot.yaml").is_file()), None)
        if hh is None:
            raise RuntimeError("HHTools robot template not found; specify --hhtools_root")
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    names = bank["canonical_semantic_contract"]["joint_names"]
    weights = native_weights()
    source = {}
    for name in weights:
        path = ROOT/"data/motions/rpo_lab"/(name+".pkl")
        source[name] = dict(path=str(path), sha256=sha(path))
    plan = dict(joint_names=names, source_urdf=bank["original_template"], hhtools_root=str(hh))
    from hhtools_backend import Backend
    backend = Backend(plan, out)
    cache, bindings = {}, {}
    for design in bank["designs"]:
        key = kinematic_signature(design["urdf"])
        if key not in cache:
            directory = out/key; directory.mkdir()
            target = backend.model(key, design["urdf"])
            hashes = {}
            for name, record in source.items():
                payload = load_payload(record["path"], names, record["sha256"])
                q = validate_payload(payload, names)
                solved, diagnostic = backend.solve(target, q, float(payload["fps"]), name)
                if solved.shape != q.shape:
                    raise ValueError("HHTools changed frame/joint count; no resampling or silent truncation allowed")
                points, soles = backend.fk_and_soles(target, solved, KEY_BODIES)
                value = encode_payload(solved, points, payload, names)
                destination = directory/(name+".pkl")
                dump_portable(destination, value)
                hashes[name] = sha(destination)
                write_new(directory/(name+".diagnostics.json"), dict(solver=diagnostic,
                    min_collision_sole_z=float(soles.min()), max_collision_sole_z=float(soles.max()),
                    contact_repair=False, dynamic_feasibility_verified=False))
            cache[key] = dict(directory=str(directory), clips=hashes)
        bindings[design["id"]] = dict(cache[key], geometry_key=key, urdf_sha256=design["urdf_sha256"])
    manifest = out/"library.json"
    write_new(manifest, dict(kind="spatial_hhtools_library_v1", bank_sha256=sha(a.bank), source_clips=source,
        joint_names=names, key_bodies=list(KEY_BODIES), weights=weights, designs=bindings,
        unique_geometry_count=len(cache), contact_repair=False, actual_gait_validated=False,
        hhtools_source_sha256={str(p.relative_to(hh)): sha(p) for p in sorted((hh/"hhtools").rglob("*.py"))},
        hhtools_template_sha256=sha(hh/"configs/robots/_template/robot.yaml")))
    validate_library(bank, a.bank, manifest)
    print("SPATIAL_AMP_REFERENCES_READY", manifest, flush=True)


if __name__ == "__main__":
    main()
