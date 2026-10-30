"""Build a separate, hash-verified BO source tree for the completed Rough model."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


EXPECTED_REPAIRS = {
    "hybrid:persistent_worker.py": "8dc97306cba7812c141fcb72a8233fd288a3ee4be31b31c4d0217a803a71c0c7",
    "hybrid:train.py": "872cad0dece779c38be133e199904fa541f496a58939b2a693b5890991958558",
    "hybrid:train_stage.py": "88648149221118542cbb986d14005d0008a5e96fa23d2fc52aab848bc2714108",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def expected_files(identity):
    if not identity.get("native_sources") or not identity.get("hybrid_sources"):
        raise ValueError("Checkpoint source identity is incomplete")
    return {**{f"native:{k}": v for k, v in identity["native_sources"].items()},
            **{f"hybrid:{k}": v for k, v in identity["hybrid_sources"].items()}}


def source_path(root, key):
    group, relative = key.split(":", 1)
    return (root if group == "native" else root / "experiments/rough_hybrid") / relative


def audit(root, identity):
    rows = {}
    boundary = root.resolve().parent
    for key, expected in expected_files(identity).items():
        path = source_path(root, key)
        if not path.resolve().is_relative_to(boundary):
            raise ValueError(f"Source identity escapes the repository: {key}")
        actual = sha(path.read_bytes()) if path.is_file() else None
        rows[key] = {"path": str(path), "expected": expected, "actual": actual, "match": actual == expected}
    return rows


def extract_code_archive(repo, destination):
    paths = ["robolab/robolab", "robolab/experiments", "robolab/scripts", "rsl_rl"]
    data = subprocess.check_output(["git", "-C", str(repo), "archive", "HEAD", *paths])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe path in code archive")
            path = destination.joinpath(*relative.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, path.open("xb") as target:
                    target.write(source.read())
            else:
                raise ValueError(f"Unsupported link in code archive: {member.name}")


def ensure_source_snapshot(code_root, identity, cache_root, bundle_root, repo_root):
    """Recover only three known orchestration files; reject any other mismatch.

    All checkpoint-identified files are copied with their verified original
    bytes. Assets remain in their existing locations, linked read-only by use.
    Neither the working repository nor checkpoints are overwritten.
    """
    code_root, repo_root = Path(code_root).resolve(), Path(repo_root).resolve()
    rows = audit(code_root, identity)
    mismatches = {key: value for key, value in rows.items() if not value["match"]}
    if not mismatches:
        return code_root
    unknown = sorted(set(mismatches) - set(EXPECTED_REPAIRS))
    if unknown:
        raise ValueError("Unrecognized checkpoint source changes; no snapshot built: " + ", ".join(unknown))
    if code_root != repo_root / "robolab":
        raise ValueError("Automatic recovery requires this repository's robolab directory")
    replacements = {}
    for key, row in mismatches.items():
        data = (Path(bundle_root) / key.split(":", 1)[1]).read_bytes()
        if row["expected"] != EXPECTED_REPAIRS[key] or sha(data) != row["expected"]:
            raise ValueError(f"Bundled source does not match checkpoint: {key}")
        replacements[key] = data
    fingerprint = sha(json.dumps({"root": str(code_root), "sources": expected_files(identity)},
                                 sort_keys=True).encode())[:24]
    destination = Path(cache_root) / f"checkpoint_source_{fingerprint}"
    target_root = destination / "robolab"
    manifest = destination / "source_snapshot_audit.json"
    if destination.exists():
        if not manifest.is_file():
            raise ValueError(f"Incomplete source snapshot retained; inspect before recovery: {destination}")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("original_code_root") != str(code_root):
            raise ValueError("Existing source snapshot belongs to a different source tree")
        verified = audit(target_root, identity)
        if any(not row["match"] for row in verified.values()):
            raise ValueError("Cached source snapshot changed; no training started")
        print(f"SOURCE_SNAPSHOT: {len(verified)}/{len(verified)} verified; reusing {target_root}", flush=True)
        return target_root
    destination.mkdir(parents=True, exist_ok=False)
    extract_code_archive(repo_root, destination)
    for key, row in rows.items():
        path = source_path(target_root, key)
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"Snapshot destination escapes its root: {key}")
        data = replacements[key] if key in replacements else Path(row["path"]).read_bytes()
        if sha(data) != row["expected"]:
            raise ValueError(f"Source changed while building snapshot: {key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for relative in ("robolab/data", "robolab/logs", "vendor"):
        original, link = repo_root / relative, destination / relative
        if original.exists() and not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(original.resolve(), target_is_directory=original.is_dir())
    verified = audit(target_root, identity)
    if any(not row["match"] for row in verified.values()):
        raise ValueError(f"Recovered source audit failed; retained evidence at {destination}")
    manifest.write_text(json.dumps({
        "result": "CHECKPOINT_SOURCE_SNAPSHOT_VERIFIED", "original_code_root": str(code_root),
        "code_root": str(target_root), "restored_files": sorted(replacements),
        "matched_sources": len(verified), "source_mismatches": 0, "files": verified,
    }, indent=2), encoding="utf-8")
    print(f"SOURCE_SNAPSHOT: restored {', '.join(sorted(replacements))}; "
          f"{len(verified)}/{len(verified)} verified; {target_root}", flush=True)
    return target_root


def prepare_checkpoint_source(checkpoint, code_root, repo_root):
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return ensure_source_snapshot(
        code_root, saved.get("hybrid_identity", {}),
        Path(repo_root) / "robolab/logs/rough_bo/source_snapshots",
        Path(__file__).resolve().parent / "checkpoint_sources", repo_root,
    )
