"""Verify that a frozen Rough checkpoint is evaluated by its recorded source tree."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source_files(code_root: Path, identity: dict) -> dict:
    """Hash every checkpoint-identified Python/config source under ``code_root``.

    The checkpoint stores native paths relative to ``robolab`` (with rsl_rl as a
    sibling) and hybrid paths relative to ``experiments/rough_hybrid``.
    """
    code_root = code_root.resolve()
    expected_groups = {
        "native": identity.get("native_sources", {}),
        "hybrid": identity.get("hybrid_sources", {}),
    }
    records = {}
    missing_groups = [name for name, files in expected_groups.items() if not files]
    if missing_groups:
        raise ValueError(f"checkpoint is missing source identity groups: {missing_groups}")

    for group, files in expected_groups.items():
        for relative, expected in sorted(files.items()):
            base = code_root if group == "native" else code_root / "experiments" / "rough_hybrid"
            path = (base / Path(relative)).resolve()
            exists = path.is_file()
            actual = _sha256(path) if exists else None
            records[f"{group}:{relative}"] = {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": bool(exists and actual == expected),
            }

    mismatches = [key for key, value in records.items() if not value["match"]]
    return {
        "code_root": str(code_root),
        "native_source_count": len(expected_groups["native"]),
        "hybrid_source_count": len(expected_groups["hybrid"]),
        "matched_source_count": len(records) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "files": records,
    }


def audit_loaded_modules(code_root: Path, identity: dict, module_names: list[str]) -> dict:
    """Ensure imported critical runtime modules came from the audited files."""
    code_root = code_root.resolve()
    expected_by_path = {}
    for relative, digest in identity.get("native_sources", {}).items():
        expected_by_path[(code_root / Path(relative)).resolve()] = ("native", relative, digest)
    hybrid_root = code_root / "experiments" / "rough_hybrid"
    for relative, digest in identity.get("hybrid_sources", {}).items():
        expected_by_path[(hybrid_root / relative).resolve()] = ("hybrid", relative, digest)

    records = []
    for name in module_names:
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(f"critical runtime module was not imported: {name}")
        path = Path(module.__file__).resolve()
        expected = expected_by_path.get(path)
        actual = _sha256(path) if path.is_file() else None
        records.append({
            "module": name,
            "path": str(path),
            "source_group": expected[0] if expected else None,
            "source_key": expected[1] if expected else None,
            "expected_sha256": expected[2] if expected else None,
            "actual_sha256": actual,
            "match": bool(expected and actual == expected[2]),
        })
    mismatches = [row["module"] for row in records if not row["match"]]
    return {"modules": records, "mismatch_count": len(mismatches), "mismatches": mismatches}

