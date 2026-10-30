"""Source recovery tests use a tiny real Git repository and no simulator."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

BO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BO))
import source_snapshot as snapshot


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=BO / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.code = self.repo / "robolab"
        self.identity = {"native_sources": {}, "hybrid_sources": {}}
        for relative, data in [("robolab/core.py", b"value = 1\n"),
                               ("../rsl_rl/rsl_rl/core.py", b"value = 2\n")]:
            path = self.code / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.identity["native_sources"][relative] = snapshot.sha(data)
        scripts = self.code / "scripts/tools"
        scripts.mkdir(parents=True)
        (scripts / "audit_helper.py").write_text("# supporting tool\n")
        for key, expected in snapshot.EXPECTED_REPAIRS.items():
            name = key.split(":", 1)[1]
            data = (BO / "checkpoint_sources" / name).read_bytes()
            self.assertEqual(snapshot.sha(data), expected)
            path = self.code / "experiments/rough_hybrid" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# current improved controller\n")
            self.identity["hybrid_sources"][name] = expected
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=BO Test", "-c", "user.email=bo@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.STDOUT)

    def recover(self):
        with redirect_stdout(io.StringIO()):
            return snapshot.ensure_source_snapshot(self.code, self.identity, self.base / "cache",
                                                   BO / "checkpoint_sources", self.repo)

    def test_recovery_restores_exact_bytes_and_preserves_repo(self):
        before = {str(p): p.read_bytes() for p in self.repo.rglob("*.py")}
        target = self.recover()
        self.assertNotEqual(target, self.code)
        self.assertTrue(all(row["match"] for row in snapshot.audit(target, self.identity).values()))
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.repo.rglob("*.py")})
        manifest = json.loads((target.parent / "source_snapshot_audit.json").read_text())
        self.assertEqual(manifest["source_mismatches"], 0)
        self.assertEqual(len(manifest["restored_files"]), 3)
        self.assertEqual(target, self.recover())

    def test_unknown_runtime_change_is_rejected_before_copy(self):
        (self.code / "robolab/core.py").write_text("changed physics\n")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            self.recover()
        self.assertFalse((self.base / "cache").exists())

    def test_tampered_bundle_is_rejected(self):
        with patch.object(snapshot, "EXPECTED_REPAIRS", {**snapshot.EXPECTED_REPAIRS,
                         "hybrid:train.py": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "Bundled source"):
                self.recover()

    def test_tampered_cached_snapshot_is_rejected(self):
        target = self.recover()
        (target / "experiments/rough_hybrid/train.py").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "Cached source snapshot changed"):
            self.recover()

    def test_matching_source_returns_original(self):
        for key in snapshot.EXPECTED_REPAIRS:
            path = snapshot.source_path(self.code, key)
            path.write_bytes((BO / "checkpoint_sources" / path.name).read_bytes())
        self.assertEqual(self.recover(), self.code)
        self.assertFalse((self.base / "cache").exists())

    def test_escaping_identity_is_rejected(self):
        self.identity["native_sources"]["../../outside.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.recover()


if __name__ == "__main__":
    unittest.main()
