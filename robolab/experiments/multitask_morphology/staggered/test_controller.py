import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from controller import native_controller_path, phase_identity, require_staggered_plan, worker_command


class TrialIsolationTests(unittest.TestCase):
    def test_keeps_native_worker_visible_to_duplicate_bank_guard(self):
        for task in ("rough", "backpack", "amp"):
            filename = "amp_persistent_worker.py" if task == "amp" else "persistent_worker.py"
            original = str(native_controller_path(task).with_name(filename))
            args = ["python", "-u", original, "--bank", "existing/bank.json", "--root", "new/run"]
            updated = worker_command(args, task)
            self.assertEqual(Path(updated[2]).name, "staggered_worker.py")
            self.assertEqual(updated[:2] + updated[3:], args)

    def test_accepts_only_matching_frozen_staggered_plan(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            plan = output / "plan.json"
            base = {"execution_protocol": "persistent_v1"}
            plan.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "没有错开回合"):
                require_staggered_plan(output, "rough")
            base["episode_phase"] = phase_identity("rough")
            plan.write_text(json.dumps(base), encoding="utf-8")
            self.assertEqual(require_staggered_plan(output, "rough"), base)
            with self.assertRaisesRegex(ValueError, "不一致"):
                require_staggered_plan(output, "backpack")
            base["episode_phase"]["sources"]["episode_phases.py"] = "changed"
            plan.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不一致"):
                require_staggered_plan(output, "rough")


if __name__ == "__main__":
    unittest.main()
