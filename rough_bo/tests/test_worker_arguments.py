"""Check the actual BO child commands against both worker argument parsers."""
import argparse
import ast
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

BO = Path(__file__).resolve().parents[1]


def load_function(filename, name, namespace):
    namespace.setdefault("Any", object)
    tree = ast.parse((BO / filename).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(BO / filename), "exec"), namespace)
    return namespace[name]


class WorkerArgumentTests(unittest.TestCase):
    def test_actual_train_and_evaluation_commands(self):
        try:
            from isaaclab.app import AppLauncher
            app_module = sys.modules["isaaclab.app"]
        except ImportError:
            class AppLauncher:
                @staticmethod
                def add_app_launcher_args(parser):
                    parser.add_argument("--headless", action="store_true")
                    parser.add_argument("--device", default="cuda:0")
                    parser.add_argument("--kit_args", default="")
            app_module = types.ModuleType("isaaclab.app")
            app_module.AppLauncher = AppLauncher
        kit = "--/renderer/activeGpu=5 --/physics/cudaDevice=5 --/renderer/multiGpu/enabled=false"
        parsed = []
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp:
            root = Path(tmp)
            attempt = root / "attempt"
            attempt.mkdir()
            request = dict(thigh_m=0.3, calf_m=0.4, knee_motor="K", ankle_motor="A")
            provenance = dict(checkpoint="base.pt", bank="bank.json", checkpoint_sha256="basehash", bank_sha256="bankhash")
            args = argparse.Namespace(python=sys.executable, code_root=root / "robolab", kit_args=kit,
                device="cuda:5", num_envs=4096, seed=42, eval_seed=42, learning_rate=None,
                allow_source_matched_asset_hash_override=True, fine_tune_timeout=10, eval_timeout=10)
            def atomic(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            def run_child(command, **kwargs):
                filename = Path(command[2]).name
                namespace = dict(argparse=argparse, DEFAULT_NUM_ENVS=4096,
                                 DEFAULT_FINETUNE_UPDATES=200, EVAL_SEED=42, __doc__="Parser test")
                parser = load_function(filename, "parse_args", namespace)
                with patch.dict(sys.modules, {"isaaclab.app": app_module}), patch.object(sys, "argv", command[2:]):
                    value = parser()
                parsed.append(value)
                self.assertEqual(value.kit_args, kit)
                self.assertEqual(value.device, "cuda:5")
                self.assertTrue(value.headless)
                self.assertNotIn("--kit-args", command)
                self.assertIn("--kit_args=" + kit, command)
                if filename == "bo_finetune_candidate.py":
                    atomic(attempt / "finetune/finetune_complete.json", {"fine_tune_updates": 200, "checkpoint": str(root / "model_199.pt")})
                else:
                    atomic(attempt / "evaluation/summary.json", {
                        "result": "ROUGH_CANDIDATE_200_UPDATE_SPEED_TRACKING_EVALUATED",
                        "trial_number": 0, "base_checkpoint_sha256": "basehash", "bank_sha256": "bankhash",
                        "fine_tune_updates": 200, "objective_value": 0.1})
            namespace = dict(Path=Path, os=os, json=json, math=math, HERE=BO,
                DEFAULT_FINETUNE_UPDATES=200, next_attempt_dir=lambda *a: attempt,
                atomic_json=atomic, run_child=run_child)
            run = load_function("bo_finetune_controller.py", "run_candidate", namespace)
            result = run(args, root, provenance, request, 0, "ocd50")
            self.assertEqual(len(parsed), 2)
            self.assertEqual(result["objective_value"], 0.1)


if __name__ == "__main__":
    unittest.main()
