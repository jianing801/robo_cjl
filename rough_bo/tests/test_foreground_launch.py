"""CPU-only launch/logging checks; never start Isaac or GPU training."""
import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

BO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BO))
import start


def child_runner():
    tree = ast.parse((BO / "bo_finetune_controller.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_child")
    scope = dict(Path=Path, subprocess=subprocess, sys=sys, threading=threading, time=time,
                 atomic_json=lambda p, v: p.write_text(json.dumps(v), encoding="utf-8"),
                 tail=lambda p: p.read_text(encoding="utf-8"))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(BO / "bo_finetune_controller.py"), "exec"), scope)
    return scope["run_child"]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.gpus = [dict(index=0, free_mib=21000, utilization=95),
                     dict(index=2, free_mib=18000, utilization=10),
                     dict(index=5, free_mib=20000, utilization=5)]

    def test_auto_avoids_busy_card_and_chooses_free_memory(self):
        self.assertEqual(start.choose_gpu(self.gpus)["index"], 5)

    def test_explicit_card_and_insufficient_memory(self):
        self.assertEqual(start.choose_gpu(self.gpus, requested=0)["index"], 0)
        with self.assertRaises(RuntimeError):
            start.choose_gpu([dict(index=5, free_mib=12000, utilization=0)], requested=5)
        with self.assertRaises(RuntimeError):
            start.choose_gpu(self.gpus, requested=4)

    def test_no_available_card(self):
        with self.assertRaises(RuntimeError):
            start.choose_gpu([dict(index=0, free_mib=5000, utilization=0)])
        with self.assertRaises(RuntimeError):
            start.choose_gpu([dict(index=0, free_mib=24000, utilization=90)])

    def test_query_csv(self):
        with patch.object(start.subprocess, "check_output", return_value="0, 22000, 5\n5, 13000, 70\n"):
            self.assertEqual(start.query_gpus()[1], dict(index=5, free_mib=13000, utilization=70))

    def test_physical_gpu_indices_match(self):
        args = Mock(checkpoint=Path("checkpoint.pt"), bank=Path("bank.json"),
                    code_root=Path("code"), bo_trials=50, resume=None)
        command = start.build_command(args, Path("repo"), self.gpus[2], Path("output"))
        self.assertEqual(command[command.index("--device") + 1], "cuda:5")
        kit = command[command.index("--kit-args") + 1]
        self.assertIn("activeGpu=5", kit)
        self.assertIn("cudaDevice=5", kit)
        self.assertEqual(command[command.index("--bo-trials") + 1], "50")

    def test_duplicate_process_detection(self):
        self.assertTrue(start.is_bo_command(["python", "/repo/rough_bo/bo_finetune_candidate.py"]))
        self.assertFalse(start.is_bo_command(["python", "/repo/amp/train.py"]))

    def test_lock_released_after_failure(self):
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp:
            path = Path(tmp) / "launch.lock"
            with self.assertRaises(ValueError):
                with start.launch_lock(path):
                    raise ValueError("simulated launch error")
            with start.launch_lock(path):
                with self.assertRaises(RuntimeError):
                    with start.launch_lock(path):
                        pass

    def test_missing_file_does_not_launch(self):
        with patch.object(start.subprocess, "Popen") as launch:
            with self.assertRaises(FileNotFoundError):
                start.main(["--checkpoint", "/definitely_missing/bo_checkpoint.pt"])
            launch.assert_not_called()

    def test_real_stdout_stderr_are_live_and_saved(self):
        run = child_runner()
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp:
            root = Path(tmp)
            output, errors = io.StringIO(), io.StringIO()
            exception = []
            def execute():
                try:
                    run([sys.executable, "-u", "-c", "import time,sys; print('Learning iteration 1/200',flush=True); print('diagnostic',file=sys.stderr,flush=True); time.sleep(1.5); print('finished',flush=True)"],
                        cwd=root, env=dict(__import__('os').environ), stdout_path=root / "stdout.log",
                        stderr_path=root / "stderr.log", timeout=10, label="CPU output test")
                except BaseException as exc:
                    exception.append(exc)
            with redirect_stdout(output), redirect_stderr(errors):
                thread = threading.Thread(target=execute)
                thread.start()
                deadline = time.monotonic() + 5
                while "Learning iteration 1/200" not in output.getvalue() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertIn("Learning iteration 1/200", output.getvalue())
                self.assertTrue(thread.is_alive(), "output must be visible before child completion")
                thread.join(timeout=10)
            self.assertFalse(exception)
            self.assertIn("finished", (root / "stdout.log").read_text())
            self.assertIn("diagnostic", (root / "stderr.log").read_text())
            self.assertIn("diagnostic", errors.getvalue())

    def test_child_error_is_not_masked(self):
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "rc=7"):
                child_runner()([sys.executable, "-c", "import sys; print('failure detail',file=sys.stderr); sys.exit(7)"],
                    cwd=root, env=dict(__import__('os').environ), stdout_path=root / "out",
                    stderr_path=root / "err", timeout=10, label="CPU error test")

    def test_child_timeout_reaps_process(self):
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                child_runner()([sys.executable, "-c", "import time; time.sleep(60)"],
                    cwd=root, env=dict(__import__('os').environ), stdout_path=root / "out",
                    stderr_path=root / "err", timeout=0.2, label="CPU timeout test")
            self.assertTrue((root / "err.timeout.json").is_file())

    def fake_repo(self, root):
        relative = ["rough_bo/bo_finetune_controller.py", "rough_bo/bo_finetune_candidate.py",
                    "rough_bo/bo_finetune_evaluate.py", "rough_bo/bo_finetune_common.py",
                    "rough_bo/ocd_mixed_design_50_full_cooling/ocd_design_50.csv",
                    "robolab/scripts/tools/pretrained_rough_source_audit.py", "checkpoint.pt", "bank.json"]
        for name in relative:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        return ["--checkpoint", str(root / "checkpoint.pt"), "--bank", str(root / "bank.json")]

    def test_check_only_validates_without_launch(self):
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp:
            root = Path(tmp)
            argv = self.fake_repo(root)
            with patch.object(start, "__file__", str(root / "rough_bo/start.py")), \
                 patch.object(start, "prepare_checkpoint_source", return_value=root / "robolab"), \
                 patch.object(start, "query_gpus", return_value=self.gpus), \
                 patch.object(start, "reject_running_bo"), \
                 patch.object(start.subprocess, "Popen") as process, redirect_stdout(io.StringIO()):
                self.assertEqual(start.main(argv + ["--check-only"]), 0)
                process.assert_not_called()

    def test_foreground_main_passes_gpu_and_unmasks_child_only(self):
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp:
            root = Path(tmp)
            argv = self.fake_repo(root)
            with patch.object(start, "__file__", str(root / "rough_bo/start.py")), \
                 patch.object(start, "prepare_checkpoint_source", return_value=root / "robolab"), \
                 patch.object(start, "query_gpus", return_value=self.gpus), \
                 patch.object(start, "reject_running_bo"), \
                 patch.dict(start.os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
                 patch.object(start.subprocess, "Popen") as process, redirect_stdout(io.StringIO()):
                process.return_value.wait.return_value = 0
                self.assertEqual(start.main(argv + ["--gpu", "2"]), 0)
                command = process.call_args.args[0]
                self.assertEqual(command[command.index("--device") + 1], "cuda:2")
                self.assertNotIn("CUDA_VISIBLE_DEVICES", process.call_args.kwargs["env"])
                self.assertEqual(start.os.environ["CUDA_VISIBLE_DEVICES"], "2")
                self.assertNotIn("stdout", process.call_args.kwargs)
                self.assertNotIn("stderr", process.call_args.kwargs)
                process.return_value.wait.assert_called_once()

    def test_interrupt_waits_for_worker_exit(self):
        with tempfile.TemporaryDirectory(dir=BO / "tests") as tmp:
            root = Path(tmp)
            process = Mock(stdout=io.StringIO("iteration\n"), stderr=io.StringIO(""))
            process.wait.side_effect = [KeyboardInterrupt(), 0]
            with patch.object(subprocess, "Popen", return_value=process), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    child_runner()(["fixture"], cwd=root, env={}, stdout_path=root / "out",
                                   stderr_path=root / "err", timeout=10, label="interrupt")
                self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
