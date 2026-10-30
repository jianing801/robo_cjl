"""Launch Rough BO in the foreground; omit --gpu to choose an available GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime


BO_WORKERS = {"bo_finetune_controller.py", "bo_finetune_candidate.py", "bo_finetune_evaluate.py"}


def query_gpus():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, encoding="utf-8")
    return [{"index": int(row[0]), "free_mib": int(row[1]), "utilization": int(row[2])}
            for row in csv.reader(output.splitlines(), skipinitialspace=True) if row]


def choose_gpu(gpus, requested=None, min_free_mib=12800, max_utilization=20):
    if requested is not None:
        candidates = [g for g in gpus if g["index"] == requested]
        if not candidates:
            raise RuntimeError(f"GPU {requested} is not visible to nvidia-smi")
        if candidates[0]["free_mib"] < min_free_mib:
            raise RuntimeError(f"GPU {requested} has {candidates[0]['free_mib']} MiB free; need {min_free_mib}")
        return candidates[0]
    candidates = [g for g in gpus if g["free_mib"] >= min_free_mib
                  and g["utilization"] <= max_utilization]
    if not candidates:
        raise RuntimeError(f"No GPU has >= {min_free_mib} MiB free and utilization <= {max_utilization}%")
    return max(candidates, key=lambda g: (g["free_mib"], -g["utilization"], -g["index"]))


def is_bo_command(argv):
    return any(Path(arg).name in BO_WORKERS for arg in argv)


def reject_running_bo():
    if sys.platform != "linux":
        return
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
            if is_bo_command(argv):
                raise RuntimeError(f"An existing BO process is still alive: PID {entry.name}. Nothing launched.")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


@contextmanager
def launch_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt
            if path.stat().st_size == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another BO launcher is already running") from exc
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another BO launcher is already running") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def build_command(args, root, gpu, output):
    index = gpu["index"]
    return [sys.executable, "-u", str(root / "rough_bo/bo_finetune_controller.py"),
            "--python", sys.executable, "--checkpoint", str(args.checkpoint),
            "--bank", str(args.bank), "--code-root", str(args.code_root),
            "--output", str(output), "--device", f"cuda:{index}",
            "--kit-args", f"--/renderer/activeGpu={index} --/physics/cudaDevice={index} --/renderer/multiGpu/enabled=false",
            "--allow-source-matched-asset-hash-override", "--bo-trials", str(args.bo_trials),
            *(["--resume"] if args.resume else [])]


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="Physical nvidia-smi GPU index; default: automatically choose")
    parser.add_argument("--min-free-gb", type=float, default=12.5)
    parser.add_argument("--max-utilization", type=int, default=20, help="Automatic selection utilization limit")
    parser.add_argument("--bo-trials", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume this existing BO directory")
    parser.add_argument("--checkpoint", type=Path, default=root / "robolab/logs/rough_hybrid/persistent_20261026_194622/attempts/group_000141_20261027_235200_111604/result/models/model_11999.pt")
    parser.add_argument("--bank", type=Path, default=root / "robolab/logs/rough_hybrid/dynamic_20261026_174456/bank/bank.json")
    parser.add_argument("--code-root", type=Path, default=root / "robolab")
    parser.add_argument("--check-only", action="store_true", help="Check inputs and GPU, print command, do not train")
    args = parser.parse_args(argv)
    if args.output and args.resume:
        parser.error("use either --output or --resume")
    if (args.min_free_gb < 12.5 or not 0 <= args.max_utilization <= 100 or args.bo_trials < 0
            or (args.gpu is not None and args.gpu < 0)):
        parser.error("require min-free-gb >= 12.5, max-utilization 0..100 and bo-trials >= 0")
    required = [args.checkpoint, args.bank, args.code_root / "scripts/tools/pretrained_rough_source_audit.py",
                *[root / "rough_bo" / filename for filename in (
                    "bo_finetune_controller.py", "bo_finetune_candidate.py",
                    "bo_finetune_evaluate.py", "bo_finetune_common.py")],
                root / "rough_bo/ocd_mixed_design_50_full_cooling/ocd_design_50.csv"]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required file is missing or empty: {path}")
    output = (args.resume or args.output or root / "robolab/logs/rough_bo" /
              f"bo_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    if args.resume:
        if not (output / "run_contract.json").is_file():
            raise RuntimeError(f"Resume directory has no run_contract.json: {output}")
    elif output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    with launch_lock(root / "robolab/logs/rough_bo/.foreground_launcher.lock"):
        reject_running_bo()
        gpu = choose_gpu(query_gpus(), args.gpu, int(args.min_free_gb * 1024), args.max_utilization)
        command = build_command(args, root, gpu, output)
        print(f"GPU {gpu['index']}: {gpu['free_mib']} MiB free, utilization {gpu['utilization']}%", flush=True)
        print(f"BO: 50 initial designs + {args.bo_trials} GP candidates; each fine-tuned for 200 updates", flush=True)
        print(f"OUTPUT={output}", flush=True)
        if args.check_only:
            print(json.dumps(command, ensure_ascii=False), flush=True)
            return 0
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PYTHONUTF8"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        process = subprocess.Popen(command, cwd=root, env=env)
        try:
            return process.wait()
        except KeyboardInterrupt:
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            return 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, FileExistsError, subprocess.SubprocessError) as exc:
        print(f"BO was not started or failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
