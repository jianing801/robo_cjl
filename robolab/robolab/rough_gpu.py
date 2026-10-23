"""Shared-server GPU selection; never kill or inspect unrelated Python jobs."""
from contextlib import contextmanager
import csv
import os
from pathlib import Path
import subprocess


def query_gpus():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Cannot query GPUs with nvidia-smi; refusing to guess a device") from error
    rows = []
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 5:
            raise ValueError("unexpected nvidia-smi GPU inventory")
        try:
            index, uuid, total, free, util = [x.strip() for x in fields]
            row = dict(index=int(index), uuid=uuid, total_mib=int(total), free_mib=int(free), utilization=int(util))
        except ValueError:
            continue  # N/A metrics are not evidence that a GPU is idle.
        if not uuid.startswith("GPU-") or not 0 <= row["free_mib"] <= row["total_mib"] or not 0 <= row["utilization"] <= 100:
            raise ValueError("invalid GPU inventory metrics")
        rows.append(row)
    return rows


def visible_gpus(rows, environment):
    result = sorted(rows, key=lambda r: r["index"])
    for key in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if key not in environment or (key == "NVIDIA_VISIBLE_DEVICES" and environment[key] == "all"):
            continue
        tokens = [x.strip() for x in environment[key].split(",") if x.strip()]
        if not tokens or tokens == ["-1"] or tokens == ["void"] or tokens == ["none"]:
            return []
        selected = []
        for token in tokens:
            found = [r for r in result if str(r["index"]) == token or (token.startswith("GPU-") and r["uuid"].startswith(token))]
            if len(found) != 1:
                raise ValueError(f"Cannot unambiguously map {key}={token}; MIG/unknown GPUs are not supported")
            if found[0] in selected:
                raise ValueError("duplicate visible GPU")
            selected.append(found[0])
        result = selected
    return result


def candidates(rows, environment, *, min_free_mib, max_utilization, requested=None):
    if min_free_mib <= 0 or not 0 <= max_utilization <= 100:
        raise ValueError("positive GPU free-memory threshold and utilization 0..100 required")
    visible = visible_gpus(rows, environment)
    if requested and requested != "auto":
        if requested == "cuda":
            requested = "cuda:0"
        if not requested.startswith("cuda:") or not requested[5:].isdigit():
            raise ValueError("Rough requires auto or cuda:N; N is relative to the current visible GPU list")
        index = int(requested[5:])
        visible = visible[index:index+1]
    eligible = [r for r in visible if r["free_mib"] >= min_free_mib and r["utilization"] <= max_utilization]
    return sorted(eligible, key=lambda r: (-r["free_mib"], r["utilization"], r["index"]))


@contextmanager
def gpu_lease(rows, root):
    """Cooperative OS lock: auto-released on exit/crash; does not reserve others' GPUs."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = root/(row["uuid"]+".lock")
        stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    stream.write(b"0"); stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            continue
        try:
            yield row
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        return
    raise RuntimeError("No GPU meets free-memory/utilization thresholds or all eligible GPUs are leased by another Rough launcher; no job was killed")


def child_environment(gpu, environment):
    return dict(environment, CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")


def reject_duplicate_bank(bank, *, processes=None, own_tree=None):
    """Reject only another worker explicitly using THIS bank; unrelated jobs are fine."""
    import psutil
    own_tree = own_tree if own_tree is not None else {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    processes = processes if processes is not None else psutil.process_iter(["cmdline", "cwd"])
    bank = Path(bank).resolve()
    for process in processes:
        if process.pid in own_tree:
            continue
        try:
            cmd = process.info.get("cmdline") or []
            if not any(Path(x).name in ("train_spatial_rough.py", "train_spatial_expert.py") for x in cmd):
                continue
            for i, token in enumerate(cmd):
                value = token.split("=", 1)[1] if token.startswith("--bank=") else cmd[i+1] if token == "--bank" and i+1 < len(cmd) else None
                if value is not None:
                    path = Path(value)
                    if not path.is_absolute():
                        cwd = process.info.get("cwd")
                        if not cwd:
                            continue
                        path = Path(cwd)/path
                    if path.resolve() == bank:
                        raise RuntimeError(f"same Rough bank already active in PID {process.pid}; no duplicate launch")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
