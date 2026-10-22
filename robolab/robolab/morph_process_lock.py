"""Exclusive owned controller lock. Preserve stale lock records, never kill jobs."""
import atexit
import json
import os
from pathlib import Path
import time
import psutil
from robolab.morph_experiments import write_new


def claim_controller(root):
    path = Path(root).resolve() / "controller.lock.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf8"))
        if old["status"] == "active" and psutil.pid_exists(old["pid"]):
            process = psutil.Process(old["pid"])
            if process.create_time() == old["created"]:
                raise RuntimeError("owned curriculum controller still active; no duplicate")
        path.rename(path.with_name(f"controller_{time.time_ns()}.previous.json"))
    info = {"pid": os.getpid(), "created": psutil.Process().create_time(), "status": "active"}
    write_new(path, info)  # exclusive creation closes same-root duplicate race
    def release():
        current = json.loads(path.read_text(encoding="utf8"))
        if current == info:
            current["status"] = "exited"
            pending = path.with_name(f"controller_{time.time_ns()}.pending")
            write_new(pending, current)
            os.replace(pending, path)
    atexit.register(release)
    return release
