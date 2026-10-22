"""Dry-run by default; execute a finite serial two-expert morphology curriculum.

No policy per design: one continuing full checkpoint per TASK. Interrupted
blocks are retained; --resume continues from the last completed block only.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "rsl_rl")]


def main():
    from robolab.morph_experiments import load_bank, write_new, execution_budget
    from robolab.mxt_design_transfer import file_sha
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--task", choices=("run", "stairs", "both"), default="both")
    parser.add_argument("--output-root", required=True, help="New root, or same root with explicit --resume")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kit-args", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    bank_path, output = Path(args.bank).resolve(), Path(args.output_root).resolve()
    bank, tasks = load_bank(bank_path), ("run", "stairs") if args.task == "both" else (args.task,)
    from robolab.native_expert_contract import validate_native_bank
    validate_native_bank(bank)
    if any(t not in bank["training"].get("authorized_tasks", ("run", "stairs")) for t in tasks):
        raise ValueError("task is not authorized by this experiment; RUN-only means never launch STAIRS")
    limits, schedules, budget_identity = execution_budget(bank, bank_path)
    if not args.execute:
        print(json.dumps({"mode": "PLAN_ONLY_NO_LAUNCH", "bank_sha256": file_sha(bank_path),
            "tasks": {t: {"blocks": len(schedules[t]), "updates": limits[t],
                          "training_designs": [r["id"] for r in bank["designs"] if r["split"] == "train"]} for t in tasks},
            "no_holdout_training": True}, indent=2))
        return
    if output.exists() and not args.resume:
        raise FileExistsError("new curriculum root required, or explicitly resume its same identity")
    output.mkdir(parents=True, exist_ok=True)
    from robolab.morph_process_lock import claim_controller
    release = claim_controller(output)
    for task in tasks:
        task_root = output / task
        task_root.mkdir(exist_ok=True)
        state_path = task_root / "state.json"
        state = json.loads(state_path.read_text(encoding="utf8")) if state_path.exists() else {
            "bank_sha256": file_sha(bank_path), "task": task, "completed_blocks": 0, "completed_updates": 0,
            "checkpoint": None, "checkpoint_sha256": None, "status": "ready"}
        if state["bank_sha256"] != file_sha(bank_path) or state["task"] != task:
            raise ValueError("cannot resume a different bank or expert")
        if state["checkpoint"] and file_sha(state["checkpoint"]) != state["checkpoint_sha256"]:
            raise ValueError("last completed checkpoint changed; do not restart from scratch")
        if (state["completed_updates"] > limits[task] or state["completed_blocks"] > len(schedules[task])
                or state["completed_updates"] != sum(b["updates"] for b in schedules[task][:state["completed_blocks"]])):
            raise ValueError("saved progress exceeds or disagrees with revised stop budget")
        state.update(execution_budget=budget_identity, target_updates=limits[task], target_blocks=len(schedules[task]))
        for block in schedules[task][state["completed_blocks"]:]:
            if execution_budget(bank, bank_path)[2] != budget_identity:
                raise ValueError("stop budget changed while controller was running")
            # Catch a still-active owned worker before retrying an interrupted
            # controller. CPU dry-run never inspects/kills unrelated processes.
            import psutil
            for process in psutil.process_iter(["pid", "cmdline"]):
                cmd = process.info["cmdline"] or []
                if any(Path(a).name == "train_multimorph_expert_block.py" for a in cmd) and str(bank_path) in cmd:
                    raise RuntimeError("existing bank worker active; refusing duplicate launch")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run = task_root / f"b{block['block']:04}_{stamp}"
            command = [args.python, "-u", str(Path(__file__).with_name("train_multimorph_expert_block.py")),
                "--bank", str(bank_path), "--task", task, "--block", str(block["block"]), "--output-dir", str(run),
                "--execute", "--device", args.device, "--headless", "--rendering_mode", "performance", "--kit_args=" + args.kit_args]
            if state["checkpoint"]:
                command += ["--resume-checkpoint", state["checkpoint"]]
            write_new(task_root / f"launch_{stamp}.json", {"argv": command, "from_state": state})
            process_started = time.monotonic()
            with (task_root / f"launch_{stamp}.stdout.log").open("x") as out, (task_root / f"launch_{stamp}.stderr.log").open("x") as err:
                env = dict(os.environ, WARP_CACHE_PATH=str(ROOT / "logs/rsl_rl/isaac_cache/warp_training"))
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                returncode = process.wait()
                write_new(task_root / f"attempt_{stamp}.json", {"block": block, "returncode": returncode,
                    "process_total_wall_seconds": time.monotonic()-process_started,
                    "includes_Isaac_startup_and_shutdown": True, "run_directory": str(run),
                    "failed_attempt_cost_must_not_be_omitted": True})
                if returncode != 0:
                    raise RuntimeError(f"block failed; retain all files, no automatic reset: {run}")
            result = json.loads((run / "summary.json").read_text(encoding="utf8"))
            if (result["result"] != "EXPERT_BLOCK_COMPLETE_NOT_GAIT_VALIDATED" or result["block"] != block
                    or result["bank_sha256"] != state["bank_sha256"] or result["task"] != task
                    or result.get("execution_budget") != budget_identity
                    or file_sha(result["final_checkpoint"]) != result["final_sha256"]):
                raise ValueError("worker result identity or checkpoint mismatch")
            state.update(completed_blocks=block["block"] + 1, completed_updates=result["completed_updates"],
                         checkpoint=result["final_checkpoint"], checkpoint_sha256=result["final_sha256"], status="training")
            pending = task_root / f"state_{stamp}.pending"
            write_new(pending, state)
            os.replace(pending, state_path)  # only this controller's live state is mutable
        state["status"] = "training_complete_requires_all_design_gait_evaluation"
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        write_new(task_root / f"complete_{stamp}.json", state)
        pending = task_root / f"state_{stamp}.pending"
        write_new(pending, state)
        os.replace(pending, state_path)
    release()


if __name__ == "__main__":
    main()
