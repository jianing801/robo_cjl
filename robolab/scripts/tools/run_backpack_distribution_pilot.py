"""Prepare or run a fixed-schedule, single-design backpack feasibility pilot.

Preparation is CPU-only. Execution refuses competing Python GPU workflows.
No original backpack chain or expert bank is changed.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robolab.backpack_distribution import (stages, sha, write_new, validate_checkpoint,
                                           validate_plan, acquire_controller_lock)
from robolab.physical_snapshot import validate_physical_model, physical_model, generator


def gpu_conflicts():
    import psutil
    result = []
    ancestors = {p.pid for p in psutil.Process().parents()}
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            if process.pid in ancestors or process.pid == os.getpid() or "python" not in (process.info["name"] or "").lower():
                continue
            command = process.info["cmdline"] or []
            scripts = [Path(arg).name.lower() for arg in command if str(arg).lower().endswith(".py")]
            if any(name.startswith(("train_", "evaluate_", "eval_", "record_", "rollout_"))
                   or name in ("run_multimorph_experts.py", "run_native_expert_pair.py",
                               "run_backpack_distribution_pilot.py") for name in scripts):
                result.append(dict(pid=process.pid, command=command))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            raise RuntimeError("cannot establish GPU workflow ownership")
    return result


def prepare(root, template, mesh_dir):
    import xml.etree.ElementTree as ET
    if not template.is_file() or not mesh_dir.is_dir():
        raise ValueError("provide this machine's original --template and --mesh-dir")
    model = physical_model(knee_motor="DM-J10010L-2EC", ankle_motor="DM-J4340P-2EC")
    root.mkdir(parents=True, exist_ok=False)
    candidate = generator().generate_urdf(.25, .30, root / "candidate", template,
        knee_motor=model["knee_motor"], ankle_motor=model["ankle_motor"], mesh_dir=mesh_dir)
    design = dict(id="nominal_025_030_J10010L_J4340P", thigh_m=.25, calf_m=.30,
        urdf=str(candidate), urdf_sha256=sha(candidate), root_height_compensation_m=0.0)
    mesh_paths = {mesh.get("filename") for mesh in ET.parse(candidate).findall(".//mesh")}
    sources = [ROOT / "robolab/backpack_distribution.py", Path(__file__),
               ROOT / "scripts/tools/train_backpack_distribution_stage.py",
               ROOT / "scripts/tools/ppo_resume_state.py",
               ROOT / "robolab/physical_snapshot.py"]
    sources += list((ROOT / "robolab/tasks/direct/backpack_rpo_flat").glob("*.py"))
    sources += [ROOT / "robolab/tasks/direct/backpack_walk/backpack_urdf.py"]
    sources += list((ROOT / "robolab/tasks/direct/base").rglob("*.py"))
    sources += list((ROOT / "robolab/actuators").rglob("*.py"))
    sources += [ROOT.parent / "rsl_rl/rsl_rl/runners/on_policy_runner.py",
        ROOT.parent / "rsl_rl/rsl_rl/algorithms/ppo.py", ROOT.parent / "rsl_rl/rsl_rl/modules/actor_critic.py"]
    plan = dict(kind="backpack_fixed_distribution_server_v1", num_envs=4096, seed=42,
        design=design, template_sha256=sha(template), physical_model=model,
        mesh_sha256={path: sha(path) for path in sorted(mesh_paths)},
        stages=stages(), max_updates=12000,
        profile="compliant_existing_rewards_unchanged", promotion_gate=False,
        purpose="single-design feasibility, not final multimorph expert or convergence claim",
        mass_sampling="uniform per environment, fixed during each stage; no mid-episode mass jumps",
        reset_between_stages=True, preserve_policy_normalization_optimizer_lr=True,
        source_sha256={str(p.resolve()): sha(p) for p in sources})
    write_new(root / "plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--template", type=Path, help="this machine's unmodified RPO URDF")
    parser.add_argument("--mesh-dir", type=Path, help="this machine's original RPO meshes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kit-args", default="", help="optional platform-specific Kit settings")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf8"))
    else:
        if args.resume:
            raise ValueError("cannot resume a nonexistent plan")
        if args.template is None or args.mesh_dir is None:
            parser.error("new experiments require --template and --mesh-dir")
        plan = prepare(root, args.template.resolve(), args.mesh_dir.resolve())
    validate_plan(plan)
    validate_physical_model(plan["physical_model"])
    print(json.dumps(dict(plan=str(plan_path), stages=plan["stages"], execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    import atexit
    lock = acquire_controller_lock(root)
    atexit.register(lock.close)
    conflicts = gpu_conflicts()
    if conflicts:
        raise RuntimeError(f"GPU workflow already active; leave it running: {conflicts}")
    state_path = root / "state.json"
    if state_path.exists() and not args.resume:
        raise ValueError("existing state requires explicit --resume")
    state = json.loads(state_path.read_text(encoding="utf8")) if state_path.exists() else {}
    start = state.get("completed_stages", 0)
    checkpoint = state.get("checkpoint")
    if checkpoint:
        import torch
        if sha(checkpoint) != state["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False),
                            sha(plan_path), state["completed_updates"])
    # On explicit resume, recover the latest complete periodic checkpoint in
    # the interrupted stage, rather than repeating up to 9000 full-range updates.
    if args.resume and start < len(plan["stages"]):
        import torch
        candidates = []
        for path in root.glob(f"stage{start}_*/models/model_*.pt"):
            suffix = path.stem.removeprefix("model_")
            if suffix.isdigit():
                candidates.append((int(suffix), path))
        for iteration, path in sorted(candidates, key=lambda row: row[0], reverse=True):
            if not plan["stages"][start]["start"] <= iteration < plan["stages"][start]["stop"]:
                continue
            try:
                saved = torch.load(path, map_location="cpu", weights_only=False)
                validate_checkpoint(saved, sha(plan_path), iteration + 1)
                if saved["infos"]["stage"] != start:
                    raise ValueError("checkpoint stage mismatch")
            except Exception as error:
                print(f"Keep but skip invalid checkpoint {path}: {error}", flush=True)
                continue
            checkpoint = str(path)
            break
        if checkpoint is None:
            raise ValueError("no complete checkpoint exists; explicit resume must not silently start from scratch")
    for index in range(start, len(plan["stages"])):
        conflicts = gpu_conflicts()
        if conflicts:
            raise RuntimeError(f"competing workflow: {conflicts}")
        tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = root / f"stage{index}_{tag}"
        command = [sys.executable, "-u", str(ROOT / "scripts/tools/train_backpack_distribution_stage.py"),
            "--plan", str(plan_path), "--stage", str(index), "--output", str(output),
            "--headless", "--device", args.device, "--rendering_mode", "performance",
            "--kit_args=" + args.kit_args]
        if checkpoint:
            command += ["--resume-checkpoint", checkpoint]
        if args.smoke_only:
            command += ["--smoke-only"]
        environment = os.environ.copy()
        environment["WARP_CACHE_PATH"] = str(root / "cache/warp")
        Path(environment["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
        with (root / f"launch_{tag}.stdout.log").open("x") as stdout, (root / f"launch_{tag}.stderr.log").open("x") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            write_new(root / f"launch_{tag}.json", dict(pid=child.pid, command=command))
            if child.wait() != 0:
                raise RuntimeError(f"worker failed; inspect launch_{tag}.stderr.log; no automatic retries")
        if args.smoke_only:
            print("Physics smoke complete, no PPO updates", flush=True)
            return
        summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
        checkpoint = summary["checkpoint"]
        state = dict(completed_stages=index + 1, completed_updates=summary["completed_updates"],
                     checkpoint=checkpoint, checkpoint_sha256=summary["checkpoint_sha256"],
                     status="complete_requires_gait_review" if index + 1 == len(plan["stages"]) else "training")
        # Only this experiment's mutable progress pointer is replaced; all runs remain immutable.
        temporary = root / f"state_{tag}.pending"
        write_new(temporary, state)
        os.replace(temporary, state_path)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
