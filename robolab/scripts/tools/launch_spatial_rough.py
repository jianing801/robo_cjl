"""One-switch Rough workflow; all paths resolve from this checkout, never cwd."""
import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "rsl_rl")]


def original_iterations(root=ROOT):
    """Read the source Rough/Flat inherited default without importing Isaac."""
    tree = ast.parse((root/"robolab/tasks/direct/base/agents/rpo_agent_cfg.py").read_text(encoding="utf8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    def lookup(name):
        cls = classes[name]
        values = [ast.literal_eval(n.value) for n in ast.walk(cls) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                          and t.value.id == "self" and t.attr == "max_iterations" for t in n.targets)]
        if values:
            return values[-1]
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                return lookup(base.id)
        raise ValueError("cannot resolve source Rough max_iterations; specify --max_iterations")
    value = lookup("RPORoughAgentCfg")
    if type(value) is not int or value < 1:
        raise ValueError("invalid source Rough training budget")
    return value


def robot_assets(root=ROOT, override=None):
    robot = Path(override).expanduser().resolve() if override else root/"data/robots/roboparty/rpo"
    template, meshes = robot/"urdf/rpo.urdf", robot/"meshes"
    if not template.is_file() or not meshes.is_dir():
        raise FileNotFoundError(f"Original RPO assets are missing under {robot}. Install the same robot data required by original RPO there. No alternate robot will be silently substituted.")
    return template.resolve(), meshes.resolve()


def run_stage(name, command, output, *, environment=None, runner=subprocess.run):
    from robolab.rough_contract import write_new
    write_new(output/(name+".started.json"), dict(command=command, cwd=str(ROOT)))
    print(f"[Rough multi-morph] {name}", flush=True)
    try:
        result = runner(command, cwd=ROOT, check=False, env=environment)
    except KeyboardInterrupt:
        write_new(output/(name+".interrupted.json"), dict(user_interrupt=True, later_stages_not_started=True))
        raise
    write_new(output/(name+".finished.json"), dict(returncode=result.returncode))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def main(argv=None):
    from robolab.rough_contract import write_new
    from robolab.configurable_morphology import select_grid, eligible_pools
    from robolab.physical_snapshot import DEFAULT_ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--multi_morph", action="store_true", required=True)
    p.add_argument("--task", default="RPO-Rough", choices=["RPO-Rough"])
    p.add_argument("--num_envs", type=int)
    p.add_argument("--max_iterations", type=int)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device")
    p.add_argument("--gpu_min_free_gb", type=float, default=12., help="GPU selection floor in GiB, not a guaranteed memory budget")
    p.add_argument("--gpu_max_utilization", type=int, default=30, help="Maximum sampled GPU utilization percent")
    p.add_argument("--kit_args")
    p.add_argument("--multi_morph_config", default=str(ROOT/"configs/spatial_rough_grid.json"))
    p.add_argument("--multi_morph_root", help="Optional NEW output directory; normally generated automatically")
    p.add_argument("--robot-root", help="Optional nonstandard RPO data location; no need for normal installs")
    p.add_argument("--multi_morph_prepare_only", action="store_true", help="Generate assets only; no GPU or PPO")
    a = p.parse_args(argv)
    config = json.loads(Path(a.multi_morph_config).expanduser().resolve().read_text(encoding="utf8"))
    if a.num_envs is not None:
        config["num_envs"] = a.num_envs
    pools, _ = eligible_pools(DEFAULT_ROOT)
    grid = select_grid(config, pools["knee"], pools["ankle"])
    updates = original_iterations() if a.max_iterations is None else a.max_iterations
    if updates < 1:
        p.error("max_iterations must be positive")
    template, meshes = robot_assets(override=a.robot_root)
    output = (Path(a.multi_morph_root).expanduser().resolve() if a.multi_morph_root else
              ROOT/"logs/rpo_rough_multimorph"/datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"configuration.json", config)
    write_new(output/"workflow.json", dict(task="RPO-Rough", num_envs=grid["num_envs"],
        design_count=grid["design_count"], absolute_target_updates=updates, template=str(template), meshes=str(meshes),
        preflight_updates=2, preflight_policy_discarded=True, scratch_training=True))
    print(f"[Rough multi-morph] {grid['num_envs']} environments / {grid['design_count']} designs / {updates} source-PPO updates", flush=True)
    print(f"[Rough multi-morph] output: {output}", flush=True)
    prepare = [sys.executable, str(ROOT/"scripts/tools/prepare_spatial_grid.py"), "--config", str(output/"configuration.json"),
        "--template", str(template), "--mesh-dir", str(meshes), "--output-dir", str(output/"bank"), "--generate-assets"]
    run_stage("prepare", prepare, output)
    if a.multi_morph_prepare_only:
        return
    common = [sys.executable, str(ROOT/"scripts/tools/train_spatial_rough.py"), "--bank", str(output/"bank/bank.json")]
    if a.headless:
        common += ["--headless"]
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    inventory = query_gpus()
    choices = candidates(inventory, os.environ, min_free_mib=a.gpu_min_free_gb*1024,
                         max_utilization=a.gpu_max_utilization, requested=a.device)
    write_new(output/"gpu_inventory.json", dict(gpus=inventory, min_free_gib=a.gpu_min_free_gb,
        max_utilization=a.gpu_max_utilization, eligible_uuids=[r["uuid"] for r in choices]))
    with gpu_lease(choices, ROOT/"logs/rpo_rough_multimorph/gpu_locks") as gpu:
        environment = child_environment(gpu, os.environ)
        write_new(output/"gpu_selection.json", dict(**gpu, child_cuda_visible_devices=gpu["uuid"],
            child_device="cuda:0", render_physical_gpu=gpu["index"], cooperative_lock_not_external_reservation=True))
        print(f"[Rough multi-morph] selected GPU {gpu['index']} ({gpu['uuid']}), free {gpu['free_mib']} MiB, utilization {gpu['utilization']}%", flush=True)
        # CUDA sees one UUID as logical cuda:0; RTX activeGpu is a physical index.
        # The two index spaces must not be confused on shared multi-GPU machines.
        kit = (a.kit_args or "")+f" --/renderer/activeGpu={gpu['index']} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0"
        common += ["--device", "cuda:0", "--kit_args", kit]
        run_stage("preflight", common+["--output-dir", str(output/"preflight"), "--preflight-only"], output, environment=environment)
        proof = json.loads((output/"preflight/summary.json").read_text(encoding="utf8"))
        if proof.get("result") != "SPATIAL_TWO_UPDATE_PREFLIGHT_PASSED_DISCARD_POLICY":
            raise ValueError("preflight did not pass; formal training will NOT start")
        ready = candidates(query_gpus(), environment, min_free_mib=a.gpu_min_free_gb*1024,
                           max_utilization=a.gpu_max_utilization, requested="cuda:0")
        if not ready:
            raise RuntimeError("Selected GPU is no longer eligible after preflight; stopping without moving to another GPU or killing jobs")
        run_stage("train", common+["--output-dir", str(output/"train"), "--execute", "--updates", str(updates),
            "--preflight-report", str(output/"preflight/summary.json"), "--asset-cache", str(output/"preflight")], output, environment=environment)
    summary = json.loads((output/"train/summary.json").read_text(encoding="utf8"))
    if summary.get("result") != "SPATIAL_FINITE_TRAINING_COMPLETE_NOT_GAIT_VALIDATED" or summary.get("completed_updates") != updates:
        raise ValueError("training did not complete the requested update budget")
    write_new(output/"completed.json", dict(completed_updates=updates, requires_real_gait_evaluation=True))
    print(f"[Rough multi-morph] Training complete, gait NOT yet validated: {output}", flush=True)


if __name__ == "__main__":
    main()
