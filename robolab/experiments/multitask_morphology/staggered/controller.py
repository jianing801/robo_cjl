"""独立的错开回合训练控制器，复用原生调度、缓存和检查点校验。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ENTRY = HERE.parent
TASK_KEYS = {"rough": "stairs", "backpack": "backpack", "amp": "run"}
SOURCES = ("controller.py", "episode_phases.py", "staggered_worker.py")


def native_controller_path(task):
    return ENTRY / ("amp_engine" if task == "amp" else "engine") / "train.py"


def phase_identity(task):
    return dict(
        mode="staggered",
        task=task,
        sources={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES},
    )


def require_staggered_plan(output, task=None):
    """只恢复由本入口建立、且预热代码版本匹配的运行。"""
    path = Path(output) / "plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    recorded = plan.get("episode_phase")
    if not isinstance(recorded, dict) or recorded.get("task") not in TASK_KEYS:
        raise ValueError(f"该目录没有错开回合训练计划：{path}")
    expected_task = task if task is not None else recorded["task"]
    if recorded != phase_identity(expected_task):
        raise ValueError(f"错开回合训练任务或代码与冻结计划不一致：{path}")
    protocol = "amp_persistent_v1" if expected_task == "amp" else "persistent_v1"
    if plan.get("execution_protocol") != protocol or plan.get("start_stage") or plan.get("bootstrap_checkpoint"):
        raise ValueError("错开回合入口需要本方案建立的持久训练计划")
    return plan


def worker_command(command, task):
    """保留原 worker 路径参数，供现有同结构库进程检查识别。"""
    filename = "amp_persistent_worker.py" if task == "amp" else "persistent_worker.py"
    wanted = str(native_controller_path(task).with_name(filename))
    if wanted not in command:
        return command
    index = command.index(wanted)
    return [*command[:index], str(HERE / "staggered_worker.py"), *command[index:]]


def run_native(args, native_args):
    native_path = native_controller_path(args.task)
    sys.path.insert(0, str(native_path.parent))
    spec = importlib.util.spec_from_file_location("staggered_native_controller", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    root = args.output_root.resolve()
    if args.resume:
        plan = require_staggered_plan(root, args.task)
        if Path(plan["bank"]).resolve() != args.bank.resolve():
            raise ValueError("恢复时的结构库与错开回合计划不一致")
    if args.resume_paused:
        marker = root / "USER_PAUSED"
        with native.controller_lock(root):
            if marker.exists():
                if not any((root / "commits").glob("stage_*.json")):
                    raise RuntimeError("首组未提交断点，请保留此目录并使用新的错开回合实验目录")
                if args.task == "amp":
                    from amp_processes import reject_duplicate_bank_safely
                else:
                    from hybrid_processes import reject_duplicate_bank_safely
                reject_duplicate_bank_safely(args.bank)
                verified = native.load_plan(root / "plan.json", args.bank)
                if verified["configuration"] != json.loads(args.config.read_text(encoding="utf-8")):
                    raise ValueError("恢复时的配置与冻结计划不一致")
                if args.task == "amp" and str(args.reference_library.resolve()) != verified["reference_library"]:
                    raise ValueError("恢复时的 AMP 参考动作库与冻结计划不一致")
                marker.unlink()

    native_write = native.write_new
    native_child = native.run_child

    def write_plan(path, value):
        if Path(path) == root / "plan.json":
            value = dict(value, episode_phase=phase_identity(args.task))
        native_write(path, value)

    def launch_child(command, folder, env):
        wrapped = worker_command(command, args.task)
        if wrapped is command:
            raise RuntimeError("原生控制器未使用预期的持久训练 worker")
        native_child(wrapped, folder, env)

    with patch.object(native, "write_new", write_plan), patch.object(native, "run_child", launch_child):
        native.main(native_args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_KEYS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-controller", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device")
    parser.add_argument("--asset-cache", type=Path)
    parser.add_argument("--reference-library", type=Path)
    parser.add_argument("--gpu-min-free-gb", type=float)
    parser.add_argument("--gpu-max-utilization", type=int)
    parser.add_argument("--kit-args")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-paused", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.native_controller and args.native_controller.resolve() != native_controller_path(args.task):
        parser.error("--native-controller 与任务不匹配")
    if args.resume_paused and not (args.resume and args.execute):
        parser.error("--resume-paused 需要同时执行 --resume --execute")
    if any(part.casefold() == "experts" for part in args.output_root.resolve().parts):
        parser.error("请选择独立的错开回合目录")
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    if bank.get("task_key") != TASK_KEYS[args.task]:
        parser.error("任务与结构库类型不匹配")
    if args.execute and (not args.asset_cache or not (args.asset_cache / "completed.json").is_file()):
        parser.error("--execute 需要带 completed.json 的原生 USD 缓存")
    if args.task == "amp":
        if not args.reference_library or not args.reference_library.is_file():
            parser.error("AMP 需要现有的 --reference-library")
    elif args.reference_library:
        parser.error("--reference-library 仅用于 AMP")

    native_args = [
        "--config", str(args.config.resolve()),
        "--bank", str(args.bank.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--execution-mode", "persistent",
    ]
    for name in ("device", "asset_cache", "reference_library", "gpu_min_free_gb", "gpu_max_utilization", "kit_args"):
        value = getattr(args, name)
        if value is not None:
            path_or_value = value.resolve() if isinstance(value, Path) else value
            native_args.append(f"--{name.replace('_', '-')}={path_or_value}")
    if args.resume:
        native_args.append("--resume")
    if args.execute:
        native_args.append("--execute")
    run_native(args, native_args)


if __name__ == "__main__":
    main()
