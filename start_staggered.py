"""以独立目录运行三个专家任务的错开回合训练。"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import start as original


STAGGERED = original.ENTRY / "staggered"
TASKS = ("rough", "负重", "backpack", "amp")


def find_native_cache(root, task, bank):
    indexed = original.prepared_cache(root, task)
    if indexed is not None:
        return indexed
    bank_hash = hashlib.sha256(bank.read_bytes()).hexdigest()

    def completed_cache(folder):
        for proof in sorted(folder.glob("native_cache_*/completed.json"), reverse=True):
            try:
                cache = original.read(proof)
            except (OSError, ValueError):
                continue
            if (isinstance(cache, dict) and cache.get("cache_schema_version") == 2
                    and cache.get("result") == "NATIVE_ASSET_CACHE_COMPLETE"
                    and cache.get("bank_sha256") == bank_hash):
                return proof.parent
        return None

    ready = completed_cache(root / "资产准备" / task)
    if ready is not None:
        return ready
    original_output, has_plan = original.select_run(root, task)
    return completed_cache(original_output) if has_plan else None


def expert_command(root, task, gpu, resume_paused):
    bank = original.bank_path(root, task).resolve()
    if not bank.is_file():
        raise FileNotFoundError(f"缺少结构库：{bank}。请先用 start.py 准备资产")
    output = root / "experts_staggered" / task
    if output.resolve() != output or (root / "experts_staggered").is_symlink():
        raise ValueError(f"错开回合目录需要独立路径：{output}")
    resume = (output / "plan.json").is_file()
    if output.exists() and not resume:
        raise RuntimeError(f"错开回合目录已存在但缺少训练计划：{output}")
    plan = None
    if resume:
        sys.path.insert(0, str(STAGGERED))
        from controller import require_staggered_plan
        plan = require_staggered_plan(output, task)
    paused = (output / "USER_PAUSED").exists()
    if paused and not resume_paused:
        raise RuntimeError(f"该运行已暂停，继续时请加 --resume-paused：{output}")
    if resume_paused and not resume:
        raise ValueError("没有已有的训练计划可恢复")

    cache = find_native_cache(root, task, bank)
    if cache is None:
        raise FileNotFoundError(f"缺少与 {task} 结构库匹配的已完成 USD 缓存。请先用 start.py 资产准备")
    if task == "amp":
        library = root / "references" / "library.json"
        if plan is not None:
            library = Path(plan["reference_library"])
        if not library.is_file():
            raise FileNotFoundError(f"缺少 AMP 参考动作库：{library}。请先用 start.py 准备")

    engine = original.ENTRY / ("amp_engine" if task == "amp" else "engine")
    command = [
        sys.executable, "-u", str(STAGGERED / "controller.py"),
        "--task", task, "--config", str(original.ENTRY / "configs" / f"{task}.json"),
        "--bank", str(bank), "--output-root", str(output),
        "--asset-cache", str(cache.resolve()), "--device", f"cuda:{gpu}",
        "--gpu-min-free-gb", "12.5", "--gpu-max-utilization", "100", "--execute",
        "--native-controller", str(engine / "train.py"),
    ]
    if task == "amp":
        command += ["--reference-library", str(library.resolve())]
    if resume:
        command.append("--resume")
    if paused:
        command.append("--resume-paused")
    return output, command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--gpu", type=int, default=0)
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument("--root", type=Path)
    roots.add_argument("--migrated", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-paused", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("显卡编号不能为负数")

    task = "backpack" if args.task == "负重" else args.task
    root = (args.root or (original.MIGRATED_ROOT if args.migrated else original.DEFAULT_ROOT)).expanduser().resolve()
    if (args.migrated or (root / "migration_in_progress.json").is_file()) and not (root / "migration_completed.json").is_file():
        parser.error("结构库迁移尚未完成")
    output, command = expert_command(root, task, args.gpu, args.resume_paused)
    print(f"错开回合训练目录：{output}", flush=True)
    if args.dry_run:
        original.run(command, preview=True)
        return
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.chdir(original.REPO)
    if os.name != "nt":
        os.execv(sys.executable, command)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"启动未完成：{error}", file=sys.stderr)
        raise SystemExit(1)
