"""新版持久化 worker，在当前进程中接入原生场景与训练器。"""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

from controller import require_staggered_plan
from episode_phases import attach_phase_warmup

EXPERIMENT = Path(__file__).resolve().parents[1]
NATIVE_WORKERS = {
    EXPERIMENT / "engine/persistent_worker.py": "persistent_worker",
    EXPERIMENT / "amp_engine/amp_persistent_worker.py": "amp_persistent_worker",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_worker", type=Path)
    for option in ("root", "bank", "result", "asset-cache"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--kit-args", default="")
    args = parser.parse_args(argv)
    native_path = args.native_worker.resolve()
    module_name = NATIVE_WORKERS.get(native_path)
    if module_name is None:
        parser.error("此入口只支持仓库自带的专家持久化 worker。")

    root = args.root.resolve()
    plan = require_staggered_plan(root)
    expected_amp = module_name == "amp_persistent_worker"
    if (plan["episode_phase"]["task"] == "amp") != expected_amp:
        raise ValueError("专家计划与 worker 不匹配。")
    sys.path.insert(0, str(native_path.parent))
    import importlib
    native_worker = importlib.import_module(module_name)
    stage = importlib.import_module("train_stage")
    attach_logger = stage.attach_global_logger

    def attach_logger_and_phases(runner, target):
        attach_logger(runner, target)
        attach_phase_warmup(runner, root / "USER_PAUSED")

    with patch.object(stage, "attach_global_logger", attach_logger_and_phases):
        native_worker.run_groups(
            root,
            args.bank.resolve(),
            args.result.resolve(),
            args.kit_args,
            args.asset_cache.resolve(),
            stage_main=stage.main,
        )


if __name__ == "__main__":
    main()
