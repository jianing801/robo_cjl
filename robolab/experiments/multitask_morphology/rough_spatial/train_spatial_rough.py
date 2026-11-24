"""Run a fixed all-2500 Rough or backpack layout through the frozen expert engine."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ENTRY = HERE.parent
ENGINE = ENTRY / "engine"
NATIVE_WORKER = ENGINE / "persistent_worker.py"
TASK_KEYS = {"rough": "stairs", "backpack": "backpack"}


def config_path(task):
    return ENTRY / "configs" / f"{task}_spatial_all2500.json"


def profile(task="rough"):
    """Record the settings and code applied only by this entry point."""
    files = (Path(__file__).resolve(), ENTRY / "amp_engine" / "amp_layout.py")
    return dict(
        mode=f"{task}_spatial_all2500_v1",
        per_design_metrics=False,
        minibatch_sampler="vectorized_balanced",
        init_at_random_ep_len=True,
        sources={str(path.relative_to(ENTRY).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in files},
    )


def wrap_worker_command(command, task="rough"):
    # Retain the original worker argument so existing process checks identify it.
    index = command.index(str(NATIVE_WORKER))
    task_args = [] if task == "rough" else ["--task", task]
    return [*command[:index], str(Path(__file__).resolve()), "--worker", *task_args, *command[index:]]


def install_runtime(stack, runner, runtime):
    from amp_engine.amp_layout import vectorized_balanced_indices

    env = runner.env
    base_step = runtime.RslRlVecEnvWrapper.step

    def step(actions):
        obs, rewards, dones, extras = base_step(env, actions)
        extras.setdefault("log", {}).update(getattr(env.unwrapped, "_motor_health_log", {}))
        return env._extend(obs), rewards, dones, extras

    native_learn = runner.learn

    def learn(num_learning_iterations):
        # Backpack can invoke learn again when the native payload curriculum advances.
        return native_learn(num_learning_iterations, init_at_random_ep_len=runner.current_learning_iteration == 0)

    stack.enter_context(patch.object(env, "step", step))
    stack.enter_context(patch.object(runner, "learn", learn))
    stack.enter_context(patch.object(runtime, "balanced_indices", vectorized_balanced_indices))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_controller(args):
    bank = args.bank.resolve()
    root = args.output_root.resolve()
    cache = args.asset_cache.resolve()
    task = args.task
    if not (cache / "completed.json").is_file():
        raise FileNotFoundError(f"{task} USD cache has no completion proof: {cache}")
    config = config_path(task)
    cfg, bank_data = read(config), read(bank)
    rotation = cfg["rotation"]
    if (bank_data.get("task_key") != TASK_KEYS[task]
            or bank_data.get("configuration") != cfg["grid"]
            or bank_data.get("context_schema") != cfg["context_schema"]
            or sum(row["split"] == "train" for row in bank_data["designs"]) != rotation["active_designs"]):
        raise ValueError(f"{task} spatial run requires the matching complete training bank")

    sys.path.insert(0, str(ENGINE))
    native = importlib.import_module("train")
    native_write, native_child = native.write_new, native.run_child

    def write_plan(path, value):
        if Path(path) == root / "plan.json":
            value = dict(value, **{f"{task}_spatial": profile(task)})
        native_write(path, value)

    def launch_worker(command, folder, env):
        native_child(wrap_worker_command(command, task), folder, env)

    with patch.object(native, "write_new", write_plan), patch.object(native, "run_child", launch_worker):
        native.main([
            "--config", str(config), "--bank", str(bank), "--output-root", str(root),
            "--asset-cache", str(cache), "--device", args.device,
            "--gpu-min-free-gb", "12.5", "--gpu-max-utilization", "100",
            "--execution-mode", "persistent", "--execute",
        ])


def run_worker(argv):
    parser = argparse.ArgumentParser(description="Fixed-layout spatial worker")
    parser.add_argument("native_worker", type=Path)
    parser.add_argument("--task", choices=TASK_KEYS, default="rough")
    for flag in ("root", "bank", "result", "asset-cache"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--kit-args", default="")
    args = parser.parse_args(argv)
    if args.native_worker.resolve() != NATIVE_WORKER:
        parser.error("expected the native expert engine's persistent worker")

    root, bank = args.root.resolve(), args.bank.resolve()
    plan = read(root / "plan.json")
    task = args.task
    if (plan.get(f"{task}_spatial") != profile(task) or plan.get("execution_protocol") != "persistent_v1"
            or plan["configuration"] != read(config_path(task)) or Path(plan["bank"]).resolve() != bank):
        raise ValueError(f"{task} spatial runtime, bank or configuration differs from its frozen plan")

    sys.path[:0] = [str(ENGINE), str(ENTRY)]
    native_worker = importlib.import_module("persistent_worker")
    stage = importlib.import_module("train_stage")
    attach_global_logger = stage.attach_global_logger

    with ExitStack() as stack:
        # The native stage loads Isaac before invoking this hook.
        def attach(runner, target):
            attach_global_logger(runner, target)
            install_runtime(stack, runner, importlib.import_module("hybrid_runtime"))

        stack.enter_context(patch.object(stage, "attach_global_logger", attach))
        native_worker.run_groups(root, bank, args.result.resolve(), args.kit_args,
                                 args.asset_cache.resolve(), stage_main=stage.main)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--worker"]:
        run_worker(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bank", "output-root", "asset-cache"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--task", choices=TASK_KEYS, default="rough")
    run_controller(parser.parse_args(argv))


if __name__ == "__main__":
    main()
