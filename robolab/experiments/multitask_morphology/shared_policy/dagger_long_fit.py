"""Run the frozen fitter with one prioritized 20-second DAgger archive per task."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys

TASKS = ("stairs", "backpack", "run")
AUDIT_NAME = "long_replay_sampling.json"
PRIORITY_SOURCE = "real_student_DAgger_with_frozen_native_teacher_labels"
PRIORITY_STEPS = 1000
ROWS_PER_TASK_PER_STEP = 8 * 16


def read_priority_data(path):
    raw = Path(path).read_bytes()
    paths = json.loads(raw)
    if (not isinstance(paths, list) or len(paths) != len(TASKS)
            or any(not isinstance(value, str) or not Path(value).is_absolute()
                   or Path(value).name != "data.pt" for value in paths)):
        raise ValueError("Priority data must list three absolute data.pt paths in stairs, backpack, run order")
    resolved = [str(Path(value).resolve(strict=True)) for value in paths]
    if len(set(resolved)) != len(TASKS):
        raise ValueError("Each task requires a distinct priority archive")
    return dict(zip(TASKS, resolved)), hashlib.sha256(raw).hexdigest()


def training_view(groups, sources, priority):
    by_path = {source["path"]: source for source in sources}
    for task, path in priority.items():
        if path not in by_path:
            raise ValueError(f"Missing priority archive for {task} in the fit replay: {path}")
        source = by_path[path]
        if source["task"] != task:
            raise ValueError(f"The priority archive for {task} belongs to {source['task']}")
        if source["source"] != PRIORITY_SOURCE:
            raise ValueError(f"Expected a 20-second student DAgger collection for {task}: {path}")

    preferred = {}
    weighted = {}
    for key, group in groups.items():
        task, design = key
        fresh = [shard for shard in group["train"] if shard.path == priority[task]]
        others = [shard for shard in group["train"] if shard.path != priority[task]]
        if len(fresh) != 1:
            raise ValueError(f"Expected exactly one priority replay shard for {task}/{design}")
        if not others:
            raise ValueError(f"Found no baseline replay for {task}/{design}")
        preferred[task] = fresh[0]
        # The frozen batch draws one uniform source index per design. Give the
        # priority shard one third of those indices and split the rest evenly.
        # This view holds references to the original mmap shards.
        weighted[key] = dict(group, train=tuple(fresh * len(others) + others + others))

    for task, path in priority.items():
        archive = preferred[task].store.load(path)
        if archive.get("run_arguments", {}).get("steps") != PRIORITY_STEPS:
            raise ValueError(f"Expected a 20-second student DAgger collection for {task}: {path}")
        del archive

    return weighted


def sampling_audit(groups, sources, priority, priority_digest, summary):
    draws = {task: Counter() for task in TASKS}
    for source in sources:
        draws[source["task"]][source["path"]] = 0
    for (task, _), group in groups.items():
        for shard in group["train"]:
            draws[task][shard.path] += shard.sampled_draws

    source_hashes = {source["path"]: source["sha256"] for source in sources}
    expected = summary["optimizer_steps"] * ROWS_PER_TASK_PER_STEP
    tasks = {}
    for task, path in priority.items():
        counts = draws[task]
        total = sum(counts.values())
        if total != expected or total != summary["cumulative_training_exposure"][task]["sampled_draws"]:
            raise ValueError(f"Training draw totals for {task} differ from the frozen fit summary")
        tasks[task] = dict(priority_path=path, priority_sha256=source_hashes[path],
                           priority_rows=counts[path], total_rows=total, source_rows=dict(sorted(counts.items())))
    return dict(kind="dagger_long_sampling_v1", priority_data_sha256=priority_digest,
                priority_probability=1 / 3, selection_unit="16 rows per design draw", tasks=tasks)


def install_sampling(fit, priority, priority_digest):
    native_load_groups = fit["load_groups"]
    native_batch = fit["batch"]
    native_write_new = fit["write_new"]
    groups = sources = weighted = None

    def load_groups(paths, plan, *args, **kwargs):
        nonlocal groups, sources, weighted
        if tuple(plan["schema"]["tasks"]) != TASKS:
            raise ValueError("Priority replay requires the stairs, backpack, run fit plan")
        groups, sources, io = native_load_groups(paths, plan, *args, **kwargs)
        weighted = training_view(groups, sources, priority)
        return groups, sources, io

    def batch(groups, generator, plan, *args, record_exposure=True, **kwargs):
        # The empirical normalization parity probe uses the native sampling view.
        return native_batch(weighted if record_exposure else groups, generator, plan, *args,
                            record_exposure=record_exposure, **kwargs)

    def write_new(path, value):
        if path.name == "manifest.json":
            value = dict(value, sampling=(
                "equal tasks; uniform designs; one-third current 20-second DAgger source and "
                "two-thirds uniform among other sources per task/design; uniform scenario and "
                "environment/episode; rows sampled on demand from full mmap replay"
            ), long_replay_sampling_audit=AUDIT_NAME)
        if path.name == "summary.json":
            audit = sampling_audit(groups, sources, priority, priority_digest, value)
            native_write_new(path.with_name(AUDIT_NAME), audit)
        native_write_new(path, value)

    fit.update(load_groups=load_groups, batch=batch, write_new=write_new)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path, help="Path to the frozen fit.py")
    parser.add_argument("--priority-data", required=True, type=Path,
                        help="JSON list of the stairs, backpack and run data.pt paths in that order")
    parser.add_argument("native", nargs=argparse.REMAINDER, help="Frozen fit arguments after --")
    args = parser.parse_args(argv)
    if len(args.native) < 2 or args.native[0] != "--":
        parser.error("Pass frozen fit arguments after --")

    fit_path = args.fit.resolve(strict=True)
    priority, priority_digest = read_priority_data(args.priority_data)
    original_argv, original_path, original_bytecode = sys.argv, sys.path, sys.dont_write_bytecode
    try:
        sys.argv = [str(fit_path), *args.native[1:]]
        sys.path = [str(fit_path.parent), *original_path]
        sys.dont_write_bytecode = True
        native_main = runpy.run_path(str(fit_path))["main"]
        # runpy returns a copy. The loaded fitter's functions share their original globals.
        install_sampling(native_main.__globals__, priority, priority_digest)
        native_main()
    finally:
        sys.argv, sys.path, sys.dont_write_bytecode = original_argv, original_path, original_bytecode


if __name__ == "__main__":
    main()
