"""CPU coverage of prioritized replay through the native fit entry point."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit
from contract import graph_features, schema_for, unified_context
from student import SharedMLPActor, save_student

TASKS = ("stairs", "backpack", "run")
FITTER = Path(fit.__file__)
WRAPPER = FITTER.with_name("dagger_long_fit.py")


@pytest.fixture
def replay(tmp_path):
    names = [f"joint_{i}" for i in range(23)]
    semantic = dict(joint_names=names, default_joint_pos=[0.] * 23)
    nodes = ['<link name="base"><inertial><mass value="1"/></inertial></link>']
    for i, name in enumerate(names):
        nodes.extend([
            f'<link name="link_{i}"><inertial><mass value=".1"/></inertial></link>',
            f'<joint name="{name}" type="revolute"><parent link="base"/><child link="link_{i}"/>'
            '<origin xyz="0 0 .1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2"/></joint>',
        ])
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="smoke">' + "".join(nodes) + "</robot>", encoding="utf8")
    keys = [[.3, .35, "K1", "A1"], [.35, .35, "K1", "A1"]]
    teachers, banks = {}, {}
    for task in TASKS:
        designs = [
            dict(id=f"{task}_{i}", thigh_m=key[0], calf_m=key[1], knee_motor=key[2],
                 ankle_motor=key[3], context=[0., 0., 1., 1.] + [0.] * 16, urdf=str(urdf), urdf_sha256=fit.sha(urdf))
            for i, key in enumerate(keys)
        ]
        banks[task] = dict(canonical_semantic_contract=semantic, ranges_m=dict(thigh=[.25, .4], calf=[.3, .45]),
                           knees=["K1"], ankles=["A1"], designs=designs)
        bank = tmp_path / f"{task}_bank.json"
        bank.write_text(json.dumps(banks[task]), encoding="utf8")
        teacher = tmp_path / f"{task}_teacher.pt"
        teacher.write_bytes(b"frozen teacher identity")
        teachers[task] = dict(bank=str(bank), bank_sha256=fit.sha(bank), checkpoint=str(teacher),
                              checkpoint_sha256=fit.sha(teacher), design_ids=[d["id"] for d in designs])

    schema = schema_for(banks)
    schema["shared_structure_keys"] = keys
    shared_bank = tmp_path / "shared.json"
    shared_bank.write_text(json.dumps({"structures": keys}), encoding="utf8")
    plan = dict(kind="rough_backpack_amp_distillation_plan_v3", structure_scope="teacher_intersection_only",
                teachers=teachers, schema=schema, task_ids={task: i for i, task in enumerate(TASKS)},
                shared_bank=str(shared_bank), shared_bank_sha256=fit.sha(shared_bank), selected_structure_keys=keys)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf8")
    init = tmp_path / "init.pt"
    save_student(init, SharedMLPActor(schema, {"hidden_dims": [8]}), plan_sha256=fit.sha(plan_path))

    paths, priority = [], []
    graph = graph_features(str(urdf), names, semantic["default_joint_pos"])
    flattened = [value for node in graph for value in node]
    io = dict(**semantic, step_dt=.02, action_scale=.25, history_width=780,
              history_layout="frame_major_oldest_to_newest")
    for task in TASKS:
        static = torch.tensor([flattened + unified_context(banks[task], design, schema)
                               for design in banks[task]["designs"]])
        environments = torch.arange(16) % 2
        for source in range(4):
            current = source == 3
            folder = tmp_path / f"{task}_{source}"
            folder.mkdir()
            path = folder / "data.pt"
            x = torch.full((16, 780), source * .1)
            y = torch.full((16, 23), source * .01)
            torch.save(dict(kind="real_rough_backpack_teacher_data_v1", schema=schema, task=task,
                            teacher_sha256=teachers[task]["checkpoint_sha256"],
                            bank_sha256=teachers[task]["bank_sha256"], plan_sha256=fit.sha(plan_path),
                            io_contract=io, base_obs=x, teacher_actions=y, validation=environments.bool(),
                            environment_ids=environments, design_ids=teachers[task]["design_ids"],
                            design_index=torch.arange(16) // 8, static_context=static,
                            actual_payload_not_in_actor=True, source=(
                                "real_student_DAgger_with_frozen_native_teacher_labels" if current else
                                "real_teacher_execution"),
                            run_arguments=dict(steps=1000 if current else 250)), path)
            paths.append(str(path))
            if current:
                priority.append(str(path))
    return plan_path, init, paths, priority


def run_fit(tmp_path, replay, name="output", *, steps=64):
    plan, init, paths, priority = replay
    listing = tmp_path / "data_list.json"
    listing.write_text(json.dumps(paths), encoding="utf8")
    priority_file = tmp_path / "priority.json"
    priority_file.write_text(json.dumps(priority), encoding="utf8")
    output = tmp_path / name
    args = [
        sys.executable, str(WRAPPER), "--fit", str(FITTER), "--priority-data", str(priority_file), "--",
        "--plan", str(plan), "--data-list", str(listing), "--init-student", str(init), "--output", str(output),
        "--architecture", "mlp", "--device", "cpu", "--epochs", "1", "--steps-per-epoch", str(steps),
        "--validation-rows-per-design-shard", "2", "--replay-cache-shards", "1",
    ]
    process = subprocess.run(args, capture_output=True, text=True, timeout=90)
    return process, output, priority_file


def test_long_sources_receive_one_third_of_equal_training_draws(tmp_path, replay):
    process, output, priority_file = run_fit(tmp_path, replay)
    assert process.returncode == 0, process.stdout + process.stderr
    audit = json.loads((output / "long_replay_sampling.json").read_text(encoding="utf8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf8"))
    assert audit["kind"] == "dagger_long_sampling_v1"
    assert audit["priority_data_sha256"] == fit.sha(priority_file)
    assert audit["priority_probability"] == 1 / 3
    assert manifest["long_replay_sampling_audit"] == "long_replay_sampling.json"
    assert summary["full_replay_rows"] == 96 and summary["mmap"] and summary["cache_limit"] == 1
    assert manifest["validation_rows_per_design_source_split"] == 2
    assert len(manifest["sources"]) == 12
    assert set(audit["tasks"]) == set(TASKS)
    for task, priority_path in zip(TASKS, replay[-1]):
        sample = audit["tasks"][task]
        recorded = summary["cumulative_training_exposure"][task]
        assert sample["total_rows"] == recorded["sampled_draws"] == 64 * 128
        assert sample["priority_path"] == priority_path
        assert sample["priority_sha256"] == fit.sha(priority_path)
        assert sample["source_rows"][priority_path] == sample["priority_rows"]
        assert sum(sample["source_rows"].values()) == sample["total_rows"]
        assert abs(sample["priority_rows"] / sample["total_rows"] - 1 / 3) < .06
        for source_path, count in sample["source_rows"].items():
            if source_path != priority_path:
                assert abs(count / sample["total_rows"] - 2 / 9) < .06

    second, second_output, _ = run_fit(tmp_path, replay, "repeat")
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads((second_output / "long_replay_sampling.json").read_text(encoding="utf8")) == audit


@pytest.mark.parametrize("invalid", ["wrong_task", "missing_design", "no_baseline", "short_collection"])
def test_invalid_priority_replay_is_rejected_before_optimization(tmp_path, replay, invalid):
    _, _, paths, priority = replay
    if invalid == "wrong_task":
        priority[:2] = reversed(priority[:2])
        expected = "priority archive for stairs belongs to backpack"
    elif invalid == "no_baseline":
        paths[:] = [path for path in paths if task_not_stairs_baseline(path, priority[0])]
        expected = "no baseline replay for stairs/stairs_0"
    else:
        path = priority[0]
        data = torch.load(path, weights_only=True)
        if invalid == "missing_design":
            for name in ("base_obs", "teacher_actions", "validation", "environment_ids", "design_index"):
                data[name] = data[name][:8]
            data["design_ids"] = data["design_ids"][:1]
            data["static_context"] = data["static_context"][:1]
            expected = "exactly one priority replay shard for stairs/stairs_1"
        else:
            data["run_arguments"]["steps"] = 250
            expected = "20-second student DAgger collection for stairs"
        torch.save(data, path)

    process, output, _ = run_fit(tmp_path, replay, steps=1)
    assert process.returncode != 0
    assert expected in process.stderr
    assert not output.exists()


def task_not_stairs_baseline(path, stairs_priority):
    return path == stairs_priority or not Path(path).parent.name.startswith("stairs_")
