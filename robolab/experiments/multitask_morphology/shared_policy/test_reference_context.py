"""Reference designs keep the trained actor's physical feature coordinates."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[index]) for index in (0, 3)]

from contract import load_plan, sha, unified_context, write_new
from robolab.morphology_context import FIELDS, LINKS, SCHEMA, WIDTH, raw_context


def inputs(tmp_path):
    urdf = tmp_path / "reference.urdf"
    links = "".join(
        f'<link name="{name}"><inertial><mass value="1.2"/>'
        '<origin xyz="0.01 0 -0.1"/>'
        '<inertia ixx="0.03" iyy="0.04" izz="0.03" ixy="0" ixz="0" iyz="0"/>'
        '</inertial></link>' for name in LINKS
    )
    urdf.write_text(f"<robot name='reference'>{links}</robot>", encoding="utf8")
    key = [.25, .30, "DM-J10010L-2EC", "DM-J4340P-2EC"]
    pair = "__".join(key[2:])
    design = dict(id="reference", thigh_m=key[0], calf_m=key[1], knee_motor=key[2],
                  ankle_motor=key[3], pair_id=pair, urdf=str(urdf), urdf_sha256=sha(urdf),
                  context=[0.] * WIDTH)
    model = dict(knee_motor=key[2], ankle_motor=key[3], selected={
        "knee": dict(peak_torque_nm=120., max_speed_rad_s=20., armature_kgm2=.1),
        "ankle": dict(peak_torque_nm=27., max_speed_rad_s=10., armature_kgm2=.04),
    })
    values = raw_context(design, model)
    minimum = [x - .25 for x in values]
    maximum = [x + .75 for x in values]
    minimum[-1] = maximum[-1] = values[-1]
    schema = dict(tasks=["stairs", "backpack", "run"], knees=[key[2]], ankles=[key[3]],
                  shared_structure_keys=[[.2444444444, .2944444444, *key[2:]]],
                  context_schema=SCHEMA, context_width=WIDTH, context_fields=list(FIELDS),
                  context_normalization=dict(kind="training_bank_minmax_v1", minimum=minimum, maximum=maximum))
    bank = dict(schema_version=3, context_schema=SCHEMA, designs=[design], motor_pairs={pair: model})
    return key, design, schema, bank, urdf


def test_explicit_reference_uses_verified_features_in_frozen_actor_coordinates(tmp_path):
    key, design, schema, bank, urdf = inputs(tmp_path)
    before = deepcopy(schema)
    with pytest.raises(ValueError, match="intersection"):
        unified_context(bank, design, schema)
    vector = unified_context(bank, design, schema, reference_structure_key=key)
    assert vector == pytest.approx([-.5] * (WIDTH - 1) + [0.])
    assert schema == before and design["context"] == [0.] * WIDTH
    urdf.write_text("<robot name='tampered'/>", encoding="utf8")
    with pytest.raises(ValueError, match="urdf asset changed"):
        unified_context(bank, design, schema, reference_structure_key=key)


def test_reference_plan_requires_external_key_and_singleton_native_banks(tmp_path):
    key, design, schema, bank, _ = inputs(tmp_path)
    bank_path, teacher_path, shared = [tmp_path / name for name in ("bank.json", "teacher.pt", "shared.json")]
    write_new(bank_path, bank)
    teacher_path.write_bytes(b"teacher")
    write_new(shared, dict(structures=[dict(key=key, design_ids=dict.fromkeys(schema["tasks"], design["id"]))]))
    spec = dict(bank=str(bank_path), bank_sha256=sha(bank_path), checkpoint=str(teacher_path),
                checkpoint_sha256=sha(teacher_path), design_ids=[design["id"]])
    plan = dict(kind="rough_backpack_amp_distillation_plan_v3", structure_scope="teacher_intersection_only",
                schema=schema, teachers={task: spec for task in schema["tasks"]},
                task_ids={task: index for index, task in enumerate(schema["tasks"])},
                shared_bank=str(shared), shared_bank_sha256=sha(shared), selected_structure_keys=[key],
                reference_structure_key=key)
    path = tmp_path / "plan.json"
    write_new(path, plan)
    assert load_plan(path) == plan

    original_key_in_schema = deepcopy(plan)
    original_key_in_schema["schema"]["shared_structure_keys"].append(key)
    in_grid_path = tmp_path / "in_grid.json"
    write_new(in_grid_path, original_key_in_schema)
    with pytest.raises(ValueError, match="outside"):
        load_plan(in_grid_path)

    extra_row_bank = tmp_path / "extra_bank.json"
    write_new(extra_row_bank, dict(bank, designs=[design, dict(design, id="extra")]))
    non_singleton = deepcopy(plan)
    non_singleton["teachers"]["stairs"].update(bank=str(extra_row_bank), bank_sha256=sha(extra_row_bank))
    non_singleton_path = tmp_path / "non_singleton.json"
    write_new(non_singleton_path, non_singleton)
    with pytest.raises(ValueError, match="singleton"):
        load_plan(non_singleton_path)
