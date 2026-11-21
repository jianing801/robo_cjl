"""CPU fake-robot checks, not evidence of native Isaac startup execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physical_audit import audit_physics


def fixture(tmp_path, *, robust=False):
    designs = []
    for i, mass in enumerate((4., 5.)):
        path = tmp_path / f"design{i}.urdf"
        path.write_text(f'<robot name="test"><link name="torso"><inertial><mass value="{mass}"/>'
            '</inertial></link><link name="leg"><inertial><mass value="1"/></inertial></link></robot>')
        designs.append(dict(id=f"d{i}", urdf=str(path), total_mass_kg=mass + 1))
    nominal = torch.tensor([[4., 1.], [5., 1.], [4., 1.], [5., 1.]])
    inertia = torch.eye(3).repeat(4, 2, 1, 1).reshape(4, 2, 9)
    gains = torch.tensor([[20., 30.]]).repeat(4, 1)
    data = NS(default_mass=nominal.clone(), default_inertia=inertia.clone(),
        default_joint_stiffness=gains.clone(), joint_stiffness=gains.clone(),
        default_joint_damping=gains.clone() * .1, joint_damping=gains.clone() * .1,
        default_joint_armature=torch.ones(4, 2), joint_armature=torch.ones(4, 2))
    view = NS(mass=nominal.clone(), inertia=inertia.clone(), com=torch.zeros(4, 2, 7),
        material=torch.ones(4, 2, 3))
    view.get_masses = lambda: view.mass
    view.get_inertias = lambda: view.inertia
    view.get_coms = lambda: view.com
    view.get_material_properties = lambda: view.material
    raw = NS(num_envs=4, robot=NS(root_physx_view=view, data=data,
        body_names=["torso", "leg"], joint_names=["j0", "j1"], actuators={}))
    events = NS(add_base_mass=None, scale_link_mass=None)
    if robust:
        events.add_base_mass = NS(mode="startup", params=dict(asset_cfg=NS(name="robot", body_names="torso"),
            mass_distribution_params=(-.5, .5), operation="add"))
        events.scale_link_mass = NS(mode="startup", params=dict(asset_cfg=NS(name="robot", body_names=["leg"]),
            mass_distribution_params=(.8, 1.2), operation="scale"))
    cfg = NS(events=events, scene=NS(robot=NS(actuators=dict(knee=NS(joint_names_expr=["j0"], stiffness=20.,
        damping=2., spatial_parameters=[dict(armature=1., curve=[[0., 10.], [1., 0.]]),
            dict(armature=2., curve=[[0., 20.], [2., 0.]])])))))
    return raw, cfg, designs


def test_nominal_identity_json_and_copy_routing(tmp_path):
    raw, cfg, designs = fixture(tmp_path)
    result = audit_physics(raw, cfg, designs, profile="nominal")
    json.dumps(result, allow_nan=False)
    assert result["nominal_bank_urdf_mass_verified"]
    assert [row["design_id"] for row in result["environments"]] == ["d0", "d1", "d0", "d1"]
    assert result["designs"][0]["environment_ids"] == [0, 2]
    assert result["designs"][0]["nominal_actuator_fingerprint"] != result["designs"][1]["nominal_actuator_fingerprint"]
    assert "physx_joint_armature" in result["unavailable_properties"]


def test_nominal_rejects_changed_mass_even_with_same_total(tmp_path):
    raw, cfg, designs = fixture(tmp_path)
    raw.robot.root_physx_view.mass[2] += torch.tensor([.1, -.1])
    with pytest.raises(ValueError, match="nominal mass outside"):
        audit_physics(raw, cfg, designs, profile="nominal")


def test_physical_audit_checks_last_partial_design_copy(tmp_path):
    raw, cfg, designs = fixture(tmp_path)
    raw.num_envs = 3
    for owner in (raw.robot.data, raw.robot.root_physx_view):
        for key, tensor in vars(owner).items():
            if torch.is_tensor(tensor):
                setattr(owner, key, tensor[:3])
    result = audit_physics(raw, cfg, designs, profile="nominal")
    assert [row['environment_ids'] for row in result['designs']] == [[0, 2], [1]]
    raw.robot.data.default_mass[2] = raw.robot.data.default_mass[1]
    raw.robot.root_physx_view.mass[2] = raw.robot.data.default_mass[1]
    with pytest.raises(ValueError, match='env=2'):
        audit_physics(raw, cfg, designs, profile="nominal")


def test_robust_legal_body_ranges_and_default_identity(tmp_path):
    raw, cfg, designs = fixture(tmp_path, robust=True)
    raw.robot.root_physx_view.mass[:, 0] += torch.tensor([-.5, .5, .2, -.2])
    raw.robot.root_physx_view.mass[:, 1] *= torch.tensor([.8, 1.2, 1., 1.1])
    result = audit_physics(raw, cfg, designs, profile="robust")
    assert result["mass_events"]["add_base_mass"]["body_names"] == ["torso"]
    assert result["environments"][0]["nominal_total_mass_kg"] == 5.
    assert result["environments"][0]["actual_total_mass_kg"] != 5.
    raw.robot.root_physx_view.mass[3, 1] = 1.3
    with pytest.raises(ValueError, match="env=3, body=leg"):
        audit_physics(raw, cfg, designs, profile="robust")


def test_robust_cannot_hide_wrong_design_routing(tmp_path):
    raw, cfg, designs = fixture(tmp_path, robust=True)
    raw.robot.data.default_mass[2] = raw.robot.data.default_mass[1]
    raw.robot.root_physx_view.mass[2] = raw.robot.data.default_mass[1]
    with pytest.raises(ValueError, match="default_mass/bank.*env=2"):
        audit_physics(raw, cfg, designs, profile="robust")


def test_robust_unselected_body_cannot_drift(tmp_path):
    raw, cfg, designs = fixture(tmp_path, robust=True)
    cfg.events.scale_link_mass = None
    raw.robot.root_physx_view.mass[0, 1] = 1.1
    with pytest.raises(ValueError, match="body=leg"):
        audit_physics(raw, cfg, designs, profile="robust")


def test_overlapping_mass_events_are_rejected_instead_of_guessing_order(tmp_path):
    raw, cfg, designs = fixture(tmp_path, robust=True)
    cfg.events.scale_link_mass.params["asset_cfg"].body_names = ".*"
    with pytest.raises(ValueError, match="Overlapping mass events"):
        audit_physics(raw, cfg, designs, profile="robust")


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_negative_nonfinite_mass_rejected(tmp_path, value):
    raw, cfg, designs = fixture(tmp_path, robust=True)
    raw.robot.root_physx_view.mass[0, 0] = value
    with pytest.raises(ValueError, match="positive|Nonfinite"):
        audit_physics(raw, cfg, designs, profile="robust")


def test_indefinite_inertia_rejected_despite_positive_diagonal(tmp_path):
    raw, cfg, designs = fixture(tmp_path)
    raw.robot.root_physx_view.inertia[0, 0] = torch.tensor([1., 2., 0., 2., 1., 0., 0., 0., 1.])
    with pytest.raises(ValueError, match="eigenvalue"):
        audit_physics(raw, cfg, designs, profile="nominal")


def test_missing_defaults_and_urdf_identity_fail_closed(tmp_path):
    raw, cfg, designs = fixture(tmp_path)
    del raw.robot.data.default_mass
    with pytest.raises(ValueError, match="Missing robot.data.default_mass"):
        audit_physics(raw, cfg, designs, profile="nominal")
    raw, cfg, designs = fixture(tmp_path)
    designs[0]["total_mass_kg"] = 6.
    with pytest.raises(ValueError, match="bank/URDF"):
        audit_physics(raw, cfg, designs, profile="nominal")


def test_selector_and_unknown_distribution_are_not_silently_accepted(tmp_path):
    raw, cfg, designs = fixture(tmp_path, robust=True)
    cfg.events.add_base_mass.params["asset_cfg"].body_names = "missing"
    with pytest.raises(ValueError, match="matches no body"):
        audit_physics(raw, cfg, designs, profile="robust")
    cfg.events.add_base_mass.params["asset_cfg"].body_names = "torso"
    cfg.events.add_base_mass.params["distribution"] = "gaussian"
    with pytest.raises(ValueError, match="Cannot bound"):
        audit_physics(raw, cfg, designs, profile="robust")


def test_optional_actual_property_nonfinite_rejected(tmp_path):
    raw, cfg, designs = fixture(tmp_path)
    raw.robot.root_physx_view.com[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite physx_coms"):
        audit_physics(raw, cfg, designs, profile="nominal")


def test_nominal_actuator_fingerprint_ignores_actual_dr_and_task_design_id(tmp_path):
    raw, cfg, designs = fixture(tmp_path, robust=True)
    before = audit_physics(raw, cfg, designs, profile="robust")
    raw.robot.data.joint_stiffness *= 1.1
    renamed = deepcopy(designs)
    renamed[0]["id"] = "other_task_d0"
    after = audit_physics(raw, cfg, renamed, profile="robust")
    assert before["designs"][0]["nominal_actuator_fingerprint"] == after["designs"][0]["nominal_actuator_fingerprint"]
    assert before["environments"][0]["properties"]["data_joint_stiffness"] != after["environments"][0]["properties"]["data_joint_stiffness"]
