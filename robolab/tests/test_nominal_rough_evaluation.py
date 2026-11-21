import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load("nominal_rough", ROOT / "scripts/tools/nominal_rough_evaluation.py")


class Events(NS):
    def to_dict(self):
        return vars(self)


@pytest.mark.parametrize("reset_name,expected", [("reset_joints_by_offset", 0.), ("reset_joints_by_scale", 1.)])
def test_fixed_profile_no_randomization_or_external_sensor(reset_name, expected):
    def root(): pass
    def joints(): pass
    root.__name__, joints.__name__ = "reset_root_state_uniform", reset_name
    cfg = NS(events=Events(reset_base=NS(func=root, params={}),
        reset_robot_joints=NS(func=joints, params={}), push_robot=NS(mode="interval"),
        physics_material=NS(mode="startup"), mass=NS(mode="startup")),
        noise=NS(add_noise=True), commands=NS(ranges=NS()),
        scene=NS(robot=NS(actuators={"knees": NS(min_delay=0, max_delay=4)})),
        scene_context=NS(height_scanner=NS(enable_height_scan_actor=False)))
    result = profile.configure_nominal_rough(cfg)
    assert cfg.events.push_robot is None and cfg.events.physics_material is None and cfg.events.mass is None
    assert cfg.events.reset_robot_joints.params["position_range"] == (expected, expected)
    assert cfg.commands.heading_command and cfg.commands.rel_heading_envs == 1.
    assert cfg.commands.ranges.lin_vel_x == (.5, .5)
    assert not cfg.noise.add_noise and cfg.scene.robot.actuators["knees"].max_delay == 0
    assert not result["physical_domain_randomization"] and not result["pushes"]


def test_reference_experts_select_the_published_structures_in_the_same_order(monkeypatch):
    entry = ROOT / "experiments/multitask_morphology"
    reference = entry / "reference_evaluation"
    published = ROOT.parent / "docs/reference/rough_amp_expert_20261121"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    amp = load("reference_amp", reference / "evaluate_amp.py")
    rough = load("reference_rough", reference / "evaluate_rough.py")
    from hybrid_bank import select_grid

    config = json.loads((entry / "configs/rough.json").read_text(encoding="utf-8"))
    grid = config["grid"]
    bank = select_grid(grid, grid["knee_pool"], grid["ankle_pool"])
    bank.update(configuration=grid, ranges_m=dict(thigh=grid["thigh_range_m"], calf=grid["calf_range_m"]))
    selected = amp.select_group(bank, config)
    frozen_amp = json.loads((published / "amp/selection.json").read_text(encoding="utf-8"))
    assert selected["group"] == frozen_amp["group"] == 21
    assert selected["structures"] == frozen_amp["structures"]
    assert all(selected["selected_counts_3x3"]["all"])

    bank["designs"].reverse()
    rough_rows = rough.select_reference_structures(bank, config, selected)
    frozen_rough = json.loads((published / "rough/selection.json").read_text(encoding="utf-8"))
    assert rough_rows == frozen_rough["structures"]
    assert len(rough_rows) == 50
    assert len({(row["knee_motor"], row["ankle_motor"]) for row in rough_rows}) == 25


def test_inference_actions_can_be_masked_without_changing_alive_actions():
    import torch
    with torch.inference_mode():
        inferred = torch.ones(4, 23)
    actions = inferred.clone()
    alive = torch.tensor([True, False, True, False])
    actions[~alive] = 0.
    assert torch.equal(actions[alive], inferred[alive])
    assert not actions[~alive].any()
