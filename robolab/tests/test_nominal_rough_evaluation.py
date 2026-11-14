import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load("nominal_rough", "nominal_rough_evaluation.py")
driver = load("rough_auto_driver", "auto_evaluate_completed_rough.py")


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


def test_three_shapes_three_different_motor_pairs_repeatably():
    bank = dict(knees=["k0", "k1"], ankles=["a0", "a1"], designs=[])
    for thigh, calf in [(.25, .30), (.40, .45), (.2875, .3375)]:
        for knee in bank["knees"]:
            for ankle in bank["ankles"]:
                bank["designs"].append(dict(thigh_m=thigh, calf_m=calf, knee_motor=knee, ankle_motor=ankle))
    first = driver.select_cases(bank)
    assert first == driver.select_cases(bank)
    assert len({(d["knee_motor"], d["ankle_motor"]) for d in first}) == 3
    assert len({(d["thigh_m"], d["calf_m"]) for d in first}) == 3


def test_inference_actions_can_be_masked_without_changing_alive_actions():
    import torch
    with torch.inference_mode():
        inferred = torch.ones(4, 23)
    actions = inferred.clone()
    alive = torch.tensor([True, False, True, False])
    actions[~alive] = 0.
    assert torch.equal(actions[alive], inferred[alive])
    assert not actions[~alive].any()
