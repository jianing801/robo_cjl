"""CPU tests: pitch invariance, severe falls, timeouts and profile consistency."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "robolab/tasks/direct/backpack_rpo_flat"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rewards = _load_file("adaptive_rewards", PACKAGE / "rewards.py")
profiles = _load_file("backpack_profiles", ROOT / "scripts/tools/backpack_rpo_profile.py")


def _gravity(pitch_deg=0, roll_deg=0):
    pitch, roll = math.radians(pitch_deg), math.radians(roll_deg)
    return [math.sin(pitch), -math.cos(pitch) * math.sin(roll), -math.cos(pitch) * math.cos(roll)]


def _env(gravity):
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        projected_gravity_b=torch.tensor(gravity, dtype=torch.float32),
    ))})


def test_posture_does_not_select_a_pitch_angle():
    env = _env([_gravity(pitch) for pitch in (-40, -20, 0, 20, 40)])
    assert torch.allclose(rewards.roll_orientation_l2(env), torch.zeros(5))
    assert torch.allclose(rewards.roll_upward(env), torch.ones(5))


def test_roll_penalty_is_not_weakened_by_pitch():
    env = _env([_gravity(pitch, 20) for pitch in (-40, 0, 40)])
    expected_error = torch.full((3,), math.sin(math.radians(20)) ** 2)
    expected_upward = torch.full((3,), math.cos(math.radians(20)))
    assert torch.allclose(rewards.roll_orientation_l2(env), expected_error, atol=1e-6)
    assert torch.allclose(rewards.roll_upward(env), expected_upward, atol=1e-6)


def test_compliant_pitch_target_matches_tact_moment_formula_and_retains_penalty():
    masses = torch.tensor([0.0, 10.0, 15.0])
    com_x = torch.full((3,), -0.17)
    expected_targets = 0.5 * masses * 9.81 * com_x / 50.0
    target_degrees = torch.rad2deg(expected_targets)
    assert torch.allclose(target_degrees, torch.tensor([0.0, -9.555, -14.333]), atol=0.01)

    # The reward is minimized at the load-dependent target, not at zero pitch;
    # the same original posture reward also penalizes deviations around it.
    gravity_at_target = torch.tensor([_gravity(float(angle)) for angle in target_degrees])
    orientation, upward, actual_targets = rewards.compliant_posture_terms(
        gravity_at_target, masses, com_x,
    )
    assert torch.allclose(actual_targets, expected_targets, atol=1e-6)
    assert torch.allclose(orientation, torch.zeros(3), atol=1e-6)
    assert torch.allclose(upward, torch.ones(3), atol=1e-6)

    gravity_10deg_away = torch.tensor([_gravity(float(angle + 10)) for angle in target_degrees])
    off_target, _, _ = rewards.compliant_posture_terms(gravity_10deg_away, masses, com_x)
    assert torch.all(off_target > 0.0)


def test_severe_tilt_catches_backpack_supported_and_sideways_falls():
    gravity = torch.tensor([
        _gravity(0), _gravity(40), _gravity(-40), _gravity(70),
        _gravity(-70), _gravity(0, 70), [0, 0, 1], [0, 0, -1.000001],
    ])
    assert rewards.excessive_tilt(gravity, math.radians(60)).tolist() == [
        False, False, False, True, True, True, True, False,
    ]


@pytest.mark.parametrize("threshold", [0, -1, math.pi / 2, math.pi])
def test_invalid_fall_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        rewards.excessive_tilt(torch.tensor([_gravity(0)]), threshold)


class _BaseEnv:
    def _get_dones(self):
        return self.contacts, self.timeouts


tree = ast.parse((PACKAGE / "env.py").read_text(encoding="utf-8"))
env_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
namespace = {"BaseEnv": _BaseEnv, "excessive_tilt": rewards.excessive_tilt, "ground_contact": rewards.ground_contact}
exec(compile(ast.Module(body=env_classes, type_ignores=[]), str(PACKAGE / "env.py"), "exec"), namespace)


def test_contact_and_tilt_are_ored_but_timeout_is_not_a_fall():
    env = namespace["BackpackRPOFlatEnv"]()
    env.contacts = torch.tensor([True, False, False, True])
    env.timeouts = torch.tensor([False, False, True, False])
    env.robot = _env([_gravity(0), _gravity(70), _gravity(0), _gravity(70)]).scene["robot"]
    env.scene = {
        name: SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=torch.zeros(4, 3, 1, 1, 3)))
        for name in ("backpack_ground_contact", "pelvis_ground_contact")
    }
    env.cfg = SimpleNamespace(backpack_fall_tilt_rad=math.radians(60))
    terminated, timeouts = env._get_dones()
    assert terminated.tolist() == [True, True, False, True]
    assert timeouts.tolist() == [False, False, True, False]
    assert env.backpack_contact_fall.tolist() == [True, False, False, True]
    assert env.backpack_tilt_fall.tolist() == [False, True, False, True]


def test_filtered_ground_force_is_terminal_but_self_contact_is_not():
    env = namespace["BackpackRPOFlatEnv"]()
    env.contacts = torch.tensor([False, False, False])
    env.timeouts = torch.tensor([False, False, False])
    env.robot = _env([_gravity(0)] * 3).scene["robot"]
    env.cfg = SimpleNamespace(backpack_fall_tilt_rad=math.radians(60))
    env.scene = {
        name: SimpleNamespace(data=SimpleNamespace(
            force_matrix_w_history=torch.zeros(3, 3, 1, 1, 3),
            net_forces_w_history=torch.full((3, 3, 1, 3), 100.0),
        )) for name in ("backpack_ground_contact", "pelvis_ground_contact")
    }
    env.scene["backpack_ground_contact"].data.force_matrix_w_history[1, 0, 0, 0, 2] = 2
    env.scene["pelvis_ground_contact"].data.force_matrix_w_history[2, 1, 0, 0, 2] = 2
    terminated, _ = env._get_dones()
    assert terminated.tolist() == [False, True, True]


def _checkpoint(tmp_path, design):
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    import json
    (run / "params/design.json").write_text(json.dumps(design), encoding="utf-8")
    return str(run / "model_100.pt")


def test_evaluation_uses_saved_adaptive_profile(tmp_path):
    checkpoint = _checkpoint(tmp_path, {"backpack_profile": "adaptive"})
    assert profiles.evaluation_profile(checkpoint) == "adaptive"
    with pytest.raises(ValueError, match="mismatched"):
        profiles.evaluation_profile(checkpoint, "baseline")


def test_evaluation_uses_saved_compliant_profile(tmp_path):
    checkpoint = _checkpoint(tmp_path, {"backpack_profile": "compliant"})
    assert profiles.evaluation_profile(checkpoint) == "compliant"
    with pytest.raises(ValueError, match="mismatched"):
        profiles.evaluation_profile(checkpoint, "adaptive")


def test_legacy_baseline_metadata_is_supported(tmp_path):
    checkpoint = _checkpoint(tmp_path, {"task": "RPO-Flat"})
    assert profiles.evaluation_profile(checkpoint) == "baseline"


def test_missing_metadata_requires_explicit_evaluation_profile(tmp_path):
    checkpoint = str(tmp_path / "model.pt")
    with pytest.raises(ValueError, match="Missing"):
        profiles.evaluation_profile(checkpoint)
    assert profiles.evaluation_profile(checkpoint, "adaptive") == "adaptive"
