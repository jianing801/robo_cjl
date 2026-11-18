"""CPU contract tests; no Isaac import or claims about locomotion performance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import randomization
from randomization import (configure_profile, payload_spec, protocol_metadata,
    stairs_spec, student_noise_spec)


def reset_joints_by_scale():
    pass


def reset_joints_by_offset():
    pass


def native_config(task):
    run = task == "run"
    def event(mode="startup", **params):
        return NS(mode=mode, params=params)
    events = NS(
        physics_material=event(static_friction_range=(.3, 1.6), dynamic_friction_range=(.3, 1.2),
            restitution_range=(0., .5), num_buckets=64, make_consistent=True),
        add_base_mass=event(mass_distribution_params=(-3., 3.) if run else (-1., 1.), operation="add"),
        randomize_rigid_body_com=event(com_range={"x": (-.03, .03) if run else (-.025, .025)}),
        scale_link_mass=event(mass_distribution_params=(.8, 1.2) if run else (.9, 1.1), operation="scale"),
        scale_actuator_gains=event(stiffness_distribution_params=(.8, 1.2) if run else (.9, 1.1),
            damping_distribution_params=(.8, 1.2) if run else (.9, 1.1), operation="scale"),
        scale_joint_parameters=event(friction_distribution_params=(1., 1.),
            armature_distribution_params=(.8, 1.2), operation="scale"),
        reset_base=event("reset", pose_range={"x": (-.5, .5), "y": (-.5, .5), "yaw": (-3.14, 3.14)},
            velocity_range={"x": (-.2, .2) if run else (-.5, .5), "yaw": (-.2, .2) if run else (-.78, .78)}),
        reset_robot_joints=event("reset", position_range=(.8, 1.2) if run else (-.15, .15), velocity_range=(0., 0.)),
        push_robot=event("interval", velocity_range={"x": (-.5, .5), "yaw": (-1., 1.) if run else (-.78, .78)}),
    )
    events.reset_robot_joints.func = reset_joints_by_scale if run else reset_joints_by_offset
    events.push_robot.interval_range_s = (5., 10.) if run else (10., 15.)
    if run:
        events.base_external_force_torque = event("reset", force_range=(0., 0.), torque_range=(0., 0.))
    command = NS(rel_standing_envs=.02 if run else .2, rel_heading_envs=1.,
        resampling_time_range=(10., 10.), heading_command=True,
        ranges=NS(lin_vel_x=(-.5, 2.5) if run else (-.6, 1.), lin_vel_y=(-.5, .5),
            ang_vel_z=(-1.5, 1.5), heading=(-3.14, 3.14)))
    return NS(seed=2026, events=events, noise=NS(add_noise=True),
        observations=NS(policy=NS(enable_corruption=True)),
        commands=NS(base_velocity=command) if run else command,
        scene=NS(terrain=NS(marker="original terrain"), robot=NS(actuators={
            "knees": NS(min_delay=0, max_delay=2, stiffness=150., damping=5., armature=.01),
            "other": NS(stiffness=40., damping=2.),
        })))


@pytest.mark.parametrize("task", ["stairs", "backpack", "run"])
def test_nominal_reproduces_legacy_overrides_and_never_mutates_source(task):
    source = native_config(task)
    before = deepcopy(source)
    cfg, audit = configure_profile(source, task, profile="nominal", seed=7, speed=2.2, reset_yaw_range=.07)
    assert source == before
    assert cfg.seed == 7 and source.seed == 2026
    for name in ("physics_material", "add_base_mass", "scale_link_mass", "push_robot"):
        assert getattr(cfg.events, name) is None
    assert cfg.events.reset_base.params["pose_range"] == {"x": (-.01, .01), "y": (-.01, .01), "yaw": (-.07, .07)}
    assert all(value == (0., 0.) for value in cfg.events.reset_base.params["velocity_range"].values())
    assert cfg.events.reset_robot_joints.params["position_range"] == ((.99, 1.01) if task == "run" else (-.01, .01))
    assert cfg.scene.robot.actuators["knees"].max_delay == 0
    command = cfg.commands.base_velocity if task == "run" else cfg.commands
    assert command.ranges.lin_vel_x == ((2.2, 2.2) if task == "run" else (.5, .5))
    assert command.resampling_time_range == (1000., 1000.)
    assert command.rel_standing_envs == 0.
    assert not any(audit["student_noise"]["frame_amplitudes"])
    assert audit["delay_units"] == "actuator_buffer_steps"
    assert cfg.scene.terrain == source.scene.terrain
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("task,mass,gain,speed,standing", [
    ("stairs", (-1., 1.), (.9, 1.1), (.2, .6), 0.),
    ("backpack", (-1., 1.), (.9, 1.1), (.2, .8), 0.),
    ("run", (-3., 3.), (.8, 1.2), (1.5, 2.5), 0.),
])
def test_robust_preserves_task_dynamics_and_uses_declared_collection_commands(task, mass, gain, speed, standing):
    source = native_config(task)
    before = deepcopy(source)
    cfg, audit = configure_profile(source, task)
    assert source == before
    assert cfg.events.add_base_mass.params["mass_distribution_params"] == mass
    assert cfg.events.scale_actuator_gains.params["stiffness_distribution_params"] == gain
    assert cfg.events.push_robot == source.events.push_robot
    if task == "stairs":
        expected_reset = deepcopy(source.events.reset_base)
        expected_reset.params["pose_range"]["yaw"] = (-.15, .15)
        assert cfg.events.reset_base == expected_reset
    else:
        assert cfg.events.reset_base == source.events.reset_base
    assert cfg.events.reset_robot_joints == source.events.reset_robot_joints
    assert cfg.scene.robot.actuators == source.scene.robot.actuators
    assert cfg.scene.robot.actuators is not source.scene.robot.actuators
    command = cfg.commands.base_velocity if task == "run" else cfg.commands
    assert command.ranges.lin_vel_x == speed
    assert command.ranges.lin_vel_y == (0., 0.)
    assert command.rel_standing_envs == standing
    assert command.resampling_time_range == (8., 10.)
    assert command.ranges.heading == ((0., 0.) if task == "stairs" else (-.3, .3))
    assert (cfg.observations.policy.enable_corruption if task == "run" else cfg.noise.add_noise) is False
    assert audit["commands"]["source"] == "v2_continuous_locomotion"
    assert audit["commands"]["standing_commands"] is False


@pytest.mark.parametrize("speed", [0., .05, -1.])
def test_nominal_running_cannot_reintroduce_standing_commands(speed):
    with pytest.raises(ValueError, match="positive moving speed"):
        configure_profile(native_config("run"), "run", profile="nominal", speed=speed)


@pytest.mark.parametrize("task", ["stairs", "backpack", "run"])
@pytest.mark.parametrize("strength", [.25, .5, .75, 1.])
def test_every_curriculum_stage_disables_standing_without_mutating_teachers(task, strength):
    source = native_config(task)
    before = deepcopy(source)
    cfg, audit = configure_profile(source, task, profile="robust", strength=strength)
    command = cfg.commands.base_velocity if task == "run" else cfg.commands
    assert source == before
    assert command.rel_standing_envs == 0.
    assert command.ranges.lin_vel_x[0] > .05
    assert audit["commands"]["standing_commands"] is False


def test_partial_strength_respects_zero_and_one_centers_and_preserves_event_frequency():
    source = native_config("run")
    cfg, audit = configure_profile(source, "run", strength=.5)
    assert cfg.events.add_base_mass.params["mass_distribution_params"] == (-1.5, 1.5)
    assert cfg.events.randomize_rigid_body_com.params["com_range"]["x"] == (-.015, .015)
    assert cfg.events.scale_link_mass.params["mass_distribution_params"] == (.9, 1.1)
    assert cfg.events.scale_actuator_gains.params["damping_distribution_params"] == (.9, 1.1)
    assert cfg.events.scale_joint_parameters.params["armature_distribution_params"] == (.9, 1.1)
    assert cfg.events.scale_joint_parameters.params["friction_distribution_params"] == (1., 1.)
    material = cfg.events.physics_material.params
    assert material["static_friction_range"] == pytest.approx((.65, 1.3))
    assert material["dynamic_friction_range"] == pytest.approx((.65, 1.1))
    assert material["restitution_range"] == (0., .25)
    assert material["make_consistent"] is True and material["num_buckets"] == 64
    assert cfg.events.push_robot.params["velocity_range"]["yaw"] == (-.5, .5)
    assert cfg.events.push_robot.interval_range_s == source.events.push_robot.interval_range_s
    assert cfg.events.reset_robot_joints.params["position_range"] == (.9, 1.1)
    assert cfg.scene.robot.actuators["knees"].max_delay == 1
    assert audit["delays"]["knees"] == dict(native=(0, 2), applied=(0, 1))


@pytest.mark.parametrize("task", ["stairs", "run"])
def test_zero_strength_has_identity_dynamics_but_keeps_operating_condition_envelope(task):
    cfg, audit = configure_profile(native_config(task), task, strength=0.)
    assert cfg.events.add_base_mass.params["mass_distribution_params"] == (0., 0.)
    assert cfg.events.scale_link_mass.params["mass_distribution_params"] == (1., 1.)
    assert cfg.events.physics_material.params["static_friction_range"] == (1., 1.)
    assert cfg.events.physics_material.params["restitution_range"] == (0., 0.)
    assert cfg.events.reset_robot_joints.params["position_range"] == ((1., 1.) if task == "run" else (0., 0.))
    assert cfg.scene.robot.actuators["knees"].max_delay == 0
    assert not any(audit["student_noise"]["frame_amplitudes"])
    assert audit["operating_conditions_scaled_by_strength"] is False


@pytest.mark.parametrize("strength", [0., .5, 1.])
def test_stair_reset_heading_stays_near_ascent_direction_and_audit_matches(strength):
    source = native_config("stairs")
    cfg, audit = configure_profile(source, "stairs", strength=strength)
    pose = cfg.events.reset_base.params["pose_range"]
    assert pose["yaw"] == (-.15 * strength, .15 * strength)
    assert pose["x"] == (-.5 * strength, .5 * strength)
    assert pose["y"] == (-.5 * strength, .5 * strength)
    assert audit["reset"]["base"]["pose_range"] == pose
    assert source.events.reset_base.params["pose_range"]["yaw"] == (-3.14, 3.14)


def test_explicit_delay_uses_integer_actuator_steps_without_double_scaling():
    cfg, audit = configure_profile(native_config("run"), "run", strength=.1, delay_steps=(1, 3))
    assert (cfg.scene.robot.actuators["knees"].min_delay, cfg.scene.robot.actuators["knees"].max_delay) == (1, 3)
    assert audit["delay_units"] == "actuator_buffer_steps"
    for bad in ((0., 2.), (-1, 2), (2, 1), (False, 2)):
        with pytest.raises(ValueError, match="[Dd]elay"):
            configure_profile(native_config("run"), "run", delay_steps=bad)
    with pytest.raises(ValueError, match="zero delay"):
        configure_profile(native_config("run"), "run", profile="nominal", delay_steps=(0, 1))


@pytest.mark.parametrize("strength", [-.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_strength_rejected(strength):
    with pytest.raises(ValueError, match="strength"):
        configure_profile(native_config("stairs"), "stairs", strength=strength)


def test_unknown_dynamics_and_reset_semantics_refused_without_mutating_source():
    cfg = native_config("run")
    cfg.events.scale_link_mass.params["operation"] = "abs"
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match="operation"):
        configure_profile(cfg, "run")
    assert cfg == before
    cfg = native_config("run")
    cfg.events.reset_robot_joints.func = reset_joints_by_offset
    with pytest.raises(ValueError, match="reset semantics"):
        configure_profile(cfg, "run")


def test_student_noise_is_only_sensor_slots_and_has_one_task_independent_protocol_hash():
    hashes = set()
    for task, angular in (("stairs", .2), ("backpack", .2), ("run", .35)):
        spec = student_noise_spec(task)
        noise = spec["frame_amplitudes"]
        assert len(noise) == 78
        assert noise[:3] == [angular] * 3
        assert noise[3:6] == [.05] * 3
        assert noise[6:9] == [0.] * 3
        assert noise[9:32] == [.03] * 23
        assert noise[32:55] == [1.75] * 23
        assert noise[55:] == [0.] * 23
        for strength in (0., .5, 1.):
            _, audit = configure_profile(native_config(task), task, strength=strength, seed=int(strength * 10))
            hashes.add(audit["protocol_hash"])
    assert len(hashes) == 1
    assert protocol_metadata("robust", 0.)["protocol_hash"] == protocol_metadata("nominal", 1.)["protocol_hash"]
    assert protocol_metadata()["version"] == "shared_randomization_v2"


def test_stair_strata_and_payload_envelope_stay_within_declared_native_limits():
    stairs = stairs_spec()
    assert len(stairs["strata"]) == 6
    assert {(x["step_height_m"], x["step_width_m"]) for x in stairs["strata"]} == {
        (height, width) for height in (.05, .075, .1) for width in (.25, .35)}
    assert stairs["num_rows"] == 1 and stairs["num_cols"] == 6
    assert stairs_spec("nominal")["strata"] == [dict(step_height_m=.1, step_width_m=.25)]
    payload = payload_spec()
    assert payload["mass_range_kg"] == (.1, 15.)
    assert payload["inertia_size_scale_range"] == (.9, 1.1)
    assert payload["mount_surface_offset_max_m"] == (.005, .005)
    assert payload["com_offset_max_m"] == (.015, .01, .02)
    no_variation = payload_spec(strength=0.)
    assert no_variation["mass_range_kg"] == (.1, 15.)
    assert no_variation["inertia_size_scale_range"] == (1., 1.)
    assert no_variation["com_offset_max_m"] == (0., 0., 0.)
    assert stairs["speed_sampling"] == payload["speed_sampling"] == "explicit_v1_command_range_teacher_preflight_required"


def test_protocol_binds_each_collector_source_and_rejects_missing_source(tmp_path, monkeypatch):
    for name in randomization.PROTOCOL_SOURCE_FILES:
        (tmp_path / name).write_text("# original source\n", encoding="utf8")
    monkeypatch.setattr(randomization, "__file__", str(tmp_path / "randomization.py"))
    original = protocol_metadata()
    assert set(original["source_sha256"]) == set(randomization.PROTOCOL_SOURCE_FILES)
    for name in randomization.PROTOCOL_SOURCE_FILES:
        path = tmp_path / name
        path.write_text("# updated source\n", encoding="utf8")
        updated = protocol_metadata()
        assert updated["protocol_hash"] != original["protocol_hash"]
        assert updated["source_sha256"][name] != original["source_sha256"][name]
        path.write_text("# original source\n", encoding="utf8")
    assert protocol_metadata()["protocol_hash"] == original["protocol_hash"]
    (tmp_path / "scenario_sampling.py").unlink()
    with pytest.raises(ValueError, match="Missing collection protocol source"):
        protocol_metadata()


def test_protocol_rules_declare_student_clip_and_scenario_stratification():
    assert randomization.GLOBAL_RULES["student_noise"]["student_clip"] == 100.
    assert randomization.GLOBAL_RULES["scenario"]["speed_bins"] == 3
    assert randomization.GLOBAL_RULES["commands"]["standing_commands"] is False
    assert randomization.GLOBAL_RULES["payload"]["mass_strata_kg"] == ((.1, 5.), (5., 10.), (10., 15.))
