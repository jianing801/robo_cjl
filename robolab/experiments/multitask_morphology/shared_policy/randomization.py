"""Versioned collection profiles, independent of Isaac Lab imports.

The native teacher configuration is the source of dynamics ranges.  This module
copies it before changing anything.  Student observation noise is a separate
channel: both profiles leave the teacher's native observations uncorrupted.
Terrain construction and payload sampling belong to the collector; the pure
specifications below describe continuous-locomotion operating envelopes requiring a
teacher preflight, without claiming condition-dependent speed adaptation.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


PROFILE_VERSION = "shared_randomization_v2"
PROFILES = ("nominal", "robust")
TASKS = ("stairs", "backpack", "run")
FRAME_WIDTH = 78
PROTOCOL_SOURCE_FILES = ("randomization.py", "student_observations.py", "scenario_sampling.py",
    "rollout.py", "rollout_amp.py", "stair_course.py")
COMMAND_RULES = {
    "stairs": dict(lin_vel_x=(.2, .6), heading=(0., 0.), standing=0.),
    "backpack": dict(lin_vel_x=(.2, .8), heading=(-.3, .3), standing=0.),
    "run": dict(lin_vel_x=(1.5, 2.5), heading=(-.3, .3), standing=0.),
}
GLOBAL_RULES = dict(
    native_events="inherit_task_ranges_and_scale_on_a_deep_copy",
    event_centers=dict(add=0., com=0., scale=1., material_friction=1., restitution=0., push=0.),
    reset_centers=dict(root_pose=0., root_velocity=0., joints_offset=0., joints_scale=1., joint_velocity=0.),
    stairs_heading_reset=.15,
    native_teacher_noise=False,
    student_noise=dict(stairs_ang_vel=.2, backpack_ang_vel=.2, run_ang_vel=.35,
        projected_gravity=.05, joint_pos=.03, joint_vel=1.75, commands=0., actions=0.,
        frame_width=FRAME_WIDTH, history="retain_old_noise_reset_per_environment", student_clip=100.),
    delay=dict(units="actuator_buffer_steps", default="native_integer_endpoints_scaled_half_up"),
    commands=dict(tasks=COMMAND_RULES, lin_vel_y=(0., 0.), ang_vel_z=(-.4, .4),
        rel_heading_envs=1., resampling_time_range=(8., 10.), source="v2_continuous_locomotion",
        standing_commands=False, scope="continuous locomotion; no stop-and-stand capability claim"),
    stairs=dict(heights_m=(.05, .075, .10), widths_m=(.25, .35),
        terrain="inverted_pyramid_stairs", sampling="six_fixed_strata"),
    payload=dict(mass_range_kg=(.1, 15.), inertia_size_scale_range=(.9, 1.1),
        mass_strata_kg=((.1, 5.), (5., 10.), (10., 15.)),
        mount_surface_offset_max_m=(.005, .005), com_offset_max_m=(.015, .010, .020)),
    scenario=dict(speed_bins=3, speed_sampling="three_moving_speed_bands_teacher_reference_required"),
    nominal=dict(disable_startup_interval=True, delay=(0, 0), teacher_noise=False,
        student_noise=False, xy_range=(-.01, .01), default_yaw_range=(-.05, .05),
        joint_offset=(-.01, .01), joint_scale=(.99, 1.01), linear_speed=.5, run_speed=2.),
)


def _validate(task, profile, strength):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if profile not in PROFILES:
        raise ValueError(f"Unknown collection profile: {profile}")
    if isinstance(strength, bool) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Randomization strength must be finite and in [0, 1]")


def protocol_metadata(profile="robust", strength=1.0):
    """Bind global rules and collector implementation, independent of task/seed.

    Missing source files are an incomplete deployment, never a weaker protocol
    identity. Read each file afresh so edits invalidate already collected data.
    """
    _validate("stairs", profile, strength)
    directory = Path(__file__).resolve().parent
    sources = {}
    for name in PROTOCOL_SOURCE_FILES:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing collection protocol source: {path}")
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    rules = dict(version=PROFILE_VERSION, rules=GLOBAL_RULES, source_sha256=sources)
    digest = hashlib.sha256(json.dumps(rules, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf8")).hexdigest()
    return dict(version=PROFILE_VERSION, profile=profile, protocol_hash=digest,
        strength=float(strength), source_sha256=sources)


def rollout_metadata(audit, requested):
    """Freeze startup identity and report applied ranges separately from CLI inputs."""
    protocol={key:deepcopy(audit[key]) for key in
        ('version','profile','protocol_hash','strength','source_sha256')}
    return dict(collection_protocol=protocol,requested_configuration=deepcopy(requested),
        applied_configuration={key:deepcopy(audit[key]) for key in
            ('task','commands','reset','student_noise','delays','events')})


def assert_protocol_unchanged(protocol):
    if protocol_metadata(protocol['profile'],protocol['strength'])!=protocol:
        raise ValueError('Collection source changed during execution; refusing mislabeled results')


def _range(values, strength, center=0.):
    if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] > values[1]:
        raise ValueError(f"Invalid randomization range: {values}")
    # Preserve exact native endpoint values at full strength, including tuples.
    if strength == 1:
        return tuple(values)
    return tuple(center + strength * (v - center) for v in values)


def _mapping(ranges, strength, center=0.):
    return {name: _range(values, strength, center) for name, values in ranges.items()}


def _event_ranges(event, strength):
    """Scale supported native dynamics parameters without changing semantics."""
    params = event.params
    handled = []
    for name in ("static_friction_range", "dynamic_friction_range"):
        if name in params:
            params[name] = _range(params[name], strength, 1.)
            handled.append(name)
    if "restitution_range" in params:
        params["restitution_range"] = _range(params["restitution_range"], strength)
        handled.append("restitution_range")
    distribution_keys = ("mass_distribution_params", "stiffness_distribution_params",
        "damping_distribution_params", "friction_distribution_params", "armature_distribution_params")
    if any(name in params for name in distribution_keys):
        operation = params.get("operation")
        if operation not in ("add", "scale"):
            raise ValueError(f"Cannot safely scale randomization operation {operation!r}")
        center = 1. if operation == "scale" else 0.
        for name in distribution_keys:
            if name in params:
                params[name] = _range(params[name], strength, center)
                handled.append(name)
    for name in ("com_range", "velocity_range"):
        if name in params:
            params[name] = _mapping(params[name], strength)
            handled.append(name)
    if not handled:
        raise ValueError("Unsupported startup/interval randomization; add explicit range semantics")
    return handled


def _delay_range(values):
    if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError("Delay bounds must be integer actuator steps")
    if not 0 <= values[0] <= values[1]:
        raise ValueError("Delay bounds must be ordered and nonnegative")
    return tuple(values)


def _reset(cfg, task, profile, strength, reset_yaw_range):
    base = cfg.events.reset_base.params
    joints = cfg.events.reset_robot_joints
    expected = "reset_joints_by_scale" if task == "run" else "reset_joints_by_offset"
    if getattr(joints.func, "__name__", None) != expected:
        raise ValueError(f"Unknown native {task} joint reset semantics")
    if profile == "nominal":
        base["pose_range"] = {key: (-.01, .01) if key in ("x", "y") else
            (-reset_yaw_range, reset_yaw_range) if key == "yaw" else (0., 0.)
            for key in base["pose_range"]}
        base["velocity_range"] = {key: (0., 0.) for key in base["velocity_range"]}
        joints.params.update(position_range=(.99, 1.01) if task == "run" else (-.01, .01),
            velocity_range=(0., 0.))
    else:
        base["pose_range"] = _mapping(base["pose_range"], strength)
        if task == "stairs":
            yaw = GLOBAL_RULES["stairs_heading_reset"] * strength
            base["pose_range"]["yaw"] = (-yaw, yaw)
        base["velocity_range"] = _mapping(base["velocity_range"], strength)
        joints.params["position_range"] = _range(joints.params["position_range"], strength,
            1. if task == "run" else 0.)
        joints.params["velocity_range"] = _range(joints.params["velocity_range"], strength)


def configure_profile(cfg, task, *, profile="robust", strength=1.0, seed=None,
                      reset_yaw_range=.05, speed=2.0, delay_steps=None):
    """Return ``(copied_cfg, audit)`` for the selected collection profile.

    ``robust`` restores native dynamics and uses explicit forward-command
    ranges, not the teacher's entire command envelope. The caller applies
    stairs/payload specifications; standing commands are disabled in both profiles.

    ``strength`` interpolates dynamics/reset/noise magnitudes, not operating
    conditions or event frequency. Native integer delay endpoints are rounded
    to the nearest integer (halves up) after scaling. ``delay_steps`` overrides
    those endpoints explicitly; these are actuator buffer steps, not policy
    control steps. ``nominal`` always uses zero delay and no student noise.
    """
    _validate(task, profile, strength)
    if not math.isfinite(reset_yaw_range) or reset_yaw_range < 0:
        raise ValueError("Reset yaw range must be finite and nonnegative")
    if not math.isfinite(speed):
        raise ValueError("Nominal speed must be finite")
    if profile == "nominal" and task == "run" and speed <= .05:
        raise ValueError("Continuous-locomotion nominal running requires a positive moving speed")
    if delay_steps is not None:
        delay_steps = _delay_range(delay_steps)
        if profile == "nominal" and delay_steps != (0, 0):
            raise ValueError("Nominal profile requires zero delay")
    result = deepcopy(cfg)
    if seed is not None:
        result.seed = seed
    if task == "run":
        result.observations.policy.enable_corruption = False
        command = result.commands.base_velocity
    else:
        result.noise.add_noise = False
        command = result.commands
    audit = dict(**protocol_metadata(profile, strength), task=task,
        native_teacher_noise=False, events={}, delays={}, delay_units="actuator_buffer_steps",
        operating_conditions_scaled_by_strength=False)
    for name, event in vars(result.events).items():
        if getattr(event, "mode", None) not in ("startup", "interval"):
            continue
        if profile == "nominal":
            setattr(result.events, name, None)
            audit["events"][name] = dict(enabled=False)
        else:
            ranges = _event_ranges(event, strength)
            audit["events"][name] = dict(enabled=True, mode=event.mode,
                ranges={key: deepcopy(event.params[key]) for key in ranges})
    _reset(result, task, profile, strength, reset_yaw_range)
    for name, motor in result.scene.robot.actuators.items():
        if not hasattr(motor, "min_delay") and not hasattr(motor, "max_delay"):
            continue
        native = _delay_range((motor.min_delay, motor.max_delay))
        actual = (0, 0) if profile == "nominal" else delay_steps
        if actual is None:
            actual = tuple(math.floor(value * strength + .5) for value in native)
        motor.min_delay, motor.max_delay = actual
        audit["delays"][name] = dict(native=native, applied=actual)
    if profile == "nominal":
        command.rel_standing_envs = 0.
        command.rel_heading_envs = 1.
        command.resampling_time_range = (1000., 1000.)
        command.ranges.lin_vel_x = (speed, speed) if task == "run" else (.5, .5)
        command.ranges.lin_vel_y = (0., 0.)
        command.ranges.ang_vel_z = (-.4, .4)
        command.ranges.heading = (0., 0.)
    else:
        rules = COMMAND_RULES[task]
        command.rel_standing_envs = rules["standing"]
        command.rel_heading_envs = 1.
        command.resampling_time_range = (8., 10.)
        command.ranges.lin_vel_x = rules["lin_vel_x"]
        command.ranges.lin_vel_y = (0., 0.)
        command.ranges.ang_vel_z = (-.4, .4)
        command.ranges.heading = rules["heading"]
    audit["reset"] = dict(base=deepcopy(result.events.reset_base.params),
        joint_position=tuple(result.events.reset_robot_joints.params["position_range"]),
        joint_velocity=tuple(result.events.reset_robot_joints.params["velocity_range"]))
    audit["commands"] = dict(ranges={key: tuple(getattr(command.ranges, key))
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")},
        resampling_time_range=tuple(command.resampling_time_range),
        rel_standing_envs=command.rel_standing_envs, rel_heading_envs=command.rel_heading_envs,
        standing_commands=False,
        source="v2_continuous_locomotion" if profile == "robust" else "legacy_nominal")
    audit["student_noise"] = student_noise_spec(task, profile=profile, strength=strength)
    return result, audit


def student_noise_spec(task, *, profile="robust", strength=1.0):
    """Uniform physical-unit noise for a single fresh 78-value sensor frame.

    Apply once when appending a frame, retain older noisy frames, clear history
    at episode reset, and never perturb command/action/task/morphology channels.
    """
    _validate(task, profile, strength)
    effective = strength if profile == "robust" else 0.
    amplitudes = {"base_ang_vel": (.35 if task == "run" else .2) * effective,
        "projected_gravity": .05 * effective, "joint_pos": .03 * effective,
        "joint_vel": 1.75 * effective}
    frame = ([amplitudes["base_ang_vel"]] * 3 + [amplitudes["projected_gravity"]] * 3 +
        [0.] * 3 + [amplitudes["joint_pos"]] * 23 + [amplitudes["joint_vel"]] * 23 + [0.] * 23)
    return dict(version=PROFILE_VERSION, task=task, profile=profile, distribution="uniform_symmetric",
        frame_width=FRAME_WIDTH, frame_layout="ang_vel3_gravity3_commands3_joint_pos23_joint_vel23_actions23",
        amplitudes=amplitudes, frame_amplitudes=frame, apply_to="fresh_student_frame_only",
        history="retain_old_noise_reset_per_environment", teacher_noise=False)


def stairs_spec(profile="robust"):
    """Fixed strata inside the actual expert's 5–10 cm stair envelope."""
    _validate("stairs", profile, 1.)
    heights = (.05, .075, .10) if profile == "robust" else (.10,)
    widths = (.25, .35) if profile == "robust" else (.25,)
    strata = [dict(step_height_m=height, step_width_m=width)
        for height in heights for width in widths]
    return dict(version=PROFILE_VERSION, profile=profile, strata=strata,
        terrain="inverted_pyramid_stairs", fixed_strata=True,
        curriculum=True, num_rows=1, num_cols=len(strata),
        speed_sampling="explicit_v1_command_range_teacher_preflight_required")


def payload_spec(profile="robust", *, strength=1.0):
    """Native final payload curriculum envelope; collector owns actual sampling."""
    _validate("backpack", profile, strength)
    effective = strength if profile == "robust" else 0.
    return dict(version=PROFILE_VERSION, profile=profile,
        mass_range_kg=(.1, 15.) if profile == "robust" else None,
        nominal_mass_points_kg=(.3, 5., 10., 15.),
        inertia_size_scale_range=_range((.9, 1.1), effective, 1.),
        mount_surface_offset_max_m=tuple(effective * value for value in (.005, .005)),
        com_offset_max_m=tuple(effective * value for value in (.015, .010, .020)),
        speed_sampling="explicit_v1_command_range_teacher_preflight_required")
