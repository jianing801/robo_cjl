"""Isaac-only independent native environments. Import after AppLauncher.

No shared task-routing environment, cross-task reward manager or evaluation
configuration is used here. No training starts on import.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import torch
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robolab.native_expert_contract import TASKS, RECIPE, serializable, config_digest
from robolab.native_expert_policy import append_context
from robolab.physical_snapshot import configure_robot


def configure_native(options, bank, design, output):
    task_id = TASKS[options.task]
    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    source_agent = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point").to_dict()
    before = serializable(cfg.to_dict())
    cfg.scene.num_envs = options.num_envs
    cfg.seed, cfg.sim.device = source_agent["seed"], options.device or "cuda:0"
    robot = cfg.scene.robot.copy()
    robot.spawn.asset_path, robot.spawn.usd_dir = design["urdf"], str(output / "usd")
    configure_robot(robot, bank["motor_pairs"][design["pair_id"]])
    # Geometric ground placement is the only reset adaptation. Joint posture
    # and all reset distributions remain the source task's own definitions.
    state = robot.init_state
    state.pos = (*state.pos[:2], state.pos[2] + design["root_height_compensation_m"])
    cfg.scene.robot = robot
    reference_record = None
    if bank.get("training", {}).get("retargeted_references"):
        if options.task != "run":
            raise ValueError("RUN reference comparison must not be applied to STAIRS")
        from robolab.amp_reference_retarget import approved_dataset
        reference_record = approved_dataset(bank, design, cfg.motion_data.motion_dataset.motion_data_weights)
        cfg.motion_data.motion_dataset.motion_data_dir = reference_record["directory"]
    if options.task == "run":
        from robolab.native_ground import OfflineNativePlane
        cfg.scene.terrain.class_type = OfflineNativePlane
        cfg.scene.terrain.visual_material = None
        cfg.scene.sky_light.spawn.texture_file = None
        cfg.commands.base_velocity.debug_vis = False
    if options.task == "stairs":
        cfg.scene_context.num_envs = options.num_envs
        cfg.scene_context.robot = robot
        cfg.scene.terrain.visual_material = None
        cfg.scene.sky_light.spawn.texture_file = None
        cfg.commands.debug_vis = False
        for sensor in ("left_feet_scanner", "right_feet_scanner", "height_scanner"):
            getattr(cfg.scene, sensor).debug_vis = False
        cap = bank.get("training", {}).get("stairs_height_range_m")
        if cap is not None:
            if cap != [0.05, 0.10]:
                raise ValueError("only the user's frozen 5-10cm stair override is authorized")
            for name, terrain in cfg.scene.terrain.terrain_generator.sub_terrains.items():
                if "stairs" in name:
                    terrain.step_height_range = tuple(cap)
    after = serializable(cfg.to_dict())
    # Strong non-interference check: these may NEVER be changed by this adapter.
    protected = ("rewards", "reward", "commands", "events", "observations", "noise",
                 "robot", "normalization", "terminations", "episode_length_s", "decimation")
    for key in protected:
        expected = deepcopy(before.get(key))
        if options.task == "run" and key == "commands":
            expected["base_velocity"]["debug_vis"] = False
        if options.task == "stairs" and key == "commands":
            expected["debug_vis"] = False
        if expected != after.get(key):
            raise RuntimeError(f"native adapter changed protected source configuration: {key}")
    contract = {"recipe": RECIPE, "task": options.task, "source_task": task_id,
        "source_environment_sha256": config_digest(before),
        "source_agent_sha256": config_digest(source_agent),
        "protected_source_settings": {k: after[k] for k in protected if k in after},
        "context_width": len(design["context"]),
        "context_is_known_design_not_sensor": True,
        "source_symmetry_preserved_with_design_suffix_adapter": True}
    if options.task == "stairs":
        contract["display_only_overrides"] = ["remote terrain MDL and sky HDR removed", "command/raycaster debug disabled"]
        contract["stairs_height_range_m"] = bank.get("training", {}).get("stairs_height_range_m")
        contract["resolved_terrain"] = after["scene"]["terrain"]
    if options.task == "run":
        contract["offline_scene"] = {"plane": "Z=0 normal +Z; original GroundPlaneCfg spawning and original physics material",
                                     "external_ground_USD_replaced": True, "external_visual_textures_removed": True,
                                     "online_command_debug_arrow_disabled_only": True}
    if options.task == "run":
        motion = cfg.motion_data.motion_dataset
        contract["motion_files"] = {name: {"weight": weight,
            "sha256": hashlib.sha256((Path(motion.motion_data_dir) / (name + ".pkl")).read_bytes()).hexdigest()}
            for name, weight in motion.motion_data_weights.items()}
        if reference_record is not None:
            contract["retargeted_reference_binding"] = {
                "design_id": design["id"], "urdf_sha256": design["urdf_sha256"],
                "library": bank["training"]["retargeted_references"],
                "directory": reference_record["directory"],
                "review_sha256": reference_record["review_sha256"],
                "dynamic_feasibility_validated": False}
    return cfg, source_agent, contract


class NativeMorphVecEnv(RslRlVecEnvWrapper):
    def __init__(self, env, *, context, source_symmetry, clip_actions=None):
        self.native_context = torch.as_tensor(context, device=env.device, dtype=torch.float32)
        self.native_source_symmetry = source_symmetry
        super().__init__(env, clip_actions=clip_actions)

    def get_observations(self):
        return append_context(super().get_observations(), self.native_context)

    def reset(self):
        obs, extras = super().reset()
        return append_context(obs, self.native_context), extras

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        return append_context(obs, self.native_context), rewards, dones, extras


def actual_physics(raw):
    view = raw.scene["robot"].root_physx_view
    return {"body_names": list(raw.scene["robot"].body_names),
            "joint_names": list(raw.scene["robot"].joint_names),
            "masses_kg": view.get_masses().cpu().tolist(),
            "inertia_tensors": view.get_inertias().cpu().tolist(),
            "com_positions_quaternions": view.get_coms().cpu().tolist(),
            "joint_armature": raw.scene["robot"].data.joint_armature.cpu().tolist(),
            "source_task_startup_randomization_retained": True}
