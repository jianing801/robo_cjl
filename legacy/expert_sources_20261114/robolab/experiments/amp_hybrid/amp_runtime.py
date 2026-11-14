"""AMP with grouped native motion loaders; morphology never enters disc inputs."""
from copy import deepcopy
from pathlib import Path
import json
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporter
import isaaclab.sim as sim_utils
from robolab.tasks.manager_based.amp.amp_env import AmpEnv
from robolab.tasks.manager_based.amp.managers import MotionDataManager, MotionDataTerm, AnimationManager, AnimationTerm
from reference_router import ReferenceRouter
from rsl_rl.runners import AMPRunner
from rsl_rl.algorithms import PPOAMP
from rsl_rl.storage import CircularBuffer
from rsl_rl.modules import resolve_amp_config
from amp_mixed_runtime import SpatialRunner


class OfflineAmpPlane(TerrainImporter):
    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        path = self.cfg.prim_path+"/"+name
        if path in self.terrain_prim_paths:
            raise ValueError("duplicate plane")
        self.terrain_prim_paths.append(path)
        asset = Path(__file__).with_name("native_ground.usda")
        cfg = sim_utils.GroundPlaneCfg(usd_path=str(asset), physics_material=self.cfg.physics_material, size=size, color=None)
        cfg.func(path, cfg)


class SpatialMotionManager(MotionDataManager):
    def _prepare_terms(self):
        if set(vars(self.cfg)) != {"motion_dataset"}:
            raise ValueError("spatial AMP currently requires the source single motion_dataset term")
        library = json.loads(Path(self._env.cfg.spatial_reference_library).read_text(encoding="utf8"))
        directories, terms, design_group = {}, [], []
        for design in self._env.cfg.spatial_reference_designs:
            directory = library["designs"][design["id"]]["directory"]
            if directory not in directories:
                cfg = deepcopy(self.cfg.motion_dataset)
                cfg.motion_data_dir = directory
                directories[directory] = len(terms)
                terms.append(MotionDataTerm(cfg, self._env))
            design_group.append(directories[directory])
        env_groups = [design_group[i % len(design_group)] for i in range(self._env.num_envs)]
        self._terms["motion_dataset"] = ReferenceRouter(terms, env_groups, self._env.device)
        self._term_cfgs["motion_dataset"] = self.cfg.motion_dataset


class SpatialAnimationTerm(AnimationTerm):
    def reset(self, env_ids=None):
        if env_ids is None or len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = self.motion_data_term.sample_motions_for_envs(env_ids)
        self.motion_durations[env_ids] = self.motion_data_term.get_motion_durations(self.motion_ids[env_ids])
        truncate = self.num_steps*self._env.step_dt
        kwargs = dict(truncate_time_end=truncate) if self.cfg.num_steps_to_use > 0 else dict(truncate_time_start=truncate)
        self.motion_fetch_time[env_ids, 0] = (self.motion_data_term.sample_times(self.motion_ids[env_ids], **kwargs)
                                           if self.cfg.random_initialize else 0.)
        if self.num_steps > 1:
            self.motion_fetch_time[env_ids, 1:] = self.motion_fetch_time[env_ids, 0:1]+self.step_indices[1:].float()*self._env.step_dt
        self._fetch_motion_data(env_ids)


class SpatialAnimationManager(AnimationManager):
    def _prepare_terms(self):
        for name, cfg in vars(self.cfg).items():
            if cfg is not None:
                self._terms[name] = SpatialAnimationTerm(cfg, self._env)
                self._term_cfgs[name] = cfg


class SpatialAmpEnv(AmpEnv):
    def load_managers(self):
        self.robot = self.scene["robot"]
        self.spatial_groups = torch.arange(self.num_envs, device=self.device) % len(self.cfg.spatial_contexts)
        compensation = torch.tensor(self.cfg.spatial_compensations, device=self.device)
        self.robot.data.default_root_state[:, 2] += compensation[self.spatial_groups]-compensation[0]
        self.spatial_context = torch.tensor(self.cfg.spatial_contexts, device=self.device)[self.spatial_groups]
        self.motion_data_manager = SpatialMotionManager(self.cfg.motion_data, self)
        self.animation_manager = SpatialAnimationManager(self.cfg.animation, self)
        ManagerBasedRLEnv.load_managers(self)


class HybridPPOAMP(PPOAMP):
    def __init__(self, policy, storage, *, amp_cfg, device="cpu", **kwargs):
        buffers = [CircularBuffer(amp_cfg["disc_obs_buffer_size"], storage.num_envs, device) for _ in range(2)]
        super().__init__(policy, storage, *buffers, amp_cfg=amp_cfg, device=device, **kwargs)


class SpatialAMPRunner(SpatialRunner, AMPRunner):
    def _construct_algorithm(self, obs):
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        return SpatialRunner._construct_algorithm(self, obs)

_PRISTINE_AMP_CONFIG = None


def configure_amp(options, bank, design, output):
    """Preserve native AMP learning settings; replace only robot/reference/display."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from amp_support import serializable
    from robolab.physical_snapshot import configure_robot
    global _PRISTINE_AMP_CONFIG
    if _PRISTINE_AMP_CONFIG is None:
        _PRISTINE_AMP_CONFIG = deepcopy(load_cfg_from_registry("RPO-AMP", "env_cfg_entry_point"))
    cfg = deepcopy(_PRISTINE_AMP_CONFIG)
    agent = load_cfg_from_registry("RPO-AMP", "rsl_rl_cfg_entry_point").to_dict()
    before = serializable(cfg.to_dict())
    cfg.scene.num_envs = options.num_envs
    cfg.seed, cfg.sim.device = agent["seed"], options.device or "cuda:0"
    robot = cfg.scene.robot.copy()
    robot.spawn.asset_path, robot.spawn.usd_dir = design["urdf"], str(output/"usd")
    configure_robot(robot, bank["motor_pairs"][design["pair_id"]])
    robot.init_state.pos = (*robot.init_state.pos[:2],
                            robot.init_state.pos[2]+design["root_height_compensation_m"])
    cfg.scene.robot = robot
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.class_type = OfflineAmpPlane
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.robot_anim = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.animation.animation.enable_visualization = False
    cfg.spatial_reference_library = options.reference_library
    # configure_spatial replaces this with only current-group designs.
    cfg.spatial_reference_designs = []
    from amp_scene_contract import align_merged_base_events
    base_selector_remap = align_merged_base_events(cfg)
    after = serializable(cfg.to_dict())
    protected = ("rewards", "events", "observations", "terminations", "decimation", "episode_length_s")
    for key in protected:
        expected = deepcopy(before.get(key))
        if key == 'events':
            for name, remap in base_selector_remap.items():
                expected[name]['params']['asset_cfg']['body_names'] = remap['after']
        if expected != after.get(key):
            raise ValueError(f"AMP adapter changed protected source setting {key}")
    commands = deepcopy(before["commands"])
    commands["base_velocity"]["debug_vis"] = False
    if commands != after["commands"]:
        raise ValueError("AMP command changes beyond debug display")
    return cfg, agent, dict(task="run", source_task="RPO-AMP",
        original_source=before, native_agent=serializable(agent),
        protected_source_settings={k:after.get(k) for k in protected},
        references_per_physical_design=True, morphology_not_in_discriminator=True,
        merged_base_selector_remap=base_selector_remap,
        no_reward_command_learning_parameter_tuning=True)

