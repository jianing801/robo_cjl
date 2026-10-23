"""Isaac-only mixed real-URDF Rough runtime. Import after AppLauncher."""
from copy import deepcopy
from pathlib import Path
import torch
from tensordict import TensorDict
from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robolab.actuators.torque_speed_envelope import TorqueSpeedEnvelopeActuator, TorqueSpeedEnvelopeActuatorCfg
from robolab.spatial_morphology import envelope_limit, interpolate_validated_envelope, balanced_indices
from robolab.physical_snapshot import configure_robot
from robolab.tasks.direct.base.base_env import BaseEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.runners import OnPolicyRunner


def spawn_spatial_assets(prim_path, cfg, **kwargs):
    from isaaclab.sim.spawners.wrappers.wrappers import spawn_multi_asset
    import isaaclab.sim as sim_utils
    from pxr import Sdf
    first = spawn_multi_asset(prim_path, cfg, **kwargs)
    parents = sim_utils.find_matching_prim_paths(prim_path.rsplit("/", 1)[0])
    for index, parent in enumerate(parents):
        prim = sim_utils.get_current_stage().GetPrimAtPath(parent + "/" + prim_path.rsplit("/", 1)[1])
        prim.CreateAttribute("spatial:design_id", Sdf.ValueTypeNames.String).Set(
            cfg.spatial_design_ids[index % len(cfg.spatial_design_ids)])
    return first


@configclass
class SpatialSpawnCfg(MultiAssetSpawnerCfg):
    func = spawn_spatial_assets
    spatial_design_ids: list = []


class SpatialMotor(TorqueSpeedEnvelopeActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.groups = torch.arange(len(self.computed_effort), device=self._device) % len(cfg.spatial_parameters)
        for i, p in enumerate(cfg.spatial_parameters):
            ids = self.groups == i
            self.armature[ids] = p["armature"]
            self.effort_limit[ids] = p["effort_limit"]
            self.effort_limit_sim[ids] = p["effort_limit_sim"]
            self.velocity_limit[ids] = p["velocity_limit"]
            self.velocity_limit_sim[ids] = p["velocity_limit_sim"]
        # Twenty designs use only two unique motor curves per actuator group.
        # Validate on CPU once, cache GPU tables/indices; never re-upload and
        # synchronously validate twenty identical tables every physics substep.
        import json
        unique = {}
        for i, p in enumerate(cfg.spatial_parameters):
            key = json.dumps(p["curve"])
            unique.setdefault(key, dict(table=p["curve"], groups=[]))["groups"].append(i)
        self.spatial_curves = []
        for p in unique.values():
            envelope_limit(torch.zeros(1), p["table"], 1.)
            indices = torch.tensor([i for i in range(len(self.computed_effort))
                if i % len(cfg.spatial_parameters) in p["groups"]], device=self._device, dtype=torch.long)
            table = torch.tensor(p["table"], device=self._device, dtype=self.computed_effort.dtype)
            self.spatial_curves.append((indices, table))

    def _clip_effort(self, effort):
        result = torch.empty_like(effort)
        for ids, table in self.spatial_curves:
            limit = interpolate_validated_envelope(self._joint_vel[ids], table, self.effort_limit[ids])
            result[ids] = torch.clamp(effort[ids], -limit, limit)
        return result


@configclass
class SpatialMotorCfg(TorqueSpeedEnvelopeActuatorCfg):
    class_type: type = SpatialMotor
    spatial_parameters: list = []


def configure_spatial(options, bank, designs, output):
    cfg, agent, contract = configure_rough(options, bank, designs[0], output)
    cache = {}
    if options.asset_cache:
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        for entry in Path(options.asset_cache).resolve().rglob(".asset_hash"):
            if (entry.parent / "config.yaml").is_file() and (entry.parent / "rpo.usd").is_file():
                cache.setdefault(entry.read_text().strip(), entry.parent)
    assets, robots = [], []
    for d in designs:
        robot = cfg.scene.robot.copy()
        robot.spawn.asset_path = d["urdf"]
        robot.spawn.usd_dir = str(output / "usd" / d["id"])
        if cache:
            import shutil
            key = AssetConverterBase._config_to_hash(robot.spawn)
            if key not in cache:
                raise ValueError(f"no exact converter-hash cache for {d['id']}; do not substitute another design")
            shutil.copytree(cache[key], robot.spawn.usd_dir)
            # The native converter independently checks this hash and all copied
            # USD references still resolve within the dedicated new directory.
        configure_robot(robot, bank["motor_pairs"][d["pair_id"]])
        robots.append(robot)
        assets.append(robot.spawn)
    mixed = cfg.scene.robot
    mixed.spawn = SpatialSpawnCfg(assets_cfg=assets, random_choice=False, activate_contact_sensors=True,
        spatial_design_ids=[d["id"] for d in designs])
    for name in ("knees", "feet"):
        first = robots[0].actuators[name]
        params = [dict(curve=r.actuators[name].torque_speed_lookup,
            armature=r.actuators[name].armature, effort_limit=r.actuators[name].effort_limit,
            effort_limit_sim=r.actuators[name].effort_limit_sim,
            velocity_limit=r.actuators[name].velocity_limit,
            velocity_limit_sim=r.actuators[name].velocity_limit_sim) for r in robots]
        values = deepcopy(vars(first))
        values.pop("class_type", None)
        mixed.actuators[name] = SpatialMotorCfg(**values, spatial_parameters=params)
    cfg.scene.replicate_physics = False
    cfg.scene_context.robot = mixed
    cfg.spatial_compensations = [d["root_height_compensation_m"] for d in designs]
    cfg.spatial_contexts = [d["context"] for d in designs]
    cfg.spatial_design_ids = [d["id"] for d in designs]
    contract.update(spatial_real_assets=True, single_actual_URDF_per_block=False,
        env_assignment="environment index modulo design count; verified at runtime", paper_discount_regularization=False)
    return cfg, agent, contract


def configure_rough(options, bank, design, output):
    """Source RPO-Rough only; no task router, AMP configuration or old bank."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from robolab.rough_contract import RECIPE, serializable, digest
    cfg = load_cfg_from_registry("RPO-Rough", "env_cfg_entry_point")
    source_agent = load_cfg_from_registry("RPO-Rough", "rsl_rl_cfg_entry_point").to_dict()
    before = serializable(cfg.to_dict())
    cfg.scene.num_envs = options.num_envs
    cfg.seed, cfg.sim.device = source_agent["seed"], options.device or "cuda:0"
    robot = cfg.scene.robot.copy()
    robot.spawn.asset_path, robot.spawn.usd_dir = design["urdf"], str(output/"usd")
    configure_robot(robot, bank["motor_pairs"][design["pair_id"]])
    state = robot.init_state
    state.pos = (*state.pos[:2], state.pos[2]+design["root_height_compensation_m"])
    cfg.scene.robot = robot
    cfg.scene_context.num_envs, cfg.scene_context.robot = options.num_envs, robot
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.commands.debug_vis = False
    for sensor in ("left_feet_scanner", "right_feet_scanner", "height_scanner"):
        getattr(cfg.scene, sensor).debug_vis = False
    if bank["training"].get("stairs_height_range_m") != [.05, .10]:
        raise ValueError("only the authorized 5-10cm stair range is supported")
    for name, terrain in cfg.scene.terrain.terrain_generator.sub_terrains.items():
        if "stairs" in name:
            terrain.step_height_range = (.05, .10)
    after = serializable(cfg.to_dict())
    protected = ("rewards", "reward", "commands", "events", "observations", "noise", "robot",
                 "normalization", "terminations", "episode_length_s", "decimation")
    for key in protected:
        expected = deepcopy(before.get(key))
        if key == "commands":
            expected["debug_vis"] = False
        if expected != after.get(key):
            raise RuntimeError(f"Rough adapter changed protected source setting: {key}")
    contract = dict(recipe=RECIPE, task="stairs", source_task="RPO-Rough",
        source_environment_sha256=digest(before), source_agent_sha256=digest(source_agent),
        protected_source_settings={k: after[k] for k in protected if k in after},
        context_width=len(design["context"]), context_is_known_design_not_sensor=True,
        stairs_height_range_m=[.05, .10], resolved_terrain=after["scene"]["terrain"],
        display_only_overrides=["remote terrain MDL and sky HDR removed", "command/raycaster debug disabled"])
    return cfg, source_agent, contract


class SpatialRoughEnv(BaseEnv):
    def _setup_scene(self):
        # InteractiveScene already creates each heterogeneous environment before
        # spawning its assets. BaseEnv's extra single-source clone would replace
        # the differing robot definitions with env_0. Never call it here.
        self.robot = self.scene["robot"]
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def init_buffers(self):
        # Called before the original first reset, after physical startup events.
        groups = torch.arange(self.num_envs, device=self.device) % len(self.cfg.spatial_compensations)
        compensation = torch.tensor(self.cfg.spatial_compensations, device=self.device)
        self.robot.data.default_root_state[:, 2] += compensation[groups] - compensation[0]
        self.spatial_groups = groups
        self.spatial_context = torch.tensor(self.cfg.spatial_contexts, device=self.device)[groups]
        super().init_buffers()


class SpatialVecEnv(RslRlVecEnvWrapper):
    def __init__(self, env, source_symmetry, clip_actions=None):
        super().__init__(env, clip_actions=clip_actions)
        self.native_source_symmetry = source_symmetry
        self.unwrapped.native_source_symmetry = source_symmetry

    def _extend(self, obs):
        result = obs.clone() if isinstance(obs, TensorDict) else TensorDict(obs, batch_size=[self.num_envs])
        for key in ("policy", "critic"):
            result[key] = torch.cat((result[key], self.unwrapped.spatial_context), -1)
        return result

    def get_observations(self):
        return self._extend(super().get_observations())

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        # Every design contributes exactly its live environment count each step.
        extras.setdefault("log", {})
        for i, name in enumerate(self.unwrapped.cfg.spatial_design_ids):
            ids = self.unwrapped.spatial_groups == i
            extras["log"][f"Spatial/{name}/reward"] = rewards[ids].mean()
            extras["log"][f"Spatial/{name}/forward_velocity"] = self.unwrapped.robot.data.root_lin_vel_b[ids, 0].mean()
        return self._extend(obs), rewards, dones, extras


class SpatialStorage(RolloutStorage):
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("spatial storage requires feedforward PPO")
        names = ("observations", "actions", "values", "advantages", "returns", "actions_log_prob", "mu", "sigma")
        arrays = [getattr(self, name).flatten(0, 1) for name in names]
        for _ in range(num_epochs):
            for indices in balanced_indices(self.spatial_groups, self.num_transitions_per_env,
                                            num_mini_batches, device=self.device):
                yield (*[x[indices] for x in arrays], (None, None), None)


class SpatialRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs):
        alg = super()._construct_algorithm(obs)
        if alg.policy.is_recurrent:
            raise ValueError("recurrent policy not supported by this spatial sampler")
        alg.storage.__class__ = SpatialStorage
        alg.storage.spatial_groups = self.env.unwrapped.spatial_groups
        return alg

    def save(self, path, infos=None):
        path = Path(path)
        if path.exists():
            if getattr(self, "last_saved", None) == (str(path), self.current_learning_iteration):
                return
            raise FileExistsError(path)
        state = dict(model_state_dict=self.alg.policy.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(), iter=self.current_learning_iteration,
            learning_rate=self.alg.learning_rate, spatial_identity=self.spatial_identity,
            torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all())
        with path.open("xb") as stream:
            torch.save(state, stream)
        self.last_saved = str(path), self.current_learning_iteration

    def restore_spatial(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("spatial_identity") != self.spatial_identity:
            raise ValueError("not this spatial experiment; serial checkpoints cannot silently resume")
        self.alg.policy.load_state_dict(state["model_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.alg.learning_rate = state["learning_rate"]
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        self.current_learning_iteration = state["iter"] + 1
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_states"]])
