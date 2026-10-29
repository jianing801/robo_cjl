"""AMP mixed real-URDF spawner, motor groups, observation wrapper and sampler."""
from copy import deepcopy
from pathlib import Path
import torch
from tensordict import TensorDict
from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robolab.actuators.torque_speed_envelope import TorqueSpeedEnvelopeActuator, TorqueSpeedEnvelopeActuatorCfg
from amp_layout import envelope_limit, interpolate_validated_envelope, balanced_indices
from robolab.physical_snapshot import configure_robot
from amp_checkpoint import AmpCheckpointMixin
from rsl_rl.storage import RolloutStorage
from rsl_rl.runners import OnPolicyRunner


def spawn_spatial_assets(prim_path, cfg, **kwargs):
    from robolab.spatial_spawn import spawn_without_template_undo as spawn_multi_asset
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
        # Validate unique motor curves once, then cache GPU tables/indices;
        # never re-upload identical tables every physics substep.
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


def configure_spatial(options, bank, designs, output, configurator=None):
    cfg, agent, contract = configurator(options, bank, designs[0], output)
    cache = {}
    if options.asset_cache:
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        for entry in Path(options.asset_cache).resolve().rglob(".asset_hash"):
            if (entry.parent / "config.yaml").is_file() and any(entry.parent.glob("*.usd")):
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
                if not getattr(options, 'allow_cache_misses', False):
                    raise ValueError(f"no exact converter-hash cache for {d['id']}; do not substitute another design")
        if cache and key in cache:
            # USD layers are immutable on an exact native hash cache hit.
            # Link only those layers on the same volume; configuration/hash
            # metadata stays private because the native converter may rewrite it.
            import os
            import errno
            def copy_cache_file(source, target):
                if Path(source).suffix.lower() == ".usd":
                    try:
                        os.link(source, target)
                        return target
                    except OSError as exc:
                        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
                            raise
                return shutil.copy2(source, target)
            shutil.copytree(cache[key], robot.spawn.usd_dir, copy_function=copy_cache_file)
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
    if hasattr(cfg, "scene_context"):
        cfg.scene_context.robot = mixed
    cfg.spatial_compensations = [d["root_height_compensation_m"] for d in designs]
    cfg.spatial_contexts = [d["context"] for d in designs]
    cfg.spatial_design_ids = [d["id"] for d in designs]
    if hasattr(cfg, "spatial_reference_designs"):
        cfg.spatial_reference_designs = designs
    contract.update(spatial_real_assets=True, single_actual_URDF_per_block=False,
        env_assignment="environment index modulo design count; verified at runtime", paper_discount_regularization=False)
    return cfg, agent, contract


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


class SpatialRunner(AmpCheckpointMixin, OnPolicyRunner):
    def _construct_algorithm(self, obs):
        # Explicit native constructor: in the AMP multiple-inheritance runner,
        # super() would enter legacy AMPRunner's eval()-based constructor.
        alg = OnPolicyRunner._construct_algorithm(self, obs)
        if alg.policy.is_recurrent:
            raise ValueError("recurrent policy not supported by this spatial sampler")
        alg.storage.__class__ = SpatialStorage
        alg.storage.spatial_groups = self.env.unwrapped.spatial_groups
        return alg

