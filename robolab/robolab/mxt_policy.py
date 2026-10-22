"""Fixed-topology RPO adaptation of MxT's MTG-v2 behavior-distillation design.

References (independent PyTorch implementation, not copied JAX source):
https://arxiv.org/abs/2211.14296, Appendix A
https://github.com/frt03/mxt_bench/blob/main/mxt_bench/models/transformer.py

The 23 incoming-joint modules, base, and velocity/task goal share one feature
schema and encoder. Only actuator nodes are decoded. The original RPO action
affine transform is preserved: unlike Brax's bounded actions, RPO means are
NOT squashed by tanh. This is not a topology-search or payload implementation.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn
from rsl_rl.modules import ActorCritic

from robolab.multitask_distill import (
    ACTION_WIDTH, CANONICAL_HISTORY, FRAME_WIDTH, STUDENT_POLICY_WIDTH,
)

REPRESENTATION = "rpo_fixed_mtg_v2_v1"
POLICY_CLASS = "robolab.mxt_policy:RpoMxtActorCritic"
NODE_COUNT = ACTION_WIDTH + 2
STATIC_WIDTH = 14
BODY_SLICE = slice(0, 60)
JOINT_SLICE = slice(60, 90)
GOAL_SLICE = slice(90, 122)
STATIC_SLICE = slice(122, 136)
TYPE_SLICE = slice(136, 139)
ACTUATOR_INDEX = 139
TARGET_INDEX = 140
NODE_WIDTH = 141
DEFAULT_ARCHITECTURE = {"d_model": 256, "num_heads": 2, "num_layers": 3, "feedforward_dim": 512}


def _vector(element, attribute: str, width: int, default: str) -> list[float]:
    values = [float(v) for v in (element.get(attribute, default) if element is not None else default).split()]
    if len(values) != width or not all(math.isfinite(v) for v in values):
        raise ValueError(f"invalid URDF {attribute}")
    return values


def fixed_rpo_graph(urdf_path: str | Path, joint_names: list[str], default_joint_pos: list[float]) -> dict:
    """Read geometry only; never generate, modify, or rewrite a URDF.

    Node identity follows the simulator's actuator order, NOT XML file order.
    Geometry is origin xyz/rpy, axis xyz, limits, child mass, parent node index,
    and default joint position. A joint token represents its child module.
    """
    if len(joint_names) != ACTION_WIDTH or len(set(joint_names)) != ACTION_WIDTH:
        raise ValueError("MxT RPO requires exactly 23 unique actuator joint names")
    if len(default_joint_pos) != ACTION_WIDTH:
        raise ValueError("default joint position count mismatch")
    raw = Path(urdf_path).read_bytes()
    root = ET.fromstring(raw)
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    links = {link.get("name"): link for link in root.findall("link")}
    active = {name for name, joint in joints.items() if joint.get("type") != "fixed"}
    if active != set(joint_names):
        raise ValueError("URDF and simulator actuator joint names disagree")
    child_joint = {joint.find("child").get("link"): name for name, joint in joints.items()}
    roots = set(links) - set(child_joint)
    if len(roots) != 1:
        raise ValueError("expected one URDF kinematic root")
    root_link = next(iter(roots))
    indices = {name: index + 1 for index, name in enumerate(joint_names)}
    static = [[0.0] * STATIC_WIDTH for _ in range(NODE_COUNT)]
    mass = links[root_link].find("inertial/mass")
    static[0][11] = float(mass.get("value")) if mass is not None else 0.0
    parents = [-1]
    for name, q0 in zip(joint_names, default_joint_pos):
        joint = joints[name]
        if joint.get("type") not in ("revolute", "continuous"):
            raise ValueError(f"unsupported RPO joint type: {name}")
        parent_link = joint.find("parent").get("link")
        visited = set()
        while parent_link in child_joint and child_joint[parent_link] not in indices:
            if parent_link in visited:
                raise ValueError("cycle in URDF")
            visited.add(parent_link)
            parent_link = joints[child_joint[parent_link]].find("parent").get("link")
        parent_index = indices[child_joint[parent_link]] if parent_link in child_joint else 0
        parents.append(parent_index)
        limit = joint.find("limit")
        limits = [float(limit.get(key, "0")) for key in ("lower", "upper")] if limit is not None else [0., 0.]
        child = joint.find("child").get("link")
        mass = links[child].find("inertial/mass")
        values = (
            _vector(joint.find("origin"), "xyz", 3, "0 0 0")
            + _vector(joint.find("origin"), "rpy", 3, "0 0 0")
            + _vector(joint.find("axis"), "xyz", 3, "1 0 0")
            + limits + [float(mass.get("value")) if mass is not None else 0.0,
                        parent_index / ACTION_WIDTH, float(q0)]
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"nonfinite URDF attributes: {name}")
        static[indices[name]] = values
    parents.append(-1)  # MTG-v2 goal is a disjoint node, not a physical joint.
    return {"representation": REPRESENTATION, "urdf_sha256": hashlib.sha256(raw).hexdigest(),
            "joint_names": list(joint_names), "node_names": [root_link] + list(joint_names) + ["velocity_task_goal"],
            "parent_nodes": parents, "static_features": static}


def tokenize_rpo_history(policy_obs: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
    """782 canonical inputs -> [batch, 25 nodes, 141 semantic features].

    Bodies, joints, and goals use separate slots in the SAME shared schema.
    Zero slots mean 'not applicable', distinguished by node type/actuator flag;
    there are no padded/phantom actuators or invented sensor readings.
    """
    if policy_obs.ndim != 2 or policy_obs.shape[1] != STUDENT_POLICY_WIDTH:
        raise ValueError("MxT expects canonical ten-frame history plus two task bits")
    if static_features.shape != (NODE_COUNT, STATIC_WIDTH):
        raise ValueError("invalid fixed RPO graph attributes")
    frames = policy_obs[:, :-2].reshape(-1, CANONICAL_HISTORY, FRAME_WIDTH)
    batch = policy_obs.shape[0]
    nodes = policy_obs.new_zeros(batch, NODE_COUNT, NODE_WIDTH)
    nodes[:, 0, BODY_SLICE] = frames[:, :, :6].reshape(batch, -1)
    joints = torch.stack((frames[:, :, 9:32], frames[:, :, 32:55], frames[:, :, 55:78]), dim=-1)
    nodes[:, 1:24, JOINT_SLICE] = joints.permute(0, 2, 1, 3).reshape(batch, ACTION_WIDTH, -1)
    nodes[:, -1, GOAL_SLICE] = torch.cat((frames[:, :, 6:9].reshape(batch, -1), policy_obs[:, -2:]), dim=-1)
    nodes[:, :, STATIC_SLICE] = static_features.to(policy_obs)
    nodes[:, 0, TYPE_SLICE.start] = 1
    nodes[:, 1:24, TYPE_SLICE.start + 1] = 1
    nodes[:, -1, TYPE_SLICE.start + 2] = 1
    nodes[:, 1:24, ACTUATOR_INDEX] = 1
    nodes[:, 0, TARGET_INDEX] = 1  # Both goals target locomotion of the base.
    return nodes


def rpo_io_contract(robot, urdf_path, action_scale, step_dt, observation_scales) -> dict:
    names = list(robot.joint_names)
    q0 = robot.data.default_joint_pos[0].cpu().tolist()
    graph = fixed_rpo_graph(urdf_path, names, q0)
    return {"joint_names": names, "default_joint_pos": q0,
            "default_joint_vel": robot.data.default_joint_vel[0].cpu().tolist(),
            "action_scale": float(action_scale), "step_dt": float(step_dt),
            "urdf_sha256": graph["urdf_sha256"], "observation_scales": list(observation_scales),
            "coordinates": "body-frame angular velocity/gravity/commands; q-q0; qdot-qdot0; previous normalized action",
            "action_semantics": "q_des=q0+action_scale*action; position PD; clip=100",
            "mtg": graph}


class RpoMxtActor(nn.Module):
    def __init__(self, graph: dict, *, d_model=256, num_heads=2, num_layers=3, feedforward_dim=512):
        super().__init__()
        if graph.get("representation") != REPRESENTATION:
            raise ValueError("unknown MTG representation")
        if d_model < 1 or num_heads < 1 or d_model % num_heads or num_layers < 1 or feedforward_dim < 1:
            raise ValueError("invalid Transformer architecture")
        static = torch.tensor(graph["static_features"], dtype=torch.float32)
        if static.shape != (NODE_COUNT, STATIC_WIDTH) or not torch.isfinite(static).all():
            raise ValueError("invalid graph attributes")
        self.register_buffer("static_features", static)
        self.node_encoder = nn.Linear(NODE_WIDTH, d_model)
        self.position = nn.Parameter(torch.zeros(NODE_COUNT, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, feedforward_dim, dropout=0.0,
                                           activation="relu", batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model + NODE_WIDTH, 1)

    def forward(self, policy_obs: torch.Tensor) -> torch.Tensor:
        nodes = tokenize_rpo_history(policy_obs, self.static_features)
        latent = self.node_encoder(nodes) + self.position
        latent = self.transformer(latent)
        all_actions = self.decoder(torch.cat((latent, nodes), dim=-1)).squeeze(-1)
        # Mask non-actuator outputs by selection, preserving simulator joint order.
        return all_actions[:, 1:24]


class RpoMxtActorCritic(ActorCritic):
    """RSL-RL PPO-compatible MTG actor; original privileged MLP critic retained."""

    def __init__(self, *args, mxt_config: dict, **kwargs):
        self.mxt_config = copy.deepcopy(mxt_config)
        if kwargs.get("state_dependent_std", False) or kwargs.get("actor_obs_normalization", False):
            raise ValueError("MxT requires independent std and canonical unnormalized actor input")
        super().__init__(*args, **kwargs)

    def _build_actor(self, num_actor_obs, num_actions, actor_hidden_dims, activation):
        if num_actor_obs != STUDENT_POLICY_WIDTH or num_actions != ACTION_WIDTH:
            raise ValueError("RPO MxT observation/action dimensions do not match")
        return RpoMxtActor(self.mxt_config["graph"], **self.mxt_config["architecture"])

    def get_extra_state(self):
        # RSL-RL saves model state even when infos=None; architecture survives PPO.
        return {"kind": "rpo_mxt", "config": copy.deepcopy(self.mxt_config)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("checkpoint MTG architecture/physical contract differs from environment")

    def load_state_dict(self, state_dict, strict=True):
        # Validate BEFORE tensor copying; do not silently overwrite new geometry.
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("checkpoint MTG architecture/physical contract differs from environment")
        if not torch.equal(state_dict["actor.static_features"].cpu(), self.actor.static_features.detach().cpu()):
            raise ValueError("checkpoint node geometry differs from the configured original RPO")
        return super().load_state_dict(state_dict, strict=strict)


def checkpoint_mxt_config(checkpoint: str | Path | None) -> dict | None:
    if checkpoint is None:
        return None
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = data.get("model_state_dict", {}).get("_extra_state")
    if extra is None:
        return None
    if not isinstance(extra, dict) or extra.get("kind") not in ("rpo_mxt", "rpo_mxt_stairs_adapter_v1"):
        raise ValueError("unknown policy extra-state; cannot guess checkpoint architecture")
    resolve_mxt_policy_class(extra)
    return copy.deepcopy(extra["config"])


def resolve_mxt_policy_class(extra):
    """Explicit checkpoint dispatch; an adapter must never be dropped at inference."""
    has_adapter = "stairs_adapter" in extra.get("config", {})
    if extra.get("kind") == "rpo_mxt" and not has_adapter:
        return RpoMxtActorCritic
    if extra.get("kind") == "rpo_mxt_stairs_adapter_v1" and has_adapter:
        from robolab.mxt_stairs_adapter import RpoMxtStairsAdapterActorCritic, validate_config
        validate_config(extra["config"])
        return RpoMxtStairsAdapterActorCritic
    raise ValueError("MxT checkpoint kind and task-adapter configuration disagree")


def configure_mxt_policy(agent_cfg, base_env, urdf_path, *, checkpoint=None, required=False):
    """Called after Isaac creates the robot: use its REAL joint/action order."""
    saved = checkpoint_mxt_config(checkpoint)
    if not required and saved is None:
        return
    if required and checkpoint is not None and saved is None:
        raise ValueError("MLP checkpoint cannot initialize the MxT architecture; distill an MxT student first")
    scales = base_env.cfg.normalization.obs_scales
    contract = rpo_io_contract(base_env.robot, urdf_path, base_env.cfg.robot.action_scale, base_env.step_dt,
                              [getattr(scales, name) for name in
                               ("ang_vel", "projected_gravity", "commands", "joint_pos", "joint_vel", "actions")])
    if base_env.cfg.normalization.clip_actions != 100.0 or agent_cfg.clip_actions != 100.0:
        raise ValueError("MxT requires the original action clipping contract")
    config = {"graph": contract["mtg"], "io_contract": contract,
              "architecture": dict(DEFAULT_ARCHITECTURE)} if saved is None else saved
    if config["io_contract"] != contract:
        raise ValueError("checkpoint and environment physical/observation/action contracts do not match")
    agent_cfg.policy.class_name = ("robolab.mxt_stairs_adapter:RpoMxtStairsAdapterActorCritic"
                                   if "stairs_adapter" in config else POLICY_CLASS)
    agent_cfg.policy.mxt_config = config
    agent_cfg.policy.actor_obs_normalization = False
    agent_cfg.policy.critic_obs_normalization = False
