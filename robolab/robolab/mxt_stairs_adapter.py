"""Frozen original MxT actor plus a bounded STAIRS-only residual decoder.

This adds no sensors or joints. RUN returns the original mean exactly for the
same input; shared simulation RNG/reset effects still require closed-loop tests.
The old critic/optimizers are explicitly migrated, not silently reinitialized.
"""
import copy
import hashlib
import math

import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from robolab.mxt_policy import RpoMxtActor, RpoMxtActorCritic, NODE_WIDTH, tokenize_rpo_history
from robolab.task_reward_routing import task_ids_from_context
from robolab.mxt_online_expert import OnlineExpertPPOAMP

KIND = "rpo_mxt_stairs_adapter_v1"
POLICY_CLASS = "robolab.mxt_stairs_adapter:RpoMxtStairsAdapterActorCritic"
PREFIX = "actor.stairs_adapter."


def adapter_config(base, *, bound=.2, seed=91818):
    if "stairs_adapter" in base:
        raise ValueError("cannot silently stack task adapters")
    result = copy.deepcopy(base)
    result["stairs_adapter"] = {"version": 1, "hidden_dim": 64, "raw_action_bound": float(bound),
        "seed": int(seed), "task_id": 1, "initialization": "zero_output", "base_actor_frozen": True}
    validate_config(result)
    return result


def validate_config(config):
    adapter = config.get("stairs_adapter", {})
    if (set(adapter) != {"version", "hidden_dim", "raw_action_bound", "seed", "task_id", "initialization", "base_actor_frozen"}
            or adapter["version"] != 1 or adapter["hidden_dim"] != 64 or adapter["task_id"] != 1
            or adapter["initialization"] != "zero_output" or adapter["base_actor_frozen"] is not True
            or not math.isfinite(adapter["raw_action_bound"]) or not 0 < adapter["raw_action_bound"] <= .2
            or not isinstance(adapter["seed"], int)):
        raise ValueError("explicit bounded original-structure STAIRS adapter contract required")


class RpoMxtStairsAdapterActor(RpoMxtActor):
    def __init__(self, graph, *, adapter, **architecture):
        super().__init__(graph, **architecture)
        validate_config({"stairs_adapter": adapter})
        self.raw_action_bound = adapter["raw_action_bound"]
        # New module initialization must not change reset/noise RNG compared to
        # constructing the original actor. CPU modules are moved by RSL later.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(adapter["seed"])
            self.stairs_adapter = nn.Sequential(nn.Linear(self.decoder.in_features, 64), nn.ELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.stairs_adapter[-1].weight)
            nn.init.zeros_(self.stairs_adapter[-1].bias)
        self.freeze_base()

    def freeze_base(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("stairs_adapter."):
                parameter.requires_grad_(False)
                parameter.grad = None
        self.node_encoder.eval()
        self.transformer.eval()
        self.decoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_base()  # RSL train() must never unfreeze/update the RUN path.
        return self

    def forward(self, policy_obs):
        ids = task_ids_from_context(policy_obs[:, -2:])
        with torch.no_grad():
            nodes = tokenize_rpo_history(policy_obs, self.static_features)
            latent = self.transformer(self.node_encoder(nodes) + self.position)
            features = torch.cat((latent, nodes), dim=-1)
            baseline = self.decoder(features).squeeze(-1)[:, 1:24]
        stairs = ids == 1
        if not stairs.any():
            return baseline
        delta = self.raw_action_bound * torch.tanh(self.stairs_adapter(features[stairs, 1:24]).squeeze(-1))
        result = baseline.clone()
        result[stairs] = result[stairs] + delta
        return result


class RpoMxtStairsAdapterActorCritic(RpoMxtActorCritic):
    def _build_actor(self, num_actor_obs, num_actions, actor_hidden_dims, activation):
        if num_actor_obs != 782 or num_actions != 23:
            raise ValueError("fixed original RPO actor input/action contract required")
        validate_config(self.mxt_config)
        return RpoMxtStairsAdapterActor(self.mxt_config["graph"],
            adapter=self.mxt_config["stairs_adapter"], **self.mxt_config["architecture"])

    def get_extra_state(self):
        return {"kind": KIND, "config": copy.deepcopy(self.mxt_config)}

    def load_state_dict(self, state_dict, strict=True):
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("adapter architecture/physical contract differs; no silent plain-policy load")
        if not torch.equal(state_dict["actor.static_features"].cpu(), self.actor.static_features.detach().cpu()):
            raise ValueError("adapter original node geometry differs")
        return ActorCritic.load_state_dict(self, state_dict, strict=strict)


def frozen_actor_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.actor.state_dict().items()):
        if not name.startswith("stairs_adapter."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(policy.std.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_plain_source(source):
    if source["model_state_dict"]["_extra_state"]["kind"] != "rpo_mxt":
        raise ValueError("first adapter must start from a verified plain shared policy")
    with torch.random.fork_rng(devices=[]):
        obs = TensorDict({"policy": torch.zeros(1, 782), "critic": torch.zeros(1, 3262)}, batch_size=[1])
        model = RpoMxtActorCritic(obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
            num_actions=23, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
            actor_obs_normalization=False, critic_obs_normalization=False, activation="elu",
            mxt_config=source["model_state_dict"]["_extra_state"]["config"])
    model.load_state_dict(source["model_state_dict"], strict=True)
    return model.eval()


def audit_original_actions(original, adapted, directory, source_sha256):
    """Frozen replay probes are algebraic isolation evidence, NOT gait holdout."""
    from pathlib import Path
    from robolab.multitask_distill import make_student_policy_input
    device = next(adapted.parameters()).device
    previous_mode = adapted.training
    adapted.eval()
    original.eval()
    generator = torch.Generator().manual_seed(20261017)
    report = {"same_input_action_function_check_not_closed_loop_proof": True}
    try:
        for task, name in ((0, "run"), (1, "stairs")):
            data = torch.load(Path(directory) / f"{name}.pt", map_location="cpu", weights_only=True)
            if (data.get("roll_in_student_sha256") != source_sha256 or data.get("schema_version") != 3 or
                    data.get("io_contract") != original.mxt_config["io_contract"] or
                    data.get("history_layout") != "frame_major" or
                    data.get("enable_urdf_update") is not False or data.get("enable_motor_curve") is not False or
                    not torch.all(data["task_ids"] == task)):
                raise ValueError("verified original-source real replay states required")
            indices = torch.randperm(len(data["policy_obs"]), generator=generator)[:256]
            inputs = make_student_policy_input(data["policy_obs"][indices].float(), data["task_ids"][indices]).to(device)
            maximum = 0.
            with torch.inference_mode():
                for batch in inputs.split(64):
                    td = TensorDict({"policy": batch}, batch_size=[len(batch)])
                    maximum = max(maximum, float((original.act_inference(td) - adapted.act_inference(td)).abs().max()))
            report[name] = {"real_state_samples": len(inputs), "maximum_raw_action_difference": maximum}
        if report["run"]["maximum_raw_action_difference"] != 0.:
            raise RuntimeError("frozen RUN changed on real source states")
        if report["stairs"]["maximum_raw_action_difference"] > adapted.actor.raw_action_bound + 1e-5:
            raise RuntimeError("STAIRS correction exceeded its explicit bound")
        return report
    finally:
        adapted.train(previous_mode)


def migrate_checkpoint(source, policy, source_parameter_names, *, adapter_learning_rate=1e-4):
    """Append only fresh adapter state; preserve all named original Adam states."""
    original_config = copy.deepcopy(policy.mxt_config)
    original_config.pop("stairs_adapter")
    if source["model_state_dict"]["_extra_state"] != {"kind": "rpo_mxt", "config": original_config}:
        raise ValueError("source architecture does not match frozen adapter base")
    names = [name for name, _ in policy.named_parameters() if not name.startswith(PREFIX)]
    if names != list(source_parameter_names):
        raise ValueError("original optimizer parameter order/names changed")
    old_opt = source["optimizer_state_dict"]
    if len(old_opt["param_groups"]) != 1 or len(old_opt["param_groups"][0]["params"]) != len(names):
        raise ValueError("explicit single original optimizer group required")
    original = [p for n, p in policy.named_parameters() if not n.startswith(PREFIX)]
    added = [p for n, p in policy.named_parameters() if n.startswith(PREFIX)]
    optimizer = torch.optim.Adam([
        {"params": original, "lr": source["learning_rate"], "name": "original_parameters"},
        {"params": added, "lr": adapter_learning_rate, "name": "stairs_adapter"}])
    migrated = copy.deepcopy(source)
    target = policy.state_dict()
    expected = set(source["model_state_dict"]) | {n for n in target if n.startswith(PREFIX)}
    if set(target) != expected:
        raise ValueError("only the four adapter tensors may be added")
    for name, value in source["model_state_dict"].items():
        if name != "_extra_state":
            target[name] = value.detach().clone()
    migrated["model_state_dict"] = target
    new_opt = optimizer.state_dict()
    ids = dict(zip(old_opt["param_groups"][0]["params"], new_opt["param_groups"][0]["params"]))
    new_opt["state"] = {ids[k]: copy.deepcopy(v) for k, v in old_opt["state"].items()}
    settings = {k: copy.deepcopy(v) for k, v in old_opt["param_groups"][0].items() if k != "params"}
    new_opt["param_groups"][0].update(settings)
    new_opt["param_groups"][0]["name"] = "original_parameters"
    migrated["optimizer_state_dict"] = new_opt
    migrated["infos"] = {**migrated["infos"], "stairs_adapter_migration": {
        "original_parameter_names": names, "new_parameter_names": [n for n, _ in policy.named_parameters() if n.startswith(PREFIX)],
        "original_optimizer_states_preserved_by_name": True, "adapter_optimizer_initially_empty": True}}
    return migrated, optimizer


class FrozenRunPPOAMP(OnlineExpertPPOAMP):
    def update(self):
        for name, parameter in self.policy.actor.named_parameters():
            if not name.startswith("stairs_adapter.") and (parameter.requires_grad or parameter.grad is not None):
                raise RuntimeError("frozen RUN/base parameter became trainable")
        result = super().update()
        if self.online_sums.get("run_source_mse", 0.) != 0.:
            raise RuntimeError("RUN mean differs from its frozen source on current PPO states")
        if self.policy.std.requires_grad or self.policy.std.grad is not None:
            raise RuntimeError("RUN exploration std must stay frozen")
        return result
