"""Native MLP/PPO semantics with a bank-sized known design suffix."""
from copy import deepcopy
import torch
from torch import nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization


class PrefixNormalization(nn.Module):
    """Normalize original observations only; known design suffix is already scaled."""
    def __init__(self, base_width):
        super().__init__()
        if base_width < 1:
            raise ValueError("context must not consume the original observations")
        self.base_width = base_width
        self.base = EmpiricalNormalization(base_width)

    def forward(self, x):
        return torch.cat((self.base(x[..., :self.base_width]), x[..., self.base_width:]), -1)

    def update(self, x):
        self.base.update(x[..., :self.base_width])


class SpatialActorCritic(ActorCritic):
    def __init__(self, *args, native_config, **kwargs):
        self.native_config = deepcopy(native_config)
        width = native_config["context_width"]
        if type(width) is not int or width < 20 or native_config.get("role") != "stairs":
            raise ValueError("explicit Rough spatial context schema required")
        super().__init__(*args, **kwargs)
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = PrefixNormalization(self.actor_obs_normalizer.mean.numel()-width)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = PrefixNormalization(self.critic_obs_normalizer.mean.numel()-width)

    def get_extra_state(self):
        return dict(kind="spatial_configurable_policy_v1", config=deepcopy(self.native_config))

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValueError("incompatible spatial bank/context/checkpoint")


@torch.no_grad()
def spatial_design_symmetry(env, obs=None, actions=None):
    width = env.unwrapped.spatial_context.shape[-1]
    base, tails = None, {}
    if obs is not None:
        base = obs.clone()
        for key in ("policy", "critic"):
            if key in obs:
                tails[key] = obs[key][..., -width:]
                base[key] = obs[key][..., :-width]
    augmented, result_actions = env.native_source_symmetry(env, obs=base, actions=actions)
    if augmented is not None:
        for key, tail in tails.items():
            repeats, remainder = divmod(augmented[key].shape[0], tail.shape[0])
            if remainder:
                raise ValueError("source symmetry changed sample grouping")
            augmented[key] = torch.cat((augmented[key], tail.repeat(repeats, 1)), -1)
    return augmented, result_actions
