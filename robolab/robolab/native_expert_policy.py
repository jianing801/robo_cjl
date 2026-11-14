"""Native AMP/Rough MLPs with only a known, bilateral design-context suffix."""
from copy import deepcopy
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization
from robolab.morph_experiments import CONTEXT_WIDTH
from robolab.native_expert_contract import RECIPE


def append_context(obs, context):
    if context.shape != (CONTEXT_WIDTH,) or not torch.isfinite(context).all():
        raise ValueError("invalid known morphology context")
    result = obs.clone() if isinstance(obs, TensorDict) else dict(obs)
    for key in ("policy", "critic"):
        x = obs[key]
        if x.ndim != 2:
            raise ValueError("source-native flattened observation required")
        result[key] = torch.cat((x, context.to(x).expand(x.shape[0], -1)), dim=-1)
    # AMP disc/disc_demo are intentionally not extended, reordered or retargeted.
    return result


class PrefixNormalization(nn.Module):
    """Use the original normalizer on source observations, not constant design IDs.

    The already-scaled known design suffix must not acquire zero variance from
    a single-design block. Source normalization settings/epsilon are unchanged.
    """
    def __init__(self, base_width):
        super().__init__()
        self.base_width = base_width
        self.base = EmpiricalNormalization(base_width)

    def forward(self, x):
        return torch.cat((self.base(x[..., :self.base_width]), x[..., self.base_width:]), -1)

    def update(self, x):
        self.base.update(x[..., :self.base_width])


class NativeMorphActorCritic(ActorCritic):
    def __init__(self, *args, native_config, **kwargs):
        self.native_config = deepcopy(native_config)
        if native_config.get("recipe") != RECIPE or native_config.get("role") not in ("run", "stairs"):
            raise ValueError("native expert identity required; mixed experts are incompatible")
        super().__init__(*args, **kwargs)
        # Keep original network architecture, distribution, activation, and
        # normalization flags. Only the first-layer observation width grows.
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = PrefixNormalization(self.actor_obs_normalizer.mean.numel() - CONTEXT_WIDTH)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = PrefixNormalization(self.critic_obs_normalizer.mean.numel() - CONTEXT_WIDTH)

    def get_extra_state(self):
        return {"kind": RECIPE, "config": deepcopy(self.native_config)}

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValueError("wrong native recipe, bank or expert role")


@torch.no_grad()
def native_design_symmetry(env, obs=None, actions=None):
    """Delegate ALL original base/action mirroring; bilateral design stays fixed."""
    base, tails = None, {}
    if obs is not None:
        base = obs.clone()
        for key in ("policy", "critic"):
            if key in obs:
                tails[key] = obs[key][..., -CONTEXT_WIDTH:]
                base[key] = obs[key][..., :-CONTEXT_WIDTH]
    augmented, augmented_actions = env.native_source_symmetry(env, obs=base, actions=actions)
    if augmented is not None:
        for key, tail in tails.items():
            repeats = augmented[key].shape[0] // tail.shape[0]
            augmented[key] = torch.cat((augmented[key], tail.repeat(repeats, 1)), -1)
    return augmented, augmented_actions
