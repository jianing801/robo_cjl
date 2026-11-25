"""Bridge task-native privileged observations into mixed PPO storage.

The deployable actor keeps the existing ten-frame streaming input. The critic
uses the original native observations for its own task, including morphology
context. Padding is storage-only: task critics consume their real widths.
"""
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import math
import re

import torch
from tensordict import TensorDict


TASKS = ("stairs", "backpack", "run")
CORE_WIDTHS = (3260, 1390, 243)
ACTOR_HISTORY_WIDTH = 780
JOINT_INDICES = (1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21)
JOINT_SIGNS = (-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1)

# These are the two native RPO conventions (direct/base/agents/rpo_agent_cfg.py
# and manager_based/amp/mdp/symmetry/rpo.py). They use the same joint order.
POLICY_INDICES = tuple(range(9)) + tuple(i + start for start in (9, 32, 55) for i in JOINT_INDICES)
POLICY_SIGNS = (-1, 1, -1, 1, -1, 1, 1, -1, -1) + JOINT_SIGNS * 3
FLAT_CRITIC_INDICES = (
    POLICY_INDICES + (78, 79, 80, 82, 81, 86, 87, 88, 83, 84, 85, 90, 89, 92, 91)
    + tuple(i + start for start in (93, 116) for i in JOINT_INDICES)
)
FLAT_CRITIC_SIGNS = POLICY_SIGNS + (1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1) + JOINT_SIGNS * 2
HEIGHT_INDICES = tuple(139 + (10 - row) * 17 + col for row in range(11) for col in range(17))
ROUGH_CRITIC_INDICES = FLAT_CRITIC_INDICES + HEIGHT_INDICES
ROUGH_CRITIC_SIGNS = FLAT_CRITIC_SIGNS + (1,) * 187


@lru_cache(maxsize=None)
def _map_tensors(indices, signs, device, dtype):
    # Cache can be initialized from an inference-only pass before PPO updates.
    # Always allocate regular tensors so autograd can save them for the
    # subsequent mirror-action loss.
    with torch.inference_mode(False):
        return (torch.tensor(indices, device=device, dtype=torch.long),
                torch.tensor(signs, device=device, dtype=dtype))


def _mirror(x, indices, signs):
    index, sign = _map_tensors(indices, signs, x.device, x.dtype)
    return x.index_select(-1, index) * sign


def _mirror_history(x, indices, signs, frames):
    return _mirror(x.reshape(-1, frames, len(indices)), indices, signs).reshape(x.shape)


def _mirror_amp_critic(x):
    # AMP is term-major, not frame-major. The first term is privileged base
    # linear velocity, then angular velocity, gravity, command, q, qd, action.
    # Apply each native transformation within all three frames of that term.
    pieces = []
    offset = 0
    for signs in ((1, -1, 1), (-1, 1, -1), (1, -1, 1), (1, -1, -1),
                  JOINT_SIGNS, JOINT_SIGNS, JOINT_SIGNS):
        width = len(signs) * 3
        indices = tuple(range(3)) if len(signs) == 3 else JOINT_INDICES
        pieces.append(_mirror_history(x[:, offset:offset + width], indices, signs, 3))
        offset += width
    assert offset == CORE_WIDTHS[2]
    return torch.cat(pieces, -1)


class NativeObservationAdapter:
    """Pack task-specific observations with explicit task IDs, then mirror.

    Native critics have three different dynamic widths. Both original spatial
    wrappers append the same static design-context width after those dynamics.
    Keep that tail as-is and leave zero padding untouched during augmentation.
    """

    def __init__(self, critic_widths):
        if isinstance(critic_widths, Mapping):
            if set(critic_widths) != set(TASKS):
                raise ValueError("Expected native critic widths for all three tasks")
            critic_widths = tuple(critic_widths[t] for t in TASKS)
        else:
            critic_widths = tuple(critic_widths)
        if len(critic_widths) != 3 or any(type(w) is not int for w in critic_widths):
            raise ValueError("Expected three integer native critic widths")
        tails = tuple(w - core for w, core in zip(critic_widths, CORE_WIDTHS, strict=True))
        if tails[0] < 1 or len(set(tails)) != 1:
            raise ValueError("Native critic history/layout or shared physical context width differs")
        self.critic_widths = critic_widths
        self.context_width = tails[0]
        self.max_width = max(critic_widths)
        self._pending_action_mask = None

    def pack(self, task, base, native_critic):
        task_id = TASKS.index(task)
        if base.ndim != 2 or base.shape[1] < ACTOR_HISTORY_WIDTH + len(TASKS):
            raise ValueError("Shared policy requires ten canonical frames and task indicator")
        if native_critic.shape != (len(base), self.critic_widths[task_id]):
            raise ValueError(f"{task} native critic shape mismatch: {tuple(native_critic.shape)}")
        if native_critic.device != base.device:
            raise ValueError("Policy/critic device mismatch")
        critic = native_critic
        if critic.shape[-1] < self.max_width:
            critic = torch.cat((critic, critic.new_zeros((len(base), self.max_width - critic.shape[-1]))), -1)
        ids = torch.full((len(base), 1), task_id, device=base.device, dtype=torch.long)
        return TensorDict({"policy": base, "critic": critic, "task_id": ids}, batch_size=[len(base)])

    @staticmethod
    def concat(packed):
        """Concatenate row batches without altering the task/native layouts."""
        return torch.cat(tuple(packed), dim=0)

    def augment(self, obs=None, actions=None, env=None):
        """Native symmetry callback: originals then task-native augmentation.

        Rough and AMP enable symmetry in their source agents. The flat
        backpack source agent disables it. Repeat backpack observations and
        actions *unchanged* so every task has the same weight in the native
        fixed-size augmented minibatch, with zero backpack mirror loss.
        """
        obs_aug = None
        if obs is not None:
            self._pending_action_mask = None
            tasks = obs["task_id"].squeeze(-1)
            if tasks.shape != (len(obs),) or tasks.dtype != torch.long or torch.any((tasks < 0) | (tasks >= len(TASKS))):
                raise ValueError("Unknown task identity for native symmetry minibatch")
            # The bundled native update calls this with observations and then
            # once with only action means for its mirror target. Cache the
            # exact shuffled task route, not the original environment order.
            with torch.inference_mode(False):
                enabled = (tasks != 1)
                self._pending_action_mask = enabled.clone()
            obs_mirror = obs.clone()
            base = obs["policy"]
            obs_mirror["policy"][enabled, :ACTOR_HISTORY_WIDTH] = _mirror_history(
                base[enabled, :ACTOR_HISTORY_WIDTH], POLICY_INDICES, POLICY_SIGNS, 10)
            # Task indicator, graph features and physical context describe the
            # unchanged robot. They are not dynamic left/right states.
            if "critic" in obs.keys():
                source = obs["critic"]
                for task in (0, 2):
                    width = CORE_WIDTHS[task]
                    mask = tasks == task
                    core = source[mask, :width]
                    if task == 0:
                        core = _mirror_history(core, ROUGH_CRITIC_INDICES, ROUGH_CRITIC_SIGNS, 10)
                    else:
                        core = _mirror_amp_critic(core)
                    obs_mirror["critic"][mask, :width] = core
            obs_aug = torch.cat((obs, obs_mirror), dim=0)
        actions_aug = None
        if actions is not None:
            if obs is None:
                enabled = self._pending_action_mask
                self._pending_action_mask = None  # exactly one action-only target per minibatch
            if (enabled is None or actions.ndim != 2 or actions.shape[1] != len(JOINT_INDICES)
                    or enabled.shape != (len(actions),) or enabled.device != actions.device):
                self._pending_action_mask = None
                raise ValueError("Native mirror action has no matching current observation minibatch route")
            mirrored = actions.clone()
            mirrored[enabled] = _mirror(actions[enabled], JOINT_INDICES, JOINT_SIGNS)
            actions_aug = torch.cat((actions, mirrored), dim=0)
        return obs_aug, actions_aug


def amp_native_observations(obs):
    """Return only native AMP discriminator/demo rows from the actual env.

    These values must NOT enter shared actor/critic storage or be generated by
    reconstructing AMP's own three-frame history from the student's history.
    """
    disc = obs.get("disc")
    demo = obs.get("disc_demo")
    if (not isinstance(disc, torch.Tensor) or not isinstance(demo, torch.Tensor)
            or disc.ndim != 3 or disc.shape[1:] != (3, 67)
            or demo.shape != disc.shape or demo.device != disc.device):
        raise ValueError("AMP native discriminator and demonstration require matching [N,3,67] observations")
    return TensorDict({"disc": disc, "disc_demo": demo}, batch_size=[len(disc)])


def native_amp_setup(agent, teacher_state, obs, *, step_dt):
    """Independent native AMP initialization for the mixed learner.

    Native resolve_amp_config fills these dimensions and dt from the actual
    env. Return them here with copies of the checkpoint weights and statistics;
    the mixed learner owns its own trainable discriminator and optimizer.
    """
    native = amp_native_observations(obs)
    amp = deepcopy(agent["algorithm"]["amp_cfg"])
    groups = deepcopy(agent["obs_groups"])
    if groups.get("discriminator") != ["disc"] or groups.get("discriminator_demonstration") != ["disc_demo"]:
        raise ValueError("AMP native discriminator/demo group contract changed")
    if not math.isclose(step_dt, .02, rel_tol=0., abs_tol=1e-10):
        raise ValueError("AMP native step dt changed")
    amp.update(disc_obs_steps=native["disc"].shape[1], disc_obs_dim=native["disc"].shape[2],
               step_dt=float(step_dt))
    # prepare_rollout runs in inference mode. Ensure checkpoint tensors remain
    # ordinary independent tensors even if copied at that point.
    with torch.inference_mode(False):
        state = {key: deepcopy(teacher_state[key]) for key in
                 ("amp_discriminator_state_dict", "amp_discriminator_normalizer_state_dict")}
    return {"amp_cfg": amp, "amp_state": state, "obs_groups": groups}


def native_expert_setup(task, checkpoint, teacher, obs, *, checkpoint_sha256):
    """Extract the verified native value network and learned exploration prior.

    The caller has already validated the checkpoint SHA in load_plan() and
    loaded its native model_state_dict strictly into the frozen expert. Copy
    directly from that same checkpoint, never from a randomly initialized
    critic or by computing inverse transforms on the shared student input.
    """
    if task not in TASKS or not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("Native expert checkpoint identity differs")
    state = checkpoint["model_state_dict"]
    critic = obs["critic"]
    core = CORE_WIDTHS[TASKS.index(task)]
    if critic.ndim != 2 or len(critic) < 1:
        raise ValueError("Native expert critic observation shape differs")
    width = critic.shape[-1]
    config = state.get("_extra_state", {}).get("config", {})
    context = config.get("context_width")
    if (type(context) is not int or context < 20 or config.get("role") != task
            or core + context != width):
        raise ValueError("Native expert critic context/layout differs")

    critic_prefix, normalizer_prefix = "critic.", "critic_obs_normalizer."
    critic_state = {key[len(critic_prefix):]: value for key, value in state.items()
                    if key.startswith(critic_prefix)}
    norm_state = {key[len(normalizer_prefix):]: value for key, value in state.items()
                  if key.startswith(normalizer_prefix)}
    expected = {}
    for layer, inputs, outputs in ((0, width, 512), (2, 512, 256), (4, 256, 128), (6, 128, 1)):
        expected[f"{layer}.weight"] = (outputs, inputs)
        expected[f"{layer}.bias"] = (outputs,)

    def finite_tensor(value, shape):
        return (isinstance(value, torch.Tensor) and value.shape == shape
                and torch.is_floating_point(value) and bool(torch.isfinite(value).all()))

    if set(critic_state) != set(expected) or any(
            not finite_tensor(critic_state[key], shape) for key, shape in expected.items()):
        raise ValueError("Native expert critic checkpoint architecture or weights differ")

    # The actual RPOFlat/RPORough experts were trained without a critic
    # normalizer. AMP has the native three-frame PrefixNormalization, which
    # intentionally leaves the known design suffix unchanged.
    base_width = core if task == "run" else None
    if bool(teacher.critic_obs_normalization) != (base_width is not None):
        raise ValueError("Native expert critic normalizer contract differs")
    if base_width is None:
        if norm_state:
            raise ValueError("Native identity critic has unexpected normalizer state")
    else:
        keys = {"base._mean", "base._var", "base._std", "base.count"}
        if (set(norm_state) != keys
                or any(not finite_tensor(norm_state[k], (1, base_width)) for k in keys - {"base.count"})
                or not isinstance(norm_state["base.count"], torch.Tensor)
                or norm_state["base.count"].shape != ()
                or norm_state["base.count"].dtype != torch.long
                or bool(norm_state["base.count"] < 0)
                or bool((norm_state["base._var"] < 0).any())
                or bool((norm_state["base._std"] < 0).any())):
            raise ValueError("Native AMP critic normalizer state or prefix width differs")

    if getattr(teacher, "state_dependent_std", False) or ("std" in state) == ("log_std" in state):
        raise ValueError("Native expert action std checkpoint is not a single per-joint parameter")
    raw_std = state.get("std", state.get("log_std"))
    if not finite_tensor(raw_std, (23,)):
        raise ValueError("Native expert action std is not finite")
    with torch.inference_mode(False):
        action_std = raw_std.detach().cpu().clone()
        if "log_std" in state:
            action_std = action_std.exp()
        if not finite_tensor(action_std, (23,)) or not bool((action_std > 0).all()):
            raise ValueError("Native expert action std must be positive and finite")
        prior = dict(
            critic_state_dict={k: v.detach().cpu().clone() for k, v in critic_state.items()},
            critic_normalizer_state_dict={k: v.detach().cpu().clone() for k, v in norm_state.items()},
            critic_normalizer_base_width=base_width,
            action_std=action_std,
            checkpoint_sha256=checkpoint_sha256)
    with torch.inference_mode():
        value = teacher.evaluate(obs)
    if not finite_tensor(value, (len(critic), 1)):
        raise ValueError("Trusted native expert initial value output differs or is nonfinite")
    with torch.inference_mode(False):
        # Ordinary, independent tensor on the physical observation device for
        # the learner's initial exact-output parity check; not a prior field.
        value = value.detach().clone()
    return {"expert_prior": prior, "teacher_value": value}
