"""A native RSL-RL PPOAMP learner for one gated Actor and three physical tasks.

Only the policy adapter, task-specific advantage/entropy weighting, and the
AMP-only discriminator collection differ from the native optimizer.
"""
from copy import deepcopy
import math
from collections.abc import Mapping

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_amp import PPOAMP
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.storage import CircularBuffer, RolloutStorage


TASKS = ('stairs', 'backpack', 'run')
TASK_ENTROPY = (.005, .005, .01)
ACTOR_INITIAL_STD = 1.
LEARNING_RATE = 1.e-4
PPO_NATIVE_CONFIG = dict(learning_rate=LEARNING_RATE, schedule='adaptive', desired_kl=.01,
    gamma=.99, lam=.95, clip_param=.2, value_loss_coef=1., use_clipped_value_loss=True,
    max_grad_norm=1., entropy_coef=.005, normalize_advantage_per_mini_batch=False)
CRITIC_HIDDEN = (512, 256, 128)


def _task_tuple(values):
    result = tuple(values[k] for k in TASKS) if isinstance(values, Mapping) else tuple(values)
    if len(result) != len(TASKS) or any(type(v) is not int or v < 1 for v in result):
        raise ValueError('Three positive integer task dimensions are required')
    return result


class NativeSharedPolicy(nn.Module):
    """RSL-RL actor/critic interface without altering the distilled Actor."""
    is_recurrent = False

    def __init__(self, actor, critic_widths, device):
        super().__init__()
        self.actor = actor
        self.critic_widths = _task_tuple(critic_widths)
        self.critic_normalizers = nn.ModuleList(EmpiricalNormalization(width) for width in self.critic_widths)
        self.critic_encoders = nn.ModuleList(nn.Sequential(nn.Linear(width, CRITIC_HIDDEN[0]), nn.ELU())
            for width in self.critic_widths)
        self.critic_trunk = nn.Sequential(nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1]), nn.ELU(),
            nn.Linear(CRITIC_HIDDEN[1], CRITIC_HIDDEN[2]), nn.ELU(), nn.Linear(CRITIC_HIDDEN[2], 1))
        self.log_std = nn.Parameter(torch.full((23,), math.log(ACTOR_INITIAL_STD)))
        self.obs_groups = {'policy': ['policy'], 'critic': ['critic'], 'discriminator': ['disc'],
                           'discriminator_demonstration': ['disc_demo']}
        self.distribution = None
        self.to(device)

    def _critic_inputs(self, obs):
        native = obs['critic']
        ids = obs['task_id']
        if native.ndim != 2 or native.shape[1] != max(self.critic_widths) or ids.shape != (len(native), 1):
            raise ValueError('Invalid mixed native critic observations')
        if ids.dtype != torch.long:
            raise ValueError('Task identities must be integer indices')
        ids = ids.squeeze(-1)
        if torch.any((ids < 0) | (ids >= len(TASKS))):
            raise ValueError('Unknown task identity')
        return native, ids

    def _update_distribution(self, obs):
        mean = self.actor(obs['policy'])
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean), validate_args=False)

    def act(self, obs, **kwargs):
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(obs['policy'])

    def evaluate(self, obs, **kwargs):
        native, ids = self._critic_inputs(obs)
        encoded = native.new_zeros((len(native), CRITIC_HIDDEN[0]))
        for i, (width, normalizer, encoder) in enumerate(zip(self.critic_widths, self.critic_normalizers,
                self.critic_encoders, strict=True)):
            rows = ids == i
            encoded[rows] = encoder(normalizer(native[rows, :width]))
        return self.critic_trunk(encoded)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    @torch.no_grad()
    def update_normalization(self, obs):
        native, ids = self._critic_inputs(obs)
        for i, (width, normalizer) in enumerate(zip(self.critic_widths, self.critic_normalizers, strict=True)):
            rows = native[ids == i, :width]
            if len(rows):
                normalizer.update(rows)

    def reset(self, dones=None):
        pass


class NativeMixedPPO(PPOAMP):
    """The native PPOAMP update is shared; only run observations reach AMP."""

    def __init__(self, *args, counts, critic_warmup_updates, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = _task_tuple(counts)
        self.critic_warmup_updates = int(critic_warmup_updates)
        if self.critic_warmup_updates < 0:
            raise ValueError('Invalid critic warm-up budget')
        self.critic_only = False
        self.task_entropy = torch.tensor(TASK_ENTROPY, device=self.device)
        self.task_ids = torch.arange(len(TASKS), device=self.device).repeat_interleave(
            torch.tensor(self.counts, device=self.device)).reshape(-1, 1)

    def _entropy_regularization(self, obs_batch, entropy_batch):
        return (entropy_batch * self.task_entropy[obs_batch['task_id'].squeeze(-1)]).mean()

    def configure_critic_only(self, enabled):
        self.critic_only = bool(enabled)
        self.policy.train()
        self.policy.actor.requires_grad_(not self.critic_only)
        self.policy.log_std.requires_grad_(not self.critic_only)
        self.amp_discriminator.requires_grad_(not self.critic_only)
        self.amp_discriminator.train(not self.critic_only)

    @torch.no_grad()
    def process_mixed_env_step(self, next_obs, raw_rewards, dones, timeouts, amp_obs):
        n = sum(self.counts)
        if any(t.shape != (n,) for t in (raw_rewards,dones,timeouts)):
            raise ValueError('Mixed rewards/dones/timeout shape does not match the rollout')
        if dones.dtype != torch.bool or timeouts.dtype != torch.bool or (timeouts & ~dones).any():
            raise ValueError('Invalid mixed termination masks')
        if (self.transition.observations is None
                or not torch.equal(self.transition.observations['task_id'], self.task_ids)
                or not torch.equal(next_obs['task_id'], self.task_ids)):
            raise ValueError('Mixed rollout task order changed before AMP reward routing')
        disc = self.amp_discriminator.get_disc_obs(amp_obs)
        demo = self.amp_discriminator.get_disc_demo_obs(amp_obs)
        expected = (self.counts[-1], self.amp_discriminator.disc_obs_steps, self.amp_discriminator.disc_obs_dim)
        if disc.shape != expected or demo.shape != expected:
            raise ValueError('Only the native AMP environments may feed the discriminator')
        if not torch.isfinite(disc).all() or not torch.isfinite(demo).all():
            raise ValueError('Nonfinite native AMP observations')
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc,
            dt=self.amp_cfg['step_dt'])
        rewards = raw_rewards.clone()
        rewards[-self.counts[-1]:] = self.amp_discriminator.lerp_reward(raw_rewards[-self.counts[-1]:], self.style_rewards)
        if not torch.isfinite(rewards).all():
            raise ValueError('Nonfinite mixed PPO rewards')
        self.rewards_lerp = rewards
        # During critic-only warmup the discriminator, including normalizer, is frozen.
        self.disc_obs_buffer.append(disc)
        self.disc_demo_obs_buffer.append(demo)
        PPO.process_env_step(self, next_obs, rewards, dones, {'time_outs': timeouts})
        return rewards

    def compute_returns_and_update(self, next_obs, *, critic_only=False):
        if self.storage.step != self.storage.num_transitions_per_env:
            raise ValueError('Native PPO rollout is incomplete')
        if bool(critic_only) != self.critic_only:
            raise ValueError('Set critic-only mode before collecting the rollout')
        with torch.no_grad():
            PPO.compute_returns(self, next_obs)
            # Preserve the explicit task-balanced advantage contract. All other
            # GAE and PPO arithmetic is native RSL-RL.
            raw_advantages = self.storage.returns - self.storage.values
            for part, target in zip(raw_advantages.split(self.counts, dim=1),
                                    self.storage.advantages.split(self.counts, dim=1), strict=True):
                target.copy_((part - part.mean()) / (part.std(unbiased=part.numel() > 1) + 1.e-8))
        if critic_only:
            # Native PPO's value loss/clipping and data augmentation remain
            # active. No actor, std or discriminator optimization, and a fixed
            # learning rate avoids adapting on an intentionally unchanged actor.
            schedule, entropy, symmetry = self.schedule, self.entropy_coef, self.symmetry
            self.schedule, self.entropy_coef = 'fixed', 0.
            if symmetry:
                self.symmetry = dict(symmetry, use_mirror_loss=False)
            try:
                result = PPO.update(self)
            finally:
                self.schedule, self.entropy_coef, self.symmetry = schedule, entropy, symmetry
        else:
            result = PPOAMP.update(self)
        result.update(samples=self.storage.num_transitions_per_env * sum(self.counts),
            learning_rate=float(self.learning_rate), mean_action_std=float(self.policy.log_std.detach().exp().mean()),
            tasks=self._task_diagnostics(), critic_only=bool(critic_only))
        if not all(math.isfinite(v) for v in result.values() if isinstance(v,(int,float))):
            raise ValueError('Nonfinite native PPO statistics')
        if any(not torch.isfinite(p).all() for p in self.policy.parameters()):
            raise ValueError('Nonfinite native shared policy')
        if any(not torch.isfinite(p).all() for p in self.amp_discriminator.parameters()):
            raise ValueError('Nonfinite native AMP discriminator')
        return result

    @torch.no_grad()
    def _task_diagnostics(self):
        st = self.storage
        # RSL-RL clears only the cursor; its same rollout arrays stay available.
        # Evaluate a single step per chunk to avoid additional full-rollout copies.
        summaries = {task: dict(kl=[], clip_fraction=[], value=[]) for task in TASKS}
        for t in range(st.num_transitions_per_env):
            obs = st.observations[t]
            mu = self.policy.act_inference(obs)
            sigma = self.policy.log_std.exp().expand_as(mu)
            old_mu, old_sigma = st.mu[t], st.sigma[t]
            kl = (torch.log(sigma/old_sigma) + (old_sigma.square() + (old_mu-mu).square())
                    / (2.*sigma.square())-.5).sum(-1).clamp_min(0.)
            new_logp = Normal(mu,sigma,validate_args=False).log_prob(st.actions[t]).sum(-1)
            ratio = (new_logp - st.actions_log_prob[t].squeeze(-1)).exp()
            clipped = ((ratio < 1.-self.clip_param) | (ratio > 1.+self.clip_param)).float()
            value = self.policy.evaluate(obs)
            for task, kl_part, clip_part, val_part in zip(TASKS, kl.split(self.counts),
                    clipped.split(self.counts), value.split(self.counts), strict=True):
                summaries[task]['kl'].append(kl_part.mean())
                summaries[task]['clip_fraction'].append(clip_part.mean())
                summaries[task]['value'].append(val_part)
        for task, returns in zip(TASKS, st.returns.split(self.counts,dim=1), strict=True):
            summary = summaries[task]
            value = torch.stack(summary.pop('value'))
            var = returns.var(unbiased=False)
            variance = (1.-(returns-value).var(unbiased=False)/var) if var > 1.e-10 else None
            summary.update(kl=float(torch.stack(summary['kl']).mean()),
                clip_fraction=float(torch.stack(summary['clip_fraction']).mean()),
                explained_variance=None if variance is None else float(variance))
            if not all(math.isfinite(v) for v in summary.values() if v is not None):
                raise ValueError(f'Nonfinite {task} diagnostics')
        return summaries


def build_native_learner(actor, counts, critic_widths, amp_cfg, amp_state, symmetry_callback, device,
        *, steps=24, symmetry_env=None, num_epochs=5, num_minibatches=4, critic_warmup_updates=10):
    """Create the candidate-local native learner and clone the teacher AMP prior."""
    counts, widths = _task_tuple(counts), _task_tuple(critic_widths)
    policy = NativeSharedPolicy(actor, widths, device)
    n = sum(counts)
    if steps < 1 or steps*n % num_minibatches or steps*counts[-1] % num_minibatches:
        raise ValueError('Rollout samples must divide evenly across the native minibatches')
    obs = TensorDict(dict(policy=torch.empty(n,actor.width,device=device),
        critic=torch.empty(n,max(widths),device=device),task_id=torch.empty(n,1,dtype=torch.long,device=device)),
        batch_size=[n], device=device)
    storage = RolloutStorage('rl', n, steps, obs, [23], device=device)
    cfg = deepcopy(amp_cfg)
    buffer_size = int(cfg['disc_obs_buffer_size'])
    if buffer_size < steps:
        raise ValueError('Native AMP buffer is shorter than a complete rollout')
    disc = CircularBuffer(buffer_size, counts[-1], device)
    demo = CircularBuffer(buffer_size, counts[-1], device)
    symmetry = None if symmetry_callback is None else dict(use_data_augmentation=True, use_mirror_loss=True,
        mirror_loss_coeff=.2, data_augmentation_func=symmetry_callback, _env=symmetry_env)
    alg = NativeMixedPPO(policy,storage,disc,demo,counts=counts,
        critic_warmup_updates=critic_warmup_updates,num_learning_epochs=num_epochs,num_mini_batches=num_minibatches,
        device=device,symmetry_cfg=symmetry,amp_cfg=cfg,**PPO_NATIVE_CONFIG)
    alg.amp_discriminator.load_state_dict(amp_state['amp_discriminator_state_dict'], strict=True)
    alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
        amp_state['amp_discriminator_normalizer_state_dict'],strict=True)
    if any(not torch.isfinite(v).all() for v in alg.amp_discriminator.state_dict().values()):
        raise ValueError('Nonfinite native AMP teacher state')
    alg.configure_critic_only(False)
    return alg
