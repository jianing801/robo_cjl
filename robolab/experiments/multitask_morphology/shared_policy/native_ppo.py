"""A native RSL-RL PPOAMP learner for one gated Actor and three physical tasks.

Only the policy adapter, task-specific advantage/entropy weighting, and the
AMP-only discriminator collection differ from the native optimizer.
"""
from copy import deepcopy
import math
import re
from collections.abc import Mapping

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_amp import PPOAMP
from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.storage import CircularBuffer, RolloutStorage


TASKS = ('stairs', 'backpack', 'run')
TASK_ENTROPY = (.005, .005, .01)
LEARNING_RATE = 1.e-4
PPO_NATIVE_CONFIG = dict(learning_rate=LEARNING_RATE, schedule='adaptive', desired_kl=.01,
    gamma=.99, lam=.95, clip_param=.2, value_loss_coef=1., use_clipped_value_loss=True,
    max_grad_norm=1., entropy_coef=.005, normalize_advantage_per_mini_batch=False)
CRITIC_HIDDEN = (512, 256, 128)


def project_actor_gradients(gradients, weights=None):
    """PCGrad on task contributions; cyclic order avoids altering rollout RNG."""
    if gradients.ndim != 2 or gradients.shape[0] != len(TASKS):
        raise ValueError('Expected three flattened Actor gradients')
    if weights is not None:
        weights=torch.as_tensor(weights, dtype=gradients.dtype, device=gradients.device)
        if weights.shape!=(len(TASKS),) or not torch.isfinite(weights).all() or not (weights>0).all():
            raise ValueError('Expected three positive, finite Actor task weights')
        gradients=gradients*(weights/weights.mean())[:,None]
    projected=[]
    for i in range(len(TASKS)):
        g=gradients[i]
        for shift in range(1,len(TASKS)):
            other=gradients[(i+shift)%len(TASKS)]
            dot=torch.dot(g,other)
            g=g-torch.minimum(dot,torch.zeros_like(dot))*other/torch.dot(other,other).clamp_min(1e-30)
        projected.append(g)
    return torch.stack(projected).sum(0)


def _task_tuple(values):
    result = tuple(values[k] for k in TASKS) if isinstance(values, Mapping) else tuple(values)
    if len(result) != len(TASKS) or any(type(v) is not int or v < 1 for v in result):
        raise ValueError('Three positive integer task dimensions are required')
    return result


class NativePrefixNormalization(nn.Module):
    """Exact spatial teacher normalization: dynamic prefix, unchanged context.

    Matches PrefixNormalization in the native AMP expert, including its
    EmpiricalNormalization epsilon, count, and continuing update semantics.
    """
    def __init__(self, base_width):
        super().__init__()
        self.base_width = base_width
        self.base = EmpiricalNormalization(base_width)

    def forward(self, x):
        return torch.cat((self.base(x[..., :self.base_width]), x[..., self.base_width:]), -1)

    def update(self, x):
        self.base.update(x[..., :self.base_width])


def _load_exact_prior(module, state, task, role):
    target = module.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(target):
        raise ValueError(f'{task}: {role} teacher state keys differ')
    for key, expected in target.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                or value.dtype != expected.dtype or not torch.isfinite(value).all()):
            raise ValueError(f'{task}: invalid {role} teacher tensor {key}')
    module.load_state_dict(state, strict=True)


class NativeSharedPolicy(nn.Module):
    """One distilled Actor and three native teacher-initialized critics."""
    is_recurrent = False

    def __init__(self, actor, critic_widths, device, *, teacher_priors):
        super().__init__()
        self.actor = actor
        self.critic_widths = _task_tuple(critic_widths)
        if not isinstance(teacher_priors, Mapping) or set(teacher_priors) != set(TASKS):
            raise ValueError('Exact teacher priors for all three tasks are required')
        self.critics = nn.ModuleList()
        self.critic_normalizers = nn.ModuleList()
        self.teacher_checkpoint_sha256 = {}
        stds = []
        fields = {'critic_state_dict','critic_normalizer_state_dict','critic_normalizer_base_width',
                  'action_std','checkpoint_sha256'}
        for task, width in zip(TASKS,self.critic_widths,strict=True):
            prior = teacher_priors[task]
            if not isinstance(prior,Mapping) or not fields.issubset(prior):
                raise ValueError(f'{task}: incomplete teacher prior')
            digest = prior['checkpoint_sha256']
            if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):
                raise ValueError(f'{task}: invalid teacher checkpoint identity')
            self.teacher_checkpoint_sha256[task] = digest
            # Use precisely the same native ELU MLP layout as the independently
            # trained teachers; strict load rejects all shape/key changes.
            critic = MLP(width,1,CRITIC_HIDDEN,activation='elu')
            _load_exact_prior(critic,prior['critic_state_dict'],task,'critic')
            base_width = prior['critic_normalizer_base_width']
            if task == 'run':
                if (type(base_width) is not int or not 0 < base_width < width
                        or (width > 243 and base_width != 243)):
                    raise ValueError('run: teacher AMP critic prefix differs')
                normalizer = NativePrefixNormalization(base_width)
            else:
                if base_width is not None:
                    raise ValueError(f'{task}: native critic normalization must be Identity')
                normalizer = nn.Identity()
            _load_exact_prior(normalizer,prior['critic_normalizer_state_dict'],task,'critic normalizer')
            if task == 'run' and (normalizer.base.count < 0
                    or (normalizer.base._var < 0).any() or (normalizer.base._std < 0).any()):
                raise ValueError('run: invalid teacher critic normalization statistics')
            std = prior['action_std']
            if (not isinstance(std,torch.Tensor) or std.shape != (23,) or std.dtype != critic[0].weight.dtype
                    or not torch.isfinite(std).all() or (std <= 0).any()):
                raise ValueError(f'{task}: teacher action std must be a positive finite native 23-vector')
            self.critics.append(critic)
            self.critic_normalizers.append(normalizer)
            stds.append(std.detach().cpu().clone())
        # Native experts use a direct scalar std parameter per joint. Keep
        # exactly that parameterization and the bitwise teacher values.
        self.std = nn.Parameter(torch.stack(stds))
        self.obs_groups = {'policy': ['policy'], 'critic': ['critic'], 'discriminator': ['disc'],
                           'discriminator_demonstration': ['disc_demo']}
        self.distribution = None
        self.to(device)

    def _task_ids(self, obs):
        ids = obs['task_id']
        if ids.shape != (len(obs),1) or ids.dtype != torch.long:
            raise ValueError('Task identities must be integer indices')
        ids = ids.squeeze(-1)
        if torch.any((ids < 0) | (ids >= len(TASKS))):
            raise ValueError('Unknown task identity')
        return ids

    def _critic_inputs(self, obs):
        native = obs['critic']
        if native.ndim != 2 or native.shape[1] != max(self.critic_widths):
            raise ValueError('Invalid mixed native critic observations')
        return native,self._task_ids(obs)

    def _require_positive_std(self):
        if not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Native learned action std must remain positive and finite')

    def action_std_for(self, obs):
        self._require_positive_std()
        return self.std[self._task_ids(obs)]

    def _update_distribution(self, obs):
        mean = self.actor(obs['policy'])
        self.distribution = Normal(mean,self.action_std_for(obs),validate_args=False)

    def act(self, obs, **kwargs):
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(obs['policy'])

    def evaluate(self, obs, **kwargs):
        native, ids = self._critic_inputs(obs)
        values = native.new_zeros((len(native),1))
        for i, (width,normalizer,critic) in enumerate(zip(self.critic_widths,self.critic_normalizers,
                self.critics,strict=True)):
            rows = ids == i
            values[rows] = critic(normalizer(native[rows,:width]))
        return values

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
            if isinstance(normalizer,NativePrefixNormalization):
                rows = native[ids == i,:width]
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
        self._fixed_std = None
        self.actor_projection = False
        self.actor_gradient_weights = None
        self.inspect_actor_gradients = False
        self._gradient_records = []

    def fix_expert_std(self, tasks):
        """Freeze only the requested exact expert std rows, keeping native PPO."""
        if self._fixed_std is not None or not tasks or not set(tasks).issubset(TASKS):
            raise ValueError('Invalid fixed exploration task selection')
        if self.optimizer.state:
            raise ValueError('Fixed exploration must be configured before optimizer updates or restore')
        row_ids = [TASKS.index(task) for task in tasks]
        parameter = self.policy.std
        mask = torch.ones_like(parameter)
        mask[row_ids] = 0.
        self._fixed_std = (row_ids, parameter.detach()[row_ids].clone())
        parameter.register_hook(lambda grad: grad * mask)

    def verify_fixed_std(self):
        if self._fixed_std is not None:
            row_ids, expected = self._fixed_std
            if not torch.equal(self.policy.std.detach()[row_ids], expected):
                raise ValueError('Frozen expert exploration changed during PPO or resume')

    def _backward_policy_loss(self, loss, obs_batch, surrogate_elements, mirror_pairs, original_batch_size):
        inspect = self.inspect_actor_gradients and not self._gradient_records
        if self.critic_only or not (self.actor_projection or inspect):
            return super()._backward_policy_loss(loss, obs_batch, surrogate_elements, mirror_pairs, original_batch_size)
        params = tuple(self.policy.actor.parameters())
        ids = obs_batch['task_id'][:,0]
        terms = surrogate_elements.reshape(-1)
        if len(terms)!=len(ids): raise ValueError('PPO actor loss/task routing differs')
        mirror = None if mirror_pairs is None else (mirror_pairs[0]-mirror_pairs[1]).square().mean(-1)
        gradients=[]
        for i in range(len(TASKS)):
            mask=ids==i
            if not mask.any(): raise ValueError('A mixed Actor minibatch lacks a task')
            objective=terms[mask].sum()/len(terms)
            if mirror is not None:
                mirror_ids=ids[original_batch_size:]
                if len(mirror)!=len(mirror_ids): raise ValueError('Mirror gradient/task routing differs')
                objective=objective+self.symmetry['mirror_loss_coeff']*mirror[mirror_ids==i].sum()/len(mirror)
            values=torch.autograd.grad(objective,params,retain_graph=True)
            gradients.append(torch.cat([v.detach().reshape(-1) for v in values]))
        flat=torch.stack(gradients)
        if inspect:
            norms=flat.norm(dim=1)
            cosine=(flat@flat.T)/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
            self._gradient_records.append(dict(norms=norms.tolist(),cosine=cosine.tolist()))
        replacement=project_actor_gradients(flat,self.actor_gradient_weights) if self.actor_projection else None
        loss.backward()
        if replacement is not None:
            offset=0
            for p in params:
                p.grad.copy_(replacement[offset:offset+p.numel()].view_as(p));offset+=p.numel()

    def _entropy_regularization(self, obs_batch, entropy_batch):
        return (entropy_batch * self.task_entropy[obs_batch['task_id'].squeeze(-1)]).mean()

    def configure_critic_only(self, enabled):
        self.critic_only = bool(enabled)
        self.policy.train()
        self.policy.actor.requires_grad_(not self.critic_only)
        self.policy.std.requires_grad_(not self.critic_only)
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
        self._gradient_records=[]
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
        self.verify_fixed_std()
        self.policy._require_positive_std()
        result.update(samples=self.storage.num_transitions_per_env * sum(self.counts),
            learning_rate=float(self.learning_rate), mean_action_std=float(self.policy.std.detach().mean()),
            tasks=self._task_diagnostics(), critic_only=bool(critic_only))
        if self._gradient_records:
            result['actor_gradient_sample']=self._gradient_records[0]
        if self.actor_gradient_weights is not None:
            result['actor_gradient_weights']=dict(zip(TASKS,self.actor_gradient_weights,strict=True))
        if self.desired_kl != PPO_NATIVE_CONFIG['desired_kl']:
            result['desired_kl']=float(self.desired_kl)
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
            sigma = self.policy.action_std_for(obs)
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
        for task,returns,std in zip(TASKS,st.returns.split(self.counts,dim=1),self.policy.std,strict=True):
            summary = summaries[task]
            value = torch.stack(summary.pop('value'))
            var = returns.var(unbiased=False)
            variance = (1.-(returns-value).var(unbiased=False)/var) if var > 1.e-10 else None
            summary.update(kl=float(torch.stack(summary['kl']).mean()),
                clip_fraction=float(torch.stack(summary['clip_fraction']).mean()),
                explained_variance=None if variance is None else float(variance),
                mean_action_std=float(std.mean()),min_action_std=float(std.min()),max_action_std=float(std.max()))
            if not all(math.isfinite(v) for v in summary.values() if v is not None):
                raise ValueError(f'Nonfinite {task} diagnostics')
        return summaries


def build_native_learner(actor, counts, critic_widths, amp_cfg, amp_state, symmetry_callback, device,
        *, teacher_priors, steps=24, symmetry_env=None, num_epochs=5, num_minibatches=4, critic_warmup_updates=10,
        desired_kl=PPO_NATIVE_CONFIG['desired_kl']):
    """Create fresh candidate optimizer with exact native teacher learning priors."""
    counts, widths = _task_tuple(counts), _task_tuple(critic_widths)
    if not math.isfinite(desired_kl) or desired_kl <= 0:
        raise ValueError('Adaptive KL target must be positive and finite')
    policy = NativeSharedPolicy(actor,widths,device,teacher_priors=teacher_priors)
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
        device=device,symmetry_cfg=symmetry,amp_cfg=cfg,**dict(PPO_NATIVE_CONFIG,desired_kl=desired_kl))
    alg.amp_discriminator.load_state_dict(amp_state['amp_discriminator_state_dict'], strict=True)
    alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
        amp_state['amp_discriminator_normalizer_state_dict'],strict=True)
    if any(not torch.isfinite(v).all() for v in alg.amp_discriminator.state_dict().values()):
        raise ValueError('Nonfinite native AMP teacher state')
    alg.configure_critic_only(False)
    return alg
