"""MxT PPO with RUN-only AMP, reusing RoboParty's discriminator and PPOAMP update.

PPO sees BOTH tasks. Discriminator replay, normalization and style rewards see
ONLY RUN, including the physical pre-reset terminal frame. Original runners
and algorithms are untouched. Task assignments are fixed per environment.
"""

import copy
import os
import torch
from rsl_rl.algorithms import PPO, PPOAMP
from rsl_rl.modules import resolve_amp_config
from rsl_rl.runners import AMPRunner, OnPolicyRunner
from rsl_rl.storage import CircularBuffer

from robolab.task_reward_routing import ROUTING_VERSION, mix_run_style, task_ids_from_context


class TaskRoutedPPOAMP(PPOAMP):
    def __init__(self, policy, storage, *, amp_cfg, device="cpu", **kwargs):
        amp_cfg = copy.deepcopy(amp_cfg)
        self.run_ids = torch.tensor(amp_cfg.pop("run_env_ids"), device=device, dtype=torch.long)
        self.expected_task_ids = torch.ones(storage.num_envs, device=device, dtype=torch.long)
        self.expected_task_ids[self.run_ids] = 0
        if not 0 < len(self.run_ids) < storage.num_envs or len(self.run_ids.unique()) != len(self.run_ids):
            raise ValueError("routed AMP requires nonempty, disjoint RUN and STAIRS environments")
        minibatches = kwargs.get("num_mini_batches", 4)
        if len(self.run_ids) * storage.num_transitions_per_env % minibatches:
            raise ValueError("RUN rollout size must be divisible by discriminator mini-batches")
        buffers = [CircularBuffer(amp_cfg["disc_obs_buffer_size"], len(self.run_ids), device) for _ in range(2)]
        super().__init__(policy, storage, *buffers, amp_cfg=amp_cfg, device=device, **kwargs)

    def process_env_step(self, obs, rewards, dones, extras):
        task_ids = task_ids_from_context(obs["policy"][:, -2:]).to(self.device)
        if not torch.equal(task_ids, self.expected_task_ids):
            raise ValueError("environment task assignment changed inside a routed AMP rollout")
        # A reset observation is NOT the frame that generated reward/done.
        transition = extras.get("amp_transition")
        if transition is None:
            raise ValueError("routed AMP requires physical pre-reset amp_transition, never reset observations")
        if not torch.equal(transition["task_ids"].to(self.device), task_ids):
            raise ValueError("AMP transition task IDs disagree with actor context")
        disc = transition["disc"].to(self.device)[self.run_ids].contiguous()
        demo = transition["disc_demo"].to(self.device)[self.run_ids].contiguous()
        expected = (len(self.run_ids), self.amp_discriminator.disc_obs_steps, self.amp_discriminator.disc_obs_dim)
        if disc.shape != expected or demo.shape != expected or not torch.isfinite(disc).all() or not torch.isfinite(demo).all():
            raise ValueError("invalid RUN discriminator transition/demonstration")
        style, score = self.amp_discriminator.predict_style_reward(disc, dt=self.amp_cfg["step_dt"])
        self.style_rewards = torch.zeros_like(rewards)
        self.disc_score = torch.zeros_like(rewards)
        self.style_rewards[self.run_ids], self.disc_score[self.run_ids] = style, score
        self.rewards_lerp = mix_run_style(rewards, self.style_rewards, task_ids,
                                         self.amp_discriminator.task_style_lerp)
        self.disc_obs_buffer.append(disc)
        self.disc_demo_obs_buffer.append(demo)
        # Skip PPOAMP.process_env_step (it would append stairs and mix twice).
        PPO.process_env_step(self, obs, self.rewards_lerp, dones, extras)


class TaskRoutedAMPRunner(AMPRunner):
    def _construct_algorithm(self, obs):
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        ids = task_ids_from_context(obs["policy"][:, -2:])
        self.alg_cfg["amp_cfg"]["run_env_ids"] = (ids == 0).nonzero(as_tuple=False).flatten().cpu().tolist()
        # The generic runner resolves fully qualified classes, unlike the legacy
        # AMP runner's eval(). The custom algorithm constructs RUN-only buffers.
        return OnPolicyRunner._construct_algorithm(self, obs)

    def _routing_contract(self):
        return {"version": ROUTING_VERSION, "task_style_lerp": self.alg.amp_discriminator.task_style_lerp,
                "style_reward_scale": self.alg.amp_discriminator.style_reward_scale,
                "disc_steps": self.alg.amp_discriminator.disc_obs_steps,
                "disc_dim": self.alg.amp_discriminator.disc_obs_dim,
                "environment": self.env.unwrapped.routing_contract}

    def save(self, path, infos=None):
        # Same original AMP checkpoint fields, plus a strict routing contract.
        if os.path.exists(path):
            if getattr(self, "_last_saved", None) == (path, self.current_learning_iteration):
                return  # legacy learn() final-save repeats the last interval-save
            raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
        state = {"model_state_dict": self.alg.policy.state_dict(),
                 "optimizer_state_dict": self.alg.optimizer.state_dict(),
                 "iter": self.current_learning_iteration, "infos": infos,
                 "amp_discriminator_state_dict": self.alg.amp_discriminator.state_dict(),
                 "amp_discriminator_normalizer_state_dict": self.alg.amp_discriminator.disc_obs_normalizer.state_dict(),
                 "amp_discriminator_optimizer_state_dict": self.alg.disc_optimizer.state_dict(),
                 "task_reward_routing": self._routing_contract(),
                 "learning_rate": self.alg.learning_rate}
        if self.alg.rnd:
            raise ValueError("RND is not supported by this explicit two-task checkpoint contract")
        with open(path, "xb") as stream:
            torch.save(state, stream)
        self._last_saved = (path, self.current_learning_iteration)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True, map_location=None):
        state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        route = state.get("task_reward_routing")
        if route is not None and route != self._routing_contract():
            raise ValueError("checkpoint reward/demo routing contract differs from environment")
        if route is None:
            if load_optimizer or "amp_discriminator_state_dict" in state:
                raise ValueError("legacy/unrouted checkpoint is not a routed AMP resume")
            # Explicit distillation warm-start only: actor/critic load, fresh
            # discriminator and optimizers. Never mislabel this as full resume.
            self.alg.policy.load_state_dict(state["model_state_dict"])
            self.current_learning_iteration = 0
            print("[task_routed_amp] Distilled student warm-start; AMP discriminator is fresh", flush=True)
            return state.get("infos")
        result = super().load(path, load_optimizer, map_location)
        if load_optimizer:
            self.alg.learning_rate = state["learning_rate"]
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate
            # Saved iter is the last COMPLETED update, not the next one.
            self.current_learning_iteration = int(state["iter"]) + 1
        return result
