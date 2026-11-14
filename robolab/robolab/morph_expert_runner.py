"""Two separate experts: pure PPO for stairs, AMP-PPO for running only."""
from pathlib import Path
import hashlib
import torch
from rsl_rl.algorithms import PPO, PPOAMP
from rsl_rl.storage import CircularBuffer
from rsl_rl.modules import resolve_amp_config
from rsl_rl.runners import AMPRunner, OnPolicyRunner
from robolab.morph_experiments import checkpoint_identity


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError("explicit tensor-only module digest required")
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RunExpertPPOAMP(PPOAMP):
    def __init__(self, policy, storage, *, amp_cfg, device="cpu", **kwargs):
        buffers = [CircularBuffer(amp_cfg["disc_obs_buffer_size"], storage.num_envs, device) for _ in range(2)]
        super().__init__(policy, storage, *buffers, amp_cfg=amp_cfg, device=device, **kwargs)

    def process_env_step(self, obs, rewards, dones, extras):
        # Preserve actual pre-reset discriminator frames, not reset poses.
        transition = extras["amp_transition"]
        if not torch.all(transition["task_ids"] == 0):
            raise RuntimeError("running expert AMP saw stairs data")
        disc, demo = transition["disc"], transition["disc_demo"]
        expected = (len(rewards), self.amp_discriminator.disc_obs_steps, self.amp_discriminator.disc_obs_dim)
        if disc.shape != expected or demo.shape != expected or not torch.isfinite(disc).all() or not torch.isfinite(demo).all():
            raise RuntimeError("invalid actual pre-reset AMP demonstration/transition")
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc, dt=self.amp_cfg["step_dt"])
        self.rewards_lerp = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=self.style_rewards)
        self.disc_obs_buffer.append(disc)
        self.disc_demo_obs_buffer.append(demo)
        PPO.process_env_step(self, obs, self.rewards_lerp, dones, extras)


class _ImmutableExpertCheckpoint:
    def save(self, path, infos=None):
        path = Path(path)
        if path.exists():
            if getattr(self, "_last_saved", None) == (str(path), self.current_learning_iteration):
                return
            raise FileExistsError(path)
        identity = checkpoint_identity(self.bank_sha, self.expert_task, self.current_learning_iteration + 1,
                                       bank=self.bank, design_id=self.design_id)
        if getattr(self, "execution_budget", None) is not None:
            identity["execution_budget"] = self.execution_budget
        state = {"model_state_dict": self.alg.policy.state_dict(), "optimizer_state_dict": self.alg.optimizer.state_dict(),
                 "iter": self.current_learning_iteration, "infos": identity, "learning_rate": self.alg.learning_rate,
                 "torch_rng_state": torch.get_rng_state(),
                 "reward_contract": self.env.unwrapped.routing_contract,
                 "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        if self.expert_task == "run":
            state.update(amp_discriminator_state_dict=self.alg.amp_discriminator.state_dict(),
                amp_discriminator_normalizer_state_dict=self.alg.amp_discriminator.disc_obs_normalizer.state_dict(),
                amp_discriminator_optimizer_state_dict=self.alg.disc_optimizer.state_dict())
        with path.open("xb") as stream:
            torch.save(state, stream)
        self._last_saved = (str(path), self.current_learning_iteration)

    def restore_expert(self, path, *, completed):
        state = torch.load(path, map_location=self.device, weights_only=True)
        info = state.get("infos", {})
        if (info.get("bank_sha256") != self.bank_sha or info.get("task") != self.expert_task
                or info.get("completed_updates") != completed or state.get("iter") != completed - 1):
            raise ValueError("wrong expert, bank, budget or reward definition; not a valid serial continuation")
        from robolab.native_checkpoint_contract import validate_continuation_contract
        validate_continuation_contract(state.get("reward_contract"), self.env.unwrapped.routing_contract,
            bank=self.bank, task=self.expert_task, previous_design=info.get("last_design_id"),
            current_design=self.design_id)
        self.alg.policy.load_state_dict(state["model_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.alg.learning_rate = state["learning_rate"]
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        if self.expert_task == "run":
            self.alg.amp_discriminator.load_state_dict(state["amp_discriminator_state_dict"])
            self.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(state["amp_discriminator_normalizer_state_dict"])
            self.alg.disc_optimizer.load_state_dict(state["amp_discriminator_optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_states"]])
        # Rollout/AMP replay histories and physics reset at a design switch.
        # Discriminator parameters, normalizer and optimizer DO NOT reset.
        self.current_learning_iteration = completed


class MorphRunRunner(_ImmutableExpertCheckpoint, AMPRunner):
    def _construct_algorithm(self, obs):
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        return OnPolicyRunner._construct_algorithm(self, obs)


class MorphStairsRunner(_ImmutableExpertCheckpoint, OnPolicyRunner):
    pass


class NativePPOAMP(PPOAMP):
    """Constructor-only bridge; all reward, update and transition methods native."""
    def __init__(self, policy, storage, *, amp_cfg, device="cpu", **kwargs):
        buffers = [CircularBuffer(amp_cfg["disc_obs_buffer_size"], storage.num_envs, device) for _ in range(2)]
        super().__init__(policy, storage, *buffers, amp_cfg=amp_cfg, device=device, **kwargs)


class NativeMorphRunRunner(_ImmutableExpertCheckpoint, AMPRunner):
    """Use qualified-name construction without legacy AMPRunner's eval()."""
    def _construct_algorithm(self, obs):
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        return OnPolicyRunner._construct_algorithm(self, obs)


class NativeMorphStairsRunner(_ImmutableExpertCheckpoint, OnPolicyRunner):
    """Original Rough PPO algorithm construction in an independent simulator."""
    pass
