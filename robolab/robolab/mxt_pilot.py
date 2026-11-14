"""Bounded actor/critic preparation around the original task-routed PPOAMP.

No new rewards, losses or task selection. Frozen actor during value/disc
preparation, then low-LR actor updates with explicitly fixed low exploration.
This is an isolated pilot, not automatic long-running production training.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from robolab.task_routed_amp import TaskRoutedPPOAMP, TaskRoutedAMPRunner


def require_bc_initialization(source):
    if "task_reward_routing" in source:
        raise ValueError("bounded preparation requires unrouted BC initialization, not resetting a routed pilot's optimizer/phase/budget")
    if (source.get("infos", {}).get("initialization") != "balanced_behavior_cloning_from_rpo_amp_run_and_rpo_rough_stairs"
            or source.get("model_state_dict", {}).get("_extra_state", {}).get("kind") != "rpo_mxt"):
        raise ValueError("MxT BC metadata required; random/smoke/MLP policies are not pilot baselines")


def actor_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.actor.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError("actor fingerprint expects explicit tensor state")
        digest.update(name.encode("utf8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_pilot_phase(algorithm, *, phase, noise_std, learning_rate):
    if phase not in ("critic_discriminator_only", "actor_pilot"):
        raise ValueError("unknown bounded pilot phase")
    if not 0 < noise_std <= .2 or not 0 < learning_rate <= .001:
        raise ValueError("pilot requires explicit bounded exploration and learning rate")
    policy = algorithm.policy
    if not isinstance(getattr(policy, "std", None), torch.nn.Parameter):
        raise ValueError("pilot requires the original independent scalar-type std parameter")
    policy.actor.requires_grad_(phase == "actor_pilot")
    policy.critic.requires_grad_(True)
    with torch.no_grad():
        policy.std.fill_(noise_std)
    policy.std.requires_grad_(False)
    policy.std.grad = None
    # Clear stale gradients when switching. Parameters remain in the SAME
    # original optimizer group/order for ordinary RSL state compatibility.
    algorithm.optimizer.zero_grad(set_to_none=True)
    algorithm.schedule = "fixed"
    algorithm.learning_rate = learning_rate
    for group in algorithm.optimizer.param_groups:
        group["lr"] = learning_rate
    algorithm.pilot_phase = phase


class CheckedPilotPPOAMP(TaskRoutedPPOAMP):
    def update(self):
        losses = super().update()
        if any(not math.isfinite(float(value)) for value in losses.values()):
            raise RuntimeError("nonfinite pilot loss; stop without replacing baseline")
        if any(not torch.isfinite(value).all() for value in self.policy.parameters()):
            raise RuntimeError("nonfinite pilot policy; stop without replacing baseline")
        if any(not torch.isfinite(value).all() for value in self.amp_discriminator.state_dict().values()
               if isinstance(value, torch.Tensor) and value.is_floating_point()):
            raise RuntimeError("nonfinite pilot discriminator")
        phase = self.pilot_phase
        self.pilot_phase_updates[phase] += 1
        total = sum(self.pilot_phase_updates.values())
        progress = {"completed_update_calls": total, "phase": phase,
                    "phase_updates": dict(self.pilot_phase_updates), "learning_rate": self.learning_rate,
                    "fixed_action_noise_std": self.policy.std.detach().cpu().tolist(),
                    "losses": {key: float(value) for key, value in losses.items()},
                    "locomotion_quality_not_proven_by_losses": True}
        if self.pilot_progress_dir is not None:
            with (Path(self.pilot_progress_dir) / f"update_{total:04d}.json").open("x", encoding="utf8") as stream:
                json.dump(progress, stream, indent=2, allow_nan=False)
        if total % 10 == 0 or total == 1:
            print("MXT_PILOT_UPDATE " + json.dumps(progress), flush=True)
        return losses


class PilotAMPRunner(TaskRoutedAMPRunner):
    def save(self, path, infos=None):
        metadata = {"bounded_mxt_pilot": dict(self.pilot_manifest), "phase": self.alg.pilot_phase,
                    "phase_updates": dict(self.alg.pilot_phase_updates),
                    "exploration_std_is_frozen": True,
                    "resume_requires_reapplying_this_pilot_phase_and_fixed_noise": True}
        return super().save(path, {**(infos or {}), **metadata})
