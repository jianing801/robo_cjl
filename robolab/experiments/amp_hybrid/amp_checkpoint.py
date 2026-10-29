"""Complete AMP/PPO stage continuation. Replay/rollout are deliberately not saved."""
from pathlib import Path
import torch


class AmpCheckpointMixin:
    def save(self, path, infos=None):
        path = Path(path)
        if path.exists():
            if getattr(self, "last_saved", None) == (str(path), self.current_learning_iteration):
                return
            raise FileExistsError(path)
        state = dict(model_state_dict=self.alg.policy.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(), iter=self.current_learning_iteration,
            learning_rate=self.alg.learning_rate, hybrid_identity=self.hybrid_identity,
            hybrid_stage=self.hybrid_stage,
            amp_discriminator_state_dict=self.alg.amp_discriminator.state_dict(),
            amp_discriminator_normalizer_state_dict=self.alg.amp_discriminator.disc_obs_normalizer.state_dict(),
            amp_discriminator_optimizer_state_dict=self.alg.disc_optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all())
        with path.open("xb") as stream:
            torch.save(state, stream)
        self.last_saved = str(path), self.current_learning_iteration

    def restore_spatial(self, path):
        from amp_contract import validate_checkpoint
        state = validate_checkpoint(path, self.hybrid_identity, self.hybrid_stage["first_update"])
        self.alg.policy.load_state_dict(state["model_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.alg.amp_discriminator.load_state_dict(state["amp_discriminator_state_dict"], strict=True)
        self.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
            state["amp_discriminator_normalizer_state_dict"], strict=True)
        self.alg.disc_optimizer.load_state_dict(state["amp_discriminator_optimizer_state_dict"])
        self.alg.learning_rate = state["learning_rate"]
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        self.current_learning_iteration = state["iter"]+1
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_states"]])
