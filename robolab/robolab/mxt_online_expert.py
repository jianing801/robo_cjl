"""Opt-in on-policy PPO auxiliary; never an offline replacement for PPO.

The base PPO hook supplies ONLY original, unmirrored minibatch rows. A frozen
source actor anchors both skills on these current rollout states. The original
ROUGH actor labels STAIRS rows using the SAME noisy proprioceptive history;
no clean/privileged teacher sensor is introduced. Teacher actions are targets,
never executed or interpolated. All losses share the original backward/clip.
"""
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from robolab.mxt_pilot import CheckedPilotPPOAMP
from robolab.task_reward_routing import task_ids_from_context
from robolab.multitask_distill import source_teacher_inputs


class OnlineExpertPPOAMP(CheckedPilotPPOAMP):
    def configure_online_expert(self, reference, stairs_teacher, *, anchor_weight,
                                expert_weight, per_task, seed, progress_dir):
        if not 0 < anchor_weight <= 100 or not 0 <= expert_weight <= 1 or not 1 <= per_task <= 128:
            raise ValueError("bounded source anchor, small expert coefficient and batch required")
        self.online_reference = reference.requires_grad_(False).eval()
        self.online_teacher = stairs_teacher
        self.anchor_weight, self.expert_weight = float(anchor_weight), float(expert_weight)
        self.online_batch = per_task
        self.online_generator = torch.Generator(device=self.device).manual_seed(seed)
        self.online_progress = Path(progress_dir)
        self.online_progress.mkdir(exist_ok=False)
        self.online_sums, self.online_calls = {}, 0

    def auxiliary_policy_loss(self, observations):
        x = observations["policy"]
        if x.ndim != 2 or x.shape[1] != 782 or not torch.isfinite(x).all():
            raise ValueError("actual ten-frame proprioception and task ID required")
        ids = task_ids_from_context(x[:, -2:])
        selected, counts = [], []
        for task in (0, 1):
            indices = (ids == task).nonzero().flatten()
            count = min(len(indices), self.online_batch)
            if count:
                order = torch.randperm(len(indices), device=x.device, generator=self.online_generator)[:count]
                selected.append(indices[order])
            counts.append(count)
        if not selected:
            raise ValueError("empty PPO minibatch")
        inputs = x[torch.cat(selected)].detach()
        td = TensorDict({"policy": inputs}, batch_size=[len(inputs)])
        # These inference-only targets must be normal tensors for autograd MSE.
        with torch.no_grad():
            reference = self.online_reference.act_inference(td).detach().clone()
            teacher = (self.online_teacher(source_teacher_inputs(inputs[counts[0]:, :-2], "stairs"))
                       .detach().clone() if counts[1] else None)
        prediction = self.policy.act_inference(td)  # does not overwrite PPO distribution
        if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
            raise RuntimeError("nonfinite online source action")
        terms, offset = [], 0
        metrics = {}
        for name, count in zip(("run", "stairs"), counts):
            if count:
                error = (prediction[offset:offset + count] - reference[offset:offset + count]).square().mean()
                terms.append(error)
                metrics[name + "_source_mse"] = float(error.detach())
            offset += count
        anchor = torch.stack(terms).mean()
        expert = prediction.sum() * 0.
        if teacher is not None:
            if not torch.isfinite(teacher).all() or teacher.shape != prediction[counts[0]:].shape:
                raise RuntimeError("invalid frozen STAIRS teacher target")
            expert = (prediction[counts[0]:] - teacher).square().mean()
        loss = self.anchor_weight * anchor + self.expert_weight * expert
        metrics.update(source_anchor_loss=float((self.anchor_weight * anchor).detach()),
                       expert_mse=float(expert.detach()), expert_loss=float((self.expert_weight * expert).detach()),
                       source_equal_std_gaussian_kl=float(((prediction - reference).square().sum(-1) / .02).mean().detach()))
        for key, value in metrics.items():
            self.online_sums[key] = self.online_sums.get(key, 0.) + value
        self.online_calls += 1
        return loss

    def update(self):
        self.online_sums, self.online_calls = {}, 0
        result = super().update()
        if self.online_calls != self.num_learning_epochs * self.num_mini_batches:
            raise RuntimeError("online auxiliary must accompany every PPO minibatch")
        metrics = {key: value / self.online_calls for key, value in self.online_sums.items()}
        metrics.update(minibatches=self.online_calls, anchor_weight=self.anchor_weight, expert_weight=self.expert_weight)
        total = sum(self.pilot_phase_updates.values())
        with (self.online_progress / f"update_{total:04d}.json").open("x", encoding="utf8") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
        result.update({"online_aux/" + key: value for key, value in metrics.items()})
        return result
