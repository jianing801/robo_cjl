"""Trainable shared distilled actor with a fresh PPO value function.

Environment adapters must supply canonical student inputs under policy and
an explicit task-conditioned value observation under critic. No teacher
actor or expert optimizer is restored by this adapter.
"""
import torch
from rsl_rl.modules import ActorCritic
from student import load_student
from contract import sha


class SharedPPOActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, *, student_checkpoint,
                 student_sha256, **kwargs):
        if sha(student_checkpoint)!=student_sha256:
            raise ValueError('Shared actor checkpoint identity mismatch')
        if kwargs.get('actor_obs_normalization',False):
            raise ValueError('Keep the distilled actor internal normalization intact')
        if kwargs.get('state_dependent_std',False):
            raise ValueError('This adapter requires state-independent action noise')
        model,metadata=load_student(student_checkpoint)
        if num_actions!=23 or obs['policy'].shape[-1]!=model.width:
            raise ValueError('Canonical shared actor observation/action contract mismatch')
        super().__init__(obs,obs_groups,num_actions,**kwargs)
        self.actor=model
        self.initial_student_sha256=student_sha256
        self.shared_schema=metadata['schema']
        self.actor_obs_normalization=False

    def get_extra_state(self):
        return dict(kind='three_task_shared_ppo_v1',
                    initial_student_sha256=self.initial_student_sha256,
                    schema=self.shared_schema)

    def set_extra_state(self,value):
        if value!=self.get_extra_state():
            raise ValueError('Shared PPO identity/schema mismatch')
