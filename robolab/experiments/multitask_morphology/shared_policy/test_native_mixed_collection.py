from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mixed_finetune import collect_native, verify_expert_initialization


def test_mixed_collection_uses_learner_action_and_only_amp_discriminator_rows(tmp_path):
    counts = (2, 1, 1)
    tasks = [SimpleNamespace(name=name, base=torch.ones(n, 2),
                             observation={'critic': torch.full((n, 3), i),
                                          'disc': torch.ones(n, 3, 67),
                                          'disc_demo': torch.ones(n, 3, 67)},
                             raw=SimpleNamespace(cfg=SimpleNamespace(decimation=1)))
             for i, (name, n) in enumerate(zip(('stairs', 'backpack', 'run'), counts))]
    calls = []
    for i, (task, count) in enumerate(zip(tasks, counts)):
        task.begin = lambda action, i=i, count=count: calls.append(('action', i, tuple(action.shape)))
        task.apply = task.sync = lambda: None
        task.finish = lambda i=i, count=count: dict(environment_reward=torch.full((count,), float(i+1)),
                                                     done=torch.zeros(count, dtype=torch.bool),
                                                     timeout=torch.zeros(count, dtype=torch.bool))
    adapter = SimpleNamespace(pack=lambda name, base, critic: TensorDict(dict(policy=base, critic=critic), batch_size=[len(base)]),
                              concat=lambda parts: torch.cat(parts))
    class Learner:
        def act(self, obs):
            return torch.zeros(len(obs), 23)

        def process_mixed_env_step(self, obs, raw_reward, done, timeout, amp_obs):
            assert amp_obs['disc'].shape == (1, 3, 67)
            calls.append(('reward', raw_reward.tolist()))
            result = raw_reward.clone(); result[-1] *= 2
            return result

    batches, final = collect_native(None, SimpleNamespace(step=lambda **kwargs: None),
                                    tasks, counts, Learner(), adapter, tmp_path, steps=2)
    assert len(calls) == 8
    assert [b['environment_reward'].shape for b in batches] == [(2, 2), (2, 1), (2, 1)]
    assert batches[2]['reward'].tolist() == [[6.0], [6.0]]
    assert batches[2]['environment_reward'].tolist() == [[3.0], [3.0]]
    assert final['policy'].shape == (4, 2)


def test_real_teacher_prior_parity_checks_each_critic_and_std():
    import pytest
    names=('stairs','backpack','run')
    sigmas=torch.stack([torch.full((23,),value) for value in (.2,.3,.4)])
    tasks=[]
    for i,name in enumerate(names):
        critic=torch.tensor([[float(i+1),2.]])
        prior=dict(action_std=sigmas[i].clone(),checkpoint_sha256=str(i)*64,
                   critic_normalizer_base_width=1 if i==2 else None)
        tasks.append(SimpleNamespace(name=name,base=torch.zeros(1,2),observation={'critic':critic},
                                     training_prior={'expert_prior':prior,'teacher_value':critic[:,:1].clone()}))
    adapter=SimpleNamespace(pack=lambda name,base,critic: {'critic':critic})
    learner=SimpleNamespace(policy=SimpleNamespace(std=sigmas,evaluate=lambda obs:obs['critic'][:,:1]))
    audit=verify_expert_initialization(tasks,learner,adapter,{t.name:t.training_prior['expert_prior']['checkpoint_sha256'] for t in tasks})
    assert audit['run']['critic_max_abs_error']==0.
    assert audit['backpack']['action_std_mean']==pytest.approx(.3)
    tasks[-1].training_prior['teacher_value'] += 1.
    with pytest.raises(ValueError,match='critic'):
        verify_expert_initialization(tasks,learner,adapter,{t.name:t.training_prior['expert_prior']['checkpoint_sha256'] for t in tasks})
