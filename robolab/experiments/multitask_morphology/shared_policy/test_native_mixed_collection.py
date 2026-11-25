from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mixed_finetune import collect_native


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
