"""CPU proof of the native learner's complete committed state and BO isolation."""
from pathlib import Path
from types import SimpleNamespace
import random
import sys

import numpy as np
import pytest
import torch

from finetune_state import (checkpoint_history, load_native_checkpoint,
                            prune_native_checkpoints, save_checkpoint, save_native_checkpoint)
from finetune_settings import NATIVE_PPO_ALGORITHM, NATIVE_PPO_SETTINGS
from contract import sha, write_new


def learner():
    actor = torch.nn.Linear(3, 2)
    actor.schema = {'tasks': ['stairs', 'backpack', 'run']}
    actor.architecture = {'hidden_dims': []}
    policy = torch.nn.Module()
    policy.actor = actor
    policy.critic = torch.nn.Linear(4, 1)
    policy.log_std = torch.nn.Parameter(torch.zeros(2))
    policy.register_buffer('normalization_count', torch.ones(1))
    discriminator = torch.nn.Linear(4, 1)
    discriminator.register_buffer('normalization_count', torch.ones(1))
    def ring():
        return SimpleNamespace(_buffer=torch.randn(5, 2, 4), _num_pushes=torch.full((2,), 7),
                               _pointer=1, _batch_size=2, _device='cpu', max_length=5)
    instance = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=1e-4),
        amp_discriminator=discriminator,
        disc_optimizer=torch.optim.Adam(discriminator.parameters(), lr=1e-4),
        learning_rate=1e-4, disc_obs_buffer=ring(), disc_demo_obs_buffer=ring(), counts=(2,2,2))
    return instance


def update(instance):
    instance.optimizer.zero_grad()
    (sum(p.square().sum() for p in instance.policy.parameters())).backward()
    instance.optimizer.step()
    instance.disc_optimizer.zero_grad()
    (sum(p.square().sum() for p in instance.amp_discriminator.parameters())).backward()
    instance.disc_optimizer.step()
    instance.learning_rate = .0003
    for group in instance.optimizer.param_groups:
        group['lr'] = instance.learning_rate


def test_native_checkpoint_roundtrip_and_prune(tmp_path):
    original=learner(); update(original)
    digest='0'*64
    for n in range(5):
        record=save_native_checkpoint(tmp_path,original,n,digest,{'kl':.01},warmup_completed=10)
        assert record['metrics']['kl']==.01 and record['ppo_algorithm']==NATIVE_PPO_ALGORITHM
        update(original)
    history=checkpoint_history(tmp_path,digest)
    prune_native_checkpoints(tmp_path,history,retain_updates=(0,2))
    assert all((tmp_path/'checkpoints'/history[i]['path']).exists() for i in (0,2,3,4))
    assert not (tmp_path/'checkpoints'/history[1]['path']).exists()
    # Learning state, optimizer moments, current adaptive LR, native AMP replay and random streams survive.
    saved=torch.load(tmp_path/'checkpoints'/history[-1]['path'],weights_only=True)
    restored=learner()
    result=load_native_checkpoint(tmp_path,restored,digest)
    assert result==dict(completed_updates=4,warmup_completed=10)
    for name,value in restored.policy.state_dict().items():
        assert torch.equal(value,saved['policy'][name])
    for name,value in restored.amp_discriminator.state_dict().items():
        assert torch.equal(value,saved['amp_discriminator'][name])
    assert restored.learning_rate==restored.optimizer.param_groups[0]['lr']==.0003
    assert torch.equal(restored.disc_obs_buffer._buffer,saved['disc_obs_buffer']['buffer'])
    assert restored.disc_obs_buffer._pointer==1
    assert restored.optimizer.state and restored.disc_optimizer.state
    rng=(torch.rand(4),np.random.random(4),random.random())
    load_native_checkpoint(tmp_path,restored,digest)
    assert torch.equal(rng[0],torch.rand(4)) and np.array_equal(rng[1],np.random.random(4)) and rng[2]==random.random()
    assert len(checkpoint_history(tmp_path,digest))==5


def test_native_resume_rejects_legacy_learning_state(tmp_path):
    old=learner()
    save_checkpoint(tmp_path,old.policy.actor,old.policy.critic,old.optimizer,0,'a'*64)
    with pytest.raises(ValueError,match='native|Native'):
        load_native_checkpoint(tmp_path,learner(),'a'*64)


def test_native_settings_match_joint_ppo_contract():
    assert NATIVE_PPO_SETTINGS['critic_warmup_updates']==10
    assert NATIVE_PPO_SETTINGS['initial_noise_std']==1.
    assert NATIVE_PPO_SETTINGS['desired_kl']==.01
    assert NATIVE_PPO_SETTINGS['adaptive_learning_rate_bounds']==[1e-5,1e-2]
    assert NATIVE_PPO_SETTINGS['entropy_coefficients']=={'stairs':.005,'backpack':.005,'run':.01}


def test_native_bo_actor_protocol_never_accepts_legacy(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bo
    root=tmp_path.resolve(); actor=root/'shared_actor_finetuned.pt'; actor.write_bytes(b'old')
    candidate=dict(structure_key=[.25,.3,'k','a'])
    manifest=dict(plan_sha256='p',initial_actor_sha256='i',candidate=candidate,updates=100,
                  ppo_schedule=bo.PPO_SCHEDULE,envs_per_task=bo.MIXED_BO_LAYOUT,init_at_random_ep_len=True)
    write_new(root/'manifest.json',manifest)
    folder=root/'checkpoints'; folder.mkdir(); blob=folder/'valid.pt'; blob.write_bytes(b'blob')
    mh=sha(root/'manifest.json'); receipt=folder/'update_000100.json'
    write_new(receipt,dict(completed_updates=100,manifest_sha256=mh,path=blob.name,sha256=sha(blob)))
    write_new(root/'training_completed.json',dict(result='CANDIDATE_FINETUNED',candidate=candidate,
        ppo_updates=100,manifest_sha256=mh,student=str(actor),student_sha256=sha(actor),checkpoint_sha256=sha(receipt)))
    assert bo.completed_actor_from_hashes(root,'p','i',candidate,100,mixed_bo=True)==actor
    with pytest.raises(ValueError,match='algorithm'):
        bo.completed_actor_from_hashes(root,'p','i',candidate,100,mixed_bo=True,ppo_algorithm=NATIVE_PPO_ALGORITHM)


def test_native_bo_does_not_score_mixed_observations():
    from shared_bo_metrics import objective
    native_scoring = {'benchmark': {'ppo_algorithm': NATIVE_PPO_ALGORITHM}}
    old_report = {'domain': 'simulation'}
    with pytest.raises(ValueError,match='different PPO algorithms'):
        objective(old_report, native_scoring)
    with pytest.raises(ValueError,match='different PPO algorithms'):
        objective(dict(old_report, ppo_algorithm=NATIVE_PPO_ALGORITHM), {'benchmark': {}})
