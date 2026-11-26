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
from finetune_settings import (NATIVE_PPO_ALGORITHM, NATIVE_PPO_SETTINGS,
                               NATIVE_PPO_ALGORITHM_V3, NATIVE_PPO_SETTINGS_V3,
                               NATIVE_PPO_STATE_KIND)
from contract import sha, write_new


def learner():
    actor = torch.nn.Linear(3, 2)
    actor.schema = {'tasks': ['stairs', 'backpack', 'run']}
    actor.architecture = {'hidden_dims': []}
    policy = torch.nn.Module()
    policy.actor = actor
    policy.critics = torch.nn.ModuleList(torch.nn.Linear(width, 1) for width in (4,5,6))
    policy.critic_widths = (4, 5, 6)
    policy.teacher_checkpoint_sha256 = dict(stairs='1'*64, backpack='2'*64, run='3'*64)
    policy.std = torch.nn.Parameter(torch.arange(69, dtype=torch.float32).view(3,23)/100+.1)
    policy.critic_normalizers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.BatchNorm1d(6)])
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
    assert saved['kind'] == NATIVE_PPO_STATE_KIND == 'shared_ppo_learning_state_v4'
    assert saved['teacher_checkpoint_sha256'] == original.policy.teacher_checkpoint_sha256
    assert saved['critic_widths'] == [4,5,6]
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
    save_checkpoint(tmp_path,old.policy.actor,old.policy.critics,old.optimizer,0,'a'*64)
    with pytest.raises(ValueError,match='native|Native'):
        load_native_checkpoint(tmp_path,learner(),'a'*64)


def test_formal_scheme_b_checkpoint_isolated_from_v4(tmp_path):
    from finetune_settings import CURRENT_PPO_ALGORITHM
    original=learner(); update(original)
    digest='e'*64
    for n in range(3):
        receipt=save_native_checkpoint(tmp_path,original,n,digest,ppo_algorithm=CURRENT_PPO_ALGORITHM)
        assert receipt['ppo_algorithm']==CURRENT_PPO_ALGORITHM
    history=checkpoint_history(tmp_path,digest)
    prune_native_checkpoints(tmp_path,history)
    with pytest.raises(ValueError,match='Cannot resume'):
        load_native_checkpoint(tmp_path,learner(),digest)
    assert load_native_checkpoint(tmp_path,learner(),digest,ppo_algorithm=CURRENT_PPO_ALGORITHM)['completed_updates']==2


def test_formal_scheme_b_sparse_checkpoint_keeps_all_metrics_and_restores(tmp_path):
    from finetune_settings import CURRENT_PPO_ALGORITHM
    from finetune_state import save_native_metrics,native_checkpoint_history_for_resume
    original=learner(); update(original); digest='b'*64
    for n in range(12):
        save=save_native_checkpoint if n%10==0 else save_native_metrics
        save(tmp_path,original,n,digest,{'kl':.01},ppo_algorithm=CURRENT_PPO_ALGORITHM)
    history=checkpoint_history(tmp_path,digest)
    assert len(history)==12 and len(list((tmp_path/'checkpoints').glob('*.pt')))==2
    recovered=native_checkpoint_history_for_resume(tmp_path,digest,
        recovery_dir=tmp_path/'attempts'/'b_recovered',ppo_algorithm=CURRENT_PPO_ALGORITHM)
    assert len(recovered)==11 and (tmp_path/'attempts/b_recovered/update_000011.json').is_file()
    assert load_native_checkpoint(tmp_path,learner(),digest,ppo_algorithm=CURRENT_PPO_ALGORITHM)['completed_updates']==10


def test_native_interval_keeps_dense_rewards_and_eleven_full_commits(tmp_path):
    from finetune_state import native_checkpoint_due, save_native_metrics
    from training_reward import mean_reward_rows

    agent = learner()
    digest = 'd'*64
    full = []
    for n in range(101):
        update(agent)
        metrics = {'kl': n/1000}
        reward = {'session_id': 'one', 'mean_reward': {'all': {'mean_reward': n, 'window_episodes': 100}}}
        if native_checkpoint_due(n, 100):
            row = save_native_checkpoint(tmp_path, agent, n, digest, metrics, training_reward=reward)
            full.append(n)
            assert (tmp_path/'checkpoints'/row['path']).is_file()
            prune_native_checkpoints(tmp_path, checkpoint_history(tmp_path,digest), retain_updates=(0,100))
        else:
            row = save_native_metrics(tmp_path, agent, n, digest, metrics, training_reward=reward)
            assert 'path' not in row and 'sha256' not in row
    assert full == [0,10,20,30,40,50,60,70,80,90,100]
    history = checkpoint_history(tmp_path,digest)
    assert len(history) == 101
    assert [r['completed_updates'] for r in history if 'path' in r] == full
    assert len(list((tmp_path/'checkpoints').glob('*.pt'))) == 3
    assert [r['mean_reward'] for r in mean_reward_rows(history, ('stairs','backpack','run'))] == list(range(1,101))
    assert load_native_checkpoint(tmp_path,learner(),digest)['completed_updates'] == 100


def test_native_resume_mixed_old_dense_prefix_preserves_replayed_metrics(tmp_path):
    from finetune_state import native_checkpoint_due, native_checkpoint_history_for_resume, save_native_metrics
    from training_reward import mean_reward_rows

    agent=learner(); digest='e'*64
    for n in range(18):
        update(agent)
        save_native_checkpoint(tmp_path,agent,n,digest,{'old': n})
        prune_native_checkpoints(tmp_path,checkpoint_history(tmp_path,digest))
    for n in (18,19):
        update(agent)
        save_native_metrics(tmp_path,agent,n,digest,{'discarded': n})
    with pytest.raises(ValueError,match='reconcil|rollback'):
        load_native_checkpoint(tmp_path,learner(),digest)
    archived = tmp_path/'attempts'/'recovery'/'receipts'
    prefix = native_checkpoint_history_for_resume(tmp_path,digest,recovery_dir=archived)
    assert len(prefix) == 18 and prefix[-1]['completed_updates'] == 17
    assert sorted(p.name for p in archived.iterdir()) == ['update_000018.json','update_000019.json']
    assert not (tmp_path/'checkpoints/update_000019.json').exists()
    restored=learner()
    assert load_native_checkpoint(tmp_path,restored,digest)['completed_updates'] == 17
    for n in range(18,24):
        update(restored)
        reward={'session_id':'replay','mean_reward':{'all':{'mean_reward':n,'window_episodes':100}}}
        saver=save_native_checkpoint if native_checkpoint_due(n,23) else save_native_metrics
        saver(tmp_path,restored,n,digest,{'replayed':n},training_reward=reward)
        if saver is save_native_checkpoint:
            prune_native_checkpoints(tmp_path,checkpoint_history(tmp_path,digest), retain_updates=(0,23))
    history=checkpoint_history(tmp_path,digest)
    assert len(history)==24
    assert 'discarded' not in history[18]['metrics']
    assert history[19]['metrics']=={'replayed':19}
    assert history[20]['path'].endswith('.pt') and history[23]['path'].endswith('.pt')
    assert [r['update'] for r in mean_reward_rows(history, ('stairs','backpack','run'))] == list(range(1,24))
    assert load_native_checkpoint(tmp_path,learner(),digest)['completed_updates'] == 23


def test_native_metrics_recovery_does_not_hide_corrupted_full_state(tmp_path):
    from finetune_state import native_checkpoint_history_for_resume, save_native_metrics

    agent=learner();digest='f'*64
    row=save_native_checkpoint(tmp_path,agent,0,digest)
    save_native_metrics(tmp_path,agent,1,digest,{'kl':.01})
    (tmp_path/'checkpoints'/row['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checkpoint changed'):
        native_checkpoint_history_for_resume(tmp_path,digest,recovery_dir=tmp_path/'attempts/recovery')
    assert (tmp_path/'checkpoints/update_000001.json').exists()


def test_native_lightweight_updates_do_not_hash_existing_blob(tmp_path, monkeypatch):
    import finetune_state

    agent=learner();digest='1'*64
    record=save_native_checkpoint(tmp_path,agent,0,digest)
    original_blob = tmp_path/'checkpoints'/record['path']
    def forbidden_hash(path):
        pytest.fail(f'Metrics-only update unexpectedly hashes a checkpoint: {path}')
    monkeypatch.setattr(finetune_state,'sha',forbidden_hash)
    finetune_state.save_native_metrics(tmp_path,agent,1,digest,{'kl':.01})
    prune_native_checkpoints(tmp_path,checkpoint_history(tmp_path,digest))
    assert original_blob.is_file()
    assert list((tmp_path/'checkpoints').glob('*.pt')) == [original_blob]
    with pytest.raises(ValueError,match='scheduled full checkpoint'):
        finetune_state.save_native_metrics(tmp_path,agent,10,digest,{'kl':.01})


def test_native_settings_match_joint_ppo_contract():
    assert NATIVE_PPO_SETTINGS['critic_warmup_updates']==10
    assert NATIVE_PPO_ALGORITHM == 'shared_native_ppoamp_expert_prior_v4'
    assert NATIVE_PPO_SETTINGS['noise_initialization']=='frozen_expert_learned_std'
    assert NATIVE_PPO_SETTINGS['noise_std_shape'] == [3,23]
    assert NATIVE_PPO_SETTINGS['critic_initialization'] == 'frozen_expert_critic'
    assert NATIVE_PPO_SETTINGS['critic_architecture'] == 'three_independent_teacher_native'
    assert NATIVE_PPO_SETTINGS['critic_observation_normalization'] == dict(stairs='identity',backpack='identity',run='teacher_native_prefix')
    assert NATIVE_PPO_SETTINGS_V3['initial_noise_std'] == 1.
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
    with pytest.raises(ValueError,match='different PPO algorithms'):
        objective(dict(old_report, ppo_algorithm=NATIVE_PPO_ALGORITHM_V3), native_scoring)


def test_native_v4_resume_rejects_v3_receipt(tmp_path):
    folder = tmp_path/'checkpoints'; folder.mkdir()
    write_new(folder/'update_000000.json',dict(completed_updates=0,manifest_sha256='b'*64,
        ppo_algorithm=NATIVE_PPO_ALGORITHM_V3,path='unused.pt',sha256='0'*64))
    with pytest.raises(ValueError,match='native|Native'):
        load_native_checkpoint(tmp_path,learner(),'b'*64)


def test_native_v4_rejects_changed_teacher_checkpoint(tmp_path):
    first=learner(); update(first)
    save_native_checkpoint(tmp_path,first,0,'c'*64)
    other=learner(); other.policy.teacher_checkpoint_sha256['run']='4'*64
    with pytest.raises(ValueError,match='identity'):
        load_native_checkpoint(tmp_path,other,'c'*64)


def test_v3_actor_remains_verifiable_but_not_v4(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bo
    root=tmp_path.resolve(); actor=root/'shared_actor_finetuned.pt'; actor.write_bytes(b'v3 actor')
    candidate=dict(structure_key=[.25,.3,'k','a'])
    manifest=dict(plan_sha256='p',initial_actor_sha256='i',candidate=candidate,updates=100,
                  ppo_schedule=bo.PPO_SCHEDULE,envs_per_task=bo.MIXED_BO_LAYOUT,init_at_random_ep_len=True,
                  ppo_algorithm=NATIVE_PPO_ALGORITHM_V3,native_ppo_settings=NATIVE_PPO_SETTINGS_V3)
    write_new(root/'manifest.json',manifest)
    folder=root/'checkpoints'; folder.mkdir(); blob=folder/'valid.pt'; blob.write_bytes(b'v3 blob')
    mh=sha(root/'manifest.json'); receipt=folder/'update_000100.json'
    write_new(receipt,dict(completed_updates=100,manifest_sha256=mh,path=blob.name,sha256=sha(blob),
        ppo_algorithm=NATIVE_PPO_ALGORITHM_V3,warmup_completed=10))
    write_new(root/'training_completed.json',dict(result='CANDIDATE_FINETUNED',candidate=candidate,
        ppo_updates=100,manifest_sha256=mh,student=str(actor),student_sha256=sha(actor),checkpoint_sha256=sha(receipt)))
    assert bo.completed_actor_from_hashes(root,'p','i',candidate,100,mixed_bo=True,
        ppo_algorithm=NATIVE_PPO_ALGORITHM_V3)==actor
    with pytest.raises(ValueError,match='algorithm'):
        bo.completed_actor_from_hashes(root,'p','i',candidate,100,mixed_bo=True,ppo_algorithm=NATIVE_PPO_ALGORITHM)

def test_native_hundred_update_interval_keeps_every_requested_full_checkpoint(tmp_path):
    from finetune_state import native_checkpoint_due, save_native_metrics
    from training_reward import mean_reward_rows

    assert [i for i in range(1001) if native_checkpoint_due(
        i, 1000, checkpoint_interval=100)] == list(range(0, 1001, 100))
    assert native_checkpoint_due(203, 203, checkpoint_interval=100)
    assert native_checkpoint_due(10, 100)  # Default BO cadence is unchanged.

    agent=learner(); digest='7'*64
    full=[]
    for n in range(304):
        reward={'session_id':'experiment','mean_reward':{'all':{'mean_reward':n,'window_episodes':100}}}
        if native_checkpoint_due(n, 303, checkpoint_interval=100):
            row=save_native_checkpoint(tmp_path,agent,n,digest,{'kl':.01},training_reward=reward)
            full.append(n)
            prune_native_checkpoints(tmp_path,checkpoint_history(tmp_path,digest),
                retain_updates=(0,100,200,300,303),checkpoint_interval=100)
        else:
            row=save_native_metrics(tmp_path,agent,n,digest,{'kl':.01},training_reward=reward,
                                    checkpoint_interval=100)
            assert 'path' not in row
    history=checkpoint_history(tmp_path,digest)
    assert len(history)==304 and full==[0,100,200,300,303]
    assert [r['completed_updates'] for r in history if 'path' in r]==full
    assert all((tmp_path/'checkpoints'/history[n]['path']).is_file() for n in full)
    assert len(list((tmp_path/'checkpoints').glob('*.pt')))==len(full)
    assert [r['mean_reward'] for r in mean_reward_rows(history,('stairs','backpack','run'))]==list(range(1,304))
    assert load_native_checkpoint(tmp_path,learner(),digest,checkpoint_interval=100)['completed_updates']==303


def test_native_hundred_update_interval_recovers_99_uncommitted_metric_rows(tmp_path):
    from finetune_state import native_checkpoint_history_for_resume, save_native_metrics

    agent=learner(); digest='8'*64
    save_native_checkpoint(tmp_path,agent,0,digest)
    for n in range(1,100):
        save_native_metrics(tmp_path,agent,n,digest,{'kl':.02},checkpoint_interval=100)
    # The interval is part of the training manifest; accidentally using the
    # default would encounter a metrics-only receipt at a required save index.
    archive=tmp_path/'attempts'/'recovered'
    with pytest.raises(ValueError,match='metrics-only'):
        native_checkpoint_history_for_resume(tmp_path,digest,recovery_dir=archive)
    with pytest.raises(ValueError,match='reconcil'):
        load_native_checkpoint(tmp_path,learner(),digest,checkpoint_interval=100)
    assert not archive.exists()
    prefix=native_checkpoint_history_for_resume(tmp_path,digest,recovery_dir=archive,checkpoint_interval=100)
    assert len(prefix)==1 and len(list(archive.glob('*.json')))==99
    assert load_native_checkpoint(tmp_path,learner(),digest,checkpoint_interval=100)['completed_updates']==0
    for n in range(1,100):
        save_native_metrics(tmp_path,agent,n,digest,{'kl':.03},checkpoint_interval=100)
    with pytest.raises(ValueError,match='scheduled full checkpoint'):
        save_native_metrics(tmp_path,agent,100,digest,checkpoint_interval=100)
    save_native_checkpoint(tmp_path,agent,100,digest)
    assert checkpoint_history(tmp_path,digest)[99]['metrics']=={'kl':.03}
    assert load_native_checkpoint(tmp_path,learner(),digest,checkpoint_interval=100)['completed_updates']==100


@pytest.mark.parametrize('bad_interval',[0,-1,True,1.5,None])
def test_native_checkpoint_interval_rejects_invalid_values(tmp_path,bad_interval):
    from finetune_state import native_checkpoint_due, save_native_metrics, native_checkpoint_history_for_resume

    for call in (
        lambda: native_checkpoint_due(3,100,checkpoint_interval=bad_interval),
        lambda: save_native_metrics(tmp_path,learner(),3,'9'*64,checkpoint_interval=bad_interval),
        lambda: native_checkpoint_history_for_resume(tmp_path,'9'*64,
            recovery_dir=tmp_path/'attempts'/'recovered',checkpoint_interval=bad_interval),
        lambda: load_native_checkpoint(tmp_path,learner(),'9'*64,checkpoint_interval=bad_interval),
    ):
        with pytest.raises(ValueError,match='checkpoint interval'):
            call()
