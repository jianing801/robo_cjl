"""CPU integration tests of the native mixed PPO learner (no simulator)."""
import sys
from pathlib import Path
from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'rsl_rl'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'amp_engine'))


class Actor(nn.Module):
    width = 5

    def __init__(self):
        super().__init__()
        self.network = nn.Linear(self.width, 23)

    def forward(self, obs):
        return self.network(obs)


def teacher_priors(widths=(4,6,3)):
    """Synthetic complete native expert state, with distinguishable task stds."""
    from amp_policy import PrefixNormalization
    from rsl_rl.networks import MLP
    result={}
    for i,(task,width) in enumerate(zip(('stairs','backpack','run'),widths,strict=True)):
        critic=MLP(width,1,[512,256,128],'elu')
        base_width=(243 if width==271 else width-1) if task=='run' else None
        normalizer=PrefixNormalization(base_width) if base_width else nn.Identity()
        if base_width:
            normalizer.update(torch.randn(5,width)*2.+1.)
        result[task]=dict(critic_state_dict=deepcopy(critic.state_dict()),
            critic_normalizer_state_dict=deepcopy(normalizer.state_dict()),
            critic_normalizer_base_width=base_width,
            action_std=torch.linspace(.1+.1*i,.2+.1*i,23),checkpoint_sha256=str(i+1)*64)
    return result


def observation(counts=(2, 2, 2), widths=(4, 6, 3)):
    n = sum(counts)
    return TensorDict(dict(policy=torch.randn(n, 5), critic=torch.randn(n, max(widths)),
        task_id=torch.arange(3).repeat_interleave(torch.tensor(counts)).reshape(-1, 1)), [n])


def amp_settings():
    from rsl_rl.modules.amp import AMPDiscriminator
    cfg = dict(disc_obs_dim=2, disc_obs_steps=3, step_dt=.02,
        disc_obs_buffer_size=8, loss_type='LSGAN', grad_penalty_scale=10.,
        disc_trunk_weight_decay=.001, disc_linear_weight_decay=.1,
        disc_learning_rate=.0001, disc_max_grad_norm=1.,
        amp_discriminator=dict(hidden_dims=[8], activation='elu', style_reward_scale=1.5,
                               task_style_lerp=.6))
    disc = AMPDiscriminator(2, 3, {'discriminator': ['disc'], 'discriminator_demonstration': ['disc_demo']},
                            **cfg['amp_discriminator'])
    with torch.no_grad():
        disc.disc_linear.bias.fill_(.42)
    state = dict(amp_discriminator_state_dict=disc.state_dict(),
                 amp_discriminator_normalizer_state_dict=disc.disc_obs_normalizer.state_dict())
    return cfg, state


def amp_obs(n=2, value=1.):
    return TensorDict(dict(disc=torch.full((n, 3, 2), value),
                           disc_demo=torch.full((n, 3, 2), 2.)), [n])


def test_shared_actor_unchanged_by_privileged_critic():
    from native_ppo import NativeSharedPolicy
    actor = Actor()
    priors=teacher_priors()
    policy = NativeSharedPolicy(actor, (4, 6, 3), 'cpu',teacher_priors=priors)
    obs = observation()
    original = actor(obs['policy'])
    torch.testing.assert_close(policy.act_inference(obs), original, rtol=0, atol=0)
    changed = obs.clone(); changed['critic'] += 2.
    torch.testing.assert_close(policy.act_inference(changed), original, rtol=0, atol=0)
    assert not torch.allclose(policy.evaluate(obs), policy.evaluate(changed))
    assert policy.evaluate(obs).shape == (6, 1)
    policy.act(obs)
    expected_std=torch.stack([priors[task]['action_std'] for task in ('stairs','backpack','run')]).repeat_interleave(2,dim=0)
    torch.testing.assert_close(policy.action_std,expected_std,rtol=0,atol=0)
    assert policy.std.requires_grad and policy.std.shape==(3,23)
    policy.update_normalization(obs)
    assert isinstance(policy.critic_normalizers[0],nn.Identity)
    assert isinstance(policy.critic_normalizers[1],nn.Identity)
    assert int(policy.critic_normalizers[2].base.count)==7
    pad_edit = obs.clone(); pad_edit['critic'][:2,4:] += 1000.
    torch.testing.assert_close(policy.evaluate(obs),policy.evaluate(pad_edit),atol=0,rtol=0)


def learner(symmetry=None):
    from native_ppo import build_native_learner
    cfg, state = amp_settings()
    return build_native_learner(Actor(), (2,2,2), (4,6,3), cfg, state, symmetry, 'cpu',
                               teacher_priors=teacher_priors(),steps=2, num_epochs=1, num_minibatches=1)


def fill_rollout(alg):
    obs = observation()
    for i in range(2):
        with torch.inference_mode():
            alg.act(obs)
            following = observation()
            raw = torch.arange(1.,7.)
            done = torch.tensor([False,True,False,False,True,False])
            timeout = torch.tensor([False,True,False,False,False,False])
            native_amp = amp_obs(value=1.+i)
            merged = alg.process_mixed_env_step(following, raw, done, timeout, native_amp)
            torch.testing.assert_close(merged[:4], raw[:4])
            assert not torch.allclose(merged[4:], raw[4:])
            assert alg.disc_obs_buffer.batch_size == 2
            torch.testing.assert_close(alg.disc_obs_buffer._buffer[i], native_amp['disc'])
            obs = following
    return obs


def test_native_amp_updates_only_amp_disc_buffer_and_all_task_actor():
    from rsl_rl.algorithms.ppo_amp import PPOAMP
    alg = learner()
    assert isinstance(alg, PPOAMP)
    assert not alg.optimizer.state and not alg.disc_optimizer.state
    final_obs = fill_rollout(alg)
    raw_actor = {k:v.clone() for k,v in alg.policy.actor.state_dict().items()}
    raw_disc = alg.amp_discriminator.disc_linear.weight.clone()
    raw_std=alg.policy.std.detach().clone()
    result = alg.compute_returns_and_update(final_obs)
    assert result['samples'] == 12
    assert alg.storage.step == 0
    assert 'amp/disc_loss' in result
    assert 'learning_rate' in result and result['learning_rate'] == alg.learning_rate
    assert set(result['tasks']) == {'stairs','backpack','run'}
    assert all('kl' in d and 'clip_fraction' in d and 'explained_variance' in d and 'mean_action_std' in d for d in result['tasks'].values())
    assert any(not torch.equal(v, raw_actor[k]) for k,v in alg.policy.actor.state_dict().items())
    assert not torch.equal(raw_disc,alg.amp_discriminator.disc_linear.weight)
    assert all(not torch.equal(a,b) for a,b in zip(raw_std,alg.policy.std,strict=True))
    for section in alg.storage.advantages.split((2,2,2),dim=1):
        assert abs(float(section.mean())) < 1e-5
        assert float(section.std()) == pytest.approx(1., abs=1e-5)


def test_critic_warmup_keeps_actor_std_discriminator_and_lr_unchanged():
    alg = learner()
    alg.configure_critic_only(True)
    actor = {k:v.clone() for k,v in alg.policy.actor.state_dict().items()}
    disc = {k:v.clone() for k,v in alg.amp_discriminator.state_dict().items()}
    critic = {k:v.clone() for k,v in alg.policy.critics.state_dict().items()}
    std = alg.policy.std.clone()
    lr = alg.learning_rate
    obs = fill_rollout(alg)
    result = alg.compute_returns_and_update(obs,critic_only=True)
    assert result['samples']==12
    for k,v in alg.policy.actor.state_dict().items():
        torch.testing.assert_close(v,actor[k],rtol=0,atol=0)
    for k,v in alg.amp_discriminator.state_dict().items():
        torch.testing.assert_close(v,disc[k],rtol=0,atol=0)
    assert all(any(not torch.equal(v,critic[k]) for k,v in alg.policy.critics.state_dict().items()
                   if k.startswith(f'{i}.')) for i in range(3))
    torch.testing.assert_close(alg.policy.std,std,rtol=0,atol=0)
    assert alg.learning_rate==lr
    alg.configure_critic_only(False)
    assert alg.policy.std.requires_grad


def test_native_entropy_coefficients_are_task_specific():
    from rsl_rl.algorithms.ppo_amp import PPOAMP
    alg = learner()
    obs = observation()
    entropy = torch.full((len(obs),), 2.)
    assert alg._entropy_regularization(obs,entropy).item() == pytest.approx((.005+.005+.01)*2/3)
    # The unmodified expert scalar entropy behavior remains exactly the same.
    assert PPOAMP._entropy_regularization(alg,obs,entropy).item() == pytest.approx(.005*2)


def test_amp_teacher_copy_and_native_timeout_bootstrapping():
    from native_ppo import build_native_learner
    cfg,state=amp_settings()
    teacher={k:v.clone() for k,v in state['amp_discriminator_state_dict'].items()}
    alg=build_native_learner(Actor(),(2,2,2),(4,6,3),cfg,state,None,'cpu',
                             teacher_priors=teacher_priors(),steps=2,num_epochs=1,num_minibatches=1)
    for key,value in alg.amp_discriminator.state_dict().items():
        torch.testing.assert_close(value,teacher[key],rtol=0,atol=0)
    obs=fill_rollout(alg)
    # Native inputs 1..4 from the two direct tasks; only timeout row 1
    # receives the gamma*value correction in native storage.
    expected=torch.arange(1.,5.).repeat(2,1)
    expected[:,1] += alg.gamma*alg.storage.values[:,1,0]
    torch.testing.assert_close(alg.storage.rewards[:,:4,0],expected)
    alg.compute_returns_and_update(obs)
    for key,value in state['amp_discriminator_state_dict'].items():
        torch.testing.assert_close(value,teacher[key],rtol=0,atol=0)


def test_native_symmetry_callback_and_amp_update_share_a_minibatch():
    # Exercise the actual PPOAMP data-augmentation and mirror-loss code path
    # separately from the physical mirror transform tested elsewhere.
    def mirror(obs=None, actions=None, env=None):
        mirrored_obs = torch.cat((obs, obs.clone()),dim=0) if obs is not None else None
        mirrored_actions = torch.cat((actions,-actions),dim=0) if actions is not None else None
        return mirrored_obs,mirrored_actions
    alg=learner(mirror)
    obs=fill_rollout(alg)
    metrics=alg.compute_returns_and_update(obs)
    assert metrics['symmetry'] > 0.
    assert metrics['amp/disc_loss'] > 0.


def test_disordered_task_rows_cannot_send_non_amp_rewards_into_amp():
    alg = learner()
    obs = observation()
    alg.act(obs)
    bad = observation()
    bad['task_id'][0],bad['task_id'][-1] = bad['task_id'][-1].clone(),bad['task_id'][0].clone()
    with pytest.raises(ValueError,match='task order'):
        alg.process_mixed_env_step(bad,torch.ones(6),torch.zeros(6,dtype=torch.bool),
                                   torch.zeros(6,dtype=torch.bool),amp_obs())


def test_native_ppo_with_exact_observation_widths_and_real_symmetry_adapter():
    from native_observations import NativeObservationAdapter
    from native_ppo import build_native_learner, TASKS
    class WideActor(Actor):
        width=1161
    widths=(3288,1418,271)
    adapter=NativeObservationAdapter(widths)
    def current_obs():
        return adapter.concat([adapter.pack(task,torch.randn(2,1161),torch.randn(2,width))
                               for task,width in zip(TASKS,widths,strict=True)])
    cfg, state=amp_settings()
    alg=build_native_learner(WideActor(),(2,2,2),widths,cfg,state,adapter.augment,'cpu',
                             teacher_priors=teacher_priors(widths),steps=2,num_epochs=1,num_minibatches=1)
    obs=current_obs()
    for _ in range(2):
        with torch.inference_mode():
            alg.act(obs)
            obs=current_obs()
            alg.process_mixed_env_step(obs,torch.arange(1.,7.),torch.zeros(6,dtype=torch.bool),
                                       torch.zeros(6,dtype=torch.bool),amp_obs())
    result=alg.compute_returns_and_update(obs)
    assert result['samples']==12
    assert result['symmetry']>0.
    assert result['amp/disc_loss']>0.


def test_corrupt_amp_demonstration_is_rejected_before_storage():
    alg=learner()
    obs=observation()
    alg.act(obs)
    native=amp_obs()
    native['disc_demo'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite native AMP'):
        alg.process_mixed_env_step(observation(),torch.ones(6),torch.zeros(6,dtype=torch.bool),
                                   torch.zeros(6,dtype=torch.bool),native)
    assert alg.storage.step==0 and alg.disc_obs_buffer._buffer is None


def test_exact_teacher_critics_and_noise_on_shuffled_and_mirrored_rows():
    from amp_policy import PrefixNormalization
    from rsl_rl.networks import MLP
    from native_ppo import NativeSharedPolicy,TASKS
    widths=(4,6,3)
    priors=teacher_priors(widths)
    actor=Actor()
    policy=NativeSharedPolicy(actor,widths,'cpu',teacher_priors=priors)
    obs=observation()[torch.tensor([5,0,3,1,4,2])]
    doubled=torch.cat((obs,obs.clone()))
    expected=torch.empty(len(doubled),1)
    expected_noise=torch.empty(len(doubled),23)
    for i,(task,width) in enumerate(zip(TASKS,widths,strict=True)):
        prior=priors[task]
        critic=MLP(width,1,[512,256,128],'elu')
        critic.load_state_dict(prior['critic_state_dict'],strict=True)
        normalizer=PrefixNormalization(prior['critic_normalizer_base_width']) if task=='run' else nn.Identity()
        normalizer.load_state_dict(prior['critic_normalizer_state_dict'],strict=True)
        rows=doubled['task_id'].squeeze(-1)==i
        x=doubled['critic'][rows,:width]
        expected[rows]=critic(normalizer(x))
        expected_noise[rows]=prior['action_std']
        for key,value in policy.critics[i].state_dict().items():
            torch.testing.assert_close(value,prior['critic_state_dict'][key],rtol=0,atol=0)
        for key,value in policy.critic_normalizers[i].state_dict().items():
            torch.testing.assert_close(value,prior['critic_normalizer_state_dict'][key],rtol=0,atol=0)
    torch.testing.assert_close(policy.evaluate(doubled),expected,rtol=0,atol=0)
    policy.act(doubled)
    torch.testing.assert_close(policy.action_std,expected_noise,rtol=0,atol=0)
    torch.testing.assert_close(policy.act_inference(doubled),actor(doubled['policy']),rtol=0,atol=0)
    assert policy.teacher_checkpoint_sha256=={t:priors[t]['checkpoint_sha256'] for t in TASKS}


@pytest.mark.parametrize('corruption',('missing_task','critic_shape','std_shape','std_negative','std_nan',
                                       'norm_shape','wrong_prefix','identity_state','missing_hash'))
def test_bad_or_missing_teacher_prior_is_rejected(corruption):
    from native_ppo import NativeSharedPolicy
    priors=teacher_priors()
    if corruption=='missing_task': del priors['stairs']
    if corruption=='critic_shape': priors['stairs']['critic_state_dict']['0.weight']=torch.ones(512,3)
    if corruption=='std_shape': priors['run']['action_std']=torch.ones(22)
    if corruption=='std_negative': priors['stairs']['action_std'][1]=-1.
    if corruption=='std_nan': priors['run']['action_std'][0]=float('nan')
    if corruption=='norm_shape': priors['run']['critic_normalizer_state_dict']['base._mean']=torch.ones(1,1)
    if corruption=='wrong_prefix': priors['run']['critic_normalizer_base_width']=None
    if corruption=='identity_state': priors['backpack']['critic_normalizer_state_dict']={'base.count':torch.zeros(())}
    if corruption=='missing_hash': del priors['run']['checkpoint_sha256']
    with pytest.raises(ValueError):
        NativeSharedPolicy(Actor(),(4,6,3),'cpu',teacher_priors=priors)


def test_teacher_prior_is_required_and_bad_updated_std_is_not_silently_clamped():
    from native_ppo import NativeSharedPolicy
    with pytest.raises(TypeError): NativeSharedPolicy(Actor(),(4,6,3),'cpu')
    policy=NativeSharedPolicy(Actor(),(4,6,3),'cpu',teacher_priors=teacher_priors())
    with torch.no_grad(): policy.std[1,0]=-1.
    with pytest.raises(ValueError,match='positive'):
        policy.act(observation())


def test_amp_prefix_uses_exact_native_statistics_update_and_preserves_context():
    from native_ppo import NativeSharedPolicy
    from amp_policy import PrefixNormalization
    widths=(4,6,3)
    priors=teacher_priors(widths)
    original=PrefixNormalization(2)
    original.load_state_dict(priors['run']['critic_normalizer_state_dict'],strict=True)
    policy=NativeSharedPolicy(Actor(),widths,'cpu',teacher_priors=priors)
    obs=observation()[torch.tensor([4,1,2,5,3,0])]
    run_rows=obs['task_id'].squeeze(-1)==2
    native=obs['critic'][run_rows,:3]
    for _ in range(3):
        original.update(native)
        policy.update_normalization(obs)
    for key,value in original.state_dict().items():
        torch.testing.assert_close(value,policy.critic_normalizers[2].state_dict()[key],rtol=0,atol=0)
    normalized=policy.critic_normalizers[2](native)
    torch.testing.assert_close(normalized,original(native),rtol=0,atol=0)
    torch.testing.assert_close(normalized[:,2:],native[:,2:],rtol=0,atol=0)
    assert int(priors['run']['critic_normalizer_state_dict']['base.count'])==5
