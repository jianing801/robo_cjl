"""Isolated fine-tuning ablations; never alter the formal V4 configuration."""
import torch
from test_native_ppo import Actor, teacher_priors, amp_settings, learner, fill_rollout


def test_fixed_amp_std_preserves_exact_inherited_vector():
    from native_ppo import build_native_learner
    cfg,state=amp_settings()
    learner=build_native_learner(Actor(),(2,2,2),(4,6,3),cfg,state,None,'cpu',
                                 teacher_priors=teacher_priors(),steps=2,num_epochs=1,num_minibatches=1)
    before=learner.policy.std.detach().clone()
    learner.fix_expert_std(('run',))
    for _ in range(3):
        learner.optimizer.zero_grad(); learner.policy.std.sum().backward(); learner.optimizer.step()
        learner.verify_fixed_std()
    assert torch.equal(learner.policy.std[2],before[2])
    assert not torch.equal(learner.policy.std[:2],before[:2])


def test_legacy_ablation_identity_and_formal_scheme_b():
    from finetune_settings import NATIVE_PPO_SETTINGS, correction_settings
    assert correction_settings('native')==NATIVE_PPO_SETTINGS
    fixed=correction_settings('fixed_amp_std')
    assert fixed!=NATIVE_PPO_SETTINGS and fixed['fixed_expert_std_tasks']==['run']
    assert 'fixed_expert_std_tasks' not in NATIVE_PPO_SETTINGS
    from finetune_settings import (CURRENT_PPO_ALGORITHM,CURRENT_PPO_SETTINGS,NATIVE_PPO_ALGORITHM,
                                   NATIVE_PPO_PROFILES,resolve_training_profile)
    algorithm,profile,settings=resolve_training_profile()
    assert algorithm==CURRENT_PPO_ALGORITHM!=NATIVE_PPO_ALGORITHM
    assert profile=='fixed_all_std_pcgrad' and settings==correction_settings(profile)
    assert settings==CURRENT_PPO_SETTINGS==NATIVE_PPO_PROFILES[algorithm]
    assert resolve_training_profile(profile)==(CURRENT_PPO_ALGORITHM,profile,settings)


def test_pcgrad_removes_pairwise_conflicting_component_without_changing_aligned_sum():
    from native_ppo import project_actor_gradients
    aligned=torch.tensor([[1.,0.],[2.,0.],[3.,0.]])
    assert torch.equal(project_actor_gradients(aligned),aligned.sum(0))
    conflict=torch.tensor([[1.,0.],[-1.,1.],[0.,0.]])
    actual=project_actor_gradients(conflict)
    assert torch.allclose(actual,torch.tensor([.5,1.5]))
    assert torch.isfinite(actual).all()


def test_weighted_stairs_profile_preserves_formal_b_and_scales_contributions():
    from finetune_settings import STAIRS_WEIGHT_PROFILE, CURRENT_PPO_SETTINGS, correction_settings, resolve_training_profile
    from native_ppo import project_actor_gradients
    settings=correction_settings(STAIRS_WEIGHT_PROFILE)
    assert settings['actor_gradient_weights']==dict(stairs=2.,backpack=1.,run=1.)
    assert settings['actor_gradient_weight_normalization']=='mean_one'
    assert {k:v for k,v in settings.items() if not k.startswith('actor_gradient_weight')}==correction_settings('fixed_amp_std_pcgrad')
    assert resolve_training_profile()[2]==CURRENT_PPO_SETTINGS
    aligned=torch.eye(3)
    weights=(2.,1.,1.)
    assert torch.allclose(project_actor_gradients(aligned,weights), torch.tensor([1.5,.75,.75]))
    assert torch.equal(project_actor_gradients(aligned),aligned.sum(0))


def test_pcgrad_real_ppo_update_keeps_frozen_amp_and_native_discriminator():
    def mirror(obs=None,actions=None,env=None):
        return (torch.cat((obs,obs.clone()),dim=0) if obs is not None else None,
                torch.cat((actions,-actions),dim=0) if actions is not None else None)
    alg=learner(mirror)
    alg.fix_expert_std(('run',));alg.actor_projection=True;alg.inspect_actor_gradients=True
    initial=alg.policy.std.detach().clone();disc=alg.amp_discriminator.disc_linear.weight.clone()
    actor={k:v.clone() for k,v in alg.policy.actor.state_dict().items()}
    metrics=alg.compute_returns_and_update(fill_rollout(alg))
    assert len(metrics['actor_gradient_sample']['cosine'])==3
    assert torch.equal(alg.policy.std[2],initial[2])
    assert not torch.equal(disc,alg.amp_discriminator.disc_linear.weight)
    assert any(not torch.equal(v,actor[k]) for k,v in alg.policy.actor.state_dict().items())


def test_weighted_stairs_real_ppo_update_keeps_frozen_amp_and_discriminator():
    alg=learner(None)
    alg.fix_expert_std(('run',));alg.actor_projection=True;alg.actor_gradient_weights=(2.,1.,1.)
    initial=alg.policy.std.detach().clone();disc=alg.amp_discriminator.disc_linear.weight.clone()
    metrics=alg.compute_returns_and_update(fill_rollout(alg))
    assert metrics['actor_gradient_weights']==dict(stairs=2.,backpack=1.,run=1.)
    assert torch.equal(alg.policy.std[2],initial[2])
    assert not torch.equal(disc,alg.amp_discriminator.disc_linear.weight)


def test_kl_pilot_changes_only_desired_kl_and_uses_native_value():
    from finetune_settings import KL_PILOT_PROFILE, CURRENT_PPO_SETTINGS, correction_settings
    from native_ppo import build_native_learner
    settings=correction_settings(KL_PILOT_PROFILE)
    assert settings==dict(correction_settings('fixed_amp_std_pcgrad'),desired_kl=.02)
    cfg,state=amp_settings()
    alg=build_native_learner(Actor(),(2,2,2),(4,6,3),cfg,state,None,'cpu',teacher_priors=teacher_priors(),
                            steps=2,num_epochs=1,num_minibatches=1,desired_kl=settings['desired_kl'])
    assert alg.desired_kl==.02
    alg.fix_expert_std(('run',));alg.actor_projection=True
    metrics=alg.compute_returns_and_update(fill_rollout(alg))
    assert metrics['desired_kl']==.02 and 'actor_gradient_weights' not in metrics


def test_stairs_larger_sample_profile_uses_unweighted_unequal_native_batches():
    from finetune_settings import STAIRS_SAMPLE_PROFILE,CURRENT_PPO_SETTINGS,correction_settings
    from native_ppo import build_native_learner
    from test_native_ppo import observation,amp_obs
    settings=correction_settings(STAIRS_SAMPLE_PROFILE)
    assert settings==dict(correction_settings('fixed_amp_std_pcgrad'),envs_per_task=dict(stairs=2048,backpack=1024,run=1024))
    counts=(4,2,2);cfg,state=amp_settings()
    alg=build_native_learner(Actor(),counts,(4,6,3),cfg,state,None,'cpu',teacher_priors=teacher_priors(),
                            steps=2,num_epochs=1,num_minibatches=1)
    alg.fix_expert_std(('run',));alg.actor_projection=True
    std=alg.policy.std[2].detach().clone();obs=observation(counts)
    for i in range(2):
        with torch.inference_mode():
            alg.act(obs);following=observation(counts);raw=torch.arange(1.,9.);dones=torch.zeros(8,dtype=torch.bool)
            merged=alg.process_mixed_env_step(following,raw,dones,dones,amp_obs(value=1.+i))
            torch.testing.assert_close(merged[:6],raw[:6])
            assert alg.disc_obs_buffer.batch_size==2
            obs=following
    metrics=alg.compute_returns_and_update(obs)
    assert metrics['samples']==16 and 'actor_gradient_weights' not in metrics
    assert torch.equal(alg.policy.std[2],std)


def test_all_teacher_exploration_profile_changes_only_frozen_rows():
    from finetune_settings import (ALL_FIXED_STD_PROFILE, ALL_FIXED_STD_ALGORITHM,
        CURRENT_PPO_ALGORITHM, CURRENT_PPO_SETTINGS, EXPERT_PRIOR_ALGORITHMS, resolve_training_profile)
    algorithm,profile,settings=resolve_training_profile(ALL_FIXED_STD_PROFILE)
    assert profile==ALL_FIXED_STD_PROFILE and algorithm==ALL_FIXED_STD_ALGORITHM==CURRENT_PPO_ALGORITHM
    assert algorithm in EXPERT_PRIOR_ALGORITHMS
    assert settings==dict(CURRENT_PPO_SETTINGS,fixed_expert_std_tasks=['stairs','backpack','run'])
    assert CURRENT_PPO_SETTINGS['fixed_expert_std_tasks']==['stairs','backpack','run']
    alg=learner(None)
    alg.fix_expert_std(tuple(settings['fixed_expert_std_tasks']));alg.actor_projection=True
    initial=alg.policy.std.detach().clone();disc=alg.amp_discriminator.disc_linear.weight.detach().clone()
    actor={k:v.detach().clone() for k,v in alg.policy.actor.state_dict().items()}
    for _ in range(2):
        alg.compute_returns_and_update(fill_rollout(alg));alg.verify_fixed_std()
        assert torch.equal(alg.policy.std,initial)
    assert not torch.equal(disc,alg.amp_discriminator.disc_linear.weight)
    assert any(not torch.equal(v,actor[k]) for k,v in alg.policy.actor.state_dict().items())
