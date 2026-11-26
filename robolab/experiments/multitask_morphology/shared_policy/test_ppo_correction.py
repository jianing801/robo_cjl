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
    assert profile=='fixed_amp_std_pcgrad' and settings==correction_settings(profile)
    assert settings==CURRENT_PPO_SETTINGS==NATIVE_PPO_PROFILES[algorithm]
    assert resolve_training_profile(profile)==(NATIVE_PPO_ALGORITHM,profile,settings)


def test_pcgrad_removes_pairwise_conflicting_component_without_changing_aligned_sum():
    from native_ppo import project_actor_gradients
    aligned=torch.tensor([[1.,0.],[2.,0.],[3.,0.]])
    assert torch.equal(project_actor_gradients(aligned),aligned.sum(0))
    conflict=torch.tensor([[1.,0.],[-1.,1.],[0.,0.]])
    actual=project_actor_gradients(conflict)
    assert torch.allclose(actual,torch.tensor([.5,1.5]))
    assert torch.isfinite(actual).all()


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
