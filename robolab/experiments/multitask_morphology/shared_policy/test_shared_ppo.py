import os
import pytest
import torch
from tensordict import TensorDict
from contract import sha
from student import load_student
from shared_ppo import SharedPPOActorCritic


def test_actor_parity_and_trainable_value_function():
    path=os.environ.get('SHARED_PPO_TEST_CHECKPOINT')
    if not path:pytest.skip('Requires explicitly selected real shared checkpoint')
    original,_=load_student(path)
    torch.manual_seed(881)
    obs=TensorDict(dict(policy=torch.randn(32,original.width),critic=torch.randn(32,96)),batch_size=[32])
    policy=SharedPPOActorCritic(obs,dict(policy=['policy'],critic=['critic']),23,
        student_checkpoint=path,student_sha256=sha(path),
        init_noise_std=.1,actor_obs_normalization=False,critic_obs_normalization=False)
    with torch.no_grad():
        torch.testing.assert_close(policy.act_inference(obs),original(obs['policy']),atol=1e-6,rtol=1e-6)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-5)
    actor_before=next(policy.actor.parameters()).detach().clone()
    critic_before=next(policy.critic.parameters()).detach().clone()
    for _ in range(2):
        actions=policy.act(obs).detach()
        loss=-policy.get_actions_log_prob(actions).mean()+policy.evaluate(obs).square().mean()
        optimizer.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(actor_before,next(policy.actor.parameters()))
    assert not torch.equal(critic_before,next(policy.critic.parameters()))
    assert len(optimizer.state)>0
