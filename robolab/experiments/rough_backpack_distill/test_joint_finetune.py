import torch
from joint_finetune import returns

def test_gae_does_not_cross_episode_boundary():
    reward=torch.tensor([[1.],[2.],[100.]])
    done=torch.tensor([[False],[True],[False]])
    actual=returns(reward,done,torch.zeros_like(done),torch.zeros_like(reward),torch.tensor([0.]),gamma=1.,lam=1.)
    torch.testing.assert_close(actual,torch.tensor([[3.],[2.],[100.]]))

def test_truncation_bootstraps_without_bootstrapping_fall():
    reward=torch.tensor([[1.,1.]])
    value=torch.tensor([[2.,2.]])
    actual=returns(reward,torch.tensor([[True,True]]),torch.tensor([[True,False]]),value,torch.tensor([50.,50.]),gamma=.5,lam=1.)
    torch.testing.assert_close(actual,torch.tensor([[2.,1.]]))
