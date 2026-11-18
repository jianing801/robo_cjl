import torch
from joint_finetune import returns, worker_command

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


def test_each_ppo_task_explicitly_receives_training_randomization(tmp_path):
    for task in ('stairs','backpack','run'):
        command=worker_command('plan.json','student.pt',tmp_path,task,'design','robust',.75,42)
        assert command[command.index('--mode')+1]=='evaluate'
        assert command[command.index('--profile')+1]=='robust'
        assert command[command.index('--randomization-strength')+1]=='0.75'
        assert command[command.index('--seed')+1]=='42'
        assert ('--task' in command)==(task!='run')
        assert '--ppo-channel' in command
