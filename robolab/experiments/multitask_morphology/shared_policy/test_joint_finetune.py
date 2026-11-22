import torch
from joint_finetune import ppo_update, returns, worker_command
from training_reward import RolloutRewards


def test_ppo_reuses_each_rollout_for_five_epochs_of_four_batches():
    model = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam([*model.parameters(), *critic.parameters()])
    shape = (24, 8)
    batches = [dict(obs=torch.zeros(*shape, 2), next_obs=torch.zeros(8, 2),
                    actions=torch.zeros(*shape, 1), logp=torch.zeros(shape),
                    reward=torch.ones(shape), done=torch.zeros(shape, dtype=torch.bool),
                    timeout=torch.zeros(shape, dtype=torch.bool)) for _ in range(3)]
    assert ppo_update(model, critic, optimizer, batches)['samples'] == 576
    assert int(optimizer.state[model.weight]['step']) == 20


def test_mean_reward_tracks_global_and_task_episode_returns():
    logger = RolloutRewards()

    def rollout(task, step_reward, done):
        reward = torch.tensor([step_reward])
        return dict(task=task, environment_reward=reward, reward=reward * 2,
                    done=torch.tensor([done]), timeout=torch.zeros(1, len(done), dtype=torch.bool))

    first = logger.summarize([
        rollout('stairs', [2., 4.], [True, False]),
        rollout('run', [20.], [False]),
    ], 'session')
    assert first['mean_reward']['all']['mean_reward'] == 2.
    assert first['mean_reward']['run']['mean_reward'] is None
    second = logger.summarize([
        rollout('stairs', [4., 4.], [True, True]),
        rollout('run', [22.], [True]),
    ], 'session')
    assert second['mean_reward']['all'] == dict(mean_reward=14., window_episodes=4)
    assert second['mean_reward']['stairs'] == dict(mean_reward=14. / 3, window_episodes=3)
    assert second['mean_reward']['run'] == dict(mean_reward=42., window_episodes=1)

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
