import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from rsl_rl.utils.amp_logger import LoggerAMP
from rsl_rl.utils.logger import Logger

from episode_phases import attach_phase_warmup, warm_up_episode_phases


class FixedHorizonEnv:
    device = "cpu"
    max_episode_length = 6

    def __init__(self):
        self.episode_length_buf = torch.zeros(12, dtype=torch.long)
        self.resets = torch.zeros_like(self.episode_length_buf)
        self.initial_ages = None

    def get_observations(self):
        return self.episode_length_buf.unsqueeze(1)

    def step(self, actions):
        if self.initial_ages is None:
            self.initial_ages = self.episode_length_buf.clone()
        rewards = torch.where(self.resets == 0, 100.0, 2.0)
        self.episode_length_buf.add_(1)
        done = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[done] = 0
        self.resets.add_(done)
        return self.get_observations(), rewards, done, {}


class PhaseTests(unittest.TestCase):
    def make_runner(self, amp=False):
        env = FixedHorizonEnv()
        logger = (LoggerAMP if amp else Logger).__new__(LoggerAMP if amp else Logger)
        logger.log_dir = "test"
        logger.ep_extras = []
        logger.rewbuffer = deque(maxlen=100)
        logger.lenbuffer = deque(maxlen=100)
        logger.cur_reward_sum = torch.zeros(12)
        logger.cur_episode_length = torch.zeros(12)
        storage = SimpleNamespace(spatial_groups=torch.arange(12) % 2)
        alg = SimpleNamespace(
            policy=SimpleNamespace(is_recurrent=False, act=lambda obs: obs),
            storage=storage,
        )
        if amp:
            logger.max_episode_length_s = 0.12
            logger.style_rewbuffer = deque(maxlen=100)
            logger.total_rewbuffer = deque(maxlen=100)
            logger.cur_style_reward_sum = torch.zeros(12)
            logger.cur_total_reward_sum = torch.zeros(12)

            class Disc(torch.nn.Module):
                def get_disc_obs(self, obs, flatten_history_dim):
                    return obs

                def predict_style_reward(self, obs, dt):
                    if self.training:
                        raise AssertionError("Warmup discriminator must be in evaluation mode")
                    return torch.ones(len(obs)), None

                def lerp_reward(self, task_reward, style_reward):
                    return task_reward + style_reward

            alg.amp_discriminator = Disc()
            alg.amp_cfg = {"step_dt": 0.02}
        return SimpleNamespace(
            env=env, logger=logger, alg=alg, alg_cfg={"rnd_cfg": None},
            device="cpu", cfg={"num_steps_per_env": 2}, hybrid_stage={"index": 0},
            current_learning_iteration=0, train_mode=lambda: None,
        )

    def test_counts_only_complete_ordinary_episodes(self):
        runner = self.make_runner()
        audit = warm_up_episode_phases(runner)
        self.assertEqual(audit["warmup_control_steps"], 14)
        self.assertEqual(audit["provisional_episodes_excluded"], 12)
        self.assertEqual(set(runner.logger.rewbuffer), {12.0})
        self.assertEqual(set(runner.logger.lenbuffer), {6.0})
        self.assertEqual(runner.logger.ep_extras, [])
        for group in range(2):
            self.assertEqual(set(runner.env.initial_ages[group::2].tolist()), set(range(6)))

    def test_independent_timer_mixes_all_terrain_types(self):
        from episode_phases import assign_initial_episode_ages
        torch.manual_seed(42)
        age = torch.zeros(4096, dtype=torch.long)
        groups = torch.zeros_like(age)
        assign_initial_episode_ages(age, groups, 1000, independent=True)
        terrain = torch.arange(4096) * 20 // 4096
        table = torch.bincount(terrain*20 + age*20//1000, minlength=400).reshape(20,20)
        self.assertTrue(torch.all((table>0).sum(1)>=18))
        self.assertTrue(torch.all((age>=0)&(age<1000)))

    def test_independent_warmup_excludes_partial_episodes(self):
        runner=self.make_runner()
        audit=warm_up_episode_phases(runner, independent=True)
        self.assertEqual(audit['phase_randomization'],'per_environment_iid_uniform')
        self.assertEqual(audit['provisional_episodes_excluded'],12)
        self.assertEqual(set(runner.logger.rewbuffer),{12.0})
        self.assertEqual(set(runner.logger.lenbuffer),{6.0})

    def test_counts_amp_task_style_and_total_separately(self):
        runner = self.make_runner(amp=True)
        warm_up_episode_phases(runner)
        self.assertEqual(set(runner.logger.rewbuffer), {12.0})
        self.assertEqual(set(runner.logger.style_rewbuffer), {6.0})
        self.assertEqual(set(runner.logger.total_rewbuffer), {18.0})
        self.assertEqual(runner.logger.ep_extras, [])
        self.assertTrue(runner.alg.amp_discriminator.training)

    def test_reuses_native_learn_after_each_physical_reset(self):
        runner = self.make_runner()
        learn_calls = []

        def native_learn(count):
            self.assertEqual(set(runner.logger.rewbuffer), {12.0})
            learn_calls.append(count)

        runner.learn = native_learn
        with TemporaryDirectory() as folder:
            marker = Path(folder) / "USER_PAUSED"
            attach_phase_warmup(runner, marker)
            runner.learn(50)
            runner.current_learning_iteration = 50
            runner.hybrid_stage = {"index": 1}
            runner.env = FixedHorizonEnv()
            runner.logger.rewbuffer.append(999.0)
            runner.learn(40)
            marker.touch()
            with self.assertRaisesRegex(RuntimeError, "USER_PAUSED"):
                runner.learn(10)
        self.assertEqual(learn_calls, [50, 40])


if __name__ == "__main__":
    unittest.main()
