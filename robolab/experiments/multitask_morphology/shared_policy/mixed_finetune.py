"""Single-structure pilot with one Isaac Lab app and one shared GPU PPO learner."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

from contract import load_plan, sha, write_new
from finetune_settings import PPO_SCHEDULE
from finetune_state import select_candidate, save_checkpoint
from joint_finetune import ACTION_STD, STEPS_PER_TASK, check_pause, ppo_update, publish_training_completion
from mixed_batch import advance, task_counts
from randomization import protocol_metadata, assert_protocol_unchanged
from student import load_student
from training_reward import RolloutRewards, mean_reward_rows, reward_history_rows, write_mean_reward_csv, write_training_reward_csv


HERE = Path(__file__).resolve().parent
TASKS = ('stairs', 'backpack', 'run')


class NativeTask:
    @torch.inference_mode()
    def __init__(self, name, backend, protocol):
        self.name = name
        self.wrapper, self.raw, obs, self.canonical, self.clip, self.reward_adapter, policy = backend
        if policy != protocol:
            raise ValueError(f'{name} rollout protocol differs')
        self.base = self.canonical(obs, None)
        self.records = []

    def begin(self, action):
        action = action.clamp(-self.clip, self.clip)
        if self.wrapper.clip_actions is not None:
            action = action.clamp(-self.wrapper.clip_actions, self.wrapper.clip_actions)
        if self.name == 'run':
            self.raw.action_manager.process_action(action)
            self.raw.recorder_manager.record_pre_step()
        else:
            self.raw._pre_physics_step(action)

    def apply(self):
        raw = self.raw
        raw._sim_step_counter += 1
        if self.name == 'run':
            raw.action_manager.apply_action()
        else:
            raw._apply_action()
        raw.scene.write_data_to_sim()

    def sync(self):
        raw = self.raw
        if self.name == 'run':
            raw.recorder_manager.record_post_physics_decimation_step()
        raw.scene.update(dt=raw.physics_dt)

    def finish(self):
        obs, reward, terminated, truncated, _ = self.raw._finish_control_step()
        done = terminated | truncated
        timeout = truncated if not self.raw.cfg.is_finite_horizon else torch.zeros_like(truncated)
        obs = self.wrapper._extend(obs)
        step_reward = self.reward_adapter(obs, reward.clone()) if self.reward_adapter else reward
        self.base = self.canonical(obs, done)
        return dict(reward=step_reward, environment_reward=reward, done=done, timeout=timeout)


def prepare_tasks(a, plan, root, protocol, counts):
    # Load the pilot's native step hooks before the frozen teacher runtime paths
    # are placed on sys.path by the per-task rollout builders.
    import robolab.tasks
    from mixed_scene import create_subscenes
    from rollout import prepare_rollout
    from rollout_amp import prepare_rollout as prepare_amp_rollout

    if Path(robolab.__file__).resolve().parent != HERE.parents[2]/'robolab':
        raise RuntimeError('The active RoboLab package is outside the mixed pilot worktree')
    with ExitStack() as stack:
        preparers, recipes = [], []
        for task, count in zip(TASKS, counts, strict=True):
            definition = plan['teachers'][task]
            args = ['--plan', str(root/'plan.json'), '--design-ids', definition['design_ids'][0],
                    '--output', str(root/task), '--mode', 'evaluate', '--ppo-channel', str(root),
                    '--profile', a.profile, '--randomization-strength', str(a.randomization_strength),
                    '--seed', str(a.seed), '--headless', '--device', a.device]
            entry = prepare_amp_rollout
            if task != 'run':
                args.extend(('--task', task))
                entry = prepare_rollout
            generator = entry(args, mixed_count=count)
            preparers.append(generator)
            stack.callback(generator.close)
            recipes.append(next(generator))
        sim, subscenes = create_subscenes(recipes, TASKS)
        backends = []
        for task, preparer, raw in zip(TASKS, preparers, subscenes, strict=True):
            try:
                preparer.send(raw)
            except StopIteration as result:
                backends.append(NativeTask(task, result.value, protocol))
            else:
                raise RuntimeError(f'{task} did not finish preparing the PPO rollout')
    return sim, backends


@torch.inference_mode()
def collect(a, sim, tasks, counts, model, root):
    for task in tasks:
        task.records.clear()
    for _ in range(STEPS_PER_TASK):
        check_pause(root)
        obs = [task.base for task in tasks]
        mean = model(torch.cat(obs))
        distribution = torch.distributions.Normal(mean, ACTION_STD)
        actions = distribution.sample()
        logp = distribution.log_prob(actions).sum(-1)
        outcomes = advance(tasks, counts, actions, tasks[0].raw.cfg.decimation,
                           lambda: sim.step(render=False))
        for task, base, action, action_logp, outcome in zip(
                tasks, obs, actions.split(counts), logp.split(counts), outcomes, strict=True):
            sample = dict(obs=base, actions=action, logp=action_logp, **outcome)
            task.records.append({key: value.clone() for key, value in sample.items()})
    batches = []
    for task in tasks:
        batch = {key: torch.stack([row[key] for row in task.records]) for key in task.records[0]}
        task.records.clear()
        for key in ('obs', 'actions', 'logp', 'reward', 'environment_reward'):
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f'Nonfinite {task.name} {key}')
        batches.append(dict(task=task.name, next_obs=task.base, **batch))
    return batches


def train(a, root, plan, candidate, protocol, counts):
    sim, tasks = prepare_tasks(a, plan, root, protocol, counts)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model, _ = load_student(a.student, a.device)
    if model.schema != plan['schema']:
        raise ValueError('Candidate actor schema differs')
    critic = torch.nn.Sequential(torch.nn.Linear(model.width, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 1)).to(a.device)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 3e-6},
                                  {'params': critic.parameters(), 'lr': 1e-3}])
    digest = sha(root/'manifest.json')
    write_new(root/'mixed_scene.json', dict(physics_scenes=1, pid=os.getpid(),
              tasks={task.name: dict(envs=task.raw.num_envs, scope=task.raw.scene.scope,
                                    joints=task.wrapper.num_actions) for task in tasks},
              physics_dt=sim.get_physics_dt(), control_dt=tasks[0].raw.step_dt))
    print('MIXED_SCENE_READY', counts, flush=True)
    save_checkpoint(root, model, critic, optimizer, 0, digest)
    rewards = RolloutRewards()
    step_history, mean_history = [], []
    for update in range(1, a.updates+1):
        assert_protocol_unchanged(protocol)
        torch.cuda.reset_peak_memory_stats(a.device)
        begin = time.monotonic()
        batches = collect(a, sim, tasks, counts, model, root)
        torch.cuda.synchronize(a.device)
        collected = time.monotonic()
        reward_summary = rewards.summarize(batches, 'mixed')
        metrics = ppo_update(model, critic, optimizer, batches)
        torch.cuda.synchronize(a.device)
        metrics.update(collection_seconds=collected-begin, update_seconds=time.monotonic()-collected,
                       torch_peak_mib=torch.cuda.max_memory_allocated(a.device)/2**20)
        del batches
        check_pause(root)
        receipt = save_checkpoint(root, model, critic, optimizer, update, digest, metrics,
                                  training_reward=reward_summary)
        step_history.extend(reward_history_rows([receipt], TASKS))
        mean_history.extend(mean_reward_rows([receipt], TASKS))
        write_training_reward_csv(root, step_history)
        write_mean_reward_csv(root, TASKS, mean_history)
        print('MIXED_PPO_UPDATE', update, '/', a.updates, metrics, flush=True)
    publish_training_completion(a, root, model, candidate, digest, protocol)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design-id', required=True)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--profile', choices=('nominal', 'robust'), default='robust')
    parser.add_argument('--randomization-strength', type=float, default=1.)
    AppLauncher.add_app_launcher_args(parser)
    a = parser.parse_args()
    if not a.headless or not a.device.startswith('cuda') or a.updates < 1:
        parser.error('Mixed training requires a CUDA device, headless mode and a positive update budget')
    counts = task_counts(a.num_envs)
    plan, candidate = select_candidate(load_plan(a.plan), design_id=a.design_id)
    if tuple(plan['schema']['tasks']) != TASKS:
        raise ValueError('Mixed pilot requires the three-task student')
    protocol = protocol_metadata(a.profile, a.randomization_strength)
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    files = ['mixed_finetune.py', 'mixed_scene.py', 'mixed_batch.py', 'joint_finetune.py',
             'rollout.py', 'rollout_amp.py', 'student.py', 'training_reward.py']
    write_new(root/'manifest.json', dict(kind='shared_candidate_finetuning_mixed_pilot',
              plan_sha256=sha(a.plan), initial_actor_sha256=sha(a.student), candidate=candidate,
              updates=a.updates, seed=a.seed, collection_protocol=protocol, ppo_schedule=PPO_SCHEDULE,
              envs_per_task=dict(zip(TASKS, counts, strict=True)), steps_per_task=STEPS_PER_TASK,
              sources={name: sha(HERE/name) for name in files}))
    write_new(root/'plan.json', plan)
    a.kit_args = (a.kit_args or '') + f' --/renderer/multiGpu/enabled=false --portable-root {root/"portable"} --portable'
    app = AppLauncher(a).app
    try:
        train(a, root, plan, candidate, protocol, counts)
    except BaseException:
        write_new(root/'failure.json', dict(traceback=traceback.format_exc()))
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    app.close()


if __name__ == '__main__':
    main()
