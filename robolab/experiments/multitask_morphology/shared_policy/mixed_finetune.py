"""Single-structure pilot with one Isaac Lab app and one shared GPU PPO learner."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import os
import random
import sys
import time
import traceback
from uuid import uuid4

import numpy as np
import torch

from contract import load_plan, sha, write_new
from finetune_settings import MIXED_BO_ENVS, PPO_SCHEDULE
from finetune_state import checkpoint_history, select_candidate
from finetune_common import STEPS_PER_TASK, check_pause, publish_training_completion
from mixed_batch import advance, initialize_episode_length, task_counts
from persistence import resume_pause, run_lock
from randomization import protocol_metadata, assert_protocol_unchanged
from student import load_student
from training_reward import RolloutRewards, mean_reward_rows, reward_history_rows, write_mean_reward_csv, write_training_reward_csv


HERE = Path(__file__).resolve().parent
TASKS = ('stairs', 'backpack', 'run')


class NativeTask:
    @torch.inference_mode()
    def __init__(self, name, backend, protocol):
        self.name = name
        self.wrapper, self.raw, obs, self.canonical, self.clip, self.reward_adapter, policy = backend[:7]
        if len(backend) != 8 or 'expert_prior' not in backend[7] or 'teacher_value' not in backend[7]:
            raise ValueError(f'{name} backend did not supply the exact frozen expert prior')
        self.training_prior = backend[7]
        self.amp_training = self.training_prior if name == 'run' else None
        if policy != protocol:
            raise ValueError(f'{name} rollout protocol differs')
        if self.reward_adapter is not None or (name == 'run') != (self.amp_training is not None):
            raise ValueError('Mixed native training requires an independent trainable AMP discriminator')
        self.observation = obs
        self.base = self.canonical(obs, None)

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
        self.observation = obs
        self.base = self.canonical(obs, done)
        return dict(environment_reward=reward, done=done, timeout=timeout)


def prepare_tasks(a, plan, root, attempt, protocol, counts, worker_seed):
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
                    '--output', str(attempt/task), '--mode', 'evaluate', '--ppo-channel', str(root),
                    '--profile', a.profile, '--randomization-strength', str(a.randomization_strength),
                    '--seed', str(worker_seed), '--headless', '--device', a.device]
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
def collect_native(a, sim, tasks, counts, learner, adapter, root, *, steps=STEPS_PER_TASK):
    """Use the native learner's sampling, transition storage and AMP reward path."""
    from native_observations import amp_native_observations
    def observations():
        packed = adapter.concat([adapter.pack(task.name, task.base, task.observation['critic'])
                                 for task in tasks])
        if not torch.isfinite(packed['policy']).all() or not torch.isfinite(packed['critic']).all():
            raise ValueError('Nonfinite native mixed policy or critic observation')
        return packed
    obs = observations()
    records = [[] for task in tasks]
    for _ in range(steps):
        check_pause(root)
        actions = learner.act(obs)
        if not torch.isfinite(actions).all():
            raise ValueError('Nonfinite native PPO action')
        outcomes = advance(tasks, counts, actions, tasks[0].raw.cfg.decimation,
                           lambda: sim.step(render=False))
        obs = observations()
        raw_reward, done, timeout = (torch.cat([outcome[key] for outcome in outcomes])
                                     for key in ('environment_reward', 'done', 'timeout'))
        amp_obs = amp_native_observations(tasks[-1].observation)
        if not torch.isfinite(amp_obs['disc']).all() or not torch.isfinite(amp_obs['disc_demo']).all():
            raise ValueError('Nonfinite native AMP discriminator samples')
        ppo_reward = learner.process_mixed_env_step(obs, raw_reward, done, timeout, amp_obs)
        if not torch.isfinite(ppo_reward).all() or not torch.isfinite(raw_reward).all():
            raise ValueError('Nonfinite reward during mixed PPO collection')
        for rows, outcome, reward in zip(records, outcomes, ppo_reward.split(counts), strict=True):
            rows.append({key: value.clone() for key, value in dict(reward=reward, **outcome).items()})
    batches = []
    for task, rows in zip(tasks, records, strict=True):
        batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        batches.append(dict(task=task.name, **batch))
    return batches, obs


@torch.no_grad()
def verify_expert_initialization(tasks, learner, adapter, expected_hashes):
    """Check the initialized full Critic and raw action std against live teachers.

    This occurs before restoring a training checkpoint or adapting the Critic.
    The first real native observations, including context, are used for parity.
    """
    if tuple(task.name for task in tasks) != TASKS or set(expected_hashes) != set(TASKS):
        raise ValueError('Missing or disordered expert initialization tasks')
    audits = {}
    for index, task in enumerate(tasks):
        prior = task.training_prior['expert_prior']
        if prior['checkpoint_sha256'] != expected_hashes[task.name]:
            raise ValueError(f'{task.name} expert checkpoint identity differs')
        expected_std = prior['action_std'].detach().cpu()
        actual_std = learner.policy.std[index].detach().cpu()
        if expected_std.shape != (23,) or not torch.equal(expected_std, actual_std):
            raise ValueError(f'{task.name} initial exploration std differs from frozen teacher')
        packed = adapter.pack(task.name, task.base, task.observation['critic'])
        predicted = learner.policy.evaluate(packed)
        expected = task.training_prior['teacher_value'].to(predicted.device)
        if (predicted.shape != expected.shape or not torch.isfinite(predicted).all()
                or not torch.isfinite(expected).all()
                or not torch.allclose(predicted, expected, rtol=1e-5, atol=1e-6)):
            raise ValueError(f'{task.name} initial expert critic prediction parity failed')
        audits[task.name] = dict(checkpoint_sha256=prior['checkpoint_sha256'],
            critic_max_abs_error=float((predicted-expected).abs().max()),
            critic_native_normalizer_prefix=prior['critic_normalizer_base_width'],
            action_std=expected_std.tolist(), action_std_mean=float(expected_std.mean()),
            action_std_min=float(expected_std.min()), action_std_max=float(expected_std.max()),
            observation_rows=len(expected))
    return audits


def train(a, root, attempt, plan, candidate, protocol, counts):
    from finetune_settings import NATIVE_PPO_SETTINGS
    from finetune_state import load_native_checkpoint, save_native_checkpoint, prune_native_checkpoints
    from native_observations import NativeObservationAdapter
    # Pin the current bundled RSL-RL before frozen teacher runtime paths are installed.
    sys.path.insert(0, str(HERE.parents[3] / 'rsl_rl'))
    from native_ppo import build_native_learner
    import inspect
    from rsl_rl.algorithms.ppo_amp import PPOAMP
    if Path(inspect.getfile(PPOAMP)).resolve() != HERE.parents[3]/'rsl_rl/rsl_rl/algorithms/ppo_amp.py':
        raise RuntimeError('Mixed learner did not load the current bundled native PPOAMP')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model, _ = load_student(a.student, a.device)
    if model.schema != plan['schema']:
        raise ValueError('Candidate actor schema differs')
    digest = sha(root/'manifest.json')
    history = checkpoint_history(root, digest)
    completed = history[-1]['completed_updates'] if history else 0
    if completed > a.updates:
        raise ValueError('Mixed PPO checkpoint exceeds its frozen update budget')
    worker_seed = (a.seed + completed*1009) % 2**32
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    sim, tasks = prepare_tasks(a, plan, root, attempt, protocol, counts, worker_seed)
    # Native scene creation seeds its sampling. Protect the coordinator's RNG,
    # and use a deterministic CUDA stream for each fresh physical session.
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    torch.cuda.manual_seed_all(worker_seed)
    widths = tuple(task.observation['critic'].shape[1] for task in tasks)
    adapter = NativeObservationAdapter(widths)
    amp = tasks[-1].amp_training
    teacher_priors = {task.name:task.training_prior['expert_prior'] for task in tasks}
    learner = build_native_learner(model, counts, widths, amp['amp_cfg'], amp['amp_state'],
                                   adapter.augment, a.device, teacher_priors=teacher_priors, steps=STEPS_PER_TASK)
    expected_hashes = {name:plan['teachers'][name]['checkpoint_sha256'] for name in TASKS}
    write_new(attempt/'expert_prior_audit.json', dict(initial_actor_sha256=sha(a.student),
        tasks=verify_expert_initialization(tasks, learner, adapter, expected_hashes)))
    print('NATIVE_EXPERT_PRIORS_VERIFIED',
          {task:float(learner.policy.std[i].detach().mean()) for i,task in enumerate(TASKS)},flush=True)
    restored = load_native_checkpoint(root, learner, digest)
    if restored is not None and restored['completed_updates'] != completed:
        raise ValueError('Restored native learning state differs from its committed update')
    retain_updates = {0, a.updates, *[i for i in (100, 200, 500, 800, 1000) if i <= a.updates]}
    if history:
        prune_native_checkpoints(root, history, retain_updates=retain_updates)
    step_history = reward_history_rows(history, TASKS)
    mean_history = list(mean_reward_rows(history, TASKS))
    if completed:
        write_training_reward_csv(root, step_history)
        write_mean_reward_csv(root, TASKS, mean_history)
    if completed == a.updates:
        publish_training_completion(a, root, model.cpu(), candidate, digest, protocol)
        return
    if a.init_at_random_ep_len:
        for task in tasks:
            initialize_episode_length(task.raw.episode_length_buf, task.raw.max_episode_length)
    write_new(attempt/'episode_start.json', {
        task.name: dict(min_step=int(task.raw.episode_length_buf.min()),
                        max_step=int(task.raw.episode_length_buf.max()),
                        distinct_steps=task.raw.episode_length_buf.unique().numel(),
                        max_episode_steps=int(task.raw.max_episode_length))
        for task in tasks})
    write_new(attempt/'mixed_scene.json', dict(physics_scenes=1, pid=os.getpid(),
              tasks={task.name: dict(envs=task.raw.num_envs, scope=task.raw.scene.scope,
                                    joints=task.wrapper.num_actions) for task in tasks},
              physics_dt=sim.get_physics_dt(), control_dt=tasks[0].raw.step_dt))
    print('MIXED_SCENE_READY', counts, flush=True)
    rewards = RolloutRewards(history[-1].get('training_reward', {}).get('mean_reward_windows') if history else None)
    warmup = NATIVE_PPO_SETTINGS['critic_warmup_updates']
    if restored is None:
        original_actor = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        original_std = learner.policy.std.detach().cpu().clone()
        learner.configure_critic_only(True)
        warmup_rows = []
        for update in range(1, warmup+1):
            check_pause(root); assert_protocol_unchanged(protocol)
            batches, next_obs = collect_native(a, sim, tasks, counts, learner, adapter, root)
            reward_summary = rewards.summarize(batches, attempt.name)
            metrics = learner.compute_returns_and_update(next_obs, critic_only=True)
            warmup_rows.append(dict(update=update, metrics=metrics))
            print('NATIVE_CRITIC_WARMUP', update, '/', warmup, metrics, flush=True)
        learner.configure_critic_only(False)
        if any(not torch.equal(value.detach().cpu(), original_actor[name])
               for name, value in model.state_dict().items()):
            raise RuntimeError('Critic-only warmup unexpectedly modified the original shared Actor')
        if not torch.equal(learner.policy.std.detach().cpu(), original_std):
            raise RuntimeError('Critic-only warmup unexpectedly modified inherited exploration std')
        del original_actor
        write_new(attempt/'critic_warmup.json', warmup_rows)
        receipt = save_native_checkpoint(root, learner, 0, digest, warmup_completed=warmup,
                                         training_reward=reward_summary)
        history.append(receipt)
    elif restored['warmup_completed'] != warmup:
        raise ValueError('Native checkpoint has an incomplete Critic warmup')
    for update in range(completed+1, a.updates+1):
        assert_protocol_unchanged(protocol)
        torch.cuda.reset_peak_memory_stats(a.device)
        begin = time.monotonic()
        batches, next_obs = collect_native(a, sim, tasks, counts, learner, adapter, root)
        torch.cuda.synchronize(a.device)
        collected = time.monotonic()
        reward_summary = rewards.summarize(batches, attempt.name)
        metrics = learner.compute_returns_and_update(next_obs)
        torch.cuda.synchronize(a.device)
        metrics.update(collection_seconds=collected-begin, update_seconds=time.monotonic()-collected,
                       torch_peak_mib=torch.cuda.max_memory_allocated(a.device)/2**20)
        del batches
        check_pause(root)
        receipt = save_native_checkpoint(root, learner, update, digest, metrics,
                                         warmup_completed=warmup, training_reward=reward_summary)
        history.append(receipt)
        prune_native_checkpoints(root, history, retain_updates=retain_updates)
        step_history.extend(reward_history_rows([receipt], TASKS))
        mean_history.extend(mean_reward_rows([receipt], TASKS))
        write_training_reward_csv(root, step_history)
        write_mean_reward_csv(root, TASKS, mean_history)
        print('MIXED_PPO_UPDATE', update, '/', a.updates, metrics, flush=True)
    publish_training_completion(a, root, model.cpu(), candidate, digest, protocol)


def main():
    from finetune_settings import NATIVE_PPO_ALGORITHM, NATIVE_PPO_SETTINGS
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design-id', required=True)
    parser.add_argument('--num-envs', type=int, default=MIXED_BO_ENVS)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--init-at-random-ep-len', action=argparse.BooleanOptionalAction, default=True,
                        help='Use ordinary RSL-RL episode counter initialization before the first update')
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--profile', choices=('nominal', 'robust'), default='robust')
    parser.add_argument('--randomization-strength', type=float, default=1.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-paused', action='store_true')
    parser.add_argument('--compact-training-artifacts', action='store_true')
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
    files = ['mixed_finetune.py', 'finetune_common.py', 'mixed_scene.py', 'mixed_batch.py', 'native_ppo.py', 'native_observations.py',
             'rollout.py', 'rollout_amp.py', 'student.py', 'training_reward.py', 'finetune_state.py',
             'finetune_settings.py', '../../../../rsl_rl/rsl_rl/algorithms/ppo_amp.py',
             '../../../../rsl_rl/rsl_rl/algorithms/ppo.py', '../../../../rsl_rl/rsl_rl/storage/rollout_storage.py',
             '../../../../rsl_rl/rsl_rl/storage/circular_buffer.py', '../../../../rsl_rl/rsl_rl/modules/amp.py']
    manifest = dict(kind='shared_candidate_finetuning_mixed',
              plan_sha256=sha(a.plan), initial_actor_sha256=sha(a.student), candidate=candidate,
              updates=a.updates, seed=a.seed, collection_protocol=protocol, ppo_schedule=PPO_SCHEDULE,
              ppo_algorithm=NATIVE_PPO_ALGORITHM, native_ppo_settings=NATIVE_PPO_SETTINGS,
              teacher_checkpoint_sha256={name:plan['teachers'][name]['checkpoint_sha256'] for name in TASKS},
              init_at_random_ep_len=a.init_at_random_ep_len,
              envs_per_task=dict(zip(TASKS, counts, strict=True)), steps_per_task=STEPS_PER_TASK,
              compact_training_artifacts=a.compact_training_artifacts,
              sources={name: sha(HERE/name) for name in files})
    with run_lock(root):
        if a.resume or a.resume_paused:
            if (json.loads((root/'manifest.json').read_text(encoding='utf8')) != manifest
                    or load_plan(root/'plan.json') != plan):
                raise ValueError('Mixed PPO candidate, budget, episode timing or sources changed')
        else:
            root.mkdir(parents=True, exist_ok=True)
            frozen_plan = root/'plan.json'
            if any(path != frozen_plan for path in root.iterdir()):
                raise ValueError('Mixed PPO output already exists; resume the frozen run')
            if frozen_plan.is_file():
                if load_plan(frozen_plan) != plan:
                    raise ValueError('Mixed PPO preparation belongs to a different candidate')
            else:
                write_new(frozen_plan, plan)
            write_new(root/'manifest.json', manifest)
        resume_pause(root, a.resume_paused)
        attempt = root/'attempts'/uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        a.kit_args = (a.kit_args or '') + f' --/renderer/multiGpu/enabled=false --portable-root {attempt/"portable"} --portable'
        app = AppLauncher(a).app
        try:
            train(a, root, attempt, plan, candidate, protocol, counts)
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                (root/'USER_PAUSED').touch()
            write_new(attempt/'failure.json', dict(traceback=traceback.format_exc()))
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
    # All artifacts are committed before exit. The three subscenes share one
    # context, so running the native per-environment Kit teardown would clear
    # the context repeatedly and the global close can hang.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
