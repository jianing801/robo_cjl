"""The isolated original-RPO 5120/512/512, 10-epoch V4 convergence experiment."""
import pytest
import torch

from test_native_ppo import Actor, teacher_priors, amp_settings, observation, amp_obs


def test_profile_changes_only_the_approved_v4_experimental_parameters():
    from finetune_settings import (V4_STAIRS5120_EPOCH10_PROFILE, TRAINING_PROFILES,
        NATIVE_PPO_SETTINGS, NATIVE_PPO_ALGORITHM, CURRENT_PPO_SETTINGS,
        CURRENT_PPO_ALGORITHM, resolve_training_profile)
    profile = V4_STAIRS5120_EPOCH10_PROFILE
    assert profile == 'v4_stairs5120_epoch10'
    assert profile in TRAINING_PROFILES
    algorithm, chosen, settings = resolve_training_profile(profile)
    assert (algorithm, chosen) == (NATIVE_PPO_ALGORITHM, profile)
    assert settings == dict(NATIVE_PPO_SETTINGS, epochs=10,
                            envs_per_task={'stairs':5120,'backpack':512,'run':512},
                            checkpoint_interval=100)
    assert not any(key in settings for key in ('fixed_expert_std_tasks', 'actor_gradient_projection','actor_gradient_weights'))
    assert resolve_training_profile()[0] == CURRENT_PPO_ALGORITHM
    assert resolve_training_profile()[2] == CURRENT_PPO_SETTINGS
    assert 'checkpoint_interval' not in CURRENT_PPO_SETTINGS
    assert CURRENT_PPO_SETTINGS['epochs'] == 5


def test_experimental_layout_and_manifest_schedule_are_exact():
    from finetune_settings import V4_STAIRS5120_EPOCH10_PROFILE, correction_settings, PPO_SCHEDULE
    from mixed_finetune import configured_task_counts, configured_ppo_schedule, inspect_actor_gradients
    settings = correction_settings(V4_STAIRS5120_EPOCH10_PROFILE)
    counts = configured_task_counts(6144, settings)
    assert counts == (5120,512,512)
    assert tuple(n * settings['steps_per_env'] for n in counts) == (122880,12288,12288)
    assert configured_ppo_schedule(settings) == dict(epochs=10, minibatches=4)
    with pytest.raises(ValueError, match='allocation'):
        configured_task_counts(4096, settings)
    assert configured_task_counts(4096, correction_settings('native')) == (1366,1365,1365)
    assert configured_ppo_schedule(correction_settings('native')) == PPO_SCHEDULE
    assert not inspect_actor_gradients(V4_STAIRS5120_EPOCH10_PROFILE, 25)
    assert inspect_actor_gradients('fixed_amp_std_pcgrad', 25)


def test_real_v4_learner_executes_all_ten_epochs_without_correction():
    from finetune_settings import V4_STAIRS5120_EPOCH10_PROFILE, correction_settings
    from mixed_finetune import build_configured_learner
    settings = correction_settings(V4_STAIRS5120_EPOCH10_PROFILE)
    cfg, state = amp_settings()
    counts = (20,2,2)
    # Same per-task ratio and optimizer cadence; reduced physics-free CPU sample count.
    alg = build_configured_learner(Actor(), counts, (4,6,3), cfg, state, None, 'cpu',
                                  teacher_priors(), settings, steps=2)
    assert alg.num_learning_epochs == 10 and alg.num_mini_batches == 4
    assert alg.actor_projection is False and alg.actor_gradient_weights is None
    assert alg._fixed_std is None and alg.policy.std.requires_grad
    seen = 0
    original_step = alg.optimizer.step
    def counted_step(*args, **kwargs):
        nonlocal seen
        seen += 1
        return original_step(*args, **kwargs)
    alg.optimizer.step = counted_step
    obs = observation(counts)
    for i in range(2):
        with torch.inference_mode():
            alg.act(obs)
            following = observation(counts)
            raw = torch.arange(float(sum(counts))) + 1.
            dones = torch.zeros(sum(counts),dtype=torch.bool)
            alg.process_mixed_env_step(following,raw,dones,dones,amp_obs(n=counts[-1],value=1.+i))
            obs = following
    metrics = alg.compute_returns_and_update(obs)
    assert seen == 40 and metrics['samples'] == 2*sum(counts)
    assert 'actor_gradient_weights' not in metrics and 'actor_gradient_sample' not in metrics


def test_experiment_requires_exact_resume_sources():
    from copy import deepcopy
    from finetune_settings import V4_STAIRS5120_EPOCH10_PROFILE, correction_settings
    from native_resume_compat import validate_native_resume_manifest
    from test_native_resume_compat import LATEST_SOURCES, manifest, new_sources
    settings = correction_settings(V4_STAIRS5120_EPOCH10_PROFILE)
    current = manifest(new_sources())
    current.update(experimental_profile=V4_STAIRS5120_EPOCH10_PROFILE,
                   native_ppo_settings=settings, ppo_schedule={'epochs':10,'minibatches':4},
                   envs_per_task=settings['envs_per_task'])
    assert validate_native_resume_manifest(current, deepcopy(current))
    saved = deepcopy(current)
    saved['sources'] = dict(LATEST_SOURCES)
    assert not validate_native_resume_manifest(saved, current)
