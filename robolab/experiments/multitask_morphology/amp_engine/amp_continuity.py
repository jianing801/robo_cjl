"""CPU-testable AMP group transition; never resets learned state."""
import torch


def learned_objects(runner):
    alg = runner.alg
    return tuple(id(value) for value in (alg.policy, alg.optimizer, alg.storage,
        alg.amp_discriminator, alg.disc_optimizer,
        alg.amp_discriminator.disc_obs_normalizer))


def rebind_group(runner, env, groups):
    obs = env.get_observations()
    storage = runner.alg.storage
    if set(obs.keys()) != set(storage.observations.keys()) or any(
            obs[key].shape != storage.observations[key].shape[1:] for key in obs.keys()):
        raise ValueError('AMP group changed observation/storage shapes')
    # Replay contains the previous group's physical trajectories/references.
    # Keep allocations, but never mix those samples into the new group.
    with torch.inference_mode():
        storage.clear()
        runner.alg.disc_obs_buffer.reset()
        runner.alg.disc_demo_obs_buffer.reset()
        for name in ('cur_reward_sum', 'cur_episode_length', 'cur_total_reward_sum',
                     'cur_style_reward_sum', 'cur_ereward_sum', 'cur_ireward_sum'):
            value = getattr(runner.logger, name, None)
            if value is not None:
                value.zero_()
    storage.spatial_groups = groups
    runner.env = env
    if runner.alg.symmetry:
        runner.alg.symmetry['_env'] = env
    for name in ('ep_extras', 'rewbuffer', 'lenbuffer', 'total_rewbuffer',
                 'style_rewbuffer', 'erewbuffer', 'irewbuffer'):
        value = getattr(runner.logger, name, None)
        if value is not None:
            value.clear()


def capture_rng(session):
    if session is not None:
        session['learning_rng'] = (torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(session):
    cpu, cuda = session['learning_rng']
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)
