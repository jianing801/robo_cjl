"""Optimizer schedule for candidate PPO and its BO compatibility checks."""

PPO_SCHEDULE = {'epochs': 5, 'minibatches': 4}
LEGACY_PPO_SCHEDULE = {'epochs': 4, 'minibatches': 3}
MIXED_BO_ENVS = 4096
MIXED_BO_LAYOUT = {'stairs': 1366, 'backpack': 1365, 'run': 1365}

# Preserve the exact v3 historical identity for independent evaluation/audit.
# It had a randomly initialized shared Critic and a single std initialized at 1.
NATIVE_PPO_ALGORITHM_V3 = 'shared_native_ppoamp_v3'
NATIVE_PPO_SETTINGS_V3 = dict(
    steps_per_env=24, epochs=5, minibatches=4, gamma=.99, lam=.95,
    clip_param=.2, value_loss_coef=1., use_clipped_value_loss=True,
    schedule='adaptive', desired_kl=.01, initial_learning_rate=1e-4,
    adaptive_learning_rate_bounds=[1e-5, 1e-2], max_grad_norm=1.,
    initial_noise_std=1., learn_noise_std=True,
    entropy_coefficients={'stairs': .005, 'backpack': .005, 'run': .01},
    symmetry_augmentation=True, symmetry_tasks=['stairs', 'run'], mirror_loss_coef=.2,
    critic_warmup_updates=10, critic_hidden_dims=[512, 256, 128],
    critic_observation_normalization=True, amp_discriminator_training=True,
    amp_disc_learning_rate=1e-4, amp_disc_buffer_size=100, amp_style_reward_scale=1.5,
    amp_task_style_lerp=.6, amp_loss_type='LSGAN', amp_gradient_penalty=10.,
    amp_disc_trunk_weight_decay=1e-3, amp_disc_linear_weight_decay=1e-1,
)

# V4 copies each teacher's native critic and 23 learned action standard
# deviations, then fits those critic priors to the distilled student. PPO
# updates a single shared actor and three independent native value networks.
NATIVE_PPO_ALGORITHM = 'shared_native_ppoamp_expert_prior_v4'
NATIVE_PPO_STATE_KIND = 'shared_ppo_learning_state_v4'
NATIVE_PPO_SETTINGS = dict(NATIVE_PPO_SETTINGS_V3,
    actor_initialization='frozen_round5_shared_actor',
    noise_initialization='frozen_expert_learned_std', noise_std_shape=[3, 23],
    critic_initialization='frozen_expert_critic', critic_architecture='three_independent_teacher_native',
    critic_observation_normalization={'stairs': 'identity', 'backpack': 'identity', 'run': 'teacher_native_prefix'},
    ppo_optimizer_initialization='fresh_adam', amp_disc_optimizer_initialization='fresh_adam',
)
del NATIVE_PPO_SETTINGS['initial_noise_std'], NATIVE_PPO_SETTINGS['critic_hidden_dims']

NATIVE_PPO_PROFILES = {NATIVE_PPO_ALGORITHM_V3: NATIVE_PPO_SETTINGS_V3,
                       NATIVE_PPO_ALGORITHM: NATIVE_PPO_SETTINGS}

CORRECTION_PROFILES = ('native', 'fixed_amp_std', 'fixed_amp_std_pcgrad')
STAIRS_WEIGHT_PROFILE = 'fixed_amp_std_pcgrad_stairs_weight2'
KL_PILOT_PROFILE = 'fixed_amp_std_pcgrad_kl002'
STAIRS_SAMPLE_PROFILE = 'fixed_amp_std_pcgrad_stairs_samples_2048'
V4_STAIRS5120_EPOCH10_PROFILE = 'v4_stairs5120_epoch10'
TRAINING_PROFILES = CORRECTION_PROFILES + (STAIRS_WEIGHT_PROFILE, KL_PILOT_PROFILE, STAIRS_SAMPLE_PROFILE,
                                          V4_STAIRS5120_EPOCH10_PROFILE)


def correction_settings(profile):
    """The tested correction profiles retain distinct settings."""
    if profile == 'native':
        return dict(NATIVE_PPO_SETTINGS)
    if profile == 'fixed_amp_std':
        return dict(NATIVE_PPO_SETTINGS, fixed_expert_std_tasks=['run'])
    if profile == 'fixed_amp_std_pcgrad':
        return dict(NATIVE_PPO_SETTINGS, fixed_expert_std_tasks=['run'], actor_gradient_projection='pcgrad_cyclic')
    if profile == STAIRS_WEIGHT_PROFILE:
        return dict(correction_settings('fixed_amp_std_pcgrad'),
                    actor_gradient_weights={'stairs': 2., 'backpack': 1., 'run': 1.},
                    actor_gradient_weight_normalization='mean_one')
    if profile == KL_PILOT_PROFILE:
        return dict(correction_settings('fixed_amp_std_pcgrad'), desired_kl=.02)
    if profile == STAIRS_SAMPLE_PROFILE:
        return dict(correction_settings('fixed_amp_std_pcgrad'),
                    envs_per_task={'stairs':2048,'backpack':1024,'run':1024})
    if profile == V4_STAIRS5120_EPOCH10_PROFILE:
        return dict(NATIVE_PPO_SETTINGS, epochs=10,
                    envs_per_task={'stairs':5120,'backpack':512,'run':512}, checkpoint_interval=100)
    raise ValueError(f'Unknown experimental PPO profile: {profile}')


# A new identity keeps the selected B protocol distinct from the archived V4
# calibration and from the original controlled ablation receipts.
CURRENT_PPO_PROFILE = 'fixed_amp_std_pcgrad'
CURRENT_PPO_ALGORITHM = 'shared_native_ppoamp_expert_prior_pcgrad_fixed_amp_std'
CURRENT_PPO_SETTINGS = correction_settings(CURRENT_PPO_PROFILE)
NATIVE_PPO_PROFILES[CURRENT_PPO_ALGORITHM] = CURRENT_PPO_SETTINGS
EXPERT_PRIOR_ALGORITHMS = (NATIVE_PPO_ALGORITHM, CURRENT_PPO_ALGORITHM)


def resolve_training_profile(experimental_profile=None):
    """Default to the selected B; explicit profiles reproduce the isolated studies."""
    if experimental_profile is None:
        return CURRENT_PPO_ALGORITHM, CURRENT_PPO_PROFILE, CURRENT_PPO_SETTINGS
    return NATIVE_PPO_ALGORITHM, experimental_profile, correction_settings(experimental_profile)
