"""Optimizer schedule for candidate PPO and its BO compatibility checks."""

PPO_SCHEDULE = {'epochs': 5, 'minibatches': 4}
LEGACY_PPO_SCHEDULE = {'epochs': 4, 'minibatches': 3}
MIXED_BO_ENVS = 4096
MIXED_BO_LAYOUT = {'stairs': 1366, 'backpack': 1365, 'run': 1365}

# New native PPOAMP learner. Keep PPO_SCHEDULE and the original layout names
# unchanged so that historical 100-update hand-written PPO evidence is readable.
NATIVE_PPO_ALGORITHM = 'shared_native_ppoamp_v3'
NATIVE_PPO_SETTINGS = dict(
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
