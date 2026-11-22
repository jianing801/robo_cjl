"""Optimizer schedule for candidate PPO and its BO compatibility checks."""

PPO_SCHEDULE = {'epochs': 5, 'minibatches': 4}
LEGACY_PPO_SCHEDULE = {'epochs': 4, 'minibatches': 3}
MIXED_BO_ENVS = 4096
MIXED_BO_LAYOUT = {'stairs': 1366, 'backpack': 1365, 'run': 1365}
