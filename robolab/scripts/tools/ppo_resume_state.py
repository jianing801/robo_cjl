"""Restore the adaptive scheduler scalar as well as Adam parameter groups."""

import math


def restore_learning_rate(algorithm):
    rates = [float(group["lr"]) for group in algorithm.optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Checkpoint optimizer learning rate is invalid")
    if any(not math.isclose(rate, rates[0], rel_tol=1e-9) for rate in rates):
        raise ValueError("PPO's scalar scheduler cannot represent different group learning rates")
    algorithm.learning_rate = rates[0]
    return rates[0]
