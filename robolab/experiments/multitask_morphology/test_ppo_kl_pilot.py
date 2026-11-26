from ppo_kl_pilot import pilot_assessment


def series(n, stairs, backpack=30., run=31.):
    return [dict(update=i,stairs_mean_reward=stairs(i) if callable(stairs) else stairs,
                 backpack_mean_reward=backpack,run_mean_reward=run) for i in range(1,n+1)]


def test_pilot_distinguishes_early_plateau_from_mere_slope():
    b=series(1000,lambda i:32 if i<200 else 38)
    early=pilot_assessment(series(200,37),b)
    assert early['promising'] and early['early_plateau_candidate']
    slow=pilot_assessment(series(200,lambda i:32+i/100),b)
    assert not slow['early_plateau_candidate']
    broken=pilot_assessment(series(200,37,run=20),b)
    assert not broken['promising'] and not broken['other_tasks_preserved']
