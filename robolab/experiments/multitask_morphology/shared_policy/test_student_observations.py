import json
import pytest
import torch
from student_observations import StudentHistory, EpisodeTracker, TrackingError, speed_bins


def test_noise_only_new_sensor_frame_and_global_rng_unchanged():
    torch.manual_seed(47)
    state = torch.random.get_rng_state().clone()
    h = StudentHistory(2, 'cpu', 'run', profile='robust', strength=1., seed=9)
    clean = torch.ones(2, 78)
    first = h.append(clean).reshape(2, 10, 78)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(first[:, -1, 6:9], clean[:, 6:9])
    assert torch.equal(first[:, -1, 55:], clean[:, 55:])
    assert not torch.equal(first[:, -1, :3], clean[:, :3])
    second = h.append(clean * 2).reshape(2, 10, 78)
    assert torch.equal(second[:, :-1], first[:, 1:])
    reset = h.append(clean * 3, torch.tensor([True, False])).reshape(2, 10, 78)
    assert torch.equal(reset[0], reset[0, -1:].expand(10, -1))
    assert torch.equal(reset[1, :-1], second[1, 1:])


def test_nominal_is_clean_and_seeded_robust_is_repeatable():
    x = torch.randn(3, 78)
    clean = StudentHistory(3, 'cpu', 'stairs', profile='nominal', strength=1., seed=1)
    assert torch.equal(clean.append(x).reshape(3, 10, 78)[:, -1], x)
    a = StudentHistory(3, 'cpu', 'backpack', profile='robust', strength=.5, seed=12)
    b = StudentHistory(3, 'cpu', 'backpack', profile='robust', strength=.5, seed=12)
    assert torch.equal(a.append(x), b.append(x))


def test_bins_and_episode_identity_and_tracking_excludes_resets():
    cmd = torch.tensor([[.2, 0., 0.], [.4, 0., 0.], [.6, 0., 0.]])
    assert speed_bins(cmd, 'stairs').tolist() == [0, 1, 2]
    episodes = EpisodeTracker(2, 'cpu')
    episodes.advance(torch.tensor([True, False]))
    assert episodes.episode.tolist() == [1, 0]
    error = TrackingError(2, 'cpu')
    error.update(torch.tensor([.5, .5]), torch.tensor([100., .3]), torch.tensor([False, True]))
    assert torch.isnan(error.values()[0])
    torch.testing.assert_close(error.values()[1], torch.tensor(.2, dtype=torch.float64))


@pytest.mark.parametrize('speed', [0., .04, .05, -1., float('nan')])
def test_collection_speed_bins_reject_disabled_standing_or_invalid_commands(speed):
    with pytest.raises(ValueError, match='standing commands are disabled'):
        speed_bins(torch.tensor([[speed, 0., 0.]]), 'stairs')


def test_tracking_missing_measurements_are_null_with_separate_command_exposure():
    error = TrackingError(2, 'cpu')
    error.update(torch.tensor([2., .5]), torch.tensor([100., .3]), torch.tensor([False, True]),
                 dt=.02, command_valid=torch.tensor([True, True]))
    rows = error.rows()
    json.dumps(rows, allow_nan=False)
    assert rows[0]['tracking_rmse'] is None and rows[0]['tracking_samples'] == 0
    assert rows[0]['command_seconds'] == .02 and rows[0]['tracking_seconds'] == 0.
    assert rows[0]['commanded_distance_m'] == .04 and rows[0]['commanded_vx_rms_m_s'] == 2.
    assert rows[1]['command_aligned_distance_m'] == pytest.approx(.006)
    assert rows[1]['tracking_rmse'] == pytest.approx(.2)
    assert rows[0]['moving_command_samples'] == 1 and rows[0]['moving_tracking_samples'] == 0
    assert rows[0]['moving_tracking_rmse'] is None and rows[0]['standing_vx_rms_m_s'] is None


def test_standing_drift_does_not_count_as_commanded_motion_progress():
    error = TrackingError(1, 'cpu')
    yes = torch.tensor([True])
    error.update(torch.tensor([0.]), torch.tensor([5.]), yes, dt=.02)
    error.update(torch.tensor([.04]), torch.tensor([5.]), yes, dt=.02)
    error.update(torch.tensor([2.]), torch.tensor([1.]), yes, dt=.02)
    row = error.rows()[0]
    assert row['command_seconds'] == .06 and row['active_command_seconds'] == .02
    assert row['commanded_distance_m'] == .04
    assert row['command_aligned_distance_m'] == .02
    assert row['commanded_vx_rms_m_s'] == pytest.approx(((.04**2 + 2**2) / 3)**.5)
    assert row['moving_commanded_vx_rms_m_s'] == 2.
    assert row['moving_tracking_rmse'] == 1.
    assert row['standing_commanded_vx_rms_m_s'] == pytest.approx((.04**2 / 2)**.5)
    assert row['standing_vx_rms_m_s'] == 5.
    assert row['standing_command_samples'] == row['standing_tracking_samples'] == 2


def test_negative_command_uses_matching_direction_for_progress():
    error = TrackingError(1, 'cpu')
    error.update(torch.tensor([-2.]), torch.tensor([-1.]), torch.tensor([True]), dt=.02)
    row = error.rows()[0]
    assert row['commanded_distance_m'] == .04
    assert row['command_aligned_distance_m'] == .02


def test_tracking_rejects_nonfinite_valid_samples_and_untimed_reports():
    error = TrackingError(1, 'cpu')
    with pytest.raises(ValueError, match='Nonfinite'):
        error.update(torch.tensor([1.]), torch.tensor([float('nan')]), torch.tensor([True]), dt=.02)
    error.update(torch.tensor([1.]), torch.tensor([1.]), torch.tensor([True]))
    with pytest.raises(ValueError, match='Timed'):
        error.rows()
