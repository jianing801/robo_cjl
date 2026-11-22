from types import SimpleNamespace

import pytest
import torch

from mixed_batch import advance, initialize_episode_length, task_counts


def test_fixed_mixed_layout():
    assert task_counts(24) == (8, 8, 8)
    assert task_counts(4096) == (1366, 1365, 1365)
    with pytest.raises(ValueError):
        task_counts(2)


def test_episode_counter_initialization_matches_native_rsl_rl():
    counters = torch.zeros(32, dtype=torch.long)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        expected = torch.randint_like(counters, high=1000)
        torch.manual_seed(7)
        initialize_episode_length(counters, 1000)
    torch.testing.assert_close(counters, expected)
    assert counters.unique().numel() > 1


def test_each_native_scene_writes_before_one_shared_physics_tick():
    trace = []
    tasks = [
        SimpleNamespace(begin=lambda a, i=i: trace.append(("prepare", i, a.item())),
                        apply=lambda i=i: trace.append(("write", i)),
                        sync=lambda i=i: trace.append(("update", i)),
                        finish=lambda i=i: trace.append(("finish", i)) or i)
        for i in range(3)
    ]
    assert advance(tasks, (1, 1, 1), torch.tensor([[7], [8], [9]]), 2, lambda: trace.append("physics")) == [0, 1, 2]
    assert trace == ([
        ("prepare", 0, 7), ("prepare", 1, 8), ("prepare", 2, 9),
    ] + [("write", 0), ("write", 1), ("write", 2), "physics",
         ("update", 0), ("update", 1), ("update", 2)] * 2
       + [("finish", 0), ("finish", 1), ("finish", 2)])
