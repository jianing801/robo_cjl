import json
import threading
from types import SimpleNamespace
from hybrid_resources import prepare_scene, arm_failed_cleanup_timeout


def test_boundary_releases_only_unused_cache(tmp_path, monkeypatch):
    import hybrid_resources as module
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='0, gpu-id, 100, 200'))
    monkeypatch.setattr(module.gc, 'collect', lambda: calls.append('gc'))
    cuda = SimpleNamespace(is_initialized=lambda: True, mem_get_info=lambda: (100, 300),
        current_device=lambda: 0, memory_allocated=lambda: 50, memory_reserved=lambda: 80,
        synchronize=lambda: calls.append('sync'), empty_cache=lambda: calls.append('empty'))
    path = tmp_path/'memory.jsonl'
    prepare_scene(path, SimpleNamespace(cuda=cuda))
    assert calls == ['gc', 'sync', 'empty']
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['phase'] for r in rows] == ['before_unused_cache_release', 'before_scene_creation']
    assert all(r['allocated'] == 50 for r in rows)


def test_failure_watchdog_runs_without_cleanup_returning():
    done = threading.Event()
    timer = arm_failed_cleanup_timeout(0.02, done.set)
    assert timer.daemon
    assert done.wait(2)


def test_parent_watchdog_kills_failed_child_but_not_quiet_healthy_child(tmp_path):
    import subprocess
    import sys
    from hybrid_resources import watch_failed_worker
    (tmp_path/'result').mkdir()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    stop, thread = watch_failed_worker(child, tmp_path, seconds=0.1, interval=0.02)
    try:
        # Quiet alone is never a failure signal.
        assert not stop.wait(0.2)
        assert child.poll() is None
        (tmp_path/'result/cleanup_failure.json').write_text('{}')
        child.wait(timeout=10)
        assert child.returncode != 0
    finally:
        stop.set()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        thread.join(timeout=10)


def test_snapshot_cuda_failure_is_diagnostic_only(tmp_path, monkeypatch):
    import hybrid_resources as module
    def broken():
        raise RuntimeError('CUDA query failed')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=''))
    cuda = SimpleNamespace(is_initialized=lambda: True, mem_get_info=broken)
    row = module.snapshot(tmp_path/'memory.jsonl', 'diagnostic', SimpleNamespace(cuda=cuda))
    assert row['cuda_query_error'] == 'CUDA query failed'
