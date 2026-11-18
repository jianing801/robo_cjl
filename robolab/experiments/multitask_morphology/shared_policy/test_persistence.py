"""Publication races and explicit pause handling, using actual filesystem operations."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Barrier

import pytest

from contract import write_new
from persistence import atomic_publish, run_lock, resume_pause


def test_reader_never_sees_half_written_request(tmp_path):
    partial = Event(); finish = Event(); path = tmp_path/'request.json'
    def writer(stream):
        stream.write(b'{"steps":'); stream.flush(); partial.set()
        assert finish.wait(5)
        stream.write(b'24}')
    with ThreadPoolExecutor() as executor:
        future = executor.submit(atomic_publish, path, writer)
        assert partial.wait(5)
        try: assert not path.exists()
        finally: finish.set()
        future.result(timeout=5)
    assert json.loads(path.read_text()) == {'steps': 24}
    assert not list(tmp_path.glob('*.writing'))


def test_two_publishers_cannot_overwrite_one_another(tmp_path):
    barrier = Barrier(2); path = tmp_path/'result.json'
    def publish(value):
        def writer(stream):
            stream.write(json.dumps({'value':value}).encode()); barrier.wait(timeout=5)
        try: atomic_publish(path, writer); return value
        except FileExistsError: return None
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(publish, [1, 2]))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1 and json.loads(path.read_text())['value'] == winners[0]
    with pytest.raises(FileExistsError): write_new(path, {'value': 3})


def test_failed_serialization_never_leaves_a_published_result(tmp_path):
    path = tmp_path/'result.json'
    with pytest.raises(ValueError): write_new(path, {'bad':float('nan')})
    assert not path.exists()
    def fail(stream):
        stream.write(b'partial'); raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError): atomic_publish(path, fail)
    assert not path.exists() and not list(tmp_path.glob('*.writing'))


def test_run_lock_releases_on_error_and_pause_requires_explicit_resume(tmp_path):
    root = tmp_path/'run'; root.mkdir()
    with run_lock(root):
        with pytest.raises(RuntimeError, match='coordinator'):
            with run_lock(root): pass
    with pytest.raises(RuntimeError):
        with run_lock(root): raise RuntimeError('failure')
    with run_lock(root): pass
    marker = root/'USER_PAUSED'; marker.write_text('manual stop')
    with pytest.raises(RuntimeError, match='resume-paused'): resume_pause(root, False)
    assert marker.read_text() == 'manual stop'
    resume_pause(root, True)
    assert not marker.exists()
    assert next((root/'pause_history').glob('*.marker')).read_text() == 'manual stop'
