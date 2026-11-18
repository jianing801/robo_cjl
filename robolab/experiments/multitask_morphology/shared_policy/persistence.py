"""Atomic, exclusive publication and process-lifetime locks on local filesystems."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import uuid


def atomic_publish(path, writer):
    """Expose a complete file once; never truncate/replace an existing result."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.writing', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking the closed temporary file is atomic and fails if the destination
        # exists. A check-then-rename would silently overwrite on POSIX.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json_new(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf8')
    atomic_publish(path, lambda stream: stream.write(encoded))


@contextmanager
def run_lock(root):
    """OS releases the lock on process death, so stale PID cleanup is unnecessary."""
    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / f'.{root.name}.coordinator.lock'
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another coordinator is active for {root}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resume_pause(root, explicit):
    """Call only after run identity validation and while holding the run lock."""
    root = Path(root)
    marker = root / 'USER_PAUSED'
    if not marker.exists():
        return
    if not explicit:
        raise RuntimeError('USER_PAUSED; pass --resume-paused to continue this run')
    history = root / 'pause_history'
    history.mkdir(exist_ok=True)
    event = uuid.uuid4().hex
    write_json_new(history / f'{event}.json', dict(explicit_resume=True, pid=os.getpid()))
    marker.rename(history / f'{event}.marker')
