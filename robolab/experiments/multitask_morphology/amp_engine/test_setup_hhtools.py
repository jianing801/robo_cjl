import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_hhtools as setup


def test_bundled_snapshot_matches():
    assert setup.verify_snapshot() > 300


def test_modified_source_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'vendor' / 'hhtools'
    source.mkdir(parents=True)
    (source / 'example.py').write_bytes(b'changed')
    manifest = {'files': {'example.py': hashlib.sha256(b'original').hexdigest()}}
    (tmp_path / 'vendor' / 'hhtools_snapshot.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(setup, 'HERE', tmp_path)
    monkeypatch.setattr(setup, 'SOURCE', source)
    with pytest.raises(ValueError, match='snapshot mismatch'):
        setup.verify_snapshot()


@pytest.mark.parametrize('name', ['base', 'robo_cjl', '../robo_cjl', 'amp_hhtools;bad'])
def test_training_and_invalid_envs_rejected(name):
    with pytest.raises(ValueError):
        setup.check_env_name(name)


def test_active_env_rejected(monkeypatch):
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'amp_hhtools')
    with pytest.raises(ValueError, match='active environment'):
        setup.check_env_name('amp_hhtools')


def test_dry_run_never_calls_conda_or_pip(monkeypatch, capsys):
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'robo_cjl')
    monkeypatch.setattr(sys, 'argv', ['setup_hhtools.py', '--dry-run'])
    def forbidden(*args, **kwargs):
        pytest.fail('dry-run executed a subprocess')
    monkeypatch.setattr(setup.subprocess, 'run', forbidden)
    monkeypatch.setattr(setup.subprocess, 'check_output', forbidden)
    setup.main()
    output = capsys.readouterr().out
    assert 'amp_hhtools' in output and '[robot,retarget]' in output
