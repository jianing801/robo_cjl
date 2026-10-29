import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amp_reference_prepare as prepare
import amp_references


@pytest.fixture
def checked(monkeypatch):
    calls = []
    def validate(bank, bank_path, path):
        value = json.loads(Path(path).read_text())
        if value.get('bank') != 'correct':
            raise ValueError('wrong reference identity')
        calls.append(Path(path))
        return value
    monkeypatch.setattr(amp_references, 'validate_library', validate)
    return calls


def test_preview_never_generates_or_creates_output(tmp_path, monkeypatch, checked):
    path = tmp_path / 'refs/library.json'
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *a, **k: pytest.fail('preview generated references'))
    assert prepare.ensure_library({}, 'bank', path, execute=False) is None
    assert not path.parent.exists()


def test_existing_library_checked_without_generator(tmp_path, monkeypatch, checked):
    path = tmp_path / 'library.json'
    path.write_text('{"bank":"correct"}')
    original = path.read_bytes()
    monkeypatch.setattr(prepare.subprocess, 'run', lambda *a, **k: pytest.fail('regenerated existing library'))
    assert prepare.ensure_library({}, 'bank', path, execute=True)['bank'] == 'correct'
    assert path.read_bytes() == original and checked == [path]
    path.write_text('{"bank":"wrong"}')
    with pytest.raises(ValueError, match='identity'):
        prepare.ensure_library({}, 'bank', path, execute=True)


def test_missing_resume_never_regenerates(tmp_path, checked):
    with pytest.raises(FileNotFoundError, match='original reference'):
        prepare.ensure_library({}, 'bank', tmp_path/'missing/library.json', execute=True, resume=True)
    assert not (tmp_path/'missing').exists()


def test_generation_validates_before_publication_and_keeps_absolute_paths(tmp_path, monkeypatch, checked):
    path = tmp_path / 'refs/library.json'
    def generate(command, **kwargs):
        assert not path.exists()
        assert command[command.index('-n') + 1] == 'amp_hhtools'
        output = Path(command[command.index('--output') + 1])
        output.mkdir()
        (output/'library.json').write_text(json.dumps({'bank': 'correct', 'directory': str(output)}))
    monkeypatch.setattr(prepare.subprocess, 'run', generate)
    value = prepare.ensure_library({}, 'bank', path, execute=True)
    assert Path(value['directory']).is_dir()
    assert checked == [Path(value['directory'])/'library.json']
    assert json.loads(path.read_text()) == value


@pytest.mark.parametrize('failure', ['exit', 'missing', 'invalid', 'validation_only'])
def test_generation_failures_never_publish_library(tmp_path, monkeypatch, checked, failure):
    path = tmp_path / 'refs/library.json'
    def generate(command, **kwargs):
        output = Path(command[command.index('--output')+1])
        output.mkdir()
        if failure == 'exit':
            raise subprocess.CalledProcessError(1, command)
        if failure != 'missing':
            (output/'library.json').write_text(json.dumps({'bank': 'wrong' if failure == 'invalid' else 'correct',
                                                         'validation_only': failure == 'validation_only'}))
    monkeypatch.setattr(prepare.subprocess, 'run', generate)
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
        prepare.ensure_library({}, 'bank', path, execute=True)
    assert not path.exists()
    assert len(list(path.parent.glob('generated_*'))) == 1


def test_explicit_python_does_not_use_conda():
    command = prepare.generation_command('bank', 'new', python='/separate/python')
    assert command[0] == '/separate/python' and 'conda' not in command
