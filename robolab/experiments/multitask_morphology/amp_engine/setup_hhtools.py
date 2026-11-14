"""Install the bundled reference generator in a separate Python 3.12 conda env."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'vendor' / 'hhtools'


def verify_snapshot():
    manifest = json.loads((HERE / 'vendor' / 'hhtools_snapshot.json').read_text(encoding='utf-8-sig'))
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid snapshot path: {name}')
        path = SOURCE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundled HHTools snapshot mismatch: {name}')
    return len(manifest['files'])


def check_env_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name.lower() in ('base', 'robo_cjl'):
        raise ValueError('Use a separate conda environment name, e.g. amp_hhtools')
    active = os.environ.get('CONDA_DEFAULT_ENV', '')
    if active and name.lower() == Path(active).name.lower():
        raise ValueError('Do not install into the active environment; activate robo_cjl first')
    return name


def install_commands(conda, name):
    run = [conda, 'run', '--no-capture-output', '-n', name, 'python']
    return [
        run + ['-m', 'pip', 'install', '-r', str(HERE / 'hhtools_requirements.txt'),
               '-e', str(SOURCE) + '[robot,retarget]'],
        run + ['-m', 'pip', 'check'],
        run + [str(Path(__file__).resolve()), '--verify-runtime'],
    ]


def verify_runtime():
    from importlib.metadata import version
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('HHTools reference generation requires Python 3.12')
    import hhtools
    if Path(hhtools.__file__).resolve().parent != (SOURCE / 'hhtools').resolve():
        raise RuntimeError('Imported HHTools does not come from the bundled snapshot')
    for line in (HERE / 'hhtools_requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        package, expected = line.split('==')
        if version(package) != expected:
            raise RuntimeError(f'{package}: expected {expected}, found {version(package)}')
    if version('hhtools') != '0.1.0':
        raise RuntimeError('Unexpected HHTools version')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import warp as wp
    wp.config.enable_cuda = False
    from hhtools.retarget import robot_to_robot  # noqa: F401
    from hhtools.retarget.newton_basic.robot_model import build_newton_model  # noqa: F401
    print('HHTOOLS_RUNTIME_READY', sys.executable, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-name', default='amp_hhtools')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-runtime', action='store_true')
    args = parser.parse_args()
    print('HHTOOLS_SNAPSHOT_VERIFIED', verify_snapshot(), flush=True)
    if args.verify_runtime:
        verify_runtime()
        return
    name = check_env_name(args.env_name)
    conda = os.environ.get('CONDA_EXE') or shutil.which('conda') or 'conda'
    create = [conda, 'create', '--yes', '-n', name, 'python=3.12', 'pip']
    commands = install_commands(conda, name)
    if args.dry_run:
        print(json.dumps({'create_if_absent': create, 'commands': commands}, indent=2))
        return
    info = json.loads(subprocess.check_output([conda, 'env', 'list', '--json'], text=True))
    paths = [Path(p) for p in info['envs'] if Path(p).name == name]
    if len(paths) > 1:
        raise RuntimeError('Ambiguous environment name; choose a unique name')
    if not paths:
        subprocess.run(create, check=True)
    # Reject an existing incompatible env before installing any package.
    subprocess.run([conda, 'run', '-n', name, 'python', '-c',
                    "import sys; assert sys.version_info[:2] == (3,12), 'Use a new Python 3.12 environment'"], check=True)
    for command in commands:
        subprocess.run(command, check=True)
    print('Ready. Generate references with conda run -n', name,
          'python', HERE / 'prepare_references.py', '--bank <bank.json> --output <new-directory>',
          '--hhtools-root', SOURCE, flush=True)


if __name__ == '__main__':
    main()
