"""Rebind an existing Python 3.12 reference environment without changing dependencies."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from setup_hhtools import SOURCE, verify_snapshot


def command():
    return [sys.executable, '-m', 'pip', 'install', '--no-deps',
            '--no-build-isolation', '-e', str(SOURCE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    verify_snapshot()
    if args.dry_run:
        print(json.dumps(command()))
        return
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Run inside the existing Python 3.12 amp_hhtools environment')
    # Never invoke the dependency installer or recreate the environment here.
    subprocess.run(command(), check=True)
    probe = ('import hhtools; from pathlib import Path; '
             'assert Path(hhtools.__file__).resolve().parent == Path(' +
             repr(str(SOURCE / 'hhtools')) + ').resolve(); '
             'from importlib.metadata import version; '
             'assert version("hhtools") == "0.1.0"')
    subprocess.run([sys.executable, '-c', probe], check=True)
    print('HHTOOLS_REBOUND', SOURCE, flush=True)


if __name__ == '__main__':
    main()
