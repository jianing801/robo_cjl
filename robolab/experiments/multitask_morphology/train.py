"""Unified expert entry. No task runs until --execute is explicitly supplied."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TASKS = {'rough': 'stairs', 'backpack': 'backpack', 'amp': 'run'}


def command(task, config, bank, arguments):
    value = json.loads(Path(bank).read_text(encoding='utf8'))
    if value.get('task_key') != TASKS[task]:
        raise ValueError('Task and bank identity differ')
    settings = json.loads(Path(config).read_text(encoding='utf8'))
    if value.get('configuration') != settings['grid']:
        raise ValueError('Training configuration and bank grid differ; prepare the matching bank')
    if value.get('context_schema') != settings.get('context_schema'):
        raise ValueError('Training configuration and bank context schema differ; prepare the matching bank')
    engine = HERE / ('amp_engine' if task == 'amp' else 'engine')
    return [sys.executable, '-u', str(engine / 'train.py'), '--config', str(config),
            '--bank', str(Path(bank).resolve()), *arguments]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--config', help='Defaults to this folder/configs/TASK.json')
    p.add_argument('--bank', required=True)
    a, extra = p.parse_known_args()
    cmd = command(a.task, a.config or HERE / 'configs' / (a.task + '.json'), a.bank, extra)
    # Replace the dispatcher on Unix so Ctrl-C and process ownership remain clear.
    # On Windows the child receives console interrupts in the same process group.
    if sys.platform != 'win32':
        import os
        os.execv(sys.executable, cmd)
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
