"""Prepare three exact common structure banks, generating base URDFs only once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--template', required=True)
    p.add_argument('--mesh-dir', required=True)
    p.add_argument('--generate-assets', action='store_true', help='Otherwise CPU preview only')
    a = p.parse_args()
    root = Path(a.output_root).resolve()
    configs = {t: HERE/'configs'/f'{t}.json' for t in ('rough','backpack','amp')}
    grids = [json.loads(p.read_text())['grid'] for p in configs.values()]
    if not grids[0] == grids[1] == grids[2]:
        raise ValueError('Three task morphology grids differ')
    if a.generate_assets and root.exists():
        raise ValueError('Use a NEW output root; existing banks are never overwritten')
    common = ['--generate-assets'] if a.generate_assets else []
    commands = [
        [sys.executable, str(HERE/'engine/prepare_bank.py'), '--config', str(configs['rough']),
         '--template', a.template, '--mesh-dir', a.mesh_dir, '--output-dir', str(root/'banks/rough'), *common],
        [sys.executable, str(HERE/'engine/prepare_bank.py'), '--task-key', 'backpack',
         '--config', str(configs['backpack']), '--rough-bank', str(root/'banks/rough/bank.json'),
         '--output-dir', str(root/'banks/backpack'), *common],
        [sys.executable, str(HERE/'amp_engine/prepare_bank.py'), '--config', str(configs['amp']),
         '--rough-bank', str(root/'banks/rough/bank.json'), '--output-dir', str(root/'banks/amp'), *common]]
    for cmd in commands:
        subprocess.run(cmd, check=True)
    if a.generate_assets:
        from audit_banks import audit
        report = audit(root/'banks/rough/bank.json', root/'banks/amp/bank.json', root/'banks/backpack/bank.json')
        (root/'bank_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        print('THREE_TASK_BANKS_VALIDATED', root, flush=True)


if __name__ == '__main__':
    main()
