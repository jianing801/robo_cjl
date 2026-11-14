"""Read-only validation in an isolated interpreter; prints a teacher manifest."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['stairs','backpack','run'], required=True)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    engine = HERE/('amp_engine' if a.task == 'run' else 'engine')
    sys.path[:0] = [str(engine), str(ROOT), str(ROOT.parent/'rsl_rl')]
    if a.task == 'run':
        from amp_contract import load_plan, validate_checkpoint, sha
        from amp_bank import load_configurable_bank
        from amp_references import validate_library
    else:
        from hybrid_contract import load_plan, validate_checkpoint, sha
        from hybrid_bank import load_configurable_bank
    root = Path(a.root).resolve()
    completed = json.loads((root/'completed.json').read_text(encoding='utf8'))
    raw_plan = json.loads((root/'plan.json').read_text(encoding='utf8'))
    bank_path = Path(raw_plan['bank']).resolve()
    plan = load_plan(root/'plan.json', bank_path)
    bank = load_configurable_bank(bank_path)
    if bank['task_key'] != a.task:
        raise ValueError('Wrong expert task')
    target = plan['configuration']['rotation']['total_updates']
    if completed['completed_updates'] != target:
        raise ValueError('Expert has not completed its configured absolute budget')
    checkpoint = Path(completed['checkpoint']).resolve()
    state = validate_checkpoint(checkpoint, plan['experiment'], target)
    report = dict(code_root=str(ROOT), runtime_dir=str(engine), bank=str(bank_path),
        original_bank=str(bank_path), bank_sha256=sha(bank_path), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), expected_iter=state['iter'],
        context_width=bank['context_width'], full_design_count=len(bank['designs']),
        asset_cache=[str(root)], copies=8, full_body_torque=True)
    if not plan['configuration'].get('full_body_torque'):
        raise ValueError('Expected the new full-body torque expert configuration')
    if a.task == 'run':
        library = plan['reference_library']
        validate_library(bank, bank_path, library)
        report.update(reference_library=library, reference_library_sha256=sha(library))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
