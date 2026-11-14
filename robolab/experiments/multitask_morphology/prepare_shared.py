"""Bind three completed experts to a strict common-bank shared-policy plan."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'shared_policy'))
from contract import schema_for, paired_structures, sha, write_new, load_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for task in ('rough','backpack','amp'):
        p.add_argument('--'+task+'-root', required=True, help='Completed expert training output root')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists():
        raise ValueError('Shared plan output must be new')
    teachers = {}
    for public, native in [('rough','stairs'),('backpack','backpack'),('amp','run')]:
        raw = subprocess.check_output([sys.executable, str(HERE/'validate_teacher.py'),
            '--task', native, '--root', getattr(a, public+'_root')], text=True, encoding='utf8')
        teachers[native] = json.loads(raw)
    banks = {t:json.loads(Path(s['bank']).read_text(encoding='utf8')) for t,s in teachers.items()}
    pairs = paired_structures(banks)
    if any(len(pairs) != len(b['designs']) for b in banks.values()):
        raise ValueError('New unified experts must cover exactly the same full structure library')
    schema = schema_for(banks)
    schema['shared_structure_keys'] = [list(key) for key,_ in pairs]
    for task, spec in teachers.items():
        spec['design_ids'] = [rows[task]['id'] for _,rows in pairs]
    out.mkdir(parents=True, exist_ok=False)
    shared = out/'shared_bank.json'
    write_new(shared, dict(kind='three_task_common_structure_bank_v1',
        structures=[dict(key=list(key),design_ids={t:r['id'] for t,r in rows.items()}) for key,rows in pairs]))
    plan = dict(kind='rough_backpack_amp_distillation_plan_v3',
        structure_scope='teacher_intersection_only', scope='all',
        schema=schema, teachers=teachers, task_ids={t:i for i,t in enumerate(teachers)},
        selected_structure_keys=schema['shared_structure_keys'], shared_bank=str(shared),
        shared_bank_sha256=sha(shared), full_common_design_count=len(pairs))
    write_new(out/'plan.json', plan)
    load_plan(out/'plan.json')
    print('SHARED_PLAN_VALIDATED', out/'plan.json', len(pairs), 'structures')


if __name__ == '__main__':
    main()
