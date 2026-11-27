"""Measure the original RPO leg dimensions and motors with the BO protocol."""

import argparse
from copy import deepcopy
from pathlib import Path

import bo
from contract import structure_key


RPO_REFERENCE = bo.RPO_REFERENCE_KEY
BANKS = {'stairs': 'rough', 'backpack': 'backpack', 'run': 'amp'}


def reference_plan(work, assets, plan):
    frozen = deepcopy(plan)
    ids = {}
    origin = bo.read(assets/'assets_ready.json')
    grid = bo.read(bo.HERE/'configs/rpo_original_configuration.json')['grid']
    for task, folder in BANKS.items():
        path = assets/'banks'/folder/'bank.json'
        bank_hash = bo.sha(path)
        bank = bo.read(path)
        (row,) = bank['designs']
        if (bank_hash != origin['banks'][folder] or bank['task_key'] != task
                or row['id'] != origin['design_id'] or structure_key(row) != RPO_REFERENCE
                or bank.get('context_schema') != plan['schema'].get('context_schema')
                or bank['configuration'] != grid):
            raise ValueError(f'Wrong predicted RPO reference bank: {path}')
        original_bank = bo.read(plan['teachers'][task]['bank'])
        pair = row['pair_id']
        if (bank['motor_pairs'][pair]['selected'] != original_bank['motor_pairs'][pair]['selected']
                or bank['original_template_sha256'] != original_bank['original_template_sha256']
                or bank['training']['source_hashes'] != original_bank['training']['source_hashes']):
            raise ValueError(f'RPO reference motor models, template or runtime differ from the BO bank: {task}')
        ids[task] = row['id']
        frozen['teachers'][task].update(bank=str(path), bank_sha256=bank_hash, design_ids=[row['id']])
    lib = assets/'references/library.json'
    lib_hash = bo.sha(lib)
    if lib_hash != origin['library_sha256'] or bo.read(lib)['bank_sha256'] != frozen['teachers']['run']['bank_sha256']:
        raise ValueError('AMP references do not match the predicted RPO bank')
    frozen['teachers']['run'].update(reference_library=str(lib), reference_library_sha256=lib_hash)
    shared_bank = work/'reference_bank.json'
    expected = bo.read(plan['shared_bank'])['kind']
    bo.freeze(shared_bank, dict(kind=expected, structures=[dict(key=list(RPO_REFERENCE), design_ids=ids)]))
    frozen.update(reference_structure_key=list(RPO_REFERENCE), selected_structure_keys=[list(RPO_REFERENCE)],
                  shared_bank=str(shared_bank), shared_bank_sha256=bo.sha(shared_bank))
    source = work/'reference_plan.json'
    bo.freeze(source, frozen)
    bo.load_plan(source)
    return source, frozen, ids['stairs']


def summarize(work, report):
    (pair,) = report['pairs']
    tasks = {}
    for task, trial in pair['tasks'].items():
        observed = trial.get('observed') or {}
        tasks[task] = {name: observed.get(name) for name in (
            'absolute_mechanical_cot', 'cot', 'forward_rmse_mps',
            'roll_pitch_rms_rad_s', 'knee_ankle_max_rated_ratio')}
        tasks[task].update(status=trial['status'], reason=trial['reason'], steps=trial['steps'],
                          rate_hz=1/trial['dt_s'], forward_m=trial['forward_m'],
                          mean_forward_mps=trial['mean_forward_mps'])
    result = dict(candidate=report['candidate'], seed=pair['seed'],
              jointly_feasible=pair['feasible'], ppo_updates=report['ppo_updates'], tasks=tasks,
              conditions_sha256=report['conditions_sha256'], student=report['student'],
              student_sha256=report['student_sha256'],
              report_sha256=bo.sha(work/'candidate/bo_evaluation.json'))
    if report.get('ppo_algorithm') is not None:
        result['ppo_algorithm'] = report['ppo_algorithm']
    bo.freeze(work/'reference_result.json', result)
    print('RPO_REFERENCE_EVALUATION_READY', work/'reference_result.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shared-continuation', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume-paused', action='store_true')
    args = parser.parse_args()
    args.conditions = bo.PHYSICS_CONDITIONS
    args.updates = bo.DEFAULT_BO_UPDATES
    args.timeout = 1800
    args.finetuned = None
    args.ppo_algorithm = bo.NATIVE_PPO_ALGORITHM
    work = args.output.resolve()
    with bo.run_lock(work):
        work.mkdir(exist_ok=True)
        try:
            bo.resume_pause(work, args.resume_paused)
            donor, plan, actor = bo.shared_input(args.root.resolve(), formal=True,
                require_horizontal_selection=True, continuation=args.shared_continuation.resolve())
            conditions = bo.read_conditions(args.conditions)
            bo.freeze(work/'reference_protocol.json', dict(source_plan_sha256=bo.sha(donor),
                initial_actor_sha256=bo.sha(actor), conditions_sha256=bo.digest(conditions), updates=args.updates,
                ppo_algorithm=bo.NATIVE_PPO_ALGORITHM, native_ppo_settings=bo.NATIVE_PPO_SETTINGS))
            source, candidate_plan, design_id = reference_plan(work, args.assets.resolve(), plan)
            report = bo.evaluate(args, work/'candidate', source, candidate_plan, actor, design_id, conditions)
            summarize(work, report)
        except KeyboardInterrupt:
            (work/'USER_PAUSED').touch()
            raise SystemExit(130)


if __name__ == '__main__':
    main()
