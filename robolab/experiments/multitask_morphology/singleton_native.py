"""Opt-in native single-layout training ledger and partial-stage recovery.

The native learner, rollout, curriculum and reward are unchanged.  A restarted
simulator restores learning state, not the interrupted physical episode.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import time
import sys

TASK_BUDGETS = dict(stairs=12000, backpack=12000, run=5000)
DESIGN_KEY = [.25,.30,'DM-J10010L-2EC','DM-J4340P-2EC']


def singleton_config(bank, task):
    if bank.get('task_key') != task or len(bank.get('designs', [])) != 1:
        raise ValueError('Expected the exact one-design task bank')
    design = bank['designs'][0]
    key = [design[k] for k in ('thigh_m','calf_m','knee_motor','ankle_motor')]
    if (key != DESIGN_KEY or design['split'] != 'train' or bank['configuration']['num_envs'] != 4096
            or bank['context_schema'] != 'leg_motor_inertia_v1'):
        raise ValueError('Original RPO singleton physical identity differs')
    budget = TASK_BUDGETS[task]
    return dict(grid=bank['configuration'], context_schema=bank['context_schema'], full_body_torque=True,
                rotation=dict(active_designs=1, updates_per_stage=budget, total_updates=budget, seed=42))


def check_single_stage(stage, target):
    if (stage['first_update'] != 0 or stage['end_update'] != target
            or len(stage['design_ids']) != 1):
        raise ValueError('Singleton training requires a single fixed layout for the entire budget')


def checkpoint_boundary(state, stage):
    check_single_stage(stage, stage['end_update'])
    if (state.get('hybrid_stage') != stage or type(state.get('iter')) is not int
            or not 0 < state['iter'] + 1 <= stage['end_update']):
        raise ValueError('Checkpoint does not belong to the exact singleton stage')
    return state['iter'] + 1


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, row):
    """Only publish a receipt after all model bytes have been written."""
    path = Path(path)
    temp = path.with_suffix('.writing')
    with temp.open('w', encoding='utf8') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def install_ledger(runner, output, pause_root, stage):
    check_single_stage(stage, stage['end_update'])
    output, pause_root = Path(output), Path(pause_root)
    receipts = output/'checkpoint_receipts'; receipts.mkdir(exist_ok=True)
    original_save, original_log = runner.save, runner.logger.log
    def save(path, infos=None):
        original_save(path, infos)
        path = Path(path).resolve()
        completed = runner.current_learning_iteration + 1
        receipt = receipts/f'update_{completed:06d}.json'
        row = dict(completed_updates=completed, checkpoint=str(path), sha256=file_sha(path),
                   stage=stage, timestamp=time())
        if receipt.is_file():
            previous = json.loads(receipt.read_text(encoding='utf8'))
            if any(previous[k] != row[k] for k in ('completed_updates','checkpoint','sha256','stage')):
                raise ValueError('Committed singleton checkpoint changed')
        else:
            publish(receipt, row)
    runner.save = save
    path = output/'native_mean_reward.csv'
    columns = ('update','mean_reward','episode_window','collect_seconds','learn_seconds','learning_rate')
    with path.open('x', newline='', encoding='utf8') as stream:
        csv.DictWriter(stream, fieldnames=columns).writeheader()
    def log(**values):
        original_log(**values)
        rewards = runner.logger.rewbuffer
        row = dict(update=values['it']+1, mean_reward=fmean(rewards) if rewards else '',
                   episode_window=len(rewards), collect_seconds=float(values['collect_time']),
                   learn_seconds=float(values['learn_time']), learning_rate=float(values['learning_rate']))
        if not all(math.isfinite(row[k]) for k in ('collect_seconds','learn_seconds','learning_rate')):
            raise ValueError('Nonfinite native training diagnostics')
        with path.open('a', newline='', encoding='utf8') as stream:
            csv.DictWriter(stream, fieldnames=columns).writerow(row)
            stream.flush()
        if (pause_root/'USER_PAUSED').exists():
            save(Path(runner.logger.log_dir)/f'model_{runner.current_learning_iteration}.pt')
            raise KeyboardInterrupt('USER_PAUSED')
    runner.logger.log = log


def restore_within_stage(runner, path, experiment, stage, validate, *, amp=False):
    """Strictly validate and restore the same fields as the native stage loader."""
    import torch
    raw = torch.load(path, map_location='cpu', weights_only=False)
    completed = checkpoint_boundary(raw, stage)
    del raw
    state = validate(path, experiment, completed, within_stage=stage)
    runner.alg.policy.load_state_dict(state['model_state_dict'], strict=True)
    runner.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
    if amp:
        runner.alg.amp_discriminator.load_state_dict(state['amp_discriminator_state_dict'], strict=True)
        runner.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
            state['amp_discriminator_normalizer_state_dict'], strict=True)
        runner.alg.disc_optimizer.load_state_dict(state['amp_discriminator_optimizer_state_dict'])
    runner.alg.learning_rate = state['learning_rate']
    for group in runner.alg.optimizer.param_groups: group['lr'] = runner.alg.learning_rate
    runner.current_learning_iteration = completed
    torch.set_rng_state(state['torch_rng_state'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng_states']])
    return completed


def native_modules(task):
    import importlib
    here = Path(__file__).resolve().parent
    runtime = here/('amp_engine' if task=='run' else 'engine')
    sys.path[:0] = [str(runtime), str(here/'engine'), str(here.parents[1]), str(here.parents[2]/'rsl_rl')]
    prefix = 'amp' if task=='run' else 'hybrid'
    return importlib.import_module(prefix+'_contract'), importlib.import_module(prefix+'_bank')


def prepare(task, bank_path, root, source, library=None):
    import shutil
    contract, bank_module = native_modules(task)
    from hybrid_assets import asset_files, mesh_sources
    from hybrid_native_cache import validate_cache
    bank = bank_module.load_configurable_bank(bank_path)
    config = singleton_config(bank, task)
    contract.validate_config(config, bank)
    experiment = (contract.experiment_identity(bank_path, config, library) if task=='run'
                  else contract.experiment_identity(bank_path, config))
    stage = contract.stage_spec(bank, config, 0)
    check_single_stage(stage, TASK_BUDGETS[task])
    if task=='run':
        from amp_references import validate_library
        value=validate_library(bank,bank_path,library,stage['design_ids'])
        if value.get('validation_only'): raise ValueError('AMP reference library is validation-only')
    root, source = Path(root).resolve(), Path(source).resolve()
    root.mkdir(parents=True,exist_ok=True)
    cache = root/'cache'; cache.mkdir(exist_ok=True)
    design = bank['designs'][0]
    directory = cache/design['id']
    if not (cache/'completed.json').is_file():
        if directory.exists(): raise ValueError('Incomplete cache copy; preserve it for inspection')
        shutil.copytree(source, directory)
        usd, = directory.glob('*.usd')
        row = dict(design=design['id'],converter_hash=(directory/'.asset_hash').read_text().strip(),
                   urdf_sha256=design['urdf_sha256'],usd=str(usd),usd_sha256=file_sha(usd),files_sha256=asset_files(directory))
        publish(cache/'completed.json',dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,
                mesh_sources=mesh_sources(bank),bank_sha256=file_sha(bank_path),task=task,designs=1,assets=[row]))
    validate_cache(cache,bank_path)
    plan = dict(configuration=config,experiment=experiment,bank=str(bank_path),preflight_updates_per_new_layout=0,
                execution_protocol='direct_stage_v2',total_updates_are_absolute=True)
    if task=='run': plan['reference_library']=str(library)
    plan_path=root/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan: raise ValueError('Singleton native plan changed')
    else: publish(plan_path,plan)
    print('SINGLETON_PREPARED',task,stage,flush=True)


def verify(task, plan_path, checkpoint, completed):
    contract, _ = native_modules(task)
    plan=json.loads(Path(plan_path).read_text())
    bank_path=Path(plan['bank'])
    plan=contract.load_plan(plan_path,bank_path)
    _, bank_module=native_modules(task)
    bank=bank_module.load_configurable_bank(bank_path)
    stage=contract.stage_spec(bank,plan['configuration'],0)
    state=contract.validate_checkpoint(checkpoint,plan['experiment'],completed,within_stage=stage)
    if checkpoint_boundary(state,stage)!=completed: raise ValueError('Resume boundary changed')
    print('SINGLETON_CHECKPOINT_VALIDATED',task,completed,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify'));parser.add_argument('--task',choices=TASK_BUDGETS,required=True)
    parser.add_argument('--bank',type=Path);parser.add_argument('--root',type=Path);parser.add_argument('--source-cache',type=Path)
    parser.add_argument('--library',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--completed',type=int)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.task,args.bank,args.root,args.source_cache,args.library)
    else: verify(args.task,args.plan,args.checkpoint,args.completed)
