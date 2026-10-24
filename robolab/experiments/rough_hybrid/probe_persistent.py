"""Experimental real-PPO scene rebuilding. Not a production/resume entry.

Use a separately generated SMALL bank/config. Never feed a production output.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--kit-args', default='')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    from hybrid_contract import experiment_identity, stage_spec, validate_config, validate_checkpoint
    from hybrid_bank import load_configurable_bank
    from hybrid_support import write_new
    bank_path = Path(a.bank).resolve()
    bank = load_configurable_bank(bank_path)
    config = json.loads(Path(a.config).read_text())
    validate_config(config, bank)
    if config['grid']['num_envs'] > 64 or config['rotation']['total_updates'] > 12:
        raise ValueError('probe limited to <=64 environments and <=12 total PPO updates')
    if not a.execute:
        print('Planning only; explicit --execute required'); return
    from hybrid_processes import reject_duplicate_bank_safely
    reject_duplicate_bank_safely(bank_path)
    root = Path(a.output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    experiment = experiment_identity(bank_path, config)
    write_new(root/'plan.json', dict(configuration=config, experiment=experiment))
    session = {}
    rows = []
    failed = False
    try:
        import train_stage
        index = 0
        previous = None
        while index*config['rotation']['updates_per_stage'] < config['rotation']['total_updates']:
            stage = stage_spec(bank, config, index)
            args = ['--bank', str(bank_path), '--hybrid-plan', str(root/'plan.json'),
                    '--stage-index', str(index), '--output-dir', str(root/f'group_{index}'),
                    '--execute', '--updates', str(stage['end_update']), '--headless',
                    '--device', 'cuda:0', '--kit_args', a.kit_args]
            if previous: args += ['--resume-checkpoint', str(previous)]
            start = time.monotonic()
            before = None
            if session.get('runner'):
                r = session['runner']
                before = (id(r.alg.policy), id(r.alg.optimizer), id(r.alg.storage))
            original_argv = sys.argv
            try:
                # AppLauncher inspects sys.argv inside add_app_launcher_args.
                sys.argv = [str(HERE/'train_stage.py'), *args]
                train_stage.main(args, session=session)
            finally:
                sys.argv = original_argv
            r = session['runner']
            objects = (id(r.alg.policy), id(r.alg.optimizer), id(r.alg.storage))
            if before is not None and before != objects:
                raise RuntimeError('persistent policy/Adam/storage objects replaced')
            previous = Path(r.logger.log_dir)/f"model_{stage['end_update']-1}.pt"
            state = validate_checkpoint(previous, experiment, stage['end_update'])
            steps = [float(x['step']) for x in state['optimizer_state_dict']['state'].values()]
            import torch
            rows.append(dict(stage=index, pid=os.getpid(), objects=objects,
                             completed=stage['end_update'], adam_steps=steps,
                             checkpoint=str(previous), seconds=time.monotonic()-start,
                             allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved()))
            write_new(root/f'group_{index}_audit.json', rows[-1])
            # Native learn leaves iter at the LAST completed index.
            r.current_learning_iteration = stage['end_update']
            index += 1
        write_new(root/'probe_complete.json', dict(groups=rows, production_ready=False,
                  note='Small real PPO only. Not a VRAM reservation or full-scale benchmark.'))
    except BaseException:
        failed = True
        import traceback
        write_new(root/'probe_failure.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        if session.get('app'):
            if failed:
                import atexit
                atexit.register(lambda: os._exit(1))
            session['app'].close()


if __name__ == '__main__': main()
