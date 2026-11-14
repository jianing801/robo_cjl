"""One Kit/AMP learner across groups; only physical scenes/references rotate."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]


def run_groups(root, bank_path, result, kit_args='', asset_cache=None, stage_main=None):
    from amp_contract import load_plan, stage_spec, validate_checkpoint, sha
    from amp_bank import load_configurable_bank
    from amp_support import write_new
    from amp_continuity import learned_objects
    # Load the sibling controller under a unique name, not a cached Rough 'train'.
    import importlib.util
    spec = importlib.util.spec_from_file_location('amp_controller', HERE/'train.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    plan = load_plan(root/'plan.json', bank_path)
    if plan.get('execution_protocol') != 'amp_persistent_v1':
        raise ValueError('persistent AMP plan required')
    bank = load_configurable_bank(bank_path)
    rows = controller.committed(root, plan, bank)
    index = len(rows)
    previous = json.loads(rows[-1].read_text())['checkpoint'] if rows else None
    if stage_main is None:
        from train_stage import main as stage_main
    session = {'failure_signal': result/'cleanup_failure.json'}
    result.mkdir(parents=True, exist_ok=False)
    failed = False
    try:
        target = plan['configuration']['rotation']['total_updates']
        while index*plan['configuration']['rotation']['updates_per_stage'] < target:
            if (root/'USER_PAUSED').exists():
                raise RuntimeError('USER_PAUSED; no next AMP group')
            stage = stage_spec(bank, plan['configuration'], index)
            if previous:
                validate_checkpoint(previous, plan['experiment'], stage['first_update'])
            output = root/'attempts'/f'group_{index:06d}_{datetime.now():%Y%m%d_%H%M%S_%f}'/'result'
            args = ['--bank', str(bank_path), '--hybrid-plan', str(root/'plan.json'),
                '--reference-library', plan['reference_library'], '--stage-index', str(index),
                '--output-dir', str(output), '--execute', '--updates', str(stage['end_update']),
                '--headless', '--device', 'cuda:0', '--kit_args', kit_args,
                '--asset-cache', str(asset_cache or root/'attempts')]
            if not asset_cache:
                args += ['--allow-cache-misses']
            if previous:
                args += ['--resume-checkpoint', previous]
            before = learned_objects(session['runner']) if session.get('runner') else None
            print(f'AMP GROUP {index+1}: updates {stage["first_update"]}..{stage["end_update"]-1}/{target}; PID={os.getpid()}', flush=True)
            # AppLauncher.add_app_launcher_args parses process sys.argv before
            # our explicit parse_args(argv). Expose the current group's argv,
            # not the controller/worker CLI, then restore it even on failure.
            original = sys.argv
            try:
                sys.argv = [str(HERE/'train_stage.py'), *args]
                stage_main(args, session=session)
            finally:
                sys.argv = original
            if (output/'failure.json').exists():
                raise RuntimeError(f'AMP group failure: {output}')
            summary = json.loads((output/'summary.json').read_text())
            if summary['result'] != 'AMP_HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED' or summary['completed_updates'] != stage['end_update']:
                raise ValueError('AMP group did not complete exact frozen target')
            runner = session['runner']
            objects = learned_objects(runner)
            if before is not None and before != objects:
                raise ValueError('AMP learned objects/rollout allocations replaced')
            checkpoint = Path(runner.logger.log_dir)/f'model_{stage["end_update"]-1}.pt'
            state = validate_checkpoint(checkpoint, plan['experiment'], stage['end_update'])
            if state['hybrid_stage'] != stage:
                raise ValueError('AMP checkpoint group differs')
            import torch
            memory = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else {}
            write_new(root/'commits'/f'stage_{index:06d}.json', dict(stage=stage,
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                summary=str(output/'summary.json'), pid=os.getpid(), learned_objects=objects,memory=memory))
            runner.current_learning_iteration = stage['end_update']
            previous = str(checkpoint)
            index += 1
            print('AMP_STAGE_COMMITTED', index, stage['end_update'], '/', target, flush=True)
        write_new(result/'summary.json', dict(result='PERSISTENT_AMP_COMPLETE_NOT_GAIT_VALIDATED',
            completed_updates=target, checkpoint=previous))
    except BaseException:
        failed = True
        write_new(result/'failure.json', dict(traceback=traceback.format_exc()))
        from amp_rough_alignment import rough_module
        rough_module('hybrid_resources').arm_failed_cleanup_timeout()
        raise
    finally:
        if session.get('app'):
            if failed:
                import atexit
                atexit.register(lambda: os._exit(1))
            session['app'].close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--result', required=True)
    p.add_argument('--kit-args', default='')
    p.add_argument('--asset-cache')
    a = p.parse_args()
    run_groups(Path(a.root).resolve(), Path(a.bank).resolve(), Path(a.result).resolve(), a.kit_args, a.asset_cache)


if __name__ == '__main__':
    main()
