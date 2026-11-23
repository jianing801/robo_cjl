"""Continue a live three-task distillation into formal BO calibration."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from bo import (HERE, REPO, PHYSICS_CONDITIONS, calibration_keys, check_panel_receipt,
                checked_measurement_panel, checked_report, completed_actor, digest, freeze_scoring,
                official_conditions_path, panel_scoring, paired_trials, read, read_conditions, run_lock,
                sha, shared_input, uses_current_schedule, write_calibration_summary)
from finetune_settings import MIXED_BO_LAYOUT, PPO_SCHEDULE
from shared_bo_metrics import ABSOLUTE_COT_BASIS, scoring_cot_basis


def own_distillation(command, root):
    if '--output' not in command[:-1]:
        return False
    return (any(Path(arg).resolve() == HERE/'distill.py' for arg in command[1:5])
            and Path(command[command.index('--output') + 1]).resolve() == root.resolve())


def wait_for_distillation(root, process, sleep=time.sleep):
    while True:
        if (root/'USER_PAUSED').exists():
            raise RuntimeError('USER_PAUSED: distillation paused before BO calibration')
        try:
            active = process is not None and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            active = False
        if not active:
            if (root/'completed.json').is_file():
                return
            raise RuntimeError('Distillation exited without verified completion')
        sleep(20)


def observe_calibration(command, work):
    """Report measured wall time and storage for the first complete structure."""
    first = work/'design_01'
    report = first/'bo_evaluation.json'
    logged_first = report.exists()
    with subprocess.Popen(command, cwd=REPO) as process:
        try:
            while True:
                if not logged_first and report.is_file():
                    elapsed = report.stat().st_mtime - (first/'manifest.json').stat().st_mtime
                    footprint = sum(p.stat().st_size for p in first.rglob('*') if p.is_file())
                    print(f'BO_FIRST_STRUCTURE seconds={elapsed:.1f} bytes={footprint} '
                          f'projected_50_seconds={50*elapsed:.0f} projected_50_bytes={50*footprint}', flush=True)
                    logged_first = True
                try:
                    return process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            (work/'USER_PAUSED').touch()
            # Interrupt the owned controller so its nested workers stop as well.
            if process.poll() is None:
                if os.name == 'nt':
                    from pipeline import stop_child
                    stop_child(process)
                else:
                    process.send_signal(signal.SIGINT)
            process.wait()
            raise


def initialize(command, work, *, resume_paused, observe=observe_calibration, sleep=time.sleep):
    for attempt in range(1, 4):
        args = [*command, '--resume-paused'] if attempt == 1 and resume_paused else command
        status = observe(args, work)
        if status == 0:
            return
        paused = ((work/'USER_PAUSED').exists() or any(work.glob('design_*/USER_PAUSED'))
                  or any(work.glob('design_*/tuning/USER_PAUSED')))
        if paused or status in (130, -signal.SIGINT) or attempt == 3:
            raise subprocess.CalledProcessError(status, args)
        print('BO_CALIBRATION_RETRY', attempt, 'last_exit=', status, flush=True)
        sleep(30)


def checked_historical_reports(work, source, initial, benchmark, conditions):
    conditions_sha256 = digest(conditions)
    updates = benchmark['updates']
    mixed = uses_current_schedule(updates, conditions)
    reports = []
    for index, entry in enumerate(benchmark['reports'], 1):
        folder = work/f'design_{index:02d}'
        path = (work/entry['report']).resolve()
        if path != (folder/'bo_evaluation.json').resolve() or sha(path) != entry['sha256']:
            raise ValueError(f'Calibration report identity differs at structure {index}')
        report = checked_report(path)
        actor = completed_actor(folder/'tuning', source, initial, report['candidate'], updates,
                                ppo_schedule=PPO_SCHEDULE if mixed else None, mixed_bo=mixed)
        if (sha(actor) != report['student_sha256'] or actor != Path(report['student']).resolve()
                or report['evaluation_manifest_sha256'] != sha(folder/'manifest.json')
                or report['conditions'] != conditions):
            raise ValueError(f'Calibration actor or conditions changed at structure {index}')
        measurements = []
        for measurement in report['measurements']:
            native_path = Path(measurement['report']).resolve()
            if not native_path.is_relative_to(folder.resolve()) or sha(native_path) != measurement['sha256']:
                raise ValueError(f'Measurement receipt changed at structure {index}')
            native = read(native_path)
            if native.get('conditions_sha256') != conditions_sha256:
                raise ValueError(f'Measurement conditions differ at structure {index}')
            if native['actor_sha256'] != report['student_sha256'] or sha(native['trace']) != native['trace_sha256']:
                raise ValueError(f'Measurement actor or raw trace changed at structure {index}')
            measurements.append(native)
        if paired_trials(measurements, conditions) != report['pairs']:
            raise ValueError(f'Measurement pairing differs at structure {index}')
        reports.append(report)
    return reports


def summarize(root, work, updates, *, conditions_path=None, continuation=None):
    source, _, initial = shared_input(root, formal=True, require_horizontal_selection=True,
                                      continuation=continuation)
    scoring_path = work/'scoring.json'
    check_panel_receipt(scoring_path)
    scoring = read(scoring_path)
    frozen_conditions = official_conditions_path(scoring['conditions_sha256'])
    conditions = read_conditions(conditions_path or frozen_conditions)
    conditions_sha256 = digest(conditions)
    keys = calibration_keys()
    benchmark = scoring['benchmark']
    if (benchmark['updates'] != updates or benchmark['initial_actor_sha256'] != sha(initial)
            or benchmark['source_plan_sha256'] != sha(source) or len(benchmark['reports']) != len(keys)
            or scoring['conditions_sha256'] != conditions_sha256
            or read(work/'manifest.json').get('conditions_sha256') != conditions_sha256):
        raise ValueError('Calibration receipt has a different budget, policy, plan or conditions')
    if uses_current_schedule(updates, conditions) and (
            benchmark.get('init_at_random_ep_len') is not True or benchmark.get('envs_per_task') != MIXED_BO_LAYOUT
            or benchmark.get('ppo_schedule') != PPO_SCHEDULE):
        raise ValueError('Calibration receipt uses a different mixed PPO training configuration')
    cot_basis = scoring_cot_basis(scoring)
    if cot_basis == ABSOLUTE_COT_BASIS:
        reports, verified_keys, verified_benchmark, verified_conditions = checked_measurement_panel(work)
        if (verified_keys, verified_benchmark, verified_conditions) != (keys, benchmark, conditions):
            raise ValueError('Frozen scoring and completed 200 Hz measurement panel differ')
    else:
        reports = checked_historical_reports(work, source, initial, benchmark, conditions)
    if panel_scoring(reports, keys, benchmark, conditions=conditions, cot_basis=cot_basis,
                     task_weights=scoring['task_weights']) != scoring:
        raise ValueError('Frozen scoring differs from recomputed quantiles')
    write_calibration_summary(work, reports, scoring, conditions)


def main(argv=None):
    from bo import DEFAULT_BO_UPDATES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--conditions', type=Path, help='Approved BO conditions file')
    parser.add_argument('--shared-continuation', type=Path,
                        help='Use the verified winning actor through this completed DAgger continuation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--distillation-pid', type=int)
    parser.add_argument('--resume-paused', action='store_true')
    args = parser.parse_args(argv)
    root, work = args.root.resolve(), args.output.resolve()
    continuation = args.shared_continuation.resolve() if args.shared_continuation is not None else None
    shared = continuation or root/'shared_policy'
    actor_source = {'continuation': continuation} if continuation is not None else {}
    process = None
    if args.distillation_pid is not None:
        try:
            process = psutil.Process(args.distillation_pid)
            if not own_distillation(process.cmdline(), shared):
                parser.error('PID does not match the selected distillation run')
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            process = None
    with run_lock(work.with_name(work.name + '_handoff')):
        manifest = work/'manifest.json'
        frozen = read(manifest)['conditions_sha256'] if manifest.is_file() else None
        conditions_path = args.conditions or (official_conditions_path(frozen) if frozen else PHYSICS_CONDITIONS)
        conditions = read_conditions(conditions_path)
        conditions_sha256 = digest(conditions)
        official_conditions_path(conditions_sha256)
        if continuation is not None and conditions.get('reset_mode') != 'fixed':
            raise ValueError('A continuation actor requires formal single-seed conditions')
        if frozen is not None and conditions_sha256 != frozen:
            raise ValueError('Requested conditions differ from the frozen calibration')
        print('BO_HANDOFF_WAIT', shared, 'pid=', args.distillation_pid, flush=True)
        wait_for_distillation(shared, process)
        shared_input(root, formal=True, require_horizontal_selection=True, **actor_source)
        if not (work/'completed.json').is_file():
            if (work/'measurements_completed.json').is_file():
                with run_lock(work):
                    freeze_scoring(work, resume_paused=args.resume_paused)
            else:
                command = [sys.executable, '-u', str(HERE/'bo.py'), 'initialize', '--root', str(root),
                           '--output', str(work), '--gpu', str(args.gpu), '--updates', str(DEFAULT_BO_UPDATES),
                           '--conditions', str(conditions_path.resolve())]
                if continuation is not None:
                    command += ['--shared-continuation', str(continuation)]
                print('BO_HANDOFF_START', 'updates=', DEFAULT_BO_UPDATES, work, flush=True)
                initialize(command, work, resume_paused=args.resume_paused)
        summarize(root, work, DEFAULT_BO_UPDATES, conditions_path=conditions_path, **actor_source)


if __name__ == '__main__':
    main()
