"""Continue a live three-task distillation into formal BO calibration."""
import argparse
from math import isclose
import os
from pathlib import Path
from statistics import fmean
import signal
import subprocess
import sys
import time

import psutil

from bo import (HERE, REPO, FORMAL_CONDITIONS, TASKS, calibration_keys, check_panel_receipt,
                checked_report, completed_actor, digest, freeze, panel_scoring, paired_trials,
                read, read_conditions, run_lock, sha, shared_input)


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


def summarize(root, work, updates):
    source, _, initial = shared_input(root, formal=True, require_horizontal_selection=True)
    conditions = read_conditions(FORMAL_CONDITIONS)
    scoring_path = work/'scoring.json'
    check_panel_receipt(scoring_path)
    scoring = read(scoring_path)
    keys = calibration_keys()
    benchmark = scoring['benchmark']
    if (benchmark['updates'] != updates or benchmark['initial_actor_sha256'] != sha(initial)
            or benchmark['source_plan_sha256'] != sha(source) or len(benchmark['reports']) != len(keys)
            or scoring['conditions_sha256'] != digest(conditions)):
        raise ValueError('Calibration receipt has a different budget, policy, plan or conditions')
    tasks = {task: [] for task in TASKS}
    reports = []
    for index, entry in enumerate(benchmark['reports'], 1):
        folder = work/f'design_{index:02d}'
        path = (work/entry['report']).resolve()
        if path != (folder/'bo_evaluation.json').resolve() or sha(path) != entry['sha256']:
            raise ValueError(f'Calibration report identity differs at structure {index}')
        report = checked_report(path)
        actor = completed_actor(folder/'tuning', source, initial, report['candidate'], updates)
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
            if native['actor_sha256'] != report['student_sha256'] or sha(native['trace']) != native['trace_sha256']:
                raise ValueError(f'Measurement actor or raw trace changed at structure {index}')
            measurements.append(native)
        if paired_trials(measurements, conditions) != report['pairs']:
            raise ValueError(f'Measurement pairing differs at structure {index}')
        reports.append(report)
        for pair in report['pairs']:
            for task, trial in pair['tasks'].items():
                tasks[task].append(trial)
    if panel_scoring(reports, keys, benchmark) != scoring:
        raise ValueError('Frozen scoring differs from recomputed quantiles')
    task_summary = {}
    for task, trials in tasks.items():
        speeds = [r['mean_forward_mps'] for r in trials if r['mean_forward_mps'] is not None]
        task_summary[task] = dict(trials=len(trials), successes=sum(r['status']=='success' for r in trials),
                                  mean_horizontal_mps=fmean(speeds) if speeds else None,
                                  command_mps=conditions['tasks'][task]['speed_mps'])
    coefficients = {}
    for task, metrics in scoring['metrics'].items():
        coefficients[task] = {}
        for metric, row in metrics.items():
            scale = row['p90'] - row['p10']
            if not isclose(scale, row['scale'], rel_tol=1e-12) or scale <= 0.:
                raise ValueError(f'Invalid calibration scale for {task}/{metric}')
            coefficients[task][metric] = dict(**row, multiplier=1/scale, offset=-row['p10']/scale)
    summary = dict(structures=len(reports), ppo_updates=updates, initial_actor_sha256=sha(initial),
                   scoring_sha256=sha(scoring_path), trials=sum(len(v) for v in tasks.values()),
                   paired_successes=sum(pair['feasible'] is True for r in reports for pair in r['pairs']),
                   tasks=task_summary, normalization=coefficients)
    path = work/'calibration_summary.json'
    freeze(path, summary)
    print('BO_CALIBRATION_VERIFIED', path, flush=True)


def main(argv=None):
    from bo import DEFAULT_BO_UPDATES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--distillation-pid', type=int)
    parser.add_argument('--resume-paused', action='store_true')
    args = parser.parse_args(argv)
    root, work = args.root.resolve(), args.output.resolve()
    shared = root/'shared_policy'
    process = None
    if args.distillation_pid is not None:
        try:
            process = psutil.Process(args.distillation_pid)
            if not own_distillation(process.cmdline(), shared):
                parser.error('PID does not match the selected distillation run')
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            process = None
    with run_lock(work.with_name(work.name + '_handoff')):
        print('BO_HANDOFF_WAIT', shared, 'pid=', args.distillation_pid, flush=True)
        wait_for_distillation(shared, process)
        shared_input(root, formal=True, require_horizontal_selection=True)
        if not (work/'completed.json').is_file():
            command = [sys.executable, '-u', str(HERE/'bo.py'), 'initialize', '--root', str(root),
                       '--output', str(work), '--gpu', str(args.gpu), '--updates', str(DEFAULT_BO_UPDATES)]
            print('BO_HANDOFF_START', 'updates=', DEFAULT_BO_UPDATES, work, flush=True)
            initialize(command, work, resume_paused=args.resume_paused)
        summarize(root, work, DEFAULT_BO_UPDATES)


if __name__ == '__main__':
    main()
