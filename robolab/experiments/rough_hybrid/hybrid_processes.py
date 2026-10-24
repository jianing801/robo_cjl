"""Race-safe process snapshots for shared-server duplicate-bank checks.

Never request cwd eagerly through process_iter: a process may disappear while
psutil constructs the next item, outside the caller's exception handler.
"""
from pathlib import Path
from types import SimpleNamespace


def bank_process_snapshots():
    import psutil
    for pid in psutil.pids():
        try:
            process = psutil.Process(pid)
            command = process.cmdline()
            if not any(Path(x).name == 'train_spatial_rough.py' for x in command):
                continue
            banks = [token.split('=', 1)[1] if token.startswith('--bank=') else command[i+1]
                     for i, token in enumerate(command)
                     if token.startswith('--bank=') or (token == '--bank' and i+1 < len(command))]
            cwd = None
            if any(not Path(value).is_absolute() for value in banks):
                try:
                    cwd = process.cwd()
                except (psutil.AccessDenied, PermissionError) as error:
                    raise RuntimeError(f'Cannot resolve relative Rough bank for PID {pid}; inspect before starting') from error
                except FileNotFoundError as error:
                    if process.is_running():
                        raise RuntimeError(f'Active Rough PID {pid} has unavailable cwd; inspect before starting') from error
                    continue
            yield SimpleNamespace(pid=pid, info={'cmdline':command, 'cwd':cwd})
        except (psutil.NoSuchProcess, psutil.ZombieProcess, FileNotFoundError, ProcessLookupError):
            continue
        except (psutil.AccessDenied, PermissionError):
            # No access to an unrelated process's command line is not grounds
            # for killing it or requiring sudo on a shared server.
            continue


def reject_duplicate_bank_safely(bank):
    from robolab.rough_gpu import reject_duplicate_bank
    reject_duplicate_bank(bank, processes=bank_process_snapshots())
