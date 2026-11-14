"""Prepare missing AMP references before acquiring a training GPU or creating a run."""
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def generation_command(bank_path, output, *, python=None, env_name='amp_hhtools', source=None,
                       device='cpu', warp_cache=None):
    if python:
        command = [str(python)]
    else:
        command = [os.environ.get('CONDA_EXE') or 'conda', 'run', '--no-capture-output',
                   '-n', env_name, 'python']
    command += [str(HERE / 'prepare_references.py'), '--bank', str(bank_path),
                      '--output', str(output), '--hhtools-root',
                      str(Path(source).resolve() if source else HERE / 'vendor' / 'hhtools'),
                      '--device', device]
    if warp_cache:
        command += ['--warp-cache',str(Path(warp_cache).resolve())]
    return command


@contextmanager
def reference_runtime(device, requested, min_free_gb, max_utilization):
    if device == 'cpu':
        yield 'cpu', dict(os.environ)
        return
    if device != 'cuda':
        raise ValueError('Reference mode must be cpu or cuda')
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    choices = candidates(query_gpus(), os.environ, min_free_mib=min_free_gb*1024,
                         max_utilization=max_utilization, requested=requested)
    with gpu_lease(choices, HERE.parents[1]/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
        print('AMP_REFERENCE_GPU_SELECTED',gpu['index'],gpu['uuid'],flush=True)
        yield 'cuda:0', child_environment(gpu,os.environ)


@contextmanager
def reference_lock(parent):
    # Hold an OS lock across generation and publication; a stale file is harmless.
    with (parent / 'reference_prepare.lock').open('a+b') as stream:
        stream.seek(0)
        stream.write(b'0')
        stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def ensure_library(bank, bank_path, library_path, *, execute, resume=False,
                   python=None, env_name='amp_hhtools', source=None, device='cpu',
                   requested_gpu=None, min_free_gb=12., max_utilization=30, warp_cache=None):
    from amp_references import validate_library
    library_path = Path(library_path).resolve()
    if library_path.exists():
        library = validate_library(bank, bank_path, library_path)
        if library.get('validation_only'):
            raise ValueError('validation-only library cannot train')
        print('AMP_REFERENCES_REUSED', library_path, flush=True)
        return library
    if resume:
        raise FileNotFoundError('Resume requires the original reference library; no automatic regeneration')
    if not execute:
        print('AMP_REFERENCES_REQUIRED_BEFORE_TRAINING', library_path, flush=True)
        print('With --execute, references will be generated in the isolated HHTools environment.', flush=True)
        return None
    library_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_lock(library_path.parent):
        # A concurrent launch may have completed before this lock was acquired.
        if library_path.exists():
            return ensure_library(bank, bank_path, library_path, execute=False)
        generated = library_path.parent / f'generated_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}'
        print('AMP_REFERENCE_GENERATION_BEGIN', generated, flush=True)
        with reference_runtime(device,requested_gpu,min_free_gb,max_utilization) as (actual_device,environment):
            command = generation_command(bank_path, generated, python=python, env_name=env_name,
                                         source=source,device=actual_device,warp_cache=warp_cache)
            try:
                subprocess.run(command, check=True, cwd=HERE.parents[1],env=environment)
            except (subprocess.CalledProcessError, FileNotFoundError) as error:
                raise RuntimeError(f'AMP reference preparation failed; training not started. '
                                   f'Check the independent HHTools environment and retained output: {generated}') from error
        candidate = generated / 'library.json'
        library = validate_library(bank, bank_path, candidate)
        if library.get('validation_only'):
            raise ValueError('generated validation-only library cannot train')
        if device == 'cuda' and not library.get('reference_execution',{}).get('cuda_graph_required'):
            raise ValueError('Generated references do not confirm the required CUDA Graph mode')
        # References contain absolute generated paths. Keep this directory in place;
        # publish only its validated manifest, never rename the referenced files.
        with library_path.open('x', encoding='utf8') as stream:
            stream.write(candidate.read_text(encoding='utf8'))
        print('AMP_REFERENCES_READY_FOR_TRAINING', library_path, flush=True)
        return library
