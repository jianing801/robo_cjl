"""Scene-boundary diagnostics; never change live learning tensors or physics."""
import gc
import json
import os
import subprocess
import threading
import time


def snapshot(path, phase, torch):
    row = dict(phase=phase, pid=os.getpid(), time=time.time())
    try:
        if torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            row.update(device=torch.cuda.current_device(), free=free, total=total,
                       allocated=torch.cuda.memory_allocated(),
                       reserved=torch.cuda.memory_reserved())
    except Exception as error:
        row['cuda_query_error'] = str(error)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=5, check=True)
        row['all_gpu_memory_csv'] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        row['gpu_query_error'] = str(error)
    try:
        with path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row) + '\n')
    except OSError as error:
        print('SCENE_MEMORY_LOG_ERROR', str(error), flush=True)
    return row


def prepare_scene(path, torch):
    snapshot(path, 'before_unused_cache_release', torch)
    # Previous stage_main has returned: its environment locals can now die.
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        # Only unused allocator blocks. Policy, Adam and rollout stay alive.
        torch.cuda.empty_cache()
    return snapshot(path, 'before_scene_creation', torch)


def arm_failed_cleanup_timeout(seconds=120, exit_fn=None):
    """Bound cleanup after an authoritative failure was already written/flushed."""
    timer = threading.Timer(seconds, exit_fn or (lambda: os._exit(1)))
    timer.daemon = True
    timer.start()
    return timer


def watch_failed_worker(worker, folder, seconds=120, interval=1):
    """Parent-side watchdog: no timeout for healthy training or quiet logs."""
    stop = threading.Event()
    def watch():
        first_failure = None
        while not stop.wait(interval):
            if worker.poll() is not None:
                return
            if not any((folder/'result'/name).is_file()
                       for name in ('failure.json', 'cleanup_failure.json')):
                continue
            if first_failure is None:
                first_failure = time.monotonic()
            if time.monotonic() - first_failure < seconds:
                continue
            # Only this Popen child's tree, never other GPU users or a stale PID.
            import psutil
            try:
                process = psutil.Process(worker.pid)
                children = process.children(recursive=True)
                process.terminate()
                for child in children:
                    try: child.terminate()
                    except psutil.NoSuchProcess: pass
                _, alive = psutil.wait_procs([process, *children], timeout=5)
                for process in alive:
                    try: process.kill()
                    except psutil.NoSuchProcess: pass
            except psutil.NoSuchProcess:
                pass
            return
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return stop, thread
