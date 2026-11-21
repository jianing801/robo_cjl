"""跨形态多任务共享策略流程：自动寻找结构库、识别续训并复用参考动作。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from task_timer import atomic_json, start_task_timer

ENTRY_STARTED_WALL=time.time()
ENTRY_STARTED_MONOTONIC=time.monotonic()

REPO = Path(__file__).resolve().parent
ENTRY = REPO / 'robolab/experiments/multitask_morphology'
PREVIOUS_PHYSICAL_ROOT = REPO / 'robolab/logs/multitask_full2500_balanced_v1'
PREVIOUS_FULL_ROOT = REPO / 'robolab/logs/multitask_full3000_balanced_v2'
DEFAULT_ROOT = REPO / 'robolab/logs/multitask_full2500_physical'
MIGRATED_ROOT = REPO / 'robolab/logs/multitask_full2500_physical_migrated'
ALIASES = {'准备': 'prepare', '参考动作': 'references', '负重': 'backpack',
           '蒸馏': 'distill', '微调': 'finetune', '状态': 'status', '录像': 'video',
           '资产': 'assets', '迁移': 'migrate'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(command, *, env=None, preview=False):
    if preview:
        print('预览，不执行：', subprocess.list2cmdline(list(map(str, command))), flush=True)
        return
    subprocess.run(list(map(str, command)), cwd=REPO, env=env, check=True)


def bank_path(root, task):
    return root / 'banks' / task / 'bank.json'


def ensure_banks(root, preview=False):
    # 完整库直接复用；部分生成的目录绝不覆盖或擅自删除。
    tasks = {'rough': 'stairs', 'backpack': 'backpack', 'amp': 'run'}
    paths = {task: bank_path(root, task) for task in tasks}
    if all(p.is_file() for p in paths.values()):
        for task, path in paths.items():
            bank = read(path)
            settings = read(ENTRY/'configs'/f'{task}.json')
            if (bank.get('task_key') != tasks[task] or bank.get('configuration') != settings['grid']
                    or bank.get('context_schema') != settings.get('context_schema')):
                raise ValueError(f'结构库与当前 {task} 电机池或物理输入不匹配：{path}。请使用新的实验目录。')
        print('已找到三个任务的结构库，直接复用。', flush=True)
        return
    if root.exists():
        raise RuntimeError(f'结构库不完整，保留现场并停止：{root}。请查看“异常处理与续训”。')
    assets = REPO / 'robolab/data/robots/roboparty/rpo'
    template = assets / 'urdf/rpo.urdf'
    meshes = assets / 'meshes'
    if not template.is_file() or not meshes.is_dir():
        raise FileNotFoundError('缺少原始机器人资产，请先完成“环境与资产准备”。')
    run([sys.executable, '-u', ENTRY/'prepare.py', '--output-root', root,
         '--template', template, '--mesh-dir', meshes, '--generate-assets'], preview=preview)


def select_run(root, task):
    """兼容之前手动启动的目录；优先使用本入口记下的任务目录。"""
    bank = bank_path(root, task).resolve()
    state = root / '启动记录' / f'{task}.json'
    if state.is_file():
        selected = Path(read(state)['output']).resolve()
        if not selected.is_relative_to((root/'experts').resolve()):
            raise ValueError('启动记录中的目录不属于本实验。')
        if (selected/'plan.json').is_file():
            if Path(read(selected/'plan.json')['bank']).resolve() != bank:
                raise ValueError('启动记录绑定了不同结构库。')
            return selected, True
    candidates = []
    for folder in (root/'experts').glob(task+'*'):
        if folder.name != task and not folder.name.startswith(task+'_'):
            continue
        plan = folder/'plan.json'
        if plan.is_file() and Path(read(plan)['bank']).resolve() == bank:
            candidates.append(folder)
    if candidates:
        selected = max(candidates, key=lambda p: (p/'plan.json').stat().st_mtime_ns)
        return selected, True
    selected = root/'experts'/task
    if selected.exists():
        if not selected.is_dir() or any(p.is_file() or p.is_symlink() for p in selected.rglob('*')):
            raise RuntimeError(f'目录有内容但缺少训练计划，保留并停止：{selected}')
        # 空壳目录保留，给正式运行分配新目录；之后由启动记录记住。
        selected = selected.with_name(task+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    return selected, False


def prepared_cache(root, task):
    index = root/'启动记录'/f'{task}_asset_cache.json'
    if not index.is_file():
        return None
    value = read(index)
    if value['bank_sha256'] != hashlib.sha256(bank_path(root,task).read_bytes()).hexdigest():
        raise ValueError('预转换缓存绑定了不同结构库。')
    proof = Path(value['cache'])/'completed.json'
    if not proof.is_file() or hashlib.sha256(proof.read_bytes()).hexdigest() != value['completed_sha256']:
        raise ValueError('预转换缓存完成证明缺失或变化，不能直接用于训练。')
    return Path(value['cache'])


def register_asset_cache(root, task, cache):
    from hybrid_native_cache import sha
    folder = root/'启动记录'
    folder.mkdir(exist_ok=True)
    atomic_json(folder/f'{task}_asset_cache.json',
                dict(cache=str(cache.resolve()), bank_sha256=sha(bank_path(root,task)),
                     completed_sha256=sha(cache/'completed.json')))


def expert_command(root, task, gpu, output, resume):
    command = [sys.executable, '-u', str(ENTRY/'train.py'), '--task', task,
               '--bank', str(bank_path(root, task)), '--output-root', str(output),
               '--device', f'cuda:{gpu}', '--gpu-min-free-gb', '12.5',
               '--gpu-max-utilization', '100', '--execute']
    if resume:
        command.append('--resume')
    if task == 'amp':
        library = root/'references/library.json'
        if resume and (output/'plan.json').is_file():
            library = Path(read(output/'plan.json')['reference_library'])
        command += ['--reference-library', str(library),
                    '--hhtools-env', 'amp_hhtools', '--reference-device', 'cuda']
    cached = prepared_cache(root, task)
    if cached is not None:
        command += ['--asset-cache', str(cached)]
    return command


def record_run(root, task, output):
    folder = root/'启动记录'; folder.mkdir(exist_ok=True)
    value = dict(output=str(output.resolve()), bank=str(bank_path(root, task).resolve()))
    tmp = folder/f'{task}.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')
    os.replace(tmp, folder/f'{task}.json')


def reject_active_expert(output, bank):
    """只核对同一训练目录/结构库，恢复不能制造重复任务。"""
    import psutil
    own={os.getpid(),*[p.pid for p in psutil.Process().parents()]}
    scripts={'start.py','train.py','train_stage.py','persistent_worker.py','amp_persistent_worker.py','prepare_native_cache.py'}
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid in own:continue
        args=process.info['cmdline'] or []
        if not any(Path(arg).name in scripts for arg in args):continue
        if any(arg==str(output) or arg.startswith(str(output)+'/') or arg==str(bank) for arg in args):
            raise RuntimeError(f'该结构库或训练目录仍有进程{process.pid}，不重复启动。')


def has_initial_checkpoint(output):
    """排除纯USD树；其他位置的模型都保留，不因尚未提交就丢弃。"""
    for directory,dirs,files in os.walk(output,followlinks=False):
        here=Path(directory)
        if here==output:dirs[:]=[d for d in dirs if not d.startswith('native_cache_')]
        if here==output/'attempts':dirs[:]=[d for d in dirs if not d.startswith('cache_')]
        if any(Path(name).suffix.lower() in {'.pt','.pth','.ckpt'} for name in files):return True
    return False


def resume_expert_pause(root, task, output, preview=False):
    marker=output/'USER_PAUSED'
    if not marker.exists():return
    bank=bank_path(root,task).resolve();reject_active_expert(output,bank)
    sys.path.insert(0,str(ENTRY/'engine'))
    from hybrid_native_cache import training_attempt_started
    commits=list((output/'commits').glob('stage_*.json'))
    if training_attempt_started(output) and not commits:
        raise RuntimeError('首组已中断但没有已提交组断点；组内模型保留在原目录，明确从首组重新训练请加--restart-initial。')
    if preview:
        print('预览：明确恢复该专家暂停，不删除模型。',flush=True);return
    folder=root/'启动记录';folder.mkdir(exist_ok=True)
    audit=folder/f'{task}_training_resume_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    before=marker.read_bytes()
    audit.write_text(json.dumps(dict(explicit_training_resume=True,output=str(output),marker_sha256=hashlib.sha256(before).hexdigest()),ensure_ascii=False),encoding='utf8')
    if marker.read_bytes()!=before:raise RuntimeError('暂停标记并发变化，未解除。')
    marker.unlink()
    print('已按明确恢复选项解除当前专家暂停。',flush=True)


def prepare_initial_restart(root, task, output, preview=False):
    """明确授权的首组重启：新目录、原bank/配置/USD，旧目录和暂停标记保留。"""
    bank=bank_path(root,task).resolve();reject_active_expert(output,bank)
    plan=read(output/'plan.json')
    if (output/'completed.json').exists() or any((output/'commits').glob('*.json')):
        raise RuntimeError('已有训练提交或完成记录，请使用--resume-paused续训。')
    if plan.get('bootstrap_checkpoint') or plan.get('start_stage',0):raise RuntimeError('该目录是检查点续接链，不能当作初始训练重开。')
    if plan.get('configuration')!=read(ENTRY/'configs'/f'{task}.json'):
        raise RuntimeError('旧配置与当前正式配置不一致，不隐式改变预算或训练设置。')
    # 避免旧索引漏掉后来手动启动且已经产生结果的同库实验。
    for other in (root/'experts').glob(task+'*'):
        if other==output or not (other/'plan.json').is_file():continue
        if Path(read(other/'plan.json')['bank']).resolve()==bank and (any((other/'commits').glob('stage_*.json')) or has_initial_checkpoint(other)):
            raise RuntimeError('同结构库另有已训练实验，请先核对启动索引，不自动重开。')
    record=root/'启动记录'/f'{task}_asset_cache.json'
    candidates=[Path(read(record)['cache'])] if record.is_file() else [p.parent for p in sorted(output.glob('native_cache_*/completed.json'),reverse=True)]
    bank_value=read(bank);bank_sha=hashlib.sha256(bank.read_bytes()).hexdigest();ids={d['id'] for d in bank_value['designs']};cache_record=None
    for cache in candidates:
        proof_path=cache/'completed.json'
        if not proof_path.is_file():continue
        data=read(proof_path);rows=data.get('assets',[])
        if (data.get('result')=='NATIVE_ASSET_CACHE_COMPLETE' and data.get('cache_schema_version')==2
                and data.get('bank_sha256')==bank_sha and len(rows)==len(ids) and {r['design'] for r in rows}==ids):
            cache_record=dict(cache=str(cache.resolve()),bank_sha256=bank_sha,completed_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest());break
    if cache_record is None:raise RuntimeError('没有同bank的完整USD完成证明；不在重启时隐式重新转换。')
    tag=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    selected=root/'experts'/f'{task}_restart_{tag}'
    if selected.exists():raise RuntimeError('重启目录冲突，未覆盖。')
    if not preview:
        folder=root/'启动记录';folder.mkdir(exist_ok=True)
        audit=folder/f'{task}_initial_restart_{tag}';audit.mkdir()
        for name in [f'{task}.json',f'{task}_asset_cache.json']:
            path=folder/name
            if path.is_file():(audit/name).write_bytes(path.read_bytes())
        (audit/'plan.before.json').write_bytes((output/'plan.json').read_bytes())
        (audit/'authorization.json').write_text(json.dumps(dict(explicit_restart_initial=True,previous_output=str(output),new_output=str(selected),bank=str(bank),cache=cache_record['cache'],configuration=plan['configuration'],old_pause_preserved=(output/'USER_PAUSED').exists()),ensure_ascii=False,indent=2),encoding='utf8')
        temporary=folder/f'{task}_asset_cache.{tag}.tmp';temporary.write_text(json.dumps(cache_record,ensure_ascii=False),encoding='utf8');os.replace(temporary,record)
    print('明确从第0代重新启动首组，保留原目录、组内模型和旧暂停记录。',flush=True)
    return selected,plan.get('reference_library'),cache_record['cache']


def launch_expert(root, task, gpu, preview=False, resume_paused=False, restart_initial=False):
    output, resume = select_run(root, task)
    reference_override=None;cache_override=None
    if restart_initial:
        if not resume:raise RuntimeError('尚无旧训练计划，不需要--restart-initial，直接启动即可。')
        output,reference_override,cache_override=prepare_initial_restart(root,task,output,preview)
        resume=False
    elif resume_paused:
        resume_expert_pause(root,task,output,preview)
    print(('续训目录：' if resume else '新训练目录：')+str(output), flush=True)
    if (output/'USER_PAUSED').exists() and not (preview and resume_paused):
        raise RuntimeError('该任务有用户暂停标记；续训加--resume-paused，首组无断点重开加--restart-initial。')
    if not preview:
        target=(read(output/'plan.json')['configuration'] if resume else read(ENTRY/'configs'/f'{task}.json'))['rotation']['total_updates']
        start_task_timer(root,task,output,target,ENTRY_STARTED_WALL,ENTRY_STARTED_MONOTONIC)
    if resume and task != 'amp':
        # 仅转换阶段允许通过已有审计工具迁移；PPO开始后严格拒绝改身份。
        current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ENTRY/'engine').glob('*.py') if not p.name.startswith('test_')}
        recorded=read(output/'plan.json')['experiment']
        if recorded['hybrid_sources'] != current:
            sys.path.insert(0,str(ENTRY/'engine'))
            from hybrid_contract import controller_compatible
            compatible=controller_compatible(recorded,dict(recorded,hybrid_sources=current),output)
            if not compatible:
                run([sys.executable, ENTRY/'engine/refresh_cache_plan.py', '--root', output], preview=preview)
    command = expert_command(root, task, gpu, output, resume)
    if cache_override is not None and '--asset-cache' not in command:
        command+=['--asset-cache',cache_override]
    if reference_override is not None and task=='amp':
        command[command.index('--reference-library')+1]=reference_override
    if preview:
        run(command, preview=True)
        return
    record_run(root, task, output)
    # Linux直接替换当前进程，避免额外父进程干扰中断和任务归属。
    if sys.platform != 'win32':
        os.chdir(REPO)
        os.execv(sys.executable, command)
    run(command)


def references(root, gpu, cpu, preview=False):
    library = root/'references/library.json'
    if library.is_file():
        print('参考动作库已存在；AMP启动时会严格校验复用，无需重新生成。', flush=True)
        return
    if library.parent.exists():
        raise RuntimeError(f'参考动作目录部分存在但缺少完整库，保留并停止：{library.parent}')
    command = ['conda', 'run', '--no-capture-output', '-n', 'amp_hhtools', 'python', '-u',
               ENTRY/'amp_engine/prepare_references.py', '--bank', bank_path(root, 'amp'),
               '--output', library.parent, '--hhtools-root', ENTRY/'amp_engine/vendor/hhtools',
               '--device', 'cpu' if cpu else 'cuda:0']
    env = os.environ.copy()
    if not cpu:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    run(command, env=env, preview=preview)


def cache_worker_status(pid):
    import psutil
    try:
        state=psutil.Process(pid).status()
    except psutil.Error:
        return '已退出'
    # Linux的/proc链接只用于诊断本次子进程，不读取资产内容或其他用户数据。
    if sys.platform.startswith('linux'):
        try:
            for path in sorted(Path(f'/proc/{pid}/fd').iterdir(),key=lambda p:int(p.name),reverse=True):
                target=os.readlink(path)
                if target.lower().endswith(('.usd','.urdf','.stl','.obj','.pkl')):
                    return state+'，打开文件：'+target
        except OSError:
            pass
    return state


def run_cache_child(command, folder, env):
    """资产专用监督：心跳不依赖子进程输出，中断及非零退出也保存退出记录。"""
    import psutil
    folder.mkdir(exist_ok=False)
    started=time.monotonic();worker=None;pump=None;problem=None;code=None
    last=['尚无子进程输出'];logging_errors=[]
    with (folder/'stdout.log').open('x',encoding='utf8') as out,(folder/'stderr.log').open('x',encoding='utf8') as err:
        try:
            kwargs=dict(cwd=REPO/'robolab',env=dict(env,PYTHONIOENCODING='utf-8'),
                        stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf8',errors='replace',bufsize=1)
            if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
            else:kwargs['start_new_session']=True
            worker=subprocess.Popen(command,**kwargs)
            try:created=psutil.Process(worker.pid).create_time()
            except psutil.NoSuchProcess:created=None
            (folder/'launch.json').write_text(json.dumps(dict(pid=worker.pid,create_time=created,command=command)),encoding='utf8')
            print('CACHE_CHILD_START',worker.pid,folder,flush=True)
            def copy_output():
                try:
                    for line in worker.stdout:
                        last[0]=line.rstrip();print(line,end='',flush=True)
                        out.write(line);out.flush()
                except BaseException as error:
                    logging_errors.append(error)
            pump=threading.Thread(target=copy_output,daemon=True);pump.start()
            heartbeat=time.monotonic()
            while worker.poll() is None:
                if logging_errors:raise RuntimeError('缓存日志转发失败：'+str(logging_errors[0]))
                if time.monotonic()-heartbeat>=10:
                    print(f'CACHE_HEARTBEAT pid={worker.pid} elapsed={time.monotonic()-started:.0f}s '
                          f'{cache_worker_status(worker.pid).split(chr(65292)+"打开文件",1)[0]}；阶段/进度：{last[0]}',flush=True)
                    heartbeat=time.monotonic()
                time.sleep(.2)
            code=worker.wait()
            pump.join(timeout=5)
            if logging_errors:raise RuntimeError('缓存日志转发失败：'+str(logging_errors[0]))
        except BaseException as error:
            problem=error
            if isinstance(error,KeyboardInterrupt):
                (folder.parents[2]/'USER_PAUSED').touch(exist_ok=True)
            if worker is not None and worker.poll() is None:
                from native_cache_coordinator import stop_tree
                stop_tree(worker)
                code=worker.poll()
        finally:
            if pump is not None:pump.join(timeout=1)
            if worker is not None and code is None:code=worker.poll()
            failure=(folder/'result/failure.json').is_file()
            summary=(folder/'result/summary.json').is_file()
            effective=code if code else (1 if problem is not None or failure or not summary else 0)
            record=dict(returncode=effective,raw_returncode=code,summary_exists=summary,
                        failure_recorded=failure,interrupted=isinstance(problem,KeyboardInterrupt),
                        wall_seconds=round(time.monotonic()-started,3),
                        controller_error=None if problem is None else type(problem).__name__+': '+str(problem))
            print('CACHE_CHILD_EXIT',json.dumps(record,ensure_ascii=False),flush=True)
            (folder/'exit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    if problem is not None:raise problem
    if effective:
        try:
            failure_path=folder/'result/failure.json'
            if failure_path.is_file():
                print('缓存准备失败原因：\n'+str(read(failure_path).get('traceback','')),file=sys.stderr,flush=True)
            else:
                with (folder/'stderr.log').open('rb') as stream:
                    stream.seek(0,os.SEEK_END);stream.seek(max(0,stream.tell()-8192))
                    print(stream.read().decode('utf8',errors='replace'),file=sys.stderr,flush=True)
        except (OSError,ValueError) as reading_error:
            print('无法读取错误详情：'+str(reading_error),file=sys.stderr,flush=True)
        raise RuntimeError(f'USD准备失败，退出码{code}；完整错误见{folder}/stderr.log')


def prepare_task_assets(root, task, gpu, preview=False):
    """仅准备一个任务的USD；复用原控制器的锁、日志及缓存校验，不调用PPO。"""
    output, resume = select_run(root, task)
    # 已有训练计划的目录可以直接利用之前成功的小批次；新任务另建资产目录，
    # 不在专家输出目录制造“有缓存但无plan”的半成品。
    cache_root = output if resume else root/'资产准备'/task
    if (cache_root/'USER_PAUSED').exists():
        raise RuntimeError(f'{task}资产目录有用户暂停标记，不自动取消。')
    print(f'仅准备{task}的USD，不启动PPO。缓存目录：{cache_root}', flush=True)
    if preview:
        print(f'转换选卡：cuda:{gpu}；最低空闲显存4GiB，和训练12.5GiB门槛分开。')
        return
    sys.path[:0] = [str(ENTRY/'engine'), str(REPO/'robolab'), str(REPO/'rsl_rl')]
    from train import controller_lock
    from hybrid_native_cache import ensure_cache, sha, validate_cache
    from hybrid_processes import reject_duplicate_bank_safely
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    bank = bank_path(root,task)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root/'attempts').mkdir(exist_ok=True)
    with controller_lock(cache_root):
        reject_duplicate_bank_safely(bank)
        reject_asset_workers(bank)
        # 已完成缓存只需CPU校验，不先抢占GPU，也不要求当前有空闲卡。
        cache=None
        try:
            indexed = prepared_cache(root, task)
        except (ValueError, OSError, KeyError) as error:
            print('USD_EXISTING_CACHE_REJECTED', str(error), flush=True)
            indexed = None
        candidates_to_check = [indexed] if indexed is not None else []
        candidates_to_check.extend(proof.parent for proof in sorted(cache_root.glob('native_cache_*/completed.json'),reverse=True)
                                   if proof.parent != indexed)
        for candidate in candidates_to_check:
            print('USD_EXISTING_CACHE_CHECK_BEFORE_GPU',task,str(candidate),flush=True)
            try:validate_cache(candidate,bank,resume_fast=True)
            except (ValueError,OSError,KeyError) as error:
                print('USD_EXISTING_CACHE_REJECTED',str(error),flush=True)
                continue
            cache=candidate
            break
        if cache is None:
            # 这是转换的启动门槛，不是显存峰值保证；OOM保留日志并停止。
            choices = candidates(query_gpus(), os.environ, min_free_mib=4*1024,
                                 max_utilization=100, requested=f'cuda:{gpu}')
            with gpu_lease(choices, REPO/'robolab/logs/rpo_rough_multimorph/gpu_locks') as selected:
                environment = child_environment(selected, os.environ)
                kit = (f" --/renderer/activeGpu={selected['index']}"
                       " --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0")
                cache = ensure_cache(cache_root, bank, sys.executable, kit, environment, run_cache_child)
        register_asset_cache(root, task, cache)
    print('USD_ASSETS_READY',task,cache,flush=True)


def reject_asset_workers(bank):
    """拒绝同库残留转换器；不结束其他进程或删除锁文件。"""
    import psutil
    own = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid in own:
            continue
        args = process.info['cmdline'] or []
        if not any(Path(arg).name == 'prepare_native_cache.py' for arg in args):
            continue
        for i, arg in enumerate(args):
            value = arg.split('=',1)[1] if arg.startswith('--bank=') else args[i+1] if arg=='--bank' and i+1<len(args) else None
            if value is None:
                continue
            path = Path(value)
            if not path.is_absolute():
                try:
                    path = Path(process.cwd())/path
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            if path.resolve() == Path(bank).resolve():
                raise RuntimeError(f'同一结构库已有转换进程{process.pid}，不重复启动。')


def resume_asset_pause(root, task, preview=False):
    """仅在用户明确传入恢复选项时清除预训练转换暂停，保留审计。"""
    output, resume=select_run(root,task)
    cache_root=output if resume else root/'资产准备'/task
    marker=cache_root/'USER_PAUSED'
    if not marker.exists():return
    sys.path.insert(0,str(ENTRY/'engine'))
    from hybrid_native_cache import training_attempt_started
    if training_attempt_started(cache_root) or any((cache_root/'commits').glob('*.json')):
        raise RuntimeError('该暂停目录已有PPO记录；资产恢复选项不能取消训练暂停。')
    if preview:
        print('预览：将明确恢复转换暂停',marker)
        return
    folder=root/'启动记录';folder.mkdir(exist_ok=True)
    record=folder/f'{task}_asset_resume_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    with record.open('x',encoding='utf8') as stream:
        json.dump(dict(explicit_asset_resume=True,marker=str(marker),
                       marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest()),stream)
    marker.unlink()
    print('已按本次明确指令恢复USD准备，不启动PPO。',flush=True)


def prepare_assets(root, task, gpu, preview=False, resume_paused=False):
    tasks = ('rough','backpack','amp') if task == '全部' else (ALIASES.get(task,task),)
    # 顺序执行，一次只启动一个任务的转换器；任一失败立即停止。
    for current in tasks:
        if resume_paused:resume_asset_pause(root,current,preview)
        prepare_task_assets(root,current,gpu,preview)
    print('USD_ASSETS_ALL_SELECTED_READY' if not preview else 'USD_ASSETS_PREVIEW_ONLY',flush=True)


def migrate_assets(root, source, gpu, preview=False, resume_paused=False):
    """Build a standalone 5+5 snapshot and complete only missing native assets."""
    completed = root/'migration_completed.json'
    if completed.is_file():
        print('MIGRATION_READY', root, flush=True)
        return
    if source is None:
        source = next((p for p in (PREVIOUS_FULL_ROOT, PREVIOUS_PHYSICAL_ROOT)
                      if all(bank_path(p,t).is_file() for t in ('rough','backpack','amp'))), PREVIOUS_FULL_ROOT)
    source = source.resolve()
    if root == DEFAULT_ROOT or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('迁移源和目标必须是独立目录。全新生成目录保留给“准备”和“资产”。')
    state_path = root/'migration_in_progress.json'
    tasks = ('rough','backpack','amp')
    sys.path.insert(0,str(ENTRY/'engine'))
    from hybrid_native_cache import sha
    identity = dict(source=str(source), source_banks={t:sha(bank_path(source,t)) for t in tasks})
    bank_command = [sys.executable,'-u',ENTRY/'prepare.py','--output-root',root,
                    '--reuse-banks-from',source,'--generate-assets']
    worker = ENTRY/'engine/portable_usd.py'
    from portable_usd import runtime_environment
    environment = runtime_environment(os.environ)
    command = [sys.executable,'-u',worker]
    arguments = ['--root',root,'--source',source]
    if preview:
        if not (root/'bank_audit.json').is_file():
            run(bank_command,preview=True)
        for task in tasks:
            run([*command,'import',*arguments,'--task',task],preview=True)
        run([*command,'verify',*arguments],preview=True)
        return

    # Lock adjacent to the target so this cannot be mistaken for a prepared bank.
    sys.path.insert(0,str(ENTRY/'shared_policy'))
    from persistence import run_lock
    with run_lock(root):
        if completed.is_file():
            print('MIGRATION_READY',root,flush=True)
            return
        if state_path.is_file():
            state = read(state_path)
            if state['identity'] != identity:
                raise ValueError('迁移源路径或结构清单已变化。请恢复原源目录或另选迁移目标。')
            if state.get('paused') and not resume_paused:
                raise RuntimeError('迁移有暂停标记，继续时加 --resume-paused。')
        else:
            if root.exists():
                raise FileExistsError(f'迁移目标已存在，请另选独立目录：{root}')
            root.mkdir(parents=True,exist_ok=False)
        atomic_json(state_path,dict(identity=identity,paused=False))
        try:
            if not (root/'bank_audit.json').is_file():
                run(bank_command)
            for task in tasks:
                if resume_paused:
                    resume_asset_pause(root,task)
                if prepared_cache(root,task) is not None:
                    continue
                run([*command,'import',*arguments,'--task',task],env=environment)
                cache = root/'资产准备'/task/'native_cache_portable'
                if (cache/'completed.json').is_file():
                    register_asset_cache(root,task,cache)
                else:
                    print('MIGRATION_CONVERTING_MISSING',task,flush=True)
                    prepare_task_assets(root,task,gpu)
            run([*command,'verify',*arguments],env=environment)
            atomic_json(completed,dict(result='MIGRATION_READY',root=str(root),
                bank_sha256={task:sha(bank_path(root,task)) for task in tasks},
                amp_references_ready=(root/'references/library.json').is_file()))
            state_path.unlink()
        except KeyboardInterrupt:
            atomic_json(state_path,dict(identity=identity,paused=True))
            raise
    print('MIGRATION_READY',root,flush=True)


def distill(root, gpu, preview=False, collection_profile='robust', randomization_strength=1.,
            initial_randomization_strength=.25,
            selection_seeds='20261117,20261119,20261120', samples_per_design_per_epoch=32,
            validation_rows_per_design_shard=128, replay_cache_shards=12,
            tracking_rel_tolerance=.25, min_command_progress_ratio=.5, resume_paused=False):
    teachers = {t: select_run(root, t)[0] for t in ('rough', 'backpack', 'amp')}
    if not all((p/'completed.json').is_file() for p in teachers.values()):
        raise RuntimeError('请先完成三个专家；尚未完成的任务可用“状态”查看。')
    plan = root/'shared_plan/plan.json'
    if not plan.is_file():
        command = [sys.executable, ENTRY/'prepare_shared.py', '--output', plan.parent]
        for task, folder in teachers.items():
            command += [f'--{task}-root', folder]
        run(command, preview=preview)
    if plan.is_file():
        bound = read(plan)['teachers']
        for task, folder in teachers.items():
            native = {'rough':'stairs','backpack':'backpack','amp':'run'}[task]
            if Path(bound[native]['checkpoint']).resolve() != Path(read(folder/'completed.json')['checkpoint']).resolve():
                raise ValueError('共享计划绑定的教师与当前教师不一致；保留旧计划，不自动替换。')
    output = root/'shared_policy'
    command = [sys.executable, '-u', ENTRY/'distill.py', '--plan', plan, '--output', output,
               '--device', f'cuda:{gpu}', '--collect-steps', '250',
               '--stride', '5', '--epochs', '10', '--dagger-rounds', '3',
               '--collection-profile', collection_profile,
               '--randomization-strength', str(randomization_strength),
               '--initial-randomization-strength', str(initial_randomization_strength),
               '--selection-seeds', selection_seeds,
               '--samples-per-design-per-epoch', str(samples_per_design_per_epoch),
               '--validation-rows-per-design-shard', str(validation_rows_per_design_shard),
               '--replay-cache-shards', str(replay_cache_shards),
               '--tracking-rel-tolerance', str(tracking_rel_tolerance),
               '--min-command-progress-ratio', str(min_command_progress_ratio),
               '--timeout-seconds', '1800', '--gpu-min-free-gb', '12.5', '--gpu-max-utilization', '100']
    if (output/'manifest.json').is_file():
        command.append('--resume')
    if resume_paused:command.append('--resume-paused')
    if not preview and not (output/'completed.json').is_file() and (resume_paused or not (output/'USER_PAUSED').is_file()):
        start_task_timer(root,'distill',output,None)
    run(command, preview=preview)


def finetune(root, gpu, design_id, design_task='stairs', updates=100, preview=False,
             resume_paused=False, output=None, evaluation_profile='both', evaluation_seeds='20261118',
             tracking_rel_tolerance=.25, min_command_progress_ratio=.5):
    plan_path = root/'shared_plan/plan.json'
    complete_path = root/'shared_policy/completed.json'
    if not plan_path.is_file() or not complete_path.is_file():
        raise RuntimeError('请先完成共享策略蒸馏，再选择候选结构微调。')
    plan = read(plan_path); complete = read(complete_path)
    if design_task not in plan['teachers'] or design_id not in plan['teachers'][design_task]['design_ids']:
        raise ValueError('候选结构ID不在所选任务的共享结构库中。')
    records = read(plan['shared_bank'])['structures']
    matches = [row for row in records if row['design_ids'][design_task] == design_id]
    if len(matches) != 1:
        raise ValueError('候选结构必须在共同结构表中唯一对应。')
    key = matches[0]['key']
    token = hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()[:16]
    output = Path(output).resolve() if output else root/'candidate_finetuning'/token
    command = [sys.executable, '-u', ENTRY/'finetune.py', '--plan', plan_path,
        '--student', complete['student'], '--output', output, '--design-task', design_task,
        '--design-id', design_id, '--updates', str(updates), '--profile', 'robust',
        '--randomization-strength', '1', '--device', f'cuda:{gpu}',
        '--evaluation-profile', evaluation_profile, '--evaluation-seeds', evaluation_seeds,
        '--tracking-rel-tolerance', str(tracking_rel_tolerance),
        '--min-command-progress-ratio', str(min_command_progress_ratio),
        '--timeout-seconds', '1800', '--gpu-min-free-gb', '12.5', '--gpu-max-utilization', '100']
    if (output/'manifest.json').is_file(): command.append('--resume')
    if resume_paused: command.append('--resume-paused')
    print('候选结构：', key, '微调输出：', output)
    run(command, preview=preview)


def video(root, task, gpu, index, preview=False):
    plan = root/'shared_plan/plan.json'; complete = root/'shared_policy/completed.json'
    if not plan.is_file() or not complete.is_file():
        raise RuntimeError('请先完成共享策略蒸馏，再录共享策略录像。')
    native = {'rough':'stairs', 'backpack':'backpack', 'amp':'run'}[task]
    ids = read(plan)['teachers'][native]['design_ids']
    if not 0 <= index < len(ids):
        raise ValueError(f'结构序号应在0到{len(ids)-1}之间。')
    output = root/'videos'/f'{task}_{index}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    command = [sys.executable, '-u', ENTRY/'shared_policy'/('rollout_amp.py' if task=='amp' else 'rollout.py'),
               '--plan', plan, '--design-ids', ids[index], '--student', read(complete)['student'],
               '--output', output, '--mode', 'evaluate', '--steps', '1000', '--video', '--headless',
               '--profile', 'nominal', '--randomization-strength', '0',
               '--device', 'cuda:0', f'--kit_args=--/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0']
    if task != 'amp':
        command += ['--task', native]
    if task == 'backpack':
        command += ['--payload-kg', '10']
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print('录像输出目录：', output, flush=True)
    run(command, env=env, preview=preview)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('-h', '--help', action='help', help='显示本说明并退出')
    parser._positionals.title = '操作'
    parser._optionals.title = '选项'
    parser.add_argument('操作', choices=['准备','参考动作','资产','迁移','rough','负重','amp','蒸馏','微调','状态','录像'])
    parser.add_argument('--gpu', type=int, default=0, help='显卡编号，默认0')
    parser.add_argument('--cpu', action='store_true', help='单独生成参考动作时使用CPU')
    parser.add_argument('--任务', choices=['rough','负重','amp','全部'], help='资产默认全部任务；录像默认rough')
    parser.add_argument('--结构', type=int, default=0, help='录像的结构序号，从0开始')
    parser.add_argument('--candidate-design-id', help='微调所选任务结构库中的真实结构ID')
    parser.add_argument('--candidate-task', choices=['stairs','backpack','run'], default='stairs')
    parser.add_argument('--finetune-updates', type=int, default=100)
    parser.add_argument('--finetune-output', type=Path, help='可选：为候选微调指定独立实验目录')
    parser.add_argument('--finetune-evaluation-seeds', default='20261118')
    parser.add_argument('--root', type=Path, help='高级选项：自定义实验根目录')
    parser.add_argument('--migrated', action='store_true', help='使用默认的独立迁移目录')
    parser.add_argument('--source-root', type=Path, help='仅用于迁移：指定旧实验目录')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不创建实验或启动训练')
    parser.add_argument('--resume-paused', action='store_true', help='明确恢复所选任务暂停；无首组断点重开另用--restart-initial')
    parser.add_argument('--restart-initial', action='store_true', help='明确重开无检查点首组，自动复用USD并记住新目录')
    parser.add_argument('--collection-profile', choices=['nominal','robust'], default='robust', help='共享蒸馏采集配置，默认按任务随机化')
    parser.add_argument('--randomization-strength', type=float, default=1., help='共享蒸馏目标随机化强度，0到1')
    parser.add_argument('--initial-randomization-strength', type=float, default=.25, help='共享BC首轮随机化强度，逐轮增加至目标')
    parser.add_argument('--evaluation-profile', choices=['nominal','robust','both'], default='both', help='候选微调后的步态检查配置')
    parser.add_argument('--selection-seeds', default='20261117,20261119,20261120', help='三轮共用的三个选模种子，逗号分隔')
    parser.add_argument('--samples-per-design-per-epoch', type=int, default=32, help='每结构每数据源每epoch的计算预算，不截断训练数据')
    parser.add_argument('--validation-rows-per-design-shard', type=int, default=128, help='每结构每数据源的监督验证行预算')
    parser.add_argument('--replay-cache-shards', type=int, default=12, help='内存中的回放分片缓存数量')
    parser.add_argument('--tracking-abs-tolerance', type=float, default=None, help='已停用：运动跟踪不再使用绝对误差下限')
    parser.add_argument('--standing-speed-tolerance', type=float, default=None, help='已停用：共享策略已关闭站立指令并移除站立验收项')
    parser.add_argument('--tracking-rel-tolerance', type=float, default=.25, help='仅运动阶段的跟踪RMSE相对运动指令速度RMS的容差')
    parser.add_argument('--min-command-progress-ratio', type=float, default=.5, help='运动指令下实际进度与指令进度的最小比例')
    args = parser.parse_args(argv)
    if args.tracking_abs_tolerance is not None:
        parser.error('--tracking-abs-tolerance已停用；运动跟踪仅使用--tracking-rel-tolerance')
    if args.standing_speed_tolerance is not None:
        parser.error('--standing-speed-tolerance已停用；共享策略已关闭站立指令并移除站立验收项')
    if args.gpu < 0:
        parser.error('显卡编号不能为负数')
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    action = ALIASES.get(args.操作, args.操作)
    if args.root and args.migrated:
        parser.error('--root 与 --migrated 只选一个')
    root = (args.root or (MIGRATED_ROOT if args.migrated or action == 'migrate' else DEFAULT_ROOT)).resolve()
    if args.source_root is not None and action != 'migrate':
        parser.error('--source-root 仅用于迁移')
    if args.cpu and action != 'references':
        parser.error('--cpu仅用于单独生成参考动作')
    if args.resume_paused and action not in ('migrate','assets','rough','backpack','amp','distill','finetune'):
        parser.error('--resume-paused仅用于迁移、资产准备、专家任务、蒸馏或微调')
    if args.restart_initial and action not in ('rough','backpack','amp'):
        parser.error('--restart-initial仅用于专家任务')
    if args.resume_paused and args.restart_initial:
        parser.error('续训恢复与首组重开不能同时选择')
    if action == 'migrate':
        if args.任务:
            parser.error('一次性迁移包含全部三个任务')
        migrate_assets(root,args.source_root,args.gpu,args.dry_run,args.resume_paused)
        return
    if (args.migrated or (root/'migration_in_progress.json').is_file()) and not (root/'migration_completed.json').is_file():
        parser.error('该迁移库尚未完成。先运行 start.py 迁移。')
    if action == 'status':
        print('实验目录：', root)
        for task in ('rough','backpack','amp'):
            output, resume = select_run(root, task)
            status = '存在完成记录，仍需模型与步态验收' if (output/'completed.json').is_file() else ('已有计划，可续训' if resume else '尚未开始')
            print(task, status, output)
        return
    if action == 'video':
        task = args.任务 or 'rough'
        if task == '全部':
            parser.error('录像请一次选择一个任务')
        video(root, ALIASES.get(task,task), args.gpu, args.结构, args.dry_run)
        return
    if action == 'finetune':
        if not args.candidate_design_id or args.finetune_updates < 1:
            parser.error('微调需要--candidate-design-id和正数更新预算')
        finetune(root,args.gpu,args.candidate_design_id,args.candidate_task,args.finetune_updates,
            args.dry_run,args.resume_paused,args.finetune_output,args.evaluation_profile,args.finetune_evaluation_seeds,
            args.tracking_rel_tolerance,args.min_command_progress_ratio)
        return
    ensure_banks(root, args.dry_run)
    if action == 'prepare':
        return
    if action == 'references':
        references(root, args.gpu, args.cpu, args.dry_run)
    elif action == 'assets':
        prepare_assets(root, args.任务 or '全部', args.gpu, args.dry_run, args.resume_paused)
    elif action == 'distill':
        distill(root, args.gpu, args.dry_run, args.collection_profile, args.randomization_strength,
                args.initial_randomization_strength,args.selection_seeds,
                args.samples_per_design_per_epoch,args.validation_rows_per_design_shard,args.replay_cache_shards,
                args.tracking_rel_tolerance,args.min_command_progress_ratio,args.resume_paused)
    else:
        launch_expert(root, action, args.gpu, args.dry_run, args.resume_paused, args.restart_initial)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, subprocess.CalledProcessError) as error:
        print('启动未完成：'+str(error), file=sys.stderr)
        raise SystemExit(1)
