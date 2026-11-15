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

REPO = Path(__file__).resolve().parent
ENTRY = REPO / 'robolab/experiments/multitask_morphology'
DEFAULT_ROOT = REPO / 'robolab/logs/multitask_full3000_balanced_v2'
ALIASES = {'准备': 'prepare', '参考动作': 'references', '负重': 'backpack',
           '蒸馏': 'distill', '状态': 'status', '录像': 'video', '资产': 'assets'}


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
    paths = [bank_path(root, t) for t in ('rough', 'backpack', 'amp')]
    if all(p.is_file() for p in paths):
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
    cached = root/'启动记录'/f'{task}_asset_cache.json'
    if cached.is_file():
        value = read(cached)
        # 索引只指向本结构库的完整缓存；实际几何仍由原生worker严格校验。
        if value['bank_sha256'] != hashlib.sha256(bank_path(root,task).read_bytes()).hexdigest():
            raise ValueError('预转换缓存绑定了不同结构库。')
        proof = Path(value['cache'])/'completed.json'
        if not proof.is_file() or hashlib.sha256(proof.read_bytes()).hexdigest() != value['completed_sha256']:
            raise ValueError('预转换缓存完成证明缺失或变化，不能直接用于训练。')
        command += ['--asset-cache', value['cache']]
    return command


def record_run(root, task, output):
    folder = root/'启动记录'; folder.mkdir(exist_ok=True)
    value = dict(output=str(output.resolve()), bank=str(bank_path(root, task).resolve()))
    tmp = folder/f'{task}.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')
    os.replace(tmp, folder/f'{task}.json')


def launch_expert(root, task, gpu, preview=False):
    output, resume = select_run(root, task)
    print(('续训目录：' if resume else '新训练目录：')+str(output), flush=True)
    if (output/'USER_PAUSED').exists():
        raise RuntimeError('该任务有用户暂停标记；请明确恢复后再启动，不自动取消暂停。')
    if resume and task != 'amp':
        # 仅转换阶段允许通过已有审计工具迁移；PPO开始后严格拒绝改身份。
        current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ENTRY/'engine').glob('*.py') if not p.name.startswith('test_')}
        if read(output/'plan.json')['experiment']['hybrid_sources'] != current:
            run([sys.executable, ENTRY/'engine/refresh_cache_plan.py', '--root', output], preview=preview)
    command = expert_command(root, task, gpu, output, resume)
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
                          f'{cache_worker_status(worker.pid)}；最近输出：{last[0]}',flush=True)
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
    from hybrid_native_cache import ensure_cache, sha
    from hybrid_processes import reject_duplicate_bank_safely
    from robolab.rough_gpu import query_gpus, candidates, gpu_lease, child_environment
    bank = bank_path(root,task)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root/'attempts').mkdir(exist_ok=True)
    with controller_lock(cache_root):
        reject_duplicate_bank_safely(bank)
        reject_asset_workers(bank)
        # 这是转换的启动门槛，不是显存峰值保证；OOM保留日志并停止。
        choices = candidates(query_gpus(), os.environ, min_free_mib=4*1024,
                             max_utilization=100, requested=f'cuda:{gpu}')
        with gpu_lease(choices, REPO/'robolab/logs/rpo_rough_multimorph/gpu_locks') as selected:
            environment = child_environment(selected, os.environ)
            kit = (f" --/renderer/activeGpu={selected['index']}"
                   " --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0")
            cache = ensure_cache(cache_root, bank, sys.executable, kit, environment, run_cache_child)
        folder = root/'启动记录'; folder.mkdir(exist_ok=True)
        record = dict(cache=str(cache.resolve()), bank_sha256=sha(bank),
                      completed_sha256=sha(cache/'completed.json'))
        temporary = folder/f'{task}_asset_cache.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record,ensure_ascii=False),encoding='utf8')
        os.replace(temporary,folder/f'{task}_asset_cache.json')
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


def distill(root, gpu, preview=False):
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
               '--device', f'cuda:{gpu}', '--batch-designs', '10', '--collect-steps', '1000',
               '--stride', '5', '--epochs', '20', '--dagger-rounds', '3',
               '--timeout-seconds', '1800', '--gpu-min-free-gb', '12.5', '--gpu-max-utilization', '100']
    if (output/'manifest.json').is_file():
        command.append('--resume')
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
    parser.add_argument('操作', choices=['准备','参考动作','资产','rough','负重','amp','蒸馏','状态','录像'])
    parser.add_argument('--gpu', type=int, default=0, help='显卡编号，默认0')
    parser.add_argument('--cpu', action='store_true', help='单独生成参考动作时使用CPU')
    parser.add_argument('--任务', choices=['rough','负重','amp','全部'], help='资产默认全部任务；录像默认rough')
    parser.add_argument('--结构', type=int, default=0, help='录像的结构序号，从0开始')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='高级选项：自定义实验根目录')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不创建实验或启动训练')
    parser.add_argument('--resume-paused', action='store_true', help='明确恢复仅转换阶段的用户暂停，不恢复PPO')
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error('显卡编号不能为负数')
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    root = args.root.resolve(); action = ALIASES.get(args.操作, args.操作)
    if args.cpu and action != 'references':
        parser.error('--cpu仅用于单独生成参考动作')
    if args.resume_paused and action != 'assets':
        parser.error('--resume-paused仅用于USD资产准备')
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
    ensure_banks(root, args.dry_run)
    if action == 'prepare':
        return
    if action == 'references':
        references(root, args.gpu, args.cpu, args.dry_run)
    elif action == 'assets':
        prepare_assets(root, args.任务 or '全部', args.gpu, args.dry_run, args.resume_paused)
    elif action == 'distill':
        distill(root, args.gpu, args.dry_run)
    else:
        launch_expert(root, action, args.gpu, args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, subprocess.CalledProcessError) as error:
        print('启动未完成：'+str(error), file=sys.stderr)
        raise SystemExit(1)
