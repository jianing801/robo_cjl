"""Independent training wall-clock accounting; never imports or controls simulation."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil

DISPLAY_INTERVAL_SECONDS = 3600


def timestamp(value):
    return datetime.fromtimestamp(value,timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def duration(seconds):
    seconds=max(0,int(seconds));hours,remainder=divmod(seconds,3600);minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');os.replace(temporary,path)


def process_matches(pid,created):
    try:
        p=psutil.Process(pid)
        return abs(p.create_time()-created)<.01 and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def timer_folder(root,task,output):
    key=hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:20]
    return Path(root)/'启动记录'/'计时记录'/(task+'_'+key)


def start_task_timer(root,task,output,target,started_wall=None,started_monotonic=None,pid=None,spawn=True,start_source='start.py'):
    """开始计时；不提前创建训练目录，保留engine的目录创建与进程归属规则。"""
    output=Path(output).resolve();root=Path(root).resolve();pid=os.getpid() if pid is None else pid
    created=psutil.Process(pid).create_time();now=time.time()
    start=now if started_wall is None else started_wall
    mono=time.monotonic() if started_monotonic is None else started_monotonic
    folder=timer_folder(root,task,output);folder.mkdir(parents=True,exist_ok=True)
    record=folder/f'launch_{pid}_{int(created*1000000)}.json'
    if not record.exists():
        progress=any((output/'processes').glob('*/launch.json')) if task=='distill' else any((output/'commits').glob('stage_*.json'))
        historical=(progress or (output/'completed.json').exists()) and not any(folder.glob('launch_*.json'))
        atomic_json(record,dict(task=task,root=str(root),output=str(output),target_updates=target,pid=pid,pid_create_time=created,
            started_epoch=start,started_at=timestamp(start),started_monotonic=mono,elapsed_seconds=max(0,time.monotonic()-mono),
            updated_epoch=now,status='running',completed_at_start=(output/'completed.json').exists(),
            history_before_timer_unknown=historical,start_source=start_source))
    if spawn:
        error=(folder/(record.stem+'.monitor.stderr.log')).open('a',encoding='utf8')
        try:
            options=dict(stdin=subprocess.DEVNULL,stderr=error)
            if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
            else:options['start_new_session']=True
            subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--monitor',str(record)],**options)
        finally:error.close()
    print('TASK_TIMER_STARTED',task,'计时记录：',folder/'timing.json',flush=True)
    return record


def completed_marker(record):
    path=Path(record['output'])/'completed.json'
    try:
        value=json.loads(path.read_text(encoding='utf8'))
        if record['task']=='distill':
            if value.get('result')!='DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING':
                return None
            checkpoint=Path(value['student'])
        else:
            if value.get('completed_updates')!=record['target_updates']:
                return None
            checkpoint=Path(value['checkpoint'])
        if checkpoint.is_file() and checkpoint.stat().st_size>0:
            return path.stat().st_mtime
    except (OSError,ValueError,KeyError,TypeError):pass
    return None


def active_seconds(sessions, until):
    """Add recorded running intervals once, even if a rejected launcher overlapped."""
    intervals = sorted((r['started_epoch'], min(until, r['started_epoch'] + r['elapsed_seconds']))
                       for r in sessions if r['status'] != 'already_complete'
                       and not r.get('completed_at_start') and r['started_epoch'] < until)
    total, covered_end = 0., float('-inf')
    for start, end in intervals:
        total += max(0., end - max(start, covered_end))
        covered_end = max(covered_end, end)
    return total


def summarize(folder):
    folder=Path(folder);sessions=[json.loads(p.read_text(encoding='utf8')) for p in folder.glob('launch_*.json')]
    if not sessions:return None
    sessions.sort(key=lambda r:r['started_epoch']);first=sessions[0]['started_epoch']
    completed=[r for r in sessions if r['status']=='completed']
    finished=min((r['ended_epoch'] for r in completed),default=None)
    end=finished if finished is not None else max(r['updated_epoch'] for r in sessions)
    active=active_seconds(sessions,end)
    running=[r for r in sessions if r['status']=='running' and process_matches(r['pid'],r['pid_create_time'])]
    latest=sessions[-1]
    result=dict(task=latest['task'],output=latest['output'],first_started_at=timestamp(first),
        completed_at=None if finished is None else timestamp(finished),
        status='completed' if finished is not None else 'running' if running else latest['status'],
        accumulated_run_seconds=active,launch_count=len(sessions),
        history_before_timer_unknown=any(r.get('history_before_timer_unknown',False) for r in sessions),
        includes_preparation_and_scene_loading=True,
        updated_at=timestamp(max(r['updated_epoch'] for r in sessions)),sessions=sessions)
    result['accumulated_run_hms']=duration(active)
    atomic_json(folder/'timing.json',result)
    output=Path(latest['output'])
    if output.is_dir():atomic_json(output/'timing.json',result)
    return result


@contextmanager
def monitor_lock(path):
    with Path(path).open('a+b') as stream:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            yield False;return
        try:yield True
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def monitor(path,poll_seconds=1,write_interval=30,display_interval=DISPLAY_INTERVAL_SECONDS):
    """独立观察驱动PID，保留exec/Ctrl-C行为；绝不启动、停止或恢复训练。"""
    path=Path(path)
    with monitor_lock(path.with_suffix('.lock')) as acquired:
        if not acquired:return
        record=json.loads(path.read_text(encoding='utf8'))
        if record['status']!='running':return
        prior=[json.loads(p.read_text(encoding='utf8')) for p in path.parent.glob('launch_*.json') if p != path]
        prior_active=active_seconds(prior, record['started_epoch'])
        next_display=(prior_active//display_interval+1)*display_interval
        last_write=float('-inf')
        while True:
            now=time.time();mono=time.monotonic();finish=completed_marker(record)
            alive=process_matches(record['pid'],record['pid_create_time'])
            if finish is not None and not record['completed_at_start']:
                record.update(status='completed',ended_epoch=max(record['started_epoch'],finish))
            elif not alive:
                state='already_complete' if record['completed_at_start'] and finish is not None else 'paused' if (Path(record['output'])/'USER_PAUSED').exists() else 'stopped_or_failed'
                record.update(status=state,ended_epoch=now)
            record['elapsed_seconds']=max(0,mono-record['started_monotonic'])
            if record['status']=='completed':record['elapsed_seconds']=max(0,record['ended_epoch']-record['started_epoch'])
            record['updated_epoch']=record.get('ended_epoch',now)
            due=not record['completed_at_start'] and prior_active+record['elapsed_seconds']>=next_display
            if record['status']!='running' or mono-last_write>=write_interval or due:
                atomic_json(path,record);summary=summarize(path.parent);last_write=mono
            if record['status']!='running':
                print('TASK_TIMER_FINISHED',record['task'],record['status'],
                      '累计训练用时',summary['accumulated_run_hms'],flush=True)
                return
            if due:
                next_display=(max(next_display,summary['accumulated_run_seconds'])//display_interval+1)*display_interval
                print('TASK_TIMER',record['task'],'累计训练用时',summary['accumulated_run_hms'],flush=True)
            time.sleep(poll_seconds)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--monitor',type=Path,required=True);args=parser.parse_args()
    monitor(args.monitor)


if __name__=='__main__':main()
