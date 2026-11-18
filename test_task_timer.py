"""计时真实短子进程测试；不启动仿真、训练或GPU。"""
import json,os,subprocess,sys,time
from pathlib import Path
import pytest
import task_timer as timer


def child(delay=.1):
    return subprocess.Popen([sys.executable,'-c',f'import time; time.sleep({delay})'])


def record(tmp_path,process,task='rough',target=12000,output=None,**kwargs):
    output=tmp_path/'expert' if output is None else output
    return timer.start_task_timer(tmp_path,task,output,target,pid=process.pid,spawn=False,**kwargs)


def completion(output,target):
    output.mkdir(exist_ok=True);model=output/'model.pt';model.write_bytes(b'weights')
    (output/'completed.json').write_text(json.dumps(dict(completed_updates=target,checkpoint=str(model))))


@pytest.mark.parametrize('task,target',[('rough',12000),('backpack',12000),('amp',5000)])
def test_real_short_process_completion_and_visible_result(tmp_path,task,target):
    output=tmp_path/'expert';p=child(.5)
    try:
        r=record(tmp_path,p,task,target,output)
        assert not output.exists()  # 计时器不得抢先创建训练目录。
        completion(output,target)
        timer.monitor(r,poll_seconds=.01,write_interval=.01,display_interval=99)
        result=json.loads((output/'timing.json').read_text())
        assert result['status']=='completed' and result['task']==task
        assert result['accumulated_run_seconds']>=0
        assert 'task_total_wall_seconds' not in result
        assert result['includes_preparation_and_scene_loading'] is True
        assert json.loads(r.read_text())['target_updates']==target
    finally:p.wait()


@pytest.mark.parametrize('paused',[False,True])
def test_stopped_process_never_becomes_training_complete(tmp_path,paused):
    output=tmp_path/'expert';output.mkdir()
    if paused:(output/'USER_PAUSED').touch()
    p=child(.05);r=record(tmp_path,p,output=output);p.wait()
    timer.monitor(r,poll_seconds=.01,write_interval=.01,display_interval=99)
    result=json.loads((output/'timing.json').read_text())
    assert result['status']==('paused' if paused else 'stopped_or_failed')
    assert result['completed_at'] is None


def test_resume_keeps_only_accumulated_running_time(tmp_path):
    output=tmp_path/'expert';p=child(.01);r=record(tmp_path,p,output=output);p.wait()
    first=json.loads(r.read_text());first.update(started_epoch=100,started_at=timer.timestamp(100),updated_epoch=110,ended_epoch=110,status='paused',elapsed_seconds=10)
    timer.atomic_json(r,first)
    second=dict(first,started_epoch=130,started_at=timer.timestamp(130),updated_epoch=150,ended_epoch=150,status='completed',elapsed_seconds=20)
    timer.atomic_json(r.parent/'launch_second.json',second)
    result=timer.summarize(r.parent)
    assert result['accumulated_run_seconds']==30
    assert 'task_total_wall_seconds' not in result
    assert 'between_launches_seconds' not in result
    assert result['launch_count']==2


def test_timing_includes_entry_preparation_before_monitor_started(tmp_path):
    p=child(.1)
    try:
        r=record(tmp_path,p,started_wall=time.time()-30,started_monotonic=time.monotonic()-30)
        assert json.loads(r.read_text())['elapsed_seconds']>=30
    finally:p.wait()


def test_completed_marker_requires_exact_budget_and_nonempty_checkpoint(tmp_path):
    output=tmp_path/'expert';p=child(.1)
    try:
        r=record(tmp_path,p,output=output);data=json.loads(r.read_text())
        completion(output,11999);assert timer.completed_marker(data) is None
        completion(output,12000);assert timer.completed_marker(data) is not None
        (output/'model.pt').write_bytes(b'');assert timer.completed_marker(data) is None
    finally:p.wait()


def test_old_completed_task_does_not_reset_original_finish_time(tmp_path):
    output=tmp_path/'expert';p=child(.1)
    try:
        r=record(tmp_path,p,output=output);completion(output,12000)
        timer.monitor(r,poll_seconds=.01,write_interval=.01,display_interval=99)
        before=timer.summarize(r.parent)
        extra=json.loads(r.read_text());extra.update(status='already_complete',started_epoch=time.time()+10,updated_epoch=time.time()+11,ended_epoch=time.time()+11,elapsed_seconds=1)
        timer.atomic_json(r.parent/'launch_check_again.json',extra)
        after=timer.summarize(r.parent)
        assert after['completed_at']==before['completed_at']
        assert after['accumulated_run_seconds']==before['accumulated_run_seconds']
    finally:p.wait()


def test_pid_reuse_and_long_duration_format():
    import psutil
    assert not timer.process_matches(os.getpid(),psutil.Process().create_time()-100)
    assert timer.duration(49*3600+62)=='49:01:02'


def test_hourly_display_carries_over_previous_launch(tmp_path, monkeypatch, capsys):
    times = iter([0, 1, 2, 3])
    current = {'time': 0}
    def tick(_):
        current['time'] = next(times)
    tick(None)
    monkeypatch.setattr(timer.time, 'monotonic', lambda: 100 + current['time'])
    monkeypatch.setattr(timer.time, 'time', lambda: 50_000 + current['time'])
    monkeypatch.setattr(timer.time, 'sleep', tick)
    monkeypatch.setattr(timer, 'process_matches', lambda pid, _: pid == 1 and current['time'] < 3)
    prior = dict(task='rough', output=str(tmp_path/'absent'), started_epoch=10,
                 elapsed_seconds=3598, updated_epoch=3608, status='paused', pid=2, pid_create_time=2)
    timer.atomic_json(tmp_path/'launch_prior.json', prior)
    path = tmp_path/'launch_current.json'
    timer.atomic_json(path, dict(prior, started_epoch=50_000, started_monotonic=100, elapsed_seconds=0,
        updated_epoch=50_000, status='running', pid=1, pid_create_time=1, completed_at_start=False))
    timer.monitor(path)
    lines = capsys.readouterr().out.splitlines()
    assert lines == ['TASK_TIMER rough 累计训练用时 01:00:00',
                     'TASK_TIMER_FINISHED rough stopped_or_failed 累计训练用时 01:00:01']
