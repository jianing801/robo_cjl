"""验证中文统一入口的目录判断和实际命令，不启动仿真或训练。"""
import importlib.util
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('simple_launcher', REPO/'start.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def banks(root):
    for task in ('rough','backpack','amp'):
        path = launcher.bank_path(root, task); path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'task_key': task}))


def plan(root, name):
    folder = root/'experts'/name; folder.mkdir(parents=True)
    (folder/'plan.json').write_text(json.dumps({'bank': str(launcher.bank_path(root,'rough'))}))
    return folder


def test_fresh_and_empty_outputs_never_need_manual_paths(tmp_path):
    banks(tmp_path)
    output, resume = launcher.select_run(tmp_path,'rough')
    assert output == tmp_path/'experts/rough' and not resume
    output.mkdir(parents=True)
    next_output, resume = launcher.select_run(tmp_path,'rough')
    assert next_output != output and not resume and output.is_dir()
    assert not next_output.exists()


def test_missing_plan_with_files_never_silently_starts_over(tmp_path):
    folder = tmp_path/'experts/rough'; folder.mkdir(parents=True)
    marker = folder/'model_0.pt'; marker.write_text('keep')
    with pytest.raises(RuntimeError): launcher.select_run(tmp_path,'rough')
    assert marker.read_text() == 'keep'


def test_finds_manual_timestamp_run_and_remembers_it(tmp_path):
    banks(tmp_path); existing = plan(tmp_path,'rough_20261115_123456')
    assert launcher.select_run(tmp_path,'rough') == (existing,True)
    launcher.record_run(tmp_path,'rough',existing)
    plan(tmp_path,'rough_20261115_999999')
    assert launcher.select_run(tmp_path,'rough') == (existing,True)


def test_launch_record_cannot_escape_experiment(tmp_path):
    banks(tmp_path); folder=tmp_path/'启动记录'; folder.mkdir()
    (folder/'rough.json').write_text(json.dumps({'output':str(tmp_path.parent)}))
    with pytest.raises(ValueError): launcher.select_run(tmp_path,'rough')


@pytest.mark.parametrize('task',['rough','backpack','amp'])
def test_native_expert_commands_keep_official_budget_and_gpu_interface(tmp_path,task):
    command=launcher.expert_command(tmp_path,task,4,tmp_path/'output',True)
    assert command[command.index('--device')+1]=='cuda:4'
    assert '--execute' in command and '--resume' in command
    assert '--config' not in command  # 正式入口自行使用冻结配置。
    if task=='amp':
        assert command[command.index('--reference-library')+1]==str(tmp_path/'references/library.json')


def test_preview_does_not_create_output_or_launch_record(tmp_path,capsys):
    banks(tmp_path)
    launcher.main(['rough','--root',str(tmp_path),'--gpu','0','--dry-run'])
    assert not (tmp_path/'experts').exists() and not (tmp_path/'启动记录').exists()
    assert '--device cuda:0' in capsys.readouterr().out


def test_pause_marker_is_not_removed(tmp_path):
    banks(tmp_path); folder=plan(tmp_path,'rough'); marker=folder/'USER_PAUSED'; marker.touch()
    with pytest.raises(RuntimeError): launcher.launch_expert(tmp_path,'rough',0,True)
    assert marker.exists()


def test_partial_bank_directory_is_not_overwritten(tmp_path,monkeypatch):
    keep=tmp_path/'partial.txt'; keep.write_text('keep')
    monkeypatch.setattr(launcher,'run',lambda *a,**k:pytest.fail('不应启动生成'))
    with pytest.raises(RuntimeError): launcher.ensure_banks(tmp_path)
    assert keep.read_text()=='keep'


def test_cpu_references_command_uses_separate_environment(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(launcher,'run',lambda cmd,**kwargs:calls.append((cmd,kwargs)))
    launcher.references(tmp_path,4,True,True)
    command=calls[0][0]
    assert command[:5]==['conda','run','--no-capture-output','-n','amp_hhtools']
    assert command[-1]=='cpu'


def test_shared_video_keeps_one_design_and_ten_kg(tmp_path,monkeypatch):
    folder=tmp_path/'shared_plan';folder.mkdir()
    (folder/'plan.json').write_text(json.dumps({'teachers':{'backpack':{'design_ids':['one','two']}}}))
    output=tmp_path/'shared_policy';output.mkdir()
    (output/'completed.json').write_text(json.dumps({'student':'model.pt'}))
    calls=[];monkeypatch.setattr(launcher,'run',lambda cmd,**kwargs:calls.append(cmd))
    launcher.video(tmp_path,'backpack',0,1,True)
    command=calls[0]
    assert command[command.index('--design-ids')+1]=='two'
    assert command[command.index('--payload-kg')+1]=='10'


def test_amp_resume_keeps_original_reference_library(tmp_path):
    output=tmp_path/'amp';output.mkdir()
    library=str(tmp_path/'original_reference/library.json')
    (output/'plan.json').write_text(json.dumps({'reference_library':library}))
    command=launcher.expert_command(tmp_path,'amp',0,output,True)
    assert command[command.index('--reference-library')+1]==library


def test_existing_shared_plan_cannot_silently_switch_teachers(tmp_path,monkeypatch):
    teachers={}
    for task in ('rough','backpack','amp'):
        folder=tmp_path/'experts'/task;folder.mkdir(parents=True)
        (folder/'completed.json').write_text(json.dumps({'checkpoint':str(folder/'new.pt')}))
        teachers[task]=folder
    monkeypatch.setattr(launcher,'select_run',lambda root,task:(teachers[task],True))
    folder=tmp_path/'shared_plan';folder.mkdir()
    (folder/'plan.json').write_text(json.dumps({'teachers':{'stairs':{'checkpoint':str(tmp_path/'old.pt')}}}))
    with pytest.raises(ValueError,match='教师'):launcher.distill(tmp_path,0,True)


def test_prepare_all_assets_is_sequential_and_never_calls_training(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(launcher,'prepare_task_assets',lambda root,task,gpu,preview:calls.append((task,gpu,preview)))
    monkeypatch.setattr(launcher,'launch_expert',lambda *a,**k:pytest.fail('资产准备不得启动PPO'))
    banks(tmp_path)
    launcher.main(['资产','--gpu','5','--root',str(tmp_path),'--dry-run'])
    assert calls==[('rough',5,True),('backpack',5,True),('amp',5,True)]


def test_prepare_all_assets_stops_on_first_failure(tmp_path,monkeypatch):
    calls=[]
    def convert(root,task,gpu,preview):
        calls.append(task)
        if task=='backpack':raise RuntimeError('保留失败现场')
    monkeypatch.setattr(launcher,'prepare_task_assets',convert)
    with pytest.raises(RuntimeError):launcher.prepare_assets(tmp_path,'全部',5)
    assert calls==['rough','backpack']


def test_asset_preparation_has_separate_gpu_threshold_and_remembered_cache(tmp_path,monkeypatch):
    import sys,hashlib
    from types import SimpleNamespace
    from contextlib import nullcontext
    banks(tmp_path);events=[]
    def ensure(root,bank,python,kit,environment,child):
        events.append(('convert',root))
        cache=root/'native_cache_fixture';cache.mkdir()
        (cache/'completed.json').write_text('{}')
        return cache
    monkeypatch.setattr(launcher,'reject_asset_workers',lambda bank:None)
    monkeypatch.setitem(sys.modules,'train',SimpleNamespace(controller_lock=lambda root:nullcontext(),run_child=lambda *a:pytest.fail('测试不应执行原生子进程')))
    monkeypatch.setitem(sys.modules,'hybrid_native_cache',SimpleNamespace(validate_cache=lambda *a:None,ensure_cache=ensure,sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()))
    monkeypatch.setitem(sys.modules,'hybrid_processes',SimpleNamespace(reject_duplicate_bank_safely=lambda bank:None))
    def candidates(rows,env,**kwargs):
        assert kwargs['min_free_mib']==4096 and kwargs['requested']=='cuda:5'
        return [{'index':5,'uuid':'GPU-fixture','free_mib':10240}]
    monkeypatch.setitem(sys.modules,'robolab.rough_gpu',SimpleNamespace(query_gpus=lambda:[],candidates=candidates,
        gpu_lease=lambda choices,root:nullcontext(choices[0]),child_environment=lambda gpu,env:dict(env)))
    launcher.prepare_task_assets(tmp_path,'rough',5)
    assert events[0][1]==tmp_path/'资产准备/rough'
    assert not (tmp_path/'experts').exists()
    command=launcher.expert_command(tmp_path,'rough',5,tmp_path/'experts/rough',False)
    assert command[command.index('--gpu-min-free-gb')+1]=='12.5'
    assert command[command.index('--asset-cache')+1]==str(tmp_path/'资产准备/rough/native_cache_fixture')


def test_asset_index_cannot_reuse_another_bank(tmp_path):
    banks(tmp_path);folder=tmp_path/'启动记录';folder.mkdir()
    (folder/'rough_asset_cache.json').write_text(json.dumps({'bank_sha256':'wrong'}))
    with pytest.raises(ValueError):launcher.expert_command(tmp_path,'rough',0,tmp_path/'output',False)


def test_existing_converter_for_same_bank_is_not_duplicated(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    bank=tmp_path/'bank.json'
    other=SimpleNamespace(pid=90001,info={'cmdline':['python','prepare_native_cache.py','--bank',str(bank)]})
    fake=SimpleNamespace(Process=lambda:SimpleNamespace(parents=lambda:[]),process_iter=lambda fields:[other],
        NoSuchProcess=ProcessLookupError,ZombieProcess=ProcessLookupError)
    monkeypatch.setitem(sys.modules,'psutil',fake)
    with pytest.raises(RuntimeError,match='90001'):launcher.reject_asset_workers(bank)
    launcher.reject_asset_workers(tmp_path/'different_bank.json')


@pytest.mark.parametrize('mode',['success','failure','nonzero','missing_summary'])
def test_cache_supervisor_always_records_child_exit(tmp_path,mode):
    import sys
    folder=tmp_path/'run/attempts/cache_one/train';folder.parent.mkdir(parents=True)
    result=folder/'result'
    script='from pathlib import Path; import json,sys; p=Path('+repr(str(result))+'); p.mkdir(); print("TEST_CACHE_STAGE",flush=True); '
    if mode=='success':script+='(p/"summary.json").write_text("{}")'
    elif mode=='failure':script+='(p/"failure.json").write_text(json.dumps({"traceback":"test failure"}))'
    elif mode=='nonzero':script+='sys.exit(7)'
    else:script+='pass'
    if mode=='success':launcher.run_cache_child([sys.executable,'-u','-c',script],folder,{**__import__('os').environ})
    else:
        with pytest.raises(RuntimeError):launcher.run_cache_child([sys.executable,'-u','-c',script],folder,{**__import__('os').environ})
    record=json.loads((folder/'exit.json').read_text())
    assert record['returncode']==(0 if mode=='success' else 7 if mode=='nonzero' else 1)
    assert record['raw_returncode']==(7 if mode=='nonzero' else 0)
    assert 'TEST_CACHE_STAGE' in (folder/'stdout.log').read_text()


def test_cache_supervisor_parent_interrupt_records_pause_and_exit(tmp_path,monkeypatch):
    import sys,os
    folder=tmp_path/'run/attempts/cache_one/train';folder.parent.mkdir(parents=True)
    original_sleep=launcher.time.sleep;once=[True]
    def interrupt_once(seconds):
        if once[0]:once[0]=False;raise KeyboardInterrupt()
        original_sleep(min(seconds,.01))
    monkeypatch.setattr(launcher.time,'sleep',interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        launcher.run_cache_child([sys.executable,'-u','-c','import time; time.sleep(30)'],folder,dict(os.environ))
    assert (tmp_path/'run/USER_PAUSED').exists()
    assert json.loads((folder/'exit.json').read_text())['interrupted'] is True


def test_cache_supervisor_prints_heartbeat_without_child_output(tmp_path,monkeypatch,capsys):
    import sys,os
    folder=tmp_path/'run/attempts/cache_one/train';folder.parent.mkdir(parents=True)
    ticks=iter(range(0,100000,20))
    monkeypatch.setattr(launcher.time,'monotonic',lambda:next(ticks))
    script='import time; from pathlib import Path; time.sleep(.5); p=Path('+repr(str(folder/'result'))+'); p.mkdir(); (p/"summary.json").write_text("{}")'
    launcher.run_cache_child([sys.executable,'-u','-c',script],folder,dict(os.environ))
    assert 'CACHE_HEARTBEAT' in capsys.readouterr().out


def test_explicit_asset_resume_audits_marker_but_cannot_resume_ppo(tmp_path):
    banks(tmp_path);folder=tmp_path/'资产准备/rough';folder.mkdir(parents=True)
    marker=folder/'USER_PAUSED';marker.touch()
    launcher.resume_asset_pause(tmp_path,'rough')
    assert not marker.exists() and list((tmp_path/'启动记录').glob('rough_asset_resume_*.json'))
    marker.touch();(folder/'commits').mkdir();(folder/'commits/stage_0.json').write_text('{}')
    with pytest.raises(RuntimeError,match='PPO'):launcher.resume_asset_pause(tmp_path,'rough')
    assert marker.exists()


def restart_fixture(root,monkeypatch):
    import hashlib
    banks(root)
    bank=launcher.bank_path(root,'rough')
    bank.write_text(json.dumps({'task_key':'stairs','designs':[{'id':'one'}]}))
    output=plan(root,'rough_old');(output/'USER_PAUSED').touch()
    (output/'attempts').mkdir();(output/'commits').mkdir()
    config=launcher.read(launcher.ENTRY/'configs/rough.json')
    (output/'plan.json').write_text(json.dumps({'bank':str(bank.resolve()),'configuration':config}))
    cache=output/'native_cache_old';cache.mkdir()
    (cache/'completed.json').write_text(json.dumps(dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,
        bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),assets=[{'design':'one'}])))
    monkeypatch.setattr(launcher,'reject_active_expert',lambda *args:None)
    return output,cache


def test_explicit_initial_restart_preserves_old_pause_and_reuses_cache(tmp_path,monkeypatch):
    output,cache=restart_fixture(tmp_path,monkeypatch)
    before=(output/'plan.json').read_bytes()
    new,reference,selected_cache=launcher.prepare_initial_restart(tmp_path,'rough',output)
    assert new!=output and not new.exists() and selected_cache==str(cache.resolve())
    assert (output/'USER_PAUSED').exists() and (output/'plan.json').read_bytes()==before
    assert launcher.read(tmp_path/'启动记录/rough_asset_cache.json')['cache']==str(cache.resolve())
    audit=next((tmp_path/'启动记录').glob('rough_initial_restart_*/authorization.json'))
    assert launcher.read(audit)['explicit_restart_initial'] is True


def test_restart_preview_does_not_write_or_remove_anything(tmp_path,monkeypatch):
    output,cache=restart_fixture(tmp_path,monkeypatch)
    launcher.launch_expert(tmp_path,'rough',4,preview=True,restart_initial=True)
    assert not (tmp_path/'启动记录').exists() and (output/'USER_PAUSED').exists()


@pytest.mark.parametrize('fault',['checkpoint','commit','bootstrap','config','cache','other_checkpoint','active'])
def test_initial_restart_refuses_to_discard_learned_state_or_change_settings(tmp_path,monkeypatch,fault):
    output,cache=restart_fixture(tmp_path,monkeypatch)
    if fault=='checkpoint':(output/'model_0.pt').write_text('keep')
    if fault=='commit':(output/'commits/stage_000000.json').write_text('{}')
    if fault in ('bootstrap','config'):
        value=launcher.read(output/'plan.json')
        if fault=='bootstrap':value['bootstrap_checkpoint']='keep.pt'
        else:value['configuration']={}
        (output/'plan.json').write_text(json.dumps(value))
    if fault=='cache':(cache/'completed.json').unlink()
    if fault=='other_checkpoint':
        other=plan(tmp_path,'rough_manual');(other/'model_0.pt').write_text('keep')
    if fault=='active':
        def reject(*args):raise RuntimeError('active')
        monkeypatch.setattr(launcher,'reject_active_expert',reject)
    with pytest.raises(RuntimeError):launcher.prepare_initial_restart(tmp_path,'rough',output)
    assert (output/'USER_PAUSED').exists() and not (tmp_path/'启动记录').exists()


def test_restart_launch_uses_unified_entry_and_remembers_new_directory(tmp_path,monkeypatch):
    output,cache=restart_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(launcher.sys,'platform','win32')
    monkeypatch.setattr(launcher,'start_task_timer',lambda *a,**k:None)
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        selected=Path(command[command.index('--output-root')+1]);selected.mkdir()
        (selected/'plan.json').write_text(json.dumps({'bank':str(launcher.bank_path(tmp_path,'rough').resolve())}))
    monkeypatch.setattr(launcher,'run',run)
    launcher.launch_expert(tmp_path,'rough',4,restart_initial=True)
    assert len(commands)==1
    command=commands[0]
    assert '--resume' not in command
    assert command[command.index('--device')+1]=='cuda:4'
    assert command[command.index('--asset-cache')+1]==str(cache.resolve())
    selected,resume=launcher.select_run(tmp_path,'rough')
    assert resume and selected!=output and (output/'USER_PAUSED').exists()


def test_training_pause_resume_is_explicit_and_audited(tmp_path,monkeypatch):
    output,cache=restart_fixture(tmp_path,monkeypatch)
    (output/'commits/stage_000000.json').write_text('{}')
    launcher.resume_expert_pause(tmp_path,'rough',output)
    assert not (output/'USER_PAUSED').exists()
    audit=next((tmp_path/'启动记录').glob('rough_training_resume_*.json'))
    assert launcher.read(audit)['explicit_training_resume'] is True


def test_no_committed_checkpoint_cannot_be_disguised_as_pause_resume(tmp_path,monkeypatch):
    output,cache=restart_fixture(tmp_path,monkeypatch)
    launch=output/'attempts/session_0/train';launch.mkdir(parents=True)
    (launch/'launch.json').write_text(json.dumps({'command':['python','persistent_worker.py']}))
    with pytest.raises(RuntimeError,match='restart-initial'):
        launcher.resume_expert_pause(tmp_path,'rough',output)
    assert (output/'USER_PAUSED').exists()
