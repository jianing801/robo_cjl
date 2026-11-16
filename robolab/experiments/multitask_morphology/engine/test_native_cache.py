"""Never reuse another morphology bank or a changed native asset."""
import json
from pathlib import Path
import pytest
from hybrid_native_cache import validate_cache,ensure_cache,sha

def make_cache(tmp_path):
    from hybrid_assets import asset_files,mesh_sources
    mesh=tmp_path/'mesh.stl';mesh.write_text('mesh source')
    urdf=tmp_path/'robot.urdf';urdf.write_text(f'<robot><link name="base"><visual><geometry><mesh filename="{mesh.as_posix()}"/></geometry></visual></link></robot>')
    data={'designs':[{'id':'one','urdf':str(urdf),'urdf_sha256':sha(urdf)}]}
    bank=tmp_path/'bank.json';bank.write_text(json.dumps(data))
    root=tmp_path/'run';root.mkdir();cache=root/'native_cache_verified';asset=cache/'one';asset.mkdir(parents=True)
    usd=asset/'robot.usd';usd.write_text('native asset')
    (asset/'.asset_hash').write_text('converter');(asset/'config.yaml').write_text('valid')
    (asset/'geometry.usd').write_text('actual collision geometry')
    proof=dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,mesh_sources=mesh_sources(data),bank_sha256=sha(bank),assets=[dict(design='one',
        usd=str(usd.resolve()),usd_sha256=sha(usd),urdf_sha256=sha(urdf),converter_hash='converter',files_sha256=asset_files(asset))])
    (cache/'completed.json').write_text(json.dumps(proof))
    return root,cache,bank,usd

def test_verified_cache_reused_before_launching_worker(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path)
    def unexpected(*args):raise AssertionError('must not launch another converter')
    assert ensure_cache(root,bank,'python','',{},unexpected)==cache
    usd.write_text('changed')
    with pytest.raises(ValueError):validate_cache(cache,bank)

def test_wrong_bank_and_incomplete_cache_rejected(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path)
    other=tmp_path/'other.json';other.write_text('{"designs":[]}')
    with pytest.raises(ValueError):validate_cache(cache,other)
    (cache/'one/config.yaml').unlink()
    with pytest.raises(ValueError):validate_cache(cache,bank)

def test_completed_proof_excludes_abandoned_partial_directories(tmp_path):
    from hybrid_native_cache import cache_directories
    root,cache,bank,usd=make_cache(tmp_path)
    partial=cache/'partial';partial.mkdir()
    (partial/'.asset_hash').write_text('other')
    (partial/'config.yaml').write_text('partial')
    (partial/'robot.usd').write_text('not an approved conversion')
    assert cache_directories(cache,bank)==[usd.parent.resolve()]

def test_reuse_verified_batch_keeps_original_files(tmp_path):
    from native_cache_coordinator import reuse_verified_batches
    root,cache,bank,usd=make_cache(tmp_path)
    artifact=tmp_path/'proofs/batch_0000';artifact.mkdir(parents=True)
    artifact.joinpath('summary.json').write_bytes((cache/'completed.json').read_bytes())
    destination=tmp_path/'new_cache';destination.mkdir()
    parsed=json.loads(bank.read_text());original=usd.read_bytes()
    rows=reuse_verified_batches([artifact.parent],parsed,sha(bank),destination)
    assert len(rows)==1 and Path(rows[0]['usd']).read_bytes()==original
    assert usd.read_bytes()==original
    usd.write_text('corrupt original')
    other=tmp_path/'another';other.mkdir()
    with pytest.raises(ValueError):reuse_verified_batches([artifact.parent],parsed,sha(bank),other)

def test_geometry_layer_corruption_is_detected_without_main_usd_change(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path);before=sha(usd)
    (usd.parent/'geometry.usd').write_text('corrupted collision geometry')
    assert sha(usd)==before
    with pytest.raises(ValueError,match='geometry dependency'):validate_cache(cache,bank)

def test_external_mesh_change_is_not_hidden_by_unchanged_urdf(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path)
    (tmp_path/'mesh.stl').write_text('changed mesh')
    with pytest.raises(ValueError,match='mesh dependencies'):validate_cache(cache,bank)

def test_fullbody_cannot_accept_a_partial_unproved_cache(tmp_path):
    from hybrid_native_cache import cache_directories
    root,cache,bank,usd=make_cache(tmp_path)
    (cache/'completed.json').unlink()
    with pytest.raises(ValueError,match='completed native cache'):cache_directories(cache,bank,require_complete=True)

def test_cache_interruption_is_not_mistaken_for_started_ppo(tmp_path):
    from hybrid_native_cache import training_attempt_started
    folder=tmp_path/'attempts/cache_001/train';folder.mkdir(parents=True)
    (folder/'launch.json').write_text(json.dumps({'command':['python','-u','prepare_native_cache.py']}))
    assert not training_attempt_started(tmp_path)
    session=tmp_path/'attempts/session_001/train';session.mkdir(parents=True)
    (session/'launch.json').write_text(json.dumps({'command':['python','-u','persistent_worker.py']}))
    assert training_attempt_started(tmp_path)



def test_asset_proof_is_atomic_and_interrupted_session_can_be_reused(tmp_path):
    from native_cache_coordinator import publish_asset_proof,reuse_verified_batches
    root,cache,bank,usd=make_cache(tmp_path)
    folder=tmp_path/'sessions/batch_0000/assets';folder.mkdir(parents=True)
    proof=json.loads((cache/'completed.json').read_text())
    publish_asset_proof(folder/'one.json',proof)
    assert not list(folder.glob('*.tmp'))
    (folder/'broken.json.tmp').write_text('{')
    destination=tmp_path/'resumed';destination.mkdir()
    rows=reuse_verified_batches([folder.parent.parent],json.loads(bank.read_text()),sha(bank),destination)
    assert len(rows)==1 and Path(rows[0]['usd']).read_bytes()==usd.read_bytes()
    with pytest.raises(FileExistsError):publish_asset_proof(folder/'one.json',proof)


@pytest.mark.parametrize('fault',['healthy','crash','stall','no_progress','failure','pause'])
def test_continuous_converter_watchdog_and_resume(tmp_path,monkeypatch,fault):
    import native_cache_coordinator as c
    import hybrid_bank,hybrid_assets
    from types import SimpleNamespace
    bank={'task_key':'stairs','designs':[{'id':str(i),'urdf_sha256':'fixture'} for i in range(12)]}
    bank_path=tmp_path/'bank.json';bank_path.write_text(json.dumps(bank))
    monkeypatch.setattr(hybrid_bank,'load_configurable_bank',lambda p:bank)
    monkeypatch.setattr(hybrid_assets,'mesh_sources',lambda b,**kwargs:{})
    monkeypatch.setattr(hybrid_assets,'validate_asset',lambda row:None)
    monkeypatch.setattr(hybrid_assets,'validate_asset_resume',lambda row:None)
    ticks=iter(range(10000));monkeypatch.setattr(c.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(c.time,'sleep',lambda seconds:None)
    calls=[];pause=tmp_path/'control';pause.mkdir()
    class Child:
        returncode=None
        def __init__(self,cmd,**kwargs):
            self.ids=json.loads(Path(cmd[cmd.index('--design-ids-file')+1]).read_text())
            self.output=Path(cmd[cmd.index('--output-dir')+1]);self.output.mkdir(parents=True)
            self.cache=Path(cmd[cmd.index('--cache-root')+1]);self.index=0
            self.first=not calls;calls.append(self.ids)
        def poll(self):
            if self.returncode is not None:return self.returncode
            if fault=='no_progress':self.returncode=7;return 7
            if self.first and fault=='failure':
                (self.output/'failure.json').write_text('{"traceback":"explicit conversion error"}')
                self.returncode=1;return 1
            if self.first and fault=='pause':
                (pause/'USER_PAUSED').touch();return None
            if self.first and self.index==2 and fault in ('crash','stall'):
                if fault=='crash':self.returncode=7;return 7
                return None
            if self.index==len(self.ids):self.returncode=0;return 0
            key=self.ids[self.index];self.index+=1
            c.publish_asset_proof(self.output/'assets'/f'{key}.json',dict(
                result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,bank_sha256=sha(bank_path),
                mesh_sources={},assets=[dict(design=key,urdf_sha256='fixture',usd=str(self.cache/key/'robot.usd'))]))
            return None
    monkeypatch.setattr(c.subprocess,'Popen',Child)
    monkeypatch.setattr(c,'stop_tree',lambda child:setattr(child,'returncode',-9))
    args=SimpleNamespace(bank=bank_path,cache_root=tmp_path/'cache',output_dir=tmp_path/'proof',
        batch_size=0,startup_timeout=3,stall_timeout=3,reuse_proof_root=[],pause_root=pause)
    if fault in ('no_progress','failure','pause'):
        with pytest.raises(RuntimeError):c.coordinate(args,[])
        assert len(calls)==(3 if fault=='no_progress' else 1)
        assert not (args.cache_root/'completed.json').exists()
    else:
        c.coordinate(args,[])
        proof=json.loads((args.cache_root/'completed.json').read_text())
        assert [row['design'] for row in proof['assets']]==[str(i) for i in range(12)]
        assert len(calls)==(1 if fault=='healthy' else 2)
        if fault!='healthy':assert calls[1]==[str(i) for i in range(2,12)]


def test_live_asset_proofs_validate_geometry_and_bank_before_accepting(tmp_path):
    from native_cache_coordinator import publish_asset_proof,read_asset_proofs
    root,cache,bank,usd=make_cache(tmp_path)
    folder=tmp_path/'live';proof=json.loads((cache/'completed.json').read_text())
    publish_asset_proof(folder/'assets/one.json',proof)
    parsed=json.loads(bank.read_text())
    assert len(read_asset_proofs(folder,cache,parsed,sha(bank),set()))==1
    with pytest.raises(ValueError,match='proof'):
        read_asset_proofs(folder,cache,parsed,'wrong-bank',set())
    (usd.parent/'geometry.usd').write_text('changed')
    with pytest.raises(ValueError,match='geometry'):
        read_asset_proofs(folder,cache,parsed,sha(bank),set())


@pytest.mark.parametrize('change',['converter_only','native','runtime','unknown_converter','checkpoint','commit','worker'])
def test_cache_plan_refresh_only_accepts_known_pre_ppo_converter_change(tmp_path,monkeypatch,change):
    import refresh_cache_plan as refresh
    from copy import deepcopy
    previous={name:values[0] for name,values in refresh.KNOWN_CONVERTERS.items()}
    previous['train.py']='unchanged-controller'
    old=dict(native_sources={'physics':'same'},hybrid_sources=previous)
    current=deepcopy(old)
    for name in refresh.KNOWN_CONVERTERS:current['hybrid_sources'][name]='new-converter'
    current['hybrid_sources']['refresh_cache_plan.py']='new-helper'
    plan=dict(experiment=old,bank='fixture',configuration={'budget':12000})
    if change=='native':current['native_sources']['physics']='changed'
    if change=='runtime':current['hybrid_sources']['train.py']='changed'
    if change=='unknown_converter':old['hybrid_sources']['prepare_native_cache.py']='unknown'
    if change=='checkpoint':(tmp_path/'model_0.pt').touch()
    if change=='commit':
        (tmp_path/'commits').mkdir();(tmp_path/'commits/stage_0.json').write_text('{}')
    if change=='worker':
        folder=tmp_path/'attempts/session_0/train';folder.mkdir(parents=True)
        (folder/'launch.json').write_text(json.dumps({'command':['python','persistent_worker.py']}))
    monkeypatch.setattr(refresh,'experiment_identity',lambda bank,config:current)
    if change=='converter_only':
        updated=refresh.refreshed_plan(tmp_path,plan)
        assert updated['experiment']==current and plan['experiment']==old
        assert updated['configuration']==plan['configuration']
    else:
        with pytest.raises((ValueError,RuntimeError)):refresh.refreshed_plan(tmp_path,plan)


def test_early_bank_validation_failure_is_recorded(tmp_path,monkeypatch):
    import native_cache_coordinator as c
    import hybrid_bank
    from types import SimpleNamespace
    bank=tmp_path/'bank.json';bank.write_text('{"task_key":"stairs"}')
    def fail(path):raise OSError('test disk read failure')
    monkeypatch.setattr(hybrid_bank,'load_configurable_bank',fail)
    output=tmp_path/'result'
    with pytest.raises(OSError,match='test disk'):
        c.coordinate(SimpleNamespace(bank=bank,output_dir=output),[])
    assert 'OSError: test disk read failure' in json.loads((output/'failure.json').read_text())['traceback']


def test_exception_formatter_never_reads_source_lines(monkeypatch):
    import native_cache_coordinator as c
    import linecache
    monkeypatch.setattr(linecache,'getline',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not read source')))
    try:raise OSError('test storage failure')
    except OSError as error:text=c.exception_text(error)
    assert 'OSError: test storage failure' in text and 'test_exception_formatter' in text


def test_known_persistent_converter_and_helper_migrate_only_before_ppo(tmp_path,monkeypatch):
    import refresh_cache_plan as refresh
    from copy import deepcopy
    source={name:values[-1] for name,values in refresh.KNOWN_CONVERTERS.items()}
    source['refresh_cache_plan.py']=refresh.KNOWN_REFRESH_HELPERS[0]
    source['train.py']='unchanged'
    old=dict(native_sources={'physics':'same'},hybrid_sources=source)
    new=deepcopy(old)
    for name in (*refresh.KNOWN_CONVERTERS,'refresh_cache_plan.py'):new['hybrid_sources'][name]='new'
    monkeypatch.setattr(refresh,'experiment_identity',lambda bank,config:new)
    plan=dict(experiment=old,bank='fixture',configuration={})
    assert refresh.refreshed_plan(tmp_path,plan)['experiment']==new
    old['hybrid_sources']['refresh_cache_plan.py']='unrecognized'
    with pytest.raises(ValueError,match='helper'):refresh.refreshed_plan(tmp_path,plan)


def test_coordinator_bootstrap_never_imports_isaac_or_native_libraries(tmp_path,monkeypatch):
    import importlib.util,sys,builtins
    import native_cache_coordinator as coordinator
    script=Path(coordinator.__file__).with_name('prepare_native_cache.py')
    spec=importlib.util.spec_from_file_location('pure_cache_bootstrap',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'meta_path',list(sys.meta_path))
    calls=[];argv=['--bank','bank.json','--cache-root','cache','--output-dir','result',
        '--headless','--device','cuda:0','--kit_args=--/renderer/activeGpu=5 --/physics/cudaDevice=0']
    monkeypatch.setattr(sys,'argv',[str(script),*argv])
    monkeypatch.setattr(coordinator,'coordinate',lambda args,raw:calls.append((args,raw)))
    original_import=builtins.__import__
    def forbidden_native(name,*args,**kwargs):
        if name.split('.')[0] in ('isaaclab','isaacsim','carb','omni'):
            raise AssertionError('协调器不得导入原生库：'+name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',forbidden_native)
    module.main()
    assert len(calls)==1 and calls[0][1]==argv
    assert calls[0][0].bank=='bank.json' and calls[0][0].device=='cuda:0'


def test_coordinator_rejects_unknown_flags_before_work(tmp_path,monkeypatch):
    import importlib.util,sys
    import native_cache_coordinator as coordinator
    script=Path(coordinator.__file__).with_name('prepare_native_cache.py')
    spec=importlib.util.spec_from_file_location('pure_cache_bad_flags',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'argv',[str(script),'--bank','bank.json','--cache-root','cache',
        '--output-dir','result','--unknown-setting','x'])
    monkeypatch.setattr(coordinator,'coordinate',lambda *args:pytest.fail('不能执行未知参数'))
    with pytest.raises(SystemExit) as error:module.main()
    assert error.value.code==2


def test_coordinator_import_guard_blocks_native_components_only():
    import importlib.util
    import native_cache_coordinator as coordinator
    script=Path(coordinator.__file__).with_name('prepare_native_cache.py')
    spec=importlib.util.spec_from_file_location('cache_import_guard_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    guard=module.CoordinatorImportGuard()
    for name in ('isaaclab.app','isaaclab_tasks','isaacsim','carb','omni.usd','pxr.Usd'):
        with pytest.raises(ImportError):guard.find_spec(name)
    assert guard.find_spec('numpy') is None and guard.find_spec('sklearn') is None


def test_validation_reads_each_asset_file_once(tmp_path,monkeypatch,capsys):
    root,cache,bank,usd=make_cache(tmp_path)
    original=Path.read_bytes;reads=[]
    def counted(path):
        reads.append(path.resolve())
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',counted)
    validate_cache(cache,bank)
    proof=json.loads((cache/'completed.json').read_text())
    for name in proof['assets'][0]['files_sha256']:
        assert reads.count((usd.parent/name).resolve())==1
    assert 'CACHE_VALIDATE_PROGRESS 1 / 1' in capsys.readouterr().out


def test_selected_validation_does_not_parse_unselected_urdfs(tmp_path,monkeypatch):
    import hybrid_assets
    root,cache,bank,usd=make_cache(tmp_path)
    data=json.loads(bank.read_text())
    data['designs'].append(dict(id='unused',urdf=str(tmp_path/'must-not-open.urdf'),urdf_sha256='unused'))
    bank.write_text(json.dumps(data))
    proof=json.loads((cache/'completed.json').read_text())
    proof['bank_sha256']=sha(bank)
    proof['assets'].append(dict(design='unused'))
    (cache/'completed.json').write_text(json.dumps(proof))
    validate_cache(cache,bank,['one'])


@pytest.mark.parametrize('mismatch',[False,True])
def test_fresh_coordinator_reports_finalize_without_second_asset_read(tmp_path,monkeypatch,mismatch):
    import hybrid_native_cache as native
    root=tmp_path/'run';root.mkdir();(root/'attempts').mkdir()
    bank=tmp_path/'bank.json';bank.write_text(json.dumps({'designs':[{'id':'one'}]}))
    def child(command,folder,environment):
        cache=Path(command[command.index('--cache-root')+1]);cache.mkdir()
        result=Path(command[command.index('--output-dir')+1]);result.mkdir(parents=True)
        proof=dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,bank_sha256=sha(bank),assets=[{'design':'one'}])
        (cache/'completed.json').write_text(json.dumps(proof))
        if mismatch:proof=dict(proof,bank_sha256='wrong')
        (result/'summary.json').write_text(json.dumps(proof))
    monkeypatch.setattr(native,'validate_cache',lambda *a:pytest.fail('must not reread just-verified USD'))
    if mismatch:
        with pytest.raises(ValueError,match='reports disagree'):ensure_cache(root,bank,'python','',{},child)
    else:assert (ensure_cache(root,bank,'python','',{},child)/'completed.json').is_file()


def test_plan_checkpoint_scan_prunes_native_and_proven_conversion_trees(tmp_path,monkeypatch):
    import refresh_cache_plan as refresh
    (tmp_path/'native_cache_old/design/configuration').mkdir(parents=True)
    folder=tmp_path/'attempts/cache_old/train';folder.mkdir(parents=True)
    (folder/'launch.json').write_text(json.dumps({'command':['python','prepare_native_cache.py']}))
    (folder/'result/huge_assets').mkdir(parents=True)
    visited=[]
    original=refresh.os.scandir
    def guarded(path):
        path=Path(path)
        assert 'native_cache_old' not in path.parts
        assert 'cache_old' not in path.parts or path==tmp_path/'attempts'
        visited.append(path)
        return original(path)
    monkeypatch.setattr(refresh.os,'scandir',guarded)
    assert not refresh.checkpoint_evidence(tmp_path)
    assert tmp_path/'attempts' in visited


def test_plan_checkpoint_scan_keeps_unknown_locations_and_attempts(tmp_path):
    import refresh_cache_plan as refresh
    p=tmp_path/'attempts/cache_unproven/model.pt';p.parent.mkdir(parents=True);p.write_text('evidence')
    assert refresh.checkpoint_evidence(tmp_path)
    p.unlink()
    q=tmp_path/'unknown_models/model.pt';q.parent.mkdir();q.write_text('evidence')
    assert refresh.checkpoint_evidence(tmp_path)


@pytest.mark.parametrize('change_unchanged_converter',[False,True])
def test_server_actual_unchanged_importer_identity_is_not_a_migration(tmp_path,monkeypatch,change_unchanged_converter):
    import refresh_cache_plan as refresh
    from copy import deepcopy
    # 服务器计划与实际文件逐字节核实得到的真实源身份；不是临时关闭校验。
    old_sources={
        'prepare_native_cache.py':'fa2c445c5190002375ccbf6014f728521ac4a4bc19b0fafa253d0091f69830f1',
        'native_cache_coordinator.py':'48682f956f90326b9bee272e6c14767d96735840ab7ce6895dd2f12ce0724029',
        'hybrid_assets.py':'c9eab750c7a712966572623fd1aa22c29b579f721741f23efa81d979794f0898',
        'hybrid_native_cache.py':'03153f0b0e568439f295761903c68e6a1c06201cea5dd97893386d44ad49bad0',
        'train.py':'24f2c69b1f41f726cc4fe4726278dfa135d3e94f1f39329f1618bc4c8230760c',
        'refresh_cache_plan.py':'9d5f2088a3ccfb6e4d47de8b1bbcce9268c02a433724dbb2ee06393c1790854a'}
    new_sources=dict(old_sources)
    for name in new_sources:
        if name!='prepare_native_cache.py':new_sources[name]='updated-reviewed-source'
    if change_unchanged_converter:
        old_sources['prepare_native_cache.py']='unknown-unreviewed-source'
        new_sources['prepare_native_cache.py']='changed-importer'
    old=dict(native_sources={'physics':'same'},hybrid_sources=old_sources)
    new=dict(native_sources={'physics':'same'},hybrid_sources=new_sources)
    monkeypatch.setattr(refresh,'experiment_identity',lambda bank,config:new)
    plan=dict(experiment=old,bank='fixture',configuration={})
    if change_unchanged_converter:
        with pytest.raises(ValueError,match='prepare_native_cache'):refresh.refreshed_plan(tmp_path,plan)
    else:assert refresh.refreshed_plan(tmp_path,plan)['experiment']==new


def test_concise_mesh_progress_never_prints_file_or_design_names(tmp_path,capsys):
    from hybrid_assets import mesh_sources
    root,cache,bank,usd=make_cache(tmp_path)
    parsed=json.loads(bank.read_text())
    parsed['designs'].append(dict(parsed['designs'][0],id='second'))
    actual=mesh_sources(parsed)
    output=capsys.readouterr().out
    assert 'CACHE_URDF_PROGRESS 2 / 2' in output
    assert 'CACHE_MESH_PROGRESS 1 / 1' in output
    assert str(tmp_path) not in output and 'second' not in output
    assert actual==json.loads((cache/'completed.json').read_text())['mesh_sources']


def test_concise_asset_validation_retains_corrupt_file_in_error(tmp_path,capsys):
    root,cache,bank,usd=make_cache(tmp_path)
    (usd.parent/'geometry.usd').write_text('corrupt')
    with pytest.raises(ValueError,match='geometry.usd'):validate_cache(cache,bank)
    output=capsys.readouterr().out
    assert 'CACHE_VALIDATE_FILE_BEGIN' not in output
    assert 'CACHE_VALIDATE_ASSET_BEGIN' not in output


def test_training_fast_check_never_reads_usd_contents_and_survives_hardlink(tmp_path,monkeypatch,capsys):
    import hybrid_native_cache as native
    import os
    root,cache,bank,usd=make_cache(tmp_path)
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.suffix.lower()=='.usd':raise AssertionError('formal fast path must not read USD bytes')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    proof=native.validate_training_cache(cache,bank,['one'])
    assert len(proof['assets'])==1
    os.link(usd,tmp_path/'scene_link.usd')
    native.validate_training_cache(cache,bank,['one'])
    assert 'TRAIN_CACHE_FAST_PROGRESS 1 / 1' in capsys.readouterr().out


@pytest.mark.parametrize('fault',['missing','size','mtime','config','converter','urdf','mesh','proof'])
def test_training_fast_rejects_missing_or_changed_assets(tmp_path,fault):
    import hybrid_native_cache as native
    import os
    root,cache,bank,usd=make_cache(tmp_path)
    native.validate_training_cache(cache,bank,['one'])
    if fault=='missing':(usd.parent/'geometry.usd').unlink()
    if fault=='size':usd.write_text('different-sized data')
    if fault=='mtime':
        st=usd.stat();os.utime(usd,ns=(st.st_atime_ns,st.st_mtime_ns+1000000000))
    if fault=='config':(usd.parent/'config.yaml').write_text('changed config')
    if fault=='converter':(usd.parent/'.asset_hash').write_text('changed')
    if fault=='urdf':(tmp_path/'robot.urdf').write_text('<robot/>')
    if fault=='mesh':(tmp_path/'mesh.stl').write_text('changed mesh size')
    if fault=='proof':
        p=cache/'completed.json';d=json.loads(p.read_text());d['bank_sha256']='wrong';p.write_text(json.dumps(d))
    with pytest.raises((ValueError,OSError)):native.validate_training_cache(cache,bank,['one'])


def test_training_fast_does_not_inspect_unselected_design_or_change_deep_default(tmp_path):
    import hybrid_native_cache as native
    root,cache,bank,usd=make_cache(tmp_path)
    data=json.loads(bank.read_text());data['designs'].append(dict(id='unused',urdf='must-not-read'))
    bank.write_text(json.dumps(data));p=cache/'completed.json';proof=json.loads(p.read_text())
    proof['bank_sha256']=sha(bank);proof['assets'].append(dict(design='unused'));p.write_text(json.dumps(proof))
    assert native.cache_directories(cache,bank,['one'],require_complete=True,training_fast=True)==[usd.parent.resolve()]
    (usd.parent/'geometry.usd').write_text('corrupt')
    # 独立转换/完整审计默认依然检查真实内容。
    with pytest.raises(ValueError,match='geometry dependency'):native.cache_directories(cache,bank,['one'])


def test_training_runtime_routes_fast_mode_only_for_execute():
    import ast
    root=Path(__file__).resolve().parents[1]
    for p in [root/'engine/hybrid_runtime.py',root/'amp_engine/amp_mixed_runtime.py']:
        tree=ast.parse(p.read_text(encoding='utf8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))
               and (n.func.id if isinstance(n.func,ast.Name) else n.func.attr)=='cache_directories']
        assert len(calls)==1
        flag=next(k.value for k in calls[0].keywords if k.arg=='training_fast')
        assert ast.unparse(flag)=="bool(getattr(options, 'execute', False))"


def test_resume_receipt_survives_restart_without_reading_usd(tmp_path,monkeypatch):
    import hybrid_assets
    root,cache,bank,usd=make_cache(tmp_path)
    row=json.loads((cache/'completed.json').read_text())['assets'][0]
    hybrid_assets.validate_asset_resume(row)
    original=hybrid_assets.sha
    def no_usd(path):
        if Path(path).suffix=='.usd':raise AssertionError('unchanged USD must not be reread')
        return original(path)
    monkeypatch.setattr(hybrid_assets,'sha',no_usd)
    hybrid_assets.validate_asset_resume(row)
    (usd.parent/'geometry.usd').unlink()
    with pytest.raises(OSError):hybrid_assets.validate_asset_resume(row)


def test_resume_receipt_does_not_hide_changed_asset(tmp_path):
    from hybrid_assets import validate_asset_resume
    root,cache,bank,usd=make_cache(tmp_path)
    row=json.loads((cache/'completed.json').read_text())['assets'][0]
    validate_asset_resume(row)
    usd.write_text('changed content')
    with pytest.raises(ValueError):validate_asset_resume(row)
