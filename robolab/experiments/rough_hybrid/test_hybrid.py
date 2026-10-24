"""CPU regression tests: no Kit, GPU, asset generation or long training."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT.parent/'rsl_rl')]
from hybrid_contract import cycle_groups,stage_spec,validate_config,validate_checkpoint
from hybrid_layout import make_layout,identity,balanced_indices
from hybrid_bank import select_grid

def bank(n=37):
    return dict(configuration={'num_envs':16},context_width=22,ranges_m={'thigh':[0.25,0.4],'calf':[0.3,0.45]},
        designs=[dict(id=str(i),split='train',thigh_m=0.25+0.15*(i%10)/9,calf_m=0.3+0.15*((i//10)%10)/9,
            knee_motor='k'+str(i%2),ankle_motor='a'+str((i//2)%2),context=[float(i)]+[0.]*21,
            graph={'joint_names':['knee','ankle']}) for i in range(n)]+[dict(id='heldout',split='holdout')])
def config(b):return dict(grid=b['configuration'],rotation=dict(active_designs=8,updates_per_stage=7,total_updates=81,seed=42))

def test_full_library_can_exceed_live_envs():
    g=select_grid(dict(num_envs=2,length_count=3,thigh_range_m=[0.25,0.4],calf_range_m=[0.3,0.45],
        knee_motor_count=2,ankle_motor_count=2,motor_seed=42),['k0','k1'],['a0','a1'])
    assert g['design_count']==36 and g['num_envs']==2

def test_cycles_cover_every_design_once_deterministically():
    b=bank();c=config(b);validate_config(c,b)
    previous=None
    for cycle in range(5):
        groups=cycle_groups(b,c,cycle);flat=sum(groups,[])
        assert sorted(flat,key=int)==[str(i) for i in range(37)]
        assert max(map(len,groups))<=8 and max(map(len,groups))-min(map(len,groups))<=1
        assert groups==cycle_groups(b,c,cycle)
        if previous is not None:assert groups!=previous
        previous=groups

def test_stage_budget_and_reentry():
    b=bank();c=config(b)
    specs=[stage_spec(b,c,i) for i in range(12)]
    assert specs[0]['first_update']==0 and specs[-1]['end_update']==81
    assert sum(s['end_update']-s['first_update'] for s in specs)==81
    assert specs[5]['cycle']==1
    with pytest.raises(ValueError):stage_spec(b,c,12)

@pytest.mark.parametrize('field',['active_designs','updates_per_stage','total_updates'])
def test_required_unset_numbers_fail(field):
    b=bank();c=config(b);c['rotation'][field]=None
    with pytest.raises(ValueError):validate_config(c,b)

def test_active_count_not_full_library_is_limited_by_envs():
    b=bank(3500);c=config(b);validate_config(c,b)
    c['rotation']['active_designs']=17
    with pytest.raises(ValueError):validate_config(c,b)

def test_global_context_not_reindexed_for_subset():
    b=bank();c=config(b);groups=cycle_groups(b,c,0)
    layout=make_layout(b,16,groups[0])
    assert layout['context_width']==22
    for key in layout['design_ids']:
        assert next(d for d in b['designs'] if d['id']==key)['context'][0]==float(key)
    assert max(layout['counts'].values())-min(layout['counts'].values())<=1
    with pytest.raises(KeyError):make_layout(b,16,['heldout'])

def test_each_onpolicy_sample_used_once_after_group_change():
    for n in (3,7,8):
        groups=torch.arange(16)%n
        indices=balanced_indices(groups,24,4)
        assert torch.equal(torch.cat(indices).sort().values,torch.arange(16*24))

def runner_class():
    # Compile the actual copied runner class without importing the Isaac-only scene module.
    tree=ast.parse((HERE/'hybrid_runtime.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpatialRunner')
    namespace=dict(torch=torch,Path=Path,OnPolicyRunner=object)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'hybrid_runtime.py','exec'),namespace)
    return namespace['SpatialRunner']

class TinyPolicy(torch.nn.Module):
    def __init__(self,experiment):
        super().__init__();self.actor=torch.nn.Linear(3,2);self.critic=torch.nn.Linear(3,1)
        self.register_buffer('normalization_mean',torch.tensor([0.1,0.2,0.3]))
        self.experiment=experiment
    def get_extra_state(self):return dict(config=dict(role='stairs',spatial_identity=identity(self.experiment)))
    def set_extra_state(self,value):
        if value!=self.get_extra_state():raise ValueError('wrong policy identity')

def new_runner(experiment,stage):
    runner=runner_class()();policy=TinyPolicy(experiment)
    runner.alg=SimpleNamespace(policy=policy,optimizer=torch.optim.Adam(policy.parameters(),lr=0.0003),learning_rate=0.0003)
    runner.device='cpu';runner.hybrid_identity=experiment;runner.hybrid_stage=stage
    return runner

def test_three_stage_exact_restore_normalization_adam_and_live_gradients(tmp_path):
    experiment={'kind':'test_hybrid','bank_sha256':'fixed'}
    previous=None
    for index in range(3):
        stage=dict(index=index,first_update=index*2,end_update=index*2+2)
        runner=new_runner(experiment,stage)
        if previous:
            loaded=validate_checkpoint(previous,experiment,stage['first_update'])
            runner.restore_spatial(previous)
            for k,v in runner.alg.policy.state_dict().items():
                if isinstance(v,torch.Tensor):assert torch.equal(v,loaded['model_state_dict'][k])
            assert runner.alg.optimizer.state_dict()['state'].keys()==loaded['optimizer_state_dict']['state'].keys()
            for key,slots in loaded['optimizer_state_dict']['state'].items():
                for name,value in slots.items():assert torch.equal(value,runner.alg.optimizer.state_dict()['state'][key][name])
            assert runner.current_learning_iteration==stage['first_update']
        before=runner.alg.policy.actor.weight.detach().clone()
        for it in range(stage['first_update'],stage['end_update']):
            runner.alg.optimizer.zero_grad()
            x=torch.randn(8,3)
            loss=runner.alg.policy.actor(x).square().mean()+runner.alg.policy.critic(x).square().mean()
            loss.backward();assert runner.alg.policy.actor.weight.grad is not None
            runner.alg.optimizer.step();runner.current_learning_iteration=it
        assert not torch.equal(before,runner.alg.policy.actor.weight)
        previous=tmp_path/f'model_{stage["end_update"]-1}.pt';runner.save(previous)
        state=validate_checkpoint(previous,experiment,stage['end_update'])
        assert all(v['step'].item()==stage['end_update'] for v in state['optimizer_state_dict']['state'].values())
    with pytest.raises(ValueError):validate_checkpoint(previous,{'kind':'other'},6)
    with pytest.raises(ValueError):validate_checkpoint(previous,experiment,7)
    broken=deepcopy(state);next(iter(broken['optimizer_state_dict']['state'].values()))['exp_avg'].fill_(float('nan'))
    bad=tmp_path/'bad.pt';torch.save(broken,bad)
    with pytest.raises(ValueError):validate_checkpoint(bad,experiment,6)

def test_copy_provenance_and_runtime_isolation():
    # Historical copy hashes document origin, not a dependency on uncommitted
    # versions of the original AMP/backpack/spatial entry points.
    manifest=json.loads((HERE/'COPY_PROVENANCE.json').read_text())
    for row in manifest['files']:
        assert len(row['source_sha256'])==64 and (HERE/row['copy']).is_file()
    forbidden={'robolab.rough_contract','robolab.configurable_morphology',
               'robolab.spatial_policy','spatial_rough_runtime','train_spatial_expert'}
    for path in HERE.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):assert node.module not in forbidden

def test_source_identity_is_rough_only():
    from hybrid_support import source_identity, extend_agent
    sources=source_identity()
    assert sources and all((ROOT/p).is_file() for p in sources)
    assert not any('scripts/tools/' in p or 'spatial_expert' in p for p in sources)
    agent={'policy':{},'algorithm':{'optimizer':'adam','share_cnn_encoders':False,
                                 'symmetry_cfg':{'data_augmentation_func':'original'}}}
    result=extend_agent(agent,bank_sha='test',context_width=24)
    assert result['policy']['class_name']=='hybrid_policy:SpatialActorCritic'
    assert agent['algorithm']['optimizer']=='adam'
    with pytest.raises(ValueError):source_identity('run')

def test_worker_schedule_not_overwritten_by_usd_stage():
    tree=ast.parse((HERE/'train_stage.py').read_text())
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='stage' for t in n.targets)]
    assert len(assignments)==1 and isinstance(assignments[0].value,ast.Call) and assignments[0].value.func.id=='stage_spec'

def test_controller_stage_commit_failure_and_resume(tmp_path,monkeypatch):
    import importlib.util
    from contextlib import nullcontext
    import psutil
    import hybrid_bank
    import robolab.rough_gpu as gpu
    spec=importlib.util.spec_from_file_location('hybrid_test_controller',HERE/'train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    b=bank(9);c=config(b);c['rotation'].update(active_designs=4,updates_per_stage=2,total_updates=5)
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(c))
    bank_path=tmp_path/'bank.json';bank_path.write_text(json.dumps(b))
    root=tmp_path/'run'
    monkeypatch.setattr(module,'load_configurable_bank',lambda p:b)
    monkeypatch.setattr(hybrid_bank,'load_configurable_bank',lambda p:b)
    monkeypatch.setattr(psutil,'process_iter',lambda *a,**kw:[])
    monkeypatch.setattr(gpu,'query_gpus',lambda:[])
    monkeypatch.setattr(gpu,'candidates',lambda *a,**kw:[{'index':0,'uuid':'test'}])
    monkeypatch.setattr(gpu,'gpu_lease',lambda *a,**kw:nullcontext({'index':0,'uuid':'test'}))
    monkeypatch.setattr(gpu,'child_environment',lambda _,environment:dict(environment))
    monkeypatch.setattr(gpu,'reject_duplicate_bank',lambda *a,**kw:None)
    calls=[];fail=[True]
    def child(command,folder,env):
        assert '--preflight-only' not in command and '--preflight-report' not in command
        index=int(command[command.index('--stage-index')+1]);mode='preflight' if '--preflight-only' in command else 'train'
        calls.append((index,mode))
        if index==1 and mode=='train' and fail[0]:
            folder.mkdir();(folder/'launch.json').write_text('{}');raise RuntimeError('injected stage interruption')
        out=Path(command[command.index('--output-dir')+1]);out.mkdir(parents=True)
        plan=json.loads((root/'plan.json').read_text());stage=stage_spec(b,c,index)
        if mode=='preflight':summary=dict(result='HYBRID_TWO_UPDATE_PREFLIGHT_PASSED_DISCARD_POLICY')
        else:
            runner=new_runner(plan['experiment'],stage)
            if '--resume-checkpoint' in command:
                path=command[command.index('--resume-checkpoint')+1];runner.restore_spatial(path)
                assert runner.current_learning_iteration==stage['first_update']
            for it in range(stage['first_update'],stage['end_update']):
                runner.alg.optimizer.zero_grad();x=torch.randn(4,3)
                (runner.alg.policy.actor(x).square().mean()+runner.alg.policy.critic(x).square().mean()).backward()
                runner.alg.optimizer.step();runner.current_learning_iteration=it
            (out/'models').mkdir();runner.save(out/f'models/model_{stage["end_update"]-1}.pt')
            summary=dict(result='HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED',completed_updates=stage['end_update'])
        (out/'summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(module,'run_child',child)
    args=['--config',str(config_path),'--bank',str(bank_path),'--output-root',str(root)]
    module.main(args) # planning does not invoke a worker
    assert calls==[]
    with pytest.raises(RuntimeError,match='injected'):module.main(args+['--resume','--execute'])
    assert len(list((root/'commits').glob('stage_*.json')))==1
    assert not (root/'completed.json').exists()
    fail[0]=False
    module.main(args+['--resume','--execute'])
    assert json.loads((root/'completed.json').read_text())['completed_updates']==5
    assert len(list((root/'commits').glob('stage_*.json')))==3
    assert calls==[(0,'train'),(1,'train'),(1,'train'),(2,'train')]
    assert json.loads((root/'plan.json').read_text())['preflight_updates_per_new_layout']==0
    saved=list(calls);module.main(args+['--resume','--execute']);assert calls==saved
    (root/'USER_PAUSED').touch()
    with pytest.raises(RuntimeError,match='USER_PAUSED'):module.main(args+['--resume','--execute'])

def test_template_cannot_implicitly_start_a_training():
    c=json.loads((HERE/'config.template.json').read_text())
    with pytest.raises(ValueError):validate_config(c,bank())


def test_child_logs_each_update_to_terminal_and_file(tmp_path,capsys):
    import importlib.util
    spec=importlib.util.spec_from_file_location('hybrid_tee_test',HERE/'train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=tmp_path/'attempts'/'stage_0'/'train'
    folder.parent.mkdir(parents=True)
    module.run_child([sys.executable,'-u','-c',
                      f"from pathlib import Path; p=Path({str(folder / 'result')!r}); p.mkdir(); (p/'summary.json').write_text('{{}}'); print('Learning iteration 0/12000',flush=True); print('Learning iteration 1/12000',flush=True)"],
                     folder, __import__('os').environ)
    output=capsys.readouterr().out
    assert 'Learning iteration 0/12000' in output and 'Learning iteration 1/12000' in output
    assert (folder/'stdout.log').read_text().count('Learning iteration')==2
    assert json.loads((folder/'exit.json').read_text())['returncode']==0


def test_zero_exit_with_failure_or_missing_summary_is_rejected(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('hybrid_exit_test',HERE/'train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for failure in (False, True):
        folder=tmp_path/str(failure)
        script="print('worker exiting zero')"
        if failure:
            script=f"from pathlib import Path; p=Path({str(folder/'result')!r}); p.mkdir(); (p/'failure.json').write_text('CUDA out of memory'); (p/'summary.json').write_text('{{}}')"
        with pytest.raises(RuntimeError,match='recorded failure|without required summary'):
            module.run_child([sys.executable,'-c',script],folder,__import__('os').environ)
        exit_record=json.loads((folder/'exit.json').read_text())
        assert exit_record['returncode']==1 and exit_record['raw_returncode']==0


def test_persistent_observation_contract_uses_tensordict_keys():
    from tensordict import TensorDict
    obs=TensorDict({'policy':torch.zeros(8,12),'critic':torch.zeros(8,20)},batch_size=[8])
    storage=TensorDict({'policy':torch.zeros(24,8,12),'critic':torch.zeros(24,8,20)},batch_size=[24,8])
    assert all(obs[k].shape==storage[k].shape[1:] for k in obs.keys())
    text=(HERE/'train_stage.py').read_text()
    assert 'for k in obs.keys()' in text and 'RayCaster.meshes.clear()' in text


def test_native_logger_global_target_without_training_changes():
    import importlib.util
    spec=importlib.util.spec_from_file_location('hybrid_logging_test',HERE/'train_stage.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[];runner=SimpleNamespace(logger=SimpleNamespace(log=lambda **v:calls.append(v)))
    module.attach_global_logger(runner,12000)
    for it in (0,49,50,949):
        runner.logger.log(it=it,total_it=it+1,learning_rate=1e-4,loss_dict={'value':1.})
    assert [v['it'] for v in calls]==[0,49,50,949]
    assert all(v['total_it']==12000 and v['learning_rate']==1e-4 for v in calls)


def test_worker_has_no_preflight_execution_path():
    source=(HERE/'train_stage.py').read_text()
    assert '--preflight-only' not in source and '--preflight-report' not in source


def test_process_exit_race_and_unrelated_cwd_not_read(tmp_path,monkeypatch):
    import psutil
    from hybrid_processes import bank_process_snapshots
    bank=tmp_path/'bank.json'
    def missing():raise FileNotFoundError('/proc/282642/cwd')
    def never():raise AssertionError('unrelated cwd must never be queried')
    rows={
        1:SimpleNamespace(cmdline=lambda:['python','other.py'],cwd=never),
        2:SimpleNamespace(cmdline=lambda:['python','train_spatial_rough.py','--bank','bank.json'],
                          cwd=missing,is_running=lambda:False),
        3:SimpleNamespace(cmdline=lambda:['python','train_spatial_rough.py','--bank',str(bank)],cwd=never),
        4:SimpleNamespace(cmdline=lambda:['python','train_spatial_rough.py','--bank=bank.json'],cwd=lambda:str(tmp_path))}
    monkeypatch.setattr(psutil,'pids',lambda:list(rows))
    monkeypatch.setattr(psutil,'Process',lambda pid:rows[pid])
    snapshots=list(bank_process_snapshots())
    assert [p.pid for p in snapshots]==[3,4]
    from robolab.rough_gpu import reject_duplicate_bank
    with pytest.raises(RuntimeError,match='same Rough bank'):
        reject_duplicate_bank(bank,processes=snapshots,own_tree=set())


def test_disappearing_process_constructor_does_not_abort_scan(monkeypatch):
    import psutil
    from hybrid_processes import bank_process_snapshots
    monkeypatch.setattr(psutil,'pids',lambda:[1,2])
    def process(pid):
        if pid==1:raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(cmdline=lambda:['python','other.py'])
    monkeypatch.setattr(psutil,'Process',process)
    assert list(bank_process_snapshots())==[]


def test_live_relative_bank_cwd_unavailable_fails_closed(monkeypatch):
    import psutil
    from hybrid_processes import bank_process_snapshots
    def missing():raise FileNotFoundError('/proc/42/cwd')
    monkeypatch.setattr(psutil,'pids',lambda:[42])
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(
        cmdline=lambda:['python','train_spatial_rough.py','--bank','bank.json'],
        cwd=missing,is_running=lambda:True))
    with pytest.raises(RuntimeError,match='Active Rough PID 42'):
        list(bank_process_snapshots())

def test_exact_900_group_uniformity_over_eight_cycles():
    from collections import Counter
    b=bank(1);b['configuration']={'num_envs':4096}
    b['designs']=[dict(id=f'{t}-{c}-{m}',split='train',thigh_m=0.25+0.15*t/9,calf_m=0.3+0.15*c/9,
        knee_motor=f'k{m//3}',ankle_motor=f'a{m%3}') for t in range(10) for c in range(10) for m in range(9)]
    cfg=config(b);cfg['rotation'].update(active_designs=30,updates_per_stage=50,total_updates=12000)
    lookup={d['id']:d for d in b['designs']}
    for cycle in range(8):
        groups=cycle_groups(b,cfg,cycle)
        assert len(groups)==30 and len(set(sum(groups,[])))==900
        assert len(sum(groups,[]))==900
        for group in groups:
            rows=[lookup[k] for k in group]
            assert set(Counter(d['thigh_m'] for d in rows).values())=={3}
            assert set(Counter(d['calf_m'] for d in rows).values())=={3}
            assert sorted(Counter((d['knee_motor'],d['ankle_motor']) for d in rows).values())==[3]*6+[4]*3
    assert stage_spec(b,cfg,239)['end_update']==12000
    assert stage_spec(b,cfg,239)['cycle']==7
