"""Production persistent-loop and explicit legacy upgrade CPU regressions."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_hybrid import bank,config,new_runner
from hybrid_contract import experiment_identity,stage_spec,validate_checkpoint,sha,load_plan


def setup(tmp_path,monkeypatch):
    import hybrid_bank
    import hybrid_processes
    b=bank(9);c=config(b);c['rotation'].update(active_designs=4,updates_per_stage=2,total_updates=5)
    bp=tmp_path/'bank.json';bp.write_text(json.dumps(b))
    monkeypatch.setattr(hybrid_bank,'load_configurable_bank',lambda _:b)
    monkeypatch.setattr(hybrid_processes,'reject_duplicate_bank_safely',lambda _:None)
    return b,c,bp


def update(r,stage):
    for it in range(stage['first_update'],stage['end_update']):
        r.alg.optimizer.zero_grad();x=torch.randn(4,3)
        (r.alg.policy.actor(x).square().mean()+r.alg.policy.critic(x).square().mean()).backward()
        assert all(p.grad is not None for p in r.alg.policy.parameters())
        r.alg.optimizer.step();r.current_learning_iteration=it


def fake_stage(b,c,root,fail=None):
    def main(args,session):
        i=int(args[args.index('--stage-index')+1])
        if fail is not None and fail[0]==i:raise RuntimeError('injected interruption')
        out=Path(args[args.index('--output-dir')+1]);out.mkdir(parents=True)
        plan=json.loads((root/'plan.json').read_text());stage=stage_spec(b,c,i)
        r=session.get('runner')
        if r is None:
            r=new_runner(plan['experiment'],stage);r.alg.storage=object()
            r.logger=SimpleNamespace(log_dir=str(out/'models'));(out/'models').mkdir()
            if '--resume-checkpoint' in args:r.restore_spatial(args[args.index('--resume-checkpoint')+1])
            session['runner']=r
        r.hybrid_stage=stage
        update(r,stage)
        r.save(Path(r.logger.log_dir)/f'model_{stage["end_update"]-1}.pt')
        (out/'summary.json').write_text(json.dumps(dict(result='HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED',completed_updates=stage['end_update'])))
    return main


def test_formal_loop_interrupt_resume_and_absolute_tail(tmp_path,monkeypatch):
    from persistent_worker import run_groups
    b,c,bp=setup(tmp_path,monkeypatch);root=tmp_path/'run';root.mkdir()
    (root/'commits').mkdir();(root/'attempts').mkdir()
    plan=dict(configuration=c,experiment=experiment_identity(bp,c),execution_protocol='persistent_v1')
    (root/'plan.json').write_text(json.dumps(plan))
    fail=[1]
    with pytest.raises(RuntimeError,match='injected'):
        run_groups(root,bp,root/'session1',stage_main=fake_stage(b,c,root,fail))
    assert len(list((root/'commits').glob('*.json')))==1
    assert not (root/'session1/summary.json').exists()
    run_groups(root,bp,root/'session2',stage_main=fake_stage(b,c,root))
    rows=[json.loads(x.read_text()) for x in sorted((root/'commits').glob('*.json'))]
    assert [x['stage']['end_update'] for x in rows]==[2,4,5]
    assert rows[1]['objects']==rows[2]['objects']
    state=validate_checkpoint(rows[-1]['checkpoint'],plan['experiment'],5)
    assert all(float(v['step'])==5 for v in state['optimizer_state_dict']['state'].values())
    (root/'USER_PAUSED').touch()
    # Already complete is read-only, never adds another group/budget.
    run_groups(root,bp,root/'session3',stage_main=lambda *a,**k:pytest.fail('unexpected training'))


def test_legacy_upgrade_preserves_states_and_resumes(tmp_path,monkeypatch):
    import upgrade_run
    from persistent_worker import run_groups
    b,c,bp=setup(tmp_path,monkeypatch)
    old=tmp_path/'old';old.mkdir();(old/'commits').mkdir();(old/'attempts').mkdir()
    e=experiment_identity(bp,c)
    e['hybrid_sources']=next(iter(json.loads((Path(upgrade_run.__file__).with_name('legacy_sources.json')).read_text()).values()))
    plan=dict(configuration=c,experiment=e,execution_protocol='direct_stage_v2')
    (old/'plan.json').write_text(json.dumps(plan));stage=stage_spec(b,c,0)
    r=new_runner(e,stage);update(r,stage);r.save(old/'model_1.pt')
    (old/'summary.json').write_text(json.dumps(dict(result='HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED',completed_updates=2)))
    (old/'commits/stage_000000.json').write_text(json.dumps(dict(stage=stage,checkpoint=str(old/'model_1.pt'),checkpoint_sha256=sha(old/'model_1.pt'),summary=str(old/'summary.json'))))
    old_hashes={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
    monkeypatch.setattr(upgrade_run.time,'sleep',lambda seconds:None)
    root=tmp_path/'upgraded';upgrade_run.upgrade(old,bp,root)
    assert old_hashes=={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
    new=load_plan(root/'plan.json',bp)
    assert new['start_stage']==1
    run_groups(root,bp,root/'session',stage_main=fake_stage(b,c,root))
    rows=sorted((root/'commits').glob('*.json'))
    assert [p.name for p in rows]==['stage_000001.json','stage_000002.json']
    state=validate_checkpoint(json.loads(rows[-1].read_text())['checkpoint'],new['experiment'],5)
    assert all(float(v['step'])==5 for v in state['optimizer_state_dict']['state'].values())
    with (root/'bootstrap.pt').open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='identity changed'):load_plan(root/'plan.json',bp)


def test_unknown_legacy_code_rejected(tmp_path,monkeypatch):
    import upgrade_run
    b,c,bp=setup(tmp_path,monkeypatch);old=tmp_path/'old';old.mkdir()
    e=experiment_identity(bp,c);e['hybrid_sources']={'unreviewed':'code'}
    (old/'plan.json').write_text(json.dumps(dict(configuration=c,experiment=e,execution_protocol='direct_stage_v2')))
    with pytest.raises(ValueError,match='reviewed b313646'):upgrade_run.upgrade(old,bp,tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_environment_reseed_does_not_reset_learning_rng(monkeypatch):
    from train_stage import capture_learning_rng,restore_learning_rng
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda _:None)
    torch.manual_seed(123);session={};capture_learning_rng(session)
    expected=torch.rand(12)
    torch.manual_seed(42);torch.rand(100)  # New scene seed and reset sampling.
    restore_learning_rng(session)
    assert torch.equal(torch.rand(12),expected)
