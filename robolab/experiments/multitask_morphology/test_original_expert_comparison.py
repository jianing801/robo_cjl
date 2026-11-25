import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from original_expert_comparison import singleton_config, comparison_scores, TASK_BUDGETS, latest_receipt
from singleton_native import check_single_stage, checkpoint_boundary, install_ledger


def test_singleton_config_removes_rotation_schedule():
    for task, budget in TASK_BUDGETS.items():
        design = {'id': 'tr_792ca9f47877', 'thigh_m': .25, 'calf_m': .30,
                  'knee_motor': 'DM-J10010L-2EC', 'ankle_motor': 'DM-J4340P-2EC', 'split': 'train'}
        bank = {'task_key': task, 'configuration': {'num_envs': 4096},
                'context_schema': 'leg_motor_inertia_v1', 'designs': [design]}
        config = singleton_config(bank, task)
        assert config['rotation'] == dict(active_designs=1, updates_per_stage=budget, total_updates=budget, seed=42)
        assert config['grid'] == bank['configuration'] and config['full_body_torque'] is True
        assert 'group_schedule' not in config
        with pytest.raises(ValueError): singleton_config(dict(bank, designs=[design, design]), task)


def test_partial_checkpoint_is_exact_single_stage_only():
    stage = {'first_update': 0, 'end_update': 12000, 'design_ids': ['tr_792ca9f47877']}
    state = {'iter': 1000, 'hybrid_stage': stage}
    check_single_stage(stage, 12000)
    assert checkpoint_boundary(state, stage) == 1001
    with pytest.raises(ValueError): checkpoint_boundary(dict(state, iter=12000), stage)
    with pytest.raises(ValueError): checkpoint_boundary(dict(state, hybrid_stage=dict(stage, design_ids=['other'])), stage)
    with pytest.raises(ValueError): check_single_stage(dict(stage, first_update=10), 12000)


def test_comparison_uses_frozen_ratios_and_preserves_failure():
    tasks = ('stairs', 'backpack', 'run')
    keys = ('cot','forward_rmse_mps','roll_pitch_rms_rad_s','knee_ankle_max_rated_ratio')
    norm = {'metrics': {t: {k: {'baseline': 2., 'weight': .25} for k in keys} for t in tasks},
            'task_weights': dict(stairs=.4,backpack=.3,run=.3)}
    def trial(x):
        return {'status':'success','observed': {'absolute_mechanical_cot': x, **{k:x for k in keys[1:]}}}
    student = {t: trial(2.) for t in tasks}; teacher = {t: trial(1.) for t in tasks}
    output = comparison_scores(student, teacher, norm)
    assert output['shared_objective'] == pytest.approx(1.)
    assert output['expert_objective'] == pytest.approx(.5)
    assert output['shared_vs_expert_percent'] == pytest.approx(100.)
    teacher['run']['status'] = 'failure'
    assert comparison_scores(student,teacher,norm)['expert_objective'] is None


def test_native_ledger_saves_after_update_and_never_trusts_partial_blob(tmp_path):
    stage={'first_update':0,'end_update':12000,'design_ids':['one']}
    output=tmp_path/'attempts'/'first'/'result';output.mkdir(parents=True)
    logger=SimpleNamespace(rewbuffer=[2.,4.],log=lambda **kw: None,log_dir=str(output))
    runner=SimpleNamespace(logger=logger,current_learning_iteration=0,save=lambda path,infos=None: Path(path).write_bytes(b'checkpoint'))
    install_ledger(runner,output,tmp_path,stage)
    logger.log(it=0,collect_time=2.,learn_time=1.,learning_rate=.001)
    runner.save(output/'model_0.pt')
    receipt=latest_receipt(tmp_path)
    assert receipt['completed_updates']==1
    assert '1,3.0,2,2.0,1.0,0.001' in (output/'native_mean_reward.csv').read_text()
    (output/'model_1000.pt').write_bytes(b'partial')
    assert latest_receipt(tmp_path)==receipt
    (output/'model_0.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError): latest_receipt(tmp_path)
