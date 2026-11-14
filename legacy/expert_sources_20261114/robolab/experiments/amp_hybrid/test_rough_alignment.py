"""Cross-task physical identity and imported Rough reliability regressions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from amp_rough_alignment import (SHARED_FIELDS, adopt_rough_bank,
    rough_module, validate_alignment, validate_bound_bank)


def banks(tmp_path, monkeypatch):
    import amp_rough_alignment as module
    import amp_support
    grid=json.loads((HERE/'config.local900.json').read_text())['grid']
    rough={key: {'value':key} for key in SHARED_FIELDS}
    rough.update(kind='hybrid_rough_grid_bank_v1',task_key='stairs',
        configuration=grid,designs=[dict(id='shared',context=[0.]*24)],
        context_width=24,training=dict(authorized_tasks=['stairs']))
    path=tmp_path/'rough.json';path.write_text(json.dumps(rough))
    monkeypatch.setattr(module,'rough_module',lambda name:SimpleNamespace(load_configurable_bank=lambda p:rough))
    monkeypatch.setattr(amp_support,'source_identity',lambda task:dict(amp='unchanged'))
    amp=adopt_rough_bank(path,grid)
    return rough,amp,path


def test_config_grid_ids_encoding_and_schedule_match_latest_rough():
    from amp_bank import select_grid
    from amp_contract import cycle_groups
    amp=json.loads((HERE/'config.local900.json').read_text())
    rough=json.loads((HERE.parent/'rough_hybrid/config.local900.json').read_text())
    assert amp['grid']==rough['grid']
    assert 'DM-J6216P-2EC' in amp['grid']['ankle_pool']
    assert 'DM-J4310P-2EC-48V' not in amp['grid']['ankle_pool']
    assert amp['rotation']['total_updates']==5000
    grid=select_grid(amp['grid'],amp['grid']['knee_pool'],amp['grid']['ankle_pool'])
    other=rough_module('hybrid_bank').select_grid(rough['grid'],rough['grid']['knee_pool'],rough['grid']['ankle_pool'])
    assert grid==other
    bank=dict(designs=grid['designs'])
    for cycle in range(4):
        assert cycle_groups(bank,amp,cycle)==rough_module('hybrid_contract').cycle_groups(bank,rough,cycle)


def test_adoption_changes_only_task_identity_and_never_rough(tmp_path,monkeypatch):
    rough,amp,path=banks(tmp_path,monkeypatch)
    from amp_bank import KIND
    assert amp['kind']==KIND and amp['task_key']=='run'
    assert amp['training']['retargeted_references'] is True
    assert rough['task_key']=='stairs'
    assert all(amp[key]==rough[key] for key in SHARED_FIELDS)
    assert amp['designs'] is not rough['designs']
    assert validate_bound_bank(amp)['verified']


@pytest.mark.parametrize('field',SHARED_FIELDS)
def test_each_morphology_field_is_strict(tmp_path,monkeypatch,field):
    rough,amp,path=banks(tmp_path,monkeypatch)
    amp[field]=None
    with pytest.raises(ValueError,match='morphology differs'):
        validate_alignment(amp,rough,path)


def test_changed_rough_manifest_rejected(tmp_path,monkeypatch):
    rough,amp,path=banks(tmp_path,monkeypatch)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Rough bank changed'):
        validate_bound_bank(amp)


def test_adoption_grid_drift_rejected(tmp_path,monkeypatch):
    rough,amp,path=banks(tmp_path,monkeypatch)
    changed=deepcopy(rough['configuration']);changed['num_envs']=2048
    with pytest.raises(ValueError,match='grid must exactly match'):
        adopt_rough_bank(path,changed)


def test_shared_helper_code_is_part_of_experiment_identity(tmp_path,monkeypatch):
    import amp_contract,amp_support,amp_rough_alignment
    monkeypatch.setattr(amp_support,'source_identity',lambda _:dict(source='same'))
    bank=tmp_path/'bank';bank.write_text('{}')
    refs=tmp_path/'refs';refs.write_text('{}')
    before=amp_contract.experiment_identity(bank,{},refs)
    helpers=deepcopy(before['shared_rough_runtime_helpers'])
    helpers['hybrid_resources.py']='changed'
    monkeypatch.setattr(amp_rough_alignment,'shared_runtime_sources',lambda:helpers)
    assert amp_contract.experiment_identity(bank,{},refs)!=before


def test_controller_zero_exit_records_effective_failure(tmp_path):
    spec=importlib.util.spec_from_file_location('amp_failed_child_test',HERE/'train.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    folder=tmp_path/'attempts/session/train';folder.parent.mkdir(parents=True)
    command=[sys.executable,'-c',f"from pathlib import Path;p=Path({str(folder/'result')!r});p.mkdir();(p/'cleanup_failure.json').write_text('original CUDA error')"]
    import os
    with pytest.raises(RuntimeError,match='AMP worker failed'):
        controller.run_child(command,folder,os.environ)
    result=json.loads((folder/'exit.json').read_text())
    assert result['raw_returncode']==0 and result['returncode']==1
    assert result['failure_recorded'] and not result['summary_exists']


def test_merged_base_remap_changes_only_selectors():
    from amp_scene_contract import align_merged_base_events
    mass=SimpleNamespace(params=dict(asset_cfg=SimpleNamespace(body_names='base_link'),
        mass_distribution_params=(-3.,3.),operation='add'))
    com=SimpleNamespace(params=dict(asset_cfg=SimpleNamespace(body_names=['torso_link','base_link']),
        com_range={'x':(-.03,.03)}))
    cfg=SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(merge_fixed_joints=True))),
        events=SimpleNamespace(add_base_mass=mass,randomize_rigid_body_com=com))
    remap=align_merged_base_events(cfg)
    assert set(remap)=={'add_base_mass','randomize_rigid_body_com'}
    assert mass.params['asset_cfg'].body_names==['torso_link']
    assert com.params['asset_cfg'].body_names==['torso_link']
    assert mass.params['mass_distribution_params']==(-3.,3.) and mass.params['operation']=='add'
    assert com.params['com_range']=={'x':(-.03,.03)}
    assert align_merged_base_events(cfg)=={}
    cfg.scene.robot.spawn.merge_fixed_joints=False
    mass.params['asset_cfg'].body_names='base_link'
    assert align_merged_base_events(cfg)=={} and mass.params['asset_cfg'].body_names=='base_link'
