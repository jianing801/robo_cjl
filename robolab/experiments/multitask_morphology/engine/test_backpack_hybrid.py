"""Backpack pool alignment, attachment isolation and absolute curriculum."""
import json
from pathlib import Path
import torch
import pytest
from hybrid_payload import payload_range,payload_segments

def test_payload_is_absolute_not_restarted_by_group():
    assert payload_range(999)==(.1,.5)
    assert payload_range(1000)==(.1,4.125)
    assert payload_range(2999)==(.1,15.)
    assert payload_range(3000)==(.1,15.)
    assert list(payload_segments(999,1002))==[(999,1000,(.1,.5)),(1000,1002,(.1,4.125))]
    assert list(payload_segments(4000,4001))==[(4000,4001,(.1,15.))]

def test_all_three_configs_have_identical_complete_pools():
    here=Path(__file__).resolve().parent.parent
    configs=[json.loads((here/'configs'/f'{task}.json').read_text())
             for task in ('rough','amp','backpack')]
    assert all(c['grid']==configs[0]['grid'] and c['full_body_torque'] for c in configs)
    assert configs[2]['grid']['knee_motor_count']==5 and configs[2]['grid']['ankle_motor_count']==5
    assert set(configs[2]['grid']['ankle_pool'])=={
        'RS06','DM-J4340P-2EC','DM-J4340P-2EC-V1.1-48V','DM-J6216P-2EC','DM-J8009P-2EC-V1.1'}

def test_attachment_reuses_base_without_mutation(tmp_path,monkeypatch):
    import hybrid_backpack_bank as module
    import xml.etree.ElementTree as ET
    source=tmp_path/'base.urdf';source.write_text('<robot name="test"><link name="torso_link"><inertial><origin xyz="0 0 0"/><mass value="10"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>')
    rough_path=tmp_path/'bank.json';rough_path.write_text('{}')
    original=source.read_bytes();grid={'test':1}
    rough=dict(task_key='stairs',configuration=grid,training={'authorized_tasks':['stairs']},
        designs=[dict(id='one',urdf=str(source),total_mass_kg=10.,context=[1.,2.])])
    monkeypatch.setattr(module,'load_configurable_bank',lambda path:rough)
    monkeypatch.setattr(module,'source_identity',lambda task:{'task':task})
    bank=module.adopt_rough_bank(rough_path,grid,tmp_path/'new')
    assert source.read_bytes()==original and rough['task_key']=='stairs'
    assert bank['task_key']=='backpack' and bank['designs'][0]['context']==[1.,2.]
    attached=ET.parse(bank['designs'][0]['urdf'])
    assert attached.find("link[@name='backpack_link']") is not None
    assert bank['designs'][0]['total_mass_kg']==10.3
    with pytest.raises(ValueError):module.adopt_rough_bank(rough_path,{},tmp_path/'bad')
    assert not (tmp_path/'bad').exists()

def test_backpack_policy_keeps_role_in_checkpoint_identity():
    from hybrid_policy import SpatialActorCritic
    from tensordict import TensorDict
    obs=TensorDict({'policy':torch.zeros(8,49),'critic':torch.zeros(8,49)},batch_size=[8])
    native=dict(role='backpack',context_width=29,spatial_identity='test')
    p=SpatialActorCritic(obs,{'policy':['policy'],'critic':['critic']},23,native_config=native,
        actor_hidden_dims=[16],critic_hidden_dims=[16],actor_obs_normalization=False,critic_obs_normalization=False)
    assert p.get_extra_state()['config']['role']=='backpack'
    bad=p.get_extra_state();bad['config']['role']='stairs'
    with pytest.raises(ValueError):p.set_extra_state(bad)
