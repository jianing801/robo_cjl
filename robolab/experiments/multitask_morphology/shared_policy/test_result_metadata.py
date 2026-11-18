from copy import deepcopy
import pytest
import torch

import randomization
from randomization import configure_profile, rollout_metadata, assert_protocol_unchanged
from student import save_student, load_student
from test_randomization import native_config
from test_shared import model_and_input


@pytest.mark.parametrize('task', ['stairs','backpack','run'])
@pytest.mark.parametrize('profile', ['nominal','robust'])
def test_results_separate_requested_values_from_applied_ranges(task, profile):
    _, audit = configure_profile(native_config(task), task, profile=profile, strength=.5,
                                 reset_yaw_range=.05, speed=2.)
    metadata = rollout_metadata(audit, {'reset_yaw_range_rad':.05,'speed_command':2.})
    effective = metadata['applied_configuration']
    expected_yaw = (-.05,.05) if profile=='nominal' else ((-.075,.075) if task=='stairs' else (-1.57,1.57))
    assert effective['reset']['base']['pose_range']['yaw'] == expected_yaw
    assert effective['commands']['ranges']['lin_vel_x'] == (
        ((2.,2.) if task=='run' else (.5,.5)) if profile=='nominal'
        else {'stairs':(.2,.6),'backpack':(.2,.8),'run':(1.5,2.5)}[task])
    assert 'speed_command' not in metadata and 'reset_yaw_range_rad' not in metadata
    before = deepcopy(metadata)
    audit['commands']['ranges']['lin_vel_x'] = (99.,99.)
    assert metadata == before


def test_source_change_cannot_relabel_an_existing_rollout(monkeypatch):
    protocol = randomization.protocol_metadata('robust',.5)
    assert_protocol_unchanged(protocol)
    monkeypatch.setattr(randomization,'protocol_metadata',lambda *args:dict(protocol,protocol_hash='changed'))
    with pytest.raises(ValueError,match='source changed'): assert_protocol_unchanged(protocol)


def test_one_ppo_counter_and_backward_read_of_unambiguous_bc_checkpoint(tmp_path):
    model, _ = model_and_input()
    path = tmp_path/'actor.pt'; save_student(path,model,ppo_updates=7)
    _, value = load_student(path)
    assert value['ppo_updates']==7 and value['critic_trained'] is True and 'PPO_updates' not in value
    legacy = dict(value,critic_trained=False,PPO_updates=0); legacy.pop('ppo_updates')
    old = tmp_path/'old.pt'; torch.save(legacy,old)
    _, migrated = load_student(old)
    assert migrated['ppo_updates']==0 and 'PPO_updates' not in migrated
    legacy.update(ppo_updates=7)
    torch.save(legacy,old)
    with pytest.raises(ValueError,match='conflicting'): load_student(old)
    with pytest.raises(ValueError,match='contradicts'):
        save_student(tmp_path/'bad.pt',model,ppo_updates=7,critic_trained=False)
