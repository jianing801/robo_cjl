from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from test_randomization import native_config
from native_training_conditions import (configure_native_training, native_payload_properties,
    native_terrain_scenarios, NATIVE_EXPERT_CONDITIONS, LEGACY_CONDITIONS, validate_training_conditions)

@pytest.mark.parametrize('task',['stairs','backpack','run'])
def test_native_commands_events_resets_and_delay_preserved(task):
    source=native_config(task)
    original=deepcopy(source)
    result,audit=configure_native_training(source,task,seed=17,profile='robust',strength=1.)
    command=lambda cfg:cfg.commands.base_velocity if task=='run' else cfg.commands
    assert command(result)==command(original)
    assert result.events==original.events and source.events==original.events
    assert result.scene.robot.actuators==original.scene.robot.actuators
    assert result.seed==17 and source.seed==original.seed
    assert audit['commands']['ranges']['lin_vel_x']==command(original).ranges.lin_vel_x
    assert audit['commands']['standing_commands'] is True
    assert audit['commands']['source']==NATIVE_EXPERT_CONDITIONS
    assert audit['reset']['base']==original.events.reset_base.params
    assert (not result.observations.policy.enable_corruption if task=='run' else not result.noise.add_noise)
    assert audit['student_noise']['apply_to']=='fresh_student_frame_only'


def test_native_only_full_strength_and_legacy_independent():
    assert validate_training_conditions(NATIVE_EXPERT_CONDITIONS,'robust',1.)==NATIVE_EXPERT_CONDITIONS
    assert validate_training_conditions(LEGACY_CONDITIONS,'nominal',0.)==LEGACY_CONDITIONS
    for profile,strength in [('nominal',1.),('robust',.5)]:
        with pytest.raises(ValueError):validate_training_conditions(NATIVE_EXPERT_CONDITIONS,profile,strength)


def test_payload_uses_native_mature_sampler_once():
    calls=[]
    def sample(n,**kw):
        calls.append((n,kw));return [dict(env_id=i,mass_kg=1.+i) for i in range(n)]
    def native_range(update):
        assert update==12000;return (.1,15.)
    rows=native_payload_properties(NS(sample_backpack_properties=sample),3,seed=42,payload_range=native_range)
    assert calls==[(3,dict(seed=42,mass_range_kg=(.1,15.)))]
    assert all(r['sampling_mass_range_kg']==(.1,15.) for r in rows)


def test_native_terrain_audit_does_not_assume_six_stair_strata():
    rows=native_terrain_scenarios(2,['rpo'],[11,2],[4,1])
    assert rows==[dict(environment_id=0,design_id='rpo',task='stairs',terrain_column=11,terrain_level=4),dict(environment_id=1,design_id='rpo',task='stairs',terrain_column=2,terrain_level=1)]


def test_formal_scheme_b_identity_rejects_prechange_measurements():
    from finetune_settings import CURRENT_PPO_SETTINGS, CURRENT_PPO_ALGORITHM, NATIVE_PPO_PROFILES
    assert CURRENT_PPO_SETTINGS['training_conditions']==NATIVE_EXPERT_CONDITIONS
    legacy={k:v for k,v in CURRENT_PPO_SETTINGS.items() if k!='training_conditions'}
    assert legacy!=NATIVE_PPO_PROFILES[CURRENT_PPO_ALGORITHM]


def test_live_terrain_statistics_are_not_startup_snapshot():
    import torch
    from training_reward import rough_terrain_snapshot, terrain_curriculum_rows
    raw=NS(scene=NS(terrain=NS(terrain_levels=torch.tensor([0,1,1,4]))),num_envs=4)
    a=rough_terrain_snapshot(raw)
    assert a==dict(mean=1.5,min=0,max=4,environment_count=4,level_counts={'0':1,'1':2,'4':1})
    raw.scene.terrain.terrain_levels[:] = torch.tensor([2,2,3,5])
    b=rough_terrain_snapshot(raw)
    assert b['mean']==3. and a['mean']==1.5
    rows=list(terrain_curriculum_rows([dict(completed_updates=1,metrics=dict(rough_terrain=a)),dict(completed_updates=2,metrics=dict(rough_terrain=b))]))
    assert [r['mean'] for r in rows]==[1.5,3.]
