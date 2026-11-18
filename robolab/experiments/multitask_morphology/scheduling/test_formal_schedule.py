"""Actual three-task formal dispatch, frozen identity and stage boundaries."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'engine'),str(HERE.parent/'amp_engine'),str(HERE)]
import hybrid_contract
import amp_contract
from hybrid_bank import select_grid
from hybrid_schedule import load_schedule,groups_for,schedule_identity
from audit import audit


def setup(task):
    cfg=json.loads((HERE.parent/'configs'/f'{task}.json').read_text(encoding='utf8'))
    grid=cfg['grid'];selected=select_grid(grid,grid['knee_pool'],grid['ankle_pool'])
    bank=dict(configuration=grid,designs=selected['designs'],context_schema=cfg['context_schema'],
              ranges_m={'thigh':grid['thigh_range_m'],'calf':grid['calf_range_m']})
    return cfg,bank


@pytest.mark.parametrize('task',['rough','backpack','amp'])
def test_actual_dispatch_and_full_absolute_budget(task):
    cfg,b=setup(task);module=amp_contract if task=='amp' else hybrid_contract
    module.validate_config(cfg,b)
    first=module.cycle_groups(b,cfg,0)
    assert first==groups_for(b,cfg,0)
    assert len(first)==50 and all(len(g)==50 for g in first)
    for cycle in range(5):
        groups=module.cycle_groups(b,cfg,cycle)
        assert [set(g) for g in groups]==[set(g) for g in first]
        assert Counter(i for g in groups for i in g)==Counter(d['id'] for d in b['designs'])
        assert groups[0]==first[0][cycle*46%50:]+first[0][:cycle*46%50]
    assert module.stage_spec(b,cfg,49)['end_update']==2500
    assert module.stage_spec(b,cfg,50)['cycle']==1
    last=cfg['rotation']['total_updates']//50-1
    assert module.stage_spec(b,cfg,last)['end_update']==cfg['rotation']['total_updates']
    with pytest.raises(ValueError):module.stage_spec(b,cfg,last+1)


def test_exact_hard_constraints_independent_audit_and_joint_task_binding():
    cfg,b=setup('rough');g=groups_for(b,cfg,0)
    report=audit(g,b,windows=())
    assert report['pair_count_range']==[2,2]
    assert report['thigh_count_range']==report['calf_count_range']==[5,5]
    assert [json.loads((HERE.parent/'configs'/f'{t}.json').read_text())['group_schedule']
            for t in ('rough','backpack','amp')]==[cfg['group_schedule']]*3
    identity=schedule_identity(b,cfg)
    assert identity['sha256']==cfg['group_schedule']['sha256']
    assert identity['repeat_group_order'] is True


def test_corrupt_hash_grid_and_modified_membership_are_rejected(tmp_path):
    cfg,b=setup('rough');bad=deepcopy(cfg);bad['group_schedule']['sha256']='0'*64
    with pytest.raises(ValueError,match='hash changed'):load_schedule(b,bad)
    bad=deepcopy(b);bad['configuration']['thigh_range_m']=[.1,.2]
    with pytest.raises(ValueError,match='different morphology'):load_schedule(bad,cfg)
    value,_=load_schedule(b,cfg);value['groups'][0][0]=value['groups'][1][0]
    path=tmp_path/'bad.json';path.write_text(json.dumps(value),encoding='utf8')
    bad=deepcopy(cfg);bad['group_schedule']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='duplicate'):load_schedule(b,bad)


def test_replica_allocation_rotates_without_modifying_frozen_bytes():
    cfg,b=setup('rough');value,path=load_schedule(b,cfg);original=path.read_bytes()
    extra_counts=Counter()
    for cycle in range(25):
        group=groups_for(b,cfg,cycle)[0]
        extra_counts.update(group[:4096%50])
    # A stride of 46 modulo 50 has period 25; every slot receives 23 extras.
    assert set(extra_counts.values())=={23} and len(extra_counts)==50
    assert path.read_bytes()==original
