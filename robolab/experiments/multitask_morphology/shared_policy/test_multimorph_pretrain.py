from collections import Counter
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from contract import structure_key

def grid():
    return list(product((.2,.2666666667,.3333333333,.4),(.25,.3166666667,.3833333333,.45),
                        tuple(f'k{i}' for i in range(5)),tuple(f'a{i}' for i in range(5))))

@pytest.mark.parametrize('count,low,high', [(4096,10,11),(1024,2,3)])
def test_balanced_fixed_population(count,low,high):
    from multimorph_pretrain import balanced_order
    keys=grid()
    order=balanced_order(keys,count,seed=20261104)
    assert order==balanced_order(list(reversed(keys)),count,seed=20261104)
    assert len(order)==len(set(order))==400 and set(order)==set(keys)
    copies=Counter(order[i%400] for i in range(count))
    assert set(copies.values())=={low,high}
    assert sum(copies.values())==count
    for axes in [(0,),(1,),(2,),(3,),(0,1),(2,3)]:
        totals=Counter()
        for key,n in copies.items():totals[tuple(key[i] for i in axes)]+=n
        assert max(totals.values())-min(totals.values())<=1

def test_each_task_maps_physical_keys_independently(tmp_path):
    from multimorph_pretrain import select_training_plan,LAYOUT
    keys=grid()
    plan={'selected_structure_keys':[list(k) for k in keys], 'teachers':{}, 'schema':{'tasks':list(LAYOUT),'shared_structure_keys':[list(k) for k in keys]}}
    for task in LAYOUT:
        rows=[dict(id=f'{task}-{i}',thigh_m=k[0],calf_m=k[1],knee_motor=k[2],ankle_motor=k[3]) for i,k in enumerate(keys)]
        bank=tmp_path/f'{task}.json';bank.write_text(json.dumps({'designs':rows}))
        plan['teachers'][task]={'bank':str(bank),'design_ids':[r['id'] for r in rows]}
    frozen,audit=select_training_plan(plan,keys,20261104)
    assert frozen['schema']==plan['schema']
    assert set(map(tuple,frozen['selected_structure_keys']))==set(keys)
    for task,n in LAYOUT.items():
        ids=frozen['teachers'][task]['design_ids']
        assert len(ids)==len(set(ids))==400 and all(i.startswith(task) for i in ids)
        assert sum(audit[task]['envs_per_design'].values())==n
        assert audit[task]['design_count']==400
        assert set(audit[task]['envs_per_design'])==set(ids)
    assert plan['teachers']['stairs']['design_ids']==[f'stairs-{i}' for i in range(400)]
    broken=plan.copy();broken['teachers']={**plan['teachers'],'backpack':dict(plan['teachers']['backpack'],design_ids=plan['teachers']['backpack']['design_ids'][:-1])}
    with pytest.raises(ValueError,match='backpack'):
        select_training_plan(broken,keys,20261104)

def test_balanced_morphology_training_stairs_and_payload():
    from scenario_sampling import route_stairs,payload_properties
    n=4096
    terrain=SimpleNamespace(terrain_types=torch.zeros(n,dtype=torch.long),terrain_levels=torch.zeros(n,dtype=torch.long),
        terrain_origins=torch.arange(18.).reshape(1,6,3),env_origins=torch.zeros(n,3))
    scene=SimpleNamespace(terrain=terrain,env_origins=torch.zeros(n,3))
    route_stairs(scene,400,profile='robust',balanced_morphology_training=True)
    assert all(set(terrain.terrain_types[i::400].tolist())==set(range(6)) for i in range(400))
    assert max(torch.bincount(terrain.terrain_types))-min(torch.bincount(terrain.terrain_types))<=2
    module=SimpleNamespace(sample_backpack_properties=lambda n,**kw:[{'mass_kg':sum(kw['mass_range_kg'])/2}])
    rows=payload_properties(module,1024,400,seed=20261104,profile='robust',strength=1.,balanced_morphology_training=True)
    bins=[tuple(r['sampling_mass_range_kg']) for r in rows]
    assert all(len(set(bins[i::400]))==len(bins[i::400]) for i in range(400))
    assert max(Counter(bins).values())-min(Counter(bins).values())<=2

def test_pretraining_scenario_receipt_requires_all_400_per_task():
    from multimorph_pretrain import LAYOUT,validate_scenarios
    plan={'teachers':{t:{'design_ids':[f'{t}-{i}' for i in range(400)]} for t in LAYOUT}}
    rows={}
    for task,n in LAYOUT.items():
        ids=plan['teachers'][task]['design_ids']
        rows[task]=[dict(environment_id=i,design_id=ids[i%400],task=task) for i in range(n)]
        if task=='stairs':
            for i,r in enumerate(rows[task]):r['terrain_stratum']=(i//400+i%400)%6
        if task=='backpack':
            for i,r in enumerate(rows[task]):r['payload']={'sampling_mass_range_kg':[(.1,5.),(5.,10.),(10.,15.)][(i//400+i%400)%3]}
    audit=validate_scenarios(plan,rows)
    assert {t:audit[t]['design_count'] for t in LAYOUT}=={t:400 for t in LAYOUT}
    rows['run'][-1]['design_id']='invalid'
    with pytest.raises(ValueError,match='run'):validate_scenarios(plan,rows)
