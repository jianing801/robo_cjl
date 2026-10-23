"""CPU-only scheduling and strict stage/checkpoint contracts for the isolated copy."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
KIND='rough_hybrid_rotating_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def positive(x,name):
    if type(x) is not int or x<1:raise ValueError(f'{name} must be explicitly set to a positive integer')
    return x

def validate_config(config, bank):
    if set(config)!={'grid','rotation'}:raise ValueError('expected grid and rotation configuration')
    g,r=config['grid'],config['rotation']
    if set(r)!={'active_designs','updates_per_stage','total_updates','seed'}:raise ValueError('unknown/missing rotation fields')
    e=positive(g['num_envs'],'grid.num_envs')
    for key in ('active_designs','updates_per_stage','total_updates'):positive(r[key],'rotation.'+key)
    if type(r['seed']) is not int:raise ValueError('rotation.seed must be integer')
    if g!=bank['configuration']:raise ValueError('grid differs from frozen full bank; generate a matching independent bank')
    designs=[d for d in bank['designs'] if d['split']=='train']
    if len({d['id'] for d in designs})!=len(designs):raise ValueError('duplicate designs')
    if not designs or r['active_designs']>len(designs) or r['active_designs']>e:
        raise ValueError('active_designs must not exceed full training library or environment count')
    return config

def cycle_groups(bank, config, cycle):
    """Full coverage exactly once per sweep; stratified interleaving, seeded reshuffle."""
    if type(cycle) is not int or cycle<0:raise ValueError('invalid cycle')
    rows=[d for d in bank['designs'] if d['split']=='train']
    cap=config['rotation']['active_designs']
    count=(len(rows)+cap-1)//cap
    rng=random.Random(f"{config['rotation']['seed']}:{cycle}")
    thighs=sorted({d['thigh_m'] for d in rows});calves=sorted({d['calf_m'] for d in rows})
    pairs=sorted({(d['knee_motor'],d['ankle_motor']) for d in rows})
    if len(rows)==900 and len(thighs)==len(calves)==10 and len(pairs)==9 and cap==30:
        # Three offset perfect matchings per group. Every axis value appears
        # exactly three times; motor frequencies are six x3 and three x4.
        # Across 30 groups all 10x10x9 designs appear exactly once.
        rng.shuffle(pairs)
        lookup={(d['thigh_m'],d['calf_m'],d['knee_motor'],d['ankle_motor']):d['id'] for d in rows}
        shift=rng.randrange(10);reverse=rng.choice((False,True));groups=[]
        for r in range(3):
            for g in range(10):
                group=[]
                for j in range(3):
                    for t in range(10):
                        c=(t+g+3*j+shift)%10
                        if reverse:c=9-c
                        knee,ankle=pairs[(t+r+3*j)%9]
                        group.append(lookup[(thighs[t],calves[c],knee,ankle)])
                rng.shuffle(group);groups.append(group)
        rng.shuffle(groups)
        return groups
    buckets=defaultdict(list)
    def region(value, bounds):return min(2,max(0,int(3*(value-bounds[0])/(bounds[1]-bounds[0]))))
    for d in rows:
        key=(d['knee_motor'],d['ankle_motor'],region(d['thigh_m'],bank['ranges_m']['thigh']),region(d['calf_m'],bank['ranges_m']['calf']))
        buckets[key].append(d['id'])
    for values in buckets.values():rng.shuffle(values)
    ordered=[]
    while buckets:
        keys=sorted(buckets);rng.shuffle(keys)
        for key in keys:
            ordered.append(buckets[key].pop())
            if not buckets[key]:del buckets[key]
    groups=[[] for _ in range(count)]
    # Round-robin allocation also avoids a singleton tail group when N is not divisible.
    for i,key in enumerate(ordered):groups[i%count].append(key)
    rng.shuffle(groups)
    return groups

def stage_spec(bank,config,index):
    if type(index) is not int or index<0:raise ValueError('invalid stage index')
    r=config['rotation'];first=index*r['updates_per_stage']
    if first>=r['total_updates']:raise ValueError('stage exceeds absolute total budget')
    count=math.ceil(sum(d['split']=='train' for d in bank['designs'])/r['active_designs'])
    cycle,within=divmod(index,count)
    return dict(index=index,cycle=cycle,group_in_cycle=within,first_update=first,
        end_update=min(first+r['updates_per_stage'],r['total_updates']),
        design_ids=cycle_groups(bank,config,cycle)[within])

def experiment_identity(bank_path,config):
    from hybrid_support import source_identity
    return dict(kind=KIND,task='stairs',bank_sha256=sha(bank_path),configuration=config,
        native_sources=source_identity('stairs'),hybrid_sources={p.name:sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')},
        full_bank_context_is_stable=True,no_forgetting_metrics=True)

def load_plan(path,bank_path):
    from hybrid_bank import load_configurable_bank
    plan=json.loads(Path(path).read_text(encoding='utf8'))
    bank=load_configurable_bank(bank_path)
    validate_config(plan['configuration'],bank)
    if plan['experiment']!=experiment_identity(bank_path,plan['configuration']):raise ValueError('bank/config/source identity changed')
    return plan

def validate_checkpoint(path,experiment,expected_completed):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('hybrid_identity')!=experiment:raise ValueError('not this hybrid experiment; old Rough checkpoints rejected')
    if type(state.get('iter')) is not int or state['iter']+1!=expected_completed:raise ValueError('checkpoint absolute update differs from scheduled boundary')
    stage=state.get('hybrid_stage',{})
    if stage.get('end_update')!=expected_completed:raise ValueError('only complete previous-stage checkpoint may resume')
    weights=state['model_state_dict'];optimizer=state['optimizer_state_dict']
    from hybrid_layout import identity
    extra=weights.get('_extra_state',{})
    native=extra.get('config',{})
    if native.get('spatial_identity')!=identity(experiment) or native.get('role')!='stairs':raise ValueError('policy extra-state identity mismatch')
    if not any(k.startswith('actor.') for k in weights) or not any(k.startswith('critic.') for k in weights):raise ValueError('actor/critic missing')
    def finite(value):
        if isinstance(value,torch.Tensor):
            if not torch.isfinite(value).all():raise ValueError('nonfinite checkpoint tensor')
        elif isinstance(value,dict):
            for x in value.values():finite(x)
        elif isinstance(value,(list,tuple)):
            for x in value:finite(x)
        elif isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite checkpoint scalar')
    finite(weights);finite(optimizer)
    if not optimizer.get('state') or not optimizer.get('param_groups'):raise ValueError('empty Adam optimizer')
    if any(not {'exp_avg','exp_avg_sq','step'}<=set(v) for v in optimizer['state'].values()):raise ValueError('Adam slots incomplete')
    lr=state.get('learning_rate')
    if not isinstance(lr,(float,int)) or not math.isfinite(lr) or lr<=0:raise ValueError('invalid learning rate')
    if not isinstance(state.get('torch_rng_state'),torch.Tensor) or not isinstance(state.get('cuda_rng_states'),list):raise ValueError('RNG state missing')
    return state
