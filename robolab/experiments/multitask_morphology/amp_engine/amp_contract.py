"""CPU-only scheduling and strict stage/checkpoint contracts for the isolated copy."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from amp_rough_alignment import rough_module

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
KIND='amp_hybrid_rotating_v1'
_schedule=rough_module('hybrid_schedule')
single_pass_order=_schedule.single_pass_order
stage_count=_schedule.stage_count
LEGACY_HYBRID_SOURCE_HASHES={
    'amp_layout.py':'c52419784de1fb6759de5783a5c5bb86855e9b521529ea0a00cba4a733ee10be',
    'amp_mixed_runtime.py':'9df29380ae7a43a0e55c8ba00b97f876c5930126d72d89118104844b6db3dad7',
    'amp_contract.py':'5074a5092e172684143face4aa8ee21716a6c3787b3b65b1aba4d8d1ea4e3511',
    'train_stage.py':'618ff97a912a44a9239f6c597b833033fab946cbfb56c10b8cc0b8db4fa66213',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def positive(x,name):
    if type(x) is not int or x<1:raise ValueError(f'{name} must be explicitly set to a positive integer')
    return x

def validate_config(config, bank):
    if not {'grid','rotation'}.issubset(config) or set(config)-{'grid','rotation','full_body_torque','group_schedule','context_schema','runtime'}:raise ValueError('expected grid, rotation and recognized optional settings')
    if 'full_body_torque' in config and type(config['full_body_torque']) is not bool:raise ValueError('full_body_torque must be boolean')
    if 'runtime' in config:
        runtime=config['runtime']
        if (not isinstance(runtime,dict) or set(runtime)!={'per_design_metrics','minibatch_sampler','init_at_random_ep_len'}
                or type(runtime['per_design_metrics']) is not bool
                or type(runtime['init_at_random_ep_len']) is not bool
                or runtime['minibatch_sampler'] not in ('group_balanced','vectorized_balanced')):
            raise ValueError('runtime must define Boolean per_design_metrics and init_at_random_ep_len, and a supported minibatch_sampler')
    g,r=config['grid'],config['rotation']
    fields={'active_designs','updates_per_stage','total_updates','seed'}
    if set(r) not in (fields,fields|{'single_pass'}):raise ValueError('unknown/missing rotation fields')
    e=positive(g['num_envs'],'grid.num_envs')
    for key in ('active_designs','updates_per_stage','total_updates'):positive(r[key],'rotation.'+key)
    if type(r['seed']) is not int:raise ValueError('rotation.seed must be integer')
    if g!=bank['configuration']:raise ValueError('grid differs from frozen full bank; generate a matching independent bank')
    from robolab.morphology_context import validate_context_config
    validate_context_config(config, bank)
    designs=[d for d in bank['designs'] if d['split']=='train']
    if len({d['id'] for d in designs})!=len(designs):raise ValueError('duplicate designs')
    if not designs or r['active_designs']>len(designs) or r['active_designs']>e:
        raise ValueError('active_designs must not exceed full training library or environment count')
    if 'single_pass' in r:
        _schedule.validate_single_pass(bank,config,len(designs))
    elif 'group_schedule' in config:
        _schedule.load_schedule(bank,config)
    return config

def cycle_groups(bank, config, cycle):
    """Full coverage exactly once per sweep; stratified interleaving, seeded reshuffle."""
    if type(cycle) is not int or cycle<0:raise ValueError('invalid cycle')
    if 'group_schedule' in config:
        from amp_rough_alignment import rough_module
        return rough_module('hybrid_schedule').groups_for(bank,config,cycle)
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

def stage_spec(bank,config,index,single_pass_ids=None):
    if type(index) is not int or index<0:raise ValueError('invalid stage index')
    r=config['rotation'];first=index*r['updates_per_stage']
    if r.get('single_pass'):
        order=single_pass_ids if single_pass_ids is not None else single_pass_order(bank,config)
        return _schedule.single_pass_stage(config,index,order)
    if first>=r['total_updates']:raise ValueError('stage exceeds absolute total budget')
    count=math.ceil(sum(d['split']=='train' for d in bank['designs'])/r['active_designs'])
    cycle,within=divmod(index,count)
    return dict(index=index,cycle=cycle,group_in_cycle=within,first_update=first,
        end_update=min(first+r['updates_per_stage'],r['total_updates']),
        design_ids=cycle_groups(bank,config,cycle)[within])

def hybrid_source_identity(config):
    sources={p.name:sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}
    # Older frozen expert plans were created before the isolated large-bank
    # runtime existed. Keep their source identity stable for safe resumption.
    if 'runtime' not in config and not config.get('rotation',{}).get('single_pass'):
        sources.update(LEGACY_HYBRID_SOURCE_HASHES)
    return sources

def experiment_identity(bank_path,config,reference_library):
    from amp_support import source_identity
    from amp_rough_alignment import shared_runtime_sources
    result = dict(kind=KIND,task='run',bank_sha256=sha(bank_path),configuration=config,
        native_sources=source_identity('run'),hybrid_sources=hybrid_source_identity(config),
        full_bank_context_is_stable=True,no_forgetting_metrics=True,reference_library_sha256=sha(reference_library),
        local_assets={'native_ground.usda':sha(HERE/'native_ground.usda')},
        shared_rough_runtime_helpers=shared_runtime_sources())
    if config.get('rotation',{}).get('single_pass'):
        result['group_schedule']=dict(binding=config['group_schedule'],order='flatten_frozen_groups_once',
                                      loader_sha256=sha(_schedule.__file__))
    elif 'group_schedule' in config:
        result['group_schedule']=_schedule.schedule_identity(json.loads(Path(bank_path).read_text(encoding='utf8')),config)
    if config.get('full_body_torque'):
        from amp_rough_alignment import rough_module
        result['full_body_sources']=rough_module('full_body_torque').source_identity()
        result['mesh_sources']=rough_module('hybrid_assets').mesh_sources(json.loads(Path(bank_path).read_text(encoding='utf8')))
    return result

def load_plan(path,bank_path):
    from amp_bank import load_configurable_bank
    plan=json.loads(Path(path).read_text(encoding='utf8'))
    bank=load_configurable_bank(bank_path)
    validate_config(plan['configuration'],bank)
    if plan['experiment']!=experiment_identity(bank_path,plan['configuration'],plan['reference_library']):raise ValueError('bank/config/source identity changed')
    return plan

def validate_checkpoint(path,experiment,expected_completed,*,within_stage=None):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('hybrid_identity')!=experiment:raise ValueError('not this hybrid experiment; old AMP checkpoints rejected')
    if type(state.get('iter')) is not int or state['iter']+1!=expected_completed:raise ValueError('checkpoint absolute update differs from scheduled boundary')
    stage=state.get('hybrid_stage',{})
    if within_stage is not None:
        if (stage != within_stage or stage.get('first_update') != 0 or len(stage.get('design_ids',[])) != 1
                or not 0 < expected_completed <= stage['end_update']):
            raise ValueError('partial checkpoint must belong to the exact fixed singleton stage')
    elif stage.get('end_update')!=expected_completed:raise ValueError('only complete previous-stage checkpoint may resume')
    weights=state['model_state_dict'];optimizer=state['optimizer_state_dict']
    from amp_layout import identity
    extra=weights.get('_extra_state',{})
    native=extra.get('config',{})
    if native.get('spatial_identity')!=identity(experiment) or native.get('role')!='run':raise ValueError('policy extra-state identity mismatch')
    if not any(k.startswith('actor.') for k in weights) or not any(k.startswith('critic.') for k in weights):raise ValueError('actor/critic missing')
    if not all(any(k.startswith(prefix) for k in weights) for prefix in
               ('actor_obs_normalizer.', 'critic_obs_normalizer.')):
        raise ValueError('actor/critic normalization missing')
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
    required = ('amp_discriminator_state_dict','amp_discriminator_normalizer_state_dict',
                'amp_discriminator_optimizer_state_dict')
    for name in required:
        if not isinstance(state.get(name), dict) or not state[name]:
            raise ValueError(f'missing AMP checkpoint state: {name}')
        finite(state[name])
    disc = state['amp_discriminator_state_dict']
    for key,value in state['amp_discriminator_normalizer_state_dict'].items():
        if 'disc_obs_normalizer.'+key not in disc or not torch.equal(value,disc['disc_obs_normalizer.'+key]):
            raise ValueError('AMP normalizer snapshots disagree')
    adam = state['amp_discriminator_optimizer_state_dict']
    if not adam.get('state') or not adam.get('param_groups'):
        raise ValueError('empty discriminator Adam')
    if any(not {'exp_avg','exp_avg_sq','step'}<=set(v) for v in adam['state'].values()):
        raise ValueError('incomplete discriminator Adam slots')
    return state
