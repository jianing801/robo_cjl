"""CPU-only scheduling and strict stage/checkpoint contracts for the isolated copy."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
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
    if not {'grid','rotation'}.issubset(config) or set(config)-{'grid','rotation','torque_soft_constraint','full_body_torque','group_schedule'}:
        raise ValueError('expected grid, rotation and optional torque_soft_constraint')
    if 'full_body_torque' in config and type(config['full_body_torque']) is not bool:raise ValueError('full_body_torque must be boolean')
    if config.get('full_body_torque') and config.get('torque_soft_constraint') not in (None,{'weight':0,'limit_ratio':.8}):raise ValueError('full-body and six-joint penalties cannot both be active')
    from hybrid_torque import validate_settings
    validate_settings(config.get('torque_soft_constraint'))
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
    if 'group_schedule' in config:
        from hybrid_schedule import load_schedule
        load_schedule(bank,config)
    return config

def cycle_groups(bank, config, cycle):
    """Full coverage exactly once per sweep; stratified interleaving, seeded reshuffle."""
    if type(cycle) is not int or cycle<0:raise ValueError('invalid cycle')
    if 'group_schedule' in config:
        from hybrid_schedule import groups_for
        return groups_for(bank,config,cycle)
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
    task=json.loads(Path(bank_path).read_text(encoding='utf8')).get('task_key','stairs')
    result = dict(kind=KIND,task=task,bank_sha256=sha(bank_path),configuration=config,
        native_sources=source_identity(task),hybrid_sources={p.name:sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')},
        full_bank_context_is_stable=True,no_forgetting_metrics=True)
    if 'group_schedule' in config:
        from hybrid_schedule import schedule_identity
        result['group_schedule']=schedule_identity(json.loads(Path(bank_path).read_text(encoding='utf8')),config)
    if config.get('full_body_torque'):
        from full_body_torque import source_identity as torque_identity
        result['full_body_sources']=torque_identity()
        from hybrid_assets import mesh_sources
        result['mesh_sources']=mesh_sources(json.loads(Path(bank_path).read_text(encoding='utf8')),use_cache=True)
    return result

def load_plan(path,bank_path):
    from hybrid_bank import load_configurable_bank
    plan=json.loads(Path(path).read_text(encoding='utf8'))
    bank=load_configurable_bank(bank_path)
    validate_config(plan['configuration'],bank)
    if not controller_compatible(plan['experiment'],experiment_identity(bank_path,plan['configuration']),Path(path).parent):raise ValueError('bank/config/source identity changed')
    if plan.get('start_stage',0):
        root=Path(path).resolve().parent
        if plan.get('execution_protocol')!='persistent_v1' or type(plan['start_stage']) is not int or plan['start_stage']<1:
            raise ValueError('invalid upgrade start stage')
        if sha(plan['bootstrap_checkpoint'])!=plan['bootstrap_sha256'] or sha(root/'upgrade_audit.json')!=plan['upgrade_audit_sha256']:
            raise ValueError('upgrade bootstrap/audit identity changed')
        validate_checkpoint(plan['bootstrap_checkpoint'],plan['experiment'],
                            stage_spec(bank,plan['configuration'],plan['start_stage'])['first_update'])
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
    if native.get('spatial_identity')!=identity(experiment) or native.get('role')!=experiment.get('task','stairs'):raise ValueError('policy extra-state identity mismatch')
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


# BEGIN_CONTROLLER_COMPATIBILITY

def controller_compatible(recorded,current,root):
    """仅接受已审阅的跨任务进程检查修复；旧模型身份完整保留。"""
    if recorded==current:return True
    import hashlib
    a=dict(recorded);b=dict(current)
    before=dict(a.pop('hybrid_sources'));after=dict(b.pop('hybrid_sources'))
    if a!=b:return False
    expected={'train.py':'33e38a4c10a79c8f50fd62c975f2a54572c7c719bb31b9331fa83421ddfbd7c3',
              'hybrid_contract.py':'4a86085ab8a5800ee838966e7126f1910ad3882a4c0f79e7a2fab42cc2006ccd',
              'hybrid_assets.py':'82a28ec1eb686d14dc677ef50262d89b2837004464fafec37b010f6287106543'}
    allowed_previous={'train.py': 'e0c542265065b39aded6d51ca7865eba8eb1ada1eee91fc6311ba7cc16bfe623', 'hybrid_contract.py': 'f503055ab9a74e4c90eabf4660db51ac9fcedb656dc3b28b327b544fa850a113', 'hybrid_assets.py': '63a9c3f657b5a089477aed55074c0b284f0634337ea19eebc7ae9e8cc3df81b0'}
    # 还原审阅替换后须符合原源码SHA；只接受已知旧版或上版控制器。
    train=(HERE/'train.py').read_text(encoding='utf8')
    old_guard="if any(Path(x).name in ('train_stage.py','persistent_worker.py','probe_persistent.py') for x in args):raise RuntimeError(f'orphan/active hybrid worker {proc.pid}; inspect first')"
    new_guard="if any(Path(x).name in ('train_stage.py','persistent_worker.py','probe_persistent.py') for x in args) and (str(bank_path) in args or str(root) in args or not any(flag in args for flag in ('--bank','--root','--hybrid-plan'))):raise RuntimeError(f'orphan/active same-bank or unknown hybrid worker {proc.pid}; inspect first')"
    if train.count(new_guard)!=1:return False
    restored_train=train.replace(new_guard,old_guard)
    contract=(HERE/'hybrid_contract.py').read_text(encoding='utf8')
    marker='\n\n# BEGIN_'+'CONTROLLER_COMPATIBILITY'
    contract=contract[:contract.index(marker)]
    restored_contract=contract.replace("if not controller_compatible(plan['experiment'],experiment_identity(bank_path,plan['configuration']),Path(path).parent):raise ValueError('bank/config/source identity changed')","if plan['experiment']!=experiment_identity(bank_path,plan['configuration']):raise ValueError('bank/config/source identity changed')")
    restored_contract=restored_contract.replace("result['mesh_sources']=mesh_sources(json.loads(Path(bank_path).read_text(encoding='utf8')),use_cache=True)","result['mesh_sources']=mesh_sources(json.loads(Path(bank_path).read_text(encoding='utf8')))")
    assets=(HERE/'hybrid_assets.py').read_text(encoding='utf8')
    marker_assets='\n\n# BEGIN_'+'TRAINING_MESH_CACHE'
    restored_assets=assets[:assets.index(marker_assets)].replace('def _mesh_sources_deep(bank,report=True):','def mesh_sources(bank,report=True):',1)
    for name,text in [('train.py',restored_train),('hybrid_contract.py',restored_contract),('hybrid_assets.py',restored_assets)]:
        digest=hashlib.sha256(text.encode('utf8')).hexdigest()
        if digest!=expected[name] or before.get(name) not in (digest,allowed_previous[name]):return False
        before.pop(name);after.pop(name,None)
    if before!=after:return False
    audit=Path(root)/'training_source_cache_compatibility.json'
    value=dict(reason='reviewed controller scope and metadata source-cache optimization; frozen checkpoint identity retained',
               original_sources=expected,actual_sources={k:current['hybrid_sources'][k] for k in expected})
    if audit.exists():
        if json.loads(audit.read_text(encoding='utf8'))!=value:return False
    else:
        from hybrid_support import write_new
        write_new(audit,value)
    return True
