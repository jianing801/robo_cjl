"""Read-only frozen schedule binding shared by all three expert tasks."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_schedule(bank,config):
    binding=config['group_schedule']
    if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:
        raise ValueError('Expected explicit group schedule path and SHA256')
    path=(ROOT/binding['path']).resolve()
    if sha(path)!=binding['sha256']:
        raise ValueError('Frozen group schedule hash changed; no silent rescheduling')
    value=json.loads(path.read_text(encoding='utf8'))
    if value.get('kind')!='balanced_fixed_grid_schedule_v1' or value.get('schema_version')!=1:
        raise ValueError('Unsupported frozen group schedule')
    if value.get('grid')!=config['grid'] or bank['configuration']!=config['grid']:
        raise ValueError('Frozen schedule belongs to a different morphology grid')
    grid=config['grid']
    capacity=config['rotation']['active_designs']
    length_count=grid['length_count']
    pair_count=grid['knee_motor_count']*grid['ankle_motor_count']
    total=length_count**2*pair_count
    group_count,remainder=divmod(total,capacity)
    axis_count,axis_remainder=divmod(capacity,length_count)
    if (remainder or axis_remainder or capacity!=2*pair_count
            or value.get('group_count')!=group_count or value.get('designs_per_group')!=capacity):
        raise ValueError('Frozen schedule dimensions do not match the full balanced grid')
    if value.get('within_group_rotation_stride')!=grid['num_envs']%capacity:
        raise ValueError('Unexpected within-group replica allocation rule')
    groups=value['groups'];rows=[d for d in bank['designs'] if d['split']=='train']
    lookup={d['id']:d for d in rows};flat=[i for group in groups for i in group]
    if len(groups)!=group_count or any(len(g)!=capacity for g in groups):
        raise ValueError('Invalid frozen group capacity')
    if len(rows)!=total or len(lookup)!=total or len(set(flat))!=total or set(flat)!=set(lookup):
        raise ValueError('Frozen schedule has missing, duplicate or foreign structures')
    axes={axis:sorted({d[axis] for d in rows}) for axis in ('thigh_m','calf_m')}
    for group in groups:
        pairs=Counter((lookup[i]['knee_motor'],lookup[i]['ankle_motor']) for i in group)
        if len(pairs)!=pair_count or set(pairs.values())!={2}:
            raise ValueError('Frozen group motor pairs are not exactly balanced')
        for axis,levels in axes.items():
            counts=Counter(lookup[i][axis] for i in group)
            if len(levels)!=length_count or set(counts)!=set(levels) or set(counts.values())!={axis_count}:
                raise ValueError('Frozen group leg lengths are not exactly balanced')
    return value,path


def groups_for(bank,config,cycle):
    if type(cycle) is not int or cycle<0:raise ValueError('Invalid schedule cycle')
    value,_=load_schedule(bank,config)
    # Keep membership and group order fixed: this preserves cyclic window audits.
    # Rotate by the replica remainder so every slot gets the same long-run exposure.
    shift=(cycle*value['within_group_rotation_stride'])%value['designs_per_group']
    return [g[shift:]+g[:shift] for g in value['groups']]


def single_pass_order(bank, config):
    """Flatten the common frozen group order into one visit per structure."""
    grid = config['grid']
    group_size = 2 * grid['knee_motor_count'] * grid['ankle_motor_count']
    grouped = dict(config, rotation=dict(config['rotation'], active_designs=group_size))
    schedule, _ = load_schedule(bank, grouped)
    return [design for group in schedule['groups'] for design in group]


def validate_single_pass(bank, config, design_count):
    rotation = config['rotation']
    if (rotation['single_pass'] is not True or rotation['active_designs'] != 1
            or 'group_schedule' not in config
            or rotation['total_updates'] // design_count != rotation['updates_per_stage']):
        raise ValueError('single-pass requires one design, a frozen order and floor(total/designs) updates per stage')
    single_pass_order(bank, config)


def stage_count(bank, config):
    rotation = config['rotation']
    if rotation.get('single_pass'):
        return sum(design['split'] == 'train' for design in bank['designs'])
    steps = rotation['updates_per_stage']
    return (rotation['total_updates'] + steps - 1) // steps


def single_pass_stage(config, index, order):
    if index >= len(order):
        raise ValueError('stage exceeds absolute total budget')
    base, remainder = divmod(config['rotation']['total_updates'], len(order))
    first = index * base + min(index, remainder)
    return dict(index=index, cycle=0, group_in_cycle=index, first_update=first,
                end_update=first + base + (index < remainder), design_ids=[order[index]])


def schedule_identity(bank,config):
    value,path=load_schedule(bank,config)
    return dict(kind=value['kind'],sha256=sha(path),loader_sha256=sha(__file__),
        group_count=value['group_count'],designs_per_group=value['designs_per_group'],repeat_group_order=True,
        within_group_rotation_stride=value['within_group_rotation_stride'])
