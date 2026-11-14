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
    if config['rotation']['active_designs']!=60 or value.get('group_count')!=50 or value.get('designs_per_group')!=60:
        raise ValueError('Frozen schedule requires 50 groups of 60 structures')
    if value.get('within_group_rotation_stride')!=16:
        raise ValueError('Unexpected within-group replica allocation rule')
    groups=value['groups'];rows=[d for d in bank['designs'] if d['split']=='train']
    lookup={d['id']:d for d in rows};flat=[i for group in groups for i in group]
    if len(groups)!=50 or any(len(g)!=60 for g in groups):
        raise ValueError('Invalid frozen group capacity')
    if len(rows)!=3000 or len(lookup)!=3000 or len(set(flat))!=3000 or set(flat)!=set(lookup):
        raise ValueError('Frozen schedule has missing, duplicate or foreign structures')
    axes={axis:sorted({d[axis] for d in rows}) for axis in ('thigh_m','calf_m')}
    for group in groups:
        pairs=Counter((lookup[i]['knee_motor'],lookup[i]['ankle_motor']) for i in group)
        if len(pairs)!=30 or set(pairs.values())!={2}:
            raise ValueError('Frozen group motor pairs are not exactly balanced')
        for axis,levels in axes.items():
            counts=Counter(lookup[i][axis] for i in group)
            if len(levels)!=10 or set(counts)!=set(levels) or set(counts.values())!={6}:
                raise ValueError('Frozen group leg lengths are not exactly balanced')
    return value,path


def groups_for(bank,config,cycle):
    if type(cycle) is not int or cycle<0:raise ValueError('Invalid schedule cycle')
    value,_=load_schedule(bank,config)
    # Keep membership and group order fixed: this preserves cyclic window audits.
    # Move the 16 extra 4096/60 replicas across slots instead of favoring one set.
    shift=(cycle*value['within_group_rotation_stride'])%60
    return [g[shift:]+g[:shift] for g in value['groups']]


def schedule_identity(bank,config):
    value,path=load_schedule(bank,config)
    return dict(kind=value['kind'],sha256=sha(path),loader_sha256=sha(__file__),
        group_count=50,designs_per_group=60,repeat_group_order=True,
        within_group_rotation_stride=value['within_group_rotation_stride'])
