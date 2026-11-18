"""Read-only teacher/physical identities and unified known-morphology input."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from persistence import write_json_new

TASKS=('stairs','backpack')
HISTORY_WIDTH=780
STATIC_WIDTH=350


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    write_json_new(path,value)


def schema_for(banks):
    first=banks['stairs']['canonical_semantic_contract']
    if any(b['canonical_semantic_contract']!=first for b in banks.values()):
        raise ValueError('Teacher joint order/default pose mismatch')
    if any(b['ranges_m']!=banks['stairs']['ranges_m'] for b in banks.values()):
        raise ValueError('Teacher length scaling mismatch')
    knees=set.intersection(*(set(b['knees']) for b in banks.values()))
    ankles=set.intersection(*(set(b['ankles']) for b in banks.values()))
    if not knees or not ankles:raise ValueError('No common exact motor models')
    tasks=[t for t in (*TASKS,'run') if t in banks]
    if len(tasks)!=len(banks):raise ValueError('Unknown teacher task')
    kind='rough_backpack_amp_intersection_v3' if 'run' in tasks else 'rough_backpack_intersection_mxt_v2'
    schema = dict(kind=kind,tasks=tasks,
        knees=sorted(knees),ankles=sorted(ankles),
        semantic_contract=first,ranges_m=banks['stairs']['ranges_m'],history_width=780,
        history_layout='frame_major_oldest_to_newest',action_scale=.25,step_dt=.02,
        physical_context='known legs, exact motor IDs and physical descriptors; no measured payload in actor',
        fixed_backpack='task indicates independent pack body; 0.3kg template is not runtime load')
    fields = ('context_schema', 'context_fields', 'context_normalization')
    for field in fields:
        if any(b.get(field) != banks['stairs'].get(field) for b in banks.values()):
            raise ValueError(f'Teacher shape description differs: {field}')
    if banks['stairs'].get('context_schema') is not None:
        from robolab.morphology_context import SCHEMA, WIDTH, FIELDS
        if any(b.get('context_schema') != SCHEMA or b.get('context_width') != WIDTH
               or b.get('context_fields') != list(FIELDS) for b in banks.values()):
            raise ValueError('Invalid physical context contract')
        schema.update({field: banks['stairs'][field] for field in fields})
        schema.update(context_width=WIDTH, physical_context='known leg and actuator physical parameters')
    return schema


def unified_context(bank,design,schema):
    if design['knee_motor'] not in schema['knees'] or design['ankle_motor'] not in schema['ankles']:
        raise ValueError('Design motor is outside the shared intersection')
    if 'shared_structure_keys' in schema and list(structure_key(design)) not in schema['shared_structure_keys']:
        raise ValueError('Design is outside the shared structure intersection')
    original=design['context']
    if bank.get('context_schema') != schema.get('context_schema'):
        raise ValueError('Native and shared context schemas differ')
    if schema.get('context_schema') is not None:
        if len(original) != schema['context_width'] or not all(math.isfinite(v) for v in original):
            raise ValueError('Invalid native physical context')
        return original.copy()
    nk,na=len(bank['knees']),len(bank['ankles'])
    if len(original)!=18+nk+na:raise ValueError('Native context width mismatch')
    expected=[float(design['knee_motor']==m) for m in bank['knees']]
    expected += [float(design['ankle_motor']==m) for m in bank['ankles']]
    if original[2:2+nk+na]!=expected:raise ValueError('Teacher category encoding mismatch')
    result=original[:2]+[float(design['knee_motor']==m) for m in schema['knees']]
    result += [float(design['ankle_motor']==m) for m in schema['ankles']]+original[2+nk+na:]
    if not all(math.isfinite(v) for v in result):raise ValueError('Nonfinite context')
    return result


def graph_features(path,names,q0):
    """Same 25-node/14-static-slot convention as the previous MxT collector.

    Fixed payload is task-specific; dynamic load remains unobserved by actor.
    This does not substitute the 0.3kg URDF initializer for actual load.
    """
    root=ET.parse(path).getroot()
    joints={x.get('name'):x for x in root.findall('joint')}
    links={x.get('name'):x for x in root.findall('link')}
    if {n for n,j in joints.items() if j.get('type')!='fixed'}!=set(names):
        raise ValueError('Physical actuated-joint set mismatch')
    child_joint={j.find('child').get('link'):n for n,j in joints.items()}
    roots=set(links)-set(child_joint)
    if len(roots)!=1:raise ValueError('Invalid physical topology')
    idx={n:i+1 for i,n in enumerate(names)}
    features=[[0.]*14 for _ in range(25)]
    mass=links[next(iter(roots))].find('inertial/mass')
    features[0][11]=float(mass.get('value')) if mass is not None else 0.
    for n,default in zip(names,q0):
        joint=joints[n];parent=joint.find('parent').get('link');visited=set()
        while parent in child_joint and child_joint[parent] not in idx:
            if parent in visited:raise ValueError('Cyclic URDF')
            visited.add(parent);parent=joints[child_joint[parent]].find('parent').get('link')
        parent_idx=idx[child_joint[parent]] if parent in child_joint else 0
        def vec(element,key,default):
            value=element.get(key,default) if element is not None else default
            return [float(v) for v in value.split()]
        lim=joint.find('limit');mass=links[joint.find('child').get('link')].find('inertial/mass')
        values=vec(joint.find('origin'),'xyz','0 0 0')+vec(joint.find('origin'),'rpy','0 0 0')
        values+=vec(joint.find('axis'),'xyz','1 0 0')
        values += [float(lim.get(k,'0')) for k in ('lower','upper')]
        values += [float(mass.get('value')) if mass is not None else 0.,parent_idx/23,float(default)]
        if len(values)!=14 or not all(math.isfinite(v) for v in values):raise ValueError('Bad static graph')
        features[idx[n]]=values
    return features


def input_width(schema):
    context_width = schema.get('context_width', 18+len(schema['knees'])+len(schema['ankles']))
    return HISTORY_WIDTH + len(schema['tasks']) + STATIC_WIDTH + context_width


def structure_key(design):
    return (round(float(design['thigh_m']),10),round(float(design['calf_m']),10),
            design['knee_motor'],design['ankle_motor'])


def paired_structures(banks):
    indexed={}
    for task,bank in banks.items():
        rows={structure_key(d):d for d in bank['designs']}
        if len(rows)!=len(bank['designs']):raise ValueError('Duplicate physical parameter design')
        indexed[task]=rows
    keys=sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    if not keys:raise ValueError('No shared leg/motor structures')
    return [(key,{task:rows[key] for task,rows in indexed.items()}) for key in keys]


def load_plan(path):
    plan=json.loads(Path(path).read_text(encoding='utf8'))
    if plan.get('kind') not in ('rough_backpack_distillation_plan_v2','rough_backpack_amp_distillation_plan_v3') or plan.get('structure_scope')!='teacher_intersection_only':
        raise ValueError('Only strict intersection plans are supported; old union plans are retired')
    if list(plan['teachers'])!=plan['schema']['tasks'] or plan['task_ids']!={t:i for i,t in enumerate(plan['schema']['tasks'])}:
        raise ValueError('Task order/ID mismatch')
    if sha(plan['shared_bank'])!=plan['shared_bank_sha256']:raise ValueError('Shared design bank changed')
    for teacher in plan['teachers'].values():
        if sha(teacher['checkpoint'])!=teacher['checkpoint_sha256']:raise ValueError('Teacher changed')
        if sha(teacher['bank'])!=teacher['bank_sha256']:raise ValueError('Physical bank changed')
    keys=[]
    for task,teacher in plan['teachers'].items():
        bank=json.loads(Path(teacher['bank']).read_text())
        chosen={d['id']:d for d in bank['designs'] if d['id'] in teacher['design_ids']}
        if len(chosen)!=len(teacher['design_ids']):raise ValueError('Selected native teacher design missing')
        keys.append({structure_key(d) for d in chosen.values()})
    if any(k!=keys[0] for k in keys) or keys[0]!={tuple(k) for k in plan['selected_structure_keys']}:
        raise ValueError('Tasks must use exactly the same paired structure library')
    return plan
