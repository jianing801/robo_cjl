"""Audit full common morphology coverage and actual attached URDF contents."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def base_elements(path):
    root=ET.parse(path).getroot()
    return {(element.tag,element.get('name')):ET.tostring(element,encoding='unicode').strip()
            for element in root if element.get('name') not in ('backpack_link','backpack_mount_joint')}

def audit(rough_path,amp_path,backpack_path):
    paths=[Path(p) for p in (rough_path,amp_path,backpack_path)]
    rough,amp,pack=[json.loads(p.read_text(encoding='utf8')) for p in paths]
    if [b['task_key'] for b in (rough,amp,pack)]!=['stairs','run','backpack']:raise ValueError('task identity mismatch')
    for field in ('configuration','knees','ankles','context_width','ranges_m','motor_pairs','canonical_semantic_contract'):
        if not rough[field]==amp[field]==pack[field]:raise ValueError('different morphology field: '+field)
    for field in ('schema_version','context_schema','context_fields','context_normalization'):
        if not rough.get(field)==amp.get(field)==pack.get(field):raise ValueError('different context field: '+field)
    config=json.loads((Path(__file__).parent/'configs/rough.json').read_text(encoding='utf8'))
    grid=config['grid']
    if (rough['configuration']!=grid or set(rough['knees'])!=set(grid['knee_pool'])
            or set(rough['ankles'])!=set(grid['ankle_pool'])
            or rough.get('context_schema')!=config.get('context_schema') or rough['context_width']!=28):
        raise ValueError('three-task audit expects the current formal motor pool and context')
    if rough['configuration'].get('gp_extrapolation'):
        if not rough['training']['gp_extrapolation']==amp['training'].get('gp_extrapolation')==pack['training'].get('gp_extrapolation'):
            raise ValueError('different GP extrapolation authorization')
    if pack['base_morphology_binding']['sha256']!=sha(paths[0]) or amp['rough_morphology_binding']['rough_bank_sha256']!=sha(paths[0]):
        raise ValueError('different bound Rough bank')
    rows=[b['designs'] for b in (rough,amp,pack)]
    if not len(rows[0])==len(rows[1])==len(rows[2]):raise ValueError('different design counts')
    seen=set();pairs=Counter();lengths=Counter()
    for r,a,p in zip(*rows):
        for key in ('id','thigh_m','calf_m','knee_motor','ankle_motor','context','pair_id','root_height_compensation_m'):
            if not r[key]==a[key]==p[key]:raise ValueError('different structure/context '+key)
        if not r.get('gp_extrapolation')==a.get('gp_extrapolation')==p.get('gp_extrapolation'):
            raise ValueError('different per-design GP extrapolation markers')
        if r['id'] in seen:raise ValueError('duplicate structure')
        seen.add(r['id']);pairs[r['knee_motor'],r['ankle_motor']]+=1;lengths[r['thigh_m'],r['calf_m']]+=1
        for d in (r,a,p):
            if sha(d['urdf'])!=d['urdf_sha256']:raise ValueError('changed actual URDF')
        if r['urdf_sha256']!=a['urdf_sha256'] or p['base_urdf_sha256']!=r['urdf_sha256']:
            raise ValueError('different physical base identity')
        if base_elements(r['urdf'])!=base_elements(p['urdf']):raise ValueError('backpack altered base robot')
        if ET.parse(p['urdf']).find("link[@name='backpack_link']") is None:raise ValueError('missing backpack')
    length_count=grid['length_count']**2
    motor_count=grid['knee_motor_count']*grid['ankle_motor_count']
    if (len(seen)!=length_count*motor_count or len(pairs)!=motor_count or set(pairs.values())!={length_count}
            or len(lengths)!=length_count or set(lengths.values())!={motor_count}):
        raise ValueError('incomplete motor and leg-length coverage')
    return dict(result='THREE_TASK_COMMON_MORPHOLOGY_VERIFIED',designs=len(seen),
        ranges_m=rough['ranges_m'],gp_extrapolation_authorized=rough['configuration'].get('gp_extrapolation',False),
        designs_outside_registered_gp_range=sum(any(x['outside_registered_prediction_range'] for x in d.get('gp_extrapolation',{}).values()) for d in rows[0]),
        knees=sorted({k for k,a in pairs}),ankles=sorted({a for k,a in pairs}),motor_pairs=len(pairs),
        designs_per_motor_pair=sorted(set(pairs.values())),length_pairs=len(lengths),
        designs_per_length_pair=sorted(set(lengths.values())),context_width=rough['context_width'],
        context_schema=rough.get('context_schema'),
        all_actual_urdf_hashes_verified=True,backpack_base_elements_identical=True,
        bank_sha256={b['task_key']:sha(p) for b,p in zip((rough,amp,pack),paths)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rough-bank',required=True)
    parser.add_argument('--amp-bank',required=True);parser.add_argument('--backpack-bank',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=audit(args.rough_bank,args.amp_bank,args.backpack_bank)
    with Path(args.output).open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
