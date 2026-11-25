import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import bo_continuous_assets as assets


@pytest.mark.parametrize('key', [(.2,.25,'K','A'),(.4,.45,'K','A'),(.3267894321,.3987654321,'K','A')])
def test_singleton_exact_key_inside_authorized_bounds(key):
    schema = dict(ranges_m={'thigh':[.2,.4], 'calf':[.25,.45]}, knees=['K'], ankles=['A'],
                  context_schema='leg_motor_inertia_v1', shared_structure_keys=[])
    grid = assets.singleton_config(key,schema)['grid']
    assert (grid['length_count'],grid['num_envs'],grid['knee_pool'],grid['ankle_pool']) == (1,4096,['K'],['A'])
    for segment, target, (lo,hi) in zip(('thigh','calf'),key[:2],schema['ranges_m'].values()):
        a,b = grid[segment+'_range_m']
        assert lo <= a < b <= hi
        mid = (a+b)/2
        assert round(mid,10) == target and abs(mid-target) <= 1e-12


def test_grid_membership_and_bounds_rejected():
    schema = dict(ranges_m={'thigh':[.2,.4], 'calf':[.25,.45]}, knees=['K'], ankles=['A'],
                  context_schema='leg_motor_inertia_v1', shared_structure_keys=[[.3,.3,'K','A']])
    for key in [(.3,.3,'K','A'),(.41,.3,'K','A'),(.3,.3,'K','B'),(.300000000001,.3,'K','A')]:
        with pytest.raises(ValueError):
            assets.singleton_config(key,schema)


def test_reuse_completed_stage_and_bound_failed_attempts(tmp_path, monkeypatch):
    attempts=[]
    class BO:
        @staticmethod
        def freeze(path,data):
            path.write_text(json.dumps(data),encoding='utf8')

    def generate(folder):
        attempts.append(folder)
        folder.mkdir()
        if len(attempts) == 1:
            raise RuntimeError('simulated failure')
        (folder/'bank.json').write_text('{}')

    def verify(folder):
        if not (folder/'bank.json').is_file():
            raise ValueError('missing output')

    stage=tmp_path/'rough'
    result=assets.stage_artifact(BO, stage, generate, verify, max_attempts=2)
    assert len(attempts)==2 and attempts[0].exists() and result==attempts[1]
    assert assets.stage_artifact(BO,stage,generate,verify,max_attempts=2)==result
    assert len(attempts)==2
    (result/'bank.json').unlink()
    with pytest.raises(ValueError, match='missing output'):
        assets.stage_artifact(BO,stage,generate,verify,max_attempts=2)


def test_exact_native_common_geometry_and_physical_sources(tmp_path):
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    bo=SimpleNamespace(sha=sha,read=lambda path:json.loads(Path(path).read_text()))
    key=[.3743567801,.2865432109,'K','A']
    schema=dict(ranges_m={'thigh':[.2,.4], 'calf':[.25,.45]}, knees=['K'], ankles=['A'],
                context_schema='leg_motor_inertia_v1', shared_structure_keys=[],
                context_fields=['x'],context_width=1)
    cfg=assets.singleton_config(key,schema)
    gp={'approved':True}
    motors={'K__A':dict(snapshot_root='same',selected={'knee':'K','ankle':'A'},verified_file_sha256={'gp':'hash'})}
    names=['left_knee_joint','right_knee_joint','left_ankle_pitch_joint','right_ankle_pitch_joint']
    body=''.join(f'<joint name="{name}"><origin xyz="0 0 {-key[0 if "knee" in name else 1]}" /></joint>'
                 for name in names)
    rough=tmp_path/'rough.urdf'; amp=tmp_path/'amp.urdf'; pack=tmp_path/'backpack.urdf'
    rough.write_text('<robot>'+body+'</robot>')
    amp.write_bytes(rough.read_bytes())
    pack.write_text('<robot>'+body+'<link name="backpack_link" /></robot>')
    report=tmp_path/'report.json';report.write_text('{}')
    donor={};paths={};base_sha=None
    for task,path in [('stairs',rough),('run',amp),('backpack',pack)]:
        donor[task]=dict(training=dict(source_hashes={'source':task},gp_extrapolation=gp),
                         original_template_sha256='template',reference_contract_sha256='contract',
                         canonical_semantic_contract={'joint_names':names}, motor_pairs=motors)
        row=dict(id='tr_test',thigh_m=key[0],calf_m=key[1],knee_motor=key[2],ankle_motor=key[3],
                 pair_id='K__A',urdf=str(path),urdf_sha256=sha(path),report=str(report),report_sha256=sha(report),
                 context=[0],root_height_compensation_m=.01,gp_extrapolation=gp)
        bank=dict(donor[task],task_key=task,schema_version=3,configuration=cfg['grid'],
                  context_schema=schema['context_schema'],context_fields=schema['context_fields'],
                  context_width=schema['context_width'],knees=['K'],ankles=['A'],ranges_m={'x':'y'},
                  context_normalization={'singleton':True},designs=[row])
        if task=='run':bank['rough_morphology_binding']={'rough_bank_sha256':base_sha}
        if task=='backpack':
            bank['base_morphology_binding']={'sha256':base_sha}
            row.update(base_urdf=str(rough),base_urdf_sha256=sha(rough))
        paths[task]=tmp_path/f'{task}.json';paths[task].write_text(json.dumps(bank))
        if task=='stairs':base_sha=sha(paths[task])
    audit=assets._audit_common(bo,paths,donor,cfg,schema,key)
    assert audit['key']==key and set(audit['actual_urdf_hashes'])=={'stairs','backpack','run'}
    # Hash a tampered URDF into its bank to verify geometry independently of hash.
    amp.write_text(amp.read_text().replace(str(-key[0]),str(-key[0]+.0001)))
    bank=bo.read(paths['run']);bank['designs'][0]['urdf_sha256']=sha(amp)
    paths['run'].write_text(json.dumps(bank))
    with pytest.raises(ValueError,match='leg length differs'):
        assets._audit_common(bo,paths,donor,cfg,schema,key)


def test_stage_attempt_limit_is_retained_across_calls(tmp_path):
    calls=[]
    def generate(folder):
        calls.append(folder)
        raise RuntimeError('not complete')
    def verify(folder):
        raise ValueError('not complete')
    bo=SimpleNamespace(freeze=lambda path,data:path.write_text(json.dumps(data)))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='exhausted 2'):
            assets.stage_artifact(bo,tmp_path/'failed',generate,verify,max_attempts=2)
    assert len(calls)==2
