"""CPU checks of checkpoint-bound active-asset verification."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parent/'engine')]
from robolab.morphology_context import LINKS, bind_physical_context
from robolab.physical_snapshot import physical_model
from hybrid_support import source_identity
from contract import sha
import rollout


def fixture(tmp_path):
    model = physical_model()
    pair = model['knee_motor'] + '__' + model['ankle_motor']
    rows = []
    mesh = tmp_path/'robot.stl'
    mesh.write_bytes(b'mesh')
    for i in range(2):
        urdf = tmp_path/f'{i}.urdf'
        links = ''.join(f'<link name="{name}"><inertial><mass value="{i+1}"/><inertia ixx="1" '
            'iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>' for name in LINKS)
        urdf.write_text(f'<robot>{links}<mesh filename="robot.stl"/></robot>')
        report = tmp_path/f'{i}.json'
        report.write_text('{}')
        rows.append(dict(id=str(i), urdf=str(urdf), urdf_sha256=sha(urdf), report=str(report),
            report_sha256=sha(report), thigh_m=.3+i*.1, calf_m=.4, pair_id=pair,
            knee_motor=model['knee_motor'], ankle_motor=model['ankle_motor']))
    sources = source_identity('stairs')
    bank = dict(task_key='stairs', configuration={}, designs=rows, motor_pairs={pair:model}, training=dict(source_hashes=sources),
        original_template=str(mesh), original_template_sha256=sha(mesh), reference_contract_source=str(mesh),
        reference_contract_sha256=sha(mesh))
    bind_physical_context(bank)
    path = tmp_path/'bank.json'
    path.write_text(json.dumps(bank))
    spec = dict(bank=str(path), original_bank=str(path), bank_sha256=sha(path), full_body_torque=True)
    identity = dict(bank_sha256=sha(path), task='stairs', full_bank_context_is_stable=True,
        native_sources=sources, mesh_sources={str(mesh):sha(mesh)})
    return bank, spec, identity


def fail_native(_path):
    pytest.fail('unexpected full-bank read')


def test_active_verifier_keeps_full_bank_and_does_not_open_inactive_assets(tmp_path):
    bank, spec, identity = fixture(tmp_path)
    Path(bank['designs'][0]['urdf']).unlink()
    Path(bank['designs'][0]['report']).unlink()
    result = rollout.load_rollout_bank(spec, 'stairs', ['1'], identity, fail_native)
    assert result == bank


@pytest.mark.parametrize('change', ['urdf', 'report', 'context', 'mesh', 'bank', 'bound_base', 'gp_authorization'])
def test_active_verifier_refuses_changed_inputs(tmp_path, change):
    bank, spec, identity = fixture(tmp_path)
    task = 'stairs'
    if change in ('urdf', 'report'):
        Path(bank['designs'][1][change]).write_text('changed')
    elif change == 'mesh':
        identity['mesh_sources'] = {next(iter(identity['mesh_sources'])):'wrong'}
    elif change == 'bank':
        Path(spec['bank']).write_text('changed')
    elif change in ('context', 'gp_authorization'):
        if change == 'context':
            bank['designs'][1]['context'][0] = .2
        else:
            bank['configuration']['gp_extrapolation'] = True
        Path(spec['bank']).write_text(json.dumps(bank))
        spec['bank_sha256'] = identity['bank_sha256'] = sha(spec['bank'])
    else:
        task = identity['task'] = bank['task_key'] = 'backpack'
        identity['native_sources'] = bank['training']['source_hashes'] = source_identity(task)
        base = tmp_path/'base.json'
        base.write_text('{}')
        bank['base_morphology_binding'] = dict(bank=str(base), sha256=sha(base))
        bank['designs'][1].update(base_urdf=bank['designs'][0]['urdf'], base_urdf_sha256='wrong')
        Path(spec['bank']).write_text(json.dumps(bank))
        spec['bank_sha256'] = identity['bank_sha256'] = sha(spec['bank'])
    with pytest.raises(ValueError):
        rollout.load_rollout_bank(spec, task, ['1'], identity, fail_native)


def test_no_native_bank_attestation_uses_existing_validator():
    assert rollout.load_rollout_bank({'bank':'legacy.json'}, 'stairs', ['x'], {}, str) == 'legacy.json'


def test_indexed_cache_reads_only_selected_native_directories(tmp_path):
    root = tmp_path/'run'
    bank = root/'banks/rough/bank.json'
    bank.parent.mkdir(parents=True)
    bank.write_text('{}')
    cache = root/'prepared'
    (cache/'b').mkdir(parents=True)
    usd = cache/'b/robot.usd'
    usd.write_text('usd')
    (usd.parent/'config.yaml').write_text('config')
    (usd.parent/'.asset_hash').write_text('native-key')
    proof = cache/'completed.json'
    proof.write_text(json.dumps(dict(result='NATIVE_ASSET_CACHE_COMPLETE', cache_schema_version=2,
        bank_sha256=sha(bank), assets=[dict(design='a'),dict(design='b', urdf_sha256='b-urdf',
            converter_hash='native-key', usd=str(usd))])))
    index = root/'启动记录/rough_asset_cache.json'
    index.parent.mkdir()
    index.write_text(json.dumps(dict(cache=str(cache), bank_sha256=sha(bank), completed_sha256=sha(proof))),encoding='utf8')
    spec = dict(bank=str(bank), bank_sha256=sha(bank))
    designs = [dict(id='b', urdf_sha256='b-urdf')]
    assert rollout.indexed_rollout_cache(spec, 'stairs', designs) == [usd.parent]
    proof.write_text('{}')
    with pytest.raises(ValueError, match='proof'):
        rollout.indexed_rollout_cache(spec, 'stairs', designs)
