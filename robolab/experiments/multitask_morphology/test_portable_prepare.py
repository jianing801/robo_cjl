"""Verify one-shot bank assets remain usable with the legacy path absent."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET


spec = importlib.util.spec_from_file_location(
    'portable_prepare', Path(__file__).with_name('prepare.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_copy_selected_banks_and_reference_payloads_without_legacy_dependencies(tmp_path):
    source, target = tmp_path/'old', tmp_path/'new'
    source.mkdir()
    (source/'mesh.stl').write_bytes(b'sample mesh')
    template = source/'template.urdf'
    template.write_text('<robot><link name="base"><visual><geometry><mesh filename="mesh.stl"/>'
                        '</geometry></visual></link></robot>', encoding='utf8')
    packed = source/'packed.urdf'
    text = template.read_text(encoding='utf8')
    packed.write_text(text.replace('</robot>', '<link name="backpack_link"/></robot>'), encoding='utf8')
    contract = source/'contract.json'
    contract.write_text('{}', encoding='utf8')
    report = source/'original.report.json'
    report.write_text(json.dumps(dict(template=str(template))), encoding='utf8')
    common = dict(snapshot_root=str(tmp_path/'repository_vendor'),
                  original_template=str(template), original_template_sha256=sha(template),
                  reference_contract_source=str(contract), reference_contract_sha256=sha(contract))
    rough = dict(id='retained', urdf=str(template), urdf_sha256=sha(template),
                 report=str(report), report_sha256=sha(report))
    pack = dict(rough, urdf=str(packed), urdf_sha256=sha(packed),
                base_urdf=str(template), base_urdf_sha256=sha(template))
    original = {task: dict(common, designs=[row])
                for task, row in zip(prepare.TASKS, [rough, pack, rough], strict=True)}
    banks = {task: deepcopy(bank) for task, bank in original.items()}
    prepare.relocate_assets(banks, source, target)
    repeated = {task: deepcopy(bank) for task, bank in original.items()}
    prepare.relocate_assets(repeated, source, target)
    assert repeated == banks
    assert banks['rough']['designs'] == banks['amp']['designs']
    bank_path = target/'banks/amp/bank.json'
    prepare.publish_json(bank_path, banks['amp'])

    clip = source/'walk.pkl'
    clip.write_bytes(b'existing serialized reference')
    library = dict(designs={'retained': dict(directory=str(source), clips={'walk':sha(clip)},
                                           geometry_key='a'*64)}, reference_execution={'warp_cache':str(source/'warp')})
    prepare.relocate_references(library, banks['amp'], bank_path, source, target)
    assert not (source/'banks').exists()
    disconnected = tmp_path/'old_disconnected'
    assert disconnected.parent.resolve() == source.parent.resolve() == tmp_path.resolve()
    source.rename(disconnected)
    for value in banks.values():
        prepare.reject_legacy_paths(value, source)
        row = value['designs'][0]
        for field in ('urdf', 'report'):
            assert sha(Path(row[field])) == row[field+'_sha256']
        mesh = Path(ET.parse(row['urdf']).find('.//mesh').get('filename'))
        assert mesh.is_relative_to(target) and mesh.read_bytes() == b'sample mesh'
        assert Path(json.loads(Path(row['report']).read_text(encoding='utf8'))['template']).is_file()
    migrated = library['designs']['retained']
    assert sha(Path(migrated['directory'])/'walk.pkl') == migrated['clips']['walk']


def test_shared_repository_mesh_preserves_urdf_hash(tmp_path):
    source, target = tmp_path/'old', tmp_path/'new'
    source.mkdir()
    mesh = tmp_path/'repository_mesh.stl'
    mesh.write_bytes(b'shared mesh')
    urdf = source/'robot.urdf'
    urdf.write_text(f'<robot><link><visual><geometry><mesh filename="{mesh.as_posix()}"/>'
                    '</geometry></visual></link></robot>', encoding='utf8')
    staged, digest = prepare.stage_urdf(urdf, sha(urdf), target/'robot.urdf', source, target, {})
    assert digest == sha(urdf) and Path(staged).read_bytes() == urdf.read_bytes()
