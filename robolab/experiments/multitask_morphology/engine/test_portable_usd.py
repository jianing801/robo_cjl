"""Check that retained cache assets can be loaded with the source removed."""
from hashlib import md5
import json
from pathlib import Path

import pytest
import yaml

from hybrid_assets import asset_files, mesh_sources, sha
from hybrid_native_cache import validate_cache
from portable_usd import RESULT, check_layers, import_task


def write_bank(path, designs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(task_key='stairs', designs=designs)), encoding='utf8')


def old_asset(source, key, body, mesh):
    asset = source/'资产准备/rough/native_cache_orig'/key
    asset.mkdir(parents=True)
    urdf = source/key/'rpo.urdf'
    urdf.parent.mkdir()
    urdf.write_text(f'<robot><link><visual><geometry><mesh filename="{mesh.as_posix()}"/></geometry></visual></link></robot>',
                    encoding='utf8')
    design = dict(id=key, urdf=str(urdf), urdf_sha256=sha(urdf))
    usd = asset/'rpo.usd'
    usd.write_text(body, encoding='utf8')
    config = dict(fix_base=False, asset_path=str(urdf), usd_dir=str(asset), usd_file_name=None)
    (asset/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
    converter_hash = md5(json.dumps(dict(fix_base=False)).encode() + urdf.read_bytes()).hexdigest()
    (asset/'.asset_hash').write_text(converter_hash, encoding='utf8')
    row = dict(design=key, usd=str(usd), usd_sha256=sha(usd), urdf_sha256=sha(urdf),
               converter_hash=converter_hash, files_sha256=asset_files(asset))
    return design, row


def fake_reader(path):
    value = Path(path).read_text(encoding='utf8')
    return [], [value] if value.endswith('.usd') else [], []


@pytest.mark.parametrize('bad_layer', [False, True])
def test_import_is_independent_and_leaves_only_invalid_design_pending(tmp_path, bad_layer):
    source, target = tmp_path/'legacy', tmp_path/'new'
    mesh = tmp_path/'shared_mesh.stl'
    mesh.write_text('repository mesh')
    old_design, old_row = old_asset(source, 'kept', 'layer', mesh)
    still_needed = [old_design]
    rows = [old_row]
    if bad_layer:
        extra, wrong = old_asset(source, 'pending', str(source/'external.usd'), mesh)
        still_needed.append(extra)
        rows.append(wrong)
    discarded, discarded_row = old_asset(source, '8006', 'layer', mesh)
    old_manifest = source/'banks/rough/bank.json'
    write_bank(old_manifest, still_needed+[discarded])
    Path(discarded_row['usd']).unlink()
    proof = dict(result=RESULT, task='stairs', bank_sha256=sha(old_manifest), cache_schema_version=2,
                 mesh_sources=mesh_sources(dict(designs=still_needed)), assets=rows+[discarded_row])
    complete = Path(old_row['usd']).parent.parent/'completed.json'
    complete.write_text(json.dumps(proof), encoding='utf8')
    before = complete.read_bytes()
    new_designs = []
    for design in still_needed:
        urdf = target/'banks/rough'/design['id']/'rpo.urdf'
        urdf.parent.mkdir(parents=True)
        urdf.write_bytes(Path(design['urdf']).read_bytes())
        new_designs.append(dict(design, urdf=str(urdf)))
    manifest = target/'banks/rough/bank.json'
    write_bank(manifest, new_designs)
    cache = import_task(source, target, 'rough', fake_reader)
    assert complete.read_bytes() == before
    if bad_layer:
        assert cache is None
        batch = target/'资产准备/rough/attempts/cache_portable/train/result/batch_portable/assets'
        assert {p.name for p in batch.iterdir()} == {'kept.json'}
        return
    assert cache is not None
    value = yaml.safe_load((cache/'kept/config.yaml').read_text(encoding='utf8'))
    assert value['asset_path'] == new_designs[0]['urdf'] and value['usd_dir'] == str(cache/'kept')
    source.rename(tmp_path/'old_offline')
    verified = validate_cache(cache, manifest, resume_fast=True)
    assert verified['bank_sha256'] == sha(manifest)
    check_layers(cache/'kept', verified['assets'][0]['files_sha256'], fake_reader)
