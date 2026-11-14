"""Never reuse another morphology bank or a changed native asset."""
import json
from pathlib import Path
import pytest
from hybrid_native_cache import validate_cache,ensure_cache,sha

def make_cache(tmp_path):
    from hybrid_assets import asset_files,mesh_sources
    mesh=tmp_path/'mesh.stl';mesh.write_text('mesh source')
    urdf=tmp_path/'robot.urdf';urdf.write_text(f'<robot><link name="base"><visual><geometry><mesh filename="{mesh.as_posix()}"/></geometry></visual></link></robot>')
    data={'designs':[{'id':'one','urdf':str(urdf),'urdf_sha256':sha(urdf)}]}
    bank=tmp_path/'bank.json';bank.write_text(json.dumps(data))
    root=tmp_path/'run';root.mkdir();cache=root/'native_cache_verified';asset=cache/'one';asset.mkdir(parents=True)
    usd=asset/'robot.usd';usd.write_text('native asset')
    (asset/'.asset_hash').write_text('converter');(asset/'config.yaml').write_text('valid')
    (asset/'geometry.usd').write_text('actual collision geometry')
    proof=dict(result='NATIVE_ASSET_CACHE_COMPLETE',cache_schema_version=2,mesh_sources=mesh_sources(data),bank_sha256=sha(bank),assets=[dict(design='one',
        usd=str(usd.resolve()),usd_sha256=sha(usd),urdf_sha256=sha(urdf),converter_hash='converter',files_sha256=asset_files(asset))])
    (cache/'completed.json').write_text(json.dumps(proof))
    return root,cache,bank,usd

def test_verified_cache_reused_before_launching_worker(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path)
    def unexpected(*args):raise AssertionError('must not launch another converter')
    assert ensure_cache(root,bank,'python','',{},unexpected)==cache
    usd.write_text('changed')
    with pytest.raises(ValueError):validate_cache(cache,bank)

def test_wrong_bank_and_incomplete_cache_rejected(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path)
    other=tmp_path/'other.json';other.write_text('{"designs":[]}')
    with pytest.raises(ValueError):validate_cache(cache,other)
    (cache/'one/config.yaml').unlink()
    with pytest.raises(ValueError):validate_cache(cache,bank)

def test_completed_proof_excludes_abandoned_partial_directories(tmp_path):
    from hybrid_native_cache import cache_directories
    root,cache,bank,usd=make_cache(tmp_path)
    partial=cache/'partial';partial.mkdir()
    (partial/'.asset_hash').write_text('other')
    (partial/'config.yaml').write_text('partial')
    (partial/'robot.usd').write_text('not an approved conversion')
    assert cache_directories(cache,bank)==[usd.parent.resolve()]

def test_reuse_verified_batch_keeps_original_files(tmp_path):
    from native_cache_coordinator import reuse_verified_batches
    root,cache,bank,usd=make_cache(tmp_path)
    artifact=tmp_path/'proofs/batch_0000';artifact.mkdir(parents=True)
    artifact.joinpath('summary.json').write_bytes((cache/'completed.json').read_bytes())
    destination=tmp_path/'new_cache';destination.mkdir()
    parsed=json.loads(bank.read_text());original=usd.read_bytes()
    rows=reuse_verified_batches([artifact.parent],parsed,sha(bank),destination)
    assert len(rows)==1 and Path(rows[0]['usd']).read_bytes()==original
    assert usd.read_bytes()==original
    usd.write_text('corrupt original')
    other=tmp_path/'another';other.mkdir()
    with pytest.raises(ValueError):reuse_verified_batches([artifact.parent],parsed,sha(bank),other)

def test_geometry_layer_corruption_is_detected_without_main_usd_change(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path);before=sha(usd)
    (usd.parent/'geometry.usd').write_text('corrupted collision geometry')
    assert sha(usd)==before
    with pytest.raises(ValueError,match='geometry dependency'):validate_cache(cache,bank)

def test_external_mesh_change_is_not_hidden_by_unchanged_urdf(tmp_path):
    root,cache,bank,usd=make_cache(tmp_path)
    (tmp_path/'mesh.stl').write_text('changed mesh')
    with pytest.raises(ValueError,match='mesh dependencies'):validate_cache(cache,bank)

def test_fullbody_cannot_accept_a_partial_unproved_cache(tmp_path):
    from hybrid_native_cache import cache_directories
    root,cache,bank,usd=make_cache(tmp_path)
    (cache/'completed.json').unlink()
    with pytest.raises(ValueError,match='completed native cache'):cache_directories(cache,bank,require_complete=True)

def test_cache_interruption_is_not_mistaken_for_started_ppo(tmp_path):
    from hybrid_native_cache import training_attempt_started
    folder=tmp_path/'attempts/cache_001/train';folder.mkdir(parents=True)
    (folder/'launch.json').write_text(json.dumps({'command':['python','-u','prepare_native_cache.py']}))
    assert not training_attempt_started(tmp_path)
    session=tmp_path/'attempts/session_001/train';session.mkdir(parents=True)
    (session/'launch.json').write_text(json.dumps({'command':['python','-u','persistent_worker.py']}))
    assert training_attempt_started(tmp_path)
