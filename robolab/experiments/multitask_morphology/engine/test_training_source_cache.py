"""训练源缓存命中、变化失效和深检旁路。"""
from pathlib import Path
import pytest
import hybrid_assets as assets


def bank_fixture(tmp_path):
    mesh=tmp_path/'mesh.stl';mesh.write_text('old mesh')
    urdf=tmp_path/'robot.urdf';urdf.write_text('<robot><mesh filename="mesh.stl"/></robot>')
    return {'designs':[{'urdf':str(urdf)}]},mesh,urdf


def test_hit_does_not_parse_or_hash_and_returns_copy(tmp_path,monkeypatch):
    bank,mesh,urdf=bank_fixture(tmp_path)
    first=assets.mesh_sources(bank,use_cache=True)
    def forbidden(*a,**k):raise AssertionError('cache must skip content read')
    monkeypatch.setattr(assets.ET,'parse',forbidden)
    monkeypatch.setattr(assets,'sha',forbidden)
    second=assets.mesh_sources(bank,use_cache=True)
    assert first==second and first is not second
    first.clear()
    assert assets.mesh_sources(bank,use_cache=True)==second


@pytest.mark.parametrize('change',['mesh','urdf','missing'])
def test_change_invalidates_cache(tmp_path,change):
    bank,mesh,urdf=bank_fixture(tmp_path);first=assets.mesh_sources(bank,use_cache=True)
    if change=='mesh':
        mesh.write_text('new changed mesh');assert assets.mesh_sources(bank,use_cache=True)!=first
    elif change=='urdf':
        another=tmp_path/'another.stl';another.write_text('other')
        urdf.write_text('<robot><mesh filename="another.stl"/></robot>')
        assert set(assets.mesh_sources(bank,use_cache=True))=={str(another.resolve())}
    else:
        mesh.unlink()
        with pytest.raises(OSError):assets.mesh_sources(bank,use_cache=True)


def test_default_remains_deep_and_does_not_trust_metadata(tmp_path,monkeypatch):
    bank,mesh,urdf=bank_fixture(tmp_path);assets.mesh_sources(bank,use_cache=True)
    original=assets.sha;calls=[]
    def count(path):calls.append(path);return original(path)
    monkeypatch.setattr(assets,'sha',count)
    assets.mesh_sources(bank)
    assert str(mesh.resolve()) in calls
