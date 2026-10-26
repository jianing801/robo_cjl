"""No Isaac required: guard local version selection and native init parity."""
import sys
from types import SimpleNamespace as NS, ModuleType
import pytest
from hybrid_importer import has_known_constructor, install_local_importer_adapter, close_without_masking


class Native:
    def __init__(self,cfg):
        if True:
            manager.enable('isaacsim.asset.importer.urdf-2.4.31')
        from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface
        self._urdf_interface = acquire_urdf_interface()
        super().__init__(cfg=cfg)
    def _convert_asset(self,cfg):return cfg


def setup(monkeypatch,tmp_path,version='2.4.30',enabled=None):
    path=tmp_path/'isaacsim/exts/isaacsim.asset.importer.urdf'
    (path/'config').mkdir(parents=True)
    (path/'config/extension.toml').write_text('[package]\nversion="'+version+'"')
    import hybrid_importer
    monkeypatch.setattr(hybrid_importer.sysconfig,'get_path',lambda name:str(tmp_path))
    calls=[]
    class Manager:
        active=enabled
        def get_enabled_extension_id(self,name):return self.active
        def set_extension_enabled_immediate(self,name,value):
            calls.append((name,value));self.active=name
        def get_extension_path(self,name):return str(path)
    manager=Manager()
    converters=NS(UrdfConverter=Native)
    monkeypatch.setitem(sys.modules,'isaaclab.sim',NS(converters=converters))
    class Base:
        def __init__(self,cfg):self.cfg=cfg
    monkeypatch.setitem(sys.modules,'isaaclab.sim.converters.asset_converter_base',NS(AssetConverterBase=Base))
    app=ModuleType('omni.kit.app');app.get_app=lambda:NS(get_extension_manager=lambda:manager)
    kit=ModuleType('omni.kit');kit.app=app
    omni=ModuleType('omni');omni.kit=kit
    for name,module in [('omni',omni),('omni.kit',kit),('omni.kit.app',app)]:monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setitem(sys.modules,'isaacsim.asset.importer.urdf._urdf',NS(acquire_urdf_interface=lambda:'local-interface'))
    return converters,calls,path


def test_only_local_2430_enabled_no_native_file_edits(monkeypatch,tmp_path):
    converters,calls,path=setup(monkeypatch,tmp_path)
    manifest=(path/'config/extension.toml').read_bytes()
    audit=install_local_importer_adapter()
    assert audit['adapter'] and audit['installed_version']=='2.4.30'
    assert calls==[('isaacsim.asset.importer.urdf-2.4.30',True)]
    obj=converters.UrdfConverter('cfg')
    assert obj.cfg=='cfg' and obj._urdf_interface=='local-interface'
    assert converters.UrdfConverter._convert_asset is Native._convert_asset
    assert install_local_importer_adapter()==audit and len(calls)==1
    assert (path/'config/extension.toml').read_bytes()==manifest


def test_other_installed_version_untouched(monkeypatch,tmp_path):
    converters,calls,_=setup(monkeypatch,tmp_path,'2.4.31')
    assert not install_local_importer_adapter()['adapter']
    assert converters.UrdfConverter is Native and not calls


def test_enabled_other_version_rejected(monkeypatch,tmp_path):
    _,calls,_=setup(monkeypatch,tmp_path,enabled='isaacsim.asset.importer.urdf-2.4.31')
    with pytest.raises(RuntimeError,match='refusing version switch'):install_local_importer_adapter()
    assert not calls


def test_unknown_constructor_not_replaced():
    def changed(self,cfg):self.extra='unknown'
    assert has_known_constructor(Native.__init__)
    assert not has_known_constructor(changed)


def test_secondary_cleanup_does_not_mask_primary(capsys):
    class Broken:
        def close(self):raise AttributeError('event_manager')
    close_without_masking(Broken(),True)
    assert 'SECONDARY_CLEANUP_ERROR' in capsys.readouterr().out
    with pytest.raises(AttributeError):close_without_masking(Broken(),False)
