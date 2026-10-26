"""Rough-only adapter for Lab's 2.4.31 pin on an installed 2.4.30.

Never edits installed packages, downloads extensions, or falsifies versions.
Other native converters and all physical conversion methods remain inherited.
"""
import ast
import inspect
from pathlib import Path
import sysconfig
import textwrap
import tomllib


def has_known_constructor(constructor):
    body=ast.parse(textwrap.dedent(inspect.getsource(constructor))).body[0].body
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant):
        body=body[1:]
    if len(body)!=4 or not isinstance(body[0],ast.If):return False
    if 'isaacsim.asset.importer.urdf-2.4.31' not in ast.unparse(body[0]):return False
    expected=ast.parse('from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface\n'
        'self._urdf_interface = acquire_urdf_interface()\nsuper().__init__(cfg=cfg)').body
    return all(ast.dump(a)==ast.dump(b) for a,b in zip(body[1:],expected))


def install_local_importer_adapter():
    from isaaclab.sim import converters
    from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
    native=converters.UrdfConverter
    if getattr(native,'_rough_local_importer',False):return native._rough_importer_audit
    if not has_known_constructor(native.__init__):
        return dict(adapter=False,reason='native constructor not the known 2.4.31 pin; unchanged')
    path=Path(sysconfig.get_path('purelib'))/'isaacsim/exts/isaacsim.asset.importer.urdf'
    config=path/'config/extension.toml'
    if not config.is_file():
        raise RuntimeError('No installed URDF importer manifest; refusing downloads or version changes')
    version=tomllib.loads(config.read_text(encoding='utf8'))['package']['version']
    if version!='2.4.30':
        return dict(adapter=False,installed_version=version,reason='not the installed 2.4.30 case; unchanged')
    import omni.kit.app
    manager=omni.kit.app.get_app().get_extension_manager()
    extension='isaacsim.asset.importer.urdf-2.4.30'
    enabled=manager.get_enabled_extension_id('isaacsim.asset.importer.urdf')
    if enabled and not (enabled==extension or enabled.startswith(extension+'+')):
        raise RuntimeError(f'Unexpected enabled URDF importer {enabled}; refusing version switch')
    if not enabled:
        manager.set_extension_enabled_immediate(extension,True)
    enabled=manager.get_enabled_extension_id('isaacsim.asset.importer.urdf')
    if not enabled or not (enabled==extension or enabled.startswith(extension+'+')):
        raise RuntimeError('Installed URDF importer 2.4.30 did not enable; no download fallback')
    actual=Path(manager.get_extension_path(enabled)).resolve()
    if actual!=path.resolve():
        raise RuntimeError(f'URDF importer path mismatch: {actual}; refusing unknown package')
    from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface
    audit=dict(adapter=True,installed_version=version,enabled_id=enabled,path=str(actual),
        native_pin_bypassed='2.4.31',installed_files_modified=False,physics_methods_unchanged=True)
    class LocalUrdfConverter(native):
        _rough_local_importer=True
        _rough_importer_audit=audit
        def __init__(self,cfg):
            self._urdf_interface=acquire_urdf_interface()
            AssetConverterBase.__init__(self,cfg=cfg)
    converters.UrdfConverter=LocalUrdfConverter
    return audit


def close_without_masking(resource, primary_error):
    """Preserve authoritative initialization failure if native close also fails."""
    try:
        resource.close()
    except Exception:
        if not primary_error:raise
        import traceback
        print('SECONDARY_CLEANUP_ERROR (primary failure retained)',flush=True)
        traceback.print_exc()
