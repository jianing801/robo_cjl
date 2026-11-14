"""Hash native geometry layers and the URDF's external mesh dependencies."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mesh_sources(bank):
    paths=set()
    for design in bank['designs']:
        urdf=Path(design['urdf']).resolve()
        for mesh in ET.parse(urdf).findall('.//mesh'):
            name=mesh.get('filename','')
            if name.startswith('file://'):name=name[7:]
            if not name or '://' in name:raise ValueError('mesh must resolve to a local file before training')
            path=Path(name)
            paths.add(str((path if path.is_absolute() else urdf.parent/path).resolve()))
    return {path:sha(path) for path in sorted(paths)}

def asset_files(directory):
    directory=Path(directory).resolve()
    return {p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and not p.name.endswith(('.tmp','.tmp.usd'))}

def validate_asset(row):
    usd=Path(row['usd']).resolve();directory=usd.parent
    files=row.get('files_sha256')
    if not isinstance(files,dict) or not {usd.name,'.asset_hash','config.yaml'}.issubset(files):
        raise ValueError('native cache lacks complete geometry hashes; rebuild cache')
    for name,digest in files.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or sha(path)!=digest:
            raise ValueError('native geometry dependency changed: '+name)
    if sha(usd)!=row['usd_sha256'] or (directory/'.asset_hash').read_text().strip()!=row['converter_hash']:
        raise ValueError('native asset identity changed')
