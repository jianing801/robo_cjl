"""Hash native geometry layers and the URDF's external mesh dependencies."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mesh_sources(bank):
    paths=set()
    for design in bank['designs']:
        urdf=Path(design['urdf']).resolve()
        print('CACHE_VALIDATE_URDF_BEGIN',design['id'],str(urdf),flush=True)
        for mesh in ET.parse(urdf).findall('.//mesh'):
            name=mesh.get('filename','')
            if name.startswith('file://'):name=name[7:]
            if not name or '://' in name:raise ValueError('mesh must resolve to a local file before training')
            path=Path(name)
            paths.add(str((path if path.is_absolute() else urdf.parent/path).resolve()))
    result={}
    for path in sorted(paths):
        print('CACHE_VALIDATE_MESH_BEGIN',path,flush=True)
        result[path]=sha(path)
    return result

def asset_files(directory):
    directory=Path(directory).resolve()
    return {p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and not p.name.endswith(('.tmp','.tmp.usd'))}

def validate_asset(row):
    usd=Path(row['usd']).resolve();directory=usd.parent
    files=row.get('files_sha256')
    if not isinstance(files,dict) or not {usd.name,'.asset_hash','config.yaml'}.issubset(files):
        raise ValueError('native cache lacks complete geometry hashes; rebuild cache')
    if files[usd.name]!=row['usd_sha256']:
        raise ValueError('native asset identity changed')
    converter_bytes=None
    for name,digest in files.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('native geometry dependency changed: '+name)
        print('CACHE_VALIDATE_FILE_BEGIN',str(path),flush=True)
        if name=='.asset_hash':
            converter_bytes=path.read_bytes()
            actual=hashlib.sha256(converter_bytes).hexdigest()
        else:actual=sha(path)
        if actual!=digest:
            raise ValueError('native geometry dependency changed: '+name)
    if converter_bytes.decode('utf8').strip()!=row['converter_hash']:
        raise ValueError('native asset identity changed')
