"""Hash native geometry layers and the URDF's external mesh dependencies."""
import hashlib
from time import monotonic as progress_clock
from pathlib import Path
import xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

_progress_times={}

def progress(phase,done,total):
    # 每阶段最多约20次常规进度；不输出文件或结构路径。
    step=max(1,(total+19)//20)
    now=progress_clock();key=(phase,total)
    if done in (0,total) or done%step==0 or now-_progress_times.get(key,now)>=5:
        print(phase,done,'/',total,flush=True)
        _progress_times[key]=now

def mesh_sources(bank,report=True):
    paths=set()
    report=report and len(bank['designs'])>1
    if report:progress('CACHE_URDF_PROGRESS',0,len(bank['designs']))
    for index,design in enumerate(bank['designs'],1):
        urdf=Path(design['urdf']).resolve()
        for mesh in ET.parse(urdf).findall('.//mesh'):
            name=mesh.get('filename','')
            if name.startswith('file://'):name=name[7:]
            if not name or '://' in name:raise ValueError('mesh must resolve to a local file before training')
            path=Path(name)
            paths.add(str((path if path.is_absolute() else urdf.parent/path).resolve()))
        if report:progress('CACHE_URDF_PROGRESS',index,len(bank['designs']))
    result={}
    if report:progress('CACHE_MESH_PROGRESS',0,len(paths))
    for index,path in enumerate(sorted(paths),1):
        result[path]=sha(path)
        if report:progress('CACHE_MESH_PROGRESS',index,len(paths))
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
            raise ValueError('native geometry dependency changed: '+str(path))
        if name=='.asset_hash':
            converter_bytes=path.read_bytes()
            actual=hashlib.sha256(converter_bytes).hexdigest()
        else:actual=sha(path)
        if actual!=digest:
            raise ValueError('native geometry dependency changed: '+str(path))
    if converter_bytes.decode('utf8').strip()!=row['converter_hash']:
        raise ValueError('native asset identity changed')
