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

def _mesh_sources_deep(bank,report=True):
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


def validate_asset_resume(row):
    """恢复检查：首次深检，后续元数据未变时跳过大USD读取。"""
    import json,os,stat,time
    usd=Path(row['usd']).resolve();directory=usd.parent
    receipt=directory.parent/('.resume_'+directory.name+'.json')
    if receipt.is_symlink():raise ValueError('resume receipt is a symlink')
    signatures={}
    for name in row.get('files_sha256',{}):
        path=(directory/name).resolve()
        if not path.is_relative_to(directory):raise ValueError('asset path escapes directory')
        value=path.stat()
        if not stat.S_ISREG(value.st_mode) or value.st_size<=0:raise ValueError('empty/nonregular asset')
        signatures[name]=[value.st_size,value.st_mtime_ns,value.st_dev,value.st_ino]
    binding=dict(row=row,files=signatures)
    previous=json.loads(receipt.read_text(encoding='utf8')) if receipt.is_file() else None
    if previous==binding:return
    validate_asset(row)
    temporary=receipt.with_name(receipt.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temporary.open('x',encoding='utf8') as stream:
        json.dump(binding,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,receipt)


# BEGIN_TRAINING_MESH_CACHE

_training_mesh_cache={}

def urdf_mesh_paths(path):
    urdf=Path(path).resolve()
    paths=set()
    for mesh in ET.parse(urdf).findall('.//mesh'):
        name=mesh.get('filename','')
        if name.startswith('file://'):name=name[7:]
        if not name or '://' in name:raise ValueError('mesh must resolve to a local file before training')
        mesh_path=Path(name)
        paths.add(str((mesh_path if mesh_path.is_absolute() else urdf.parent/mesh_path).resolve()))
    return paths

def mesh_sources(bank,report=True,*,use_cache=False):
    """训练进程内复用源校验；独立USD准备和手动深检默认仍完整读取。"""
    if not use_cache:return _mesh_sources_deep(bank,report)
    import stat
    key=tuple(str(d['urdf']) for d in bank['designs'])
    def signatures(paths):
        result={}
        for name in paths:
            path=Path(name);value=path.stat()
            if not stat.S_ISREG(value.st_mode):raise ValueError('source is not a regular file: '+name)
            result[name]=(value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_dev,value.st_ino)
        return result
    previous=_training_mesh_cache.get(key)
    if previous is not None:
        value,stamps=previous
        if signatures(stamps)==stamps:
            if report:progress('TRAIN_SOURCE_CACHE_HIT',len(key),len(key))
            return dict(value)
        _training_mesh_cache.pop(key,None)
    # 冷启动先发现依赖并取元数据，再完整哈希，最后确认读取期间没有变化。
    sources=set(key)
    for name in key:
        sources.update(urdf_mesh_paths(name))
    before=signatures(sources)
    value=_mesh_sources_deep(bank,report)
    after=signatures(sources)
    if before!=after:raise ValueError('source changed during training cache verification')
    if len(_training_mesh_cache)>=8:_training_mesh_cache.pop(next(iter(_training_mesh_cache)))
    _training_mesh_cache[key]=(dict(value),after)
    return value
