"""Audit and relocate supplied AMP evaluation assets without changing originals."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from contract import sha,write_new

def main():
    w=Path(__file__).resolve().parents[4]
    repo=w/'rough_hybrid_integration_git';root=w/'am30';dest=root/'code3'
    dest.mkdir(exist_ok=False)
    plan=json.loads((root/'bundle/plan.json').read_text(encoding='utf8'))
    bank=json.loads((root/'bundle/bank.json').read_text(encoding='utf8'))
    paths=subprocess.check_output(['git','-c','safe.directory='+str(repo),'-C',str(repo),'ls-files','-z']).decode().split('\0')
    prefixes=('robolab/robolab/','robolab/configs/','robolab/experiments/amp_hybrid/',
        'robolab/experiments/rough_hybrid/','robolab/data/motions/rpo_lab/','rsl_rl/rsl_rl/')
    for name in paths:
        if not name.startswith(prefixes):continue
        source=repo/name
        if source.is_file():
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    restored=[]
    for group,prefix in [('native_sources','robolab'),('hybrid_sources','robolab/experiments/amp_hybrid'),
                         ('local_assets','robolab/experiments/amp_hybrid'),
                         ('shared_rough_runtime_helpers','robolab/experiments/rough_hybrid')]:
        for name,expected in plan['experiment'][group].items():
            target=(dest/prefix/name).resolve()
            if not target.is_relative_to(dest):raise ValueError('Source escapes snapshot')
            raw=target.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected:
                raw=raw.replace(b'\r\n',b'\n')
                if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Missing exact source '+name)
                target.write_bytes(raw);restored.append(name)
    # Copy the supplied pinned physical snapshot and original template/meshes.
    for prefix in ('vendor/physical_1450a51','robolab/data/robots/roboparty/rpo'):
        shutil.copytree(root/'assets'/prefix,dest/prefix)
    # Restore any archive filename decoding differences by immutable content hash.
    physical=dest/'vendor/physical_1450a51'
    indexed={sha(f):f for f in physical.rglob('*') if f.is_file()}
    for name,expected in next(iter(bank['motor_pairs'].values()))['verified_file_sha256'].items():
        target=physical/name
        if not target.is_file():
            if expected not in indexed:raise ValueError('Missing pinned physical file '+name)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(indexed[expected],target)
        if sha(target)!=expected:raise ValueError('Pinned physical bytes differ '+name)
    server='/data/chenjiale/robo_cjl/'
    def relocated(value):
        if isinstance(value,str) and value.startswith(server):return str(dest/value[len(server):])
        if isinstance(value,list):return [relocated(v) for v in value]
        if isinstance(value,dict):return {k:relocated(v) for k,v in value.items()}
        return value
    local=relocated(bank);asset_audits=[]
    for original,design in zip(bank['designs'],local['designs']):
        for kind in ('urdf','report'):
            source=root/'assets'/original[kind][len(server):]
            if sha(source)!=original[kind+'_sha256']:raise ValueError('Original supplied asset hash differs')
        source=root/'assets'/original['urdf'][len(server):]
        target=Path(design['urdf']);target.parent.mkdir(parents=True,exist_ok=True)
        text=source.read_text(encoding='utf8')
        def mesh(match):
            name=match.group(1)
            if name.startswith(server):path=dest/name[len(server):]
            elif name.startswith('../meshes/'):
                path=dest/'robolab/data/robots/roboparty/rpo/meshes'/Path(name).name
            else:path=(source.parent/name.removeprefix('file://')).resolve()
            if not path.is_file():raise ValueError('Missing exact mesh '+name)
            return 'filename="'+path.as_posix()+'"'
        text=re.sub(r'filename="([^"]+)"',mesh,text)
        target.write_text(text,encoding='utf8',newline='')
        shutil.copyfile(root/'assets'/original['report'][len(server):],design['report'])
        design['urdf_sha256']=sha(target);design['report_sha256']=sha(design['report'])
        asset_audits.append(dict(id=design['id'],original_sha256=original['urdf_sha256'],
            local_sha256=design['urdf_sha256'],change='Mesh filename relocation only'))
    # The binding shares every nominal physical field with a relocated Rough bank.
    rough=json.loads((w/'rough_training_source_recovery_20261029/bank.json').read_text(encoding='utf8'))
    shared=('configuration','snapshot_root','num_envs','knees','ankles','context_width','ranges_m',
        'designs','motor_pairs','original_template','original_template_sha256','reference_contract_source',
        'reference_contract_sha256','canonical_semantic_contract')
    for key in shared:rough[key]=deepcopy(local[key])
    write_new(root/'rough_local3.json',rough)
    local['rough_morphology_binding'].update(rough_bank=str(root/'rough_local3.json'),
        rough_bank_sha256=sha(root/'rough_local3.json'))
    write_new(root/'bank_local3.json',local)
    library=json.loads((root/'bundle/library.json').read_text(encoding='utf8'));library['bank_sha256']=sha(root/'bank_local3.json')
    for name,record in library['source_clips'].items():record['path']=str(dest/'robolab/data/motions/rpo_lab'/(name+'.pkl'))
    for identity,row in library['designs'].items():
        source_directory=root/'references'/row['directory'][len(server):]
        row['directory']=str(source_directory)
        design=next(d for d in local['designs'] if d['id']==identity)
        row['urdf_sha256']=design['urdf_sha256']
        for clip,expected in row['clips'].items():
            if sha(source_directory/(clip+'.pkl'))!=expected:raise ValueError('Reference clip differs')
    write_new(root/'library_local3.json',library)
    write_new(root/'import_audit3.json',dict(original_bank_sha256=sha(root/'bundle/bank.json'),
        local_bank_sha256=sha(root/'bank_local3.json'),checkpoint_sha256=sha(root/'bundle/model_4999.pt'),
        source_files_exact=True,server_LF_restored=restored,assets=asset_audits,
        reference_payload_bytes_unchanged=True,teacher_weights_unchanged=True,
        physical_changes='None; only filenames and dependency locations relocated',
        source_root=str(dest/'robolab')))
    print('AMP_IMPORT_AUDITED',len(asset_audits),'designs',len(library['designs']),'reference bindings',flush=True)

if __name__=='__main__':main()
