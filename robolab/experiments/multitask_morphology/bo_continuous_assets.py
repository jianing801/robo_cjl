"""Fresh native singleton assets for an off-grid candidate; no actor retraining."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from uuid import uuid4


BANKS = {'stairs': 'rough', 'backpack': 'backpack', 'run': 'amp'}
BOUNDS = {'thigh': [.2, .4], 'calf': [.25, .45]}
TOL = 1e-12


def singleton_config(key, schema):
    if (len(key) != 4 or any(type(v) not in (int, float) or not math.isfinite(v)
                             or round(v, 10) != v for v in key[:2])
            or key[2] not in schema['knees'] or key[3] not in schema['ankles']
            or list(key) in schema['shared_structure_keys']
            or schema['context_schema'] != 'leg_motor_inertia_v1'):
        raise ValueError('Invalid off-grid design for frozen actor schema')
    grid = dict(num_envs=4096, length_count=1, knee_motor_count=1, ankle_motor_count=1,
                motor_seed=42, knee_pool=[key[2]], ankle_pool=[key[3]], gp_extrapolation=True)
    for segment, value in zip(('thigh', 'calf'), key[:2], strict=True):
        lo, hi = schema['ranges_m'][segment]
        amin, amax = BOUNDS[segment]
        if not amin <= lo <= value <= hi <= amax or lo == hi:
            raise ValueError('Continuous design outside authorized GP/actor length bounds')
        # Native singleton generator uses midpoint of a nonempty interval. At an
        # exact boundary choose adjacent floats *inside* authorization. The actual
        # geometry is independently checked against the canonical 10-digit key.
        half = min(1e-5, value-lo, hi-value)
        if half > 0:
            a, b = value-half, value+half
        elif value == lo:
            a, b = lo, math.nextafter(lo, hi)
        else:
            a, b = math.nextafter(hi, lo), hi
        midpoint = (a+b)/2
        if not lo <= a < b <= hi or round(midpoint, 10) != value or abs(midpoint-value) > TOL:
            raise ValueError(f'Cannot represent exact canonical singleton {segment} length')
        grid[segment+'_range_m'] = [a,b]
    return dict(context_schema=schema['context_schema'], grid=grid)


def stage_artifact(bo, stage, generate, verify, *, max_attempts=3):
    """Preserve unsuccessful outputs; verify any recovery and freeze its receipt."""
    stage.mkdir(parents=True, exist_ok=True)
    receipt = stage/'completed.json'
    if receipt.is_file():
        row = json.loads(receipt.read_text(encoding='utf8'))
        folder = stage/row['attempt']
        if folder.parent != stage or verify(folder) != row['evidence']:
            raise ValueError(f'Previously validated native artifact changed: {stage}')
        return folder
    failures = []
    for index in range(1,max_attempts+1):
        folder = stage/f'attempt_{index:02d}'
        started = stage/f'{folder.name}.json'
        try:
            if not started.exists():
                if folder.exists():
                    raise ValueError(f'Untracked native artifact path: {folder}')
                bo.freeze(started, dict(attempt=folder.name))
                generate(folder)
            evidence = verify(folder)
        except (OSError, RuntimeError, TimeoutError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
            failures.append(f'{folder.name}: {exc}')
            print('BO_NATIVE_PREPARATION_RETRY', stage.name, failures[-1], flush=True)
            continue
        bo.freeze(receipt, dict(attempt=folder.name,evidence=evidence))
        return folder
    raise RuntimeError(f'Native preparation exhausted {max_attempts} retained attempts: {stage}: {failures}')


def _base_elements(path):
    root = ET.parse(path).getroot()
    return {(element.tag,element.get('name')): ET.tostring(element,encoding='unicode').strip()
            for element in root if element.get('name') not in ('backpack_link','backpack_mount_joint')}


def _actual_lengths(path, key):
    root = ET.parse(path).getroot()
    for target, suffix in zip(key[:2], ('knee_joint','ankle_pitch_joint'),strict=True):
        for side in ('left','right'):
            origin = root.find(f"joint[@name='{side}_{suffix}']/origin")
            if origin is None:
                raise ValueError('Missing exact native leg joint')
            length = -float(origin.get('xyz').split()[2])
            if not math.isfinite(length) or abs(length-target)>TOL:
                raise ValueError('Generated URDF leg length differs from candidate')


def _key(row):
    return [round(float(row['thigh_m']),10),round(float(row['calf_m']),10),
            row['knee_motor'],row['ankle_motor']]


def _verify_bank(bo, path, key, task, config, donor, schema):
    bank = bo.read(path)
    (row,) = bank['designs']
    if (bank['task_key'] != task or _key(row) != list(key)
            or bank['schema_version']!=3 or bank['context_schema'] != schema['context_schema']
            or bank['context_fields']!=schema['context_fields'] or bank['context_width']!=schema['context_width']
            or bank['configuration']!=config['grid']
            or bank['canonical_semantic_contract']!=donor['canonical_semantic_contract']
            or bank['original_template_sha256']!=donor['original_template_sha256']
            or bank['reference_contract_sha256']!=donor['reference_contract_sha256']
            or bank['training']['source_hashes']!=donor['training']['source_hashes']
            or bank['training']['gp_extrapolation']!=donor['training']['gp_extrapolation']):
        raise ValueError(f'Generated native bank/model/source identity differs: {task}')
    pair = row['pair_id']
    # Physical provenance is pinned: compare all source hashes, selected motor
    # curves and model properties, permitting only relocated snapshot root paths.
    physical = bank['motor_pairs'][pair]
    if ({k:v for k,v in physical.items() if k!='snapshot_root'}
            != {k:v for k,v in donor['motor_pairs'][pair].items() if k!='snapshot_root'}):
        raise ValueError(f'Generated native motor physical identity differs: {task}')
    for field in ('urdf','report'):
        if bo.sha(row[field]) != row[field+'_sha256']:
            raise ValueError(f'Generated native {field} changed: {task}')
    _actual_lengths(row['urdf'],key)
    return bank, row


def _audit_common(bo, paths, donor, config, schema, key):
    verified = {task:_verify_bank(bo, path, key, task, config, donor[task],schema)
                for task,path in paths.items()}
    rough, r = verified['stairs']; amp, a = verified['run']; pack, p = verified['backpack']
    base_sha = bo.sha(paths['stairs'])
    for field in ('configuration','knees','ankles','context_width','ranges_m','motor_pairs',
                  'canonical_semantic_contract','schema_version','context_schema','context_fields','context_normalization'):
        if not rough[field]==amp[field]==pack[field]:
            raise ValueError('New singleton common morphology differs: '+field)
    for field in ('id','thigh_m','calf_m','knee_motor','ankle_motor','context','pair_id',
                  'root_height_compensation_m','gp_extrapolation'):
        if not r[field]==a[field]==p[field]:
            raise ValueError('Three task singleton identity differs: '+field)
    if (pack['base_morphology_binding']['sha256']!=base_sha
            or amp['rough_morphology_binding']['rough_bank_sha256']!=base_sha
            or not r['urdf_sha256']==a['urdf_sha256']==p['base_urdf_sha256']
            or bo.sha(p['base_urdf'])!=r['urdf_sha256']
            or _base_elements(p['urdf'])!=_base_elements(r['urdf'])
            or ET.parse(p['urdf']).find("link[@name='backpack_link']") is None):
        raise ValueError('New singleton has nonidentical base geometry/incorrect task attachment')
    return dict(kind='BO_CONTINUOUS_SINGLETON_COMMON_MORPHOLOGY_VERIFIED', key=list(key),
                banks={task:bo.sha(path) for task,path in paths.items()},
                design_ids={task:row['id'] for task, (_,row) in verified.items()},
                actual_urdf_hashes={task:row['urdf_sha256'] for task,(_,row) in verified.items()})


def _mesh_directory(donor):
    template = Path(donor['original_template'])
    sibling = template.parent.parent/'meshes'
    if sibling.is_dir():
        return sibling
    # Migrated bank may store the verified, relocated mesh directory separately.
    path = Path(donor['designs'][0]['urdf'])
    filenames = [m.get('filename') for m in ET.parse(path).findall('.//mesh')]
    roots = {(path.parent/filename).resolve().parent for filename in filenames}
    if len(roots)!=1 or not next(iter(roots)).is_dir():
        raise ValueError('Cannot infer a common validated template mesh directory')
    return next(iter(roots))


def _mesh_hashes(bo, urdf):
    path=Path(urdf)
    result={}
    for element in ET.parse(path).findall('.//mesh'):
        filename=element.get('filename').removeprefix('file://')
        target=(path.parent/filename).resolve()
        checksum=bo.sha(target)
        if target.name in result and result[target.name]!=checksum:
            raise ValueError('Ambiguous template mesh basename')
        result[target.name]=checksum
    if not result:
        raise ValueError('Missing original robot display meshes')
    return result


def _native_worker(bo, a, work, native_repo, command, name, *, gpu=False):
    native_repo = Path(native_repo)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(native_repo/p) for p in ('robolab','rsl_rl')),
               PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    pause_root = Path(getattr(a,'search_root',None) or getattr(a,'output',None) or work).resolve()
    # Keep large temporary reference artifacts and caches on the selected data
    # disk, independent of the shell environment's default system-disk paths.
    for variable, default in (('TMPDIR',pause_root/'native_tmp'),('XDG_CACHE_HOME',pause_root/'native_cache')):
        directory=Path(env.get(variable) or default)
        directory.mkdir(parents=True,exist_ok=True)
        env[variable]=str(directory.resolve())
    device = getattr(a,'reference_device',None) or 'cuda:0'
    if gpu and device.startswith('cuda'):
        env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    # Frozen bo.run_worker provides bounded subprocesses, child cleanup on pause,
    # and complete launch/stdout/stderr/exit receipts. Execute in native checkout.
    launcher = [sys.executable,'-B','-c',
                'import os,sys; os.chdir(sys.argv[1]); os.execv(sys.argv[2],sys.argv[2:])',
                native_repo,*command]
    bo.run_worker(launcher,work/'native_logs'/f'{name}_{uuid4().hex[:10]}',env,
                  max(1,int(getattr(a,'timeout',1800))),pause_root)


def prepare_candidate(bo, a, work, donor_source, donor_plan, key):
    """Construct exactly one new common structure using the frozen native runtimes."""
    work = Path(work).resolve()
    work.mkdir(parents=True,exist_ok=True)
    schema = donor_plan['schema']
    key = list(key)
    config = singleton_config(key,schema)
    specs = donor_plan['teachers']
    if bo.load_plan(donor_source)!=donor_plan or any(bo.sha(specs[t]['bank'])!=specs[t]['bank_sha256'] for t in BANKS):
        raise ValueError('Frozen donor plan or native bank identity changed')
    donor = {task:bo.read(specs[task]['bank']) for task in BANKS}
    repo = {task:Path(specs[task]['code_root']).parent for task in BANKS}
    runtime = {task:Path(specs[task]['runtime_dir']) for task in BANKS}
    template = Path(getattr(a,'template',None) or donor['stairs']['original_template']).resolve()
    mesh = Path(getattr(a,'mesh_dir',None) or _mesh_directory(donor['stairs'])).resolve()
    if bo.sha(template)!=donor['stairs']['original_template_sha256']:
        raise ValueError('Frozen original template identity changed')
    mesh_hashes=_mesh_hashes(bo,donor['stairs']['designs'][0]['urdf'])
    if any(bo.sha(mesh/name)!=checksum for name,checksum in mesh_hashes.items()):
        raise ValueError('Original template display mesh contents changed')
    bo.freeze(work/'native_identity.json', dict(key=key, donor_plan_sha256=bo.sha(donor_source),
              native_repos={task:str(path) for task,path in repo.items()},
              donor_banks={task:specs[task]['bank_sha256'] for task in BANKS},
              native_preparers={task:bo.sha(runtime[task]/'prepare_bank.py') for task in BANKS},
              template=str(template),template_sha256=bo.sha(template),mesh_dir=str(mesh),mesh_sha256=mesh_hashes))
    config_path = work/'singleton_config.json'
    bo.freeze(config_path,config)
    paths = {}
    for task,label in BANKS.items():
        def generate(folder, task=task):
            args = [sys.executable,'-B',runtime[task]/'prepare_bank.py','--generate-assets',
                    '--config',config_path,'--output-dir',folder]
            if task=='stairs':
                args += ['--template',template,'--mesh-dir',mesh]
            else:
                args += ['--rough-bank',paths['stairs']]
                if task=='backpack':args += ['--task-key','backpack']
            _native_worker(bo,a,work,repo[task],args,f'{task}_generate')

        def verify(folder, task=task):
            path=folder/'bank.json'
            bank,row=_verify_bank(bo,path,key,task,config,donor[task],schema)
            if _mesh_hashes(bo,row.get('base_urdf',row['urdf']))!=mesh_hashes:
                raise ValueError('Generated native display meshes differ from donor')
            module='amp_bank' if task=='run' else 'hybrid_bank'
            code=('import sys; sys.path.insert(0,sys.argv[1]); '
                  f'from {module} import load_configurable_bank; load_configurable_bank(sys.argv[2])')
            _native_worker(bo,a,work,repo[task],
                           [sys.executable,'-B','-c',code,runtime[task],path],f'{task}_verify')
            return dict(bank_sha256=bo.sha(path), design_id=row['id'])

        folder=stage_artifact(bo,work/'native_banks'/label,generate,verify)
        paths[task]=folder/'bank.json'
    common=_audit_common(bo,paths,donor,config,schema,key)
    bo.freeze(work/'bank_audit.json',common)
    references_python=getattr(a,'reference_python',None)
    if references_python is None:
        train=Path(sys.executable)
        references_python=(train.parent.parent.parent/'amp_hhtools'/'bin'/'python'
                           if train.parent.name=='bin' else train.parent.parent/'amp_hhtools'/'python.exe')
    references_python=Path(references_python).resolve()
    if not references_python.is_file():
        raise FileNotFoundError(f'Missing dedicated HHTools Python: {references_python}')
    device=getattr(a,'reference_device',None) or 'cuda:0'
    if device not in ('cpu','cuda:0'):
        raise ValueError('Reference device must be cpu or local cuda:0 of selected GPU')
    search_root=Path(getattr(a,'search_root',None) or getattr(a,'output',None) or work.parent)
    warp_cache=Path(getattr(a,'warp_cache',None) or search_root/'warp_reference_cache').resolve()
    bo.freeze(work/'reference_identity.json',dict(python=str(references_python),device=device,
              source_script_sha256=bo.sha(runtime['run']/'prepare_references.py'),
              amp_bank_sha256=common['banks']['run'], warp_cache=str(warp_cache)))

    def generate_reference(folder):
        command=[references_python,'-B',runtime['run']/'prepare_references.py',
                 '--bank',paths['run'],'--output',folder,'--hhtools-root',runtime['run']/'vendor/hhtools',
                 '--device',device,'--warp-cache',warp_cache]
        _native_worker(bo,a,work,repo['run'],command,'amp_reference_generate',gpu=True)

    def verify_reference(folder):
        path=folder/'library.json'
        library=bo.read(path)
        if (library.get('bank_sha256')!=bo.sha(paths['run']) or library.get('validation_only')
                or set(library['designs'])!={common['design_ids']['run']}
                or (device.startswith('cuda') and not library.get('reference_execution',{}).get('cuda_graph_required'))):
            raise ValueError('Fresh AMP reference library not bound to exact candidate')
        # Original native validate_library checks geometry key computed from actual
        # URDF, native eight source clips, all retargeted hashes/shapes and weights.
        code=('import sys; sys.path.insert(0,sys.argv[1]); '
              'from amp_bank import load_configurable_bank; from amp_references import validate_library; '
              'validate_library(load_configurable_bank(sys.argv[2]),sys.argv[2],sys.argv[3])')
        _native_worker(bo,a,work,repo['run'],[references_python,'-B','-c',code,runtime['run'],paths['run'],path],
                       'amp_reference_verify')
        return dict(library_sha256=bo.sha(path),urdf_sha256=common['actual_urdf_hashes']['run'])

    reference=stage_artifact(bo,work/'native_references',generate_reference,verify_reference)/'library.json'
    frozen=deepcopy(donor_plan)
    for task,path in paths.items():
        frozen['teachers'][task].update(bank=str(path),bank_sha256=common['banks'][task],
                                       design_ids=[common['design_ids'][task]])
    frozen['teachers']['run'].update(reference_library=str(reference),reference_library_sha256=bo.sha(reference))
    shared=work/'candidate_bank.json'
    bo.freeze(shared,dict(kind=bo.read(donor_plan['shared_bank'])['kind'],
              structures=[dict(key=key,design_ids=common['design_ids'])]))
    frozen.update(reference_structure_key=key,selected_structure_keys=[key],shared_bank=str(shared),
                  shared_bank_sha256=bo.sha(shared))
    source=work/'candidate_plan.json'
    bo.freeze(source,frozen)
    # Keep the actor's full trained schema, frozen physical normalization and
    # original teacher identities; singleton row context is never used by actor.
    bo.load_plan(source)
    from contract import reference_context
    context={}
    for task,path in paths.items():
        bank=bo.read(path)
        (row,)=bank['designs']
        context[task]=reference_context(bank,row,schema,key)
    if not context['stairs']==context['run']==context['backpack']:
        raise ValueError('Candidate actor physical normalization differs across tasks')
    bo.freeze(work/'actor_context.json',dict(schema=schema['context_schema'],key=key,
              normalization=schema['context_normalization'],values=context['stairs']))
    return source,frozen,common['design_ids']['stairs']
