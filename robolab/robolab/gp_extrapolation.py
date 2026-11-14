"""Explicit, isolated GP extrapolation for the user-authorized wider leg grid.

The pinned source and fitted coefficients remain unchanged. Physical validity
checks (positive mass and physically realizable inertia) remain mandatory.
"""
import ast,hashlib,sys,types
from functools import lru_cache
from pathlib import Path

AUTHORIZED_BOUNDS={'thigh':[.20,.40],'calf':[.25,.45]}

def authorization_audit():
    from copy import deepcopy
    return dict(kind='explicit_gp_extrapolation_v1',authorized_bounds_m=deepcopy(AUTHORIZED_BOUNDS),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        authorization='user explicitly authorized GP extrapolation; 2026-11-13',
        original_fits_unchanged=True,physical_validity_checks_retained=True)

def validate_authorization(config,training):
    if type(config.get('gp_extrapolation',False)) is not bool:raise ValueError('gp_extrapolation must be boolean')
    if config.get('gp_extrapolation'):
        if training.get('gp_extrapolation')!=authorization_audit():raise ValueError('GP extrapolation authorization/source changed')
        for segment in ('thigh','calf'):
            lo,hi=config[segment+'_range_m'];a,b=AUTHORIZED_BOUNDS[segment]
            if not a<=lo<hi<=b:raise ValueError('range exceeds explicitly authorized GP extrapolation bounds')

@lru_cache(maxsize=None)
def authorized_generator(root):
    from robolab.physical_snapshot import generator,catalog
    original=generator(root);pinned_catalog=catalog(root)
    path=Path(original.__file__);tree=ast.parse(path.read_text(encoding='utf8'))
    remapped=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=='robolab.assets.rpo_motor_catalog':
            node.module=pinned_catalog.__name__;remapped+=1
    if remapped!=1:raise ValueError('unexpected pinned GP generator import interface')
    name='_authorized_gp_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    clone=types.ModuleType(name);clone.__file__=str(path);sys.modules[name]=clone
    exec(compile(tree,str(path),'exec'),clone.__dict__)
    for model in clone.MODELS.values():
        original_bounds=list(model['prediction_range_m'])
        model['registered_prediction_range_m']=original_bounds
        allowed=AUTHORIZED_BOUNDS[model['segment']]
        model['prediction_range_m']=[min(original_bounds[0],allowed[0]),max(original_bounds[1],allowed[1])]
    return clone

def get_generator(root,allow_extrapolation=False):
    from robolab.physical_snapshot import generator
    return authorized_generator(str(Path(root).resolve())) if allow_extrapolation else generator(root)

def design_extrapolation(gen,physical,row):
    result={}
    for role,segment in (('knee','thigh'),('ankle','calf')):
        model=gen.MODELS[physical['selected'][role]['urdf_model_id']]
        length=row[segment+'_m'];training=model['training_range_m']
        registered=model.get('registered_prediction_range_m',model['prediction_range_m'])
        result[segment]=dict(length_m=length,training_range_m=training,registered_prediction_range_m=registered,
            outside_training_range=not training[0]<=length<=training[1],
            outside_registered_prediction_range=not registered[0]<=length<=registered[1])
    return result
