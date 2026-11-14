"""Reuse exact Rough base morphologies; add only the native task's backpack."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from hybrid_bank import load_configurable_bank
from hybrid_support import source_identity
from robolab.rough_geometry import file_sha

def adopt_rough_bank(rough_path,grid,output):
    rough_path=Path(rough_path).resolve();rough=load_configurable_bank(rough_path)
    if rough.get('task_key','stairs')!='stairs' or rough['configuration']!=grid:
        raise ValueError('Backpack requires the exact matching Rough grid')
    out=Path(output).resolve()
    # All checks above happen before creating the new output. Never alter Rough.
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('_bank_payload',root/'robolab/tasks/direct/backpack_walk/backpack_urdf.py')
    payload=importlib.util.module_from_spec(spec);spec.loader.exec_module(payload)
    bank=deepcopy(rough);bank['task_key']='backpack'
    bank['training']['authorized_tasks']=['backpack']
    bank['training']['source_hashes']=source_identity('backpack')
    bank['base_morphology_binding']=dict(bank=str(rough_path),sha256=file_sha(rough_path),
        task_attachment='backpack_link; initial mass .3 kg overwritten by native payload curriculum')
    for design in bank['designs']:
        source=Path(design['urdf']);target=out/design['id']/'robot_with_backpack.urdf'
        target.parent.mkdir()
        payload.write_backpack_urdf(source,target,.3)
        design.update(base_urdf=str(source),base_urdf_sha256=file_sha(source),
            urdf=str(target),urdf_sha256=file_sha(target),total_mass_kg=design['total_mass_kg']+.3)
    return bank
