import json
from pathlib import Path
import numpy as np
import pytest
from robolab.physical_snapshot import DEFAULT_ROOT,generator,physical_model
from robolab.gp_extrapolation import get_generator,authorization_audit,validate_authorization

def test_explicit_extrapolation_does_not_modify_pinned_generator():
    original=generator(DEFAULT_ROOT);extended=get_generator(DEFAULT_ROOT,True)
    key='DM10010L_thigh'
    with pytest.raises(ValueError):original.predict_properties(key,.20)
    assert extended.predict_properties(key,.20)['extrapolated']
    with pytest.raises(ValueError):original.predict_properties(key,.20)
    assert original.MODELS[key]['prediction_range_m']==[.25,.4]
    for length in (.26,.30,.40):
        a=original.predict_properties(key,length);b=extended.predict_properties(key,length)
        assert a['mass_kg']==b['mass_kg']
        np.testing.assert_array_equal(a['inertia_kg_m2'],b['inertia_kg_m2'])
    with pytest.raises(ValueError):extended.predict_properties(key,.199)

def test_all_eleven_motors_all_ten_lengths_have_valid_physics():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/rough.json').read_text())['grid']
    gen=get_generator(DEFAULT_ROOT,True)
    assert cfg['thigh_range_m']==[.2,.4] and cfg['calf_range_m']==[.25,.45] and cfg['gp_extrapolation']
    for role,segment in (('knee','thigh'),('ankle','calf')):
        for motor in cfg[role+'_pool']:
            physical=physical_model(knee_motor=motor if role=='knee' else cfg['knee_pool'][0],
                ankle_motor=motor if role=='ankle' else cfg['ankle_pool'][0])
            for length in np.linspace(*cfg[segment+'_range_m'],10):
                result=gen.predict_properties(physical['selected'][role]['urdf_model_id'],float(length))
                moments=np.linalg.eigvalsh(result['inertia_kg_m2'])
                assert result['mass_kg']>0 and moments[0]>0 and moments[0]+moments[1]>=moments[2]-1e-12

def test_authorization_and_source_are_part_of_new_bank_identity():
    cfg=dict(gp_extrapolation=True,thigh_range_m=[.2,.4],calf_range_m=[.25,.45])
    training={'gp_extrapolation':authorization_audit()}
    validate_authorization(cfg,training)
    with pytest.raises(ValueError):validate_authorization(cfg,{})
    bad=dict(cfg,thigh_range_m=[.19,.4])
    with pytest.raises(ValueError):validate_authorization(bad,training)
    training['gp_extrapolation']['adapter_sha256']='changed'
    with pytest.raises(ValueError):validate_authorization(cfg,training)
