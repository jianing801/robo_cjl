import csv
from ppo_correction_study import curve_assessment, choose_best


def make_curve(decline=False):
    return [dict(update=i,mean_reward=30,
                 stairs_mean_reward=34+min(i/200,1)*3,
                 backpack_mean_reward=30,
                 run_mean_reward=29-(7*(i-150)/850 if decline and i>150 else 0))
            for i in range(1,1001)]


def test_all_task_gate_rejects_cancellation_even_when_total_is_flat():
    prior=make_curve(False)
    assert curve_assessment(prior,prior)['passed']
    result=curve_assessment(make_curve(True),prior)
    assert not result['passed'] and not result['tasks']['run']['passed']
    assert result['tasks']['stairs']['passed'] and result['tasks']['backpack']['passed']


def test_selection_prefers_balanced_measured_performance():
    tasks=('stairs','backpack','run'); keys=('cot','forward_rmse_mps','roll_pitch_rms_rad_s','knee_ankle_max_rated_ratio')
    norm={'task_weights':dict(stairs=.4,backpack=.3,run=.3),'metrics':{t:{k:{'baseline':1.,'weight':.25} for k in keys} for t in tasks}}
    def candidate(name,values):
        return {'profile':name,'pairs':[{'feasible':True,'tasks': {t:{'status':'success','observed':{'absolute_mechanical_cot':v,**{k:v for k in keys[1:]}}} for t,v in zip(tasks,values)}}]}
    winner=choose_best([candidate('low_average',[.5,.5,1.2]),candidate('balanced',[.9,.9,.9])],norm)
    assert winner['profile']=='balanced'
