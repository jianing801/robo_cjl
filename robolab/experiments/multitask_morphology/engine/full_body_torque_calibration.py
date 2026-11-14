"""Offline full-body reward scaling; does not change any live training config."""
REFERENCE_JOINTS=6
REFERENCE_SUM_WEIGHT=-4.0

def equivalent_sum_weight(joint_count):
    if type(joint_count) is not int or joint_count<1:
        raise ValueError('Positive fixed controlled joint count required')
    return REFERENCE_SUM_WEIGHT*REFERENCE_JOINTS/joint_count

def mean_utilization_reward(utilizations,dt=.02,threshold=.8):
    """Calibration only; inputs are valid dynamic utilization measurements."""
    if not utilizations or dt<=0:raise ValueError('Nonempty measurements and positive dt required')
    cost=sum(max(abs(u)-threshold,0.)**2 for u in utilizations)/len(utilizations)
    return REFERENCE_SUM_WEIGHT*REFERENCE_JOINTS*dt*cost
