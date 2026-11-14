"""Independent counts from exported design IDs; never trust generator scores."""
from collections import Counter
import numpy as np


def audit(groups, bank, windows=(1,5,10)):
    rows={d['id']:d for d in bank['designs'] if d['split']=='train'}
    flat=[i for group in groups for i in group]
    if len(groups)!=50 or any(len(g)!=60 for g in groups):
        raise ValueError('Expected 50 groups of 60')
    if len(flat)!=len(set(flat)) or set(flat)!=set(rows):
        raise ValueError('Duplicate, missing or foreign structure')
    knees=sorted({d['knee_motor'] for d in rows.values()})
    ankles=sorted({d['ankle_motor'] for d in rows.values()})
    thighs=sorted({d['thigh_m'] for d in rows.values()})
    calves=sorted({d['calf_m'] for d in rows.values()})
    def region(v, bounds):return min(2,max(0,int(3*(v-bounds[0])/(bounds[1]-bounds[0]))))
    def cells(ids, motor_field, motor, dim):
        result=np.zeros((3,3),dtype=int) if dim=='joint' else np.zeros(3,dtype=int)
        for i in ids:
            d=rows[i]
            if d[motor_field]!=motor:continue
            t=region(d['thigh_m'],bank['ranges_m']['thigh'])
            c=region(d['calf_m'],bank['ranges_m']['calf'])
            if dim=='joint':result[t,c]+=1
            else:result[t if dim=='thigh' else c]+=1
        return result.flatten()
    groups_stats=[]
    for group in groups:
        pair=Counter((rows[i]['knee_motor'],rows[i]['ankle_motor']) for i in group)
        t=Counter(rows[i]['thigh_m'] for i in group);c=Counter(rows[i]['calf_m'] for i in group)
        groups_stats.append(dict(pair_counts=[pair[k,a] for k in knees for a in ankles],
            thigh_counts=[t[x] for x in thighs],calf_counts=[c[x] for x in calves]))
    report=dict(valid=True,design_count=len(flat),group_count=len(groups),group_size=60,
        pair_count_range=[min(min(s['pair_counts']) for s in groups_stats),max(max(s['pair_counts']) for s in groups_stats)],
        thigh_count_range=[min(min(s['thigh_counts']) for s in groups_stats),max(max(s['thigh_counts']) for s in groups_stats)],
        calf_count_range=[min(min(s['calf_counts']) for s in groups_stats),max(max(s['calf_counts']) for s in groups_stats)],
        per_group=groups_stats,windows={})
    for width in windows:
        dimensions={dim:[] for dim in ('thigh','calf','joint')};worst={dim:None for dim in dimensions}
        # Include sweep-boundary windows; a repeated frozen schedule is cyclic.
        for start in range(len(groups)):
            ids=[i for j in range(width) for i in groups[(start+j)%len(groups)]]
            for field,motors in [('knee_motor',knees),('ankle_motor',ankles)]:
                for motor in motors:
                    for dim,values in dimensions.items():
                        counts=cells(ids,field,motor,dim)
                        target=cells(flat,field,motor,dim).astype(float);target/=target.sum()
                        actual=counts/counts.sum()
                        tv=float(np.abs(actual-target).sum()/2)
                        values.append(tv)
                        if worst[dim] is None or tv>worst[dim]['total_variation']:
                            worst[dim]=dict(start_group=start,motor_field=field,motor=motor,
                                counts=counts.tolist(),target_proportions=target.tolist(),total_variation=tv)
        report['windows'][str(width)]={dim:dict(mean_total_variation=float(np.mean(values)),
            max_total_variation=float(max(values)),worst=worst[dim]) for dim,values in dimensions.items()}
    return report
