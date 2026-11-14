"""Offline fixed-grid partition experiment: exact cyclic construction + SA order.

This is a bespoke constrained partition, not an implementation of SLHD/OCD.
No simulator, robot assets, teacher weights or training environments are loaded.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'engine'),str(HERE)]
from hybrid_bank import select_grid
from hybrid_contract import cycle_groups
from audit import audit


def construct(bank, seed):
    """Each group: every pair twice and every leg-axis level six times.

    Pair p belongs to one of six blocks of five pairs. Within each block,
    a[p] and z[p] are permutations of 0..4. For group (r,d):
    t=(r+a[p]) mod 5; c=(t+d+z[p]-a[p]) mod 10.
    Use both (t,c) and (t+5,(c+5) mod 10).
    Over r=0..4,d=0..9, each pair sees all 100 leg combinations once.
    """
    rng=np.random.default_rng(seed)
    rows=bank['designs'];ts=sorted({d['thigh_m'] for d in rows});cs=sorted({d['calf_m'] for d in rows})
    pairs=sorted({(d['knee_motor'],d['ankle_motor']) for d in rows})
    if len(ts)!=10 or len(cs)!=10 or len(pairs)!=30 or len(rows)!=3000:
        raise ValueError('This construction is specific to the complete 10x10x5x6 grid')
    pairs=[pairs[i] for i in rng.permutation(30)]
    lookup={(ts.index(x['thigh_m']),cs.index(x['calf_m']),x['knee_motor'],x['ankle_motor']):x['id'] for x in rows}
    if len(lookup)!=3000:raise ValueError('Not an exact full grid')
    a=np.concatenate([rng.permutation(5) for _ in range(6)])
    z=np.concatenate([rng.permutation(5) for _ in range(6)])
    groups=[]
    for r in range(5):
        for d in range(10):
            group=[]
            for p,(knee,ankle) in enumerate(pairs):
                t=int((r+a[p])%5);c=int((t+d+z[p]-a[p])%10)
                group.extend([lookup[t,c,knee,ankle],lookup[t+5,(c+5)%10,knee,ankle]])
            groups.append(group)
    return [groups[i] for i in rng.permutation(50)]


def features(groups, bank):
    rows={d['id']:d for d in bank['designs']}
    motors=[(field,m) for field in ('knee_motor','ankle_motor') for m in sorted({d[field] for d in rows.values()})]
    out=np.zeros((50,len(motors)*9))
    for g,ids in enumerate(groups):
        for j,(field,motor) in enumerate(motors):
            selected=[rows[i] for i in ids if rows[i][field]==motor]
            for row in selected:
                bins=[]
                for field2,segment in [('thigh_m','thigh'),('calf_m','calf')]:
                    lo,hi=bank['ranges_m'][segment]
                    bins.append(min(2,max(0,int(3*(row[field2]-lo)/(hi-lo)))))
                out[g,j*9+3*bins[0]+bins[1]]+=1/len(selected)
    return out


def objective(x):
    score=0.
    for width in (1,5,10):
        means=sum(np.roll(x,-i,axis=0) for i in range(width))/width
        error=(means-x.mean(axis=0)).reshape(50,11,9)
        tv=np.abs(error).sum(axis=-1)/2
        # Average + worst-window term. Width weights fixed before experiments.
        score+=float((tv**2).mean()+.25*tv.max()**2)
    return score


def optimize(groups, bank, seed, iterations):
    rng=np.random.default_rng(seed);x=features(groups,bank)
    order=np.arange(50);current=objective(x);best=current;best_order=order.copy()
    accepted=0;trace=[]
    for i in range(iterations):
        a,b=rng.choice(50,2,replace=False)
        x[[a,b]]=x[[b,a]];trial=objective(x)
        temperature=.003*(.000001/.003)**(i/max(iterations-1,1))
        if trial<=current or rng.random()<math.exp(min(0.,(current-trial)/temperature)):
            order[[a,b]]=order[[b,a]];current=trial;accepted+=1
            if trial<best:best=trial;best_order=order.copy()
        else:x[[a,b]]=x[[b,a]]
        if i%1000==0:trace.append(dict(iteration=i,best=best,temperature=temperature))
    return [groups[int(i)] for i in best_order],dict(initial=objective(features(groups,bank)),
        best=best,iterations=iterations,accepted=accepted,trace=trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(HERE.parent/'configs/rough.json'))
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--candidates',type=int,default=64)
    p.add_argument('--iterations',type=int,default=12000)
    args=p.parse_args();out=Path(args.output).resolve()
    if out.exists():raise ValueError('Use a NEW output; no overwrite')
    if args.candidates<1 or args.iterations<1:raise ValueError('Positive bounded search required')
    config=json.loads(Path(args.config).read_text(encoding='utf8'));grid=config['grid']
    if config['rotation']['active_designs']!=60:raise ValueError('Expected 60 designs/group')
    selected=select_grid(grid,grid['knee_pool'],grid['ankle_pool'])
    bank=dict(designs=selected['designs'],ranges_m={'thigh':grid['thigh_range_m'],'calf':grid['calf_range_m']})
    out.mkdir(parents=True,exist_ok=False)
    (out/'grid.json').write_text(json.dumps(bank,indent=2),encoding='utf8')
    results=[]
    for seed in [int(v) for v in args.seeds.split(',')]:
        started=time.monotonic();cfg=json.loads(json.dumps(config));cfg['rotation']['seed']=seed
        # Explicitly retain the old stratified baseline after formal adoption.
        cfg.pop('group_schedule',None)
        baseline=cycle_groups(bank,cfg,0)
        candidates=[construct(bank,seed*100000+i) for i in range(args.candidates)]
        initial=min(candidates,key=lambda groups:objective(features(groups,bank)))
        final,search=optimize(initial,bank,seed,args.iterations)
        reports={name:audit(groups,bank) for name,groups in [('baseline',baseline),('constructed',initial),('optimized',final)]}
        for key in ('pair_count_range','thigh_count_range','calf_count_range'):
            assert reports['optimized'][key]==([2,2] if key=='pair_count_range' else [6,6])
        bundle=dict(seed=seed,candidates=args.candidates,search=search,metrics=reports,
            seconds=time.monotonic()-started)
        (out/f'audit_seed_{seed}.json').write_text(json.dumps(bundle,indent=2),encoding='utf8')
        schedule=dict(kind='offline_balanced_schedule_v1',seed=seed,groups=final,
            grid_sha256=hashlib.sha256((out/'grid.json').read_bytes()).hexdigest(),
            formal_training_modified=False,algorithm='exact cyclic construction + best-of-candidates + simulated annealing group order')
        (out/f'schedule_seed_{seed}.json').write_text(json.dumps(schedule,indent=2),encoding='utf8')
        lookup={d['id']:d for d in bank['designs']}
        with (out/f'schedule_seed_{seed}.csv').open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=['group','slot','id','thigh_m','calf_m','knee_motor','ankle_motor'])
            writer.writeheader()
            for g,ids in enumerate(final):
                for slot,i in enumerate(ids):writer.writerow(dict(group=g,slot=slot,**{k:lookup[i][k] for k in writer.fieldnames if k not in ('group','slot')}))
        results.append(dict(seed=seed,seconds=bundle['seconds'],search=search,
            metrics={name:{k:v for k,v in value.items() if k!='per_group'} for name,value in reports.items()}))
        print('SCHEDULE_COMPLETE',seed,'seconds',round(bundle['seconds'],1),'objective',search['initial'],'->',search['best'],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(arguments=vars(args),results=results,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'audit.py']}),indent=2),encoding='utf8')


if __name__=='__main__':main()
