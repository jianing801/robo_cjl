"""Bounded isolated native conversion followed by the intersection-only pipeline."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from pipeline import child,HERE
from contract import load_plan,write_new

def main():
    workspace=HERE.parents[3]
    original=workspace/'rough_backpack_shared_20261029/intersection/plan_all.json'
    plan=load_plan(original)
    root=workspace/'rbc2';root.mkdir(exist_ok=False)
    environment=dict(os.environ)
    caches=[str(workspace/'rbs1'),str(workspace/'rbc1'),str(root)]
    failures=0
    for index in range(60):
        out=root/f'a{index:02d}'
        command=[sys.executable,'-u',str(HERE/'prepare_assets.py'),'--plan',str(original),
            '--output',str(out),'--headless','--device','cuda:0','--max-conversions','4']
        for cache in caches:command+=['--cache-root',cache]
        print('PREPARE_ASSETS',index,flush=True)
        try:
            child(command,root/f'log{index:02d}',environment,120,root)
        except (TimeoutError,RuntimeError) as error:
            write_new(root/f'recovery_{index:02d}.json',dict(error=str(error),action='Preserve assets; retry in fresh isolated process'))
            failures+=1
            if failures>=3:raise
            continue
        failures=0
        summary=json.loads((out/'summary.json').read_text())
        print('ASSETS_REMAINING',summary['remaining'],flush=True)
        if summary['remaining']==0:break
    else:raise RuntimeError('Bounded asset preparation did not finish')
    plan['teachers']['stairs']['asset_cache']=caches+plan['teachers']['stairs']['asset_cache']
    cached=root/'plan.json';write_new(cached,plan)
    command=[sys.executable,'-u',str(HERE/'pipeline.py'),'--plan',str(cached),
        '--output',str(workspace/'rbs2'),'--batch-designs','200',
        '--collect-steps','500','--stride','2','--epochs','20','--steps-per-epoch','100',
        '--dagger-rounds','1','--timeout-seconds','1200','--device','cuda:0']
    child(command,root/'pipeline',environment,18000,root)
    write_new(root/'completed.json',dict(pipeline=str(workspace/'rbs2'),finished=datetime.now().isoformat()))

if __name__=='__main__':main()
