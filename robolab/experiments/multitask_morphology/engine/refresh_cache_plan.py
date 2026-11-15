"""Explicitly refresh conversion-only provenance before any PPO attempt exists."""
import argparse,json,hashlib,os
from datetime import datetime
from pathlib import Path
from hybrid_contract import experiment_identity
from hybrid_native_cache import training_attempt_started
from hybrid_support import write_new

KNOWN_CONVERTERS = {'prepare_native_cache.py': ['1a7814e36b72064a4130c693885d4fc95ea450fac80cf6d5b392e63132c1da5d', '49bb0275ce660e9bc3d0d61c445c90f6d98d592819019b890d37b290189f2f31'], 'native_cache_coordinator.py': ['86a3acdc5be4b9e6fcb4ba4d2b68140af2401076619a0d62024410a642f9206a', '9da8fa47dbfdc880b427010e4b585d2698cd087613ff0f2abb4fad3baf04e057']}

def refreshed_plan(root,plan):
    if training_attempt_started(root) or any((root/'commits').glob('*.json')) or any(root.rglob('*.pt')):
        raise RuntimeError('PPO evidence exists: use the original frozen code; plan cannot be refreshed')
    old=plan['experiment'];new=experiment_identity(plan['bank'],plan['configuration'])
    if old==new:return plan
    a=dict(old);b=dict(new);previous=dict(a.pop('hybrid_sources'));current=dict(b.pop('hybrid_sources'))
    if a!=b:raise ValueError('Non-converter experiment identity changed')
    for name,known in KNOWN_CONVERTERS.items():
        if previous.pop(name,None) not in known:raise ValueError('Unknown previous converter source: '+name)
        current.pop(name,None)
    if 'refresh_cache_plan.py' in previous:raise ValueError('Unknown previous refresh helper identity')
    current.pop('refresh_cache_plan.py',None)
    if previous!=current:raise ValueError('Training/runtime source changed; refusing refresh')
    return dict(plan,experiment=new)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root).resolve();path=root/'plan.json'
    if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED present')
    import psutil
    ancestors={os.getpid(),*[p.pid for p in psutil.Process().parents()]}
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid in ancestors:continue
        command=' '.join(proc.info['cmdline'] or []).replace('\\','/').lower()
        if str(root).replace('\\','/').lower() in command:
            raise RuntimeError(f'Process {proc.pid} still uses this output root; stop it before refreshing')
    before=path.read_bytes();plan=json.loads(before);updated=refreshed_plan(root,plan)
    if updated==plan:print('CACHE_PLAN_ALREADY_CURRENT');return
    tag=datetime.now().strftime('%Y%m%d_%H%M%S_%f');audit=root/('cache_plan_refresh_'+tag);audit.mkdir()
    (audit/'plan.before.json').write_bytes(before)
    write_new(audit/'audit.json',dict(kind='pre_ppo_converter_only_refresh',
        old_plan_sha256=hashlib.sha256(before).hexdigest(),old_experiment=plan['experiment'],new_experiment=updated['experiment']))
    tmp=root/('plan.cache_refresh_'+tag+'.tmp');write_new(tmp,updated)
    if path.read_bytes()!=before:raise RuntimeError('Plan changed concurrently; not replaced')
    os.replace(tmp,path)
    print('CACHE_PLAN_REFRESHED_PRE_PPO',root)

if __name__=='__main__':main()
