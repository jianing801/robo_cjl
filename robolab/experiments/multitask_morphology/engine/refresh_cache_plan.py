"""Explicitly refresh conversion-only provenance before any PPO attempt exists."""
import argparse,json,hashlib,os
from datetime import datetime
from pathlib import Path
from hybrid_contract import experiment_identity
from hybrid_native_cache import training_attempt_started
from hybrid_support import write_new

KNOWN_CONVERTERS = {'prepare_native_cache.py': ['0914a11665f442dc6822cac05b944fc9750d5dbd9231fdca4607f53b41deec8e', '1a7814e36b72064a4130c693885d4fc95ea450fac80cf6d5b392e63132c1da5d', '49bb0275ce660e9bc3d0d61c445c90f6d98d592819019b890d37b290189f2f31', 'b642f6695a8a92ebb7cfcfce418bfefcb842cd51f1cf8ba88928bb305747be9f'], 'native_cache_coordinator.py': ['2a9ba775c12a474e33b961a009cece02c22c6c62ed8907725b5a9b08a441eefe', '48682f956f90326b9bee272e6c14767d96735840ab7ce6895dd2f12ce0724029', '86a3acdc5be4b9e6fcb4ba4d2b68140af2401076619a0d62024410a642f9206a', '9da8fa47dbfdc880b427010e4b585d2698cd087613ff0f2abb4fad3baf04e057', 'b6554ba09d3a4b03c307b2fcce120e3036f816c424b5bbf7aa04b2013af4d214']}
KNOWN_REFRESH_HELPERS = ['42a1eeffa9c451010d8d34086e5eabbc061da2be158645dea794fab11c727aee', '5de7b34f719dd0f33ad35c135d1c861158593294cba3339eebe0bed3823fbf49', '4d75c167794e1011b3bd0ba9941cfe83d98cccdf3d220b36f46a76cb543671da', '9d5f2088a3ccfb6e4d47de8b1bbcce9268c02a433724dbb2ee06393c1790854a', '7001575b91e39ed067be8d2624804e01dc23350d713331017c83f12184ea86f3', 'cc5723dc26c1d150e1508885e3c755decb9fa9eafdd9cf47341514150ac09f1e']

KNOWN_LOGGING_IO_SOURCES = {'hybrid_assets.py': 'eea0ecd8265f914e945cef8e324ad311f447377f1ca092671f91fed07c9240ea', 'hybrid_native_cache.py': '6a60466b93e536efdb0fc0d8e757353a4fdd37641ef2d1ff0574b3a01c8cf95f'}

KNOWN_IO_SOURCES = {'hybrid_assets.py': 'c9eab750c7a712966572623fd1aa22c29b579f721741f23efa81d979794f0898', 'hybrid_native_cache.py': '03153f0b0e568439f295761903c68e6a1c06201cea5dd97893386d44ad49bad0', 'train.py': '24f2c69b1f41f726cc4fe4726278dfa135d3e94f1f39329f1618bc4c8230760c'}


def checkpoint_evidence(root):
    """转换输出不是PPO；跳过已知缓存树，其他位置仍检查未知检查点。"""
    root=Path(root)
    checked=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        here=Path(directory)
        if here==root:
            dirs[:]=[d for d in dirs if not d.startswith('native_cache_')]
        if here==root/'attempts':
            remaining=[]
            for name in dirs:
                if name.startswith('cache_'):
                    launch=here/name/'train/launch.json'
                    try:
                        command=json.loads(launch.read_text(encoding='utf8'))['command']
                        scripts=[Path(a).name for a in command if isinstance(a,str) and a.endswith('.py')]
                    except (OSError,ValueError,KeyError):scripts=[]
                    if scripts==['prepare_native_cache.py']:
                        continue
                remaining.append(name)
            dirs[:]=remaining
        checked+=1
        if any(name.endswith('.pt') for name in files):return True
    print('CACHE_PLAN_NO_CHECKPOINTS',checked,'directories',flush=True)
    return False


def refreshed_plan(root,plan):
    print('CACHE_PHASE CHECKING_PPO_EVIDENCE',flush=True)
    if training_attempt_started(root) or any((root/'commits').glob('*.json')) or checkpoint_evidence(root):
        raise RuntimeError('PPO evidence exists: use the original frozen code; plan cannot be refreshed')
    print('CACHE_PLAN_CHECK_SOURCE_IDENTITY',flush=True)
    old=plan['experiment'];new=experiment_identity(plan['bank'],plan['configuration'])
    if old==new:return plan
    a=dict(old);b=dict(new);previous=dict(a.pop('hybrid_sources'));current=dict(b.pop('hybrid_sources'))
    if a!=b:raise ValueError('Non-converter experiment identity changed')
    for name,known in KNOWN_CONVERTERS.items():
        before=previous.pop(name,None);after=current.pop(name,None)
        # 未改变的文件由新旧身份直接证明一致；只有迁移改变时需要旧版白名单。
        if before is None or (before!=after and before not in known):
            raise ValueError('Unknown previous converter source: '+name)
    previous_helper=previous.pop('refresh_cache_plan.py',None)
    if previous_helper is not None and previous_helper not in KNOWN_REFRESH_HELPERS:
        raise ValueError('Unknown previous refresh helper identity')
    current.pop('refresh_cache_plan.py',None)
    for name,digest in KNOWN_IO_SOURCES.items():
        if previous.get(name)!=current.get(name):
            if previous.get(name) is None or previous.get(name) not in (digest,KNOWN_LOGGING_IO_SOURCES.get(name)):
                raise ValueError('Unknown previous cache IO source: '+name)
            previous.pop(name,None);current.pop(name,None)
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
