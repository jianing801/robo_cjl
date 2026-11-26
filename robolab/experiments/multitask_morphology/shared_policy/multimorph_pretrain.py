"""Independent spatial multi-morphology PPO warm-start of the selected round-5 actor."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import os
import random
import signal
import sys
import traceback
from uuid import uuid4

from collection_grid import select_collection_grid
from contract import load_plan, sha, structure_key, write_new
from finetune_settings import CURRENT_PPO_ALGORITHM, CURRENT_PPO_PROFILE, CURRENT_PPO_SETTINGS, PPO_SCHEDULE
from persistence import resume_pause, run_lock
from randomization import protocol_metadata, assert_protocol_unchanged

HERE = Path(__file__).resolve().parent
LAYOUT = {'stairs':4096, 'backpack':1024, 'run':1024}
TASKS = tuple(LAYOUT)
DEFAULT_UPDATES = 1000
STEPS = 24
SOURCE_FILES = (
    'multimorph_pretrain.py','mixed_finetune.py','finetune_common.py','mixed_scene.py','mixed_batch.py',
    'native_ppo.py','native_observations.py','rollout.py','rollout_amp.py','scenario_sampling.py',
    'student.py','training_reward.py','finetune_state.py','finetune_settings.py','amp_fast_reference.py',
    'collection_grid.py','contract.py','persistence.py','randomization.py','student_observations.py',
    '../../../../rsl_rl/rsl_rl/algorithms/ppo_amp.py','../../../../rsl_rl/rsl_rl/algorithms/ppo.py',
    '../../../../rsl_rl/rsl_rl/storage/rollout_storage.py','../../../../rsl_rl/rsl_rl/storage/circular_buffer.py',
    '../../../../rsl_rl/rsl_rl/modules/amp.py',
)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def balanced_order(keys, count, *, seed):
    """Order a Cartesian grid so round-robin remainders balance leg and motor marginals."""
    keys=list(map(tuple,keys))
    if len(keys)!=400 or len(set(keys))!=400:
        raise ValueError('Expected 400 distinct physical keys')
    axes=[sorted(set(axis)) for axis in zip(*keys)]
    if [len(a) for a in axes]!=[4,4,5,5] or set(product(*axes))!=set(keys):
        raise ValueError('Expected the exact 4x4x5x5 shape grid')
    if count not in set(LAYOUT.values()):
        raise ValueError('Unexpected frozen environment count')
    rng=random.Random(seed)
    lengths=list(product(*axes[:2]));rng.shuffle(lengths)
    knees,ankles=axes[2:];rng.shuffle(knees);rng.shuffle(ankles)
    shifts=list(range(5));rng.shuffle(shifts)
    # Every consecutive five pairs covers every knee and every ankle once.
    motor_pairs=[(knees[i],ankles[(i+shift)%5]) for shift in shifts for i in range(5)]
    remaining=count%400
    per_length,tail=divmod(remaining,16)
    if tail:
        raise ValueError('Frozen remainder must divide among the 16 leg geometries')
    extra=[(*length,*motor_pairs[(i*per_length+j)%25])
           for i,length in enumerate(lengths) for j in range(per_length)]
    extra_set=set(extra)
    if len(extra_set)!=remaining:
        raise ValueError('Duplicate supplemental environment key')
    rest=sorted(set(keys)-extra_set)
    # Shuffle only within the two groups; the first remainder keys remain the
    # ones receiving their extra round-robin replica.
    rng.shuffle(extra);rng.shuffle(rest)
    result=extra+rest
    copies=Counter(result[i%400] for i in range(count))
    for indices in ((0,),(1,),(2,),(3,),(0,1),(2,3)):
        marginal=Counter()
        for key,n in copies.items():
            marginal[tuple(key[i] for i in indices)] += n
        if max(marginal.values())-min(marginal.values())>1:
            raise ValueError('Unbalanced spatial assignment')
    return result


def select_training_plan(plan, keys, seed):
    keys=list(map(tuple,keys))
    if set(plan['teachers'])!=set(TASKS) or tuple(plan['schema']['tasks'])!=TASKS:
        raise ValueError('Expected the existing three-task shared schema')
    if not set(keys)<=set(map(tuple,plan['selected_structure_keys'])):
        raise ValueError('Training grid outside exact common physical bank')
    frozen=deepcopy(plan)
    audit={}
    for i,(task,n) in enumerate(LAYOUT.items()):
        spec=frozen['teachers'][task]
        rows=read(spec['bank'])['designs'];allowed=set(spec['design_ids'])
        by_key={}
        for row in rows:
            if row['id'] not in allowed:continue
            key=structure_key(row)
            if key in by_key:raise ValueError(f'Duplicate physical key for {task}')
            by_key[key]=row['id']
        ordered=balanced_order(keys,n,seed=seed+1009*i)
        if any(key not in by_key for key in ordered):
            raise ValueError(f'Missing exact paired structure for {task}')
        ids=[by_key[key] for key in ordered]
        if len(set(ids))!=400:raise ValueError(f'Duplicate structure IDs for {task}')
        spec['design_ids']=ids
        audit[task]=dict(environments=n,design_count=400,design_ids=ids,
                         envs_per_design=dict(Counter(ids[j%400] for j in range(n))))
    frozen['selected_structure_keys']=[list(k) for k in sorted(keys)]
    frozen['scope']='shared_spatial_pretrain_400'
    return frozen,audit


def load_round5(round5):
    round5=Path(round5).resolve()
    receipt=read(round5/'completed.json')
    continuation=read(round5/'manifest.json')
    if (receipt.get('kind')!='shared_dagger_continuation_v1' or receipt.get('round_index')!=5
            or receipt.get('result')!='DISTILLATION_EVALUATED_REQUIRES_CANDIDATE_FINETUNING'
            or receipt['manifest_sha256']!=sha(round5/'manifest.json')
            or continuation['round_index']!=5 or receipt['plan_sha256']!=continuation['plan_sha256']):
        raise ValueError('Round 5 does not have a verified completion')
    base=Path(continuation['base'])
    base_manifest=base/'manifest.json'
    original=base/'stair_evaluation_upgrade/original_manifest.json'
    archive_hash=continuation['parent_provenance'].get('original_manifest_sha256')
    if archive_hash:
        if sha(original)!=archive_hash:raise ValueError('Original DAgger collection identity changed')
        archived=read(original)
    else:
        archived=read(base_manifest)
    present=read(base_manifest)
    scope=archived['protocol']['collection_scope']
    if scope!=present['protocol']['collection_scope'] or scope['design_count']!=400:
        raise ValueError('The original DAgger training grid changed')
    plan_path=Path(archived['arguments']['plan'])
    actor=Path(receipt['student'])
    if sha(plan_path)!=receipt['plan_sha256'] or sha(actor)!=receipt['student_sha256']:
        raise ValueError('Frozen DAgger model or paired plan changed')
    plan=load_plan(plan_path)
    keys=list(map(tuple,scope['structure_keys']))
    if set(keys)!=set(select_collection_grid(plan['selected_structure_keys'])):
        raise ValueError('Frozen collection grid differs from the original 4x4x5x5 selection')
    provenance=dict(round5=str(round5),completion_sha256=sha(round5/'completed.json'),
        round5_manifest_sha256=sha(round5/'manifest.json'),collection_manifest_sha256=sha(original if archive_hash else base_manifest),
        source_plan=str(plan_path),source_plan_sha256=sha(plan_path),initial_actor=str(actor),
        initial_actor_sha256=sha(actor))
    return plan,keys,actor,provenance


def validate_scenarios(plan, rows_by_task):
    if set(rows_by_task)!=set(TASKS):raise ValueError('Missing task scenario audit')
    summary={}
    for task,n in LAYOUT.items():
        ids=plan['teachers'][task]['design_ids']
        rows=rows_by_task[task]
        if len(ids)!=400 or len(set(ids))!=400 or len(rows)!=n:
            raise ValueError(f'{task}: scenario count does not match frozen layout')
        by_design=defaultdict(list)
        for i,row in enumerate(rows):
            if (row['environment_id']!=i or row['task']!=task or row['design_id']!=ids[i%400]):
                raise ValueError(f'{task}: live scenario routing differs')
            by_design[row['design_id']].append(row)
        counts={key:len(group) for key,group in by_design.items()}
        if len(counts)!=400 or max(counts.values())-min(counts.values())>1:
            raise ValueError(f'{task}: incomplete or nonuniform morphology coverage')
        audit=dict(environments=n,design_count=400,envs_per_design=counts)
        if task=='stairs':
            strata=Counter()
            for key,group in by_design.items():
                bins=Counter(r['terrain_stratum'] for r in group)
                if set(bins)!=set(range(6)) or max(bins.values())-min(bins.values())>1:
                    raise ValueError(f'{task}: missing terrain for {key}')
                strata.update(bins)
            audit['terrain_counts']=dict(strata)
        if task=='backpack':
            strata=Counter()
            allowed={(.1,5.),(5.,10.),(10.,15.)}
            for key,group in by_design.items():
                bins=[tuple(r['payload']['sampling_mass_range_kg']) for r in group]
                if not set(bins)<=allowed or len(set(bins))!=len(bins):
                    raise ValueError(f'{task}: repeated or invalid load stratum for {key}')
                strata.update(bins)
            audit['payload_strata']={f'{a}-{b}':n for (a,b),n in strata.items()}
        summary[task]=audit
    return summary


def audit_live_layout(root,attempt,plan,tasks,counts):
    if tuple(counts)!=tuple(LAYOUT.values()) or tuple(t.name for t in tasks)!=TASKS:
        raise ValueError('Mixed scene differs from frozen environment allocation')
    rows={t:read(attempt/t/'scenarios.json') for t in TASKS}
    audit=validate_scenarios(plan,rows)
    for t in tasks:
        ids=plan['teachers'][t.name]['design_ids']
        if (list(t.raw.cfg.spatial_design_ids)!=ids or t.raw.num_envs!=LAYOUT[t.name]
                or t.raw.spatial_groups.bincount(minlength=400).cpu().tolist()!=[audit[t.name]['envs_per_design'][i] for i in ids]):
            raise ValueError(f'{t.name}: live PhysX context count differs')
    amp=read(attempt/'run/reference_routing_audit.json')
    if not amp['verified'] or amp['design_ids']!=[r['design_id'] for r in rows['run']]:
        raise ValueError('AMP native reference routing was not verified for the 400 structures')
    audit['run']['reference_directories']=len(set(amp['per_env_libraries']))
    write_new(attempt/'pretrain_layout_verified.json',audit)
    print('SPATIAL_PRETRAIN_LAYOUT_VERIFIED', {t:(v['environments'],v['design_count']) for t,v in audit.items()},flush=True)


def publish_pretraining(a,root,model,study,digest,protocol):
    from finetune_common import check_pause
    from student import load_student,save_student
    import torch
    check_pause(root);assert_protocol_unchanged(protocol)
    actor=root/'shared_actor_pretrained.pt'
    expected=dict(pretrain_updates=a.updates,initial_actor_sha256=sha(a.student),manifest_sha256=digest,
                  collection_protocol=protocol,training_scope='shared_spatial_pretrain_400')
    if actor.exists():
        old,meta=load_student(actor)
        if (any(meta.get(k)!=v for k,v in expected.items()) or old.schema!=model.schema
                or old.architecture!=model.architecture
                or any(not torch.equal(v,old.state_dict()[k]) for k,v in model.state_dict().items())):
            raise ValueError('Existing pretrained actor differs from learner checkpoint')
    else:save_student(actor,model,**expected)
    commit=root/f'checkpoints/update_{a.updates:06d}.json'
    record=dict(result='SHARED_SPATIAL_PRETRAINED_REQUIRES_VALIDATION',pretrain_updates=a.updates,
        manifest_sha256=digest,student=str(actor),student_sha256=sha(actor),checkpoint_receipt_sha256=sha(commit),
        envs_per_task=LAYOUT,structures_per_task=400)
    receipt=root/'training_completed.json'
    if receipt.exists():
        if read(receipt)!=record:raise ValueError('Pretraining completion receipt changed')
    else:write_new(receipt,record)
    print('SHARED_SPATIAL_PRETRAINED',receipt,flush=True)


def worker_main():
    from isaaclab.app import AppLauncher
    from mixed_finetune import train
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--round5',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=DEFAULT_UPDATES)
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-paused',action='store_true')
    AppLauncher.add_app_launcher_args(parser)
    a=parser.parse_args()
    if a.updates<1 or not a.headless or not a.device.startswith('cuda'):
        parser.error('Positive updates and headless CUDA are required')
    a.profile='robust';a.randomization_strength=1.;a.init_at_random_ep_len=True
    a.experimental_profile=CURRENT_PPO_PROFILE;a.ppo_algorithm=CURRENT_PPO_ALGORITHM
    a.compact_training_artifacts=False
    plan,keys,a.student,provenance=load_round5(a.round5)
    frozen,allocation=select_training_plan(plan,keys,a.seed)
    protocol=protocol_metadata(a.profile,a.randomization_strength)
    root=a.output.resolve()
    study=dict(structures_per_task=400,envs_per_task=LAYOUT,allocation=allocation)
    manifest=dict(kind='shared_spatial_pretrain_400',provenance=provenance,plan_sha256=provenance['source_plan_sha256'],
        initial_actor_sha256=provenance['initial_actor_sha256'],updates=a.updates,seed=a.seed,
        collection_protocol=protocol,ppo_schedule=PPO_SCHEDULE,ppo_algorithm=CURRENT_PPO_ALGORITHM,
        native_ppo_settings=CURRENT_PPO_SETTINGS,teacher_checkpoint_sha256={t:frozen['teachers'][t]['checkpoint_sha256'] for t in TASKS},
        init_at_random_ep_len=True,envs_per_task=LAYOUT,steps_per_task=STEPS,study=study,
        sources={name:sha(HERE/name) for name in SOURCE_FILES})
    with run_lock(root):
        if a.resume or a.resume_paused:
            if read(root/'manifest.json')!=manifest or load_plan(root/'plan.json')!=frozen:
                raise ValueError('Pretraining inputs, settings, layout or sources changed')
        else:
            root.mkdir(parents=True,exist_ok=True)
            if any(root.iterdir()):raise ValueError('New pretraining output already exists')
            write_new(root/'plan.json',frozen);write_new(root/'manifest.json',manifest)
        resume_pause(root,a.resume_paused)
        attempt=root/'attempts'/uuid4().hex;attempt.mkdir(parents=True,exist_ok=False)
        a.kit_args=(a.kit_args or '')+f' --/renderer/multiGpu/enabled=false --portable-root {attempt/"portable"} --portable'
        try:
            app=AppLauncher(a).app
            train(a,root,attempt,frozen,study,protocol,tuple(LAYOUT.values()),
                  layout_callback=audit_live_layout,completion_callback=publish_pretraining)
        except BaseException as error:
            if isinstance(error,KeyboardInterrupt):(root/'USER_PAUSED').touch()
            write_new(attempt/'failure.json',dict(traceback=traceback.format_exc()))
            traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


def supervisor_main():
    sys.path.insert(0,str(HERE.parent))
    from original_expert_comparison import run_child,wait_gpu,status,check_pause
    from robolab.rough_gpu import gpu_lease,child_environment
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round5',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=DEFAULT_UPDATES)
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--resume-paused',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    def pause(*_):(root/'USER_PAUSED').touch(exist_ok=True)
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
    with run_lock(root):
        try:
            resume_pause(root,args.resume_paused);check_pause(root)
            tuning=root/'tuning'
            cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',
                 '--round5',str(args.round5.resolve()),'--output',str(tuning),'--updates',str(args.updates),
                 '--seed',str(args.seed),'--headless','--device','cuda:0']
            if (tuning/'manifest.json').exists():cmd.append('--resume')
            if (tuning/'USER_PAUSED').exists():
                if not args.resume_paused:raise KeyboardInterrupt('Worker is paused')
                cmd.append('--resume-paused')
            status(root,'training_spatial_pretrain',target=args.updates,envs_per_task=LAYOUT,structures_per_task=400)
            # Use the standard per-checkout GPU lease; no other local workers are stopped.
            lease_root=HERE.parents[2]/'logs/rpo_rough_multimorph/gpu_locks'
            with gpu_lease(wait_gpu(root,args.gpu,22000),lease_root) as gpu:
                cmd += [f'--kit_args=--/renderer/activeGpu={gpu["index"]} --/physics/cudaDevice=0']
                run_child(cmd,root/'training_logs'/uuid4().hex,child_environment(gpu,os.environ),root,
                          grace=15,heartbeat=tuning/'mean_reward.csv',timeout=12*3600)
            result=read(tuning/'training_completed.json')
            if (result['result']!='SHARED_SPATIAL_PRETRAINED_REQUIRES_VALIDATION'
                    or result['pretrain_updates']!=args.updates or sha(result['student'])!=result['student_sha256']):
                raise ValueError('Pretraining worker completion was not verified')
            status(root,'pretrained_requires_validation',target=args.updates,actor=result['student'],actor_sha256=result['student_sha256'])
        except KeyboardInterrupt:
            (root/'USER_PAUSED').touch(exist_ok=True);status(root,'paused');raise SystemExit(130)
        except Exception as e:
            status(root,'failed',error=str(e));raise


if __name__=='__main__':
    if '--worker' in sys.argv:worker_main()
    else:supervisor_main()
