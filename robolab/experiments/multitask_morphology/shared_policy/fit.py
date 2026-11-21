"""Balanced real-data BC/DAgger for one morphology- and task-conditioned actor; no expert/PPO updates."""
import argparse
from collections import defaultdict,OrderedDict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from contract import load_plan,sha,write_new,input_width,graph_features,unified_context,design_ids_for_keys
from collection_grid import select_collection_grid
from student import SharedActor,SharedMLPActor,load_student,save_student
from randomization import PROTOCOL_SOURCE_FILES,PROFILE_VERSION


def _sha256_digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def collection_protocol(data):
    """Validate shared collection semantics without equating teacher internals."""
    protocol=data.get('collection_protocol')
    if protocol is None:return None
    if not isinstance(protocol,dict) or set(protocol)!={'version','profile','protocol_hash','strength','source_sha256'}:
        raise ValueError('Invalid collection_protocol metadata')
    digest=protocol['protocol_hash'];strength=protocol['strength']
    if (protocol['version']!=PROFILE_VERSION or protocol['profile'] not in ('nominal','robust')
        or not _sha256_digest(digest)
        or isinstance(strength,bool) or not isinstance(strength,(int,float)) or not math.isfinite(strength) or not 0<=strength<=1):
        raise ValueError('Invalid collection_protocol metadata')
    sources=protocol['source_sha256']
    if (not isinstance(sources,dict) or set(sources)!=set(PROTOCOL_SOURCE_FILES)
        or not all(_sha256_digest(value) for value in sources.values())):
        raise ValueError('Invalid collection_protocol source_sha256 metadata')
    contract=data['io_contract']
    if (contract.get('student_action_clip')!=100. or contract.get('student_observation_clip')!=100.
        or contract.get('previous_action')!='executed_unscaled'):
        raise ValueError('Shared student action/observation boundary semantics mismatch')
    return protocol


class ReplayStore:
    """Bounded cache of read-only memory-mapped torch archives, never a RAM fallback."""
    def __init__(self,max_cached=2):
        if max_cached<1:raise ValueError('At least one memory-mapped replay cache slot is required')
        self.max_cached=max_cached;self.cache=OrderedDict();self.identities={};self.loads=0

    def load(self,path):
        path=str(Path(path).resolve());stat=Path(path).stat()
        identity=(stat.st_size,stat.st_mtime_ns)
        if path in self.identities and self.identities[path]!=identity:
            raise ValueError(f'Replay archive changed during fitting: {path}')
        self.identities[path]=identity
        if path in self.cache:
            self.cache.move_to_end(path);return self.cache[path]
        # Evict before opening another mapping. Shards retain only copied
        # metadata indices and static context, never views into an old archive.
        if len(self.cache)>=self.max_cached:self.cache.popitem(last=False)
        try:data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        except (RuntimeError,TypeError,ValueError) as error:
            raise ValueError(f'Memory-mapped replay unavailable for {path}; re-save with torch.save default ZIP format. '
                             'No full-RAM fallback is permitted.') from error
        self.cache[path]=data;self.loads+=1
        return data

    def rows(self,path,indices):
        data=self.load(path)
        # Advanced indexing copies only this requested minibatch from the map.
        return data['base_obs'][indices].float(),data['teacher_actions'][indices].float()


class ReplayShard:
    """All eligible rows, sampled by scenario then environment/episode on demand."""
    def __init__(self,store,path,indices,static,environment_ids,scenario_bin=None,episode_id=None,*,track_exposure=True):
        self.store=store;self.path=str(Path(path).resolve());self.static=static.clone()
        if not len(indices):raise ValueError('Empty physical replay split')
        scenario=torch.zeros_like(indices) if scenario_bin is None else scenario_bin[indices]
        episode=torch.zeros_like(indices) if episode_id is None else episode_id[indices]
        keys=torch.stack((scenario,environment_ids[indices],episode),dim=1)
        unique,inverse,counts=torch.unique(keys,dim=0,sorted=True,return_inverse=True,return_counts=True)
        order=torch.argsort(inverse,stable=True)
        self.indices=indices[order].clone()
        self.episode_row_counts=counts
        self.episode_row_starts=counts.cumsum(0)-counts
        _,scenario_counts=torch.unique_consecutive(unique[:,0],return_counts=True)
        self.scenario_episode_counts=scenario_counts
        self.scenario_episode_starts=scenario_counts.cumsum(0)-scenario_counts
        self.seen=torch.zeros(len(indices),dtype=torch.bool) if track_exposure else None
        self.sampled_draws=0
        # This checks the complete split, not a sample or the number of bins.
        if not torch.equal(self.indices.sort().values,indices.sort().values):
            raise ValueError('Replay hierarchy lost or duplicated eligible rows')

    def __len__(self):return len(self.indices)

    def sample_indices(self,count,generator,*,record_exposure=True):
        # Vectorized conditional draws: every stored row has nonzero probability.
        draws=torch.rand((3,count),generator=generator,dtype=torch.float64)
        scenario=(draws[0]*len(self.scenario_episode_counts)).long()
        episode=self.scenario_episode_starts[scenario]+(draws[1]*self.scenario_episode_counts[scenario]).long()
        row=self.episode_row_starts[episode]+(draws[2]*self.episode_row_counts[episode]).long()
        if record_exposure and self.seen is not None:self.seen[row]=True;self.sampled_draws+=count
        return self.indices[row]

    def sample(self,count,generator,*,record_exposure=True):
        return self.store.rows(self.path,self.sample_indices(count,generator,record_exposure=record_exposure))

    def chunks(self,size):
        for start in range(0,len(self),size):
            yield self.store.rows(self.path,self.indices[start:start+size])

    def metadata_bytes(self):
        return sum(value.numel()*value.element_size() for value in vars(self).values() if isinstance(value,torch.Tensor))


def _shard_size(shard):return len(shard) if isinstance(shard,ReplayShard) else len(shard[0])


def _shard_chunks(shard,size=256):
    if isinstance(shard,ReplayShard):
        for x,y in shard.chunks(size):yield x,y,shard.static
    else:
        x,y,static=shard
        for start in range(0,len(x),size):yield x[start:start+size],y[start:start+size],static


def replay_exposure(groups):
    result=defaultdict(lambda:dict(eligible_rows=0,sampled_draws=0,unique_sampled_rows=0))
    for (task,_),group in groups.items():
        for shard in group['train']:
            if not isinstance(shard,ReplayShard):continue
            row=result[task];row['eligible_rows']+=len(shard)
            row['sampled_draws']+=shard.sampled_draws
            row['unique_sampled_rows']+=int(shard.seen.sum())
    for row in result.values():row['unique_coverage']=row['unique_sampled_rows']/row['eligible_rows']
    return dict(result)


def replay_memory(groups):
    shards=[shard for group in groups.values() for split in group.values() for shard in split
            if isinstance(shard,ReplayShard)]
    return dict(mmap=True,cache_limit=shards[0].store.max_cached if shards else 0,
        full_replay_rows=sum(_shard_size(shard) for group in groups.values() for shard in group['train']),
        replay_index_bytes=sum(value.numel()*value.element_size() for shard in shards
            for name,value in vars(shard).items() if isinstance(value,torch.Tensor) and name not in ('static','seen')),
        access_bitmap_bytes=sum(shard.seen.numel()*shard.seen.element_size() for shard in shards if shard.seen is not None),
        static_context_bytes=sum(shard.static.numel()*shard.static.element_size() for shard in shards))


def _validate_teacher_tensors(x,y,protocol,chunk_rows=256):
    if (x.ndim!=2 or x.shape!=(len(y),780) or y.shape[1:]!=(23,)
        or not x.is_floating_point() or not y.is_floating_point()):
        raise ValueError('Bad actual teacher tensors')
    for start in range(0,len(x),chunk_rows):
        observations=x[start:start+chunk_rows];actions=y[start:start+chunk_rows]
        if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
            raise ValueError('Bad actual teacher tensors')
        if protocol is not None and (observations.abs().gt(100.).any() or actions.abs().gt(100.).any()):
            raise ValueError('Collected student observations/actions exceed the declared numerical boundary')


def _fair_counts(capacities,budget,generator):
    """Water-fill a fixed budget, randomizing only the indivisible remainder."""
    counts=torch.zeros_like(capacities)
    while budget:
        active=(counts<capacities).nonzero().flatten()
        active=active[torch.randperm(len(active),generator=generator)]
        increment=min(budget,max(1,budget//len(active)))
        for index in active.tolist():
            take=min(increment,int(capacities[index]-counts[index]),budget)
            counts[index]+=take;budget-=take
            if not budget:break
    return counts


def stratified_rows(indices,environment_ids,scenario_bin=None,episode_id=None,*,max_rows=0,seed=0):
    """Retain a fixed total budget, balanced by scenario then physical episode.

    Episode IDs are local to an environment. Old shards have no scenario or
    episode annotations, so their environments are the only reliable strata.
    Missing scenario coverage is refused; partial episode coverage is reported.
    """
    if max_rows<0:raise ValueError('Negative replay row budget')
    if (scenario_bin is None)!=(episode_id is None):
        raise ValueError('Scenario and episode annotations must be supplied together')
    legacy=scenario_bin is None
    if legacy:
        scenario_bin=torch.zeros_like(environment_ids)
        episode_id=torch.zeros_like(environment_ids)
    scenario=scenario_bin[indices]
    episodes=torch.stack((environment_ids[indices],episode_id[indices]),dim=1)
    scenario_ids=torch.unique(scenario,sorted=True)
    budget=min(len(indices),max_rows) if max_rows else len(indices)
    if budget<len(scenario_ids):
        raise ValueError(f'Replay budget {budget} cannot cover {len(scenario_ids)} scenario bins; increase --max-rows-per-design-shard')
    generator=torch.Generator().manual_seed(seed)
    scenario_masks=[(scenario==value).nonzero().flatten() for value in scenario_ids]
    counts=_fair_counts(torch.tensor([len(rows) for rows in scenario_masks]),budget,generator)
    kept=[];coverage=[]
    for value,rows,count in zip(scenario_ids.tolist(),scenario_masks,counts.tolist()):
        episode_keys,inverse=torch.unique(episodes[rows],dim=0,return_inverse=True)
        episode_rows=[rows[(inverse==i).nonzero().flatten()] for i in range(len(episode_keys))]
        episode_counts=_fair_counts(torch.tensor([len(r) for r in episode_rows]),count,generator)
        for candidates,take in zip(episode_rows,episode_counts.tolist()):
            kept.append(indices[candidates[torch.randperm(len(candidates),generator=generator)[:take]]])
        coverage.append(dict(scenario_bin=value,available_rows=len(rows),retained_rows=count,
            available_episodes=len(episode_keys),retained_episodes=int((episode_counts>0).sum())))
    selected=torch.cat(kept)
    selected=selected[torch.randperm(len(selected),generator=generator)]
    return selected,dict(method='scenario_then_environment_episode',legacy_environment_only=legacy,
        available_rows=len(indices),retained_rows=len(selected),row_budget=max_rows,
        scenario_coverage=coverage,all_scenarios_retained=True,
        all_episodes_retained=all(r['available_episodes']==r['retained_episodes'] for r in coverage))


def load_groups(paths,plan,plan_sha,*,sampling_seed=0,validation_rows=128,replay_cache_shards=2,
                collection_grid=False):
    if validation_rows<1:raise ValueError('A positive fixed validation row budget is required')
    if len(paths)!=len({str(Path(path).resolve()) for path in paths}):
        raise ValueError('Duplicate replay archives would duplicate original-row coverage')
    groups=defaultdict(lambda:dict(train=[],validation=[]));provenance=[];io=None
    store=ReplayStore(replay_cache_shards)
    protocol_identity=None
    banks={t:json.loads(Path(s['bank']).read_text()) for t,s in plan['teachers'].items()}
    designs={t:{d['id']:d for d in b['designs']} for t,b in banks.items()}
    required_ids = {task: spec['design_ids'] for task,spec in plan['teachers'].items()}
    if collection_grid:
        selected = select_collection_grid(plan['selected_structure_keys'])
        required_ids = design_ids_for_keys(plan,selected,banks)
    required = {(task,design) for task,ids in required_ids.items() for design in ids}
    static_checked={}
    for path in paths:
        data=store.load(path)
        task=data['task'];spec=plan['teachers'][task]
        legacy=plan.get('compatible_legacy_data',{}).get(data['plan_sha256'])
        schema_ok=data['schema']==plan['schema'] and data['plan_sha256']==plan_sha
        if legacy is not None:
            schema_ok=data['schema']==legacy and task in legacy['tasks']
            for key in ('knees','ankles','semantic_contract','ranges_m','shared_structure_keys'):
                schema_ok=schema_ok and legacy[key]==plan['schema'][key]
        if (data.get('kind')!='real_rough_backpack_teacher_data_v1' or not schema_ok
            or data['teacher_sha256']!=spec['checkpoint_sha256']
            or data['bank_sha256']!=spec['bank_sha256'] or data.get('actual_payload_not_in_actor') is not True):
            raise ValueError('Wrong real teacher/data/physical identity')
        protocol=collection_protocol(data)
        identity=('legacy',) if protocol is None else (
            *(protocol[k] for k in ('version','profile','protocol_hash')),
            tuple(sorted(protocol['source_sha256'].items())))
        if protocol_identity is None:protocol_identity=identity
        elif identity!=protocol_identity:
            raise ValueError('Incompatible collection protocols: do not mix legacy/new data, profiles or protocol hashes')
        contract=data['io_contract']
        semantic=plan['schema']['semantic_contract']
        if (contract['joint_names']!=semantic['joint_names'] or contract['default_joint_pos']!=semantic['default_joint_pos']
            or contract['step_dt']!=.02 or contract['action_scale']!=.25 or contract['history_width']!=780
            or contract['history_layout']!='frame_major_oldest_to_newest'):
            raise ValueError('Native control semantics mismatch')
        if io is None:io=contract
        if contract!=io:raise ValueError('Native teacher observation/action contracts differ')
        x,y=data['base_obs'],data['teacher_actions']
        _validate_teacher_tensors(x,y,protocol)
        validation=data['validation'].bool();env=data['environment_ids']
        if validation.shape!=(len(x),) or env.shape!=(len(x),) or env.dtype!=torch.int64 or (env<0).any():
            raise ValueError('Invalid environment/split row annotations')
        routing=data['design_index'];design_ids=data['design_ids']
        if (len(design_ids)!=len(set(design_ids)) or routing.shape!=(len(x),) or routing.dtype!=torch.int64
            or (routing<0).any() or (routing>=len(design_ids)).any()):
            raise ValueError('Invalid design routing: every original row must belong to exactly one design')
        scenario=data.get('scenario_bin');episode=data.get('episode_id')
        if (scenario is None)!=(episode is None):raise ValueError('Incomplete scenario/episode annotations')
        if protocol is not None:
            command=data.get('actual_command')
            if (scenario is None or command is None or command.shape!=(len(x),3)
                or not command.is_floating_point() or not torch.isfinite(command).all()):
                raise ValueError('New collection protocol requires scenario, episode and actual_command annotations')
            if (command[:,0] <= .05).any():
                raise ValueError('Continuous-locomotion replay cannot contain standing or non-forward commands')
        for name,values in [('scenario_bin',scenario),('episode_id',episode)]:
            if values is not None and (values.shape!=(len(x),) or values.dtype!=torch.int64 or (values<0).any()):
                raise ValueError(f'Invalid {name} row annotations')
        if set(env[validation].tolist())&set(env[~validation].tolist()):raise ValueError('Environment validation leakage')
        sampling=[];source_sha=sha(path)
        for i,design in enumerate(data['design_ids']):
            if (task,design) not in required:raise ValueError(f'Design outside expected collection scope: {task}/{design}')
            indices=data['design_index']==i
            static=data['static_context'][i].float().clone()
            key=(task,design)
            if key not in static_checked:
                physical=designs[task][design]
                if sha(physical['urdf'])!=physical['urdf_sha256']:raise ValueError('Physical source changed')
                graph=graph_features(physical['urdf'],semantic['joint_names'],semantic['default_joint_pos'])
                expected=[v for row in graph for v in row]+unified_context(banks[task],physical,plan['schema'])
                static_checked[key]=torch.tensor(expected,dtype=torch.float32)
            if not torch.equal(static,static_checked[key]):raise ValueError('Recorded static morphology differs from actual asset')
            for split,mask in [('train',indices&~validation),('validation',indices&validation)]:
                if not mask.any():raise ValueError('Missing independent physical stream')
                identity=f'{sampling_seed}:{source_sha}:{task}:{design}:{split}'
                seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big')%(2**63)
                eligible=mask.nonzero().flatten()
                if split=='validation':
                    selected,coverage=stratified_rows(eligible,env,scenario,episode,max_rows=validation_rows,seed=seed)
                else:
                    selected=eligible
                    coverage=dict(method='full_replay_on_demand_scenario_then_environment_episode',
                        available_rows=len(eligible),retained_rows=len(eligible),all_original_rows_eligible=True,
                        legacy_environment_only=scenario is None)
                shard=ReplayShard(store,path,selected,static,env,scenario,episode,track_exposure=split=='train')
                coverage['index_and_static_bytes']=shard.metadata_bytes()
                sampling.append(dict(design=design,split=split,seed=seed,**coverage))
                groups[key][split].append(shard)
        provenance.append(dict(path=str(Path(path).resolve()),sha256=source_sha,source=data['source'],
                               teacher=spec['checkpoint_sha256'],task=task,rows=len(x),sampling=sampling,
                               collection_protocol=protocol,native_boundaries=data.get('native_boundaries'),
                               full_observation_action_bytes=x.numel()*x.element_size()+y.numel()*y.element_size(),
                               replay_storage='torch.load mmap=True; bounded archive cache; no eager fallback'))
    if set(groups)!=required:raise ValueError(f'Incomplete teacher/task/design coverage: {len(required-set(groups))} missing')
    return dict(groups),provenance,io


def assemble(x,static,task_id,task_count=2):
    task=x.new_zeros(len(x),task_count);task[:,task_id]=1
    return torch.cat((x,task,static.expand(len(x),-1)),-1)


def training_budget(groups,requested_steps,samples_per_task=128,samples_per_design_shard=32):
    """Bound compute independently of full replay size; never discard rows.

    The default matches the old 32-row/source compute scale. It is a sampling
    budget, not an epoch over all rows; measured unique coverage is separate.
    """
    if samples_per_design_shard<1:raise ValueError('A positive per-design/source sampling budget is required')
    counts=defaultdict(int);budget=defaultdict(int)
    for (task,_),group in groups.items():
        counts[task]+=sum(_shard_size(shard) for shard in group['train'])
        budget[task]+=sum(min(_shard_size(shard),samples_per_design_shard) for shard in group['train'])
    if requested_steps<0 or not counts or min(counts.values())<1:
        raise ValueError('Invalid training budget or empty task')
    steps=requested_steps or math.ceil(max(budget.values())/samples_per_task)
    return steps,dict(counts)


def batch(groups,generator,plan,designs_per_task=8,samples=16,*,record_exposure=True):
    xs=[];ys=[]
    for task in plan['schema']['tasks']:
        keys=sorted(k for k in groups if k[0]==task)
        chosen=torch.randint(len(keys),(designs_per_task,),generator=generator)
        for i in chosen:
            shards=groups[keys[int(i)]]['train']
            shard=shards[int(torch.randint(len(shards),(1,),generator=generator))]
            if isinstance(shard,ReplayShard):
                x,y=shard.sample(samples,generator,record_exposure=record_exposure);static=shard.static
            else:
                x,y,static=shard;rows=torch.randint(len(x),(samples,),generator=generator);x,y=x[rows],y[rows]
            xs.append(assemble(x,static,plan['task_ids'][task],len(plan['schema']['tasks'])));ys.append(y)
    return torch.cat(xs),torch.cat(ys)


def validation(model,groups,plan,device,cap=64):
    result={};model.eval()
    with torch.inference_mode():
        for key,group in groups.items():
            total=0.;count=0
            for shard in group['validation']:
                # Fixed held-out rows, read in bounded chunks. No training draws.
                for x,y,static in _shard_chunks(shard,cap):
                    prediction=model(assemble(x,static,plan['task_ids'][key[0]],len(plan['schema']['tasks'])).to(device))
                    total+=float((prediction-y.to(device)).square().sum());count+=len(x)*23
            result['/'.join(key)]=total/count
    means={t:sum(v for k,v in result.items() if k.startswith(t+'/'))/sum(k.startswith(t+'/') for k in result)
           for t in plan['schema']['tasks']}
    return result,means


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--data-list',required=True)
    p.add_argument('--output',required=True);p.add_argument('--init-student')
    p.add_argument('--collection-grid',action='store_true',help='Require complete replay for the common 400-structure grid')
    p.add_argument('--architecture',choices=['mxt','mlp'],default='mxt')
    p.add_argument('--max-rows-per-design-shard',type=int,default=None,
        help='Retired: training now samples full mmap replay. Use --samples-per-design-per-epoch for compute.')
    p.add_argument('--samples-per-design-per-epoch',type=int,default=32,
        help='Compute budget per design/source, independent of full replay rows; not a row-retention cap')
    p.add_argument('--validation-rows-per-design-shard',type=int,default=128,
        help='Fixed stratified held-out subset per design/source; all selected rows are evaluated in chunks')
    p.add_argument('--replay-cache-shards',type=int,default=2,
        help='Maximum concurrently cached memory-mapped archives; no eager RAM fallback')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--steps-per-epoch',type=int,default=0,
        help='0 uses the explicit design/source sampling budget; positive values directly set updates per epoch')
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--device',default='cuda:0')
    p.add_argument('--observation-normalization',choices=['native','empirical'],default='native',
        help='Default keeps native sensor scales, as in the earlier successful fixed-RPO MxT recipe')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--yaw-loss-weight',type=float,default=1.,help='Explicit supervised yaw-joint error emphasis; no change to actions or physics')
    p.add_argument('--stairs-loss-weight',type=float,default=1.,help='Explicit stair imitation priority; retains all task replay')
    p.add_argument('--reparameterize-observations',action='store_true',help='Calibrate initialized MLP with exact first-layer coordinate compensation')
    a=p.parse_args()
    if a.max_rows_per_design_shard is not None:
        raise ValueError('--max-rows-per-design-shard is retired; use --samples-per-design-per-epoch for compute '
                         'and --validation-rows-per-design-shard for the fixed validation subset. Training never drops a row by budget.')
    if not 1<=a.yaw_loss_weight<=8:raise ValueError('Bounded yaw loss emphasis required')
    if not 1<=a.stairs_loss_weight<=4:raise ValueError('Bounded stairs loss emphasis required')
    if (a.epochs<1 or a.steps_per_epoch<0 or min(a.samples_per_design_per_epoch,a.validation_rows_per_design_shard,a.replay_cache_shards)<1
        or not 0<a.learning_rate<=.001):raise ValueError('Explicit finite BC budget required')
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    plan=load_plan(a.plan);paths=json.loads(Path(a.data_list).read_text())
    groups,sources,io=load_groups(paths,plan,sha(a.plan),sampling_seed=a.seed,
        validation_rows=a.validation_rows_per_design_shard,replay_cache_shards=a.replay_cache_shards,
        collection_grid=a.collection_grid)
    resolved_steps,training_rows=training_budget(groups,a.steps_per_epoch,
        samples_per_design_shard=a.samples_per_design_per_epoch)
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    actor_type=SharedMLPActor if a.architecture=='mlp' else SharedActor
    model=load_student(a.init_student,a.device)[0] if a.init_student else actor_type(plan['schema']).to(a.device)
    if not isinstance(model,actor_type):raise ValueError('Explicit architecture differs from initialized student')
    if model.schema!=plan['schema']:raise ValueError('Wrong student initialization')
    if a.reparameterize_observations and (not a.init_student or a.architecture!='mlp' or a.observation_normalization!='empirical'):
        raise ValueError('Reparameterization requires initialized empirical MLP')
    normalization_parity=None
    if (not a.init_student or a.reparameterize_observations) and a.observation_normalization=='empirical':
        total=torch.zeros(780,dtype=torch.float64);squared=total.clone();count=0
        for group in groups.values():
            for shard in group['train']:
                for x,_,_ in _shard_chunks(shard):
                    total+=x.double().sum(0);squared+=x.double().square().sum(0);count+=len(x)
        mean=total/count;std=(squared/count-mean.square()).clamp(min=.01).sqrt()
        if a.reparameterize_observations:
            with torch.no_grad():
                probe,_=batch(groups,torch.Generator().manual_seed(184),plan,record_exposure=False)
                probe=probe.to(a.device);before=model(probe).clone()
                layer=model.network[0];old_weight=layer.weight[:,:780].clone()
                old_mean=model.observation_mean.clone();old_std=model.observation_std.clone()
                new_mean=mean.float().to(a.device);new_std=std.float().to(a.device)
                layer.bias.add_(old_weight@((new_mean-old_mean)/old_std))
                layer.weight[:,:780].copy_(old_weight*(new_std/old_std))
        model.observation_mean.copy_(mean.float());model.observation_std.copy_(std.float())
        if a.reparameterize_observations:
            with torch.no_grad():
                normalization_parity=float((model(probe)-before).abs().max())
                if normalization_parity>2e-4:raise ValueError('Normalization reparameterization changed policy')
    if a.observation_normalization=='native' and (model.observation_mean.abs().max()!=0 or not model.observation_std.eq(1).all()):
        raise ValueError('Native-scale mode cannot silently reinterpret a normalized student')
    optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
    joint_weights=torch.tensor([a.yaw_loss_weight if 'yaw' in name else 1.
        for name in plan['schema']['semantic_contract']['joint_names']],device=a.device)
    generator=torch.Generator().manual_seed(a.seed)
    write_new(output/'manifest.json',dict(plan_sha256=sha(a.plan),schema=plan['schema'],
        sources=sources,io_contract=io,fit_and_validation_streams_disjoint=True,
        sampling='equal tasks; uniform design, source, scenario and environment/episode; rows sampled on demand from full mmap replay',
        input_width=input_width(plan['schema']),architecture=model.architecture,epochs=a.epochs,steps_per_epoch=resolved_steps,
        requested_steps_per_epoch=a.steps_per_epoch,training_rows_by_task=training_rows,
        replay_row_cap_per_design_source_split=None,samples_per_design_per_epoch=a.samples_per_design_per_epoch,
        replay_cache_shards=a.replay_cache_shards,
        validation_rows_per_design_source_split=a.validation_rows_per_design_shard,
        validation_scope='fixed stratified held-out subset, not all held-out rows or closed-loop gait evaluation',
        exposure_scope='Unique archive-row IDs drawn for optimization within this fit run; does not combine earlier fit runs',
        replay_metadata_bytes=sum(shard.metadata_bytes() for group in groups.values() for split in group.values() for shard in split),
        memory_scope='Observation/action archives are mmap-backed; metadata indices and exposure bitmaps remain in RAM. '
                     'Mapped resident pages and the OS page cache are not a hard process-RSS limit.',
        expected_sampled_passes_by_task={t:a.epochs*resolved_steps*128/n for t,n in training_rows.items()},
        observation_normalization=a.observation_normalization,normalization_reparameterization_max_error=normalization_parity,yaw_loss_weight=a.yaw_loss_weight,stairs_loss_weight=a.stairs_loss_weight,
        teacher_updates=0,critic_trained=False,ppo_updates=0,init_student=a.init_student))
    checkpoints=[];started=time.monotonic()
    for epoch in range(a.epochs):
        model.train();losses=[]
        for _ in range(resolved_steps):
            x,y=batch(groups,generator,plan)
            prediction=model(x.to(a.device))
            task_weights=(1+(a.stairs_loss_weight-1)*x[:,780+plan['task_ids']['stairs']]).to(a.device)
            per_row=((prediction-y.to(a.device)).square()*joint_weights).mean(-1)/joint_weights.mean()
            loss=(per_row*task_weights).sum()/task_weights.sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite distillation')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite distillation gradients')
            optimizer.step();losses.append(float(loss.detach()))
        details,means=validation(model,groups,plan,a.device)
        score=sum(means.values())/len(means)
        row=dict(epoch=epoch,fit_MSE=sum(losses)/len(losses),task_validation_MSE=means,
                 balanced_validation_MSE=score,per_design_validation_MSE=details,
                 cumulative_training_exposure=replay_exposure(groups),wall_seconds=time.monotonic()-started)
        write_new(output/f'epoch_{epoch:04d}.json',row)
        print('DISTILL_EPOCH',json.dumps({k:v for k,v in row.items() if k!='per_design_validation_MSE'}),flush=True)
        checkpoint=output/f'student_epoch_{epoch:04d}.pt'
        save_student(checkpoint,model,plan_sha256=sha(a.plan),teachers={t:s['checkpoint_sha256'] for t,s in plan['teachers'].items()},
                     validation_MSE=means,epoch=epoch,closed_loop_required=True)
        checkpoints.append(dict(epoch=epoch,student=str(checkpoint),student_sha256=sha(checkpoint),
            validation_MSE=means,balanced_validation_MSE=score))
    write_new(output/'summary.json',dict(result='BC_COMPLETE_NOT_GAIT_VALIDATED',student=checkpoints[-1]['student'],
        student_sha256=checkpoints[-1]['student_sha256'],balanced_validation_MSE=checkpoints[-1]['balanced_validation_MSE'],optimizer_steps=a.epochs*resolved_steps,
        tasks=plan['schema']['tasks'],covered_task_designs=len(groups),checkpoints=checkpoints,
        student_selection='Final supervised epoch; DAgger rounds compared by independent closed-loop trials',
        closed_loop_required=True,cumulative_training_exposure=replay_exposure(groups),
        exposure_scope='This fit run only; accessible rows are not a claim that every row was sampled',**replay_memory(groups)))


if __name__=='__main__':main()
