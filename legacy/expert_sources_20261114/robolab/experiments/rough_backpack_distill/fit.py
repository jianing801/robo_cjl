"""Balanced real-data BC/DAgger for ONE rough+backpack actor; no expert/PPO updates."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import torch
from contract import load_plan,sha,write_new,input_width,graph_features,unified_context
from student import SharedActor,SharedMLPActor,load_student,save_student


def load_groups(paths,plan,plan_sha,max_rows=0):
    groups=defaultdict(lambda:dict(train=[],validation=[]));provenance=[];io=None
    banks={t:json.loads(Path(s['bank']).read_text()) for t,s in plan['teachers'].items()}
    designs={t:{d['id']:d for d in b['designs']} for t,b in banks.items()}
    static_checked={}
    for path in paths:
        data=torch.load(path,map_location='cpu',weights_only=True)
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
        contract=data['io_contract']
        semantic=plan['schema']['semantic_contract']
        if (contract['joint_names']!=semantic['joint_names'] or contract['default_joint_pos']!=semantic['default_joint_pos']
            or contract['step_dt']!=.02 or contract['action_scale']!=.25 or contract['history_width']!=780
            or contract['history_layout']!='frame_major_oldest_to_newest'):
            raise ValueError('Native control semantics mismatch')
        if io is None:io=contract
        if contract!=io:raise ValueError('Native teacher observation/action contracts differ')
        x,y=data['base_obs'].float(),data['teacher_actions'].float()
        if x.shape!=(len(y),780) or y.shape[1:]!=(23,) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Bad actual teacher tensors')
        validation=data['validation'].bool();env=data['environment_ids']
        if set(env[validation].tolist())&set(env[~validation].tolist()):raise ValueError('Environment validation leakage')
        for i,design in enumerate(data['design_ids']):
            if design not in spec['design_ids']:raise ValueError('Unknown design')
            indices=data['design_index']==i
            static=data['static_context'][i].float()
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
                selected=mask.nonzero().flatten()
                if max_rows and len(selected)>max_rows:
                    selected=selected[torch.linspace(0,len(selected)-1,max_rows).long()]
                groups[key][split].append((x[selected],y[selected],static))
        provenance.append(dict(path=str(Path(path).resolve()),sha256=sha(path),source=data['source'],
                               teacher=spec['checkpoint_sha256'],task=task,rows=len(x)))
    required={(t,d) for t,s in plan['teachers'].items() for d in s['design_ids']}
    if set(groups)!=required:raise ValueError(f'Incomplete teacher/task/design coverage: {len(required-set(groups))} missing')
    return dict(groups),provenance,io


def assemble(x,static,task_id,task_count=2):
    task=x.new_zeros(len(x),task_count);task[:,task_id]=1
    return torch.cat((x,task,static.expand(len(x),-1)),-1)


def training_budget(groups,requested_steps,samples_per_task=128):
    """An automatic epoch is one expected pass of the largest task pool.

    Balanced random sampling is with replacement, so this is an expected
    exposure count, never a guarantee that every individual row was visited.
    """
    counts=defaultdict(int)
    for (task,_),group in groups.items():
        counts[task]+=sum(len(x) for x,_,_ in group['train'])
    if requested_steps<0 or not counts or min(counts.values())<1:
        raise ValueError('Invalid training budget or empty task')
    steps=requested_steps or math.ceil(max(counts.values())/samples_per_task)
    return steps,dict(counts)


def batch(groups,generator,plan,designs_per_task=8,samples=16):
    xs=[];ys=[]
    for task in plan['schema']['tasks']:
        keys=sorted(k for k in groups if k[0]==task)
        chosen=torch.randint(len(keys),(designs_per_task,),generator=generator)
        for i in chosen:
            shards=groups[keys[int(i)]]['train']
            shard=shards[int(torch.randint(len(shards),(1,),generator=generator))]
            x,y,static=shard
            rows=torch.randint(len(x),(samples,),generator=generator)
            xs.append(assemble(x[rows],static,plan['task_ids'][task],len(plan['schema']['tasks'])));ys.append(y[rows])
    return torch.cat(xs),torch.cat(ys)


def validation(model,groups,plan,device,cap=64):
    result={};model.eval()
    with torch.inference_mode():
        for key,group in groups.items():
            total=0.;count=0
            for x,y,static in group['validation']:
                rows=torch.linspace(0,len(x)-1,min(len(x),cap)).long()
                prediction=model(assemble(x[rows],static,plan['task_ids'][key[0]],len(plan['schema']['tasks'])).to(device))
                total+=float((prediction-y[rows].to(device)).square().sum());count+=len(rows)*23
            result['/'.join(key)]=total/count
    means={t:sum(v for k,v in result.items() if k.startswith(t+'/'))/sum(k.startswith(t+'/') for k in result)
           for t in plan['schema']['tasks']}
    return result,means


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--data-list',required=True)
    p.add_argument('--output',required=True);p.add_argument('--init-student')
    p.add_argument('--architecture',choices=['mxt','mlp'],default='mxt')
    p.add_argument('--max-rows-per-design-shard',type=int,default=0,
        help='Bound replay memory with deterministic time-spread rows per design/source/split; 0 keeps all')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--steps-per-epoch',type=int,default=0,
        help='0 scales each epoch to the largest task training pool; positive values explicitly cap updates')
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--device',default='cuda:0')
    p.add_argument('--observation-normalization',choices=['native','empirical'],default='native',
        help='Default keeps native sensor scales, as in the earlier successful fixed-RPO MxT recipe')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--yaw-loss-weight',type=float,default=1.,help='Explicit supervised yaw-joint error emphasis; no change to actions or physics')
    p.add_argument('--stairs-loss-weight',type=float,default=1.,help='Explicit stair imitation priority; retains all task replay')
    p.add_argument('--reparameterize-observations',action='store_true',help='Calibrate initialized MLP with exact first-layer coordinate compensation')
    a=p.parse_args()
    if not 1<=a.yaw_loss_weight<=8:raise ValueError('Bounded yaw loss emphasis required')
    if not 1<=a.stairs_loss_weight<=4:raise ValueError('Bounded stairs loss emphasis required')
    if a.epochs<1 or a.steps_per_epoch<0 or a.max_rows_per_design_shard<0 or not 0<a.learning_rate<=.001:raise ValueError('Explicit finite BC budget required')
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    plan=load_plan(a.plan);paths=json.loads(Path(a.data_list).read_text())
    groups,sources,io=load_groups(paths,plan,sha(a.plan),a.max_rows_per_design_shard)
    resolved_steps,training_rows=training_budget(groups,a.steps_per_epoch)
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
            for x,_,_ in group['train']:
                total+=x.double().sum(0);squared+=x.double().square().sum(0);count+=len(x)
        mean=total/count;std=(squared/count-mean.square()).clamp(min=.01).sqrt()
        if a.reparameterize_observations:
            with torch.no_grad():
                probe,_=batch(groups,torch.Generator().manual_seed(184),plan)
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
        sampling='equal tasks, uniform design within task, uniform shard within design',
        input_width=input_width(plan['schema']),architecture=model.architecture,epochs=a.epochs,steps_per_epoch=resolved_steps,
        requested_steps_per_epoch=a.steps_per_epoch,training_rows_by_task=training_rows,
        replay_row_cap_per_design_source_split=a.max_rows_per_design_shard,
        expected_sampled_passes_by_task={t:a.epochs*resolved_steps*128/n for t,n in training_rows.items()},
        observation_normalization=a.observation_normalization,normalization_reparameterization_max_error=normalization_parity,yaw_loss_weight=a.yaw_loss_weight,stairs_loss_weight=a.stairs_loss_weight,
        teacher_updates=0,critic_trained=False,PPO_updates=0,init_student=a.init_student))
    best=float('inf');best_path=None;started=time.monotonic()
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
                 balanced_validation_MSE=score,per_design_validation_MSE=details,wall_seconds=time.monotonic()-started)
        write_new(output/f'epoch_{epoch:04d}.json',row)
        print('DISTILL_EPOCH',json.dumps({k:v for k,v in row.items() if k!='per_design_validation_MSE'}),flush=True)
        if score<best:
            best=score;best_path=output/f'student_epoch_{epoch:04d}.pt'
            save_student(best_path,model,plan_sha256=sha(a.plan),teachers={t:s['checkpoint_sha256'] for t,s in plan['teachers'].items()},
                         validation_MSE=means,epoch=epoch,closed_loop_required=True)
    write_new(output/'summary.json',dict(result='BC_COMPLETE_NOT_GAIT_VALIDATED',student=str(best_path),
        student_sha256=sha(best_path),balanced_validation_MSE=best,optimizer_steps=a.epochs*resolved_steps,
        tasks=plan['schema']['tasks'],covered_task_designs=len(groups)))


if __name__=='__main__':main()
