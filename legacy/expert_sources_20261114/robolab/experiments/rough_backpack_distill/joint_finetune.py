"""Exactly N synchronized PPO updates of one actor from three native tasks."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch
from student import load_student,save_student
from contract import load_plan,sha,write_new
from ppo_channel import publish
HERE=Path(__file__).resolve().parent

def returns(reward,done,timeout,value,last,gamma=.99,lam=.95):
    result=torch.empty_like(reward);adv=torch.zeros_like(last)
    reward=reward+gamma*value*timeout.float()
    for t in reversed(range(len(reward))):
        nxt=last if t==len(reward)-1 else value[t+1]
        live=1-done[t].float()
        adv=reward[t]+gamma*nxt*live-value[t]+gamma*lam*live*adv
        result[t]=adv+value[t]
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--student',required=True)
    p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=100)
    p.add_argument('--full-body-torque',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    root=Path(a.output).resolve();root.mkdir(exist_ok=False);plan=load_plan(a.plan)
    selected={}
    for task,spec in plan['teachers'].items():
        bank=json.loads(Path(spec['bank']).read_text(encoding='utf8'))
        matching=[d for d in bank['designs'] if abs(d['thigh_m']-.25)<1e-8 and abs(d['calf_m']-.30)<1e-8
            and d['knee_motor']=='DM-J10010L-2EC' and d['ankle_motor']=='RS06' and d['id'] in spec['design_ids']]
        if len(matching)!=1:raise ValueError(f'Target exact design missing: {task}')
        selected[task]=matching[0];spec['copies']=8
    frozen=root/'plan.json';write_new(frozen,plan);load_plan(frozen)
    model,_=load_student(a.student);critic=torch.nn.Sequential(torch.nn.Linear(model.width,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,1))
    optimizer=torch.optim.Adam([{'params':model.parameters(),'lr':3e-6},{'params':critic.parameters(),'lr':1e-3}])
    manifest=dict(initial_actor_sha256=sha(a.student),target=selected,updates=a.updates,
        full_body_torque=a.full_body_torque,
        method='joint clipped PPO; all three tasks collected before each shared update',
        steps_per_task=24,envs_per_task=8,actor_lr=3e-6,critic_lr=1e-3,std=.05,
        AMP='frozen native trained discriminator and normalization; native task/style reward interpolation',
        teacher_actor_updates=0,critic_initialization='fresh task-conditioned canonical input')
    write_new(root/'manifest.json',manifest)
    workers=[];handles=[]
    try:
        for task,d in selected.items():
            folder=root/f'worker_{task}';script=HERE/('rollout_amp.py' if task=='run' else 'rollout.py')
            cmd=[sys.executable,'-u',str(script),'--plan',str(frozen),'--design-ids',d['id'],
                '--output',str(folder),'--mode','evaluate','--student',a.student,'--ppo-channel',str(root),
                '--headless','--device','cuda:0','--seed','20261104']
            if task!='run':cmd+=['--task',task]
            if a.full_body_torque:cmd+=['--full-body-torque']
            outs=[open(root/f'{task}.{s}.log','x',encoding='utf8') for s in ('stdout','stderr')];handles+=outs
            kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            proc=subprocess.Popen(cmd,stdout=outs[0],stderr=outs[1],**kwargs);workers.append(proc)
            write_new(root/f'{task}.launch.json',dict(pid=proc.pid,command=cmd))
            # Stagger scene initialization to avoid converter/Kit peaks.
            time.sleep(3)
        for index in range(a.updates):
            if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
            actor=root/f'actor_{index:03d}.pt';save_student(actor,model,ppo_updates=index)
            write_new(root/f'request_{index:03d}.json',dict(student=str(actor),std=.05,steps=24))
            started=time.monotonic();batches=[]
            while not all((root/f'{task}_{index:03d}.pt').exists() for task in selected):
                if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED')
                for proc in workers:
                    if proc.poll() is not None:raise RuntimeError(f'Physical worker exited {proc.pid}: {proc.returncode}; inspect native logs')
                if time.monotonic()-started>900:raise TimeoutError('Physical task batch timeout')
                time.sleep(.2)
            for task in selected:
                data=torch.load(root/f'{task}_{index:03d}.pt',weights_only=True)
                x=data['obs'];shape=x.shape[:2]
                with torch.no_grad():
                    v=critic(x.flatten(0,1)).reshape(shape);last=critic(data['next_obs']).squeeze(-1)
                    target=returns(data['reward'],data['done'],data['timeout'],v,last)
                    adv=target-v;adv=(adv-adv.mean())/(adv.std()+1e-8)
                batches.append((x.flatten(0,1),data['actions'].flatten(0,1),data['logp'].flatten(),target.flatten(),adv.flatten()))
            x,act,oldlog,target,adv=[torch.cat(parts) for parts in zip(*batches)]
            logs=[]
            for epoch in range(4):
                order=torch.randperm(len(x))
                for ids in order.split(192):
                    dist=torch.distributions.Normal(model(x[ids]),.05);log=dist.log_prob(act[ids]).sum(-1)
                    ratio=(log-oldlog[ids]).exp();policy=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                    value=(critic(x[ids]).squeeze(-1)-target[ids]).square().mean()
                    loss=policy+value
                    optimizer.zero_grad();loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_([*model.parameters(),*critic.parameters()],1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite PPO update')
                    optimizer.step();logs.append(float(loss.detach()))
            publish(root/f'ppo_{index+1:03d}.pt',dict(actor=model.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),completed_updates=index+1))
            write_new(root/f'update_{index+1:03d}.json',dict(completed_updates=index+1,loss=sum(logs)/len(logs),tasks=list(selected),samples=len(x)))
            print('JOINT_PPO_UPDATE',index+1,'/',a.updates,flush=True)
        final=root/'shared_actor_finetuned.pt';save_student(final,model,ppo_updates=a.updates,initial_actor_sha256=sha(a.student))
        write_new(root/'completed.json',dict(completed_updates=a.updates,student=str(final),student_sha256=sha(final),requires_evaluation=True))
    finally:
        (root/'STOP').touch()
        for proc in workers:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.terminate()
        for handle in handles:handle.close()

if __name__=='__main__':main()
