"""Persistent physical sampler for synchronized three-task PPO rounds."""
import json,time
from pathlib import Path
import torch
from student import load_student

def publish(path,value):
    path=Path(path);temporary=path.with_suffix('.writing')
    torch.save(value,temporary);temporary.replace(path)

def serve(channel,task,env,raw,obs,canonical,clip,reward_adapter=None):
    root=Path(channel);index=0;reset=None
    base=canonical(obs,reset)
    while True:
        request=root/f'request_{index:03d}.json';started=time.monotonic()
        while not request.exists():
            if (root/'STOP').exists() or (root/'USER_PAUSED').exists():return
            if time.monotonic()-started>900:raise TimeoutError('PPO coordinator request missing')
            time.sleep(.1)
        spec=json.loads(request.read_text());model,_=load_student(spec['student'],raw.device)
        std=float(spec['std']);rows={k:[] for k in ('obs','actions','logp','reward','done','timeout')}
        torque_rows=[]
        for step in range(spec['steps']):
            if (root/'STOP').exists() or (root/'USER_PAUSED').exists():return
            with torch.inference_mode():
                mean=model(base);dist=torch.distributions.Normal(mean,std)
                action=dist.sample();logp=dist.log_prob(action).sum(-1)
                executed=action if clip is None else action.clamp(-clip,clip)
                following,reward,done,extra=env.step(executed)
                if hasattr(raw,'_full_body_torque_diagnostics'):
                    diag=raw._full_body_torque_diagnostics
                    manager=raw.reward_manager;term=manager._term_names.index('full_body_torque_limits')
                    term_cfg=manager._term_cfgs[term]
                    actual=manager._step_reward[:,term]*raw.step_dt
                    expected=diag['cost']*(-12.)*raw.step_dt
                    if term_cfg.weight != -12. or not torch.allclose(actual,expected,atol=1e-6,rtol=1e-5):
                        raise ValueError('Full-body native reward weight/dt mismatch')
                    torque_rows.append(dict(mean_penalty=float(actual.mean()),
                        max_hard_clip_excess_nm=float(diag['hard_clip_excess_nm'].max()),
                        unavailable=int(diag['unavailable'].sum()),
                        over80=float((diag['utilization']>.8).float().mean())))
                if reward_adapter:reward=reward_adapter(following,reward)
                values=dict(obs=base,actions=action,logp=logp,reward=reward,
                    done=done,timeout=extra.get('time_outs',torch.zeros_like(done)))
                for key,value in values.items():rows[key].append(value.detach().cpu().clone())
                base=canonical(following,done.bool())
        audit=None
        if hasattr(raw,'_full_body_torque_diagnostics'):
            audit=dict(joint_count=raw._full_body_torque_diagnostics['utilization'].shape[-1],
                weight=-12.,threshold=.8,control_dt=float(raw.step_dt),dt_verified=True,steps=torque_rows,
                reward_terms=list(raw.reward_manager._term_names),
                groups={name:dict(class_name=type(motor).__name__,joints=list(motor.joint_names),
                    limits_finite=bool(torch.isfinite(motor.dynamic_effort_limit).all())) for name,motor in raw.robot.actuators.items()})
        publish(root/f'{task}_{index:03d}.pt',dict(task=task,index=index,full_body_audit=audit,
            **{k:torch.stack(v) for k,v in rows.items()},next_obs=base.cpu()))
        print('PPO_PHYSICAL_BATCH',task,index,flush=True);index+=1
