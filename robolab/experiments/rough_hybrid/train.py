"""Independent Rough hybrid controller. Default CPU planning; --execute is explicit."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT.parent/'rsl_rl')]
from hybrid_contract import experiment_identity,load_plan,stage_spec,validate_config,validate_checkpoint,sha
from hybrid_bank import load_configurable_bank
from hybrid_support import write_new

@contextmanager
def controller_lock(root):
    # Persistent lock file, OS advisory lock releases after crash; never unlink a live lock.
    with (root/'controller.lock').open('a+b') as stream:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)

def run_child(command,folder,env):
    import psutil
    folder.mkdir(exist_ok=False)
    started=time.monotonic()
    with (folder/'stdout.log').open('x',encoding='utf8') as out,(folder/'stderr.log').open('x',encoding='utf8') as err:
        environment=dict(env, PYTHONIOENCODING='utf-8')
        kwargs=dict(cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=err,
                    text=True,encoding='utf8',errors='replace',bufsize=1)
        if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
        worker=subprocess.Popen(command,**kwargs)
        write_new(folder/'launch.json',dict(pid=worker.pid,create_time=psutil.Process(worker.pid).create_time(),command=command))
        print('CHILD_START',worker.pid,str(folder),flush=True)
        # Interrupting a parent must never leave an automatically advancing sequence.
        try:
            for line in worker.stdout:
                out.write(line);out.flush()
                print(line,end='',flush=True)
            code=worker.wait()
        except KeyboardInterrupt:
            (folder.parents[2]/'USER_PAUSED').touch(exist_ok=True)
            process=psutil.Process(worker.pid)
            children=process.children(recursive=True)
            process.terminate()
            for child in children:
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            _,alive=psutil.wait_procs([process,*children],timeout=15)
            for remaining in alive:
                try:remaining.kill()
                except psutil.NoSuchProcess:pass
            raise
    write_new(folder/'exit.json',dict(returncode=code,wall_seconds=time.monotonic()-started))
    if code:raise RuntimeError(f'worker exited {code}; inspect {folder}; no next stage started')

def committed(root,plan,bank):
    commits=sorted((root/'commits').glob('stage_*.json'))
    for index,path in enumerate(commits):
        value=json.loads(path.read_text())
        spec=stage_spec(bank,plan['configuration'],index)
        if value['stage']!=spec:raise ValueError('committed schedule changed')
        if sha(value['checkpoint'])!=value['checkpoint_sha256']:raise ValueError('committed checkpoint hash changed')
        summary=json.loads(Path(value['summary']).read_text())
        if summary['result']!='HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED' or summary['completed_updates']!=spec['end_update']:raise ValueError('committed summary invalid')
    return commits

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--bank',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--device')
    p.add_argument('--kit-args',default='')
    p.add_argument('--gpu-min-free-gb',type=float,default=12.)
    p.add_argument('--gpu-max-utilization',type=int,default=30)
    p.add_argument('--asset-cache')
    a=p.parse_args(argv)
    config=json.loads(Path(a.config).read_text(encoding='utf8'))
    bank_path=Path(a.bank).resolve();bank=load_configurable_bank(bank_path)
    validate_config(config,bank)
    root=Path(a.output_root).resolve()
    if a.resume:
        plan=load_plan(root/'plan.json',bank_path)
        if config!=plan['configuration']:raise ValueError('resume configuration differs')
    else:
        root.mkdir(parents=True,exist_ok=False)
        plan=dict(configuration=config,experiment=experiment_identity(bank_path,config),bank=str(bank_path),
            preflight_updates_per_new_layout=0,execution_protocol='direct_stage_v2',total_updates_are_absolute=True)
        write_new(root/'plan.json',plan)
        (root/'commits').mkdir();(root/'attempts').mkdir()
    if not a.execute:
        print(json.dumps(dict(output=str(root),num_envs=config['grid']['num_envs'],full_designs=len(bank['designs']),
            active_limit=config['rotation']['active_designs'],first_stage=stage_spec(bank,config,0),simulation_started=False),indent=2));return
    with controller_lock(root):
        if (root/'USER_PAUSED').exists():raise RuntimeError('USER_PAUSED present; no automatic restart')
        commits=committed(root,plan,bank)
        target=config['rotation']['total_updates'];index=len(commits)
        if commits and json.loads(commits[-1].read_text())['stage']['end_update']==target:
            last=json.loads(commits[-1].read_text())
            validate_checkpoint(last['checkpoint'],plan['experiment'],target)
            if not (root/'completed.json').exists():
                write_new(root/'completed.json',dict(completed_updates=target,checkpoint=last['checkpoint'],requires_gait_evaluation=True))
            print('TRAINING_ALREADY_COMPLETE_NOT_GAIT_VALIDATED');return
        if not commits and any((root/'attempts').glob('*/train/launch.json')):
            raise RuntimeError('initial stage interrupted with no committed checkpoint; inspect manually, no implicit scratch restart')
        from robolab.rough_gpu import query_gpus,candidates,gpu_lease,child_environment,reject_duplicate_bank
        reject_duplicate_bank(bank_path)
        # This isolated controller additionally rejects orphan workers from this new entry.
        import psutil
        for proc in psutil.process_iter(['cmdline']):
            args=proc.info['cmdline'] or []
            if any(Path(x).name=='train_stage.py' for x in args):raise RuntimeError(f'orphan/active hybrid worker {proc.pid}; inspect first')
        choices=candidates(query_gpus(),os.environ,min_free_mib=a.gpu_min_free_gb*1024,max_utilization=a.gpu_max_utilization,requested=a.device)
        with gpu_lease(choices,ROOT/'logs/rpo_rough_multimorph/gpu_locks') as gpu:
            env=child_environment(gpu,os.environ)
            kit=a.kit_args+f" --/renderer/activeGpu={gpu['index']} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0"
            previous=json.loads(commits[-1].read_text()) if commits else None
            while index*config['rotation']['updates_per_stage']<target:
                if (root/'USER_PAUSED').exists():raise RuntimeError('paused before next stage')
                stage=stage_spec(bank,config,index)
                if previous:validate_checkpoint(previous['checkpoint'],plan['experiment'],stage['first_update'])
                attempt=root/'attempts'/f"stage_{index:06d}_{datetime.now():%Y%m%d_%H%M%S_%f}"
                attempt.mkdir()
                common=[a.python,'-u',str(HERE/'train_stage.py'),'--bank',str(bank_path),'--hybrid-plan',str(root/'plan.json'),
                    '--stage-index',str(index),'--headless','--device','cuda:0','--kit_args',kit]
                cache=a.asset_cache
                if cache:
                    common+=['--asset-cache',str(Path(cache).resolve())]
                else:
                    # Reuse only exact native hashes from prior groups in this
                    # experiment. New designs convert normally on first use.
                    common+=['--asset-cache',str(root/'attempts'),'--allow-cache-misses']
                print(f'GROUP {index+1}: {len(stage["design_ids"])} designs; '
                      f'global updates {stage["first_update"]}..{stage["end_update"]-1}; '
                      f'target={target}; resume={previous["checkpoint"] if previous else "fresh initial policy"}',flush=True)
                command=common+['--output-dir',str(attempt/'train/result'),'--execute','--updates',str(stage['end_update'])]
                if previous:command+=['--resume-checkpoint',previous['checkpoint']]
                run_child(command,attempt/'train',env)
                summary_path=attempt/'train/result/summary.json';summary=json.loads(summary_path.read_text())
                if summary['result']!='HYBRID_STAGE_COMPLETE_NOT_GAIT_VALIDATED' or summary['completed_updates']!=stage['end_update']:raise ValueError('stage did not complete exact target')
                checkpoint=attempt/f"train/result/models/model_{stage['end_update']-1}.pt"
                validate_checkpoint(checkpoint,plan['experiment'],stage['end_update'])
                previous=dict(stage=stage,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),summary=str(summary_path))
                write_new(root/'commits'/f'stage_{index:06d}.json',previous)
                print('STAGE_COMMITTED',index,stage['end_update'],'/',target,flush=True);index+=1
            write_new(root/'completed.json',dict(completed_updates=target,checkpoint=previous['checkpoint'],requires_gait_evaluation=True))

if __name__=='__main__':main()
