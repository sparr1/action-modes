"""Measure and launch the pinned native TD-MPC2 Humanoid Run baseline."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MANIFEST=ROOT/'configs/dmcontrol/experiments/tdmpc2_humanoid_run_state_checkpoint_bank_1m.json'
CAMPAIGN=Path('/oscar/scratch/rgao48/ambi/tdmpc2-humanoid-run-20261108')
SOURCE_COMMIT='9002963d20880d8ed354f663040c52daadb3136f'

def read(path): return json.loads(Path(path).read_text())

def write_new(path,data):
    with Path(path).open('x') as stream: json.dump(data,stream,indent=2,allow_nan=False)

def recipe():
    manifest=read(MANIFEST);name=manifest['configs'][0]
    run={'name':name,**read(ROOT/'configs/dmcontrol/algs'/f'{name}.json'),**manifest['overrides_alg']}
    return name,manifest,run

def binding():
    name,_,_=recipe()
    paths=[MANIFEST,ROOT/'configs/dmcontrol/algs'/f'{name}.json',ROOT/'environments/dmcontrol/uv.lock']
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}

def runtime():
    import torch
    return dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),executable=sys.executable,
                float32_matmul_precision=torch.get_float32_matmul_precision())

def verify_recipe():
    _,manifest,run=recipe()
    source=read(ROOT/'configs/dmcontrol/algs/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m.json')
    source.update(seed=55,total_steps=1000000)
    target={k:v for k,v in run.items() if k!='name'}
    for c in (source,target): c['alg_params']={k:v for k,v in c['alg_params'].items() if not k.startswith('wandb')}
    assert target==source
    assert manifest['env_params']==dict(task='humanoid-run',obs='state',render_mode=None)
    assert manifest['trials']==1 and manifest['checkpoint_every']==25000 and manifest['save_strat']==['all']

def space(bank_bytes):
    rows=[line.split() for line in subprocess.check_output(['checkquota'],text=True).splitlines()]
    row=next(r for r in rows if len(r)>5 and r[0]=='rgao48' and r[1]=='/oscar/scratch')
    available=min(int((float(row[4])-float(row[2]))*2**30),shutil.disk_usage(CAMPAIGN).free)
    required=bank_bytes*1.3+2*2**30
    assert available>=required,(available,required)
    return dict(available_bytes=available,checkpoint_bank_bytes=bank_bytes,required_bytes=required)

def prepare():
    verify_recipe()
    assert Path(os.environ['AMBI_TDMPC2_CAMPAIGN'])==CAMPAIGN
    subprocess.run(['git','diff','--exit-code',SOURCE_COMMIT,'HEAD','--','RL','domains','utils','environments','main.py'],cwd=ROOT,check=True)
    write_new(CAMPAIGN/'preflight.json',dict(binding=binding(),source_commit=SOURCE_COMMIT,
        unchanged_learner=True,space=space(40*100*2**20)))
    print('PREPARED',json.dumps(binding()),flush=True)

def worker(args):
    assert os.environ['WANDB_MODE']=='disabled' and os.environ['SLURM_JOB_ID']
    cpus=[int(c) for c in args.cpus.split(',')]
    assert set(cpus)<=os.sched_getaffinity(0)
    os.sched_setaffinity(0,cpus)
    import gymnasium as gym
    import torch
    import domains
    from RL.TDMPC2 import TDMPC2Baseline
    torch.set_num_threads(len(cpus));torch.set_num_interop_threads(1)
    name,manifest,run=recipe();manifest=deepcopy(manifest);run=deepcopy(run)
    # Preserve all model, planner and update shapes; shorten seed/pretraining only.
    overrides=dict(wandb=False,wandb_mode='disabled',seed_steps=500,pretrain_steps=2)
    run['alg_params'].update(overrides)
    first,total=534,789
    run.update(total_steps=total,checkpoint_every=None,save_strat='none')
    manifest.update(checkpoint_every=None,save_strat='none',logs='none',save_trials='none')
    env=gym.make(run['env'],**manifest['env_params'])
    learner=TDMPC2Baseline(name+'-disabled-sizing',env,run['alg_params'],run,manifest)
    cfg=learner.cfg
    assert cfg.mpc and not cfg.compile and cfg.action_dim==21 and cfg.model_size==5
    assert (cfg.train_unroll_horizon,cfg.outer_planning_horizon,cfg.iterations)==(3,3,8)
    assert (cfg.num_samples,cfg.num_elites,cfg.num_pi_trajs)==(512,64,24)
    assert cfg.eval_freq is None and cfg.batch_size==256
    timing={};act,log=learner._act_agent,learner._log_wandb_step
    with tempfile.TemporaryDirectory(prefix='tdmpc2-disabled-smoke-') as temp:
        def timed_act(*pos,**kw):
            step=learner._global_step+1
            if step==first:
                torch.cuda.synchronize();timing.update(start=time.perf_counter(),cpu=time.process_time(),updates=learner._num_updates)
            action=act(*pos,**kw)
            assert torch.isfinite(action).all()
            return action
        def timed_log(*pos,**kw):
            result=log(*pos,**kw)
            if learner._global_step==total and 'seconds' not in timing:
                torch.cuda.synchronize();timing['seconds']=time.perf_counter()-timing['start']
                timing['cpu_seconds']=time.process_time()-timing['cpu']
                timing['updates']=learner._num_updates-timing['updates']
                timing['checkpoint']=learner.save(temp,'checkpoint.pt');learner.flush_checkpoints()
            return result
        learner._act_agent,learner._log_wandb_step=timed_act,timed_log
        try:
            learner.learn(total)
            assert learner._global_step==total and timing['updates']==total-first+1
            assert all(math.isfinite(float(v)) for v in learner._last_train_metrics.values())
            assert all(torch.isfinite(p).all() for p in learner.agent.model.parameters())
            checkpoint=Path(timing['checkpoint'])
            assert checkpoint.is_file() and Path(str(checkpoint)+'.metadata.json').is_file()
            weights={k:v.detach().clone() for k,v in learner.agent.model.state_dict().items()}
            learner.agent.load(checkpoint)
            for k,v in weights.items(): torch.testing.assert_close(v,learner.agent.model.state_dict()[k],rtol=0,atol=0)
            report=dict(passed=True,binding=binding(),runtime=runtime(),cpus=len(cpus),
                decisions=total-first+1,seconds_per_decision=timing['seconds']/(total-first+1),
                average_cpu_cores_used=timing['cpu_seconds']/timing['seconds'],
                peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
                checkpoint_bytes=checkpoint.stat().st_size,checkpoint_roundtrip=True,
                benchmark_overrides=overrides,wandb=False)
            write_new(args.output,report)
        finally: env.close()

def select_plan(rows):
    fastest=min(rows,key=lambda r:r['seconds_per_decision']);one=next(r for r in rows if r['cpus']==1)
    chosen=one if one['seconds_per_decision']<=fastest['seconds_per_decision']*1.05 else fastest
    return dict(cpus=chosen['cpus'],seconds_per_decision=chosen['seconds_per_decision'],
                wall_hours=math.ceil(chosen['seconds_per_decision']*1000000*1.15/3600+.5),
                memory_gib=max(8,math.ceil(max(r['peak_rss_kib'] for r in rows)/2**20*1.25+1)))

def gate():
    assert os.environ['WANDB_MODE']=='disabled' and os.environ['SLURM_JOB_ID']
    assert read(CAMPAIGN/'preflight.json')['binding']==binding()
    cores=sorted(os.sched_getaffinity(0));assert len(cores)>=2
    directory=CAMPAIGN/'gate';directory.mkdir()
    rows=[]
    for n in (1,2):
        output=directory/f'cpu-{n}.json'
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(n)
        with output.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable,__file__,'worker','--cpus',','.join(map(str,cores[:n])),'--output',str(output)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows.append(read(output));assert rows[-1]['passed'] and rows[-1]['binding']==binding()
    assert rows[0]['runtime']==rows[1]['runtime']
    plan=select_plan(rows);quota=space(40*max(r['checkpoint_bytes'] for r in rows))
    write_new(directory/'receipt.json',dict(passed=True,binding=binding(),runtime=rows[0]['runtime'],plan=plan,space=quota,cases=rows))
    print('GATE_PASSED',json.dumps(plan),flush=True)

def train():
    verify_recipe()
    assert Path(os.environ['AMBI_TDMPC2_CAMPAIGN'])==CAMPAIGN
    receipt=read(CAMPAIGN/'gate/receipt.json');state=read(CAMPAIGN/'submission.json');arm=state['runs'][0]
    assert receipt['passed'] and receipt['binding']==binding() and receipt['runtime']==runtime()
    assert int(os.environ['SLURM_CPUS_PER_TASK'])==receipt['plan']['cpus']==arm['cpus']
    assert int(os.environ['SLURM_JOB_ID'])==arm['job']
    assert state['workspace_api_verified'] and state['workspace_browser_verified']
    subprocess.run([os.environ['AMBI_WORKSPACE_PYTHON'],'slurm/tdmpc2_humanoid_run_workspace.py',str(CAMPAIGN/'submission.json')],cwd=ROOT,check=True)
    output=CAMPAIGN/'production/0';output.mkdir()
    write_new(output/'launch.json',dict(binding=binding(),runtime=runtime(),plan=receipt['plan'],job=arm['job'],run_id=arm['id'],space=space(receipt['space']['checkpoint_bank_bytes'])))
    import torch
    torch.set_num_threads(arm['cpus']);torch.set_num_interop_threads(1)
    os.environ.update(WANDB_MODE='online',WANDB_RUN_ID=arm['id'])
    sys.argv=['main.py','--run',str(MANIFEST),'--alg-dir','configs/dmcontrol/algs','--log-dir',str(output),'--alg-index','0','--trial-index','0','--num-runs','1']
    runpy.run_path(str(ROOT/'main.py'),run_name='__main__')

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','worker','gate','train'));p.add_argument('--cpus');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a)
    else:globals()[a.mode]()
