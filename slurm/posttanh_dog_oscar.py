"""Task-matched Dog gates and measured, automatic Oscar training launches."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK = os.environ.get('AMBI_DOG_TASK', 'run')
if TASK not in ('run','walk','trot'):
    raise ValueError('AMBI_DOG_TASK must be run, walk, or trot')
os.environ['AMBI_ENTROPY_VARIANT']='dog'
os.environ['AMBI_DOG_TASK']=TASK
from slurm import inner_entropy_oscar as base

CAMPAIGN = Path('/oscar/scratch/rgao48/ambi/posttanh-dog-20261111')
OUTPUT = CAMPAIGN / TASK
SOURCE_COMMIT='5f6089fe70441b968ed814d1b1e73ddbd3f08a71'
SOURCE_ROOT=Path('/oscar/scratch/rgao48/ambi/inner-entropy-cudagraph-20261102')
DOG_TEST='tests/test_posttanh_dog.py::test_cuda_dog_capture_matches_uncaptured'


def read(path):
    return json.loads(Path(path).read_text())


def write_state(state):
    temporary=CAMPAIGN/f'submission.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    temporary.replace(CAMPAIGN/'submission.json')


@contextmanager
def locked_state():
    with (CAMPAIGN/'state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield read(CAMPAIGN/'submission.json')


def arm(state):
    return next(row for row in state['runs'] if row['task']==TASK)


def phase(value):
    with locked_state() as state:
        arm(state)['phase']=value
        write_state(state)


def publish():
    subprocess.run(['flock',str(CAMPAIGN/'workspace.lock'),os.environ['AMBI_WORKSPACE_PYTHON'],
                    'slurm/posttanh_dog_workspace.py',str(CAMPAIGN/'submission.json')],cwd=ROOT,check=True)


def verify_recipe():
    source=read(ROOT/'configs/dmcontrol/algs/ambi_inner_entropy_posttanh_outer_adapt.json')
    target=read(ROOT/f'configs/dmcontrol/algs/ambi_posttanh_outer_adapt_dog_{TASK}.json')
    for value in (source,target):
        value['alg_params']={k:v for k,v in value['alg_params'].items() if not k.startswith('wandb')}
    assert source==target
    original=read(ROOT/'configs/dmcontrol/experiments/ambi_inner_entropy.json')
    original['env_params']['task']='dog-'+TASK
    manifest=read(base.MANIFEST)
    for key in ('overrides_alg','env_params','trials','logs','save_trials','checkpoint_every','save_strat','log_info','log_type'):
        assert original[key]==manifest[key],key
    assert manifest['configs']==['ambi_posttanh_outer_adapt_dog_'+TASK]
    assert base.CAMPAIGN_ROOT==OUTPUT==Path(os.environ['AMBI_ENTROPY_CAMPAIGN'])


def prepare():
    verify_recipe()
    protected=['RL','domains','utils','environments','main.py',base.GPU_PARITY_TEST.split('::')[0]]
    subprocess.run(['git','diff','--exit-code',SOURCE_COMMIT,'HEAD','--',*protected],cwd=ROOT,check=True)
    old=read(SOURCE_ROOT/'gpu-parity.json')
    assert old['binding']['commit']==SOURCE_COMMIT
    base.validate_global_parity(old,old['binding'],old['runtime'])
    for relative,expected in [(old['binding']['manifest'],old['binding']['manifest_sha256']),
                              ('environments/dmcontrol/uv.lock',old['binding']['lock_sha256']),
                              *[(f'configs/dmcontrol/algs/{name}.json',digest) for name,digest in old['binding']['config_sha256'].items()]]:
        content=subprocess.check_output(['git','show',f'{SOURCE_COMMIT}:{relative}'],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest()==expected
    junit=Path(old['junit_path'])
    assert hashlib.sha256(junit.read_bytes()).hexdigest()==old['junit_sha256']
    assert base.parity_cases(junit)==old['case_names']
    copied=OUTPUT/'reused-gpu-parity.xml'
    with copied.open('xb') as stream:stream.write(junit.read_bytes())
    evidence={'original_commit':SOURCE_COMMIT,'protected_source_paths':protected,
              'original_receipt':str(SOURCE_ROOT/'gpu-parity.json'),
              'additional_gate':'Fresh 223-observation/38-action CUDA parity; each Dog task gets full-size 1/2-core and native checks.'}
    base.write_new(OUTPUT/'gpu-parity.json',{**old,'binding':base.binding(base.MANIFEST),
        'original_binding':old['binding'],'junit_path':str(copied),'source_evidence':evidence})
    base.write_new(OUTPUT/'preflight.json',{'binding':base.binding(base.MANIFEST),'source_evidence':evidence})
    print('PREPARED',TASK,flush=True)


def select_plan(rows):
    plan=base.select_plan(rows)
    plan['wall_hours']=math.ceil(plan['seconds_per_decision']*1000000*1.15/3600+.5)
    plan['memory_gib']=max(8,math.ceil(max(r['peak_host_memory_bytes'] for r in rows)*1.25/2**30))
    plan['timing_margin']=.15
    plan['startup_allowance_hours']=.5
    return plan


def tres(text,key):
    match=re.search(r'(?:^|,)'+re.escape(key)+r'=([0-9.]+)([KMGT]?)',text)
    if not match:return 0
    value=float(match.group(1))
    if key=='mem':value*=dict(K=1/2**20,M=1/1024,G=1,T=1024,**{'':1/1024})[match.group(2)]
    return value


def check_headroom(jobs,qos,plan):
    _,counts,minutes,*_=qos.split('|')
    active=[j for j in jobs if j['qos']=='pri-gpu+']
    usage={'cpu':plan['cpus'],'gres/gpu':1,'mem':plan['memory_gib']}
    reserved={'cpu':plan['cpus']*plan['wall_hours']*60,'gres/gpu':plan['wall_hours']*60}
    for job in active:
        for key in usage:usage[key]+=tres(job['tres_req_str'],key)
        for key in reserved:reserved[key]+=tres(job['tres_req_str'],key)*job['time_limit']['number']
    for key,value in usage.items():assert value<=tres(counts,key),(key,value,counts)
    for key,value in reserved.items():assert value<=tres(minutes,key),(key,value,minutes)
    return {'qos':qos,'combined_counts':usage,'conservative_reserved_minutes':reserved}


def launch_production(plan):
    with locked_state() as state:
        row=arm(state)
        assert row['gate_job']==int(os.environ['SLURM_JOB_ID'])
        assert not row.get('training_job'), 'Refusing duplicate training submission'
        jobs=json.loads(subprocess.check_output(['squeue','--json','-u','rgao48'],text=True))['jobs']
        qos=subprocess.check_output(['sacctmgr','-nP','show','qos','pri-gpu+','format=Name,MaxTRESPU,MaxTRESRunMinsPU'],text=True).strip()
        row['resources']=check_headroom(jobs,qos,plan)
        exports=','.join(['ALL','AMBI_DOG_TASK='+TASK,'AMBI_ENTROPY_CAMPAIGN='+str(OUTPUT),
            'AMBI_EXPECTED_COMMIT='+state['commit'],'AMBI_DMCONTROL_PYTHON='+sys.executable,
            'AMBI_WORKSPACE_PYTHON='+os.environ['AMBI_WORKSPACE_PYTHON']])
        command=['sbatch','--hold','--parsable','--chdir='+str(ROOT),'--job-name=posttanh-dog-'+TASK,
            '--cpus-per-task='+str(plan['cpus']),'--mem='+str(plan['memory_gib'])+'G',
            '--time='+str(plan['wall_hours'])+':00:00','--output='+str(OUTPUT/'slurm/train-%j.log'),
            '--export='+exports,'slurm/run_posttanh_dog_oscar.sbatch','train']
        job=int(subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0])
        row.update(training_job=job,job=job,plan=plan,phase='validation passed; training queued')
        write_state(state)
    # A failed panel setup leaves this new job held and raises visibly.
    publish()
    base.resize_pending(job,plan)
    print('DOG_TRAINING_SUBMITTED',TASK,job,json.dumps(plan),flush=True)


def gate():
    verify_recipe()
    assert os.environ['WANDB_MODE']=='disabled' and os.environ['SLURM_JOB_ID']
    assert int(os.environ['SLURM_CPUS_PER_TASK'])==2
    assert arm(read(CAMPAIGN/'submission.json'))['gate_job']==int(os.environ['SLURM_JOB_ID'])
    phase('running GPU validation and CPU sizing');publish()
    output=OUTPUT/'gates/0';output.mkdir(parents=True,exist_ok=False)
    current=base.accelerated_runtime()
    parity=base.load_global_parity(OUTPUT,current)
    junit=output/'dog-parity.xml'
    program="import sys,torch,pytest;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('high');raise SystemExit(pytest.main(sys.argv[1:]))"
    with (output/'dog-parity.log').open('x') as log:
        subprocess.run([sys.executable,'-c',program,DOG_TEST,'-q','--junitxml='+str(junit)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=list(ET.parse(junit).getroot().iter('testcase'))
    assert len(cases)==1 and not any(cases[0].find(tag) is not None for tag in ('skipped','failure','error'))
    cores=base.physical_core_cpus(os.sched_getaffinity(0));assert len(cores)>=2
    rows=[]
    for count,native in ((1,False),(2,False),(1,True)):
        path=output/(('native-' if native else '')+f'cpu-{count}.json')
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(count)
        command=[sys.executable,'slurm/inner_entropy_oscar.py','worker','--index','0',
                 '--worker-cpus',','.join(map(str,cores[:count])),'--output',str(path)]
        if native:command.append('--native-reference')
        with path.with_suffix('.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=read(path)
        assert report['passed'] and report['runtime']==current and report['binding']==base.binding(base.MANIFEST)
        if not native:
            assert report['graph_native_graph_continuation'] and report['gpu_replay']
            rows.append(report)
    plan=select_plan(rows)
    space=base.checkpoint_space(output,3*40*max(r['checkpoint_bytes'] for r in rows))
    receipt={'passed':True,'binding':base.binding(base.MANIFEST),'runtime':current,'plan':plan,'space':space,
             'cases':[str(output/f'cpu-{n}.json') for n in (1,2)],'native_reference':str(output/'native-cpu-1.json'),
             'dog_cuda_parity_junit':str(junit),'dog_cuda_parity_sha256':hashlib.sha256(junit.read_bytes()).hexdigest(),
             'graph_native_graph_continuation':True,'source_evidence':parity['source_evidence'],
             'scope':'Actual Dog task; production model and update dose; 128 graph decisions on 1/2 CPUs, 32 native decisions, checkpoint/geometry/continuation checks. W&B disabled.',
             'action_repeat':2,'decision_horizon':500,'observation_dim':223,'action_dim':38}
    base.write_new(output/'receipt.json',receipt)
    print('DOG_GATE_PASSED',TASK,json.dumps(plan),flush=True)
    launch_production(plan)


def train():
    verify_recipe()
    row=arm(read(CAMPAIGN/'submission.json'))
    assert row['training_job']==int(os.environ['SLURM_JOB_ID'])
    receipt=read(OUTPUT/'gates/0/receipt.json')
    assert receipt['passed'] and receipt['binding']==base.binding(base.MANIFEST)
    junit=Path(receipt['dog_cuda_parity_junit'])
    assert hashlib.sha256(junit.read_bytes()).hexdigest()==receipt['dog_cuda_parity_sha256']
    phase('training starting');publish()
    os.environ.update(WANDB_MODE='online',WANDB_RUN_ID=row['id'])
    base.train(SimpleNamespace(index=0,gate=OUTPUT/'gates/0/receipt.json',output=OUTPUT/'production/0'))
    phase('training finished');publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=('prepare','gate','train','publish'))
    mode=parser.parse_args().mode
    try:globals()[mode]()
    except BaseException:
        if mode in ('gate','train'):
            phase(mode+' failed; inspect Slurm log')
            publish()
        raise
