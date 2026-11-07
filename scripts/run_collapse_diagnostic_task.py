"""Run one manifest-owned diagnostic task; all outputs are immutable per attempt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--stage',choices=['bank','smoke','checkpoint'],required=True)
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    index=args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    manifest=json.loads(args.manifest.read_text())
    source=Path(__file__).resolve().parents[1]
    os.chdir(source)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
    if commit!=manifest['evaluator_commit'] or dirty:
        raise RuntimeError('Evaluator checkout does not match the clean pinned manifest commit.')
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',MUJOCO_GL='egl',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1')
    root=Path(manifest['output_root'])
    bank=Path(manifest.get('bank_path',root/'bank'/'bank.json'))
    task=(dict(key='shared-bank',checkpoint=manifest['bank_checkpoint']) if args.stage=='bank'
          else (manifest['smoke_tasks'] if args.stage=='smoke' else manifest['checkpoints'])[index])
    output=root/'bank-status' if args.stage=='bank' else root/args.stage/task['key']
    output.mkdir(parents=True,exist_ok=True)
    def run(mode,checkpoint,destination,*extra):
        command=[sys.executable,str(source/'diagnose_policy_collapse.py'),mode,
            '--checkpoint',checkpoint,'--output',str(destination),'--torch-threads','1',*map(str,extra)]
        print(json.dumps(dict(command=command)),flush=True)
        subprocess.run(command,check=True)
        name='bank.json' if mode=='bank' else 'result.json'
        result=json.loads((destination/name).read_text())
        if mode!='bank' and not result.get('complete'):raise RuntimeError('Incomplete result.')
    if args.stage=='bank':
        run('bank',manifest['bank_checkpoint'],root/'bank','--device','cpu')
    elif args.stage=='smoke':
        run('parity',task['checkpoint'],output/'parity')
        run('episodes',task['checkpoint'],output/'episodes','--seeds',101,'--steps',50,
            '--arms','prior_mean','prior_sample')
        if not bank.is_file():raise RuntimeError('Shared bank not ready after parity gate; retry with a fresh output root.')
        run('gradients',task['checkpoint'],output/'gradients','--bank',bank,'--roots',2,'--repeats',1)
    else:
        if not bank.is_file():raise RuntimeError('Shared bank is missing.')
        run('episodes',task['checkpoint'],output/'episodes')
        run('gradients',task['checkpoint'],output/'gradients','--bank',bank)
    completion=dict(complete=True,stage=args.stage,index=index,key=task['key'],commit=commit,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    temporary=output/'complete.json.tmp'
    temporary.write_text(json.dumps(completion,indent=2)+'\n')
    temporary.replace(output/'complete.json')
    print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
