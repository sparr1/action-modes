"""Compare one and two physical CPU cores on one allocated GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-id',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--metadata',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--expected-sha',required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this benchmark inside a Slurm GPU allocation.')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=args.expected_sha:
        raise RuntimeError('Unexpected source commit')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise RuntimeError('Benchmark source must be clean')
    available=sorted(os.sched_getaffinity(0))
    if len(available)<2:
        raise RuntimeError('Allocate two physical CPU cores for the scaling benchmark')
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for count in (1,2):
        directory=args.output/f'cpu{count}'
        command=[sys.executable,str(repo/'evaluate_ambi_entropy_probe.py'),
                 '--checkpoint-id',args.checkpoint_id,'--checkpoint',args.checkpoint,
                 '--metadata',args.metadata,'--environment-seed','101',
                 '--output',str(directory),'--device','cuda',
                 '--torch-threads',str(count),'--decisions','0','10']
        env=dict(os.environ,OMP_NUM_THREADS=str(count),MKL_NUM_THREADS=str(count),
                 OPENBLAS_NUM_THREADS=str(count),NUMEXPR_NUM_THREADS=str(count),
                 WANDB_MODE='disabled',MUJOCO_GL='disable')
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        started=time.perf_counter()
        def bind():
            os.sched_setaffinity(0,available[:count])
        with (args.output/f'cpu{count}.log').open('x') as stream:
            result=subprocess.run(command,cwd=repo,env=env,preexec_fn=bind,
                                  stdout=stream,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter()-started
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        row=dict(cpus=count,affinity=available[:count],seconds=elapsed,
                 child_cpu_seconds=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime),
                 peak_child_rss_kib=after.ru_maxrss,exit_code=result.returncode,
                 output=str(directory))
        # ru_maxrss is cumulative across the sequential child processes; it is
        # a conservative allocation bound, not a per-case isolated RSS estimate.
        records.append(row)
        (args.output/'benchmark.json').write_text(json.dumps(dict(
            source_commit=args.expected_sha,job_id=os.environ['SLURM_JOB_ID'],
            checkpoint_id=args.checkpoint_id,records=records,
            rss_scope='cumulative child maximum on Linux'),indent=2))
        print(json.dumps(row),flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
