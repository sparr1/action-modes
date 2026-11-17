"""Validate and train the unchanged sampled gSDE8 recipe on Hydra Dog Walk."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import gsde_noise_scopes_oscar as base

MANIFEST = ROOT / 'configs/dmcontrol/experiments/ambi_gsde8_dog_walk.json'
CAMPAIGN = Path(os.environ.get('AMBI_GSDE_CAMPAIGN', '/ltmp/rgao48/ambi/gsde8-dog-walk-hydra-20261116'))
base.MANIFEST, base.CAMPAIGN_ROOT = MANIFEST, CAMPAIGN


def space(root, bank_bytes):
    root = Path(root).resolve()
    root.relative_to(Path('/ltmp/rgao48'))
    available = shutil.disk_usage(root).free
    required = int(bank_bytes * 1.3 + 2 * 2**30)
    assert available >= required, (available, required)
    return {'filesystem_free_bytes': available, 'checkpoint_bank_bytes': bank_bytes,
            'required_with_margin_bytes': required, 'storage': 'gpu2501 node-local scratch'}


base.checkpoint_space = space


def gate(args):
    assert os.environ['WANDB_MODE'] == 'disabled' and os.environ['SLURM_JOB_ID']
    assert os.uname().nodename.split('.')[0] == 'gpu2501'
    assert base.recipe(0, MANIFEST)[1]['env_params']['task'] == 'dog-walk'
    cores = base.physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    base.run_gpu_parity(args.output, CAMPAIGN)
    junit = args.output / 'dog-parity.xml'
    program = ('import sys,torch,pytest; torch.set_num_threads(1); '
               "torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('high'); "
               'raise SystemExit(pytest.main(sys.argv[1:]))')
    with (args.output / 'dog-parity.log').open('x') as log:
        subprocess.run([sys.executable, '-c', program, 'tests/test_gsde8_dog_hydra.py',
                        '-k', 'cuda_dog', '-q', '--junitxml=' + str(junit)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    assert len(cases) == 1 and not any(cases[0].find(t) is not None for t in ('skipped', 'error', 'failure'))
    rows = []
    for count in (1, 2):
        output = args.output / f'cpu-{count}.json'
        env = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = str(count)
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--index', '0',
                   '--worker-cpus', ','.join(map(str, cores[:count])), '--output', str(output)]
        with output.with_suffix('.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows.append(json.loads(output.read_text()))
        print('TIMING_COMPLETE', count, rows[-1]['measured'], flush=True)
    plan = base.select_plan(rows)
    # Completed matching gSDE8 run: 1M decisions in 179838.70855 seconds.
    # Preserve that end-to-end floor and allow 25% plus one hour for startup/I/O.
    plan['wall_hours'] = math.ceil(max(plan['seconds_per_decision'] * 1000000, 179838.70855) * 1.25 / 3600 + 1)
    plan.update(runtime_margin=0.25, startup_allowance_hours=1,
                completed_reference_seconds=179838.70855)
    receipt = {'passed': True, 'binding': base.binding(MANIFEST), 'runtime': rows[0]['runtime'],
               'plan': plan, 'space': space(args.output, 40 * plan['checkpoint_bytes']),
               'cases': [str(args.output / f'cpu-{n}.json') for n in (1, 2)],
               'global_gpu_parity_required': True, 'dog_gpu_parity': str(junit),
               'scope': 'Seven CUDA gSDE tests; Dog 223-observation/38-action graph parity; full-size Dog Walk 1/2-core training, GPU replay, checkpoint roundtrip and critic geometry; W&B disabled.'}
    base.write_new(args.output / 'receipt.json', receipt)
    print('GATE_PASSED', json.dumps(plan), flush=True)


def train(args):
    receipt = json.loads(args.gate.read_text())
    assert receipt['dog_gpu_parity'] and os.uname().nodename.split('.')[0] == 'gpu2501'
    fields = base.slurm_fields(os.environ['SLURM_JOB_ID'])
    limit = fields['TimeLimit']
    days, clock = limit.split('-') if '-' in limit else ('0', limit)
    hours, minutes, seconds = map(int, clock.split(':'))
    assert int(days) * 24 + hours + minutes / 60 + seconds / 3600 >= receipt['plan']['wall_hours']
    base.train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('gate', 'worker', 'train'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--index', type=int, default=0, choices=(0,))
    parser.add_argument('--worker-cpus')
    parser.add_argument('--native-reference', action='store_true')
    args = parser.parse_args()
    {'gate': gate, 'worker': base.worker, 'train': train}[args.mode](args)
