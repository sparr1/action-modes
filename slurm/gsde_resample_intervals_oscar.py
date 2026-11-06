"""Validate and train the four gSDE interval arms from a clean Oscar checkout."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import gsde_noise_scopes_oscar as base
from slurm.ambi_transfer_training import binding, recipe, write_new

MANIFEST = ROOT / 'configs/dmcontrol/experiments/ambi_gsde_resample_intervals.json'
CAMPAIGN_ROOT = Path('/oscar/scratch/rgao48/ambi/gsde-resample-intervals-20261106')
REFERENCE_ROOT = Path('/oscar/scratch/rgao48/ambi/gsde-noise-scopes-20261105')
REFERENCE_COMMIT = 'edec58cf7ec2b51bd54834aa4d6f806715075502'
INTERVALS = ((4, 8), (16, 8), (8, 4), (8, 16))
PUBLICATION_KEYS = ('wandb_group', 'wandb_run_name', 'wandb_tags')


def validate_recipe(index):
    name, manifest, run = recipe(index, MANIFEST)
    reference = json.loads((ROOT / 'configs/dmcontrol/algs/ambi_gsde_squashed_real_steps.json').read_text())
    actual = deepcopy({key: value for key, value in run.items() if key != 'name'})
    expected = deepcopy(reference)
    imagined, execution = INTERVALS[index]
    expected['alg_params'].update(inner_gsde_resample_steps=imagined,
                                  gsde_execution_resample_steps=execution)
    for row in (actual, expected):
        for key in PUBLICATION_KEYS:
            row['alg_params'].pop(key)
    assert actual == expected, 'Only one requested interval and publication identity may change.'
    assert manifest['checkpoint_every'] == run['checkpoint_every'] == 25000
    assert manifest['env_params']['task'] == 'humanoid-walk'
    return name, manifest, run


def inherited_cuda_evidence(current):
    """Reuse real CUDA parity only when all learner/runtime/test sources match."""
    unchanged = ['RL', 'utils', 'domains', 'main.py', 'environments/dmcontrol',
                 'tests/test_ambi_gsde_graph.py', 'tests/test_ambi_gsde_lifecycle.py',
                 'tests/test_ambi_inner_decoupling.py', 'tests/test_ambi_root_local_sac.py',
                 'tests/test_ambi_distillation_collection_graph.py']
    subprocess.run(['git', 'diff', '--exit-code', REFERENCE_COMMIT, 'HEAD', '--', *unchanged],
                   cwd=ROOT, check=True)
    receipt_path = REFERENCE_ROOT / 'gpu-parity.json'
    receipt = json.loads(receipt_path.read_text())
    expected = binding(ROOT / 'configs/dmcontrol/experiments/ambi_gsde_noise_scopes.json')
    expected['commit'] = REFERENCE_COMMIT
    base.validate_global_parity(receipt, expected, current)
    junit = Path(receipt['junit_path'])
    assert junit.is_relative_to(REFERENCE_ROOT)
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == receipt['junit_sha256']
    assert base.parity_cases(junit) == receipt['case_names']
    return {'receipt': str(receipt_path), 'source_commit': REFERENCE_COMMIT,
            'junit_sha256': receipt['junit_sha256'], 'unchanged_paths': unchanged,
            'case_count': receipt['case_count'], 'runtime': current}


def worker(args):
    validate_recipe(args.index)
    base.MANIFEST = MANIFEST
    base.CAMPAIGN_ROOT = CAMPAIGN_ROOT
    args.audit_gsde_persistence = True
    args.native_reference = False
    base.worker(args)


def validate(args):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('WANDB_MODE') == 'disabled'
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == 1
    validate_recipe(args.index)
    args.output.mkdir(parents=True, exist_ok=False)
    current = base.accelerated_runtime()
    parity = inherited_cuda_evidence(current)
    cpus = base.physical_core_cpus(os.sched_getaffinity(0))
    assert cpus
    output = args.output / 'cpu-1.json'
    with output.with_suffix('.log').open('x') as stream:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker',
                        '--index', str(args.index), '--worker-cpus', str(cpus[0]),
                        '--output', str(output)], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    report = json.loads(output.read_text())
    assert report['passed'] and report['binding'] == binding(MANIFEST)
    assert report['runtime'] == current and report['noise_lifetimes']['passed']
    assert report['physical_cpus'] == 1 and report['inner_cuda_graph']
    assert not report['compile_fallback'] and report['graph_replays']['timed'] == 96
    reference = json.loads((REFERENCE_ROOT / 'gates/5/receipt.json').read_text())
    seconds = report['measured']['seconds_per_decision']
    assert seconds <= 1.25 * reference['plan']['seconds_per_decision'], 'Unexpected steady-state slowdown.'
    submission = json.loads((CAMPAIGN_ROOT / 'submission.json').read_text())
    plan = submission['runs'][args.index]['plan']
    assert plan['cpus'] == 1 and plan['memory_gib'] >= 8
    assert seconds * 1000000 * 1.10 / 3600 + .5 <= plan['wall_hours'], 'Allocation needs a larger measured walltime.'
    space = base.checkpoint_space(args.output, 4 * 40 * report['checkpoint_bytes'])
    receipt = {'passed': True, 'binding': binding(MANIFEST), 'runtime': current,
               'plan': plan, 'space': space, 'case': str(output), 'inherited_cuda_parity': parity,
               'measured_seconds_per_decision': seconds, 'noise_lifetimes': report['noise_lifetimes'],
               'scope': 'Production architecture/dose; 96 timed CUDA graph decisions, interval boundary audit, geometry and checkpoint round-trip; W&B disabled.',
               'wandb': False, 'job_id': os.environ['SLURM_JOB_ID']}
    write_new(args.output / 'receipt.json', receipt)
    print('INTERVAL_PREFLIGHT_PASSED', args.index, json.dumps(receipt), flush=True)


def train(args):
    import torch
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.set_num_interop_threads(1)
    validate_recipe(args.index)
    receipt = json.loads(args.gate.read_text())
    assert receipt['passed'] and receipt['binding'] == binding(MANIFEST)
    assert receipt['runtime'] == base.accelerated_runtime()
    assert receipt['noise_lifetimes']['passed']
    assert receipt['plan']['index'] == args.index
    assert receipt['plan']['cpus'] == int(os.environ['SLURM_CPUS_PER_TASK'])
    assert os.environ.get('WANDB_MODE') == 'online' and os.environ.get('WANDB_RUN_ID')
    args.output.mkdir(parents=True, exist_ok=False)
    space = base.checkpoint_space(args.output, receipt['space']['checkpoint_bank_bytes'])
    write_new(args.output / 'launch.json', {
        'binding': binding(MANIFEST), 'plan': receipt['plan'], 'runtime': receipt['runtime'],
        'gate': str(args.gate), 'noise_lifetimes': receipt['noise_lifetimes'], 'space': space,
        'job_id': os.environ['SLURM_JOB_ID'], 'wandb_run_id': os.environ['WANDB_RUN_ID']})
    sys.argv = ['main.py', '--run', str(MANIFEST), '--alg-dir', 'configs/dmcontrol/algs',
                '--log-dir', str(args.output), '--alg-index', str(args.index),
                '--trial-index', '0', '--num-runs', '1']
    runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('worker', 'validate', 'train'))
    parser.add_argument('--index', type=int, choices=range(4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--worker-cpus')
    args = parser.parse_args()
    assert Path(os.environ.get('AMBI_GSDE_CAMPAIGN', str(CAMPAIGN_ROOT))) == CAMPAIGN_ROOT
    {'worker': worker, 'validate': validate, 'train': train}[args.mode](args)


if __name__ == '__main__':
    main()
