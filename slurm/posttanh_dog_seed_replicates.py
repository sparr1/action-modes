"""Replicate the validated sampled Dog recipes with seeds 56 and 57."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE_COMMIT = '5e792d64cf6b27233251f1060f6e79036b258714'
REFERENCE_ROOT = Path('/oscar/scratch/rgao48/ambi/posttanh-dog-20261111')
TASKS = ('walk', 'trot', 'run')
PROTECTED = ['main.py', 'RL', 'utils', 'domains', 'environments',
             'slurm/ambi_transfer_training.py', 'slurm/inner_entropy_oscar.py',
             'tests/test_inner_entropy_graph.py', 'tests/test_posttanh_dog.py']


def manifest_path(task, replicated=True):
    suffix = '_seeds56_57' if replicated else ''
    return ROOT / f'configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_dog_{task}{suffix}.json'


def validate_recipe(task, index):
    from slurm.ambi_transfer_training import recipe
    assert task in TASKS and index in (0, 1)
    _, original_manifest, original = recipe(0, manifest_path(task, False))
    name, manifest, run = recipe(index, manifest_path(task))
    assert run['seed'] == 56 + index
    comparable = deepcopy(run)
    comparable['seed'], comparable['name'] = original['seed'], original['name']
    for key in ('wandb_run_name', 'wandb_tags', 'wandb_group'):
        comparable['alg_params'][key] = original['alg_params'][key]
    assert comparable == original, 'Replication changed more than seed/publication labels'
    assert manifest['overrides_alg'] == {k:v for k,v in original_manifest['overrides_alg'].items() if k != 'seed'}
    for key in ('env_params', 'trials', 'logs', 'save_trials', 'checkpoint_every', 'save_strat', 'log_info', 'log_type'):
        assert manifest[key] == original_manifest[key], key
    assert manifest['env_params']['task'] == 'dog-' + task
    assert f"seed{run['seed']}" in run['alg_params']['wandb_tags']
    assert 'seed55' not in run['alg_params']['wandb_tags']
    return name, manifest, run


def validate_reference(task):
    """Reuse task-specific evidence only for identical source and recipe hashes."""
    from slurm.ambi_transfer_training import binding
    from slurm.inner_entropy_oscar import parity_cases, validate_global_parity
    assert task in TASKS
    subprocess.run(['git', 'diff', '--exit-code', REFERENCE_COMMIT, '--', *PROTECTED], cwd=ROOT, check=True)
    campaign = REFERENCE_ROOT / task
    receipt_path = campaign / 'gates/0/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    expected = binding(manifest_path(task, False)); expected['commit'] = REFERENCE_COMMIT
    assert receipt['passed'] and receipt['binding'] == expected
    assert receipt['graph_native_graph_continuation'] and receipt['action_repeat'] == 2
    assert (receipt['observation_dim'], receipt['action_dim'], receipt['decision_horizon']) == (223, 38, 500)
    parity = json.loads((campaign / 'gpu-parity.json').read_text())
    validate_global_parity(parity, expected, receipt['runtime'])
    junit = Path(parity['junit_path']); assert junit.is_relative_to(campaign)
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == parity['junit_sha256']
    assert parity_cases(junit) == parity['case_names']
    dog_junit = Path(receipt['dog_cuda_parity_junit']); assert dog_junit.is_relative_to(campaign)
    assert hashlib.sha256(dog_junit.read_bytes()).hexdigest() == receipt['dog_cuda_parity_sha256']
    cases = list(ET.parse(dog_junit).getroot().iter('testcase'))
    assert len(cases) == 1 and not any(cases[0].find(tag) is not None for tag in ('skipped', 'failure', 'error'))
    for filename in (*receipt['cases'], receipt['native_reference']):
        case = json.loads(Path(filename).read_text())
        assert case['passed'] and case['binding'] == expected and case['runtime'] == receipt['runtime']
    return receipt_path, receipt


def launch(task, index):
    import torch
    from slurm.ambi_transfer_training import binding, write_new
    from slurm.inner_entropy_oscar import accelerated_runtime, checkpoint_space
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('WANDB_MODE') == 'disabled'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert commit == os.environ['AMBI_EXPECTED_COMMIT']
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip()
    _, _, run = validate_recipe(task, index)
    receipt_path, receipt = validate_reference(task)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK'])); torch.set_num_interop_threads(1)
    current = accelerated_runtime()
    assert current == receipt['runtime'], (current, receipt['runtime'])
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == receipt['plan']['cpus']
    campaign = Path(os.environ['AMBI_ENTROPY_CAMPAIGN'])
    state = json.loads((campaign / 'submission.json').read_text()); spec = state['runs'][index]
    assert state['task'] == task and spec['seed'] == run['seed']
    assert spec['id'] == os.environ['WANDB_RUN_ID']
    assert str(spec['job']) == os.environ['SLURM_JOB_ID']
    output = campaign / 'production' / f"seed{run['seed']}"
    output.mkdir(parents=True, exist_ok=False)
    space = checkpoint_space(output, 6 * 40 * receipt['plan']['checkpoint_bytes'])
    write_new(output / 'launch.json', {
        'commit': commit, 'reference_commit': REFERENCE_COMMIT, 'task': task, 'seed': run['seed'],
        'binding': binding(manifest_path(task)), 'runtime': current, 'space': space,
        'reference_gate': str(receipt_path), 'original_plan': receipt['plan'],
        'job_id': os.environ['SLURM_JOB_ID'], 'wandb_run_id': spec['id'],
        'training_code_identical_to_reference': True,
    })
    print('REFERENCE_VALIDATED', task, run['seed'], flush=True)
    os.environ['WANDB_MODE'] = 'online'
    sys.argv = ['main.py', '--run', str(manifest_path(task)), '--alg-dir', 'configs/dmcontrol/algs',
                '--log-dir', str(output), '--alg-index', str(index), '--trial-index', '0', '--num-runs', '1']
    runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=TASKS); parser.add_argument('index', type=int, choices=(0, 1))
    args = parser.parse_args(); launch(args.task, args.index)
