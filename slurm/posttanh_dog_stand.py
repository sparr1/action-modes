"""Validate the sampled Dog Stand recipe and launch its three training seeds."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import re
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['AMBI_ENTROPY_VARIANT'] = 'dog_stand_seeds'
from slurm import inner_entropy_oscar as base

MANIFEST = ROOT / 'configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_dog_stand_seeds55_56_57.json'
CAMPAIGN = Path('/oscar/scratch/rgao48/ambi/posttanh-dog-stand-20261115')
REFERENCE_COMMIT = '5e792d64cf6b27233251f1060f6e79036b258714'
REFERENCE_ROOT = Path('/oscar/scratch/rgao48/ambi/posttanh-dog-20261111/walk')


def read(path):
    return json.loads(Path(path).read_text())


def validate_recipe(index):
    assert index in (0, 1, 2)
    _, original_manifest, original = base.recipe(0, ROOT / 'configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_dog_walk.json')
    name, manifest, run = base.recipe(index, MANIFEST)
    assert run['seed'] == 55 + index
    comparable = deepcopy(run)
    comparable['seed'], comparable['name'] = original['seed'], original['name']
    for key in ('wandb_run_name', 'wandb_tags', 'wandb_group'):
        comparable['alg_params'][key] = original['alg_params'][key]
    assert comparable == original, 'Only the task, seed and publication labels may change'
    expected = deepcopy(original_manifest)
    expected['env_params']['task'] = 'dog-stand'
    expected['overrides_alg'].pop('seed')
    for key in ('overrides_alg', 'env_params', 'trials', 'logs', 'save_trials',
                'checkpoint_every', 'save_strat', 'log_info', 'log_type'):
        assert manifest[key] == expected[key], key
    assert f"seed{run['seed']}" in run['alg_params']['wandb_tags']
    return name, manifest, run


def prepare():
    for index in range(3):
        validate_recipe(index)
    protected = ['RL', 'domains', 'utils', 'environments', 'main.py',
                 'tests/test_inner_entropy_graph.py', 'tests/test_posttanh_dog.py']
    subprocess.run(['git', 'diff', '--exit-code', REFERENCE_COMMIT, 'HEAD', '--', *protected], cwd=ROOT, check=True)
    old = read(REFERENCE_ROOT / 'gpu-parity.json')
    receipt = read(REFERENCE_ROOT / 'gates/0/receipt.json')
    assert receipt['passed'] and receipt['binding']['commit'] == REFERENCE_COMMIT
    assert (receipt['observation_dim'], receipt['action_dim'], receipt['decision_horizon']) == (223, 38, 500)
    base.validate_global_parity(old, receipt['binding'], receipt['runtime'])
    binding = receipt['binding']
    files = [(binding['manifest'], binding['manifest_sha256']),
             ('environments/dmcontrol/uv.lock', binding['lock_sha256']),
             *[(f'configs/dmcontrol/algs/{name}.json', digest) for name, digest in binding['config_sha256'].items()]]
    for relative, digest in files:
        content = subprocess.check_output(['git', 'show', f'{REFERENCE_COMMIT}:{relative}'], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
    junit = Path(old['junit_path'])
    assert junit.is_relative_to(REFERENCE_ROOT)
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old['junit_sha256']
    assert base.parity_cases(junit) == old['case_names']
    dog_junit = Path(receipt['dog_cuda_parity_junit'])
    assert dog_junit.is_relative_to(REFERENCE_ROOT)
    assert hashlib.sha256(dog_junit.read_bytes()).hexdigest() == receipt['dog_cuda_parity_sha256']
    cases = list(ET.parse(dog_junit).getroot().iter('testcase'))
    assert len(cases) == 1 and not any(cases[0].find(t) is not None for t in ('skipped', 'error', 'failure'))
    copied = CAMPAIGN / 'reused-gpu-parity.xml'
    with copied.open('xb') as stream:
        stream.write(junit.read_bytes())
    evidence = {'original_commit': REFERENCE_COMMIT, 'original_receipt': str(REFERENCE_ROOT / 'gpu-parity.json'),
                'protected_source_paths': protected, 'dog_cuda_parity_junit': str(dog_junit),
                'dog_cuda_parity_sha256': receipt['dog_cuda_parity_sha256'],
                'new_validation': 'Fresh Dog Stand environment parity and 1/2-CPU training, geometry, checkpoint, native/graph checks.'}
    base.write_new(CAMPAIGN / 'gpu-parity.json', {**old, 'binding': base.binding(MANIFEST),
                   'original_binding': old['binding'], 'junit_path': str(copied), 'source_evidence': evidence})
    base.write_new(CAMPAIGN / 'preflight.json', {'binding': base.binding(MANIFEST), 'source_evidence': evidence})
    print('DOG_STAND_PREPARED', flush=True)


def gate():
    assert os.environ['WANDB_MODE'] == 'disabled' and os.environ['SLURM_JOB_ID']
    assert base.MANIFEST == MANIFEST and base.CAMPAIGN_ROOT == CAMPAIGN
    for index in range(3):
        validate_recipe(index)
    env = dict(os.environ, AMBI_RUN_REAL_DMCONTROL_TESTS='1')
    subprocess.run([sys.executable, '-m', 'pytest', '-q',
                    'tests/test_posttanh_dog_stand.py::test_real_dog_stand_two_control_steps'],
                   cwd=ROOT, env=env, check=True)
    output = CAMPAIGN / 'gates/seed55'
    base.gate(SimpleNamespace(index=0, output=output, release_job=None))
    receipt = read(output / 'receipt.json')
    rows = [read(output / f'cpu-{n}.json') for n in (1, 2)]
    assert all(r['passed'] and r['binding'] == base.binding(MANIFEST) and r['runtime'] == receipt['runtime'] for r in rows)
    assert receipt['graph_native_graph_continuation']
    plan = base.select_plan(rows)
    # The fresh task benchmark sizes CPUs; completed Dog runs also bound end-to-end walltime.
    reference_seconds = max(read(CAMPAIGN / 'submission.json')['reference_runtime_seconds'].values())
    estimate = max(plan['seconds_per_decision'] * 1_000_000, reference_seconds)
    plan.update(wall_hours=math.ceil(estimate * 1.15 / 3600 + .5),
                memory_gib=max(8, math.ceil(max(r['peak_host_memory_bytes'] for r in rows) * 1.25 / 2**30)),
                timing_margin=.15, startup_allowance_hours=.5, reference_runtime_seconds=reference_seconds)
    base.write_new(CAMPAIGN / 'gate.json', {**receipt, 'plan': plan, 'validated_seed': 55,
                   'replicated_seeds': [55, 56, 57], 'wandb': False})
    print('DOG_STAND_GATE_PASSED', json.dumps(plan), flush=True)


def set_pending_memory(job, memory_gib):
    """scontrol accepts integer MB here; its readback may include a unit suffix."""
    megabytes = int(memory_gib * 1024)
    assert megabytes > 0 and megabytes == memory_gib * 1024
    def allocated_mb(fields):
        match = re.fullmatch(r'([0-9.]+)([KMGT]?)', fields['MinMemoryNode'])
        assert match, fields
        return float(match.group(1)) * {'': 1, 'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2}[match.group(2)]
    fields = base.slurm_fields(job)
    assert fields['JobState'] == 'PENDING' and fields['Reason'] == 'JobHeldUser'
    if allocated_mb(fields) != megabytes:
        subprocess.run(['scontrol', 'update', f'JobId={job}', f'MinMemoryNode={megabytes}'], check=True)
    verified = base.slurm_fields(job)
    assert verified['JobState'] == 'PENDING' and verified['Reason'] == 'JobHeldUser'
    assert allocated_mb(verified) == megabytes, verified


def release():
    """Apply the measured allocation to the three held, pre-submitted jobs."""
    state = read(CAMPAIGN / 'submission.json')
    receipt = read(CAMPAIGN / 'gate.json')
    assert receipt['passed'] and receipt['binding'] == base.binding(MANIFEST)
    assert str(state['release_job']) == os.environ['SLURM_JOB_ID']
    assert state['commit'] == receipt['binding']['commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip()
    plan = receipt['plan']
    own = {int(r['job']) for r in state['runs']}
    jobs = json.loads(subprocess.check_output(['squeue', '--json', '-u', 'rgao48'], text=True))['jobs']
    qos = subprocess.check_output(['sacctmgr', '-nP', 'show', 'qos', 'pri-gpu+',
                                  'format=Name,MaxTRESPU,MaxTRESRunMinsPU'], text=True).strip()
    def tres(text, key):
        match = re.search(r'(?:^|,)' + re.escape(key) + r'=([0-9.]+)([KMGT]?)', text)
        if not match:
            return 0
        value = float(match.group(1))
        return value * {'': 1/1024, 'K': 1/2**20, 'M': 1/1024, 'G': 1, 'T': 1024}[match.group(2)] if key == 'mem' else value
    counts = {'cpu': 3 * plan['cpus'], 'gres/gpu': 3, 'mem': 3 * plan['memory_gib']}
    minutes = {'cpu': 3 * plan['cpus'] * plan['wall_hours'] * 60, 'gres/gpu': 3 * plan['wall_hours'] * 60}
    for job in jobs:
        if job['qos'] != 'pri-gpu+' or job['job_id'] in own:
            continue
        for key in counts:
            counts[key] += tres(job['tres_req_str'], key)
        for key in minutes:
            minutes[key] += tres(job['tres_req_str'], key) * job['time_limit']['number']
    _, count_limits, minute_limits, *_ = qos.split('|')
    for key, value in counts.items():
        assert value <= tres(count_limits, key), (key, value, count_limits)
    for key, value in minutes.items():
        assert value <= tres(minute_limits, key), (key, value, minute_limits)
    state.update(plan=plan, resource_check={'combined_counts': counts, 'reserved_minutes_upper_bound': minutes, 'qos': qos})
    for row in state['runs']:
        fields = base.slurm_fields(row['job'])
        assert fields['JobState'] == 'PENDING' and fields['Reason'] == 'JobHeldUser'
        assert fields['WorkDir'] == str(ROOT) and fields['UserId'].startswith('rgao48(')
        set_pending_memory(row['job'], plan['memory_gib'])
        row.update(cpus=plan['cpus'], memory_gib=plan['memory_gib'], allocated_hours=plan['wall_hours'])
    temporary = CAMPAIGN / 'submission.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(CAMPAIGN / 'submission.json')
    subprocess.run(['flock', str(CAMPAIGN / 'workspace.lock'), os.environ['AMBI_WORKSPACE_PYTHON'],
                    'slurm/inner_entropy_workspace.py', str(CAMPAIGN / 'submission.json')], cwd=ROOT, check=True)
    assert read(CAMPAIGN / 'workspace.json')['api_verified']
    for row in state['runs']:
        base.resize_pending(row['job'], plan)
        row['released'] = True
        temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(CAMPAIGN / 'submission.json')
    state['status'] = 'all three training seeds released after Dog Stand validation'
    temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(CAMPAIGN / 'submission.json')
    print('DOG_STAND_TRAINING_RELEASED', json.dumps([r['job'] for r in state['runs']]), flush=True)


def train(index):
    import torch
    assert os.environ['WANDB_MODE'] == 'disabled' and os.environ['SLURM_JOB_ID']
    _, _, run = validate_recipe(index)
    receipt = read(CAMPAIGN / 'gate.json')
    assert receipt['passed'] and receipt['binding'] == base.binding(MANIFEST)
    assert receipt['graph_native_graph_continuation']
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.set_num_interop_threads(1)
    current = base.accelerated_runtime()
    assert current == receipt['runtime'], (current, receipt['runtime'])
    base.load_global_parity(CAMPAIGN, current)
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == receipt['plan']['cpus']
    spec = read(CAMPAIGN / 'submission.json')['runs'][index]
    assert spec['seed'] == run['seed'] and str(spec['job']) == os.environ['SLURM_JOB_ID']
    assert spec['id'] == os.environ['WANDB_RUN_ID']
    output = CAMPAIGN / 'production' / f"seed{run['seed']}"
    output.mkdir(parents=True, exist_ok=False)
    space = base.checkpoint_space(output, 3 * 40 * receipt['plan']['checkpoint_bytes'])
    base.write_new(output / 'launch.json', {'binding': base.binding(MANIFEST), 'seed': run['seed'],
                   'runtime': current, 'plan': receipt['plan'], 'gate': str(CAMPAIGN / 'gate.json'),
                   'space': space, 'job_id': os.environ['SLURM_JOB_ID'], 'wandb_run_id': spec['id']})
    print('DOG_STAND_STARTUP_VERIFIED', run['seed'], flush=True)
    os.environ['WANDB_MODE'] = 'online'
    sys.argv = ['main.py', '--run', str(MANIFEST), '--alg-dir', 'configs/dmcontrol/algs',
                '--log-dir', str(output), '--alg-index', str(index), '--trial-index', '0', '--num-runs', '1']
    runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'gate', 'release', 'train'))
    parser.add_argument('--index', type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.mode == 'train':
        assert args.index is not None
        train(args.index)
    else:
        globals()[args.mode]()
