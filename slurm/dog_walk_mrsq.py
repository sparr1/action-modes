"""Validate and launch the seed-55 Dog Walk five-critic MRS.Q aggregation run."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import inner_entropy_oscar as base
from slurm.dog_stand_mrsq import sized_plan

MANIFEST = ROOT / 'configs/dmcontrol/experiments/ambi_dog_walk_mrsq_q5_seed55.json'
CAMPAIGN = Path('/oscar/scratch/rgao48/ambi/dog-walk-mrsq-q5-20261118')
REFERENCE = Path('/oscar/scratch/rgao48/ambi/posttanh-dog-stand-20261115')
REFERENCE_COMMIT = '9dfd897832034a8009041a67f9ce4df0c44ed89f'


def read(path):
    return json.loads(Path(path).read_text())


def validate_recipe(index=0):
    assert index == 0
    name, manifest, run = base.recipe(index, MANIFEST)
    _, old_manifest, original = base.recipe(0, ROOT / 'configs/dmcontrol/experiments/ambi_dog_stand_mrsq_ensemble.json')
    expected = deepcopy(original)
    expected['name'] = name
    for key in ('wandb_run_name', 'wandb_tags', 'wandb_group'):
        expected['alg_params'][key] = run['alg_params'][key]
    assert run == expected, 'Only task and publication labels may change from Dog Stand Q5'
    expected_env = dict(old_manifest['env_params'], task='dog-walk')
    assert manifest['env_params'] == expected_env
    assert manifest['configs'] == ['ambi_dog_walk_mrsq_q5_seed55']
    for key in ('overrides_alg', 'trials', 'logs', 'save_trials',
                'checkpoint_every', 'save_strat', 'log_info', 'log_type'):
        assert manifest[key] == old_manifest[key], key
    return name, manifest, run


def prepare():
    validate_recipe()
    protected = ['RL', 'domains', 'utils', 'environments', 'main.py',
                 'tests/test_inner_entropy_graph.py', 'tests/test_posttanh_dog.py']
    subprocess.run(['git', 'diff', '--exit-code', REFERENCE_COMMIT, 'HEAD', '--', *protected], cwd=ROOT, check=True)
    old = read(REFERENCE / 'gpu-parity.json')
    reference = read(REFERENCE / 'gate.json')
    assert reference['passed'] and reference['binding']['commit'] == REFERENCE_COMMIT
    base.validate_global_parity(old, reference['binding'], reference['runtime'])
    junit = Path(old['junit_path'])
    assert junit.is_relative_to(REFERENCE)
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old['junit_sha256']
    assert base.parity_cases(junit) == old['case_names']
    copied = CAMPAIGN / 'reused-gpu-parity.xml'
    with copied.open('xb') as stream:
        stream.write(junit.read_bytes())
    base.write_new(CAMPAIGN / 'gpu-parity.json', {**old, 'binding': base.binding(MANIFEST),
        'original_binding': old['binding'], 'junit_path': str(copied),
        'source_evidence': {'reference': str(REFERENCE), 'protected_paths': protected,
            'new_validation': 'Fresh Q5 CUDA parity, Dog Walk environment/training, CPU sizing, geometry and checkpoints.'}})
    base.write_new(CAMPAIGN / 'preflight.json', {'binding': base.binding(MANIFEST),
                    'reference': reference, 'source_unchanged': protected})


def gate(index=0):
    assert os.environ['WANDB_MODE'] == 'disabled' and os.environ['SLURM_JOB_ID']
    assert base.MANIFEST == MANIFEST and base.CAMPAIGN_ROOT == CAMPAIGN
    validate_recipe(index)
    junit = CAMPAIGN / 'q5-cuda-parity.xml'
    program = ('import sys,torch,pytest; torch.set_num_threads(1); '
               "torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('high'); "
               'raise SystemExit(pytest.main(sys.argv[1:]))')
    subprocess.run([sys.executable, '-c', program, '-q',
                    'tests/test_dog_stand_mrsq.py::test_cuda_full_ensemble_dog_parity[5]',
                    'tests/test_dog_walk_mrsq.py::test_real_dog_walk_two_control_steps',
                    '--junitxml=' + str(junit)], cwd=ROOT,
                   env=dict(os.environ, AMBI_RUN_REAL_DMCONTROL_TESTS='1'), check=True)
    cases = list(ET.parse(junit).getroot().iter('testcase'))
    assert len(cases) == 2 and not any(c.find(t) is not None for c in cases for t in ('failure', 'error', 'skipped'))
    output = CAMPAIGN / 'gates' / 'q5'
    base.gate(SimpleNamespace(index=index, output=output, release_job=None))
    receipt = read(output / 'receipt.json')
    rows = [read(output / f'cpu-{n}.json') for n in (1, 2)]
    assert all(r['passed'] and r['binding'] == base.binding(MANIFEST) and r['runtime'] == receipt['runtime'] for r in rows)
    reference = read(CAMPAIGN / 'preflight.json')['reference']['plan']
    plan = sized_plan(rows, reference['reference_runtime_seconds'], reference['seconds_per_decision'])
    receipt.update(plan=plan, ensemble_size=5, task='dog-walk', cuda_parity_junit=str(junit),
                   cuda_parity_sha256=hashlib.sha256(junit.read_bytes()).hexdigest(), wandb=False)
    base.write_new(output / 'validated.json', receipt)
    print('DOG_WALK_MRSQ_GATE_PASSED', json.dumps(plan), flush=True)


def train(index=0):
    validate_recipe(index)
    receipt_path = CAMPAIGN / 'gates/q5/validated.json'
    receipt = read(receipt_path)
    assert receipt['passed'] and receipt['ensemble_size'] == 5 and receipt['task'] == 'dog-walk'
    assert hashlib.sha256(Path(receipt['cuda_parity_junit']).read_bytes()).hexdigest() == receipt['cuda_parity_sha256']
    spec = read(CAMPAIGN / 'submission.json')['runs'][index]
    assert spec['num_q'] == 5 and spec['seed'] == 55
    assert str(spec['job']) == os.environ['SLURM_JOB_ID'] and spec['id'] == os.environ['WANDB_RUN_ID']
    assert os.environ['WANDB_MODE'] == 'disabled'
    os.environ['WANDB_MODE'] = 'online'
    base.train(SimpleNamespace(index=index, gate=receipt_path, output=CAMPAIGN / 'production/q5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'gate', 'train'))
    parser.add_argument('--index', type=int, choices=(0,), default=0)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        globals()[args.mode](args.index)
