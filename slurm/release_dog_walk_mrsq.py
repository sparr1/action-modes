"""Release the held Dog Walk Q10 job after its pinned GPU validation succeeds."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm.release_mrsq_q5_suite import (allocation_updates, atomic, command,
    fields, reservation_check, verify_allocation)

CAMPAIGN = Path('/oscar/scratch/rgao48/ambi/dog-walk-mrsq-q10-20261118')


def validate_gate_files(gate, state, source, root):
    """Bind validation evidence to the exact training source and configuration."""
    binding = gate['binding']
    assert binding['commit'] == state['commit']
    assert binding['manifest'] == 'configs/dmcontrol/experiments/ambi_dog_walk_mrsq_q10_seed55.json'
    assert set(binding['config_sha256']) == {'ambi_dog_walk_mrsq_q10_seed55'}
    files = [(binding['manifest'], binding['manifest_sha256']),
             ('environments/dmcontrol/uv.lock', binding['lock_sha256'])]
    files += [('configs/dmcontrol/algs/' + name + '.json', digest)
              for name, digest in binding['config_sha256'].items()]
    for relative, digest in files:
        path = (source / relative).resolve()
        assert path.is_relative_to(source.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    parity = Path(gate['cuda_parity_junit'])
    assert parity.is_relative_to(root)
    assert hashlib.sha256(parity.read_bytes()).hexdigest() == gate['cuda_parity_sha256']


def release(publisher):
    assert os.environ.get('SLURM_JOB_ID')
    controller = Path(__file__).resolve().parents[1]
    assert command(['git', '-C', str(controller), 'rev-parse', 'HEAD']) == os.environ['AMBI_CONTROLLER_COMMIT']
    assert not command(['git', '-C', str(controller), 'status', '--porcelain', '--untracked-files=all'])
    with (CAMPAIGN / 'state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((CAMPAIGN / 'submission.json').read_text())
        assert len(state['runs']) == 1
        row = state['runs'][0]
        assert row['task'] == 'dog-walk' and row['num_q'] == 10 and row['seed'] == 55
        assert str(row['release_job']) == os.environ['SLURM_JOB_ID']
        assert not row.get('released'), 'Already released; do not repeat the launch'
        root = CAMPAIGN
        assert Path(state['output']) == root
        source = Path(state['remote_source'])
        assert command(['git', '-C', str(source), 'rev-parse', 'HEAD']) == state['commit']
        assert command(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}']) == state['tree']
        assert not command(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'])
        gate = json.loads((root / 'gates/q10/validated.json').read_text())
        assert gate['passed'] and gate['task'] == 'dog-walk' and gate['ensemble_size'] == 10
        validate_gate_files(gate, state, source, root)
        plan = gate['plan']
        assert plan['index'] == 0 and plan['cpus'] in (1, 2)
        assert plan['memory_gib'] > 0 and plan['wall_hours'] > 0
        job = row['job']
        current = fields(job)
        assert current['UserId'].startswith('rgao48(')
        assert current['JobName'] == 'mrsq-q10-dog-walk-s55'
        assert current['WorkDir'] == str(source)
        assert current['Command'] == str(source / 'slurm/run_dog_walk_mrsq_oscar.sbatch')
        assert current['JobState'] == 'PENDING' and current['Reason'] == 'JobHeldUser'
        budget = reservation_check(job, plan)
        subprocess.run(['scontrol', 'hold', str(job)], check=True)
        subprocess.run(allocation_updates(job, plan), check=True)
        actual = fields(job)
        assert actual['JobState'] == 'PENDING' and actual['Reason'] == 'JobHeldUser'
        verify_allocation(actual, plan)
        row.update(plan=plan, gate_passed=True, reservation_check=budget,
                   verified_allocation={k: actual[k] for k in ('NumCPUs', 'CPUs/Task', 'MinCPUsNode', 'TresPerTask', 'MinMemoryNode', 'TimeLimit')},
                   pending_status='validated; awaiting start / job ' + str(job))
        view_path = root / 'submission.json'
        atomic(view_path, state)
        subprocess.run(['flock', str(root / 'workspace.lock'), publisher,
                        'slurm/inner_entropy_workspace.py', str(view_path)], cwd=source, check=True)
        workspace = json.loads((root / 'workspace.json').read_text())
        assert workspace['api_verified'] and workspace['panel_count'] == 47
        assert workspace['run_ids'][0] == row['id']
        subprocess.run(['scontrol', 'release', str(job)], check=True)
        row['released'] = True
        atomic(view_path, state)
        atomic(root / 'gates/q10/release.json', {'training_job': job, 'controller_job': os.environ['SLURM_JOB_ID'],
               'controller_commit': os.environ['AMBI_CONTROLLER_COMMIT'], 'training_commit': state['commit'],
               'plan': plan, 'allocation_verified': True, 'workspace_verified': True})
        print('DOG_WALK_Q10_TRAINING_RELEASED', job, json.dumps(plan), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publisher-python', required=True)
    args = parser.parse_args()
    release(args.publisher_python)
