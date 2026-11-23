"""Release one held Q5 training job after its independent GPU gate succeeds.

Run this lightweight controller as an afterok CPU job. The training checkout
and gate remain pinned to the commit recorded in the campaign, independently
of this controller's commit. It never creates a training run or new GPU job.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SUITE = Path('/oscar/scratch/rgao48/ambi/mrsq-q5-suite-20261118')
TASKS = ('dog-trot', 'dog-run', 'humanoid-stand', 'humanoid-walk', 'humanoid-run')


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def fields(job):
    return dict(x.split('=', 1) for x in command(['scontrol', 'show', 'job', str(job), '-o']).split() if '=' in x)


def wall_minutes(value):
    day, clock = value.split('-') if '-' in value else ('0', value)
    parts = list(map(int, clock.split(':')))
    if len(parts) == 2:
        parts = [0] + parts
    hours, minutes, seconds = parts
    return int(day) * 1440 + hours * 60 + minutes + seconds / 60


def verify_allocation(actual, plan):
    count = str(plan['cpus'])
    for key in ('NumCPUs', 'CPUs/Task', 'MinCPUsNode'):
        assert actual[key] == count, (key, actual)
    assert actual['TresPerTask'] in ('cpu=' + count, 'cpu:' + count), actual
    assert actual['MinMemoryNode'] == str(plan['memory_gib']) + 'G', actual
    assert wall_minutes(actual['TimeLimit']) == plan['wall_hours'] * 60, actual


def allocation_updates(job, plan):
    count = str(plan['cpus'])
    # scontrol's MinMemoryNode update accepts integer MiB, unlike sbatch --mem.
    return ['scontrol', 'update', 'JobId=' + str(job),
            'NumCPUs=' + count, 'CPUsPerTask=' + count,
            'TresPerTask=cpu:' + count, 'MinCPUsNode=' + count,
            'MinMemoryNode=' + str(plan['memory_gib'] * 1024),
            'TimeLimit=' + str(plan['wall_hours']) + ':00:00']


def reservation_check(job, plan, *, concurrent=False):
    line = command(['sacctmgr', '-nP', 'show', 'qos', 'pri-gpu+',
                    'format=Name,MaxTRESRunMinsPU,MaxTRESPerUser'])
    _, run_minutes, allocated = line.split('|')[:3]
    limits = dict(x.split('=', 1) for x in run_minutes.split(','))
    wall = plan['wall_hours'] * 60
    durations = {'cpu': [wall] * plan['cpus'], 'gres/gpu': [wall]}
    # Slurm reports the full limit as time left for pending jobs. For running
    # jobs, count the remaining reservation in the concurrent budget.
    time_format = '%L' if concurrent else '%l'
    queue = command(['squeue', '-h', '-r', '-u', 'rgao48', '-o', f'%i|%q|%C|{time_format}|%b'])
    for row in queue.splitlines():
        ident, qos, cpus, time, gres = row.split('|')
        if qos != 'pri-gpu+' or ident == str(job):
            continue
        minutes = wall_minutes(time)
        match = re.search(r'gpu:(?:[^:,]+:)?([0-9]+)', gres)
        durations['cpu'].extend([minutes] * int(cpus))
        durations['gres/gpu'].extend([minutes] * (int(match[1]) if match else 0))
    # At most the user allocation limit can run simultaneously, even when
    # there are more queued jobs.
    caps = (dict(pair.split('=', 1) for pair in allocated.split(',')) if concurrent else {})
    totals = {key: sum(sorted(times, reverse=True)[:int(caps.get(key, len(times)))])
              for key, times in durations.items()}
    assert all(value <= float(limits[key]) for key, value in totals.items()), (totals, limits)
    budget_name = 'remaining_reservation_upper_bound_minutes' if concurrent else 'full_reservation_upper_bound_minutes'
    result = {budget_name: totals, 'limits': limits}
    if concurrent:
        result['allocation_limits'] = {key: int(caps[key]) for key in durations}
    return result


def release(task, publisher):
    assert task in TASKS and os.environ.get('SLURM_JOB_ID')
    controller = Path(__file__).resolve().parents[1]
    assert command(['git', '-C', str(controller), 'rev-parse', 'HEAD']) == os.environ['AMBI_CONTROLLER_COMMIT']
    assert not command(['git', '-C', str(controller), 'status', '--porcelain', '--untracked-files=all'])
    with (SUITE / 'state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((SUITE / 'submission.json').read_text())
        row = next(r for r in state['cases'] if r['task'] == task)
        assert str(row['release_job']) == os.environ['SLURM_JOB_ID']
        assert not row.get('released'), 'Already released; do not repeat the launch'
        root = SUITE / task
        assert Path(row['output']) == root
        source = Path(state['remote_source'])
        assert command(['git', '-C', str(source), 'rev-parse', 'HEAD']) == state['commit']
        assert command(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}']) == state['tree']
        assert not command(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'])
        gate = json.loads((root / 'gates/q5/validated.json').read_text())
        assert gate['passed'] and gate['task'] == task and gate['ensemble_size'] == 5
        binding = gate['binding']
        assert binding['commit'] == state['commit']
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
        plan = gate['plan']
        assert plan['index'] == 0 and plan['cpus'] in (1, 2)
        assert plan['memory_gib'] > 0 and plan['wall_hours'] > 0
        job = row['job']
        current = fields(job)
        assert current['UserId'].startswith('rgao48(')
        assert current['JobName'] == 'mrsq-q5-' + task + '-s55'
        assert current['WorkDir'] == str(source)
        assert current['Command'] == str(source / 'slurm/run_mrsq_q5_suite_oscar.sbatch')
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
        view = json.loads(view_path.read_text())
        view['runs'] = [row]
        atomic(view_path, view)
        atomic(SUITE / 'submission.json', state)
        subprocess.run(['flock', str(root / 'workspace.lock'), publisher,
                        'slurm/inner_entropy_workspace.py', str(view_path)], cwd=source, check=True)
        workspace = json.loads((root / 'workspace.json').read_text())
        assert workspace['api_verified'] and workspace['panel_count'] == 47
        assert workspace['run_ids'][0] == row['id']
        subprocess.run(['scontrol', 'release', str(job)], check=True)
        row['released'] = True
        view['runs'] = [row]
        atomic(view_path, view)
        atomic(SUITE / 'submission.json', state)
        atomic(root / 'gates/q5/release.json', {'training_job': job, 'controller_job': os.environ['SLURM_JOB_ID'],
               'controller_commit': os.environ['AMBI_CONTROLLER_COMMIT'], 'training_commit': state['commit'],
               'plan': plan, 'allocation_verified': True, 'workspace_verified': True})
        print('Q5_TRAINING_RELEASED', task, job, json.dumps(plan), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--publisher-python', required=True)
    args = parser.parse_args()
    release(args.task, args.publisher_python)
