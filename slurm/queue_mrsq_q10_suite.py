"""Queue per-task gates and release the corresponding pinned Oscar training jobs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from slurm.release_mrsq_q5_suite import allocation_updates, atomic, fields, reservation_check, wall_minutes


CAMPAIGN = Path('/oscar/scratch/rgao48/ambi/mrsq-q10-suite-20261122')
DURABLE = Path('/oscar/home/rgao48/ambi-durable/lineages/mrsq-q10-suite-20261122')
PYTHON = '/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python'
LAUNCHER = SOURCE / 'slurm/run_mrsq_q10_suite_oscar.sbatch'
TASKS = ('humanoid-walk', 'humanoid-run', 'dog-trot', 'dog-run')
PROVISIONAL = {'cpus': 1, 'memory_gib': 14, 'wall_hours': 102}


def command(*arguments):
    return subprocess.check_output(arguments, cwd=SOURCE, text=True, timeout=45).strip()


def read(path):
    return json.loads(path.read_text())


def validate_pin(commit):
    assert command('git', 'rev-parse', 'HEAD') == commit
    assert not command('git', 'status', '--porcelain', '--untracked-files=all')
    assert os.access(PYTHON, os.X_OK) and LAUNCHER.is_file()
    return command('git', 'rev-parse', 'HEAD^{tree}')


def job_name(task, role):
    return f'mrsq-q10-{task}-s55-{role}'


def job_for(task, role, job):
    current = fields(job)
    assert current['UserId'].startswith('rgao48('), current
    assert current['JobName'] == job_name(task, role), current
    assert current['WorkDir'] == str(SOURCE), current
    return current


def submit_one(task, role, command_args, row, receipt, exports):
    if role in row:
        return row[role]
    name = job_name(task, role)
    assert not command('squeue', '-h', '-u', 'rgao48', '-n', name, '-o', '%i'), name
    args = ['sbatch', '--parsable', '--account=default', '--nodes=1', '--ntasks=1',
            '--chdir=' + str(SOURCE), '--job-name=' + name,
            '--output=' + str(CAMPAIGN / 'slurm' / f'{task}-{role}-%j.log'),
            '--export=' + exports, *command_args]
    job = command(*args).split(';')[0]
    assert job.isdecimal(), job
    row[role] = job
    atomic(receipt, row)
    return job


def queue_task(task, commit, tree):
    root = CAMPAIGN / task
    root.mkdir(mode=0o700, exist_ok=True)
    receipt = root / 'scheduling.json'
    row = read(receipt) if receipt.exists() else {'task': task, 'commit': commit, 'tree': tree,
                                                  'source': str(SOURCE), 'lineage': str(DURABLE / task)}
    assert (row['commit'], row['tree'], row['source'], row['lineage']) == (
        commit, tree, str(SOURCE), str(DURABLE / task))
    if 'train' not in row:
        assert not (DURABLE / task).exists(), 'A new job cannot replace a lineage'
    exports = ','.join(['ALL', 'AMBI_EXPECTED_COMMIT=' + commit, 'AMBI_PYTHON=' + PYTHON,
                        'AMBI_Q10_CAMPAIGN=' + str(CAMPAIGN), 'WANDB_MODE=disabled',
                        'PYTHONDONTWRITEBYTECODE=1'])
    hardware = ['--gres=gpu:l40s:1', '--exclude=gpu2708', '--hint=nomultithread']
    gate = submit_one(task, 'gate', ['--partition=gpu-debug', '--qos=gpu-debug', *hardware,
                      '--cpus-per-task=2', '--mem=16G', '--time=00:30:00', '--no-requeue',
                      str(LAUNCHER), 'benchmark', task], row, receipt, exports)
    if 'train' not in row:
        reservation_check(-1, PROVISIONAL)
    train = submit_one(task, 'train', ['--partition=gpu', '--qos=pri-gpu+', *hardware,
                       '--cpus-per-task=1', '--mem=14G', '--time=102:00:00', '--requeue',
                       '--signal=USR1@1800', '--hold', '--dependency=afterok:' + gate,
                       str(LAUNCHER), 'train', task], row, receipt, exports)
    if not (root / 'release.json').is_file():
        verify_held(job_for(task, 'train', train), PROVISIONAL, gate)
    controller = shlex.join([PYTHON, '-B', str(Path(__file__).resolve()), 'release', '--task', task])
    submit_one(task, 'release', ['--partition=batch', '--qos=normal', '--cpus-per-task=1',
                '--mem=2G', '--time=00:08:00', '--no-requeue', '--dependency=afterany:' + gate,
                '--wrap', controller], row, receipt, exports)
    return row


def verify_held(current, plan, gate):
    assert current['JobState'] == 'PENDING' and current['Reason'] == 'JobHeldUser', current
    if current.get('Dependency') not in ('(null)', None):
        assert gate in current['Dependency'], current
    assert current['Partition'] == 'gpu' and current['QOS'] == 'pri-gpu+', current
    assert current['ExcNodeList'] == 'gpu2708' and current['TresPerNode'] == 'gres/gpu:l40s:1', current
    cpus = str(plan['cpus'])
    assert all(current[key] == cpus for key in ('NumCPUs', 'CPUs/Task', 'MinCPUsNode')), current
    assert current['TresPerTask'] in (f'cpu:{cpus}', f'cpu={cpus}'), current
    memory = current['MinMemoryNode']
    assert memory in (f"{plan['memory_gib']}G", f"{plan['memory_gib'] * 1024}M"), current
    assert wall_minutes(current['TimeLimit']) == plan['wall_hours'] * 60, current


def home_free_gib():
    rows = [line.split() for line in command('checkquota').splitlines()]
    selected = [row for row in rows if row[:2] == ['rgao48', '/oscar/home']]
    assert len(selected) == 1 and selected[0][-2:] == ['OK', 'None'], selected
    return float(selected[0][4]) - float(selected[0][2])


def booked_home_gib(task, new_plan):
    plans = {task: new_plan}
    for other in TASKS:
        if other != task and (CAMPAIGN / other / 'release.json').is_file():
            plans[other] = read(CAMPAIGN / other / 'release.json')['plan']
    remaining = 0
    for name, plan in plans.items():
        lineage = DURABLE / name
        if (lineage / 'DONE').is_file():
            continue
        # Keep two completed generations and space for the atomically staged third.
        bytes_required = 3 * plan['full_generation_bytes']
        committed = 0
        for file in (lineage / 'generations').glob('*/**/*'):
            try:
                if file.is_file():
                    committed += file.stat().st_size
            except FileNotFoundError:
                # A concurrent commit can retire its predecessor between those reads.
                continue
        remaining += max(0, bytes_required - committed)
    return remaining / 2**30


def release_task(task, commit, tree):
    root = CAMPAIGN / task
    row = read(root / 'scheduling.json')
    assert row['commit'] == commit and row['tree'] == tree and row['source'] == str(SOURCE)
    controller = os.environ.get('SLURM_JOB_ID')
    if controller is not None:
        assert row['release'] == controller, row
    release = root / 'release.json'
    if release.exists():
        return read(release)
    gate, train = row['gate'], row['train']
    try:
        completed = job_for(task, 'gate', gate)
        assert completed['JobState'] == 'COMPLETED' and completed['ExitCode'] == '0:0', completed
    except subprocess.CalledProcessError:
        attested = command('sacct', '-nPX', '-j', gate, '-o', 'JobIDRaw,State,ExitCode').splitlines()
        assert f'{gate}|COMPLETED|0:0' in attested, attested
    from slurm.mrsq_q10_suite import load_plan

    plan = load_plan(task, root)
    assert plan['binding']['commit'] == commit and str(plan['gate_job']) == gate
    assert plan['cpu_replay'] is True and plan['wandb'] is False
    assert plan['cpus'] in (1, 2) and plan['memory_gib'] > 0 and plan['wall_hours'] > 0
    assert plan['full_generation_bytes'] > 0 and plan['seconds_per_decision'] > 0
    current = job_for(task, 'train', train)
    intent = root / 'release-intent.json'
    if current['Reason'] != 'JobHeldUser' and intent.is_file():
        pending = read(intent)
        assert pending['training_job'] == train and pending['plan'] == plan
        atomic(release, pending)
        return pending
    verify_held(current, plan if intent.is_file() else PROVISIONAL, gate)
    budget = reservation_check(train, plan)
    needed, free = booked_home_gib(task, plan), home_free_gib()
    assert free >= needed, f'Home storage can reserve {free:.2f} GiB, needs {needed:.2f} GiB for exact resume'
    command(*allocation_updates(train, plan))
    current = job_for(task, 'train', train)
    verify_held(current, plan, gate)
    pending = {'training_job': train, 'gate_job': gate, 'controller_job': controller,
               'plan': plan, 'home_quota': {'free_gib': free, 'reserved_gib': needed},
               'reservations': budget, 'allocation_verified': True}
    atomic(intent, pending)
    command('scontrol', 'release', train)
    atomic(release, pending)
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('queue', 'release'))
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    commit = os.environ['AMBI_EXPECTED_COMMIT']
    tree = validate_pin(commit)
    CAMPAIGN.mkdir(mode=0o700, exist_ok=True)
    (CAMPAIGN / 'slurm').mkdir(mode=0o700, exist_ok=True)
    with (CAMPAIGN / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == 'queue':
            result = [queue_task(task, commit, tree) for task in (TASKS if args.task is None else (args.task,))]
        else:
            if args.task is None:
                parser.error('release requires --task')
            result = release_task(args.task, commit, tree)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
