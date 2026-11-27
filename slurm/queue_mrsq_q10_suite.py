"""Queue per-task gates and release the corresponding pinned Oscar training jobs."""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from slurm.release_mrsq_q5_suite import allocation_updates, atomic, fields, reservation_check, wall_minutes
from slurm.ambi_transfer_training import binding, recipe, write_new
from slurm.mrsq_q10_suite import CAMPAIGN, SEED56_TASKS, SEED57_TASKS, lineage_path, load_plan, manifest_path, task_path


PYTHON = '/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python'
LAUNCHER = SOURCE / 'slurm/run_mrsq_q10_suite_oscar.sbatch'
TASKS = ('humanoid-walk', 'humanoid-run', 'dog-trot', 'dog-run', 'humanoid-stand', 'dog-walk')
PROVISIONAL = {'cpus': 1, 'memory_gib': 14, 'wall_hours': 102}
GRACE_TASKS = ('dog-stand', 'humanoid-walk', 'humanoid-run')
GRACE_DEADLINE_HOURS = 20 * 24
GRACE_BUFFER_HOURS = 24
HARD_RESERVE_GIB = 80


def command(*arguments):
    return subprocess.check_output(arguments, cwd=SOURCE, text=True, timeout=45).strip()


def read(path):
    return json.loads(path.read_text())


def validate_pin(commit):
    assert command('git', 'rev-parse', 'HEAD') == commit
    assert not command('git', 'status', '--porcelain', '--untracked-files=all')
    assert os.access(PYTHON, os.X_OK) and LAUNCHER.is_file()
    return command('git', 'rev-parse', 'HEAD^{tree}')


def job_name(task, role, seed=55):
    return f'mrsq-q10-{task}-s{seed}-{role}'


def job_for(task, role, job, *, seed=55):
    current = fields(job)
    assert current['UserId'].startswith('rgao48('), current
    assert current['JobName'] == job_name(task, role, seed), current
    assert current['WorkDir'] == str(SOURCE), current
    return current


def submit_one(task, role, command_args, row, receipt, exports, *, seed=55):
    if role in row:
        return row[role]
    name = job_name(task, role, seed)
    assert not command('squeue', '-h', '-u', 'rgao48', '-n', name, '-o', '%i'), name
    args = ['sbatch', '--parsable', '--account=default', '--nodes=1', '--ntasks=1',
            '--chdir=' + str(SOURCE), '--job-name=' + name,
            '--output=' + str(CAMPAIGN / 'slurm' / f'{task_path(task, seed=seed).name}-{role}-%j.log'),
            '--export=' + exports, *command_args]
    job = command(*args).split(';')[0]
    assert job.isdecimal(), job
    row[role] = job
    atomic(receipt, row)
    return job


def verified_seed55_plan(task):
    reference_root = task_path(task, CAMPAIGN)
    reference_gate = reference_root / 'benchmark'
    reference = read(reference_gate / 'plan.json')
    release = read(reference_root / 'release.json')
    schedule = read(reference_root / 'scheduling.json')
    assert release['allocation_verified'] and release['plan'] == reference
    assert reference['passed'] and reference['task'] == task and reference['cpus'] == 1
    assert reference['runtime']['gpu'] == 'NVIDIA L40S'
    assert str(reference['gate_job']) == release['gate_job'] == schedule['gate']
    reference_commit = schedule['commit']
    assert reference['binding'] == {**binding(manifest_path(task)), 'commit': reference_commit}
    assert command('git', 'rev-parse', f'{reference_commit}^{{tree}}') == schedule['tree']
    command('git', 'diff', '--quiet', reference_commit, 'HEAD', '--',
            'main.py', 'RL', 'domains', 'utils', 'environments',
            'slurm/ambi_transfer_training.py', 'slurm/benchmark_ambi_training_cpu.py',
            'slurm/inner_entropy_oscar.py', 'slurm/posttanh_dog_trot_resume.py')
    parity = (reference_gate / 'parity.xml').read_bytes()
    assert hashlib.sha256(parity).hexdigest() == reference['parity_sha256']
    return reference, release, parity


def verify_matching_workload(reference_task, task, seed):
    identity_fields = {'wandb_run_name', 'wandb_group', 'wandb_tags'}
    manifest_labels = {'configs', 'name', 'description', 'study_note'}
    _, original_manifest, original = recipe(0, manifest_path(reference_task))
    _, next_manifest, next_seed = recipe(0, manifest_path(task, seed))
    if reference_task != task:
        manifest_labels.add('study_type')
        assert original_manifest['env_params']['task'] == reference_task
        assert next_manifest['env_params']['task'] == task
        original_manifest['env_params']['task'] = task
    assert {k: v for k, v in original_manifest.items() if k not in manifest_labels} == {
        k: v for k, v in next_manifest.items() if k not in manifest_labels}
    for run in (original, next_seed):
        run['alg_params'] = {k: v for k, v in run['alg_params'].items() if k not in identity_fields}
    assert original['seed'] == 55 and next_seed['seed'] == seed
    assert {k: v for k, v in original.items() if k not in {'name', 'seed'}} == {
        k: v for k, v in next_seed.items() if k not in {'name', 'seed'}}


def reuse_timing_plan(task, root, *, seed=56):
    """Reuse a verified Oscar L40S plan for a matching workload."""
    gate = root / 'benchmark'
    if gate.is_dir():
        return load_plan(task, root, seed=seed)
    reference_task = 'dog-walk' if task == 'dog-stand' else task
    reference, release, parity = verified_seed55_plan(reference_task)
    verify_matching_workload(reference_task, task, seed)
    plan = {**reference, 'task': task, 'binding': binding(manifest_path(task, seed)),
            'sizing_reference': {'seed': 55, 'binding': reference['binding'],
                                 'training_job': release['training_job']}}
    if task == 'dog-stand':
        # Dog Stand, Walk, Trot and Run all use 223 observations and 38 actions.
        # Stand measured 0.302855 s/decision on Oscar L40S (9256526 finished
        # 1M decisions in 304,091 seconds), and 0.301520 on Hydra L40S (225561).
        # The largest same-class sizing receipts protect the scratch allocation.
        comparable = [reference]
        for other in ('dog-trot', 'dog-run'):
            verify_matching_workload(other, task, seed)
            comparable.append(verified_seed55_plan(other)[0])
        assert all(row['runtime'] == reference['runtime'] for row in comparable)
        for key in ('memory_gib', 'wall_hours', 'checkpoint_bytes', 'full_generation_bytes'):
            plan[key] = max(row[key] for row in comparable)
        # Observed active Dog Stand history plus full replay extrapolates to
        # about 1.30 GB at 1M, inclusive of W&B history omitted from the gate.
        plan['full_generation_bytes'] = max(plan['full_generation_bytes'], 1_301_000_000)
        plan['sizing_reference'].update(
            task=reference_task, same_class_tasks=['dog-walk', 'dog-trot', 'dog-run'],
            observed_dog_stand=[{'job': '9256526', 'cluster': 'Oscar', 'seconds_per_decision': 0.302855,
                                 'full_run_seconds': 304091},
                                {'job': '225561', 'cluster': 'Hydra', 'seconds_per_decision': 0.301520}])
    with TemporaryDirectory(prefix='.timing-', dir=root) as temporary:
        staging = Path(temporary)
        (staging / 'parity.xml').write_bytes(parity)
        write_new(staging / 'plan.json', plan)
        staging.rename(gate)
    return load_plan(task, root, seed=seed)


def queue_task(task, commit, tree, *, seed=55, hold=False, scratch_grace=False):
    root = task_path(task, CAMPAIGN, seed=seed)
    root.mkdir(mode=0o700, exist_ok=True)
    receipt = root / 'scheduling.json'
    lineage = lineage_path(task, CAMPAIGN, seed=seed)
    row = read(receipt) if receipt.exists() else {'task': task, 'commit': commit, 'tree': tree,
                                                  'source': str(SOURCE), 'lineage': str(lineage)}
    assert (row['commit'], row['tree'], row['source'], row['lineage']) == (
        commit, tree, str(SOURCE), str(lineage))
    if receipt.exists():
        assert row.get('scratch_grace', False) == scratch_grace
    if scratch_grace:
        assert seed == 57 and task in GRACE_TASKS
        row['scratch_grace'] = True
    if 'train' not in row:
        assert not lineage.exists(), 'A new job cannot replace a lineage'
    exports = ','.join(['ALL', 'AMBI_EXPECTED_COMMIT=' + commit, 'AMBI_PYTHON=' + PYTHON,
                        'AMBI_Q10_CAMPAIGN=' + str(CAMPAIGN), f'AMBI_Q10_SEED={seed}', 'WANDB_MODE=disabled',
                        'PYTHONDONTWRITEBYTECODE=1'])
    exports += f',AMBI_Q10_SCRATCH_GRACE={int(scratch_grace)}'
    hardware = ['--gres=gpu:l40s:1', '--exclude=gpu2708', '--hint=nomultithread']
    if seed != 55:
        plan = reuse_timing_plan(task, root, seed=seed)
        gate = str(plan['gate_job'])
        dependency = []
        row.update(seed=seed, gate=gate)
    else:
        plan = PROVISIONAL
        gate = submit_one(task, 'gate', ['--partition=gpu-debug', '--qos=gpu-debug', *hardware,
                          '--cpus-per-task=2', '--mem=16G', '--time=00:30:00', '--no-requeue',
                          str(LAUNCHER), 'benchmark', task], row, receipt, exports)
        dependency = ['--dependency=afterok:' + gate]
    if 'train' not in row:
        reservation_check(-1, plan, concurrent=True)
    if hold:
        scratch = scratch_status(task, plan, seed=seed, allow_grace=scratch_grace)
    train = submit_one(task, 'train', ['--partition=gpu', '--qos=pri-gpu+', *hardware,
                       f"--cpus-per-task={plan['cpus']}", f"--mem={plan['memory_gib']}G",
                       f"--time={plan['wall_hours']}:00:00", '--requeue',
                       '--signal=USR1@1800', '--hold', *dependency,
                       str(LAUNCHER), 'train', task], row, receipt, exports, seed=seed)
    if seed != 55:
        if hold:
            verify_held(job_for(task, 'train', train, seed=seed), plan, gate)
            row['held_verification'] = {'cpus': plan['cpus'], 'memory_gib': plan['memory_gib'],
                                        'wall_hours': plan['wall_hours'], 'scratch_quota': scratch}
            atomic(receipt, row)
        else:
            release_task(task, commit, tree, seed=seed)
        return row
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


def scratch_free_gib():
    rows = [line.split() for line in command('checkquota').splitlines()]
    selected = [row for row in rows if row[:2] == ['rgao48', '/oscar/scratch']]
    assert len(selected) == 1 and selected[0][-2:] == ['OK', 'None'], selected
    return float(selected[0][4]) - float(selected[0][2])


def bytes_on_disk(paths):
    total = 0
    for file in paths:
        try:
            if file.is_file() and not file.is_symlink():
                total += file.stat().st_size
        except FileNotFoundError:
            # A concurrent commit can retire its predecessor between those reads.
            continue
    return total


def booked_scratch_gib(task, new_plan, *, seed=55):
    plans = {task_path(task, CAMPAIGN, seed=seed): new_plan}
    for other_seed, tasks in ((55, TASKS), (56, SEED56_TASKS), (57, SEED57_TASKS)):
        for other in tasks:
            root = task_path(other, CAMPAIGN, seed=other_seed)
            if root in plans:
                continue
            for receipt in (root / 'release.json', root / 'release-intent.json'):
                if receipt.is_file():
                    plans[root] = read(receipt)['plan']
                    break
            else:
                if other_seed == 57 and (root / 'benchmark/plan.json').is_file():
                    plans[root] = read(root / 'benchmark/plan.json')
    remaining = 0
    for root, plan in plans.items():
        lineage = root / 'lineage'
        if (lineage / 'DONE').is_file():
            continue
        # Keep two completed generations and space for the atomically staged third.
        generations = bytes_on_disk((lineage / 'generations').glob('*/**/*'))
        remaining += max(0, 3 * plan['full_generation_bytes'] - generations)
        # Portable 25k weights are siblings of the lineage on the same quota.
        models = (lineage.parent / 'models').rglob('model:*')
        snapshots = bytes_on_disk(path for path in models if path.name.rsplit('_', 1)[-1].isdigit())
        remaining += max(0, 40 * plan['checkpoint_bytes'] - snapshots)
    return remaining / 2**30


def verified_scratch_quota(task, plan, *, seed):
    needed, free = booked_scratch_gib(task, plan, seed=seed), scratch_free_gib()
    assert free >= needed, f'Scratch has {free:.2f} GiB free, needs {needed:.2f} GiB for resume and snapshots'
    return {'free_gib': free, 'reserved_gib': needed}


def verified_scratch_grace(task, plan, *, seed, deadline=None):
    assert seed == 57 and task in GRACE_TASKS
    rows = [line.split() for line in command('checkquota').splitlines()]
    selected = [row for row in rows if row[:2] == ['rgao48', '/oscar/scratch']]
    assert len(selected) == 1 and len(selected[0]) >= 11, selected
    row = selected[0]
    usage, soft, hard = map(float, (row[2], row[4], row[5]))
    assert 0 <= usage < hard and 0 < soft < hard, row
    now = datetime.now(timezone.utc)
    fresh = row[9:] == ['OK', 'None']
    if fresh:
        assert usage <= soft, row
    if deadline is None:
        # A fresh read gives a lower bound on the future grace expiry even
        # when the active quota command's time formatting varies by host.
        assert fresh, 'The opt-in requires an unexpired, fresh OK/None observation'
        deadline = now + timedelta(hours=GRACE_DEADLINE_HOURS)
    else:
        if not fresh:
            assert row[9] == 'SOFT_EXCEEDED' and usage >= soft, row
        deadline = datetime.fromisoformat(deadline)
        assert deadline.tzinfo is not None and deadline <= now + timedelta(hours=GRACE_DEADLINE_HOURS)
    needed = booked_scratch_gib(task, plan, seed=seed)
    assert usage + needed + HARD_RESERVE_GIB < hard, (
        f'Scratch projected {usage + needed:.2f} GiB plus reserve {HARD_RESERVE_GIB} GiB '
        f'reaches its {hard:.2f} GiB hard cap')
    assert now + timedelta(hours=plan['wall_hours'] + GRACE_BUFFER_HOURS) < deadline, (
        'Scratch grace deadline cannot cover a full allocation and 24-hour buffer')
    return {'observed_at_utc': now.isoformat(), 'usage_state': row[9], 'grace_deadline_utc': deadline.isoformat(),
            'used_gib': usage, 'soft_gib': soft, 'hard_gib': hard, 'reserved_gib': needed,
            'hard_reserve_gib': HARD_RESERVE_GIB}


def scratch_status(task, plan, *, seed, allow_grace=False):
    if allow_grace:
        return verified_scratch_grace(task, plan, seed=seed)
    return verified_scratch_quota(task, plan, seed=seed)


def check_scratch_grace(task, *, seed, commit):
    root = task_path(task, CAMPAIGN, seed=seed)
    scheduled = read(root / 'scheduling.json')
    released = read(root / 'release-intent.json')
    assert os.environ.get('AMBI_Q10_SCRATCH_GRACE') == '1'
    assert scheduled['scratch_grace'] is True and scheduled['seed'] == seed
    assert scheduled['commit'] == commit and scheduled['task'] == task
    assert scheduled['train'] == released['training_job'] == os.environ['SLURM_JOB_ID']
    plan = load_plan(task, root, seed=seed)
    assert released['allocation_verified'] and released['plan'] == plan
    deadline = released['scratch_quota']['grace_deadline_utc']
    return verified_scratch_grace(task, plan, seed=seed, deadline=deadline)


def release_task(task, commit, tree, *, seed=55):
    root = task_path(task, CAMPAIGN, seed=seed)
    row = read(root / 'scheduling.json')
    assert row['commit'] == commit and row['tree'] == tree and row['source'] == str(SOURCE)
    controller = os.environ.get('SLURM_JOB_ID')
    if controller is not None:
        assert row['release'] == controller, row
    release = root / 'release.json'
    if release.exists():
        return read(release)
    gate, train = row['gate'], row['train']
    if seed == 55:
        try:
            completed = job_for(task, 'gate', gate)
            assert completed['JobState'] == 'COMPLETED' and completed['ExitCode'] == '0:0', completed
        except subprocess.CalledProcessError:
            attested = command('sacct', '-nPX', '-j', gate, '-o', 'JobIDRaw,State,ExitCode').splitlines()
            assert f'{gate}|COMPLETED|0:0' in attested, attested
    plan = load_plan(task, root, seed=seed)
    assert plan['binding']['commit'] == commit and str(plan['gate_job']) == gate
    assert plan['cpu_replay'] is True and plan['wandb'] is False
    assert plan['cpus'] in (1, 2) and plan['memory_gib'] > 0 and plan['wall_hours'] > 0
    assert plan['full_generation_bytes'] > 0 and plan['checkpoint_bytes'] > 0
    assert plan['seconds_per_decision'] > 0
    current = job_for(task, 'train', train, seed=seed)
    intent = root / 'release-intent.json'
    if current['Reason'] != 'JobHeldUser' and intent.is_file():
        pending = read(intent)
        assert pending['training_job'] == train and pending['plan'] == plan
        atomic(release, pending)
        return pending
    verify_held(current, plan if seed != 55 or intent.is_file() else PROVISIONAL, gate)
    budget = reservation_check(train, plan, concurrent=True)
    scratch = scratch_status(task, plan, seed=seed, allow_grace=row.get('scratch_grace', False))
    if seed == 55:
        command(*allocation_updates(train, plan))
        current = job_for(task, 'train', train)
        verify_held(current, plan, gate)
    pending = {'training_job': train, 'gate_job': gate, 'controller_job': controller,
               'plan': plan, 'scratch_quota': scratch,
               'reservations': budget, 'allocation_verified': True}
    atomic(intent, pending)
    command('scontrol', 'release', train)
    atomic(release, pending)
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('queue', 'release', 'check-scratch-grace'))
    parser.add_argument('--task', choices=(*TASKS, 'dog-stand'))
    parser.add_argument('--seed', type=int, choices=(55, 56, 57), default=55)
    parser.add_argument('--hold', action='store_true', help='Stage a matched replicate without releasing the job')
    parser.add_argument('--allow-scratch-grace', action='store_true',
                        help='Allow bounded scratch soft-limit grace for an explicit seed-57 task')
    args = parser.parse_args()
    if args.allow_scratch_grace and (args.command != 'queue' or args.seed != 57 or args.task not in GRACE_TASKS):
        parser.error('--allow-scratch-grace applies only when queueing the three new seed-57 tasks')
    if args.task == 'dog-stand' and args.seed != 57:
        parser.error('Dog Stand in this launcher requires --seed 57')
    replicates = {56: SEED56_TASKS, 57: SEED57_TASKS}
    if args.seed in replicates and args.task not in replicates[args.seed]:
        parser.error(f'Seed {args.seed} requires an explicit --task from ' + ', '.join(replicates[args.seed]))
    if args.hold and (args.command != 'queue' or args.seed == 55):
        parser.error('--hold applies to queue for measured seed-56 and seed-57 replicates')
    commit = os.environ['AMBI_EXPECTED_COMMIT']
    tree = validate_pin(commit)
    if args.command == 'check-scratch-grace':
        if args.seed != 57 or args.task not in GRACE_TASKS:
            parser.error('Checking scratch grace requires an opt-in seed-57 task')
        print(json.dumps(check_scratch_grace(args.task, seed=args.seed, commit=commit)), flush=True)
        return
    CAMPAIGN.mkdir(mode=0o700, exist_ok=True)
    (CAMPAIGN / 'slurm').mkdir(mode=0o700, exist_ok=True)
    with (CAMPAIGN / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == 'queue':
            result = [queue_task(task, commit, tree, seed=args.seed, hold=args.hold,
                                 scratch_grace=args.allow_scratch_grace)
                      for task in (TASKS if args.task is None else (args.task,))]
        else:
            if args.task is None:
                parser.error('release requires --task')
            result = release_task(args.task, commit, tree, seed=args.seed)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
