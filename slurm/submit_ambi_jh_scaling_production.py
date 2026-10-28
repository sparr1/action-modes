"""Submit the three J/H scaling trainings after all Oscar GPU gates finish.

Run in one CPU batch job depending on the three smoke jobs. Submission intents
are immutable: an interrupted/ambiguous sbatch requires operator reconciliation,
never an automatic retry. This script does not evaluate or train on its CPU.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import ambi_reward_training as campaign

OUTPUT = Path('/oscar/scratch/rgao48/ambi/jh-scaling-training/20261028')
PYTHON = '/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python'
USER = 'rgao48'
QOS = 'pri-gpu+'
CASES = ('h3-j12-k3', 'h4-j10-k4', 'h5-j10-k5')


def command(arguments):
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()


def read(path):
    return json.loads(Path(path).read_text())


def durable_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def time_seconds(value):
    """Parse Slurm's [days-]hours:minutes:seconds or minutes:seconds."""
    assert value and value.upper() not in ('UNLIMITED', 'N/A', 'INVALID'), 'Unbounded Slurm reservation'
    days, _, clock = value.partition('-')
    if not clock:
        clock, days = days, '0'
    components = [int(item) for item in clock.split(':')]
    if len(components) == 1:
        seconds = components[0] * 60
    elif len(components) == 2:
        seconds = components[0] * 60 + components[1]
    else:
        assert len(components) == 3
        seconds = components[0] * 3600 + components[1] * 60 + components[2]
    return int(days) * 86400 + seconds


def tres(value):
    result = {}
    for field in value.split(','):
        if not field:
            continue
        key, amount = field.split('=', 1)
        result[key.removeprefix('gres/')] = amount
    return result


def gpu_count(value):
    if value in ('', '(null)', 'N/A', 'None'):
        return 0
    count = 0
    for part in value.split(','):
        if 'gpu:' in part:
            match = re.search(r'(?:gres/)?gpu(?::[^:(),]+)?:(\d+)(?:\(|$)', part)
            assert match, f'Unknown GPU request {value!r}'
            count += int(match.group(1))
    return count


def verify_smoke_jobs(jobs):
    assert len(jobs) == len(set(jobs)) == 3 and all(re.fullmatch(r'[1-9][0-9]*', value) for value in jobs)
    rows = command(['sacct', '-n', '-P', '-X', '-j', ','.join(jobs), '--format=JobIDRaw,State,ExitCode'])
    found = {}
    for line in rows.splitlines():
        fields = line.split('|')
        if fields[0] in jobs:
            assert fields[0] not in found, 'Duplicate smoke accounting row'
            found[fields[0]] = fields[1:3]
    assert set(found) == set(jobs), 'Missing smoke accounting'
    assert all(value == ['COMPLETED', '0:0'] for value in found.values()), 'All three GPU smoke jobs must succeed'
    return found


def memory_mib(value, *, cpus=1, nodes=1):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTP]?)([cn]?)', value)
    assert match, f'Unknown Slurm memory request {value!r}'
    amount, unit, scope = match.groups()
    factor = {'K': 1/1024, '': 1, 'M': 1, 'G': 1024, 'T': 1024**2, 'P': 1024**3}[unit]
    return float(amount) * factor * (cpus if scope == 'c' else nodes)


def make_plans(output, commit, smoke_jobs):
    plans = []
    for case, smoke_job, (manifest, spec) in zip(CASES, smoke_jobs, campaign.SCALING_CASES.items()):
        name, operator, rounds, interval, horizon, decisions = spec
        assert case == f'h{horizon}-j{rounds}-k{interval}'
        _, _, run = campaign.validate_campaign(manifest)
        case_root = output / 'cases' / case
        gate = case_root / 'gate' / 'receipt.json'
        for path in (gate, *gate.parents):
            assert not path.is_symlink(), 'Gate ancestors must not be symlinks'
        receipt = read(gate)
        binding = campaign.binding(manifest)
        assert binding['commit'] == commit
        runtime = receipt['runtime']
        assert runtime['gpu'] in ('NVIDIA L40S', 'L40S'), 'Gate must use L40S'
        assert runtime['executable'] == PYTHON, 'Gate used another Python environment'
        protocol = read(gate.parent / 'protocol.json')
        assert protocol['slurm_job_id'] == smoke_job and protocol['binding'] == binding
        assert protocol['config'] == name and protocol['warmup_active_decisions'] == 60 and protocol['measured_decisions'] == 120
        cpus = receipt['recommended_cpus_within_5pct_of_best']
        campaign.validate_receipt(receipt, binding, runtime, cpus, name=name, operator=operator, interval=interval)
        paths = {str(gate.parent / f'cores-{count}.json') for count in (1, 2)}
        assert set(receipt['artifacts']) == paths, 'Gate artifact inventory changed'
        for path, checksum in receipt['artifacts'].items():
            assert not Path(path).is_symlink() and campaign.digest(path) == checksum, 'Gate artifact changed'
            original = read(path)
            assert original in receipt['cases'], 'Gate case differs from its immutable artifact'
        fastest = min(row['measured']['seconds_per_decision'] for row in receipt['cases'])
        selected = min(row['physical_cores'] for row in receipt['cases']
                       if row['measured']['seconds_per_decision'] <= fastest / .95)
        assert cpus == selected, 'CPU recommendation differs from measured threshold'
        row = next(row for row in receipt['cases'] if row['physical_cores'] == cpus)
        seconds_per_decision = row['measured']['seconds_per_decision']
        wall_hours = math.ceil((seconds_per_decision * decisions * 1.15 + 2 * 3600) / 3600)
        wall_seconds = wall_hours * 3600
        lineage = output / 'lineages' / (case + '-seed55')
        plans.append(dict(case=case, config=name, manifest=str(manifest.relative_to(ROOT)),
            source_commit=commit, binding=binding, smoke_job=smoke_job, gate=str(gate),
            gate_sha256=campaign.digest(gate), output=str(case_root), lineage=str(lineage),
            cpus=cpus, gpus=1, memory_gib=12, wall_hours=wall_hours, wall_seconds=wall_seconds,
            drain_seconds=wall_seconds-1800, seconds_per_decision=seconds_per_decision,
            decisions=decisions, margin=1.15, fixed_overhead_seconds=7200))
    assert len(plans) == 3
    return plans


def check_budget(plans):
    """Conservatively count current running AND pending reservations."""
    accounts = command(['sacctmgr', '-n', '-P', 'show', 'user', USER, 'format=DefaultAccount']).splitlines()
    accounts = {line.split('|')[0] for line in accounts if line.split('|')[0]}
    assert len(accounts) == 1, 'Cannot resolve one default account'
    account = accounts.pop()
    qos_rows = command(['sacctmgr', '-n', '-P', 'show', 'qos', QOS,
                        'format=Name,MaxTRESPU,MaxTRESRunMinsPU,MaxWall,MaxSubmitPU']).splitlines()
    matches = [line.split('|') for line in qos_rows if line.split('|')[0] == QOS]
    assert len(matches) == 1, 'QOS limit lookup failed'
    _, limit_counts, limit_minutes, max_wall, max_submit, *_ = matches[0]
    count_limits, minute_limits = tres(limit_counts), tres(limit_minutes)
    assert 'gpu' in count_limits and 'cpu' in count_limits
    assert 'gpu' in minute_limits and 'cpu' in minute_limits
    rows = command(['squeue', '-h', '-u', USER, '-o', '%i|%q|%a|%C|%l|%b|%T|%M|%m|%D'])
    current = dict(cpu=0, gpu=0, cpu_minutes=0., gpu_minutes=0., memory_mib=0., jobs=0)
    reservations = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        job, qos, selected_account, cpu, limit, gres, state, elapsed, memory, nodes = [part.strip() for part in line.split('|')]
        if qos != QOS:
            continue
        cpus, gpus = int(cpu), gpu_count(gres)
        # Running allocations reserve only their remaining time; pending jobs
        # reserve their full limit. Counting elapsed time again can block an
        # otherwise feasible launch near the end of existing experiments.
        remaining = time_seconds(limit)
        if state in ('RUNNING', 'COMPLETING', 'SUSPENDED'):
            remaining = max(0, remaining - time_seconds(elapsed))
        else:
            assert state in ('PENDING', 'CONFIGURING'), f'Unknown Slurm state {state}'
        minutes = remaining / 60
        current['cpu'] += cpus
        current['gpu'] += gpus
        current['cpu_minutes'] += cpus * minutes
        current['gpu_minutes'] += gpus * minutes
        current['memory_mib'] += memory_mib(memory, cpus=cpus, nodes=int(nodes))
        current['jobs'] += 1
        reservations.append(dict(job=job, account=selected_account, cpus=cpus, gpus=gpus,
                                 state=state, remaining_minutes=minutes))
    added = dict(cpu=sum(row['cpus'] for row in plans), gpu=len(plans), jobs=len(plans),
        cpu_minutes=sum(row['cpus'] * row['wall_seconds']/60 for row in plans),
        gpu_minutes=sum(row['wall_seconds']/60 for row in plans),
        memory_mib=sum(row['memory_gib'] * 1024 for row in plans))
    for resource in ('cpu', 'gpu'):
        assert current[resource] + added[resource] <= int(count_limits[resource]), f'Insufficient {resource} QOS headroom'
        assert current[resource+'_minutes'] + added[resource+'_minutes'] <= int(minute_limits[resource]), f'Insufficient {resource} time reservation headroom'
    if max_wall:
        assert all(row['wall_seconds'] <= time_seconds(max_wall) for row in plans), 'Plan exceeds QOS MaxWall'
    if max_submit:
        assert current['jobs'] + added['jobs'] <= int(max_submit), 'Plan exceeds QOS MaxSubmitPU'
    assert 'mem' in count_limits, 'Missing memory QOS limit'
    assert current['memory_mib'] + added['memory_mib'] <= memory_mib(count_limits['mem']), 'Insufficient memory QOS headroom'
    return dict(account=account, qos=QOS, current=current, added=added,
                count_limits=count_limits, minute_limits=minute_limits, reservations=reservations,
                checked_utc=datetime.now(timezone.utc).isoformat())


def submission(plan, account):
    seconds = plan['wall_seconds']
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    limit = f'{days}-{hours:02d}:00:00'
    env = dict(AMBI_PROJECT_DIR=str(ROOT), AMBI_PYTHON=PYTHON,
        AMBI_EXPECTED_COMMIT=plan['source_commit'], AMBI_RETURN_OUTPUT=plan['output'],
        AMBI_RETURN_MANIFEST=plan['manifest'], AMBI_RETURN_MODE='train',
        AMBI_RETURN_STORAGE='scratch-jh-scaling', AMBI_RETURN_GATE=plan['gate'],
        AMBI_LINEAGE_DIR=plan['lineage'], AMBI_DRAIN_AFTER_SECONDS=str(plan['drain_seconds']),
        AMBI_RESUME_CHECKPOINT_MINUTES='30', AMBI_RESUME_MODE='new')
    return ['sbatch', '--parsable', f'--chdir={ROOT}', '--nodes=1', '--ntasks=1',
        '--hint=nomultithread', '--partition=gpu', f'--qos={QOS}', f'--account={account}',
        '--gres=gpu:l40s:1', f"--cpus-per-task={plan['cpus']}", '--mem=12G',
        f'--time={limit}', '--signal=USR1@1800', '--requeue',
        f"--job-name=ambi-jh-{plan['case']}",
        f"--output={plan['output']}/train-%j.out", f"--error={plan['output']}/train-%j.err",
        '--export=ALL,' + ','.join(key+'='+value for key,value in env.items()),
        str(ROOT / 'slurm/run_ambi_reward_training_oscar.sbatch')]


def submit_workspace_watcher(output, commit, jobs, directory):
    """Start the bounded CPU publisher after every production GPU job starts."""
    runtime = output / 'publication-env/bin/python'
    workspace = output / 'publication/workspace.json'
    assert runtime.is_file() and workspace.is_file(), 'Prepared publication runtime or workspace is missing'
    watcher = [str(runtime), str(ROOT / 'slurm/watch_ambi_jh_scaling_workspace.py'),
        '--root', str(output), '--workspace', str(workspace), '--jobs-json', str(output / 'production-jobs.json'),
        '--commit', commit, '--timeout-seconds', '1500', '--poll-seconds', '30']
    arguments = ['sbatch', '--parsable', f'--chdir={ROOT}', '--partition=batch', '--qos=normal',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=2G', '--time=00:30:00', '--no-requeue',
        '--dependency=after:' + ':'.join(jobs[case.replace('-', '_')] for case in CASES),
        '--job-name=ambi-jh-workspace', f'--output={output}/publication/watcher-%j.out',
        f'--error={output}/publication/watcher-%j.err', '--wrap', shlex.join(watcher)]
    intent = directory / 'workspace-publisher-intent.json'
    receipt = directory / 'workspace-publisher.json'
    if receipt.exists():
        saved = read(receipt)
        assert saved['command'] == arguments and re.fullmatch(r'[1-9][0-9]*', saved['job_id'])
        assert intent.exists() and saved['intent_sha256'] == campaign.digest(intent)
        return saved
    assert not intent.exists(), 'Ambiguous workspace publisher intent; inspect Slurm before retry'
    durable_new(intent, dict(command=arguments, production_jobs=jobs, source_commit=commit,
                            created_utc=datetime.now(timezone.utc).isoformat()))
    response = command(arguments)
    assert re.fullmatch(r'[1-9][0-9]*(?:;[^\s;]+)?', response), 'Ambiguous publisher submission'
    saved = dict(job_id=response.split(';')[0], command=arguments, intent_sha256=campaign.digest(intent),
                 created_utc=datetime.now(timezone.utc).isoformat())
    durable_new(receipt, saved)
    return saved


def dispatch(output, commit, smoke_jobs, *, dry_run=False):
    assert output == OUTPUT and output.resolve() == OUTPUT, 'Unexpected campaign output root'
    assert re.fullmatch(r'[0-9a-f]{40}', commit)
    assert os.environ.get('SLURM_JOB_ID'), 'Dispatcher must run as a Slurm CPU job'
    assert command(['git', 'rev-parse', 'HEAD']) == commit, 'Source commit changed'
    assert not command(['git', 'status', '--porcelain', '--untracked-files=all']), 'Source is not clean'
    verify_smoke_jobs(smoke_jobs)
    plans = make_plans(output, commit, smoke_jobs)
    directory = output / 'production-submissions'
    directory.mkdir(exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending, complete = [], []
        for plan in plans:
            intent = directory / (plan['case'] + '-intent.json')
            receipt = directory / (plan['case'] + '.json')
            if receipt.exists():
                saved = read(receipt)
                assert saved['plan'] == plan and re.fullmatch(r'[1-9][0-9]*', saved['job_id']), 'Existing receipt mismatch'
                assert intent.exists() and saved['intent_sha256'] == campaign.digest(intent), 'Existing intent mismatch'
                complete.append(saved)
            else:
                assert not intent.exists(), f'Ambiguous submission intent for {plan["case"]}; inspect Slurm before any retry'
                assert not Path(plan['lineage']).exists(), 'Unsubmitted lineage already exists'
                pending.append(plan)
        budget = check_budget(pending) if pending else None
        if dry_run:
            return dict(dry_run=True, plans=plans, pending=[row['case'] for row in pending], budget=budget,
                        submitted=[row['job_id'] for row in complete])
        for plan in pending:
            arguments = submission(plan, budget['account'])
            intent = directory / (plan['case'] + '-intent.json')
            durable_new(intent, dict(plan=plan, command=arguments, budget=budget,
                dispatcher_job=os.environ['SLURM_JOB_ID'], created_utc=datetime.now(timezone.utc).isoformat()))
            answer = command(arguments)
            assert re.fullmatch(r'[1-9][0-9]*(?:;[^\s;]+)?', answer), 'Ambiguous sbatch response; reconcile intent through Slurm'
            job_id = answer.split(';')[0]
            saved = dict(plan=plan, job_id=job_id, command=arguments, intent_sha256=campaign.digest(intent),
                         created_utc=datetime.now(timezone.utc).isoformat())
            durable_new(directory / (plan['case'] + '.json'), saved)
            complete.append(saved)
        jobs = {row['plan']['case'].replace('-', '_'): row['job_id'] for row in complete}
        mapping = output / 'production-jobs.json'
        if mapping.exists():
            assert read(mapping) == jobs, 'Production job mapping changed'
        else:
            durable_new(mapping, jobs)
        publisher = submit_workspace_watcher(output, commit, jobs, directory)
        return dict(submitted=[row['job_id'] for row in complete], plans=plans, budget=budget,
                    workspace_publisher=publisher['job_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--smoke-jobs', nargs=3, required=True, metavar='JOBID',
                        help='Completed smoke jobs in H3/J12/K3, H4/J10/K4, H5/J10/K5 order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(dispatch(args.output, args.commit, args.smoke_jobs, dry_run=args.dry_run), indent=2))


if __name__ == '__main__':
    main()
