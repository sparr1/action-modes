"""Dispatch four measured, bounded temperature-adaptation jobs after the CPU/GPU smoke gate.

Run in a short CPU allocation after the gate succeeds. Only the separately
pinned worker checkout executes scientific work; this helper uses stdlib only.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
from datetime import datetime, timezone


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def number(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'Invalid measurement: ' + label)
    return float(value)


GATES = ((0, 8, 1, True), (1, 8, 1, False), (0, 32, 1, False),
         (0, 32, 2, False), (2, 8, 1, True), (3, 8, 1, False), (2, 32, 1, False))


def benchmark(campaign_path, worker_repo, python_bin):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    campaign, root = read(campaign_path), campaign_path.parent
    workspace = read(root / 'publication/workspace.json')
    campaign_identity = hashlib.sha256(json.dumps(campaign, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(workspace.get('api_verified') is True and workspace.get('campaign_sha256') == campaign_identity,
            'Install and verify the matching publication workspace before dispatch')
    require(campaign['campaign_id'] == 'alpha-adaptation-episodes-20261021' and len(campaign['cells']) == 4
            and campaign['seeds'] == [101, 102, 103, 104, 105] and campaign['controller_seed'] == 55
            and campaign['max_steps'] == 500 and campaign['execution_action'] == 'sample', 'Unexpected campaign')
    require([(c['index'], c['checkpoint_id'], c['mode']) for c in campaign['cells']] ==
            [(i, name, mode) for i, (name, mode) in enumerate(
                [(name, mode) for name in ('sac_aux_100k', 'mppi_aux_400k') for mode in ('repeat100', 'carry')])],
            'Checkpoint or method ordering changed')
    for key in ('run_id', 'run_dir', 'output_dir', 'cell_id'):
        require(len({c[key] for c in campaign['cells']}) == 4, 'Aliased production cell: ' + key)
    commit = subprocess.check_output(['git', '-C', str(worker_repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == campaign['source_commit'] and re.fullmatch(r'[0-9a-f]{40}', commit), 'Wrong worker commit')
    require(not subprocess.check_output(['git', '-C', str(worker_repo), 'status', '--porcelain'], text=True).strip(),
            'Worker checkout is dirty')
    require(Path(python_bin).is_absolute() and Path(python_bin).is_file(), 'Use the absolute locked Python path')
    for kind, filename in (('matrix', 'ambi_alpha_adaptation_episodes.json'), ('profile', 'ambi_entropy_probe.json')):
        require(digest(worker_repo / 'configs/research' / filename) == campaign[kind + '_sha256'],
                'Worker configuration differs from campaign')
    rows, paths = [], set()
    for index, steps, cpus, baseline in GATES:
        cell = campaign['cells'][index]
        mode = 'reset' if baseline else cell['mode']
        cell_id = cell['checkpoint_id'] + '__' + mode
        path = root / 'smoke' / f'{cell_id}-steps{steps}-cpu{cpus}' / 'receipt.json'
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(root) and resolved not in paths, 'Smoke receipt aliases or escapes campaign')
        paths.add(resolved)
        row = read(path); validation = row['validation']
        require(row.get('validated') is True and row.get('smoke') is True and row.get('baseline') is baseline
                and validation.get('passed') is True and validation.get('lifecycle_validated') is True
                and row['source_commit'] == commit and row['cell_id'] == cell_id and row['mode'] == mode
                and row['checkpoint_sha256'] == cell['checkpoint_sha256'] and row['cpu_threads'] == cpus
                and validation['decisions'] == steps * 2 and validation['episodes'] == 2 and validation['seeds'] == [101, 102]
                and validation['selector'] == f'alpha_adaptation/{mode}' and validation['execution_action'] == 'sample'
                and validation['temperature_steps_per_decision'] == (3200 if mode == 'repeat100' else 32),
                'Missing, incomplete or mismatched smoke gate receipt')
        require(not baseline or validation.get('baseline_prefix_matches') is True, 'Historical baseline parity failed')
        for field, relative in (('results_sha256', 'results.json'), ('manifest_sha256', 'bundle/manifest.json'),
                                ('decision_summary_sha256', 'decision-summary.json')):
            require(row[field] == digest(path.parent / relative), 'Smoke output changed')
        require(bool(row['trace_sha256']), 'Smoke trace hashes are missing')
        for name, expected in row['trace_sha256'].items():
            trace = (path.parent / 'bundle' / name).resolve(strict=True)
            require(trace.is_relative_to((path.parent / 'bundle').resolve()) and digest(trace) == expected,
                    'Smoke trace changed or escaped bundle')
        rows.append(dict(row, receipt_path=str(resolved), receipt_sha256=digest(resolved)))
    rates = [number(r['validation']['control_seconds_per_decision'], 'controller time', True) for r in rows]
    ratio = rates[2] / rates[3]
    repeat_cpus = 2 if ratio > 1.1 else 1
    # A measured two-CPU benefit at SAC100k selects repeat workers only. MPPI's
    # walltime conservatively retains its measured one-CPU rate, without assuming
    # the same speedup. Carry workers have only a one-CPU measurement.
    measurements = [rows[3 if repeat_cpus == 2 else 2], rows[1], rows[6], rows[5]]
    cells = []
    for index, row in enumerate(measurements):
        validation = row['validation']
        rate = number(validation['control_seconds_per_decision'], 'controller time', True)
        probe_per_root = number(validation['probe_seconds'], 'probe time') / 2
        startup = max(number(row['elapsed_seconds'], 'elapsed time')
            - number(validation['control_seconds'], 'control time')
            - number(validation['probe_seconds'], 'probe time'),
            number(validation['initialization_seconds'], 'initialization time'))
        estimate = rate * 2500 + startup + 25 * probe_per_root
        minutes = math.ceil((estimate * 1.25 + 120) / 300) * 5
        require(minutes <= 180, 'Measured production time exceeds the three-hour guard; inspect before submitting')
        peak_kib = number(row['child_maxrss_kib'], 'peak child RSS', True)
        require(peak_kib * 1.5 + 512 * 1024 <= 4 * 1024 * 1024, 'Measured memory cannot safely fit 4 GiB')
        cells.append(dict(index=index, cpus=repeat_cpus if index in (0, 2) else 1,
            memory_gib=4, walltime_minutes=minutes, estimated_seconds=estimate,
            measurement_receipt=row['receipt_path'], control_seconds_per_decision=rate))
    return campaign, dict(cells=cells, cpu2_throughput_ratio=ratio,
        margin='25 percent plus two minutes, rounded to five minutes', measurements=rows)


def dispatch(campaign_path, worker_repo, python_bin):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    root, path = campaign_path.parent, campaign_path.parent / 'dispatch.json'
    with (root / '.dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign_hash = digest(campaign_path)
        if path.exists():
            existing = read(path)
            require(existing['campaign_sha256'] == campaign_hash and existing['worker_repo'] == str(worker_repo)
                    and existing['python_bin'] == str(python_bin), 'Existing dispatch targets differ')
            require(existing['status'] == 'complete', 'Dispatch was interrupted; reconcile saved intents before retrying')
            return existing
        campaign, sizing = benchmark(campaign_path, worker_repo, python_bin)
        record = dict(status='dispatching', campaign_sha256=campaign_hash, worker_repo=str(worker_repo),
            python_bin=str(python_bin), source_commit=campaign['source_commit'], resources=sizing,
            created_at=datetime.now(timezone.utc).isoformat(), jobs=[])
        (root / 'slurm').mkdir(exist_ok=True)
        save(path, record)

        def submit(kind, index, command):
            entry = dict(kind=kind, index=index, status='submitting', command=command)
            record['jobs'].append(entry); save(path, record)
            result = subprocess.check_output(command, text=True).strip()
            require(re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', result), 'Unrecognized sbatch acknowledgement')
            entry.update(status='submitted', job_id=result.split(';')[0]); save(path, record)
            return entry['job_id']

        gpu_jobs = []
        for resource in sizing['cells']:
            index, minutes = resource['index'], resource['walltime_minutes']
            walltime = f'{minutes // 60:02}:{minutes % 60:02}:00'
            command = ['sbatch', '--parsable', f'--job-name=alpha-adapt-{index}', '--partition=gpu',
                '--gres=gpu:l40s:1', '--exclude=gpu2709', f"--cpus-per-task={resource['cpus']}", '--mem=4G',
                '--time=' + walltime, '--chdir=' + str(worker_repo),
                '--output=' + str(root / 'slurm' / f'production-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'production-{index}-%j.err'),
                '--export=ALL,EXPECTED_ACTION_MODES_SHA=' + campaign['source_commit'] + ',PYTHON_BIN=' + str(python_bin),
                str(worker_repo / 'slurm/run_ambi_alpha_adaptation_episodes_oscar.sbatch'),
                '--campaign', str(campaign_path), '--index', str(index), '--cpu-threads', str(resource['cpus'])]
            gpu_jobs.append(submit('production', index, command))
        previous_publisher = None
        for index, gpu_job in enumerate(gpu_jobs):
            dependencies = [gpu_job] + ([previous_publisher] if previous_publisher else [])
            worker = ['env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                'WANDB_MODE=online', 'WANDB__REQUIRE_CORE=true', 'WANDB_DIR=' + str(root),
                'WANDB_CACHE_DIR=' + str(root / 'cache'), 'srun', str(python_bin),
                'slurm/ambi_alpha_adaptation_publish.py', 'publish', '--campaign', str(campaign_path),
                '--root', str(root / 'publication')]
            if index == 3:
                worker.append('--require-complete')
            command = ['sbatch', '--parsable', f'--job-name=alpha-adapt-publish-{index}', '--partition=batch',
                '--cpus-per-task=1', '--mem=2G', '--time=00:05:00', '--chdir=' + str(worker_repo),
                '--dependency=afterany:' + ':'.join(dependencies),
                '--output=' + str(root / 'slurm' / f'publish-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'publish-{index}-%j.err'), '--wrap', shlex.join(worker)]
            previous_publisher = submit('publisher', index, command)
        record.update(status='complete', production_job_ids=gpu_jobs, final_publisher_job_id=previous_publisher)
        save(path, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--worker-repo', required=True, type=Path)
    parser.add_argument('--python-bin', required=True)
    args = parser.parse_args()
    print(json.dumps(dispatch(args.campaign, args.worker_repo, args.python_bin), indent=2))


if __name__ == '__main__':
    main()
