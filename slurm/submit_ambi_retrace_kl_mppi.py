"""Dispatch four measured, bounded Retrace jobs after the CPU/GPU smoke gate.

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


def benchmark(campaign_path, worker_repo, python_bin):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    campaign, root = read(campaign_path), campaign_path.parent
    require(campaign['campaign_id'] == 'retrace-kl-mppi-20261021' and len(campaign['cells']) == 4
            and campaign['seeds'] == [101, 102, 103, 104, 105]
            and campaign['max_steps'] == 500 and campaign['execution_action'] == 'sample', 'Unexpected campaign')
    require([(c['index'], c['step'], c['mode']) for c in campaign['cells']] ==
            [(i, step, mode) for i, (step, mode) in enumerate(
                [(400000, 'retrace_kl'), (400000, 'retrace_both'), (500000, 'retrace_kl'), (500000, 'retrace_both')])],
            'Checkpoint or method ordering changed')
    for key in ('run_id', 'run_dir', 'output_dir', 'cell_id'):
        require(len({c[key] for c in campaign['cells']}) == 4, 'Aliased production cell: ' + key)
    commit = subprocess.check_output(['git', '-C', str(worker_repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == campaign['source_commit'] and re.fullmatch(r'[0-9a-f]{40}', commit), 'Wrong worker commit')
    require(not subprocess.check_output(['git', '-C', str(worker_repo), 'status', '--porcelain'], text=True).strip(),
            'Worker checkout is dirty')
    require(Path(python_bin).is_absolute() and Path(python_bin).is_file(), 'Use the absolute locked Python path')
    for kind in ('matrix', 'profile'):
        suffix = '' if kind == 'matrix' else '_profile'
        require(digest(worker_repo / f'configs/research/ambi_retrace_kl_mppi{suffix}.json') == campaign[kind + '_sha256'],
                'Worker configuration differs from campaign')
    rows, paths = [], []
    for index, steps, cpus in ((0, 32, 1), (0, 32, 2), (3, 8, 1)):
        cell = campaign['cells'][index]
        path = root / 'smoke' / f"{cell['cell_id']}-steps{steps}-cpu{cpus}" / 'receipt.json'
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(root) and resolved not in paths, 'Smoke receipt aliases or escapes campaign')
        paths.append(resolved)
        row, validation = read(path), read(path)['validation']
        require(row.get('validated') is True and row.get('smoke') is True and validation.get('passed') is True
                and row['source_commit'] == commit and row['cell_id'] == cell['cell_id']
                and row['checkpoint_sha256'] == cell['checkpoint_sha256'] and row['cpu_threads'] == cpus
                and validation['decisions'] == steps and validation['episodes'] == 1 and validation['seeds'] == [101]
                and validation['selector'] == cell['selector'] and validation['execution_action'] == 'sample',
                'Missing, incomplete or mismatched smoke gate receipt')
        require(row['results_sha256'] == digest(path.parent / 'results.json')
                and row['manifest_sha256'] == digest(path.parent / 'bundle/manifest.json'), 'Smoke output changed')
        require(bool(row['trace_sha256']), 'Smoke trace hashes are missing')
        for name, expected in row['trace_sha256'].items():
            trace = (path.parent / 'bundle' / name).resolve(strict=True)
            require(trace.is_relative_to((path.parent / 'bundle').resolve()) and digest(trace) == expected,
                    'Smoke trace changed or escaped bundle')
        rows.append(dict(row, receipt_path=str(resolved), receipt_sha256=digest(resolved)))
    rates = [number(r['validation']['control_seconds_per_decision'], 'controller time', True) for r in rows]
    cpus = 2 if rates[0] / rates[1] > 1.05 else 1
    # Each smoke has one probed decision. Full five-seed episodes have 25.
    probe = max(number(r['validation']['probe_seconds'], 'probe time') for r in rows)
    startup = max(max(number(r['elapsed_seconds'], 'elapsed time')
        - number(r['validation']['control_seconds'], 'control time')
        - number(r['validation']['probe_seconds'], 'probe time'),
        number(r['validation']['initialization_seconds'], 'initialization time')) for r in rows)
    estimate = max(rates[cpus - 1], rates[2]) * 2500 + startup + 25 * probe
    minutes = math.ceil((estimate * 1.2 + 120) / 300) * 5
    memory = max(4, math.ceil(max(number(r['child_maxrss_kib'], 'peak child RSS', True) for r in rows)
                            * 1.5 / (1024 * 1024) + 1))
    require(minutes <= 240, 'Measured production time exceeds the four-hour guard; inspect before submitting')
    require(memory <= 8, 'Measured memory exceeds 8 GiB; inspect before submitting')
    return campaign, dict(cpus=cpus, memory_gib=memory, walltime_minutes=minutes,
        estimated_seconds=estimate, margin='20 percent plus two minutes, rounded to five minutes',
        cpu2_throughput_ratio=rates[0] / rates[1], measurements=rows)


def dispatch(campaign_path, worker_repo, python_bin, previous_publisher=None):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    require(previous_publisher is None or re.fullmatch(r'[0-9]+', str(previous_publisher)), 'Invalid previous publisher ID')
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
            # A lost acknowledgement remains an unresolved intent; never blindly resubmit.
            result = subprocess.check_output(command, text=True).strip()
            require(re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', result), 'Unrecognized sbatch acknowledgement')
            entry.update(status='submitted', job_id=result.split(';')[0]); save(path, record)
            return entry['job_id']

        minutes = sizing['walltime_minutes']
        walltime = f'{minutes // 60:02}:{minutes % 60:02}:00'
        gpu_jobs = []
        for index in range(4):
            command = ['sbatch', '--parsable', f'--job-name=retrace-kl-{index}', '--partition=gpu',
                '--gres=gpu:l40s:1', '--exclude=gpu2709', f"--cpus-per-task={sizing['cpus']}",
                f"--mem={sizing['memory_gib']}G", '--time=' + walltime, '--chdir=' + str(worker_repo),
                '--output=' + str(root / 'slurm' / f'production-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'production-{index}-%j.err'),
                '--export=ALL,EXPECTED_ACTION_MODES_SHA=' + campaign['source_commit'] + ',PYTHON_BIN=' + str(python_bin)
                + ',RETRACE_CPU_GATE=0', str(worker_repo / 'slurm/run_ambi_retrace_kl_mppi_oscar.sbatch'),
                '--campaign', str(campaign_path), '--index', str(index)]
            gpu_jobs.append(submit('production', index, command))
        for index, gpu_job in enumerate(gpu_jobs):
            dependencies = [gpu_job] + ([str(previous_publisher)] if previous_publisher else [])
            worker = ['env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                'WANDB_MODE=online', 'WANDB__REQUIRE_CORE=true', 'WANDB_DIR=' + str(root),
                'WANDB_CACHE_DIR=' + str(root / 'cache'), 'srun', str(python_bin),
                'slurm/ambi_retrace_kl_mppi.py', 'publish', '--campaign', str(campaign_path)]
            if index == 3:
                worker.append('--require-complete')
            command = ['sbatch', '--parsable', f'--job-name=retrace-publish-{index}', '--partition=batch',
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
    parser.add_argument('--previous-publisher')
    args = parser.parse_args()
    print(json.dumps(dispatch(args.campaign, args.worker_repo, args.python_bin, args.previous_publisher), indent=2))


if __name__ == '__main__':
    main()
