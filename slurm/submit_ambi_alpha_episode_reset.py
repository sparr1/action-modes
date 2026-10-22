"""Measure and dispatch five MPPI500k histories that reset alpha between episodes.

Every selected GPU class must first pass the same two-episode, eight-decision
smoke using stream zero and the production interpreter. Benchmarking is read-only;
dispatch journals each submission intent before contacting Slurm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm import ambi_alpha_episode_reset as driver
from slurm.submit_ambi_alpha_adaptation import (
    GPU_NAMES, digest, hardware_capture_code, read, require, save, validate_hardware,
)
from slurm.submit_ambi_alpha_carry_chain import measured_resources


def assignments(gpu_types):
    values = list(gpu_types) if gpu_types is not None else []
    require(len(values) == 5 and all(value in GPU_NAMES for value in values),
            'Supply exactly five supported, concrete GPU assignments')
    return values


def smoke_path(campaign, gpu_type):
    return (Path(campaign['root']) / 'hardware' / gpu_type / 'smoke'
            / f"{campaign['cells'][0]['cell_id']}-steps8")


def benchmark(campaign_path, worker_repo, python_bin, gpu_types=None, max_walltime_minutes=90):
    require(type(max_walltime_minutes) is int and 1 <= max_walltime_minutes <= 90,
            'Episode-reset walltime guard must be between 1 and 90 minutes')
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    campaign = read(campaign_path)
    driver.validate_campaign(campaign)
    require(campaign_path.parent == Path(campaign['root']), 'Use the canonical campaign path')
    commit = subprocess.check_output(['git', '-C', str(worker_repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == campaign['source_commit'] and re.fullmatch(r'[0-9a-f]{40}', commit), 'Wrong worker commit')
    require(not subprocess.check_output(['git', '-C', str(worker_repo), 'status', '--porcelain'], text=True).strip(),
            'Worker checkout is dirty')
    require(Path(python_bin).is_absolute() and Path(python_bin).is_file(), 'Use the absolute locked Python path')
    for key, source in [('matrix', driver.MATRIX), ('profile', driver.PROFILE)]:
        require(digest(worker_repo / 'configs/research' / Path(source).name) == campaign[key + '_sha256'],
                'Worker configuration changed')
    gpu_types = assignments(gpu_types)
    gates, rows = {}, []
    for cell, gpu_type in zip(campaign['cells'], gpu_types):
        if gpu_type not in gates:
            output = smoke_path(campaign, gpu_type)
            require(output.resolve(strict=True).is_relative_to(Path(campaign['root'])), 'Smoke path escapes campaign')
            summary, receipt = driver.load_stream(campaign, campaign['cells'][0], output, 8)
            hardware_path = output.with_name(output.name + '.hardware.json')
            hardware, started = read(hardware_path), read(output / 'started.json')
            validate_hardware(hardware, gpu_type)
            require(hardware['job_id'] == started['job_id'], 'Smoke hardware belongs to another allocation')
            require(started['command'][0] == str(python_bin), 'Smoke used a different Python environment')
            require(started['command'][1] == str(worker_repo / 'evaluate_ambi_checkpoint.py'),
                    'Smoke used a different source checkout')
            resources = measured_resources(summary, receipt, max_walltime_minutes=max_walltime_minutes)
            gates[gpu_type] = dict(**resources, smoke_receipt=str(output / 'receipt.json'),
                smoke_receipt_sha256=digest(output / 'receipt.json'), hardware_path=str(hardware_path),
                hardware_sha256=digest(hardware_path))
        rows.append(dict(index=cell['index'], cell_id=cell['cell_id'], gpu_type=gpu_type, **gates[gpu_type]))
    return campaign, dict(cells=rows, gpu_types=gpu_types, independent_gpu_jobs=5,
                          total_cpu_requests=5, total_memory_gib=20)


def dispatch(campaign_path, worker_repo, python_bin, gpu_types=None, max_walltime_minutes=90):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    root, gpu_types = campaign_path.parent, assignments(gpu_types)
    path = root / 'dispatch.json'
    with (root / '.dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(campaign_sha256=digest(campaign_path), worker_repo=str(worker_repo),
                        python_bin=str(python_bin), gpu_types=gpu_types, max_walltime_minutes=max_walltime_minutes)
        if path.exists():
            old = read(path)
            require(all(old.get(key) == value for key, value in identity.items()), 'Existing dispatch targets differ')
            require(old['status'] == 'complete', 'Dispatch was interrupted; reconcile saved intents before retrying')
            return old
        campaign, sizing = benchmark(campaign_path, worker_repo, python_bin, gpu_types, max_walltime_minutes)
        require(all(not Path(cell['output_dir']).exists() for cell in campaign['cells']),
                'Production output already exists; reconcile before dispatch')
        workspace = read(root / 'publication/workspace.json')
        require(workspace.get('api_verified') is True
                and workspace.get('campaign_sha256') == driver.canonical_hash(campaign),
                'Install and verify the matching publication workspace before dispatch')
        record = dict(**identity, status='dispatching', source_commit=campaign['source_commit'], resources=sizing,
                      created_at=datetime.now(timezone.utc).isoformat(), jobs=[])
        (root / 'slurm').mkdir(exist_ok=True)
        save(path, record)

        def submit(kind, index, command):
            entry = dict(kind=kind, index=index, status='submitting', command=command)
            record['jobs'].append(entry)
            save(path, record)
            acknowledgement = subprocess.check_output(command, text=True).strip()
            require(re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', acknowledgement), 'Unrecognized sbatch acknowledgement')
            entry.update(status='submitted', job_id=acknowledgement.split(';')[0])
            save(path, record)
            return entry['job_id']

        gpu_jobs = []
        for resource in sizing['cells']:
            index, minutes, gpu_type = resource['index'], resource['walltime_minutes'], resource['gpu_type']
            capture = ['srun', '/usr/bin/env', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                str(python_bin), '-c', hardware_capture_code(gpu_type, root / 'production-hardware' / str(index))]
            worker = ['bash', str(worker_repo / 'slurm/run_ambi_alpha_episode_reset_oscar.sbatch'),
                      '--campaign', str(campaign_path), '--index', str(index)]
            command = ['sbatch', '--parsable', f'--job-name=alpha-ep-reset-{index}', '--partition=gpu',
                f'--gres=gpu:{gpu_type}:1', '--exclude=gpu2709', '--ntasks=1', '--cpus-per-task=1', '--mem=4G',
                f'--time={minutes // 60:02}:{minutes % 60:02}:00', '--chdir=' + str(worker_repo),
                '--output=' + str(root / 'slurm' / f'production-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'production-{index}-%j.err'),
                '--export=ALL,EXPECTED_ACTION_MODES_SHA=' + campaign['source_commit'] + ',PYTHON_BIN=' + str(python_bin),
                '--wrap', 'set -eu\n' + shlex.join(capture) + '\nexec ' + shlex.join(worker)]
            gpu_jobs.append(submit('production', index, command))
        publisher = ['/usr/bin/env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            'WANDB_MODE=online', 'WANDB__REQUIRE_CORE=true', 'WANDB_DIR=' + str(root),
            'WANDB_CACHE_DIR=' + str(root / 'cache'), 'srun', str(python_bin),
            'slurm/ambi_alpha_episode_reset_publish.py', 'publish', '--campaign', str(campaign_path),
            '--root', str(root / 'publication'), '--require-complete']
        checked_publisher = ('set -eu\n'
            + 'test "$(git rev-parse HEAD)" = ' + shlex.quote(campaign['source_commit']) + '\n'
            + 'test -z "$(git status --porcelain)"\nexec ' + shlex.join(publisher))
        command = ['sbatch', '--parsable', '--job-name=alpha-ep-reset-publish', '--partition=batch',
            '--ntasks=1', '--cpus-per-task=1', '--mem=2G', '--time=00:05:00', '--chdir=' + str(worker_repo),
            '--dependency=afterany:' + ':'.join(gpu_jobs),
            '--output=' + str(root / 'slurm' / 'publish-%j.out'),
            '--error=' + str(root / 'slurm' / 'publish-%j.err'), '--wrap', checked_publisher]
        publisher_id = submit('publisher', 0, command)
        record.update(status='complete', production_job_ids=gpu_jobs, final_publisher_job_id=publisher_id)
        save(path, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('benchmark', 'dispatch'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--worker-repo', type=Path, required=True)
    parser.add_argument('--python-bin', required=True)
    parser.add_argument('--gpu-types', nargs=5, choices=tuple(GPU_NAMES), required=True)
    parser.add_argument('--max-walltime-minutes', type=int, default=90)
    args = parser.parse_args()
    import json
    result = (benchmark if args.command == 'benchmark' else dispatch)(args.campaign, args.worker_repo,
        args.python_bin, args.gpu_types, args.max_walltime_minutes)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
