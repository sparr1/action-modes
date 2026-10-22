"""Gate ten independent alpha chains on measured GPU smoke tests, then dispatch.

Run this helper in a short CPU allocation after smoke completion. Every selected
GPU class needs a complete two-episode chain smoke for each checkpoint assigned
to that class. No GPU workload or publication is started by ``benchmark``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm import ambi_alpha_carry_chain as driver
from slurm.submit_ambi_alpha_adaptation import (GPU_NAMES, digest, hardware_capture_code, number,
                                              read, require, save, validate_hardware)


def assignments(gpu_types):
    values = list(gpu_types) if gpu_types is not None else ['quadro_rtx_6000'] * 10
    require(len(values) == 10 and all(v in GPU_NAMES for v in values), 'Supply exactly ten supported GPU assignments')
    return values


def smoke_path(campaign, checkpoint_id, gpu_type, steps=8):
    return Path(campaign['root']) / 'hardware' / gpu_type / 'smoke' / f'{checkpoint_id}__stream0-steps{steps}'


def measured_resources(summary, receipt, *, max_walltime_minutes=90):
    require(type(max_walltime_minutes) is int and 1 <= max_walltime_minutes <= 180, 'Invalid walltime guard')
    require(receipt['cpu_threads'] == 1, 'CPU sizing requires the one-CPU smoke')
    require(number(receipt['child_maxrss_kib'], 'host memory', positive=True) < 3 * 1024**2,
            'Measured host memory is too close to the 4 GiB allocation')
    rows = summary['rows']
    cold = [number(r['control_seconds'], 'cold control time', positive=True) for r in rows if r['decision_index'] == 0]
    warm = [number(r['control_seconds'], 'warm control time', positive=True) for r in rows if r['decision_index'] > 0]
    require(len(cold) == 2 and len(warm) >= 6, 'Two complete smoke episodes are required for cold/warm sizing')
    control = sum(cold + warm)
    probe = number(receipt['validation']['probe_seconds'], 'probe time')
    overhead = max(0., number(receipt['child_elapsed_seconds'], 'elapsed time', positive=True) - control - probe)
    # Short smoke contains 2 cold solves and 2 probed roots. Production has 3
    # cold solves and 15 probed roots, not 1500 repetitions of cold-smoke mean.
    estimated = statistics.mean(warm) * 1497 + statistics.mean(cold) * 3 + probe * 7.5 + overhead * 1.5
    minutes = math.ceil((estimated * 1.25 + 120) / 300) * 5
    require(minutes <= max_walltime_minutes, 'Measured runtime exceeds explicit walltime guard')
    return dict(cpus=1, memory_gib=4, estimated_seconds=estimated, walltime_minutes=minutes,
                warm_control_seconds_per_decision=statistics.mean(warm),
                cold_control_seconds_per_episode=statistics.mean(cold), smoke_overhead_seconds=overhead,
                smoke_peak_host_rss_kib=receipt['child_maxrss_kib'], margin_fraction=.25, margin_seconds=120)


def benchmark(campaign_path, worker_repo, python_bin, gpu_types=None, max_walltime_minutes=90):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    campaign = read(campaign_path)
    driver.validate_campaign(campaign)
    require(campaign_path.parent == Path(campaign['root']), 'Use the canonical campaign path')
    commit = subprocess.check_output(['git', '-C', str(worker_repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == campaign['source_commit'] and re.fullmatch(r'[0-9a-f]{40}', commit), 'Wrong worker commit')
    require(not subprocess.check_output(['git', '-C', str(worker_repo), 'status', '--porcelain'], text=True).strip(),
            'Worker checkout is dirty')
    require(Path(python_bin).is_absolute() and Path(python_bin).is_file(), 'Use the absolute locked Python path')
    for key, filename in [('matrix', 'ambi_alpha_carry_chain.json'), ('profile', 'ambi_entropy_probe.json')]:
        require(digest(worker_repo / 'configs/research' / filename) == campaign[key + '_sha256'], 'Worker configuration changed')
    gpu_types = assignments(gpu_types)
    gates, rows = {}, []
    for cell, gpu_type in zip(campaign['cells'], gpu_types):
        key = cell['checkpoint_id'], gpu_type
        if key not in gates:
            gate_cell = campaign['cells'][0 if cell['checkpoint_id'] == driver.CHECKPOINTS[0] else 5]
            output = smoke_path(campaign, cell['checkpoint_id'], gpu_type)
            require(output.resolve(strict=True).is_relative_to(Path(campaign['root'])), 'Smoke path escapes campaign')
            summary, receipt = driver.load_stream(campaign, gate_cell, output, 8)
            hardware_path = output.with_name(output.name + '.hardware.json')
            hardware = read(hardware_path)
            validate_hardware(hardware, gpu_type)
            require(hardware['job_id'] == read(output / 'started.json')['job_id'], 'Smoke hardware belongs to another allocation')
            resources = measured_resources(summary, receipt, max_walltime_minutes=max_walltime_minutes)
            gates[key] = dict(**resources, smoke_receipt=str(output / 'receipt.json'),
                             smoke_receipt_sha256=digest(output / 'receipt.json'),
                             hardware_path=str(hardware_path), hardware_sha256=digest(hardware_path))
        rows.append(dict(index=cell['index'], cell_id=cell['cell_id'], gpu_type=gpu_type, **gates[key]))
    return campaign, dict(cells=rows, gpu_types=gpu_types, independent_gpu_jobs=10,
                          total_cpu_requests=10, total_memory_gib=40)


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
            require(all(old.get(k) == v for k, v in identity.items()), 'Existing dispatch targets differ')
            require(old['status'] == 'complete', 'Dispatch was interrupted; reconcile saved intents before retrying')
            return old
        campaign, sizing = benchmark(campaign_path, worker_repo, python_bin, gpu_types, max_walltime_minutes)
        workspace = read(root / 'publication/workspace.json')
        require(workspace.get('api_verified') is True and workspace.get('campaign_sha256') == driver.canonical_hash(campaign),
                'Install and verify the matching publication workspace before dispatch')
        record = dict(**identity, status='dispatching', source_commit=campaign['source_commit'], resources=sizing,
                      created_at=datetime.now(timezone.utc).isoformat(), jobs=[])
        (root / 'slurm').mkdir(exist_ok=True)
        save(path, record)

        def submit(kind, index, command):
            entry = dict(kind=kind, index=index, status='submitting', command=command)
            record['jobs'].append(entry); save(path, record)
            acknowledgement = subprocess.check_output(command, text=True).strip()
            require(re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', acknowledgement), 'Unrecognized sbatch acknowledgement')
            entry.update(status='submitted', job_id=acknowledgement.split(';')[0]); save(path, record)
            return entry['job_id']

        gpu_jobs = []
        for resource in sizing['cells']:
            index, minutes, gpu_type = resource['index'], resource['walltime_minutes'], resource['gpu_type']
            capture = ['srun', '/usr/bin/env', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                str(python_bin), '-c', hardware_capture_code(gpu_type, root / 'production-hardware' / str(index))]
            worker = ['bash', str(worker_repo / 'slurm/run_ambi_alpha_carry_chain_oscar.sbatch'),
                      '--campaign', str(campaign_path), '--index', str(index)]
            command = ['sbatch', '--parsable', f'--job-name=alpha-chain-{index}', '--partition=gpu',
                f'--gres=gpu:{gpu_type}:1', '--exclude=gpu2709', '--ntasks=1', '--cpus-per-task=1', '--mem=4G',
                f'--time={minutes // 60:02}:{minutes % 60:02}:00', '--chdir=' + str(worker_repo),
                '--output=' + str(root / 'slurm' / f'production-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'production-{index}-%j.err'),
                '--export=ALL,EXPECTED_ACTION_MODES_SHA=' + campaign['source_commit'] + ',PYTHON_BIN=' + str(python_bin),
                '--wrap', 'set -eu\n' + shlex.join(capture) + '\nexec ' + shlex.join(worker)]
            gpu_jobs.append(submit('production', index, command))
        prior_publisher = None
        # Two serialized CPU merges, each after all five streams of a checkpoint.
        # The second rechecks the first and is the complete-campaign publication.
        for index in range(2):
            dependencies = gpu_jobs[index * 5:(index + 1) * 5] + ([prior_publisher] if prior_publisher else [])
            worker = ['/usr/bin/env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                'WANDB_MODE=online', 'WANDB__REQUIRE_CORE=true', 'WANDB_DIR=' + str(root),
                'WANDB_CACHE_DIR=' + str(root / 'cache'), 'srun', str(python_bin),
                'slurm/ambi_alpha_carry_chain_publish.py', 'publish', '--campaign', str(campaign_path),
                '--root', str(root / 'publication')]
            if index == 1:
                worker.append('--require-complete')
            command = ['sbatch', '--parsable', f'--job-name=alpha-chain-publish-{index}', '--partition=batch',
                '--cpus-per-task=1', '--mem=2G', '--time=00:05:00', '--chdir=' + str(worker_repo),
                '--dependency=afterany:' + ':'.join(dependencies),
                '--output=' + str(root / 'slurm' / f'publish-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'publish-{index}-%j.err'), '--wrap', shlex.join(worker)]
            prior_publisher = submit('publisher', index, command)
        record.update(status='complete', production_job_ids=gpu_jobs, final_publisher_job_id=prior_publisher)
        save(path, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('benchmark', 'dispatch'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--worker-repo', type=Path, required=True)
    parser.add_argument('--python-bin', required=True)
    parser.add_argument('--gpu-types', nargs=10, choices=tuple(GPU_NAMES))
    parser.add_argument('--max-walltime-minutes', type=int, default=90)
    args = parser.parse_args()
    import json
    result = (benchmark if args.command == 'benchmark' else dispatch)(args.campaign, args.worker_repo,
        args.python_bin, args.gpu_types, args.max_walltime_minutes)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
