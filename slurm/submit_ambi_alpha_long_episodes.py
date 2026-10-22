"""Measure ten MPPI500k ten-episode histories and dispatch five hardware pairs.

Every selected GPU class must first pass both two-episode, eight-decision
sampled-action smokes using stream zero and the production interpreter. Five
independent pairs run concurrently; reset then carry share one physical GPU and
host per stream to avoid adding a hardware confound. Benchmarking is read-only;
dispatch journals each submission intent before contacting Slurm.
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
from slurm import ambi_alpha_long_episodes as driver
from slurm.submit_ambi_alpha_adaptation import (
    GPU_NAMES, digest, hardware_capture_code, number, read, require, save, validate_hardware,
)


MODES = ('episode_reset', 'run_carry')
PRODUCTION_EPISODES = 10
DECISIONS_PER_EPISODE = 500
SMOKE_EPISODES = 2
SMOKE_STEPS = 8
HISTORICAL_EPISODES = 3


def assignments(gpu_types):
    values = list(gpu_types) if gpu_types is not None else []
    require(len(values) == 5 and all(value in GPU_NAMES for value in values),
            'Supply exactly five supported, concrete GPU assignments')
    return values


def smoke_path(campaign, gpu_type, mode):
    require(mode in MODES, 'Unknown alpha execution mode')
    cell = campaign['cells'][MODES.index(mode) * 5]
    require(cell['mode'] == mode and cell['stream_index'] == 0, 'Wrong stream-zero smoke identity')
    return (Path(campaign['root']) / 'hardware' / gpu_type / 'smoke'
            / f"{cell['cell_id']}-steps8")


def observed_history_floor(campaign):
    """Read pinned production receipts so a fast smoke cannot hide slower hosts."""
    refs = campaign['checkpoints'][driver.CHECKPOINT_ID]
    pins = refs['observed_full_history_receipts']
    require(len(pins) == 5 and len({pin['path'] for pin in pins}) == 5,
            'Exactly five compatible full-history timing receipts are required')
    rows = []
    for stream, pin in enumerate(pins):
        require(Path(pin['path']).is_absolute() and digest(pin['path']) == pin['sha256'],
                'Historical timing receipt changed')
        receipt = read(pin['path'])
        validation = receipt.get('validation', {})
        require(receipt.get('validated') is True and receipt.get('smoke') is False
                and receipt.get('protocol') == 'run-temperature-chain-v1'
                and receipt.get('checkpoint_sha256') == campaign['cells'][0]['checkpoint_sha256']
                and receipt.get('stream_index') == stream and receipt.get('cpu_threads') == 1
                and validation.get('passed') is True and validation.get('episodes') == 3
                and validation.get('decisions') == 1500
                and validation.get('temperature_steps_per_decision') == 32,
                'Historical timing receipt is not a compatible full MPPI500k history')
        rows.append(dict(**pin, elapsed_seconds=number(receipt['elapsed_seconds'], 'historical elapsed time', positive=True)))
    return dict(seconds=max(row['elapsed_seconds'] for row in rows),
                episodes=HISTORICAL_EPISODES, receipts=rows)


def measured_resources(summary, receipt, *, observed_history):
    require(receipt['cpu_threads'] == 1, 'CPU sizing requires the one-CPU smoke')
    require(number(receipt['child_maxrss_kib'], 'host memory', positive=True) < 3 * 1024**2,
            'Measured host memory is too close to the 4 GiB allocation')
    rows = summary['rows']
    require(len(rows) == SMOKE_EPISODES * SMOKE_STEPS,
            'Two complete eight-decision smoke episodes are required')
    cold = [number(row['control_seconds'], 'cold control time', positive=True)
            for row in rows if row['decision_index'] == 0]
    warm = [number(row['control_seconds'], 'warm control time', positive=True)
            for row in rows if 0 < row['decision_index'] < SMOKE_STEPS]
    require(len(cold) == SMOKE_EPISODES and len(warm) == SMOKE_EPISODES * (SMOKE_STEPS - 1),
            'Two complete smoke episodes are required for cold/warm sizing')
    probe = number(receipt['validation']['probe_seconds'], 'probe time')
    overhead = max(0., number(receipt['child_elapsed_seconds'], 'elapsed time', positive=True)
                   - sum(cold + warm) - probe)
    # Ten full episodes have ten cold solves, 4990 warm solves and 50 probed
    # roots. The smoke has two cold solves and two probed roots. Trace buffers
    # flush each episode, so the 500-decision peak does not grow tenfold.
    warm_count = PRODUCTION_EPISODES * (DECISIONS_PER_EPISODE - 1)
    smoke_estimate = (statistics.mean(warm) * warm_count
                      + statistics.mean(cold) * PRODUCTION_EPISODES
                      + probe * (PRODUCTION_EPISODES * 5 / SMOKE_EPISODES)
                      + overhead * (PRODUCTION_EPISODES / SMOKE_EPISODES))
    require(observed_history['episodes'] == HISTORICAL_EPISODES,
            'Historical timing floor must contain three full episodes')
    source_floor = number(observed_history['seconds'], 'historical full-history floor', positive=True)
    floor = source_floor * PRODUCTION_EPISODES / HISTORICAL_EPISODES
    estimated = max(smoke_estimate, floor)
    return {'cpus': 1, 'memory_gib': 4, 'smoke_estimated_seconds': smoke_estimate,
            'production_episodes': PRODUCTION_EPISODES,
            'production_warm_decisions': warm_count,
            'production_cold_decisions': PRODUCTION_EPISODES,
            'production_probed_roots': PRODUCTION_EPISODES * 5,
            'warm_control_seconds_per_decision': statistics.mean(warm),
            'cold_control_seconds_per_episode': statistics.mean(cold),
            'smoke_overhead_seconds': overhead,
            'smoke_peak_host_rss_kib': receipt['child_maxrss_kib'],
            'observed_source_history_seconds': source_floor,
            'observed_source_history_episodes': HISTORICAL_EPISODES,
            'observed_full_history_floor_seconds': floor,
            'observed_full_history_receipts': observed_history['receipts'],
            'estimated_seconds': estimated,
            'walltime_minutes': math.ceil((estimated * 1.25 + 120) / 300) * 5}


def paired_resources(first, second, *, max_walltime_minutes=480):
    require(type(max_walltime_minutes) is int and 1 <= max_walltime_minutes <= 540,
            'Paired alpha-long-episodes walltime guard must be between 1 and 540 minutes')
    estimated = first['estimated_seconds'] + second['estimated_seconds']
    minutes = math.ceil((estimated * 1.25 + 120) / 300) * 5
    require(minutes <= max_walltime_minutes, 'Measured paired runtime exceeds explicit walltime guard')
    return dict(cpus=1, memory_gib=4, estimated_seconds=estimated, walltime_minutes=minutes,
                margin_fraction=.25, margin_seconds=120,
                pairing='episode_reset then run_carry in the same GPU allocation')


def paired_hardware_capture_code(gpu_type, root, indices):
    require(len(indices) == 2 and indices[1] == indices[0] + 5, 'Invalid paired hardware indices')
    first, second = [Path(root) / 'production-hardware' / str(index) for index in indices]
    return hardware_capture_code(gpu_type, first) + f'''path=Path({str(second)!r}+"-"+row["job_id"]+".json")
with path.open("x") as stream: json.dump(row,stream,indent=2,allow_nan=False)
'''


def benchmark(campaign_path, worker_repo, python_bin, gpu_types=None, max_walltime_minutes=480):
    require(type(max_walltime_minutes) is int and 1 <= max_walltime_minutes <= 540,
            'Paired alpha-long-episodes walltime guard must be between 1 and 540 minutes')
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
    historical = observed_history_floor(campaign)
    gates, rows = {}, []
    # Qualify both lifetimes on every selected GPU class before submitting any
    # job, even when a class is assigned to production cells of just one mode.
    for gpu_type in sorted(set(gpu_types)):
        for mode_index, mode in enumerate(MODES):
            cell = campaign['cells'][mode_index * 5]
            output = smoke_path(campaign, gpu_type, mode)
            require(output.resolve(strict=True).is_relative_to(Path(campaign['root'])), 'Smoke path escapes campaign')
            summary, receipt = driver.load_stream(campaign, cell, output, 8)
            hardware_path = output.with_name(output.name + '.hardware.json')
            hardware, started = read(hardware_path), read(output / 'started.json')
            validate_hardware(hardware, gpu_type)
            require(hardware['job_id'] == started['job_id'], 'Smoke hardware belongs to another allocation')
            require(started['command'][0] == str(python_bin), 'Smoke used a different Python environment')
            require(started['command'][1] == str(worker_repo / 'evaluate_ambi_checkpoint.py'),
                    'Smoke used a different source checkout')
            resources = measured_resources(summary, receipt, observed_history=historical)
            gates[gpu_type, mode] = dict(**resources, smoke_receipt=str(output / 'receipt.json'),
                smoke_receipt_sha256=digest(output / 'receipt.json'), hardware_path=str(hardware_path),
                hardware_sha256=digest(hardware_path))
    for cell in campaign['cells']:
        gpu_type = gpu_types[cell['stream_index']]
        rows.append(dict(index=cell['index'], cell_id=cell['cell_id'], mode=cell['mode'], gpu_type=gpu_type,
                         **gates[gpu_type, cell['mode']]))
    pairs = [dict(index=stream, indices=[stream, stream + 5], gpu_type=gpu_types[stream],
                  **paired_resources(rows[stream], rows[stream + 5], max_walltime_minutes=max_walltime_minutes))
             for stream in range(5)]
    return campaign, dict(cells=rows, pairs=pairs, gpu_types=gpu_types, independent_gpu_jobs=5,
                          total_cpu_requests=5, total_memory_gib=20,
                          qualified_smokes=[dict(gpu_type=gpu, mode=mode, **resources)
                                            for (gpu, mode), resources in gates.items()])



def dispatch(campaign_path, worker_repo, python_bin, gpu_types=None, max_walltime_minutes=480):
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

        def submit(kind, index, command, indices=None):
            entry = dict(kind=kind, index=index, status='submitting', command=command)
            if indices is not None:
                entry['indices'] = indices
            record['jobs'].append(entry)
            save(path, record)
            acknowledgement = subprocess.check_output(command, text=True).strip()
            require(re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?', acknowledgement), 'Unrecognized sbatch acknowledgement')
            entry.update(status='submitted', job_id=acknowledgement.split(';')[0])
            save(path, record)
            return entry['job_id']

        gpu_jobs = []
        for resource in sizing['pairs']:
            index, indices = resource['index'], resource['indices']
            minutes, gpu_type = resource['walltime_minutes'], resource['gpu_type']
            capture = ['srun', '/usr/bin/env', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                str(python_bin), '-c', paired_hardware_capture_code(gpu_type, root, indices)]
            workers = [['bash', str(worker_repo / 'slurm/run_ambi_alpha_long_episodes_oscar.sbatch'),
                        '--campaign', str(campaign_path), '--index', str(cell_index)] for cell_index in indices]
            # Try both cells even if one fails, preserving a failed exit status.
            body = ('set -eu\n' + shlex.join(capture) + '\npair_status=0\n'
                    + '\n'.join(shlex.join(worker) + ' || pair_status=$?' for worker in workers)
                    + '\nexit "$pair_status"')
            command = ['sbatch', '--parsable', f'--job-name=alpha-long-episodes-pair-{index}', '--partition=gpu',
                f'--gres=gpu:{gpu_type}:1', '--exclude=gpu2709', '--ntasks=1', '--cpus-per-task=1', '--mem=4G',
                f'--time={minutes // 60:02}:{minutes % 60:02}:00', '--chdir=' + str(worker_repo),
                '--output=' + str(root / 'slurm' / f'production-pair-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'production-pair-{index}-%j.err'),
                '--export=ALL,EXPECTED_ACTION_MODES_SHA=' + campaign['source_commit'] + ',PYTHON_BIN=' + str(python_bin),
                '--wrap', body]
            gpu_jobs.append(submit('production', index, command, indices=indices))
        publisher = ['/usr/bin/env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            'WANDB_MODE=online', 'WANDB__REQUIRE_CORE=true', 'WANDB_DIR=' + str(root),
            'WANDB_CACHE_DIR=' + str(root / 'cache'), 'srun', str(python_bin),
            'slurm/ambi_alpha_long_episodes_publish.py', 'publish', '--campaign', str(campaign_path),
            '--root', str(root / 'publication'), '--require-complete']
        checked_publisher = ('set -eu\n'
            + 'test "$(git rev-parse HEAD)" = ' + shlex.quote(campaign['source_commit']) + '\n'
            + 'test -z "$(git status --porcelain)"\nexec ' + shlex.join(publisher))
        command = ['sbatch', '--parsable', '--job-name=alpha-long-episodes-publish', '--partition=batch',
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
    parser.add_argument('--max-walltime-minutes', type=int, default=480,
                        help='Measured paired walltime guard; default 480, hard maximum 540 minutes')
    args = parser.parse_args()
    import json
    result = (benchmark if args.command == 'benchmark' else dispatch)(args.campaign, args.worker_repo,
        args.python_bin, args.gpu_types, args.max_walltime_minutes)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
