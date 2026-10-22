"""Dispatch four measured, bounded temperature-adaptation jobs after the CPU/GPU smoke gate.

Run in a short CPU allocation after the gate succeeds. Only the separately
pinned worker checkout executes scientific work; this helper uses stdlib only.
"""
from __future__ import annotations
import argparse
import csv
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
GPU_NAMES = {'l40s': 'L40S', 'quadro_rtx_6000': 'Quadro RTX 6000',
             'nvidia_rtx_a5000': 'RTX A5000', 'a5500': 'RTX A5500',
             'nvidia_geforce_rtx_3090': 'GeForce RTX 3090', 'nvidia_a40': 'A40',
             'nvidia_rtx_a6000': 'RTX A6000'}
BASELINE_PARITY_FAILURE = 'Fresh default smoke differs from historical baseline prefix'


def baseline_review_comparisons(current, historical):
    """Permit only sub-millinat entropy roundoff beyond the old prefix check."""
    keys = ('reward', 'alpha_initial', 'alpha_final', 'entropy')
    wanted = [(seed, decision) for seed in (101, 102) for decision in range(8)]
    require([(r['seed'], r['decision_index']) for r in current['rows']] == wanted,
            'Review requires the complete two-episode eight-decision baseline smoke')
    old = {(r['seed'], r['decision_index']): r for r in historical['rows']}
    failures, maxima = [], {key: 0. for key in keys}
    for row in current['rows']:
        reference = old[row['seed'], row['decision_index']]
        for key in keys:
            actual, expected = row[key], reference[key]
            require(type(actual) in (int, float) and type(expected) in (int, float)
                    and math.isfinite(actual) and math.isfinite(expected), 'Review comparison must be finite')
            difference = abs(actual - expected)
            maxima[key] = max(maxima[key], difference)
            if not math.isclose(actual, expected, rel_tol=2e-4, abs_tol=2e-5):
                require(key == 'entropy' and difference <= 1e-3,
                        'Review cannot waive reward, alpha, or larger entropy disagreement')
                failures.append(dict(seed=row['seed'], decision_index=row['decision_index'], metric=key,
                                     original=expected, observed=actual, absolute_difference=difference))
    require(bool(failures), 'Reviewed baseline must retain a real original-tolerance failure')
    return dict(comparisons=64, original_failure_count=len(failures), original_failures=failures,
                max_absolute_differences=maxima, original_relative_tolerance=2e-4,
                original_absolute_tolerance=2e-5, reviewed_entropy_absolute_tolerance=1e-3)


def validate_baseline_review(row, output, cell, gpu_type):
    output = Path(output)
    require(gpu_type == 'quadro_rtx_6000' and row.get('cross_gpu_baseline_review_passed') is True
            and row['validation'].get('baseline_prefix_matches') is False,
            'Reviewed parity is only permitted for the explicitly reviewed Quadro baseline')
    failure = read(output / 'failure.json')
    require(failure == {'exception_type': 'ValueError', 'message': BASELINE_PARITY_FAILURE},
            'Review cannot waive another worker failure')
    review = row['baseline_review']
    files = {'failure': output / 'failure.json', 'started': output / 'started.json',
             'decision_summary': output / 'decision-summary.json',
             'historical_summary': Path(cell['baseline_decision_summary']),
             'historical_receipt': Path(cell['baseline_decision_summary']).with_name('baseline-receipt.json'),
             'hardware': output.with_name(output.name + '.hardware.json')}
    require(review['files_sha256'] == {key: digest(path) for key, path in files.items()}, 'Baseline review evidence changed')
    historical_receipt = read(files['historical_receipt'])
    require(historical_receipt['decision_summary_sha256'] == digest(files['historical_summary'])
            and historical_receipt['baseline_manifest_sha256'] == cell['baseline_manifest_sha256']
            and historical_receipt['baseline_results_sha256'] == cell['baseline_results_sha256'],
            'Baseline review historical evidence differs from pinned inputs')
    comparisons = baseline_review_comparisons(read(files['decision_summary']), read(files['historical_summary']))
    require(review['comparisons'] == comparisons, 'Baseline review comparison evidence changed')
    for kind in ('reference', 'baseline'):
        bundle = Path(cell[kind + '_bundle'])
        for suffix, path in (('manifest', bundle / 'manifest.json'), ('results', bundle.parent / 'results.json')):
            key = kind + '_' + suffix + '_sha256'
            require(row[key] == cell[key] == digest(path), 'Baseline review reference changed')


def review_baseline(campaign_path, worker_repo, python_bin, cell, output, gpu_type):
    """Revalidate old-worker outputs in a CPU allocation; retain failure verbatim."""
    output = Path(output)
    require(gpu_type == 'quadro_rtx_6000' and not (output / 'receipt.json').exists(),
            'Only a failed Quadro baseline can receive an explicit review')
    require(read(output / 'failure.json') == {'exception_type': 'ValueError', 'message': BASELINE_PARITY_FAILURE},
            'Review cannot waive another worker failure')
    hardware = read(output.with_name(output.name + '.hardware.json'))
    validate_hardware(hardware, gpu_type)
    require(read(output / 'started.json')['job_id'] == hardware['job_id'], 'Review hardware job differs')
    # The pinned validator writes a compact summary. Give it an isolated output
    # directory with read-only links to the original files, then verify that its
    # regenerated summary is identical. No original failure/result is rewritten.
    program = '''import json,sys,tempfile
from pathlib import Path
from slurm import ambi_alpha_adaptation_episodes as d
campaign_path,output,index=sys.argv[1],Path(sys.argv[2]),int(sys.argv[3])
campaign=d.read(campaign_path);d.validate_campaign(campaign)
assert d.source()==campaign['source_commit']
cell=dict(campaign['cells'][index]);cell.update(d.intervention(cell['checkpoint_id'],'reset'))
d.verify_checkpoint(cell)
pins=d.reference_pins(cell);assert all(cell[k]==v for k,v in pins.items())
assert d.sha256(cell['inventory_path'])==cell['inventory_sha256']
historical=d.load_baseline_decision_summary(cell)
with tempfile.TemporaryDirectory(prefix='baseline-review-',dir=output.parent) as tmp:
    root=Path(tmp);(root/'results.json').symlink_to(output/'results.json');(root/'bundle').symlink_to(output/'bundle',target_is_directory=True)
    validation=d.validate_result(root,cell,8)
    assert d.read(root/'decision-summary.json')==d.read(output/'decision-summary.json')
print(json.dumps(dict(validation=validation,pins=pins,trace_sha256=d.trace_hashes(output/'bundle')),allow_nan=False))
'''
    audited = json.loads(subprocess.check_output([str(python_bin), '-c', program, str(campaign_path),
        str(output), str(cell['index'])], cwd=worker_repo, text=True))
    require(audited['validation']['passed'] is True and audited['validation']['lifecycle_validated'] is True,
            'Pinned-worker review validation failed')
    comparisons = baseline_review_comparisons(read(output / 'decision-summary.json'), read(cell['baseline_decision_summary']))
    files = {'failure': output / 'failure.json', 'started': output / 'started.json',
             'decision_summary': output / 'decision-summary.json', 'historical_summary': Path(cell['baseline_decision_summary']),
             'historical_receipt': Path(cell['baseline_decision_summary']).with_name('baseline-receipt.json'),
             'hardware': output.with_name(output.name + '.hardware.json')}
    validation = dict(audited['validation'], baseline_prefix_matches=False,
                      baseline_prefix_max_absolute_differences=comparisons['max_absolute_differences'])
    # Baseline performance comes from the actual validator. Baselines never size
    # production jobs, so no elapsed-time or RSS estimate is invented here.
    row = dict(validated=True, smoke=True, baseline=True, mode='reset',
        source_commit=read(campaign_path)['source_commit'], checkpoint_id=cell['checkpoint_id'],
        checkpoint_sha256=cell['checkpoint_sha256'], cell_id=cell['checkpoint_id'] + '__reset', cpu_threads=1,
        validation=validation, cross_gpu_baseline_review_passed=True,
        baseline_review=dict(version=1, reason='Cross-GPU baseline entropy roundoff only', comparisons=comparisons,
            files_sha256={key: digest(path) for key, path in files.items()}),
        results_sha256=digest(output / 'results.json'), manifest_sha256=digest(output / 'bundle/manifest.json'),
        decision_summary_sha256=digest(output / 'decision-summary.json'), trace_sha256=audited['trace_sha256'], **audited['pins'])
    validate_baseline_review(row, output, cell, gpu_type)
    with (output / 'reviewed-receipt.json').open('x') as stream:
        json.dump(row, stream, indent=2, allow_nan=False); stream.write('\n')
    return row


def validate_hardware(row, gpu_type):
    require(gpu_type in GPU_NAMES and row.get('gpu_type') == gpu_type
            and row.get('gpu_name', '').removeprefix('NVIDIA ') == GPU_NAMES[gpu_type],
            'GPU hardware differs from requested class')
    require(all(isinstance(row.get(key), str) and row[key] for key in ('node', 'job_id', 'nvidia_smi'))
            and isinstance(row.get('device_capability'), list) and len(row['device_capability']) == 2
            and all(type(x) is int and x >= 0 for x in row['device_capability']), 'Incomplete GPU provenance')
    devices = [[v.strip() for v in values] for values in csv.reader(row['nvidia_smi'].splitlines())]
    require(any(len(values) == 4 and values[0].removeprefix('NVIDIA ') == GPU_NAMES[gpu_type]
                and values[1] and values[2] and values[3] for values in devices), 'Missing GPU UUID or driver provenance')


def hardware_capture_code(gpu_type, output_prefix):
    """Python source run only inside the GPU allocation, before the pinned worker."""
    require(gpu_type in GPU_NAMES, 'Unsupported GPU type')
    return f'''import json,os,socket,subprocess
from pathlib import Path
import torch
name=torch.cuda.get_device_name(0)
assert name.removeprefix("NVIDIA ")=={GPU_NAMES[gpu_type]!r}, "Unexpected GPU class"
row=dict(gpu_type={gpu_type!r},node=socket.gethostname(),gpu_name=name,
device_capability=list(torch.cuda.get_device_capability(0)),
cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),job_id=os.environ["SLURM_JOB_ID"],
nvidia_smi=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader,nounits"],text=True))
path=Path({str(output_prefix)!r}+"-"+row["job_id"]+".json")
path.parent.mkdir(parents=True,exist_ok=True)
with path.open("x") as stream: json.dump(row,stream,indent=2,allow_nan=False)
'''


def benchmark(campaign_path, worker_repo, python_bin, gpu_type='l40s', smoke_root=None, max_walltime_minutes=180,
              allow_reviewed_cross_gpu_entropy=False):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    campaign, root = read(campaign_path), campaign_path.parent
    require(gpu_type in GPU_NAMES, 'Unsupported GPU type')
    require(type(max_walltime_minutes) is int and 1 <= max_walltime_minutes <= 360,
            'Maximum walltime must be an integer between 1 and 360 minutes')
    smoke_root = Path(smoke_root).resolve() if smoke_root is not None else root / 'smoke'
    require(smoke_root.is_relative_to(root), 'Smoke root escapes canonical campaign')
    if smoke_root != root / 'smoke':
        require(read(smoke_root.parent / 'campaign.json') == campaign, 'Alternate smoke campaign differs from canonical campaign')
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
        path = smoke_root / f'{cell_id}-steps{steps}-cpu{cpus}' / 'receipt.json'
        reviewed = False
        if not path.exists() and baseline and allow_reviewed_cross_gpu_entropy:
            require(gpu_type == 'quadro_rtx_6000', 'Entropy review is restricted to Quadro RTX 6000')
            path = path.with_name('reviewed-receipt.json')
            if not path.exists():
                review_baseline(campaign_path, worker_repo, python_bin, cell, path.parent, gpu_type)
            reviewed = True
        resolved = path.resolve(strict=True)
        require(resolved.is_relative_to(smoke_root) and resolved not in paths, 'Smoke receipt aliases or escapes campaign')
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
        if reviewed:
            validate_baseline_review(row, path.parent, cell, gpu_type)
        else:
            require(not baseline or validation.get('baseline_prefix_matches') is True, 'Historical baseline parity failed')
        for field, relative in (('results_sha256', 'results.json'), ('manifest_sha256', 'bundle/manifest.json'),
                                ('decision_summary_sha256', 'decision-summary.json')):
            require(row[field] == digest(path.parent / relative), 'Smoke output changed')
        require(bool(row['trace_sha256']), 'Smoke trace hashes are missing')
        for name, expected in row['trace_sha256'].items():
            trace = (path.parent / 'bundle' / name).resolve(strict=True)
            require(trace.is_relative_to((path.parent / 'bundle').resolve()) and digest(trace) == expected,
                    'Smoke trace changed or escaped bundle')
        provenance = {}
        hardware_path = path.parent.with_name(path.parent.name + '.hardware.json')
        if gpu_type != 'l40s' or hardware_path.exists():
            hardware = read(hardware_path)
            validate_hardware(hardware, gpu_type)
            require(read(path.parent / 'started.json').get('job_id') == hardware['job_id'],
                    'GPU provenance belongs to another smoke job')
            provenance = dict(hardware=hardware, hardware_path=str(hardware_path), hardware_sha256=digest(hardware_path))
        rows.append(dict(row, receipt_path=str(resolved), receipt_sha256=digest(resolved), **provenance))
    if gpu_type != 'l40s':
        require(all(rows[2]['hardware'][key] == rows[3]['hardware'][key]
                    for key in ('node', 'job_id', 'cuda_visible_devices', 'nvidia_smi')),
                'CPU scaling smokes must use the same GPU allocation')
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
        require(minutes <= max_walltime_minutes,
                f'Measured production time exceeds the configured {max_walltime_minutes}-minute guard; inspect before submitting')
        peak_kib = number(row['child_maxrss_kib'], 'peak child RSS', True)
        require(peak_kib * 1.5 + 512 * 1024 <= 4 * 1024 * 1024, 'Measured memory cannot safely fit 4 GiB')
        cells.append(dict(index=index, cpus=repeat_cpus if index in (0, 2) else 1,
            memory_gib=4, walltime_minutes=minutes, estimated_seconds=estimate,
            measurement_receipt=row['receipt_path'], control_seconds_per_decision=rate))
    return campaign, dict(cells=cells, cpu2_throughput_ratio=ratio, gpu_type=gpu_type, smoke_root=str(smoke_root),
        max_walltime_minutes=max_walltime_minutes,
        margin='25 percent plus two minutes, rounded to five minutes', measurements=rows)


def dispatch(campaign_path, worker_repo, python_bin, gpu_type='l40s', smoke_root=None, max_walltime_minutes=180,
             allow_reviewed_cross_gpu_entropy=False):
    campaign_path, worker_repo = Path(campaign_path).resolve(), Path(worker_repo).resolve()
    root, path = campaign_path.parent, campaign_path.parent / 'dispatch.json'
    require(gpu_type in GPU_NAMES, 'Unsupported GPU type')
    require(type(max_walltime_minutes) is int and 1 <= max_walltime_minutes <= 360,
            'Maximum walltime must be an integer between 1 and 360 minutes')
    smoke_root = Path(smoke_root).resolve() if smoke_root is not None else root / 'smoke'
    with (root / '.dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign_hash = digest(campaign_path)
        if path.exists():
            existing = read(path)
            require(existing['campaign_sha256'] == campaign_hash and existing['worker_repo'] == str(worker_repo)
                    and existing['python_bin'] == str(python_bin)
                    and existing.get('gpu_type', 'l40s') == gpu_type
                    and existing.get('smoke_root', str(root / 'smoke')) == str(smoke_root)
                    and existing.get('max_walltime_minutes', 180) == max_walltime_minutes
                    and existing.get('allow_reviewed_cross_gpu_entropy', False) is allow_reviewed_cross_gpu_entropy,
                    'Existing dispatch targets differ')
            require(existing['status'] == 'complete', 'Dispatch was interrupted; reconcile saved intents before retrying')
            return existing
        campaign, sizing = benchmark(campaign_path, worker_repo, python_bin, gpu_type, smoke_root, max_walltime_minutes,
                                     allow_reviewed_cross_gpu_entropy)
        record = dict(status='dispatching', campaign_sha256=campaign_hash, worker_repo=str(worker_repo),
            python_bin=str(python_bin), source_commit=campaign['source_commit'], resources=sizing,
            gpu_type=gpu_type, smoke_root=str(smoke_root),
            max_walltime_minutes=max_walltime_minutes,
            allow_reviewed_cross_gpu_entropy=allow_reviewed_cross_gpu_entropy,
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
            capture = ['srun', 'env', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                       str(python_bin), '-c', hardware_capture_code(gpu_type, root / 'production-hardware' / str(index))]
            worker = ['bash', str(worker_repo / 'slurm/run_ambi_alpha_adaptation_episodes_oscar.sbatch'),
                      '--campaign', str(campaign_path), '--index', str(index), '--cpu-threads', str(resource['cpus'])]
            command = ['sbatch', '--parsable', f'--job-name=alpha-adapt-{index}', '--partition=gpu',
                f'--gres=gpu:{gpu_type}:1', '--exclude=gpu2709', '--ntasks=1', f"--cpus-per-task={resource['cpus']}", '--mem=4G',
                '--time=' + walltime, '--chdir=' + str(worker_repo),
                '--output=' + str(root / 'slurm' / f'production-{index}-%j.out'),
                '--error=' + str(root / 'slurm' / f'production-{index}-%j.err'),
                '--export=ALL,EXPECTED_ACTION_MODES_SHA=' + campaign['source_commit'] + ',PYTHON_BIN=' + str(python_bin),
                '--wrap', 'set -eu\n' + shlex.join(capture) + '\nexec ' + shlex.join(worker)]
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
    parser.add_argument('--gpu-type', choices=tuple(GPU_NAMES), default='l40s')
    parser.add_argument('--smoke-root', type=Path)
    parser.add_argument('--max-walltime-minutes', type=int, default=180)
    parser.add_argument('--allow-reviewed-cross-gpu-entropy', action='store_true')
    args = parser.parse_args()
    print(json.dumps(dispatch(args.campaign, args.worker_repo, args.python_bin, args.gpu_type, args.smoke_root,
                              args.max_walltime_minutes, args.allow_reviewed_cross_gpu_entropy), indent=2))


if __name__ == '__main__':
    main()
