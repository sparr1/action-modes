"""Qualified hardware, measured resources and submission ownership for five hardware-matched pairs of sampled ten-episode histories."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from slurm import submit_ambi_alpha_long_episodes as scheduler


DEFAULT_GPUS = ['nvidia_rtx_a5000'] * 5
HISTORICAL_SECONDS = 3317.042612573132
SCALED_HISTORY_SECONDS = HISTORICAL_SECONDS * 10 / 3


def fixture(tmp_path, monkeypatch, gpu_types=None):
    gpu_types = DEFAULT_GPUS if gpu_types is None else gpu_types
    root, repo = tmp_path / 'campaign with spaces', tmp_path / 'worker with spaces'
    root.mkdir()
    (repo / 'configs/research').mkdir(parents=True)
    python = tmp_path / 'locked python'
    python.touch()
    for name in (Path(scheduler.driver.MATRIX).name, Path(scheduler.driver.PROFILE).name):
        (repo / 'configs/research' / name).write_text('{}')
    campaign = dict(root=str(root), source_commit='a' * 40, cells=[],
                    matrix_sha256=scheduler.digest(repo / 'configs/research' / Path(scheduler.driver.MATRIX).name),
                    profile_sha256=scheduler.digest(repo / 'configs/research' / Path(scheduler.driver.PROFILE).name))
    for index in range(10):
        mode, stream = scheduler.MODES[index // 5], index % 5
        cell_id = f'mppi_aux_500k__{mode}__stream{stream}'
        campaign['cells'].append(dict(index=index, stream_index=stream, mode=mode, cell_id=cell_id,
                                     output_dir=str(root / 'production' / cell_id), checkpoint_sha256='c' * 64))
    pins = []
    for stream in range(5):
        historical = root / f'historical-receipt-{stream}.json'
        historical.write_text(json.dumps(dict(validated=True, smoke=False, protocol='run-temperature-chain-v1',
            checkpoint_sha256='c' * 64, stream_index=stream, cpu_threads=1,
            elapsed_seconds=HISTORICAL_SECONDS - (4 - stream),
            validation=dict(passed=True, episodes=3, decisions=1500, temperature_steps_per_decision=32))))
        pins.append(dict(path=str(historical), sha256=scheduler.digest(historical)))
    campaign['checkpoints'] = {scheduler.driver.CHECKPOINT_ID: {'observed_full_history_receipts': pins}}
    # Scientific validation has separate full-artifact tests. These tests
    # isolate resource gates and submission ownership after campaign acceptance.
    monkeypatch.setattr(scheduler.driver, 'validate_campaign', lambda _: None)
    gates = {}
    for gpu_type in set(scheduler.assignments(gpu_types)):
        for mode in scheduler.MODES:
            output = scheduler.smoke_path(campaign, gpu_type, mode)
            output.mkdir(parents=True)
            rows = [dict(decision_index=decision, control_seconds=30. if decision == 0 else 1.5)
                    for _ in range(2) for decision in range(8)]
            receipt = dict(cpu_threads=1, child_maxrss_kib=2 * 1024**2, child_elapsed_seconds=111.,
                           validation=dict(probe_seconds=10.))
            (output / 'receipt.json').write_text(json.dumps(receipt))
            (output / 'started.json').write_text(json.dumps(dict(job_id='123',
                command=[str(python), str(repo / 'evaluate_ambi_checkpoint.py')])))
            hardware = dict(gpu_type=gpu_type, node='gpu0001', gpu_name='NVIDIA ' + scheduler.GPU_NAMES[gpu_type],
                device_capability=[7, 5], cuda_visible_devices='0', job_id='123',
                nvidia_smi=scheduler.GPU_NAMES[gpu_type] + ', GPU-uuid, 550.54, 24576')
            output.with_name(output.name + '.hardware.json').write_text(json.dumps(hardware))
            gates[gpu_type, mode] = dict(summary=dict(rows=rows), output=output)

    def load_stream(_campaign, cell, output, smoke_steps):
        assert smoke_steps == 8 and cell['stream_index'] == 0
        gate = next(gate for gate in gates.values() if gate['output'] == output)
        return gate['summary'], scheduler.read(output / 'receipt.json')

    monkeypatch.setattr(scheduler.driver, 'load_stream', load_stream)
    path = root / 'campaign.json'
    path.write_text(json.dumps(campaign))
    (root / 'publication').mkdir()
    (root / 'publication/workspace.json').write_text(json.dumps(
        dict(api_verified=True, campaign_sha256=scheduler.driver.canonical_hash(campaign))))
    commands = []

    def call(args, **kwargs):
        if args[0] == 'git':
            return 'a' * 40 + '\n' if 'rev-parse' in args else ''
        assert args[0] == 'sbatch'
        commands.append(args)
        return f'{1000 + len(commands)};oscar\n'

    monkeypatch.setattr(scheduler.subprocess, 'check_output', call)
    return path, repo, str(python), commands, gates


def test_benchmark_floors_short_smoke_with_full_history_and_has_no_submission(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    _, sizing = scheduler.benchmark(path, repo, python, DEFAULT_GPUS)
    assert not commands
    assert sizing['independent_gpu_jobs'] == sizing['total_cpu_requests'] == 5
    assert sizing['total_memory_gib'] == 20
    assert len(sizing['cells']) == 10 and len(sizing['pairs']) == 5
    for row in sizing['cells']:
        assert row['smoke_estimated_seconds'] == pytest.approx(4990 * 1.5 + 10 * 30 + 50 * 5 + 20 * 5)
        assert row['estimated_seconds'] == SCALED_HISTORY_SECONDS
        assert row['observed_full_history_floor_seconds'] == SCALED_HISTORY_SECONDS
        assert row['observed_source_history_seconds'] == HISTORICAL_SECONDS
        assert row['observed_source_history_episodes'] == 3
        assert row['production_episodes'] == 10
        assert row['production_warm_decisions'] == 4990
        assert row['production_cold_decisions'] == 10
        assert row['production_probed_roots'] == 50
        assert row['walltime_minutes'] == 235
        assert row['cpus'] == 1 and row['memory_gib'] == 4
    for pair in sizing['pairs']:
        assert pair['estimated_seconds'] == 2 * SCALED_HISTORY_SECONDS
        assert pair['walltime_minutes'] == 465
        assert pair['indices'] == [pair['index'], pair['index'] + 5]


def test_five_parallel_pairs_one_afterany_publisher_and_idempotent_retry(tmp_path, monkeypatch):
    gpu_types = ['nvidia_rtx_a5000'] * 5
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch, gpu_types)
    result = scheduler.dispatch(path, repo, python, gpu_types)
    assert len(gates) == 2 and len(commands) == 6 and result['status'] == 'complete'
    for index, command in enumerate(commands[:5]):
        assert '--gres=gpu:nvidia_rtx_a5000:1' in command
        assert '--cpus-per-task=1' in command and '--mem=4G' in command and '--time=07:45:00' in command
        assert not any(value.startswith('--dependency=') for value in command)
        assert '--exclude=gpu2709' in command
        assert shlex.quote(python) in command[-1]
        assert shlex.quote(str(repo / 'slurm/run_ambi_alpha_long_episodes_oscar.sbatch')) in command[-1]
        assert f'--index {index}' in command[-1] and f'--index {index + 5}' in command[-1]
        assert command[-1].index(f'--index {index} ') < command[-1].index(f'--index {index + 5} ')
        assert result['jobs'][index]['indices'] == [index, index + 5]
        assert command[-1].count('get_device_name') == 1
        subprocess.run(['bash', '-n'], input=command[-1], text=True, check=True)
    publisher = commands[5]
    assert '--dependency=afterany:' + ':'.join(str(index) for index in range(1001, 1006)) in publisher
    assert '--mem=2G' in publisher and '--cpus-per-task=1' in publisher and '--time=00:05:00' in publisher
    assert 'ambi_alpha_long_episodes_publish.py publish' in publisher[-1]
    assert '--require-complete' in publisher[-1] and 'git status --porcelain' in publisher[-1]
    subprocess.run(['bash', '-n'], input=publisher[-1], text=True, check=True)
    assert scheduler.dispatch(path, repo, python, gpu_types) == result and len(commands) == 6


def test_mixed_gpu_classes_each_require_their_own_stream_zero_smoke(tmp_path, monkeypatch):
    assignment = ['nvidia_rtx_a5000', 'quadro_rtx_6000', 'l40s', 'nvidia_rtx_a5000', 'quadro_rtx_6000']
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch, assignment)
    assert len(gates) == 6
    result = scheduler.dispatch(path, repo, python, assignment)
    assert result['gpu_types'] == assignment
    assert all('--gres=gpu:' + gpu + ':1' in command for gpu, command in zip(assignment, commands[:5]))


@pytest.mark.parametrize('change', ['missing', 'model', 'job', 'memory', 'slow', 'cpu', 'python', 'source'])
def test_invalid_smoke_blocks_all_submissions(tmp_path, monkeypatch, change):
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch)
    gate = next(iter(gates.values()))
    output, summary = gate['output'], gate['summary']
    if change == 'missing':
        (output / 'receipt.json').unlink()
    elif change in ('model', 'job'):
        hardware_path = output.with_name(output.name + '.hardware.json')
        hardware = scheduler.read(hardware_path)
        hardware['gpu_name' if change == 'model' else 'job_id'] = 'wrong'
        hardware_path.write_text(json.dumps(hardware))
    elif change in ('python', 'source'):
        started = scheduler.read(output / 'started.json')
        started['command'][0 if change == 'python' else 1] = '/different/path'
        (output / 'started.json').write_text(json.dumps(started))
    elif change == 'slow':
        for row in summary['rows']:
            row['control_seconds'] = 10.
    else:
        receipt = scheduler.read(output / 'receipt.json')
        receipt['child_maxrss_kib' if change == 'memory' else 'cpu_threads'] = 3 * 1024**2 if change == 'memory' else 2
        (output / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    assert not commands and not (path.parent / 'dispatch.json').exists()


@pytest.mark.parametrize('change', ['unverified', 'wrong_campaign', 'existing_output'])
def test_unverified_workspace_or_existing_production_cannot_launch(tmp_path, monkeypatch, change):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    if change == 'existing_output':
        Path(scheduler.read(path)['cells'][2]['output_dir']).mkdir(parents=True)
    else:
        workspace = path.parent / 'publication/workspace.json'
        value = scheduler.read(workspace)
        value['api_verified' if change == 'unverified' else 'campaign_sha256'] = False if change == 'unverified' else 'wrong'
        workspace.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    assert not commands


def test_partial_submission_keeps_intent_and_blocks_duplicate_retry(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    original = scheduler.subprocess.check_output

    def fail(args, **kwargs):
        if args[0] == 'sbatch' and len(commands) == 2:
            commands.append(args)
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)

    monkeypatch.setattr(scheduler.subprocess, 'check_output', fail)
    with pytest.raises(subprocess.CalledProcessError):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    assert scheduler.read(path.parent / 'dispatch.json')['jobs'][2]['status'] == 'submitting'
    with pytest.raises(ValueError, match='reconcile saved intents'):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    assert len(commands) == 3


def test_unknown_generic_or_wrong_count_assignments_rejected():
    for values in (None, ['gpu'] * 5, ['quadro_rtx_6000'] * 10, ['v100'] * 5, []):
        with pytest.raises(ValueError):
            scheduler.assignments(values)


@pytest.mark.parametrize('change', ['commit', 'dirty', 'config'])
def test_wrong_or_changed_source_blocks_submission(tmp_path, monkeypatch, change):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    original = scheduler.subprocess.check_output
    if change == 'config':
        (repo / 'configs/research' / Path(scheduler.driver.MATRIX).name).write_text('{"changed":true}')
    else:
        def altered(args, **kwargs):
            if args[0] == 'git':
                if change == 'commit' and 'rev-parse' in args:
                    return 'b' * 40 + '\n'
                if change == 'dirty' and 'status' in args:
                    return ' M source.py\n'
            return original(args, **kwargs)
        monkeypatch.setattr(scheduler.subprocess, 'check_output', altered)
    with pytest.raises(ValueError):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    assert not commands


def test_unrecognized_acknowledgement_is_not_retried(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    original = scheduler.subprocess.check_output
    def ambiguous(args, **kwargs):
        return 'submitted but malformed acknowledgement' if args[0] == 'sbatch' else original(args, **kwargs)
    monkeypatch.setattr(scheduler.subprocess, 'check_output', ambiguous)
    with pytest.raises(ValueError, match='acknowledgement'):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    assert scheduler.read(path.parent / 'dispatch.json')['jobs'][0]['status'] == 'submitting'
    with pytest.raises(ValueError, match='reconcile saved intents'):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)


@pytest.mark.parametrize('minutes', [0, -1, 541, 10080, True, 55.5])
def test_walltime_guard_cannot_exceed_authorized_nine_hours(tmp_path, monkeypatch, minutes):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='between 1 and 540'):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS, max_walltime_minutes=minutes)
    assert not commands and not (path.parent / 'dispatch.json').exists()


def test_completed_dispatch_cannot_silently_reassign_hardware(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    with pytest.raises(ValueError, match='targets differ'):
        scheduler.dispatch(path, repo, python, ['quadro_rtx_6000'] * 5)
    assert len(commands) == 6


def test_pure_helpers_do_not_retarget_existing_campaign_schedulers():
    from slurm import submit_ambi_alpha_preservation as preservation
    from slurm import submit_ambi_alpha_carry_chain as carry
    assert scheduler.measured_resources is not carry.measured_resources
    assert scheduler.driver.__name__ == 'slurm.ambi_alpha_long_episodes'
    assert preservation.driver.__name__ == 'slurm.ambi_alpha_preservation'
    assert carry.driver.__name__ == 'slurm.ambi_alpha_carry_chain'


def test_both_modes_must_pass_on_every_gpu_class_before_any_submission(tmp_path, monkeypatch):
    assignment = ['nvidia_rtx_a5000', 'quadro_rtx_6000', 'l40s', 'nvidia_rtx_a5000', 'quadro_rtx_6000']
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch, assignment)
    (gates['quadro_rtx_6000', 'run_carry']['output'] / 'receipt.json').unlink()
    with pytest.raises(FileNotFoundError):
        scheduler.dispatch(path, repo, python, assignment)
    assert not commands and not (path.parent / 'dispatch.json').exists()


@pytest.mark.parametrize('change', ['hash', 'short', 'cpu', 'checkpoint', 'nonfinite'])
def test_historical_timing_floor_requires_pinned_compatible_receipts(tmp_path, monkeypatch, change):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    campaign = scheduler.read(path)
    pin = campaign['checkpoints'][scheduler.driver.CHECKPOINT_ID]['observed_full_history_receipts'][4]
    receipt = scheduler.read(pin['path'])
    if change == 'short':
        receipt['validation']['decisions'] = 16
    elif change == 'cpu':
        receipt['cpu_threads'] = 2
    elif change == 'checkpoint':
        receipt['checkpoint_sha256'] = 'wrong'
    elif change == 'nonfinite':
        receipt['elapsed_seconds'] = float('nan')
    else:
        receipt['elapsed_seconds'] = 100.
    Path(pin['path']).write_text(json.dumps(receipt))
    if change != 'hash':
        pin['sha256'] = scheduler.digest(pin['path'])
        path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    assert not commands


def test_pair_guard_applies_after_summing_two_compatible_histories(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='paired runtime'):
        scheduler.dispatch(path, repo, python, DEFAULT_GPUS, max_walltime_minutes=460)
    assert not commands


def test_slower_smoke_takes_precedence_over_historical_floor(tmp_path, monkeypatch):
    path, repo, python, _, gates = fixture(tmp_path, monkeypatch)
    for gate in gates.values():
        for row in gate['summary']['rows']:
            if row['decision_index']:
                row['control_seconds'] = 2.3
    with pytest.raises(ValueError, match='paired runtime'):
        scheduler.benchmark(path, repo, python, DEFAULT_GPUS)
    _, sizing = scheduler.benchmark(path, repo, python, DEFAULT_GPUS, max_walltime_minutes=540)
    assert all(row['estimated_seconds'] > SCALED_HISTORY_SECONDS for row in sizing['cells'])
    assert all(row['estimated_seconds'] == row['smoke_estimated_seconds'] for row in sizing['cells'])
    assert all(row['walltime_minutes'] == 505 for row in sizing['pairs'])


def test_pair_hardware_records_are_from_one_capture(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '777')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        get_device_name=lambda _: 'NVIDIA RTX A5000', get_device_capability=lambda _: (8, 6))))
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'NVIDIA RTX A5000, uuid-one, 550, 24576')
    exec(scheduler.paired_hardware_capture_code('nvidia_rtx_a5000', tmp_path, [2, 7]), {})
    first, second = [tmp_path / 'production-hardware' / f'{index}-777.json' for index in (2, 7)]
    assert first.read_bytes() == second.read_bytes()
    scheduler.validate_hardware(scheduler.read(first), 'nvidia_rtx_a5000')


def test_pair_still_attempts_second_cell_and_preserves_failed_exit_status(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    scheduler.dispatch(path, repo, python, DEFAULT_GPUS)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    fake_srun = bindir / 'srun'
    fake_srun.write_text('#!/bin/sh\nexit 0\n')
    fake_srun.chmod(0o755)
    worker = repo / 'slurm/run_ambi_alpha_long_episodes_oscar.sbatch'
    worker.parent.mkdir()
    log = tmp_path / 'attempts'
    worker.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> ' + shlex.quote(str(log))
                      + '\nif [[ "$*" == *"--index 0" ]]; then exit 7; fi\n')
    result = subprocess.run(['bash', '-c', commands[0][-1]],
                            env={**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH']})
    assert result.returncode == 7
    assert len(log.read_text().splitlines()) == 2
    assert '--index 0' in log.read_text() and '--index 5' in log.read_text()
