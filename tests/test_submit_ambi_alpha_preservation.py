"""Qualified hardware, measured resources and submission ownership for five chains."""
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from slurm import submit_ambi_alpha_preservation as scheduler


def fixture(tmp_path, monkeypatch, gpu_types=None):
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
    for index in range(5):
        cell_id = f'mppi_aux_500k__stream{index}'
        campaign['cells'].append(dict(index=index, stream_index=index, cell_id=cell_id,
                                     output_dir=str(root / 'production' / cell_id)))
    # Scientific validation has its own real-artifact tests. These tests isolate
    # the scheduler's behavior after the driver has accepted a campaign.
    monkeypatch.setattr(scheduler.driver, 'validate_campaign', lambda _: None)
    gates = {}
    for gpu_type in set(scheduler.assignments(gpu_types)):
        output = scheduler.smoke_path(campaign, gpu_type)
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
        gates[gpu_type] = dict(summary=dict(rows=rows), output=output)

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


def test_benchmark_uses_three_cold_solves_and_has_no_submission(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    _, sizing = scheduler.benchmark(path, repo, python)
    assert not commands
    assert sizing['independent_gpu_jobs'] == sizing['total_cpu_requests'] == 5
    assert sizing['total_memory_gib'] == 20
    for row in sizing['cells']:
        assert row['estimated_seconds'] == pytest.approx(1497 * 1.5 + 3 * 30 + 15 * 5 + 20 * 1.5)
        assert row['walltime_minutes'] == 55
        assert row['cpus'] == 1 and row['memory_gib'] == 4


def test_five_parallel_workers_one_afterany_publisher_and_idempotent_retry(tmp_path, monkeypatch):
    gpu_types = ['nvidia_rtx_a5000'] * 5
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch, gpu_types)
    result = scheduler.dispatch(path, repo, python, gpu_types)
    assert len(gates) == 1 and len(commands) == 6 and result['status'] == 'complete'
    for index, command in enumerate(commands[:5]):
        assert '--gres=gpu:nvidia_rtx_a5000:1' in command
        assert '--cpus-per-task=1' in command and '--mem=4G' in command and '--time=00:55:00' in command
        assert not any(value.startswith('--dependency=') for value in command)
        assert '--exclude=gpu2709' in command
        assert shlex.quote(python) in command[-1]
        assert shlex.quote(str(repo / 'slurm/run_ambi_alpha_preservation_oscar.sbatch')) in command[-1]
        assert f'--index {index}' in command[-1]
        subprocess.run(['bash', '-n'], input=command[-1], text=True, check=True)
    publisher = commands[5]
    assert '--dependency=afterany:1001:1002:1003:1004:1005' in publisher
    assert '--mem=2G' in publisher and '--cpus-per-task=1' in publisher and '--time=00:05:00' in publisher
    assert 'ambi_alpha_preservation_publish.py publish' in publisher[-1]
    assert '--require-complete' in publisher[-1] and 'git status --porcelain' in publisher[-1]
    subprocess.run(['bash', '-n'], input=publisher[-1], text=True, check=True)
    assert scheduler.dispatch(path, repo, python, gpu_types) == result and len(commands) == 6


def test_mixed_gpu_classes_each_require_their_own_stream_zero_smoke(tmp_path, monkeypatch):
    assignment = ['nvidia_rtx_a5000', 'quadro_rtx_6000', 'l40s', 'nvidia_rtx_a5000', 'quadro_rtx_6000']
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch, assignment)
    assert len(gates) == 3
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
        scheduler.dispatch(path, repo, python)
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
        scheduler.dispatch(path, repo, python)
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
        scheduler.dispatch(path, repo, python)
    assert scheduler.read(path.parent / 'dispatch.json')['jobs'][2]['status'] == 'submitting'
    with pytest.raises(ValueError, match='reconcile saved intents'):
        scheduler.dispatch(path, repo, python)
    assert len(commands) == 3


def test_unknown_generic_or_wrong_count_assignments_rejected():
    for values in (['gpu'] * 5, ['quadro_rtx_6000'] * 10, ['v100'] * 5, []):
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
        scheduler.dispatch(path, repo, python)
    assert not commands


def test_unrecognized_acknowledgement_is_not_retried(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    original = scheduler.subprocess.check_output
    def ambiguous(args, **kwargs):
        return 'submitted but malformed acknowledgement' if args[0] == 'sbatch' else original(args, **kwargs)
    monkeypatch.setattr(scheduler.subprocess, 'check_output', ambiguous)
    with pytest.raises(ValueError, match='acknowledgement'):
        scheduler.dispatch(path, repo, python)
    assert scheduler.read(path.parent / 'dispatch.json')['jobs'][0]['status'] == 'submitting'
    with pytest.raises(ValueError, match='reconcile saved intents'):
        scheduler.dispatch(path, repo, python)
