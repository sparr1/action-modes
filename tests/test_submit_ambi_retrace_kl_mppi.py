"""Measured resource sizing and exactly-once Slurm dispatch with mocked commands."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from slurm import submit_ambi_retrace_kl_mppi as dispatch


def fixture(tmp_path, monkeypatch, *, times=(.8, .78, .9), memory_kib=2 * 1024 * 1024):
    root, repo = tmp_path / 'campaign', tmp_path / 'worker'
    root.mkdir(); (repo / 'configs/research').mkdir(parents=True)
    py = tmp_path / 'python'; py.touch()
    campaign = dict(campaign_id='retrace-kl-mppi-20261021', source_commit='a' * 40,
        seeds=[101, 102, 103, 104, 105], max_steps=500, execution_action='sample', cells=[])
    for kind, suffix in [('matrix', ''), ('profile', '_profile')]:
        path = repo / f'configs/research/ambi_retrace_kl_mppi{suffix}.json'
        path.write_text('{}')
        campaign[kind + '_sha256'] = dispatch.digest(path)
    for index, (step, mode) in enumerate([(400000, 'retrace_kl'), (400000, 'retrace_both'),
                                        (500000, 'retrace_kl'), (500000, 'retrace_both')]):
        campaign['cells'].append(dict(index=index, step=step, mode=mode, cell_id=f'{mode}_{step}',
            selector=f'retrace_kl_mppi/{mode}_{step}', checkpoint_sha256=str(index) * 64,
            run_id=f'run{index}', run_dir=f'/registries/run{index}', output_dir=str(root / f'production/{index}')))
    paths = []
    for ordinal, (index, steps, cpus) in enumerate(((0, 32, 1), (0, 32, 2), (3, 8, 1))):
        cell = campaign['cells'][index]
        output = root / 'smoke' / f"{cell['cell_id']}-steps{steps}-cpu{cpus}"
        (output / 'bundle').mkdir(parents=True)
        (output / 'results.json').write_text('{}')
        (output / 'bundle/manifest.json').write_text('{}')
        (output / 'bundle/trace.jsonl.gz').write_bytes(b'trace fixture')
        receipt = dict(validated=True, smoke=True, source_commit=campaign['source_commit'],
            cell_id=cell['cell_id'], checkpoint_sha256=cell['checkpoint_sha256'], cpu_threads=cpus,
            elapsed_seconds=times[ordinal] * steps + 3 + 20, child_maxrss_kib=memory_kib,
            validation=dict(passed=True, decisions=steps, episodes=1, seeds=[101],
                selector=cell['selector'], execution_action='sample', control_seconds=times[ordinal] * steps,
                control_seconds_per_decision=times[ordinal], probe_seconds=3, initialization_seconds=10),
            results_sha256=dispatch.digest(output / 'results.json'),
            manifest_sha256=dispatch.digest(output / 'bundle/manifest.json'),
            trace_sha256={'trace.jsonl.gz': dispatch.digest(output / 'bundle/trace.jsonl.gz')})
        path = output / 'receipt.json'; path.write_text(json.dumps(receipt)); paths.append(path)
    path = root / 'campaign.json'; path.write_text(json.dumps(campaign))
    commands = []
    def command(args, **kwargs):
        if args[0] == 'git':
            return 'a' * 40 + '\n' if 'rev-parse' in args else ''
        assert args[0] == 'sbatch'
        commands.append(args)
        return str(1000 + len(commands)) + ';oscar\n'
    monkeypatch.setattr(dispatch.subprocess, 'check_output', command)
    return path, repo, str(py), paths, commands


@pytest.mark.parametrize('times,cpus', [((.8, .78, .9), 1), ((.8, .75, .9), 2)])
def test_resource_sizing_uses_throughput_probe_cost_startup_margin_and_peak_rss(tmp_path, monkeypatch, times, cpus):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch, times=times)
    _, sizing = dispatch.benchmark(path, repo, py)
    assert sizing['cpus'] == cpus
    assert sizing['memory_gib'] == 4
    assert sizing['estimated_seconds'] == pytest.approx(.9 * 2500 + 20 + 3 * 25)
    assert sizing['walltime_minutes'] == 50  # 2814s with margin, rounded up to five minutes.
    assert not commands


def test_slower_unused_two_cpu_controller_time_does_not_inflate_production_walltime(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch, times=(.8, 1.5, .9))
    _, sizing = dispatch.benchmark(path, repo, py)
    assert sizing['cpus'] == 1
    assert sizing['estimated_seconds'] == pytest.approx(.9 * 2500 + 20 + 3 * 25)
    assert sizing['walltime_minutes'] == 50
    assert not commands


@pytest.mark.parametrize('change', ['missing', 'unvalidated', 'wrong_cpu', 'wrong_commit', 'results', 'trace', 'alias'])
def test_gate_failure_or_changed_outputs_cannot_submit_any_job(tmp_path, monkeypatch, change):
    path, repo, py, receipts, commands = fixture(tmp_path, monkeypatch)
    selected = receipts[1]
    data = dispatch.read(selected)
    if change == 'missing': selected.unlink()
    elif change == 'results': (selected.parent / 'results.json').write_text('changed')
    elif change == 'trace': (selected.parent / 'bundle/trace.jsonl.gz').write_bytes(b'changed')
    elif change == 'alias': selected.unlink(); selected.symlink_to(receipts[0])
    else:
        data[{'unvalidated': 'validated', 'wrong_cpu': 'cpu_threads', 'wrong_commit': 'source_commit'}[change]] = False
        selected.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        dispatch.dispatch(path, repo, py)
    assert commands == [] and not (path.parent / 'dispatch.json').exists()


@pytest.mark.parametrize('time,rss', [(10., 2 * 1024 * 1024), (.8, 6 * 1024 * 1024)])
def test_large_time_or_memory_requires_review_before_any_submission(tmp_path, monkeypatch, time, rss):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch, times=(time, time, time), memory_kib=rss)
    with pytest.raises(ValueError, match='guard|8 GiB'):
        dispatch.dispatch(path, repo, py)
    assert not commands


def test_four_independent_gpus_and_serial_publishers_are_submitted_once(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    receipt = dispatch.dispatch(path, repo, py, previous_publisher='999')
    assert receipt['status'] == 'complete' and len(commands) == 8
    assert receipt['production_job_ids'] == ['1001', '1002', '1003', '1004']
    assert receipt['final_publisher_job_id'] == '1008'
    for index, command in enumerate(commands[:4]):
        assert '--gres=gpu:l40s:1' in command and '--exclude=gpu2709' in command
        assert '--cpus-per-task=1' in command and '--mem=4G' in command and '--time=00:50:00' in command
        assert not any(arg.startswith('--dependency=') for arg in command)
        assert command[-1] == str(index)
        assert str(repo / 'slurm/run_ambi_retrace_kl_mppi_oscar.sbatch') in command
    for index, command in enumerate(commands[4:]):
        prior = '999' if index == 0 else str(1004 + index)
        assert f'--dependency=afterany:{1001 + index}:{prior}' in command
        assert '--cpus-per-task=1' in command and '--mem=2G' in command and '--time=00:05:00' in command
        assert ('--require-complete' in command[-1]) == (index == 3)
    assert dispatch.dispatch(path, repo, py, previous_publisher='999') == receipt
    assert len(commands) == 8


def test_lost_submission_acknowledgement_is_not_retried(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    original = dispatch.subprocess.check_output
    def fail(args, **kwargs):
        if args[0] == 'sbatch':
            commands.append(args)
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)
    monkeypatch.setattr(dispatch.subprocess, 'check_output', fail)
    with pytest.raises(subprocess.CalledProcessError): dispatch.dispatch(path, repo, py)
    receipt = dispatch.read(path.parent / 'dispatch.json')
    assert receipt['status'] == 'dispatching' and receipt['jobs'][0]['status'] == 'submitting'
    with pytest.raises(ValueError, match='reconcile saved intents'): dispatch.dispatch(path, repo, py)
    assert len(commands) == 1


def test_worker_checkout_and_config_are_pinned_before_submission(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    config = repo / 'configs/research/ambi_retrace_kl_mppi.json'
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='configuration differs'): dispatch.dispatch(path, repo, py)
    assert not commands
