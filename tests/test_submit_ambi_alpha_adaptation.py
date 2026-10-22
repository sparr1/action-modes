"""Measured, bounded Slurm dispatch gated by all seven validated smokes."""
import json
import subprocess

import pytest

from slurm import submit_ambi_alpha_adaptation as dispatch


def fixture(tmp_path, monkeypatch, *, cpu2=.95, cpu1=1., memory_kib=2 * 1024 * 1024):
    root, repo = tmp_path / 'campaign', tmp_path / 'worker'
    root.mkdir(); (repo / 'configs/research').mkdir(parents=True)
    py = tmp_path / 'python'; py.touch()
    campaign = dict(campaign_id='alpha-adaptation-episodes-20261021', source_commit='a' * 40,
        seeds=[101, 102, 103, 104, 105], controller_seed=55, max_steps=500, execution_action='sample', cells=[])
    for kind, name in [('matrix', 'ambi_alpha_adaptation_episodes.json'), ('profile', 'ambi_entropy_probe.json')]:
        path = repo / 'configs/research' / name; path.write_text('{}')
        campaign[kind + '_sha256'] = dispatch.digest(path)
    for name in ('sac_aux_100k', 'mppi_aux_400k'):
        for mode in ('repeat100', 'carry'):
            index = len(campaign['cells'])
            campaign['cells'].append(dict(index=index, checkpoint_id=name, mode=mode,
                cell_id=f'{name}__{mode}', selector='alpha_adaptation/' + mode, checkpoint_sha256=str(index) * 64,
                run_id=f'run{index}', run_dir=f'/registries/run{index}', output_dir=str(root / f'production/{index}')))
    paths = []
    for index, steps, cpus, baseline in dispatch.GATES:
        cell = campaign['cells'][index]
        mode = 'reset' if baseline else cell['mode']
        cell_id = cell['checkpoint_id'] + '__' + mode
        output = root / 'smoke' / f'{cell_id}-steps{steps}-cpu{cpus}'
        (output / 'bundle').mkdir(parents=True)
        for name in ('results.json', 'bundle/manifest.json', 'decision-summary.json'):
            (output / name).write_text('{}')
        (output / 'bundle/trace.jsonl.gz').write_bytes(b'trace')
        rate = cpu2 if cpus == 2 else cpu1
        validation = dict(passed=True, lifecycle_validated=True, baseline_prefix_matches=baseline,
            decisions=steps * 2, episodes=2, seeds=[101, 102], selector='alpha_adaptation/' + mode,
            execution_action='sample', temperature_steps_per_decision=3200 if mode == 'repeat100' else 32,
            control_seconds_per_decision=rate, control_seconds=rate * steps * 2,
            probe_seconds=6., initialization_seconds=10.)
        receipt = dict(validated=True, smoke=True, baseline=baseline, source_commit='a' * 40,
            cell_id=cell_id, mode=mode, checkpoint_sha256=cell['checkpoint_sha256'], cpu_threads=cpus,
            validation=validation, elapsed_seconds=rate * steps * 2 + 26., child_maxrss_kib=memory_kib,
            results_sha256=dispatch.digest(output / 'results.json'), manifest_sha256=dispatch.digest(output / 'bundle/manifest.json'),
            decision_summary_sha256=dispatch.digest(output / 'decision-summary.json'),
            trace_sha256={'trace.jsonl.gz': dispatch.digest(output / 'bundle/trace.jsonl.gz')})
        path = output / 'receipt.json'; path.write_text(json.dumps(receipt)); paths.append(path)
    path = root / 'campaign.json'; path.write_text(json.dumps(campaign))
    (root / 'publication').mkdir()
    canonical = json.dumps(campaign, sort_keys=True, separators=(',', ':'), allow_nan=False)
    (root / 'publication/workspace.json').write_text(json.dumps(dict(api_verified=True,
        campaign_sha256=dispatch.hashlib.sha256(canonical.encode()).hexdigest())))
    commands = []
    def command(args, **kwargs):
        if args[0] == 'git':
            return 'a' * 40 + '\n' if 'rev-parse' in args else ''
        assert args[0] == 'sbatch'
        commands.append(args)
        return str(1000 + len(commands)) + ';oscar\n'
    monkeypatch.setattr(dispatch.subprocess, 'check_output', command)
    return path, repo, str(py), paths, commands


@pytest.mark.parametrize('cpu2,cpus', [(.95, 1), (.8, 2), (1.5, 1)])
def test_sizing_uses_measured_scaling_only_for_repeat_workers(tmp_path, monkeypatch, cpu2, cpus):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch, cpu2=cpu2)
    _, sizing = dispatch.benchmark(path, repo, py)
    assert [r['cpus'] for r in sizing['cells']] == [cpus, 1, cpus, 1]
    assert all(r['memory_gib'] == 4 for r in sizing['cells'])
    assert sizing['cells'][1]['estimated_seconds'] == pytest.approx(2500 + 20 + 25 * 3)
    assert sizing['cells'][1]['walltime_minutes'] == 60
    # MPPI repeat retains the measured one-CPU rate even when two CPUs selected.
    assert sizing['cells'][2]['estimated_seconds'] == 2595
    assert not commands


@pytest.mark.parametrize('ordinal', range(7))
def test_every_smoke_is_required_before_any_submission(tmp_path, monkeypatch, ordinal):
    path, repo, py, receipts, commands = fixture(tmp_path, monkeypatch)
    receipts[ordinal].unlink()
    with pytest.raises(FileNotFoundError):
        dispatch.dispatch(path, repo, py)
    assert not commands and not (path.parent / 'dispatch.json').exists()


@pytest.mark.parametrize('change', ['lifecycle', 'parity', 'source', 'work', 'summary', 'trace', 'alias'])
def test_changed_or_invalid_gate_blocks_dispatch(tmp_path, monkeypatch, change):
    path, repo, py, receipts, commands = fixture(tmp_path, monkeypatch)
    selected = receipts[0]; row = dispatch.read(selected)
    if change == 'summary': (selected.parent / 'decision-summary.json').write_text('changed')
    elif change == 'trace': (selected.parent / 'bundle/trace.jsonl.gz').write_bytes(b'changed')
    elif change == 'alias': selected.unlink(); selected.symlink_to(receipts[1])
    else:
        if change == 'source': row['source_commit'] = 'b' * 40
        else: row['validation'][{'lifecycle': 'lifecycle_validated', 'parity': 'baseline_prefix_matches',
                                  'work': 'temperature_steps_per_decision'}[change]] = False
        selected.write_text(json.dumps(row))
    with pytest.raises(ValueError): dispatch.dispatch(path, repo, py)
    assert not commands


@pytest.mark.parametrize('rate,rss', [(10., 2 * 1024 * 1024), (1., 3 * 1024 * 1024)])
def test_time_and_memory_guards_prevent_oversized_jobs(tmp_path, monkeypatch, rate, rss):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch, cpu1=rate, cpu2=rate, memory_kib=rss)
    with pytest.raises(ValueError, match='guard|4 GiB'):
        dispatch.dispatch(path, repo, py)
    assert not commands


def test_four_parallel_gpus_then_serial_publishers_are_idempotent(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    receipt = dispatch.dispatch(path, repo, py)
    assert receipt['status'] == 'complete' and len(commands) == 8
    assert receipt['production_job_ids'] == ['1001', '1002', '1003', '1004']
    for index, command in enumerate(commands[:4]):
        assert '--gres=gpu:l40s:1' in command and '--exclude=gpu2709' in command
        assert '--cpus-per-task=1' in command and '--mem=4G' in command and '--time=01:00:00' in command
        assert not any(arg.startswith('--dependency=') for arg in command)
        assert command[command.index('--index') + 1] == str(index)
        assert command[-2:] == ['--cpu-threads', '1']
    for index, command in enumerate(commands[4:]):
        expected = f'--dependency=afterany:{1001 + index}'
        if index:
            expected += ':' + str(1004 + index)
        assert expected in command
        assert '--cpus-per-task=1' in command and '--mem=2G' in command
        assert 'slurm/ambi_alpha_adaptation_publish.py publish' in command[-1]
        assert '--root ' + str(path.parent / 'publication') in command[-1]
        assert ('--require-complete' in command[-1]) == (index == 3)
    assert dispatch.dispatch(path, repo, py) == receipt and len(commands) == 8


def test_lost_acknowledgement_preserves_intent_and_cannot_duplicate(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    original = dispatch.subprocess.check_output
    def fail(args, **kwargs):
        if args[0] == 'sbatch':
            commands.append(args)
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)
    monkeypatch.setattr(dispatch.subprocess, 'check_output', fail)
    with pytest.raises(subprocess.CalledProcessError): dispatch.dispatch(path, repo, py)
    saved = dispatch.read(path.parent / 'dispatch.json')
    assert saved['status'] == 'dispatching' and saved['jobs'][0]['status'] == 'submitting'
    with pytest.raises(ValueError, match='reconcile saved intents'): dispatch.dispatch(path, repo, py)
    assert len(commands) == 1


@pytest.mark.parametrize('change', ['missing', 'unverified', 'wrong_campaign'])
def test_publication_layout_must_be_verified_before_dispatch(tmp_path, monkeypatch, change):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    workspace = path.parent / 'publication/workspace.json'
    row = dispatch.read(workspace)
    if change == 'missing': workspace.unlink()
    else:
        row['api_verified' if change == 'unverified' else 'campaign_sha256'] = False
        workspace.write_text(json.dumps(row))
    with pytest.raises((FileNotFoundError, ValueError)):
        dispatch.dispatch(path, repo, py)
    assert not commands
