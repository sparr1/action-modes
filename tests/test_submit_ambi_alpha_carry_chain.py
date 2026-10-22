"""Cold/warm resource sizing and idempotent submission of independent streams."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from slurm import submit_ambi_alpha_carry_chain as scheduler
from slurm import ambi_alpha_carry_chain as driver


def fixture(tmp_path, monkeypatch, gpu_types=None):
    root, repo = tmp_path / 'campaign', tmp_path / 'worker'
    root.mkdir(); (repo / 'configs/research').mkdir(parents=True)
    python = tmp_path / 'python'; python.touch()
    campaign = dict(schema_version=1, campaign_id=driver.CAMPAIGN_ID, root=str(root), protocol=driver.PROTOCOL,
        source_commit='a' * 40, matrix_sha256=driver.sha256(driver.MATRIX), profile_sha256=driver.sha256(driver.PROFILE),
        streams=driver.STREAMS, controller_seed=55, max_steps=500, execution_action='sample', independent_unit='stream',
        checkpoints={name: {} for name in driver.CHECKPOINTS}, cells=[])
    for source in (driver.MATRIX, driver.PROFILE):
        shutil.copy2(source, repo / 'configs/research' / source.name)
    for name in driver.CHECKPOINTS:
        for index in range(5):
            cell = {**driver.read(driver.PROFILE)['checkpoints'][name], **driver.intervention(name, index),
                    'index': len(campaign['cells'])}
            cell['output_dir'] = str(root / 'production' / cell['cell_id'])
            cell['science_id'] = driver.canonical_hash(dict(protocol=driver.PROTOCOL, source=campaign['source_commit'],
                matrix=campaign['matrix_sha256'], checkpoint=cell['checkpoint_sha256'], stream=index,
                seeds=cell['seeds'], controller_seed=55))
            campaign['cells'].append(cell)
    gates = {}
    for cell, gpu_type in zip(campaign['cells'], scheduler.assignments(gpu_types)):
        key = cell['checkpoint_id'], gpu_type
        if key in gates: continue
        output = scheduler.smoke_path(campaign, *key)
        output.mkdir(parents=True)
        rows = [dict(decision_index=d, control_seconds=30. if d == 0 else 1.5) for _ in range(2) for d in range(8)]
        receipt = dict(cpu_threads=1, child_maxrss_kib=2 * 1024**2, child_elapsed_seconds=111.,
                       validation=dict(probe_seconds=10.))
        (output / 'receipt.json').write_text(json.dumps(receipt))
        (output / 'started.json').write_text(json.dumps(dict(job_id='123')))
        hardware = dict(gpu_type=gpu_type, node='gpu0001', gpu_name='NVIDIA ' + scheduler.GPU_NAMES[gpu_type],
            device_capability=[7, 5], cuda_visible_devices='0', job_id='123',
            nvidia_smi=scheduler.GPU_NAMES[gpu_type] + ', GPU-uuid, 550.54, 24576')
        output.with_name(output.name + '.hardware.json').write_text(json.dumps(hardware))
        gates[key] = (dict(rows=rows), output)
    def load_stream(campaign, cell, output, smoke_steps):
        assert smoke_steps == 8 and cell['stream_index'] == 0
        key = next(key for key, (_, path) in gates.items() if path == output)
        return gates[key][0], driver.read(output / 'receipt.json')
    monkeypatch.setattr(driver, 'load_stream', load_stream)
    path = root / 'campaign.json'; path.write_text(json.dumps(campaign))
    (root / 'publication').mkdir()
    (root / 'publication/workspace.json').write_text(json.dumps(dict(api_verified=True, campaign_sha256=driver.canonical_hash(campaign))))
    commands = []
    def call(args, **kwargs):
        if args[0] == 'git': return 'a' * 40 + '\n' if 'rev-parse' in args else ''
        assert args[0] == 'sbatch'; commands.append(args)
        return f'{1000 + len(commands)};oscar\n'
    monkeypatch.setattr(scheduler.subprocess, 'check_output', call)
    return path, repo, str(python), commands, gates


def test_cold_cost_is_charged_three_times_not_fifteen_hundred(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    _, sizing = scheduler.benchmark(path, repo, python)
    assert not commands and sizing['independent_gpu_jobs'] == 10
    for row in sizing['cells']:
        assert row['estimated_seconds'] == pytest.approx(1497 * 1.5 + 3 * 30 + 15 * 5 + 20 * 1.5)
        assert row['walltime_minutes'] == 55
        assert row['cpus'] == 1 and row['memory_gib'] == 4


def test_ten_parallel_workers_and_two_complete_checkpoint_merges(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    receipt = scheduler.dispatch(path, repo, python)
    assert len(commands) == 12 and receipt['status'] == 'complete'
    for index, command in enumerate(commands[:10]):
        assert '--gres=gpu:quadro_rtx_6000:1' in command
        assert '--cpus-per-task=1' in command and '--mem=4G' in command and '--time=00:55:00' in command
        assert '--exclude=gpu2709' in command and not any(v.startswith('--dependency=') for v in command)
        assert f'--index {index}' in command[-1]
        assert 'srun /usr/bin/env OMP_NUM_THREADS=1' in command[-1]
    assert '--dependency=afterany:1001:1002:1003:1004:1005' in commands[10]
    assert '--dependency=afterany:1006:1007:1008:1009:1010:1011' in commands[11]
    assert '--require-complete' not in commands[10][-1] and '--require-complete' in commands[11][-1]
    assert 'ambi_alpha_carry_chain_publish.py publish' in commands[10][-1]
    assert commands[10][-1].startswith('/usr/bin/env ')
    assert scheduler.dispatch(path, repo, python) == receipt and len(commands) == 12


def test_mixed_classes_require_measured_smokes_for_each_checkpoint(tmp_path, monkeypatch):
    assignment = ['quadro_rtx_6000', 'nvidia_rtx_a5000'] * 5
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch, assignment)
    assert len(gates) == 4
    receipt = scheduler.dispatch(path, repo, python, assignment)
    assert receipt['gpu_types'] == assignment
    assert all('--gres=gpu:' + gpu + ':1' in command for gpu, command in zip(assignment, commands[:10]))


@pytest.mark.parametrize('change', ['missing', 'model', 'job', 'memory', 'slow', 'cpu'])
def test_bad_or_missing_smoke_blocks_all_submission(tmp_path, monkeypatch, change):
    path, repo, python, commands, gates = fixture(tmp_path, monkeypatch)
    summary, output = next(iter(gates.values()))
    if change == 'missing': (output / 'receipt.json').unlink()
    elif change in ('model', 'job'):
        hp = output.with_name(output.name + '.hardware.json'); hardware = driver.read(hp)
        hardware['gpu_name' if change == 'model' else 'job_id'] = 'wrong'
        hp.write_text(json.dumps(hardware))
    elif change == 'slow':
        for row in summary['rows']: row['control_seconds'] = 10.
    else:
        receipt = driver.read(output / 'receipt.json')
        receipt['child_maxrss_kib' if change == 'memory' else 'cpu_threads'] = 3 * 1024**2 if change == 'memory' else 2
        (output / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)): scheduler.dispatch(path, repo, python)
    assert not commands and not (path.parent / 'dispatch.json').exists()


def test_unverified_workspace_does_not_start_experiments(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    (path.parent / 'publication/workspace.json').write_text(json.dumps(dict(api_verified=False)))
    with pytest.raises(ValueError, match='workspace'): scheduler.dispatch(path, repo, python)
    assert not commands


def test_lost_acknowledgement_retains_intent_and_cannot_duplicate(tmp_path, monkeypatch):
    path, repo, python, commands, _ = fixture(tmp_path, monkeypatch)
    original = scheduler.subprocess.check_output
    def fail(args, **kwargs):
        if args[0] == 'sbatch':
            commands.append(args); raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)
    monkeypatch.setattr(scheduler.subprocess, 'check_output', fail)
    with pytest.raises(subprocess.CalledProcessError): scheduler.dispatch(path, repo, python)
    assert driver.read(path.parent / 'dispatch.json')['jobs'][0]['status'] == 'submitting'
    with pytest.raises(ValueError, match='reconcile saved intents'): scheduler.dispatch(path, repo, python)
    assert len(commands) == 1


def test_unknown_generic_and_partial_gpu_assignments_rejected():
    for values in (['gpu'] * 10, ['quadro_rtx_6000'] * 5, ['v100'] * 10):
        with pytest.raises(ValueError): scheduler.assignments(values)
