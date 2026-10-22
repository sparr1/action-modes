"""Measured, bounded Slurm dispatch gated by all seven validated smokes."""
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

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
        assert f'--index {index} --cpu-threads 1' in command[-1]
        assert 'production-hardware' in command[-1] and 'exec bash ' in command[-1]
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


def alternate_smokes(path, gpu_type='quadro_rtx_6000'):
    root = path.parent / 'hardware' / gpu_type
    root.mkdir(parents=True)
    shutil.copytree(path.parent / 'smoke', root / 'smoke')
    shutil.copy2(path, root / 'campaign.json')
    for output in (root / 'smoke').iterdir():
        hardware = dict(gpu_type=gpu_type, node='gpu0001', gpu_name='NVIDIA ' + dispatch.GPU_NAMES[gpu_type],
            device_capability=[7, 5], cuda_visible_devices='0', job_id='999',
            nvidia_smi=dispatch.GPU_NAMES[gpu_type] + ', GPU-uuid, 550.54, 24576')
        output.with_name(output.name + '.hardware.json').write_text(json.dumps(hardware))
        (output / 'started.json').write_text(json.dumps({'job_id': '999'}))
    return root / 'smoke'


def test_alternate_gpu_uses_immutable_smokes_preserves_worker_and_scientific_campaign(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    original_campaign = path.read_bytes()
    smoke = alternate_smokes(path)
    result = dispatch.dispatch(path, repo, py, 'quadro_rtx_6000', smoke)
    assert result['gpu_type'] == 'quadro_rtx_6000' and result['smoke_root'] == str(smoke)
    assert result['source_commit'] == 'a' * 40 and path.read_bytes() == original_campaign
    assert all('--gres=gpu:quadro_rtx_6000:1' in command for command in commands[:4])
    assert all(row['hardware']['gpu_name'] == 'NVIDIA Quadro RTX 6000' for row in result['resources']['measurements'])
    assert dispatch.dispatch(path, repo, py, 'quadro_rtx_6000', smoke) == result
    with pytest.raises(ValueError, match='targets differ'):
        dispatch.dispatch(path, repo, py)
    assert len(commands) == 8


@pytest.mark.parametrize('change', ['missing', 'wrong_type', 'wrong_name', 'missing_driver', 'different_allocation', 'campaign'])
def test_alternate_gpu_requires_exact_hardware_and_campaign(tmp_path, monkeypatch, change):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    smoke = alternate_smokes(path)
    hardware_path = smoke / 'sac_aux_100k__repeat100-steps32-cpu2.hardware.json'
    hardware = dispatch.read(hardware_path)
    if change == 'missing': hardware_path.unlink()
    elif change == 'campaign': (smoke.parent / 'campaign.json').write_text('{}')
    else:
        hardware[{'wrong_type': 'gpu_type', 'wrong_name': 'gpu_name', 'missing_driver': 'nvidia_smi',
                  'different_allocation': 'job_id'}[change]] = '' if change == 'missing_driver' else 'wrong'
        hardware_path.write_text(json.dumps(hardware))
    with pytest.raises((ValueError, FileNotFoundError)):
        dispatch.dispatch(path, repo, py, 'quadro_rtx_6000', smoke)
    assert not commands


@pytest.mark.parametrize('gpu_type', ['gpu:1', 'a100', 'quadro_rtx_6000;echo bad', '', None])
def test_only_concrete_allowlisted_gpu_types_can_submit(tmp_path, monkeypatch, gpu_type):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Unsupported GPU'):
        dispatch.dispatch(path, repo, py, gpu_type)
    assert not commands


def test_capture_runs_on_selected_gpu_records_driver_and_never_overwrites(tmp_path, monkeypatch):
    fake_cuda = SimpleNamespace(get_device_name=lambda _: 'Quadro RTX 6000', get_device_capability=lambda _: (7, 5))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(dispatch.subprocess, 'check_output', lambda *a, **k: 'Quadro RTX 6000, GPU-uuid, 550.54, 24576\n')
    prefix = tmp_path / 'production-hardware/0'
    source = dispatch.hardware_capture_code('quadro_rtx_6000', prefix)
    exec(compile(source, '<hardware-capture>', 'exec'), {})
    result = dispatch.read(prefix.with_name('0-123.json'))
    dispatch.validate_hardware(result, 'quadro_rtx_6000')
    assert result['job_id'] == '123' and '550.54' in result['nvidia_smi']
    with pytest.raises(FileExistsError): exec(compile(source, '<hardware-capture>', 'exec'), {})
    fake_cuda.get_device_name = lambda _: 'NVIDIA L40S'
    with pytest.raises(AssertionError, match='GPU class'): exec(compile(source, '<hardware-capture>', 'exec'), {})


def test_explicit_walltime_ceiling_retains_measured_request_and_margin(tmp_path, monkeypatch):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch, cpu1=3.5, cpu2=3.5)
    with pytest.raises(ValueError, match='180-minute guard'):
        dispatch.dispatch(path, repo, py)
    result = dispatch.dispatch(path, repo, py, max_walltime_minutes=240)
    assert result['max_walltime_minutes'] == 240
    assert all(r['walltime_minutes'] == 190 for r in result['resources']['cells'])
    assert all('--time=03:10:00' in command for command in commands[:4])
    with pytest.raises(ValueError, match='targets differ'):
        dispatch.dispatch(path, repo, py, max_walltime_minutes=360)
    assert len(commands) == 8


@pytest.mark.parametrize('ceiling', [0, -1, 361, True, 180.5])
def test_invalid_walltime_ceilings_cannot_submit(tmp_path, monkeypatch, ceiling):
    path, repo, py, _, commands = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Maximum walltime'):
        dispatch.dispatch(path, repo, py, max_walltime_minutes=ceiling)
    assert not commands


def reviewed_fixture(tmp_path, monkeypatch):
    path, repo, py, receipts, commands = fixture(tmp_path, monkeypatch)
    campaign = dispatch.read(path)
    cell = campaign['cells'][0]
    historical = {'rows': [dict(seed=seed, decision_index=decision, reward=.1,
        alpha_initial=.003, alpha_final=.00299, entropy=2.415818452835083)
        for seed in (101, 102) for decision in range(8)]}
    reference = path.parent / 'references'; reference.mkdir()
    historical_path = reference / 'decision-summary.json'
    historical_path.write_text(json.dumps(historical))
    cell['baseline_decision_summary'] = str(historical_path)
    pins = {}
    for kind in ('baseline', 'reference'):
        bundle = reference / kind / 'bundle'; bundle.mkdir(parents=True)
        cell[kind + '_bundle'] = str(bundle)
        for suffix, artifact in (('manifest', bundle / 'manifest.json'), ('results', bundle.parent / 'results.json')):
            artifact.write_text('{}')
            pins[kind + '_' + suffix + '_sha256'] = dispatch.digest(artifact)
    cell.update(pins)
    (reference / 'baseline-receipt.json').write_text(json.dumps(dict(
        decision_summary_sha256=dispatch.digest(historical_path),
        baseline_manifest_sha256=pins['baseline_manifest_sha256'], baseline_results_sha256=pins['baseline_results_sha256'])))
    path.write_text(json.dumps(campaign))
    workspace = path.parent / 'publication/workspace.json'
    workspace.write_text(json.dumps(dict(api_verified=True,
        campaign_sha256=dispatch.hashlib.sha256(json.dumps(campaign, sort_keys=True, separators=(',', ':')).encode()).hexdigest())))
    smoke = alternate_smokes(path)
    output = smoke / 'sac_aux_100k__reset-steps8-cpu1'
    original = dispatch.read(output / 'receipt.json')
    (output / 'receipt.json').unlink()
    (output / 'failure.json').write_text(json.dumps(dict(exception_type='ValueError', message=dispatch.BASELINE_PARITY_FAILURE)))
    current = json.loads(json.dumps(historical))
    current['rows'][14]['entropy'] = 2.4163899421691895
    (output / 'decision-summary.json').write_text(json.dumps(current))
    original_command = dispatch.subprocess.check_output
    audit_commands = []
    def command(args, **kwargs):
        if args[0] == py:
            assert args[1] == '-c' and kwargs['cwd'] == repo
            assert 'd.validate_result(root,cell,8)' in args[2]
            audit_commands.append(args)
            return json.dumps(dict(validation=original['validation'], pins=pins, trace_sha256=original['trace_sha256']))
        return original_command(args, **kwargs)
    monkeypatch.setattr(dispatch.subprocess, 'check_output', command)
    return path, repo, py, smoke, output, commands, audit_commands


def test_explicit_review_revalidates_preserves_failure_and_does_not_invent_resource_measurements(tmp_path, monkeypatch):
    path, repo, py, smoke, output, commands, audits = reviewed_fixture(tmp_path, monkeypatch)
    failure = (output / 'failure.json').read_bytes()
    with pytest.raises(FileNotFoundError): dispatch.benchmark(path, repo, py, 'quadro_rtx_6000', smoke)
    _, sizing = dispatch.benchmark(path, repo, py, 'quadro_rtx_6000', smoke, allow_reviewed_cross_gpu_entropy=True)
    reviewed = dispatch.read(output / 'reviewed-receipt.json')
    assert reviewed['cross_gpu_baseline_review_passed'] is True
    assert reviewed['validation']['baseline_prefix_matches'] is False
    assert reviewed['baseline_review']['comparisons']['original_failure_count'] == 1
    assert reviewed['baseline_review']['comparisons']['comparisons'] == 64
    assert 'elapsed_seconds' not in reviewed and 'child_maxrss_kib' not in reviewed
    assert (output / 'failure.json').read_bytes() == failure and not (output / 'receipt.json').exists()
    assert all(row['measurement_receipt'] != str(output / 'reviewed-receipt.json') for row in sizing['cells'])
    dispatch.benchmark(path, repo, py, 'quadro_rtx_6000', smoke, allow_reviewed_cross_gpu_entropy=True)
    assert len(audits) == 1 and not commands


@pytest.mark.parametrize('metric,delta', [('reward', .001), ('alpha_initial', .001), ('alpha_final', .001), ('entropy', .0011)])
def test_review_never_waives_reward_alpha_or_larger_entropy_drift(tmp_path, monkeypatch, metric, delta):
    path, repo, py, smoke, output, commands, _ = reviewed_fixture(tmp_path, monkeypatch)
    current = dispatch.read(output / 'decision-summary.json')
    current['rows'][0][metric] += delta
    (output / 'decision-summary.json').write_text(json.dumps(current))
    with pytest.raises(ValueError, match='cannot waive'):
        dispatch.benchmark(path, repo, py, 'quadro_rtx_6000', smoke, allow_reviewed_cross_gpu_entropy=True)
    assert not commands and not (output / 'reviewed-receipt.json').exists()


@pytest.mark.parametrize('changed_file', ['failure.json', 'decision-summary.json', 'bundle/manifest.json'])
def test_reviewed_evidence_is_rehashed_before_dispatch(tmp_path, monkeypatch, changed_file):
    path, repo, py, smoke, output, commands, _ = reviewed_fixture(tmp_path, monkeypatch)
    dispatch.benchmark(path, repo, py, 'quadro_rtx_6000', smoke, allow_reviewed_cross_gpu_entropy=True)
    (output / changed_file).write_text('{"changed": true}')
    with pytest.raises(ValueError):
        dispatch.dispatch(path, repo, py, 'quadro_rtx_6000', smoke, allow_reviewed_cross_gpu_entropy=True)
    assert not commands


def test_review_rejects_any_other_worker_failure(tmp_path, monkeypatch):
    path, repo, py, smoke, output, commands, audits = reviewed_fixture(tmp_path, monkeypatch)
    (output / 'failure.json').write_text(json.dumps(dict(exception_type='ValueError', message='Adam state continuity mismatch')))
    with pytest.raises(ValueError, match='another worker failure'):
        dispatch.dispatch(path, repo, py, 'quadro_rtx_6000', smoke, allow_reviewed_cross_gpu_entropy=True)
    assert not commands and not audits
