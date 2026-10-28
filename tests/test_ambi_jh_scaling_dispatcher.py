"""Oscar-side production submission must be measured, bounded and idempotent."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from slurm import submit_ambi_jh_scaling_production as dispatch
from tests.test_ambi_return_jh_scaling_training import scaling_receipt

SHA = 'a' * 40
JOBS = ['1001', '1002', '1003']


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, 'OUTPUT', tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '2001')
    monkeypatch.setattr(dispatch.campaign, 'binding', lambda manifest: {
        'commit': SHA, 'manifest': str(manifest.relative_to(dispatch.ROOT)), 'config_sha256': {'same': 'same'}})
    for case, smoke_job, (manifest, spec) in zip(dispatch.CASES, JOBS, dispatch.campaign.SCALING_CASES.items()):
        gate = tmp_path / 'cases' / case / 'gate'
        gate.mkdir(parents=True)
        receipt, _ = scaling_receipt(spec)
        binding = dispatch.campaign.binding(manifest)
        runtime = dict(gpu='NVIDIA L40S', executable=dispatch.PYTHON)
        receipt.update(binding=binding, runtime=runtime, recommended_cpus_within_5pct_of_best=1, artifacts={})
        for row in receipt['cases']:
            row.update(binding=binding, runtime=runtime)
            row['measured']['seconds_per_decision'] = .4 if row['physical_cores'] == 1 else .39
            path = gate / f"cores-{row['physical_cores']}.json"
            dispatch.durable_new(path, row)
            receipt['artifacts'][str(path)] = dispatch.campaign.digest(path)
        dispatch.durable_new(gate / 'receipt.json', receipt)
        dispatch.durable_new(gate / 'protocol.json', dict(slurm_job_id=smoke_job, binding=binding,
            config=spec[0], warmup_active_decisions=60, measured_decisions=120))
    runtime = tmp_path / 'publication-env/bin/python'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('test fixture')
    workspace = tmp_path / 'publication/workspace.json'
    workspace.parent.mkdir()
    workspace.write_text('{}')
    calls = []
    def command(arguments):
        calls.append(arguments)
        if arguments[:3] == ['git', 'rev-parse', 'HEAD']:
            return SHA
        if arguments[:2] == ['git', 'status']:
            return ''
        if arguments[0] == 'sacct':
            return '\n'.join(job + '|COMPLETED|0:0|' for job in JOBS)
        if arguments[0] == 'sacctmgr':
            if 'user' in arguments:
                return 'defaultacct|'
            return 'pri-gpu+|cpu=72,gres/gpu=12,mem=576G|cpu=414720,gres/gpu=69120|7-00:00:00|1200|'
        if arguments[0] == 'squeue':
            return '1|pri-gpu+|defaultacct|1|1-00:00:00|gpu:l40s:1|RUNNING|12:00:00|8G|1'
        if arguments[0] == 'sbatch':
            intents = list((tmp_path / 'production-submissions').glob('*-intent.json'))
            assert any(dispatch.read(path)['command'] == arguments for path in intents), 'Intent must exist before sbatch'
            return str(3000 + sum(call[0] == 'sbatch' for call in calls))
        raise AssertionError(arguments)
    monkeypatch.setattr(dispatch, 'command', command)
    return tmp_path, calls, command


def test_all_three_measured_plans_submit_once_and_repeat_is_idempotent(prepared):
    output, calls, _ = prepared
    result = dispatch.dispatch(output, SHA, JOBS)
    assert result['submitted'] == ['3001', '3002', '3003']
    assert len([call for call in calls if call[0] == 'sbatch']) == 4
    for plan in result['plans']:
        assert plan['cpus'] == 1
        assert plan['wall_hours'] == 130  # ceil((.4*1M*1.15+7200)/3600)
        assert plan['drain_seconds'] == 130 * 3600 - 1800
    for arguments in [call for call in calls if call[0] == 'sbatch' and '--partition=gpu' in call]:
        assert '--gres=gpu:l40s:1' in arguments and '--mem=12G' in arguments
        assert '--account=defaultacct' in arguments and '--signal=USR1@1800' in arguments
        assert any('AMBI_RETURN_STORAGE=scratch-jh-scaling' in arg for arg in arguments)
    assert dispatch.read(output / 'production-jobs.json') == dict(h3_j12_k3='3001', h4_j10_k4='3002', h5_j10_k5='3003')
    second = dispatch.dispatch(output, SHA, JOBS)
    assert second['submitted'] == result['submitted']
    assert len([call for call in calls if call[0] == 'sbatch']) == 4
    assert result['workspace_publisher'] == second['workspace_publisher'] == '3004'
    publisher = [call for call in calls if call[0] == 'sbatch' and '--partition=batch' in call][0]
    assert '--dependency=after:3001:3002:3003' in publisher


def test_dry_run_does_not_submit_or_create_intents(prepared):
    output, calls, _ = prepared
    result = dispatch.dispatch(output, SHA, JOBS, dry_run=True)
    assert result['dry_run'] and len(result['pending']) == 3
    assert not any(call[0] == 'sbatch' for call in calls)
    assert not list(output.rglob('*-intent.json')) and not (output / 'production-jobs.json').exists()


def test_ambiguous_intent_stops_all_future_submission(prepared):
    output, calls, _ = prepared
    directory = output / 'production-submissions'
    directory.mkdir()
    dispatch.durable_new(directory / (dispatch.CASES[1] + '-intent.json'), {'unknown': True})
    with pytest.raises(AssertionError, match='Ambiguous submission intent'):
        dispatch.dispatch(output, SHA, JOBS)
    assert not any(call[0] == 'sbatch' for call in calls)


def test_sbatch_failure_retains_intent_and_retry_never_submits(prepared, monkeypatch):
    output, calls, original = prepared
    def fail(arguments):
        if arguments[0] == 'sbatch':
            raise RuntimeError('connection lost after scheduler acceptance')
        return original(arguments)
    monkeypatch.setattr(dispatch, 'command', fail)
    with pytest.raises(RuntimeError):
        dispatch.dispatch(output, SHA, JOBS)
    assert (output / 'production-submissions' / (dispatch.CASES[0] + '-intent.json')).exists()
    monkeypatch.setattr(dispatch, 'command', original)
    with pytest.raises(AssertionError, match='Ambiguous'):
        dispatch.dispatch(output, SHA, JOBS)
    assert not any(call[0] == 'sbatch' for call in calls)


@pytest.mark.parametrize('mutation', ['gpu', 'python', 'artifact', 'binding', 'smoke_job', 'cpu_choice'])
def test_bad_gate_rejects_before_any_submission(prepared, mutation):
    output, calls, _ = prepared
    gate = output / 'cases' / dispatch.CASES[0] / 'gate'
    path = gate / ('protocol.json' if mutation == 'smoke_job' else 'receipt.json')
    value = dispatch.read(path)
    if mutation == 'gpu': value['runtime']['gpu'] = 'NVIDIA A100'
    if mutation == 'python': value['runtime']['executable'] = '/other/python'
    if mutation == 'artifact': value['artifacts'][str(gate / 'cores-1.json')] = 'bad'
    if mutation == 'binding': value['binding']['commit'] = 'b' * 40
    if mutation == 'smoke_job': value['slurm_job_id'] = '9999'
    if mutation == 'cpu_choice': value['recommended_cpus_within_5pct_of_best'] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        dispatch.dispatch(output, SHA, JOBS)
    assert not any(call[0] == 'sbatch' for call in calls)


@pytest.mark.parametrize('bad', ['failed_smoke', 'gpu_budget', 'cpu_budget', 'maxwall', 'dirty'])
def test_live_slurm_and_source_checks_fail_closed(prepared, monkeypatch, bad):
    output, calls, original = prepared
    def changed(arguments):
        value = original(arguments)
        if bad == 'failed_smoke' and arguments[0] == 'sacct':
            return value.replace('COMPLETED', 'FAILED', 1)
        if bad == 'gpu_budget' and arguments[0] == 'sacctmgr' and 'qos' in arguments:
            return value.replace('gres/gpu=69120', 'gres/gpu=100')
        if bad == 'cpu_budget' and arguments[0] == 'sacctmgr' and 'qos' in arguments:
            return value.replace('cpu=414720', 'cpu=100')
        if bad == 'maxwall' and arguments[0] == 'sacctmgr' and 'qos' in arguments:
            return value.replace('7-00:00:00', '1-00:00:00')
        if bad == 'dirty' and arguments[:2] == ['git', 'status']:
            return ' M main.py'
        return value
    monkeypatch.setattr(dispatch, 'command', changed)
    with pytest.raises(AssertionError):
        dispatch.dispatch(output, SHA, JOBS)
    assert not any(call[0] == 'sbatch' for call in calls)


@pytest.mark.parametrize('text,count', [('gpu:l40s:1', 1), ('gres/gpu:1', 1), ('gpu:l40s:2(S:0)', 2), ('(null)', 0)])
def test_gpu_request_parser(text, count):
    assert dispatch.gpu_count(text) == count


@pytest.mark.parametrize('state,elapsed,expected', [('RUNNING','12:00:00',720), ('PENDING','12:00:00',1440)])
def test_budget_uses_remaining_running_time_but_full_pending_limit(prepared, monkeypatch, state, elapsed, expected):
    output, calls, original = prepared
    def query(arguments):
        if arguments[0] == 'squeue':
            return f'1|pri-gpu+|defaultacct|2|1-00:00:00|gpu:l40s:1|{state}|{elapsed}|12G|1'
        return original(arguments)
    monkeypatch.setattr(dispatch, 'command', query)
    budget = dispatch.check_budget([])
    assert budget['current']['gpu_minutes'] == expected
    assert budget['current']['cpu_minutes'] == 2 * expected
    assert budget['current']['memory_mib'] == 12 * 1024


def test_memory_budget_rejects_excess_reservation(prepared, monkeypatch):
    output, calls, original = prepared
    def query(arguments):
        if arguments[0] == 'squeue':
            return '1|pri-gpu+|defaultacct|2|1-00:00:00|gpu:l40s:1|RUNNING|12:00:00|576G|1'
        return original(arguments)
    monkeypatch.setattr(dispatch, 'command', query)
    with pytest.raises(AssertionError, match='memory'):
        dispatch.dispatch(output, SHA, JOBS)
    assert not any(call[0] == 'sbatch' for call in calls)
