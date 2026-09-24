"""Checkpoint pairing, historical reuse, and live overview publication contracts."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_checkpoint_sweep_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset):
    return [dict(seed=seed, solver_seed=solver_seed(55, 'episode', seed), length=500,
        truncated_by_evaluator=False, **{'return': float(offset + seed - 100)}) for seed in publication.SEEDS]


def fixture_campaign():
    cells = []
    for step in publication.STEPS:
        prior = dict(episodes=episodes(step / 1000), checkpoint_step=step, checkpoint_sha256=f'checkpoint-{step}',
                     performance_run_id=None, manifest_sha256=f'prior-{step}')
        cell = dict(name=f'step_{step}', checkpoint_step=step, training_decisions=step,
            checkpoint_sha256=f'checkpoint-{step}', reused=step == 575000, initial_alpha=step / 1e8,
            H=3, J=10, estimator='one_step', execution_mode='mean', alpha_mode='adaptive', critic_kind='return_only',
            performance_run_id='historical-performance' if step == 575000 else 'new-performance', training_run_id=f'training-{step}',
            prior_reference=prior, params={'inner_replay_capacity': 3840}, directory=f'directory-{step}',
            bundle=f'bundle-{step}', run_dir='historical-run' if step == 575000 else 'new-run')
        if cell['reused']:
            cell['reuse_reference'] = dict(prior, episodes=episodes(step / 1000 + 10))
        cells.append(cell)
    return dict(cells=cells, group='sweep-test', label='H3/J10 checkpoints', source_run='backbone',
        source_commit='tested', inventory='inventory.json', overview_run_id='overview', publisher_workers=3)


def completed(campaign):
    return {c['name']: [dict(e, **{'return': e['return'] + c['checkpoint_step'] / 100000},
                           paired_return_delta=c['checkpoint_step'] / 100000)
                        for e in c['prior_reference']['episodes']]
            for c in campaign['cells'] if not c['reused']}


def test_startup_shows_all_priors_one_reused_measurement_and_null_missing_points():
    result = publication.aggregate_results(fixture_campaign(), {})
    assert result['evaluated'] == result['reused'] == 1 and result['new_evaluated'] == 0
    assert len(result['points']) == 25 and len(result['episodes']) == 5
    assert [p['training_decisions'] for p in result['points']] == publication.STEPS
    assert all(p['prior_return_mean'] == p['training_decisions'] / 1000 + 3 for p in result['points'])
    for point in result['points']:
        if point['reused']:
            assert point['return_mean'] == 588 and point['paired_gain_mean'] == 10
        else:
            assert all(point[k] is None for k in ('return_mean', 'return_std', 'return_episodes',
                'paired_gain_mean', 'paired_gain_ci95_low', 'paired_gain_ci95_high', 'paired_episodes'))
    rows = publication.numeric_rows(result)
    assert len(rows) == 26 and len({identity for identity, _ in rows}) == 26
    assert all('checkpoint/training_decisions' in row and 'axis/inner_rounds' not in row for _, row in rows)


def test_full_curve_pairs_by_actual_seed_at_each_checkpoint_with_bootstrap_intervals():
    campaign = fixture_campaign()
    results = {name: list(reversed(eps)) for name, eps in completed(campaign).items()}
    result = publication.aggregate_results(campaign, results)
    assert result['evaluated'] == 25 and result['new_evaluated'] == 24
    assert len(result['episodes']) == 125
    for point in result['points']:
        expected = 10 if point['reused'] else point['training_decisions'] / 100000
        assert point['paired_gain_mean'] == pytest.approx(expected)
        assert point['paired_gain_ci95_low'] == pytest.approx(expected)
        assert point['paired_gain_ci95_high'] == pytest.approx(expected)
        assert point['paired_episodes'] == point['return_episodes'] == 5
    assert all(row['solver_seed'] == solver_seed(55, 'episode', row['seed']) for row in result['episodes'])
    chart = publication.chart_payloads(result)
    assert chart['comparison/return_vs_checkpoint']['xs'] == [publication.STEPS] * 2
    assert chart['comparison/paired_gain_vs_checkpoint']['xs'] == [publication.STEPS] * 2


@pytest.mark.parametrize('damage', ['missing_seed', 'duplicate_seed', 'solver_seed', 'short_episode',
                                   'nonfinite_return', 'wrong_recorded_gain', 'reuse_republished'])
def test_invalid_or_republished_measurements_rejected(damage):
    campaign = fixture_campaign(); values = completed(campaign)
    first = values[campaign['cells'][0]['name']]
    if damage == 'missing_seed': first.pop()
    elif damage == 'duplicate_seed': first[0] = deepcopy(first[1])
    elif damage == 'solver_seed': first[0]['solver_seed'] += 1
    elif damage == 'short_episode': first[0]['length'] -= 1
    elif damage == 'nonfinite_return': first[0]['return'] = float('inf')
    elif damage == 'wrong_recorded_gain': first[0]['paired_return_delta'] += 1
    else: values['step_575000'] = episodes(1000)
    with pytest.raises(ValueError):
        publication.aggregate_results(campaign, values)


@pytest.mark.parametrize('damage', ['duplicate_step', 'separate_curve', 'shared_training', 'wrong_prior', 'wrong_reuse',
                                   'wrong_method', 'wrong_axis', 'invalid_alpha'])
def test_publication_scope_cannot_mix_checkpoints_or_run_ownership(damage):
    campaign = fixture_campaign(); first = campaign['cells'][0]
    if damage == 'duplicate_step': campaign['cells'][-1] = deepcopy(first)
    elif damage == 'separate_curve': campaign['cells'][1]['performance_run_id'] = 'other-curve'
    elif damage == 'shared_training': campaign['cells'][1]['training_run_id'] = first['training_run_id']
    elif damage == 'wrong_prior': first['prior_reference']['checkpoint_sha256'] = 'wrong'
    elif damage == 'wrong_reuse': next(c for c in campaign['cells'] if c['reused'])['reuse_reference']['checkpoint_step'] = 600000
    elif damage == 'wrong_method': first['J'] = 12
    elif damage == 'wrong_axis': first['training_decisions'] += 25000
    else: first['initial_alpha'] = 0
    with pytest.raises(ValueError):
        publication.aggregate_results(campaign, {})


def test_adapter_preserves_per_checkpoint_state_and_refuses_historical_publication():
    campaign = fixture_campaign()
    first, second = campaign['cells'][:2]
    one, two = [publication.publication_campaign(campaign, c) for c in (first, second)]
    assert one['checkpoint_step'] == 100000 and two['checkpoint_step'] == 200000
    assert one['checkpoint_sha256'] != two['checkpoint_sha256']
    assert one['cells'][0]['performance_run_id'] == two['cells'][0]['performance_run_id']
    assert one['cells'][0]['training_run_id'] != two['cells'][0]['training_run_id']
    one['cells'][0]['params']['inner_replay_capacity'] = 1
    assert first['params']['inner_replay_capacity'] == 3840
    with pytest.raises(ValueError, match='Never republish'):
        publication.publication_campaign(campaign, next(c for c in campaign['cells'] if c['reused']))


def test_legacy_reuse_execution_proof_uses_recorded_protocol_without_fabricating_new_metric():
    cell = next(c for c in fixture_campaign()['cells'] if c['reused'])
    metrics = {key: dict(mean=value, min=value, max=value) for key, value in dict(
        inner_alpha_initial=cell['initial_alpha'], inner_alpha_final=.01, inner_critic_optimizer_steps=160,
        inner_actor_optimizer_steps=40, inner_temperature_optimizer_steps=40, inner_model_steps=3840,
        inner_buffer_size=3840, inner_compile_fallback=0).items()}
    manifest = dict(code={'commit': 'historical'}, protocol={'action_rule': 'tanh_mean'},
        runs=[dict(resolved_config={'inner_execution_action': 'mean'}, result=dict(model_metrics=metrics))])
    proof = publication.protocol_proof(manifest, cell)
    assert 'inner_eval_execution_sampled' not in proof
    assert proof['execution']['protocol_action_rule'] == 'tanh_mean'
    assert proof['execution']['configured_inner_execution_action'] == 'mean'
    assert 'deterministic_execution' not in proof['execution']
    assert proof['unavailable_historical_metrics'] == ['inner_eval_execution_sampled']
    with pytest.raises(ValueError, match='New evaluations require'):
        publication.protocol_proof(manifest, dict(cell, reused=False))
    manifest['protocol']['action_rule'] = 'squashed_gaussian_sample'
    with pytest.raises(ValueError, match='explicit protocol evidence'):
        publication.protocol_proof(manifest, cell)


def test_shared_curve_publication_uses_blocking_lock_then_releases_it(tmp_path, monkeypatch):
    active, maximum, calls = 0, 0, []
    guard = threading.Lock()
    def publish(directory):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(.02)
        with guard:
            calls.append(directory)
            active -= 1
        return {'run_id': 'curve', 'published': len(calls)}
    monkeypatch.setattr(publication, 'publish_performance', publish)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(publication.serialized_performance, [tmp_path] * 3))
    assert maximum == 1 and len(calls) == 3
    assert sorted(r['published'] for r in results) == [1, 2, 3]


def test_completion_uses_exact_checkpoint_index_not_cumulative_curve_count(tmp_path):
    cell = fixture_campaign()['cells'][0]
    cell.update(directory=str(tmp_path), run_dir=str(tmp_path))
    publication.write(tmp_path / 'publication-completion.json', dict(status='complete', cell=cell['name'],
        training_run_id=cell['training_run_id'], performance=dict(run_id=cell['performance_run_id'], published=24)))
    publication.write(tmp_path / 'training-publication.json', dict(status='complete', run_id=cell['training_run_id']))
    index = {'records': {'record': dict(checkpoint_step=100000, checkpoint_sha256=cell['checkpoint_sha256'], status='published')}}
    publication.write(tmp_path / 'publication.json', index)
    assert publication.publication_complete(cell)
    index['records']['record']['checkpoint_step'] = 200000
    publication.write(tmp_path / 'publication.json', index)
    with pytest.raises(ValueError, match='identity mismatch'):
        publication.publication_complete(cell)


def test_per_cell_adapter_invokes_existing_full_publisher_with_serialized_curve_hook(tmp_path, monkeypatch):
    from slurm import ambi_aux_hj_sweep as generic
    campaign = fixture_campaign()
    cell = campaign['cells'][0]
    cell['directory'] = str(tmp_path / 'step_100000')
    (tmp_path / 'step_100000').mkdir()
    publication.write(tmp_path / 'campaign.json', campaign)
    calls = []
    monkeypatch.setattr(publication, 'load_completed', lambda *args: calls.append('validate'))
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: bool(len(calls) == 2))
    def publish(args, *, performance_publisher):
        adapter = publication.read(args.root / 'campaign.json')
        assert args.index == 0 and len(adapter['cells']) == 1
        assert adapter['checkpoint_step'] == 100000 and adapter['checkpoint_sha256'] == 'checkpoint-100000'
        assert adapter['inventory'] == campaign['inventory'] and adapter['source_run'] == campaign['source_run']
        assert adapter['cells'][0]['params']['inner_replay_capacity'] == 3840
        assert adapter['cells'][0]['performance_run_id'] == 'new-performance'
        assert performance_publisher is publication.serialized_performance
        calls.append('publish')
    monkeypatch.setattr(generic, 'publish_cell', publish)
    publication.publish_cell(SimpleNamespace(root=tmp_path, index=0))
    assert calls == ['validate', 'publish']


def test_existing_full_trace_publisher_accepts_checkpoint_adapter_and_injected_curve_owner(tmp_path, monkeypatch):
    from slurm import ambi_aux_hj_sweep as generic
    from utils import ambi_benchmark, ambi_diagnostic_series, eval_series, eval_series_data
    campaign = fixture_campaign(); cell = campaign['cells'][0]
    cell.update(directory=str(tmp_path), bundle=str(tmp_path / 'bundle'), actual_selector='sweep/return_return')
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    cfg = dict(inner_critic_target_tau=.01, inner_replay_capacity=3840)
    manifest = dict(code={'commit': 'tested'}, runs=[dict(resolved_config=cfg, trace_files=[])])
    generic.write(bundle / 'manifest.json', manifest)
    generic.write(tmp_path / 'worker-completion.json', dict(manifest_sha256=generic.digest(bundle / 'manifest.json'), trace_sha256={}))
    adapter_root = tmp_path / 'adapter'; adapter_root.mkdir()
    generic.write(adapter_root / 'campaign.json', publication.publication_campaign(campaign, cell))
    monkeypatch.setattr(generic, 'validate', lambda *a, **k: manifest)
    record = dict(identity={'test': 'same'}, metrics={'eval/paired_episodes': 5}, episodes=episodes(100))
    monkeypatch.setattr(eval_series_data, 'load_records', lambda *a, **k: [record])
    monkeypatch.setattr(eval_series, 'load_run', lambda *a: {'identity': record['identity']})
    monkeypatch.setattr(ambi_benchmark, 'stage_completed_bundle', lambda *a, **k: {cell['actual_selector']: {'status': 'queued'}})
    monkeypatch.setattr(generic, 'training_summary', lambda *a: dict(update_curves=[], per_seed_decisions=[], decision_curves=[]))
    monkeypatch.setattr(ambi_diagnostic_series, 'record_from_model_bundle', lambda *a, **k: dict(status='complete', rows=[{}] * 27500))
    monkeypatch.setattr(ambi_diagnostic_series, 'write_diagnostic_bundle', lambda *a: None)
    monkeypatch.setattr(ambi_diagnostic_series, 'diagnostic_history', lambda *a: [])
    calls, configs = [], []
    run = SimpleNamespace(summary={}, define_metric=lambda *a, **k: None, log=lambda *a: None,
                          log_artifact=lambda *a: None, finish=lambda **k: None)
    def init(**kwargs):
        configs.append(kwargs)
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init,
        Artifact=lambda *a, **k: SimpleNamespace(add_file=lambda *a, **k: None)))
    def owned_publish(run_dir):
        calls.append(run_dir)
        return dict(run_id=cell['performance_run_id'], published=17, accepted=17)
    generic.publish_cell(SimpleNamespace(root=adapter_root, index=0), performance_publisher=owned_publish)
    assert calls == [cell['run_dir']]
    assert configs[0]['name'].endswith('100k')
    assert configs[0]['config']['checkpoint_step'] == 100000
    assert configs[0]['config']['checkpoint_sha256'] == 'checkpoint-100000'
    assert configs[0]['config']['resolved_config'] == cfg
    assert run.summary['diagnostic/paired_rows'] == 27500
    assert generic.read(tmp_path / 'publication-completion.json')['performance']['published'] == 17


def test_status_separates_evaluation_publication_and_historical_reuse(monkeypatch):
    campaign = fixture_campaign()
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: cell['checkpoint_step'] == 400000)
    values = completed(campaign)
    aggregate = publication.aggregate_results(campaign, values)
    statuses = publication.status_rows(campaign, aggregate, values, {1: object()}, {2: 'upload failed'})
    assert [s['status'] for s in statuses[:4]] == ['evaluated_awaiting_publication', 'publishing', 'publication_failed', 'published']
    assert next(s for s in statuses if s['training_decisions'] == 575000)['status'] == 'reused'
    pending = publication.status_rows(campaign, publication.aggregate_results(campaign, {}), {}, {}, {}, terminal=True)
    assert pending[0]['status'] == 'evaluation_incomplete'


def test_real_sdk_disabled_payload_has_scalar_tables_and_valid_numeric_axes(tmp_path):
    import wandb
    campaign = fixture_campaign()
    aggregate = publication.aggregate_results(campaign, {})
    statuses = [{**p, 'status': 'reused' if p['reused'] else 'queued_or_running'} for p in aggregate['points']]
    run = wandb.init(mode='disabled', dir=str(tmp_path))
    try:
        payload = publication.overview_log(wandb, aggregate, statuses)
        table = payload['comparison/points']
        assert table.columns == publication.POINT_COLUMNS and len(table.data) == 25
        assert all(not isinstance(value, (dict, list, tuple)) for row in table.data for value in row)
        assert len(payload['comparison/paired_episodes'].data) == 5
        run.define_metric('checkpoint/training_decisions')
        run.define_metric('checkpoint_sweep/*', step_metric='checkpoint/training_decisions')
        for _, row in publication.numeric_rows(aggregate):
            run.log(row)
        run.log(payload)
    finally:
        run.finish()


def fake_wandb(monkeypatch):
    logged, configs = [], []
    run = SimpleNamespace(summary={}, log=lambda row: logged.append(deepcopy(row)),
        define_metric=lambda *a, **k: None, finish=lambda **k: None)
    def init(**kwargs):
        configs.append(kwargs)
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init, Table=lambda **k: k,
        plot=SimpleNamespace(line_series=lambda **k: k)))
    return run, logged, configs


def test_finished_uploads_refresh_overview_without_republishing_or_touching_575k(tmp_path, monkeypatch):
    campaign = fixture_campaign(); values = completed(campaign)
    for cell in campaign['cells']:
        directory = tmp_path / cell['name']; directory.mkdir()
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'))
        (directory / 'bundle').mkdir()
        publication.write(directory / 'bundle' / 'manifest.json', {})
        if not cell['reused']:
            publication.write(directory / 'worker-completion.json', {})
    publication.write(tmp_path / 'campaign.json', campaign)
    run, logged, configs = fake_wandb(monkeypatch)
    monkeypatch.setattr(publication, 'verify_references', lambda campaign: None)
    monkeypatch.setattr(publication, 'protocol_proof', lambda manifest, cell: {'initial_alpha': cell['initial_alpha']})
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: True)
    monkeypatch.setattr(publication, 'load_completed', lambda campaign, cell: (values[cell['name']], {'initial_alpha': cell['initial_alpha']}))
    def unexpected(*args, **kwargs):
        pytest.fail('Already published and historical results must not launch publishers')
    monkeypatch.setattr(publication.subprocess, 'run', unexpected)
    publication.watch(SimpleNamespace(root=tmp_path))
    assert run.summary['status'] == 'complete'
    assert run.summary['evaluated'] == 25 and run.summary['published'] == 24 and run.summary['reused'] == 1
    assert len([row for row in logged if 'checkpoint/training_decisions' in row]) == 50
    assert configs[0]['config']['checkpoint_steps'] == publication.STEPS
    assert configs[0]['config']['H'] == 3 and configs[0]['config']['J'] == 10
    assert publication.read(tmp_path / 'campaign-completion.json')['status'] == 'complete'
    old = next(c for c in campaign['cells'] if c['reused'])
    assert not (tmp_path / old['name'] / 'publication-adapter').exists()
