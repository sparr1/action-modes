"""650k J-sweep pairing, immutable J10 reuse, and visible publication payloads."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_j_sweep_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset):
    return [dict(seed=s, solver_seed=solver_seed(55, 'episode', s), length=500,
        truncated_by_evaluator=False, **{'return': float(offset + s - 100)}) for s in publication.SEEDS]


def fixture_campaign(horizon=3):
    prior = dict(episodes=episodes(350), checkpoint_step=650000, checkpoint_sha256='650k',
                 performance_run_id='prior', manifest_sha256='prior-manifest')
    cells = []
    for j in publication.ROUNDS:
        cell = dict(name=f'j{j}', J=j, H=horizon, checkpoint_step=650000, training_decisions=650000,
            checkpoint_sha256='650k', reused=horizon == 3 and j == 10, initial_alpha=.0042,
            estimator='one_step', execution_mode='mean', alpha_mode='adaptive', critic_kind='return_only',
            performance_run_id=f'performance-j{j}', training_run_id=f'training-j{j}',
            prior_reference=deepcopy(prior), params={'inner_replay_capacity': max(3072, 384*j), 'inner_critic_updates_per_round': 16, 'inner_actor_updates_per_round': 4},
            directory=f'directory-j{j}', bundle=f'bundle-j{j}', run_dir=f'run-j{j}')
        if cell['reused']:
            cell['reuse_reference'] = dict(prior, episodes=episodes(344))
        cells.append(cell)
    return dict(H=horizon, cells=cells, group='test-j-sweep', label=f'650k H{horizon} J sweep', source_run='backbone',
        source_commit='tested', inventory='inventory.json', overview_run_id='overview', publisher_workers=3,
        mppi_references={kind: dict(prior, episodes=episodes(value), performance_run_id=kind)
                         for kind, value in [('soft', 428), ('return_only', 379)]})


def completed(campaign):
    return {c['name']: [dict(e, **{'return': e['return'] + c['J']}, paired_return_delta=c['J'])
            for e in c['prior_reference']['episodes']] for c in campaign['cells'] if not c['reused']}


def test_startup_shows_j10_and_fixed_historical_baselines_without_inventing_other_results():
    result = publication.aggregate_results(fixture_campaign(), {})
    assert result['evaluated'] == result['reused'] == 1 and result['new_evaluated'] == 0
    assert len(result['points']) == 8 and len(result['episodes']) == 5
    assert [p['J'] for p in result['points']] == publication.ROUNDS
    assert [r['return_mean'] for r in result['references']] == [353, 431, 382]
    for point in result['points']:
        assert point['prior_return_mean'] == 353
        assert point['mppi_soft_return_mean'] == 431 and point['mppi_return_return_mean'] == 382
        assert point['mppi_soft_paired_gain_mean'] == 78 and point['mppi_return_paired_gain_mean'] == 29
        if point['reused']:
            assert point['return_mean'] == 347 and point['paired_gain_mean'] == -6
        else:
            assert all(point[k] is None for k in ('return_mean', 'return_std', 'return_episodes',
                'paired_gain_mean', 'paired_gain_ci95_low', 'paired_gain_ci95_high', 'paired_episodes'))
    charts = publication.chart_payloads(result)
    assert charts['comparison/return_vs_J']['xs'] == [[10], publication.ROUNDS, publication.ROUNDS, publication.ROUNDS]
    assert 'fixed historical budget' in charts['comparison/return_vs_J']['keys'][2]
    rows = publication.numeric_rows(result)
    assert len(rows) == 9 and all('axis/inner_rounds' in row for _, row in rows)
    assert len({name for name, _ in rows}) == 9


def test_full_sweep_pairs_actual_seeds_not_row_position_with_fixed_bootstrap():
    campaign = fixture_campaign()
    result = publication.aggregate_results(campaign, {k: list(reversed(v)) for k, v in completed(campaign).items()})
    assert result['evaluated'] == 8 and result['new_evaluated'] == 7 and len(result['episodes']) == 40
    assert (result['bootstrap_seed'], result['bootstrap_resamples']) == (20260912, 2000)
    for point in result['points']:
        gain = -6 if point['reused'] else point['J']
        assert point['paired_gain_mean'] == gain
        assert point['paired_gain_ci95_low'] == point['paired_gain_ci95_high'] == gain
        assert point['paired_episodes'] == point['return_episodes'] == 5


@pytest.mark.parametrize('damage', ['duplicate_j', 'shared_performance', 'shared_training', 'shared_run_dir',
    'wrong_prior', 'wrong_reuse', 'wrong_checkpoint', 'wrong_method', 'different_alpha', 'invalid_alpha', 'wrong_mppi'])
def test_scope_rejects_mixed_ownership_checkpoint_or_method(damage):
    campaign = fixture_campaign(); cell = campaign['cells'][0]
    if damage == 'duplicate_j': campaign['cells'][-1] = deepcopy(cell)
    elif damage == 'shared_performance': campaign['cells'][1]['performance_run_id'] = cell['performance_run_id']
    elif damage == 'shared_training': campaign['cells'][1]['training_run_id'] = cell['training_run_id']
    elif damage == 'shared_run_dir': campaign['cells'][1]['run_dir'] = cell['run_dir']
    elif damage == 'wrong_prior': cell['prior_reference']['checkpoint_step'] = 575000
    elif damage == 'wrong_reuse': campaign['cells'][5]['reuse_reference']['checkpoint_sha256'] = '575k'
    elif damage == 'wrong_checkpoint': cell['checkpoint_step'] = 675000
    elif damage == 'wrong_method': cell['execution_mode'] = 'policy_sample'
    elif damage == 'different_alpha': cell['initial_alpha'] = .003
    elif damage == 'invalid_alpha': cell['initial_alpha'] = float('nan')
    else: campaign['mppi_references']['soft']['checkpoint_sha256'] = '575k'
    with pytest.raises(ValueError): publication.aggregate_results(campaign, {})


@pytest.mark.parametrize('damage', ['missing_seed', 'duplicate_seed', 'solver_seed', 'short_episode',
    'nonfinite_return', 'wrong_gain', 'reuse_republished'])
def test_incomplete_or_unpaired_data_never_enters_graph(damage):
    campaign = fixture_campaign(); values = completed(campaign); first = values['j1']
    if damage == 'missing_seed': first.pop()
    elif damage == 'duplicate_seed': first[0] = deepcopy(first[1])
    elif damage == 'solver_seed': first[0]['solver_seed'] += 1
    elif damage == 'short_episode': first[0]['length'] -= 1
    elif damage == 'nonfinite_return': first[0]['return'] = float('inf')
    elif damage == 'wrong_gain': first[0]['paired_return_delta'] += 1
    else: values['j10'] = episodes(1000)
    with pytest.raises(ValueError): publication.aggregate_results(campaign, values)


def test_adapter_preserves_checkpoint_and_per_j_replay_without_touching_j10():
    campaign = fixture_campaign()
    first, last = campaign['cells'][0], campaign['cells'][-1]
    one, two = [publication.publication_campaign(campaign, c) for c in (first, last)]
    assert one['checkpoint_step'] == two['checkpoint_step'] == 650000
    assert one['cells'][0]['performance_run_id'] != two['cells'][0]['performance_run_id']
    assert one['cells'][0]['params']['inner_replay_capacity'] == 3072
    assert two['cells'][0]['params']['inner_replay_capacity'] == 5376
    one['cells'][0]['params']['inner_replay_capacity'] = 1
    assert first['params']['inner_replay_capacity'] == 3072
    with pytest.raises(ValueError, match='Never republish'):
        publication.publication_campaign(campaign, campaign['cells'][5])


def test_per_j_adapter_calls_complete_training_publisher_with_correct_checkpoint(tmp_path, monkeypatch):
    from slurm import ambi_aux_hj_sweep as generic
    campaign = fixture_campaign(); cell = campaign['cells'][-1]
    cell['directory'] = str(tmp_path / 'j14'); (tmp_path / 'j14').mkdir()
    publication.write(tmp_path / 'campaign.json', campaign)
    calls = []
    monkeypatch.setattr(publication, 'load_completed', lambda *a: calls.append('validate'))
    monkeypatch.setattr(publication, 'publication_complete', lambda c: len(calls) == 2)
    def publish(args):
        adapter = publication.read(args.root / 'campaign.json')
        assert args.index == 0 and adapter['checkpoint_step'] == 650000
        assert adapter['checkpoint_sha256'] == '650k' and adapter['inventory'] == campaign['inventory']
        assert adapter['cells'][0]['J'] == 14 and adapter['cells'][0]['params']['inner_replay_capacity'] == 5376
        calls.append('publish')
    monkeypatch.setattr(generic, 'publish_cell', publish)
    publication.publish_cell(SimpleNamespace(root=tmp_path, index=7))
    assert calls == ['validate', 'publish']


def test_reused_j10_status_does_not_recheck_shared_curve_cumulative_counter(monkeypatch):
    campaign = fixture_campaign(); values = {'j1': completed(campaign)['j1']}
    def complete(cell):
        assert not cell['reused'], 'Reused J10 cannot be judged by new publication receipts'
        return cell['J'] == 2
    monkeypatch.setattr(publication, 'publication_complete', complete)
    rows = publication.status_rows(campaign, publication.aggregate_results(campaign, values), values, {2: None}, {3: 'failed'})
    assert [r['status'] for r in rows] == ['evaluated_awaiting_publication', 'published', 'publishing',
        'publication_failed', 'queued_or_running', 'reused', 'queued_or_running', 'queued_or_running']
    assert all(r['performance_url'] and r['training_url'] for r in rows)


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_real_sdk_disabled_mode_accepts_actual_plot_and_table_payload(tmp_path, horizon):
    import wandb
    campaign = fixture_campaign(horizon); aggregate = publication.aggregate_results(campaign, {})
    statuses = [dict(point, status='reused' if point['reused'] else 'queued_or_running') for point in aggregate['points']]
    run = wandb.init(mode='disabled', dir=str(tmp_path))
    try:
        payload = publication.overview_log(wandb, aggregate, statuses)
        assert payload['campaign/evaluated'] == int(horizon == 3) and payload['campaign/new_published'] == 0
        assert len(payload['comparison/points'].data) == 8
        assert len(payload['comparison/references'].data) == 3
        assert 'mppi_soft_return_mean' in payload['comparison/points'].columns
        assert 'paired_gain_ci95_low' in payload['comparison/points'].columns
        run.log(payload)
    finally:
        run.finish()


def test_generic_full_trace_publisher_uses_650k_j14_adapter_and_37500_probe_rows(tmp_path, monkeypatch):
    from slurm import ambi_aux_hj_sweep as generic
    from utils import ambi_benchmark, ambi_diagnostic_series, eval_series, eval_series_data
    campaign = fixture_campaign(); cell = campaign['cells'][-1]
    cell.update(directory=str(tmp_path), bundle=str(tmp_path / 'bundle'), actual_selector='sweep/j14')
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    cfg = dict(inner_critic_target_tau=.01, inner_replay_capacity=5376)
    manifest = dict(code={'commit': 'tested'}, runs=[dict(resolved_config=cfg, trace_files=[])])
    generic.write(bundle / 'manifest.json', manifest)
    generic.write(tmp_path / 'worker-completion.json', dict(manifest_sha256=generic.digest(bundle / 'manifest.json'), trace_sha256={}))
    adapter_root = tmp_path / 'adapter'; adapter_root.mkdir()
    generic.write(adapter_root / 'campaign.json', publication.publication_campaign(campaign, cell))
    def validate(*args, **kwargs):
        assert kwargs['checkpoint_step'] == 650000 and kwargs['checkpoint_sha'] == '650k'
        return manifest
    monkeypatch.setattr(generic, 'validate', validate)
    record = dict(identity={'test': 'same'}, metrics={'eval/paired_episodes': 5}, episodes=episodes(100))
    monkeypatch.setattr(eval_series_data, 'load_records', lambda *a, **k: [record])
    monkeypatch.setattr(eval_series, 'load_run', lambda *a: {'identity': record['identity']})
    monkeypatch.setattr(ambi_benchmark, 'stage_completed_bundle', lambda *a, **k: {cell['actual_selector']: {'status': 'queued'}})
    monkeypatch.setattr(generic, 'training_summary', lambda *a: dict(update_curves=[], per_seed_decisions=[], decision_curves=[]))
    monkeypatch.setattr(ambi_diagnostic_series, 'record_from_model_bundle', lambda *a, **k: dict(status='complete', rows=[{}] * 37500))
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
        return dict(run_id=cell['performance_run_id'], published=1, accepted=1)
    generic.publish_cell(SimpleNamespace(root=adapter_root, index=0), performance_publisher=owned_publish)
    assert calls == [cell['run_dir']]
    assert configs[0]['name'].endswith('650k')
    assert configs[0]['config']['checkpoint_step'] == 650000
    assert configs[0]['config']['J'] == 14 and configs[0]['config']['inner_replay_capacity'] == 5376
    assert run.summary['diagnostic/paired_rows'] == 37500
    assert run.summary['training/critic_updates'] == 2500 * 16 * 14
    assert run.summary['training/actor_updates'] == 2500 * 4 * 14
    assert generic.read(tmp_path / 'publication-completion.json')['status'] == 'complete'


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_watcher_finishes_all_new_publications_and_preserves_only_h3_j10(tmp_path, monkeypatch, horizon):
    campaign = fixture_campaign(horizon); values = completed(campaign)
    reused = int(horizon == 3)
    for cell in campaign['cells']:
        directory = tmp_path / cell['name']; directory.mkdir()
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'))
        (directory / 'bundle').mkdir()
        publication.write(directory / 'bundle' / 'manifest.json', {'source': 'historical' if cell['reused'] else 'new'})
        if not cell['reused']:
            publication.write(directory / 'worker-completion.json', {'status': 'complete'})
    publication.write(tmp_path / 'campaign.json', campaign)
    monkeypatch.setattr(publication, 'verify_references', lambda c: None)
    monkeypatch.setattr(publication, 'protocol_proof', lambda manifest, cell: manifest)
    monkeypatch.setattr(publication, 'load_completed', lambda c, cell: (values[cell['name']], {'source': 'new'}))
    published, configs, logs = set(), [], []
    def complete(cell):
        assert not cell['reused']
        return cell['name'] in published
    monkeypatch.setattr(publication, 'publication_complete', complete)
    def launch(command, **kwargs):
        index = int(command[-1]); cell = campaign['cells'][index]
        assert not cell['reused']
        assert horizon != 3 or cell['J'] != 10
        published.add(cell['name'])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(publication.subprocess, 'run', launch)
    class Pool:
        def __init__(self, max_workers): assert max_workers == 3
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, function, index):
            value = function(index)
            return SimpleNamespace(done=lambda: True, result=lambda: value)
    monkeypatch.setattr(publication, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(publication.time, 'sleep', lambda seconds: None)
    run = SimpleNamespace(summary={}, define_metric=lambda *a, **k: None, log=logs.append, finish=lambda **k: None)
    def init(**kwargs):
        configs.append(kwargs)
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    monkeypatch.setattr(publication, 'overview_log', lambda wandb, aggregate, statuses: {'evaluated': aggregate['evaluated']})
    publication.watch(SimpleNamespace(root=tmp_path))
    final = publication.read(tmp_path / 'campaign-completion.json')
    assert final['status'] == 'complete' and final['evaluated'] == 8 and final['published'] == 8 - reused
    assert final['reused'] == reused and ('j10' not in published) == bool(reused)
    config = configs[0]['config']
    assert config['protocol'] == ('closed-loop-h3-j-sweep-v1' if horizon == 3 else 'closed-loop-hj-sweep-v1')
    assert config['H'] == horizon
    assert config['new_settings'] == 8 - reused and config['reused_settings'] == reused
    assert config['checkpoint_step'] == 650000 and config['J'] == publication.ROUNDS
    assert config['C'] == 16 and config['A'] == 4
    assert config['inner_replay_capacity_by_J'] == {str(j): max(3072, 384*j) for j in publication.ROUNDS}
    assert run.summary['protocol_proof_by_setting']['j10'] == {'source': 'historical' if horizon == 3 else 'new'}
    assert all(row['H'] == horizon for row in final['rows'])
    assert len([row for row in logs if 'j_sweep/return_mean' in row]) == 8
    assert len([row for row in logs if 'j_sweep/prior_return_mean' in row]) == 8


@pytest.mark.parametrize('horizon', [1, 2])
def test_new_horizon_starts_with_only_historical_baselines_and_all_eight_pending(horizon, monkeypatch):
    campaign = fixture_campaign(horizon)
    result = publication.aggregate_results(campaign, {})
    assert result['H'] == horizon and result['evaluated'] == result['reused'] == result['new_evaluated'] == 0
    assert len(result['points']) == 8 and result['episodes'] == []
    assert all(point['H'] == horizon and point['return_mean'] is None and not point['reused'] for point in result['points'])
    charts = publication.chart_payloads(result)
    for chart in charts.values():
        assert chart['xs'] == [publication.ROUNDS] * 3
        assert len(chart['ys']) == len(chart['keys']) == 3
        assert all('refinement' not in label for label in chart['keys'])
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: False)
    statuses = publication.status_rows(campaign, result, {}, {}, {})
    assert len(statuses) == 8 and all(row['status'] == 'queued_or_running' and row['H'] == horizon for row in statuses)
    full = publication.aggregate_results(campaign, completed(campaign))
    assert full['evaluated'] == full['new_evaluated'] == 8 and full['reused'] == 0
    assert len(full['episodes']) == 40 and all(row['H'] == horizon for row in full['episodes'])
    assert publication.chart_payloads(full)['comparison/return_vs_J']['keys'][0] == f'H{horizon} refinement'


@pytest.mark.parametrize('horizon', [1, 2])
def test_new_horizon_reference_check_never_loads_h3_j10(horizon, monkeypatch):
    from slurm import ambi_closed_loop_j_sweep as source
    campaign = fixture_campaign(horizon); calls = []
    def prior(pin, inventory):
        calls.append('prior'); return pin
    def reused(pin, inventory):
        raise AssertionError('H1/H2 cannot use H3/J10 as their measurement')
    def mppi(pin, inventory, prior):
        calls.append('mppi'); return campaign['mppi_references']
    monkeypatch.setattr(source, 'load_prior', prior)
    monkeypatch.setattr(source, 'load_reused', reused)
    monkeypatch.setattr(source, 'load_mppi_references', mppi)
    publication.verify_references(campaign)
    assert calls == ['prior', 'mppi']


@pytest.mark.parametrize('damage', ['mixed_horizons', 'unsupported_horizon', 'declared_horizon', 'h1_reuses_j10', 'h2_reuses_j10'])
def test_horizon_scope_cannot_mix_methods_or_reuse_h3_results(damage):
    campaign = fixture_campaign()
    if damage == 'mixed_horizons': campaign['cells'][0]['H'] = 1
    elif damage == 'unsupported_horizon':
        campaign['H'] = 4
        for cell in campaign['cells']: cell['H'] = 4
    elif damage == 'declared_horizon': campaign['H'] = 2
    else:
        horizon = 1 if damage == 'h1_reuses_j10' else 2
        campaign['H'] = horizon
        for cell in campaign['cells']: cell['H'] = horizon
    with pytest.raises(ValueError): publication.aggregate_results(campaign, {})
