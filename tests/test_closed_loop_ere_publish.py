"""575k ERE paired comparison, exact J1 reuse, and guarded publication ownership."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_ere_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset):
    return [dict(seed=s, solver_seed=solver_seed(55, 'episode', s), length=500,
        truncated_by_evaluator=False, **{'return': float(offset + s - 100)}) for s in publication.SEEDS]


def fixture_campaign():
    prior = dict(episodes=episodes(350), checkpoint_step=575000, checkpoint_sha256='575k',
        performance_run_id='prior', manifest_sha256='prior-manifest')
    cells = []
    for h in publication.HORIZONS:
        for j in publication.ROUNDS:
            name = f'ere_f025_h{h}_j{j}_c16'
            uniform = dict(deepcopy(prior), H=h, J=j, episodes=episodes(350 + h*j),
                performance_run_id=f'uniform-h{h}-j{j}', training_run_id=f'uniform-training-h{h}-j{j}')
            cell = dict(name=name, J=j, H=h, checkpoint_step=575000, training_decisions=575000,
                checkpoint_sha256='575k', reused=j == 1, initial_alpha=.0042,
                estimator='one_step', execution_mode='mean', alpha_mode='adaptive', critic_kind='return_only',
                performance_run_id=uniform['performance_run_id'] if j == 1 else f'performance-{name}',
                training_run_id=uniform['training_run_id'] if j == 1 else f'training-{name}',
                prior_reference=deepcopy(prior), uniform_reference=uniform,
                params=dict(inner_replay_strategy='ere', inner_ere_final_fraction=.25, inner_ere_min_rounds=1,
                    inner_ere_actor=True, inner_replay_capacity=max(3072, 128*h*j), inner_critic_updates_per_round=16,
                    inner_actor_updates_per_round=4, inner_actor_source='sac', inner_horizon_actor_source='sac',
                    inner_critic_source='aux_return', inner_horizon_critic_source='aux_return',
                    inner_sac_critic_target='reward_only', inner_terminal_entropy='none'),
                directory=f'directory-{name}', bundle=f'bundle-{name}', run_dir=f'run-{name}')
            if cell['reused']: cell['reuse_reference'] = deepcopy(uniform)
            cells.append(cell)
    return dict(H=publication.HORIZONS, J=publication.ROUNDS, cells=cells, group='test-ere', label='575k ERE f.25',
        source_run='backbone', source_commit='tested', inventory='inventory.json', overview_run_id='overview', publisher_workers=3)


def completed(campaign):
    values = {}
    for cell in campaign['cells']:
        if cell['reused']: continue
        values[cell['name']] = episodes(350 + cell['H']*cell['J'] + cell['J'])
        for ep in values[cell['name']]: ep['paired_return_delta'] = cell['H']*cell['J'] + cell['J']
    return values


def test_missing_points_are_null_and_j1_has_exact_zero_without_new_measurement():
    result = publication.aggregate_results(fixture_campaign(), {})
    assert (result['evaluated'], result['new_evaluated'], result['reused']) == (3, 0, 3)
    assert len(result['points']) == 18 and len(result['references']) == 19 and len(result['episodes']) == 15
    for point in result['points']:
        if point['J'] == 1:
            assert point['measurement'] == 'reused uniform-equivalent J1'
            assert point['return_mean'] == point['uniform_return_mean']
            assert all(point['ere_minus_uniform_' + suffix] == 0 for suffix in publication.STATS)
        else:
            assert point['return_mean'] is point['ere_minus_uniform_mean'] is None
            assert point['uniform_return_mean'] is not None
    charts = publication.chart_payloads(result)
    assert len(charts['comparison/return_vs_J']['keys']) == 7
    assert charts['comparison/return_vs_J']['xs'][1::2] == [[1], [1], [1]]
    assert all('reused equivalent' in label for label in charts['comparison/return_vs_J']['keys'][1:6:2])
    assert charts['comparison/ere_minus_uniform_vs_J']['xs'][1:] == [[1]] * 9
    assert len(publication.numeric_rows(result)) == 21


def test_full_grid_pairs_real_seeds_with_fixed_bootstrap_and_separate_h_metrics():
    campaign = fixture_campaign()
    result = publication.aggregate_results(campaign, {k: list(reversed(v)) for k, v in completed(campaign).items()})
    assert (result['evaluated'], result['new_evaluated'], result['reused']) == (18, 15, 3)
    assert len(result['episodes']) == 90
    assert (result['bootstrap_seed'], result['bootstrap_resamples']) == (20260912, 2000)
    for point in result['points']:
        gain = 0 if point['J'] == 1 else point['J']
        assert point['ere_minus_uniform_mean'] == point['ere_minus_uniform_ci95_low'] == point['ere_minus_uniform_ci95_high'] == gain
        assert point['paired_gain_mean'] == point['H'] * point['J'] + gain
        assert point['return_episodes'] == point['paired_episodes'] == 5
    rows = publication.numeric_rows(result)
    assert len(rows) == len({identity for identity, _ in rows}) == 36
    for _, row in rows:
        horizons = {key.split('/')[1] for key in row if key.startswith('ere_sweep/')}
        assert len(horizons) == 1


@pytest.mark.parametrize('damage', ['missing_cell', 'duplicate_cell', 'shared_performance', 'shared_training',
    'shared_run_dir', 'wrong_prior', 'wrong_uniform', 'uniform_j', 'wrong_checkpoint', 'wrong_method',
    'different_alpha', 'invalid_alpha', 'wrong_fraction', 'critic_only', 'replay_too_small', 'j2_reuse',
    'j1_new_identity', 'j1_changed_returns', 'new_overwrites_uniform', 'new_overwrites_other_uniform',
    'overview_overwrites_prior'])
def test_scope_rejects_changed_protocol_references_or_publication_ownership(damage):
    campaign = fixture_campaign(); cell = campaign['cells'][1]
    if damage == 'missing_cell': campaign['cells'].pop()
    elif damage == 'duplicate_cell': campaign['cells'][-1] = deepcopy(cell)
    elif damage == 'shared_performance': campaign['cells'][2]['performance_run_id'] = cell['performance_run_id']
    elif damage == 'shared_training': campaign['cells'][2]['training_run_id'] = cell['training_run_id']
    elif damage == 'shared_run_dir': campaign['cells'][2]['run_dir'] = cell['run_dir']
    elif damage == 'wrong_prior': cell['prior_reference']['checkpoint_step'] = 650000
    elif damage == 'wrong_uniform': cell['uniform_reference']['checkpoint_sha256'] = '650k'
    elif damage == 'uniform_j': cell['uniform_reference']['J'] = 4
    elif damage == 'wrong_checkpoint': cell['checkpoint_step'] = 650000
    elif damage == 'wrong_method': cell['execution_mode'] = 'policy_sample'
    elif damage == 'different_alpha': cell['initial_alpha'] = .003
    elif damage == 'invalid_alpha': cell['initial_alpha'] = float('nan')
    elif damage == 'wrong_fraction': cell['params']['inner_ere_final_fraction'] = .5
    elif damage == 'critic_only': cell['params']['inner_ere_actor'] = False
    elif damage == 'replay_too_small': cell['params']['inner_replay_capacity'] = 128
    elif damage == 'j2_reuse': cell['reused'] = True
    elif damage == 'j1_new_identity': campaign['cells'][0]['performance_run_id'] = 'new-j1'
    elif damage == 'j1_changed_returns': campaign['cells'][0]['reuse_reference']['episodes'][0]['return'] += 1
    elif damage == 'new_overwrites_uniform': cell['performance_run_id'] = cell['uniform_reference']['performance_run_id']
    elif damage == 'new_overwrites_other_uniform': cell['performance_run_id'] = campaign['cells'][2]['uniform_reference']['performance_run_id']
    else: campaign['overview_run_id'] = cell['prior_reference']['performance_run_id']
    with pytest.raises(ValueError): publication.aggregate_results(campaign, {})


@pytest.mark.parametrize('damage', ['missing_seed', 'duplicate_seed', 'solver_seed', 'short_episode',
    'nonfinite_return', 'wrong_gain', 'reuse_republished'])
def test_incomplete_or_unpaired_data_never_enters_graph(damage):
    campaign = fixture_campaign(); values = completed(campaign); first = values[campaign['cells'][1]['name']]
    if damage == 'missing_seed': first.pop()
    elif damage == 'duplicate_seed': first[0] = deepcopy(first[1])
    elif damage == 'solver_seed': first[0]['solver_seed'] += 1
    elif damage == 'short_episode': first[0]['length'] -= 1
    elif damage == 'nonfinite_return': first[0]['return'] = float('inf')
    elif damage == 'wrong_gain': first[0]['paired_return_delta'] += 1
    else: values[campaign['cells'][0]['name']] = episodes(1000)
    with pytest.raises(ValueError): publication.aggregate_results(campaign, values)


def test_adapter_preserves_575k_and_full_trace_identity_without_republishing_j1():
    campaign = fixture_campaign(); cell = campaign['cells'][-1]
    adapter = publication.publication_campaign(campaign, cell)
    assert adapter['checkpoint_step'] == 575000 and adapter['checkpoint_sha256'] == '575k'
    assert adapter['cells'][0]['performance_run_id'] == cell['performance_run_id']
    adapter['cells'][0]['params']['inner_ere_final_fraction'] = .5
    assert cell['params']['inner_ere_final_fraction'] == .25
    with pytest.raises(ValueError, match='Never republish'):
        publication.publication_campaign(campaign, campaign['cells'][0])


def test_adapter_calls_full_trace_publisher_only_after_worker_validation(tmp_path, monkeypatch):
    from slurm import ambi_aux_hj_sweep as generic
    campaign = fixture_campaign(); cell = campaign['cells'][-1]
    cell['directory'] = str(tmp_path / cell['name']); publication.Path(cell['directory']).mkdir()
    publication.write(tmp_path / 'campaign.json', campaign)
    calls = []
    monkeypatch.setattr(publication, 'load_completed', lambda *a: calls.append('validate'))
    monkeypatch.setattr(publication, 'publication_complete', lambda c: len(calls) == 2)
    def publish(args):
        adapter = publication.read(args.root / 'campaign.json')
        assert args.index == 0 and adapter['checkpoint_step'] == 575000
        assert adapter['cells'][0]['J'] == 10 and adapter['cells'][0]['H'] == 3
        assert adapter['cells'][0]['params']['inner_ere_final_fraction'] == .25
        calls.append('publish')
    monkeypatch.setattr(generic, 'publish_cell', publish)
    publication.publish_cell(SimpleNamespace(root=tmp_path, index=17))
    assert calls == ['validate', 'publish']


def test_reference_revalidation_reopens_every_historical_source(monkeypatch):
    from slurm import ambi_closed_loop_ere as source
    campaign = fixture_campaign(); calls = []
    def prior(pin, inventory):
        calls.append('prior'); return pin
    def uniform(pin, inventory):
        calls.append((pin['H'], pin['J'])); return pin
    monkeypatch.setattr(source, 'load_prior', prior)
    monkeypatch.setattr(source, 'load_uniform', uniform)
    publication.verify_references(campaign)
    assert calls == ['prior', *[(h, j) for h in publication.HORIZONS for j in publication.ROUNDS]]
    def changed(pin, inventory):
        result = deepcopy(pin); result['episodes'][0]['return'] += 1; return result
    monkeypatch.setattr(source, 'load_uniform', changed)
    with pytest.raises(ValueError, match='changed'):
        publication.verify_references(campaign)


def test_completed_load_checks_receipt_before_semantics_and_retains_replay_proof(tmp_path, monkeypatch):
    from slurm import ambi_closed_loop_ere as source
    campaign = fixture_campaign(); cell = campaign['cells'][1]
    cell.update(directory=str(tmp_path), bundle=str(tmp_path / 'bundle'))
    publication.write(tmp_path / 'worker-completion.json', dict(cell=cell['name'], status='complete'))
    calls = []
    monkeypatch.setattr(publication, 'verify_receipt', lambda bundle, receipt: calls.append('hashes'))
    replay = {key: value for key, value in cell['params'].items() if key.startswith(('inner_replay_strategy', 'inner_ere_'))}
    manifest = dict(runs=[dict(episodes=episodes(360), resolved_config=replay,
        result=dict(model_metrics={'inner_critic_replay_round_1_sample_count': {'mean': 512.}}))])
    def validate(*args): calls.append('semantics'); return manifest
    monkeypatch.setattr(source, 'validate_completed', validate)
    monkeypatch.setattr(publication, 'protocol_proof', lambda *a: {'frozen': True})
    actual, proof = publication.load_completed(campaign, cell)
    assert calls == ['hashes', 'semantics'] and actual == episodes(360)
    assert proof['replay'] == replay and proof['replay_telemetry'] == manifest['runs'][0]['result']['model_metrics']
    publication.write(tmp_path / 'worker-completion.json', dict(cell='other', status='complete'))
    with pytest.raises(ValueError, match='identity'):
        publication.load_completed(campaign, cell)
    assert calls == ['hashes', 'semantics', 'hashes']


def test_reused_j1_status_keeps_original_links_without_new_publication_check(monkeypatch):
    campaign = fixture_campaign(); values = {campaign['cells'][1]['name']: completed(campaign)[campaign['cells'][1]['name']]}
    def complete(cell):
        assert not cell['reused']
        return cell['J'] == 4
    monkeypatch.setattr(publication, 'publication_complete', complete)
    rows = publication.status_rows(campaign, publication.aggregate_results(campaign, values), values, {3: None}, {4: 'failed'})
    assert [row['status'] for row in rows[:6]] == ['reused_uniform_equivalent', 'evaluated_awaiting_publication',
        'published', 'publishing', 'publication_failed', 'queued_or_running']
    assert all(row['performance_url'] and row['training_url'] and row['uniform_performance_url'] and row['uniform_training_url'] for row in rows)
    assert all(row['performance_url'] == row['uniform_performance_url'] for row in rows if row['J'] == 1)


def test_real_sdk_disabled_mode_accepts_plot_and_tables_with_ci_bounds(tmp_path):
    import wandb
    campaign = fixture_campaign(); aggregate = publication.aggregate_results(campaign, {})
    statuses = [dict(point, status='reused_uniform_equivalent' if point['reused'] else 'queued_or_running') for point in aggregate['points']]
    run = wandb.init(mode='disabled', dir=str(tmp_path))
    try:
        payload = publication.overview_log(wandb, aggregate, statuses)
        assert payload['campaign/evaluated'] == 3 and payload['campaign/new_published'] == 0
        assert len(payload['comparison/points'].data) == 18 and len(payload['comparison/references'].data) == 19
        assert 'ere_minus_uniform_ci95_low' in payload['comparison/points'].columns
        assert 'uniform_training_run_id' in payload['comparison/points'].columns
        run.log(payload)
    finally: run.finish()


def test_watcher_publishes_only_fifteen_new_ere_runs_and_preserves_j1(tmp_path, monkeypatch):
    campaign = fixture_campaign(); values = completed(campaign)
    for cell in campaign['cells']:
        directory = tmp_path / cell['name']; directory.mkdir(); (directory / 'bundle').mkdir()
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'))
        publication.write(directory / 'bundle' / 'manifest.json', {'source': 'historical' if cell['reused'] else 'new'})
        if not cell['reused']: publication.write(directory / 'worker-completion.json', {'status': 'complete'})
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
        cell = campaign['cells'][int(command[-1])]
        assert not cell['reused'] and cell['J'] != 1
        published.add(cell['name']); return SimpleNamespace(returncode=0)
    monkeypatch.setattr(publication.subprocess, 'run', launch)
    class Pool:
        def __init__(self, max_workers): assert max_workers == 3
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, function, index):
            value = function(index); return SimpleNamespace(done=lambda: True, result=lambda: value)
    monkeypatch.setattr(publication, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(publication.time, 'sleep', lambda seconds: None)
    run = SimpleNamespace(summary={}, define_metric=lambda *a, **k: None, log=logs.append, finish=lambda **k: None)
    def init(**kwargs): configs.append(kwargs); return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    monkeypatch.setattr(publication, 'overview_log', lambda wandb, aggregate, statuses: {'evaluated': aggregate['evaluated']})
    publication.watch(SimpleNamespace(root=tmp_path))
    final = publication.read(tmp_path / 'campaign-completion.json')
    assert final['status'] == 'complete' and final['evaluated'] == 18 and final['published'] == 15 and final['reused'] == 3
    assert len(published) == 15
    config = configs[0]['config']
    assert config['protocol'] == 'closed-loop-ere-hj-sweep-v1'
    assert config['H'] == [1, 2, 3] and config['J'] == publication.ROUNDS
    assert config['checkpoint_step'] == 575000 and config['inner_ere_final_fraction'] == .25
    assert config['new_settings'] == 15 and config['reused_settings'] == 3
    assert all(proof['source'] == 'historical' and 'equivalence' in proof for name, proof in run.summary['protocol_proof_by_setting'].items() if '_j1_' in name)
