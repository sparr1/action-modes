"""Critic-only LoRA comparisons, paired dense controls, and publication ownership."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_critic_lora_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset):
    return [dict(seed=s, solver_seed=solver_seed(55, 'episode', s), length=500,
        truncated_by_evaluator=False, **{'return': float(offset + s - 100)}) for s in publication.SEEDS]


def fixture_campaign():
    prior = dict(episodes=episodes(350), checkpoint_step=575000, checkpoint_sha256='575k',
        performance_run_id='prior', manifest_sha256='prior-manifest')
    cells = []
    for rank in publication.RANKS:
        for h in publication.HORIZONS:
            for j in publication.ROUNDS:
                name = f'lora_r{rank}_h{h}_j{j}_c16'
                uniform = dict(deepcopy(prior), H=h, J=j, episodes=episodes(350 + h*j),
                    performance_run_id=f'uniform-h{h}-j{j}', training_run_id=f'uniform-training-h{h}-j{j}')
                cells.append(dict(name=name, J=j, H=h, lora_rank=rank, lora_layers='input_hidden',
                    lora_scale=1., lora_weight_decay=.0002, replay_strategy='uniform', checkpoint_step=575000,
                    training_decisions=575000, checkpoint_sha256='575k', reused=False, initial_alpha=.0042,
                    estimator='one_step', execution_mode='mean', alpha_mode='adaptive', critic_kind='return_only',
                    performance_run_id=f'performance-{name}', training_run_id=f'training-{name}',
                    prior_reference=deepcopy(prior), uniform_reference=uniform,
                    params=dict(inner_replay_strategy='uniform', inner_actor_adaptation='clone',
                        inner_critic_adaptation='lora_rl', inner_critic_lora_layers='input_hidden',
                        inner_critic_lora_rank=rank, inner_critic_lora_scale=1., inner_critic_lora_weight_decay=.0002,
                        inner_replay_capacity=max(3072, 384*j), inner_critic_updates_per_round=16,
                        inner_actor_updates_per_round=4, inner_actor_source='sac', inner_horizon_actor_source='sac',
                        inner_critic_source='aux_return', inner_horizon_critic_source='aux_return',
                        inner_sac_critic_target='reward_only', inner_terminal_entropy='none'),
                    directory=f'directory-{name}', bundle=f'bundle-{name}', run_dir=f'run-{name}'))
    return dict(H=publication.HORIZONS, J=publication.ROUNDS, ranks=publication.RANKS,
        cells=cells, group='test-lora', label='575k critic LoRA', source_run='backbone', source_commit='tested',
        inventory='inventory.json', overview_run_id='overview', publisher_workers=3)


def completed(campaign):
    values = {}
    for cell in campaign['cells']:
        gain = cell['J'] * cell['lora_rank'] / 16
        values[cell['name']] = episodes(350 + cell['H']*cell['J'] + gain)
        for ep in values[cell['name']]: ep['paired_return_delta'] = cell['H']*cell['J'] + gain
    return values


def test_initial_overview_has_only_reused_controls_and_no_fabricated_lora_values():
    result = publication.aggregate_results(fixture_campaign(), {})
    assert (result['evaluated'], result['new_evaluated'], result['reused']) == (0, 0, 0)
    assert result['reused_dense_controls'] == 18
    assert len(result['points']) == 36 and len(result['references']) == 19 and not result['episodes']
    for point in result['points']:
        assert point['measurement'] == 'new critic-only LoRA evaluation' and not point['reused']
        assert point['return_mean'] is point['lora_minus_dense_mean'] is None
        assert point['uniform_return_mean'] is not None
    charts = publication.chart_payloads(result)
    returns = charts['comparison/return_vs_J']
    assert len(returns['keys']) == 10
    for label, xs in zip(returns['keys'], returns['xs']):
        assert xs == ([] if 'LoRA' in label else publication.ROUNDS)
    assert charts['comparison/lora_minus_dense_vs_J']['xs'][1:] == [[]] * 18
    assert len(publication.numeric_rows(result)) == 18
    assert all(len(charts[f'comparison/H{h}_return_vs_J']['keys']) == 3 for h in publication.HORIZONS)


def test_full_grid_pairs_real_seeds_with_fixed_bootstrap_and_separate_rank_h_metrics():
    campaign = fixture_campaign()
    result = publication.aggregate_results(campaign, {k: list(reversed(v)) for k, v in completed(campaign).items()})
    assert (result['evaluated'], result['new_evaluated'], result['reused']) == (36, 36, 0)
    assert len(result['episodes']) == 180 and len(result['references']) == 19
    assert (result['bootstrap_seed'], result['bootstrap_resamples']) == (20260912, 2000)
    for point in result['points']:
        gain = point['J'] * point['lora_rank'] / 16
        assert point['lora_minus_dense_mean'] == point['lora_minus_dense_ci95_low'] == point['lora_minus_dense_ci95_high'] == gain
        assert point['paired_gain_mean'] == point['H'] * point['J'] + gain
        assert point['return_episodes'] == point['paired_episodes'] == 5
    rows = publication.numeric_rows(result)
    assert len(rows) == len({identity for identity, _ in rows}) == 54
    for identity, row in rows:
        prefixes = {key.rsplit('/', 1)[0] for key in row if key.startswith('critic_lora_sweep/')}
        assert len(prefixes) == 1
        if not identity.endswith('/references'):
            rank = next(c['lora_rank'] for c in campaign['cells'] if c['name'] == identity)
            assert f'/r{rank}/' in prefixes.pop()


def test_partial_grid_keeps_each_rank_at_its_measured_j_and_shares_dense_references():
    campaign = fixture_campaign(); values = completed(campaign)
    selected = {campaign['cells'][i]['name']: values[campaign['cells'][i]['name']] for i in (0, 1, 18)}
    result = publication.aggregate_results(campaign, selected)
    chart = publication.chart_payloads(result)['comparison/H1_return_vs_J']
    assert chart['xs'] == [publication.ROUNDS, [1, 2], [1]]
    assert len(result['episodes']) == 15 and result['evaluated'] == 3
    assert len(publication.numeric_rows(result)) == 21


@pytest.mark.parametrize('damage', ['missing_cell', 'duplicate_cell', 'shared_performance', 'shared_training',
    'shared_run_dir', 'wrong_prior', 'wrong_uniform', 'uniform_j', 'wrong_checkpoint', 'wrong_method',
    'different_alpha', 'invalid_alpha', 'replay_too_small', 'reused_j1', 'wrong_rank', 'wrong_grid',
    'actor_lora', 'critic_clone', 'wrong_rank_config', 'wrong_scale', 'wrong_decay', 'wrong_layers', 'ere',
    'rank_controls_disagree', 'rank_reference_ids_disagree', 'new_overwrites_uniform',
    'new_overwrites_other_uniform', 'overview_overwrites_prior'])
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
    elif damage == 'replay_too_small': cell['params']['inner_replay_capacity'] = 128
    elif damage == 'reused_j1': campaign['cells'][0]['reused'] = True
    elif damage == 'wrong_rank': cell['lora_rank'] = 64
    elif damage == 'wrong_grid': campaign['ranks'] = [16, 64]
    elif damage == 'actor_lora': cell['params']['inner_actor_adaptation'] = 'lora_rl'
    elif damage == 'critic_clone': cell['params']['inner_critic_adaptation'] = 'clone'
    elif damage == 'wrong_rank_config': cell['params']['inner_critic_lora_rank'] = 96
    elif damage == 'wrong_scale': cell['params']['inner_critic_lora_scale'] = 2.
    elif damage == 'wrong_decay': cell['params']['inner_critic_lora_weight_decay'] = 0.
    elif damage == 'wrong_layers': cell['lora_layers'] = 'hidden'
    elif damage == 'ere': cell['params']['inner_replay_strategy'] = 'ere'
    elif damage == 'rank_controls_disagree': campaign['cells'][19]['uniform_reference']['episodes'][0]['return'] += 1
    elif damage == 'rank_reference_ids_disagree': campaign['cells'][19]['uniform_reference']['training_run_id'] = 'other-control'
    elif damage == 'new_overwrites_uniform': cell['performance_run_id'] = cell['uniform_reference']['performance_run_id']
    elif damage == 'new_overwrites_other_uniform': cell['performance_run_id'] = campaign['cells'][2]['uniform_reference']['performance_run_id']
    else: campaign['overview_run_id'] = cell['prior_reference']['performance_run_id']
    with pytest.raises(ValueError): publication.aggregate_results(campaign, {})


@pytest.mark.parametrize('damage', ['missing_seed', 'duplicate_seed', 'solver_seed', 'short_episode',
    'nonfinite_return', 'wrong_gain', 'unknown_cell'])
def test_incomplete_or_unpaired_data_never_enters_graph(damage):
    campaign = fixture_campaign(); values = completed(campaign); first = values[campaign['cells'][1]['name']]
    if damage == 'missing_seed': first.pop()
    elif damage == 'duplicate_seed': first[0] = deepcopy(first[1])
    elif damage == 'solver_seed': first[0]['solver_seed'] += 1
    elif damage == 'short_episode': first[0]['length'] -= 1
    elif damage == 'nonfinite_return': first[0]['return'] = float('inf')
    elif damage == 'wrong_gain': first[0]['paired_return_delta'] += 1
    else: values['unknown'] = episodes(1000)
    with pytest.raises(ValueError): publication.aggregate_results(campaign, values)


def test_adapter_preserves_575k_and_full_trace_identity_including_new_j1():
    campaign = fixture_campaign(); cell = campaign['cells'][0]
    adapter = publication.publication_campaign(campaign, cell)
    assert adapter['checkpoint_step'] == 575000 and adapter['checkpoint_sha256'] == '575k'
    assert adapter['cells'][0]['performance_run_id'] == cell['performance_run_id']
    adapter['cells'][0]['params']['inner_critic_lora_rank'] = 96
    assert cell['params']['inner_critic_lora_rank'] == 16
    cell['reused'] = True
    with pytest.raises(ValueError, match='Never publish'):
        publication.publication_campaign(campaign, cell)


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
        assert adapter['cells'][0]['params']['inner_critic_lora_rank'] == 96
        calls.append('publish')
    monkeypatch.setattr(generic, 'publish_cell', publish)
    publication.publish_cell(SimpleNamespace(root=tmp_path, index=35))
    assert calls == ['validate', 'publish']


def test_reference_revalidation_reopens_each_dense_source_once(monkeypatch):
    from slurm import ambi_closed_loop_critic_lora as source
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


def test_completed_load_checks_receipt_before_semantics_and_retains_adaptation_proof(tmp_path, monkeypatch):
    from slurm import ambi_closed_loop_critic_lora as source
    campaign = fixture_campaign(); cell = campaign['cells'][1]
    cell.update(directory=str(tmp_path), bundle=str(tmp_path / 'bundle'))
    publication.write(tmp_path / 'worker-completion.json', dict(cell=cell['name'], status='complete'))
    calls = []
    monkeypatch.setattr(publication, 'verify_receipt', lambda bundle, receipt: calls.append('hashes'))
    adaptation = {key: value for key, value in cell['params'].items()
        if key.startswith(('inner_critic_lora_', 'inner_replay_strategy', 'inner_actor_adaptation', 'inner_critic_adaptation'))}
    manifest = dict(runs=[dict(episodes=episodes(360), resolved_config=adaptation, result=dict(model_metrics={}))])
    def validate(*args): calls.append('semantics'); return manifest
    monkeypatch.setattr(source, 'validate_completed', validate)
    monkeypatch.setattr(publication, 'protocol_proof', lambda *a: {'frozen': True})
    actual, proof = publication.load_completed(campaign, cell)
    assert calls == ['hashes', 'semantics'] and actual == episodes(360)
    assert proof['critic_adaptation'] == adaptation
    publication.write(tmp_path / 'worker-completion.json', dict(cell='other', status='complete'))
    with pytest.raises(ValueError, match='identity'):
        publication.load_completed(campaign, cell)
    assert calls == ['hashes', 'semantics', 'hashes']


def test_statuses_keep_dense_links_and_publication_state_separate(monkeypatch):
    campaign = fixture_campaign(); values = {campaign['cells'][1]['name']: completed(campaign)[campaign['cells'][1]['name']]}
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: cell['J'] == 4)
    rows = publication.status_rows(campaign, publication.aggregate_results(campaign, values), values, {3: None}, {4: 'failed'})
    assert [row['status'] for row in rows[:6]] == ['queued_or_running', 'evaluated_awaiting_publication',
        'published', 'publishing', 'publication_failed', 'queued_or_running']
    assert all(row['performance_url'] and row['training_url'] and row['uniform_performance_url'] and row['uniform_training_url'] for row in rows)
    assert all(row['performance_url'] != row['uniform_performance_url'] for row in rows)


def test_real_sdk_disabled_mode_accepts_empty_plots_and_tables_with_ci_bounds(tmp_path):
    import wandb
    campaign = fixture_campaign(); aggregate = publication.aggregate_results(campaign, {})
    statuses = [dict(point, status='queued_or_running') for point in aggregate['points']]
    run = wandb.init(mode='disabled', dir=str(tmp_path))
    try:
        payload = publication.overview_log(wandb, aggregate, statuses)
        assert payload['campaign/evaluated'] == 0 and payload['campaign/new_published'] == 0
        assert payload['campaign/reused_dense_controls'] == 18
        assert len(payload['comparison/points'].data) == 36 and len(payload['comparison/references'].data) == 19
        assert 'lora_minus_dense_ci95_low' in payload['comparison/points'].columns
        assert 'uniform_training_run_id' in payload['comparison/points'].columns
        run.log(payload)
    finally: run.finish()


def test_watcher_publishes_all_36_new_lora_settings_including_j1(tmp_path, monkeypatch):
    campaign = fixture_campaign(); values = completed(campaign)
    for cell in campaign['cells']:
        directory = tmp_path / cell['name']; directory.mkdir(); (directory / 'bundle').mkdir()
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'))
        publication.write(directory / 'worker-completion.json', {'status': 'complete'})
    publication.write(tmp_path / 'campaign.json', campaign)
    monkeypatch.setattr(publication, 'verify_references', lambda c: None)
    monkeypatch.setattr(publication, 'load_completed', lambda c, cell: (values[cell['name']], {'source': 'new'}))
    published, configs, logs = set(), [], []
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: cell['name'] in published)
    def launch(command, **kwargs):
        cell = campaign['cells'][int(command[-1])]
        assert not cell['reused']
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
    assert final['status'] == 'complete' and final['evaluated'] == final['published'] == 36 and final['reused'] == 0
    assert final['reused_dense_controls'] == 18 and len(published) == 36
    config = configs[0]['config']
    assert config['protocol'] == 'closed-loop-critic-lora-hj-sweep-v1'
    assert config['H'] == [1, 2, 3] and config['J'] == publication.ROUNDS and config['ranks'] == [16, 96]
    assert config['checkpoint_step'] == 575000 and config['inner_actor_adaptation'] == 'clone'
    assert config['inner_critic_adaptation'] == 'lora_rl' and config['inner_replay_strategy'] == 'uniform'
    assert config['new_settings'] == 36 and config['reused_settings'] == 0
    assert len(run.summary['protocol_proof_by_setting']) == 36
    with pytest.raises(RuntimeError, match='already started'):
        publication.watch(SimpleNamespace(root=tmp_path))
