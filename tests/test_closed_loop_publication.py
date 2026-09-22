"""Exact seed pairing and visible scientific axes for closed-loop W&B results."""
from copy import deepcopy

import pytest

from slurm.ambi_closed_loop_publish import (aggregate_results, chart_payloads,
                                           campaign_horizon, gpu_jobs_active,
                                           numeric_rows, overview_log)


@pytest.fixture
def comparison():
    campaign = {'cells': [dict(name=f'{arm}_j{j}', critic_kind=arm, J=j)
                          for arm in ('soft', 'return_only') for j in (1, 2, 4)]}
    prior = [dict(seed=101+i, solver_seed=800+i, length=500,
                  truncated_by_evaluator=False, **{'return': float(10+i)}) for i in range(5)]
    completed = {}
    for cell in campaign['cells']:
        offset = cell['J'] * (2 if cell['critic_kind'] == 'soft' else 5)
        completed[cell['name']] = [{**ep, 'return': ep['return']+offset,
                                    'paired_return_delta': float(offset)} for ep in reversed(prior)]
    return campaign, prior, completed


def test_pairing_matches_seed_and_solver_not_list_order(comparison):
    campaign, prior, completed = comparison
    result = aggregate_results(campaign, prior, completed)
    assert result['prior'] == dict(mean=12., std=pytest.approx(1.58113883), episodes=5)
    assert len(result['episodes']) == 30
    assert [(p['J'], p['difference']['mean'], p['difference']['std'])
            for p in result['direct']] == [(1, 3., 0.), (2, 6., 0.), (4, 12., 0.)]
    assert all(row['return_only_return'] - row['soft_return'] == 3*point['J']
               for point in result['direct'] for row in point['episodes'])


@pytest.mark.parametrize('damage', ['solver', 'duplicate', 'missing', 'capped', 'gain', 'nonfinite'])
def test_mismatched_or_invalid_episodes_cannot_become_graph_points(comparison, damage):
    campaign, prior, completed = comparison
    episodes = completed['soft_j1']
    if damage == 'solver': episodes[0]['solver_seed'] += 1
    if damage == 'duplicate': episodes[0] = deepcopy(episodes[1])
    if damage == 'missing': episodes.pop()
    if damage == 'capped': episodes[0]['truncated_by_evaluator'] = True
    if damage == 'gain': episodes[0]['paired_return_delta'] += 1
    if damage == 'nonfinite': episodes[0]['return'] = float('nan')
    with pytest.raises(ValueError):
        aggregate_results(campaign, prior, completed)


def test_partial_results_leave_missing_rounds_out_and_do_not_connect_critics(comparison):
    campaign, prior, completed = comparison
    partial = {name: completed[name] for name in ('soft_j4', 'return_only_j1')}
    result = aggregate_results(campaign, prior, partial)
    plots = chart_payloads(campaign, result)
    assert plots['comparison/return_vs_J']['keys'] == ['Soft critic', 'Return-only critic', 'Frozen prior (reused)']
    assert plots['comparison/return_vs_J']['xs'] == [[4], [1], [1, 2, 4]]
    assert plots['comparison/return_vs_J']['ys'] == [[20.], [17.], [12., 12., 12.]]
    assert 'comparison/return_minus_soft_vs_J' not in plots
    assert result['direct'] == []


def test_launch_overview_has_baseline_but_no_invented_evaluation(comparison):
    campaign, prior, _ = comparison
    result = aggregate_results(campaign, prior, {})
    payload = chart_payloads(campaign, result)
    assert payload['comparison/return_vs_J']['keys'] == ['Frozen prior (reused)']
    assert payload['comparison/paired_gain_vs_J']['ys'] == [[0., 0., 0.]]
    assert numeric_rows(result) == []


def test_native_history_has_real_J_axis_and_paired_difference(comparison):
    campaign, prior, completed = comparison
    result = aggregate_results(campaign, prior, completed)
    rows = dict(numeric_rows(result))
    assert len(rows) == 9
    assert rows['return_only_j2']['axis/inner_rounds'] == 2
    assert rows['return_only_j2']['closed_loop/return_only/return_mean'] == 22.
    assert rows['return_only_j2']['closed_loop/return_only/paired_gain_mean'] == 10.
    assert rows['return_only_j2']['closed_loop/return_only/return_episodes'] == 5
    assert rows['paired_J2']['closed_loop/return_minus_soft/mean'] == 6.
    assert rows['paired_J2']['closed_loop/return_minus_soft/episodes'] == 5
    assert not any(key.startswith('closed_loop/soft/') for key in rows['return_only_j2'])


def test_log_contains_actual_episode_tables_and_only_real_publications(comparison):
    from types import SimpleNamespace
    fake_wandb = SimpleNamespace(Table=lambda **kw: kw,
                                 plot=SimpleNamespace(line_series=lambda **kw: kw))
    campaign, prior, completed = comparison
    result = aggregate_results(campaign, prior, completed)
    statuses = [dict(setting=c['name'], critic=c['critic_kind'], J=c['J'],
                     status='published' if i == 0 else 'publishing')
                for i, c in enumerate(campaign['cells'])]
    output = overview_log(fake_wandb, campaign, result, statuses)
    assert output['campaign/evaluated'] == 6
    assert output['campaign/published'] == 1
    assert len(output['comparison/paired_episodes']['data']) == 30
    assert len(output['comparison/prior_episodes']['data']) == 5
    assert len(output['comparison/critic_difference_episodes']['data']) == 15
    assert output['comparison/return_minus_soft_vs_J']['ys'] == [[3., 6., 12.]]


@pytest.mark.parametrize('horizon', [2, 3])
def test_overview_horizon_comes_from_the_campaign(horizon):
    assert campaign_horizon({'cells': [{'H': horizon}, {'H': horizon}]}) == horizon


def test_mixed_horizon_cannot_be_mislabeled_as_single_horizon():
    with pytest.raises(ValueError, match='shared rollout horizon'):
        campaign_horizon({'cells': [{'H': 2}, {'H': 3}]})


@pytest.mark.parametrize('queue,expected', [
    ('6596064_2\n', True), ('6596064_[0-5]\n6600000\n', True),
    ('6600000\n', False), ('', False),
])
def test_expired_gpu_array_ids_are_matched_without_querying_them(monkeypatch, queue, expected):
    from slurm import ambi_closed_loop_publish as publisher
    calls = []
    def check(command, **kwargs):
        calls.append(command)
        assert '--jobs' not in command and '6596064' not in command
        return queue
    monkeypatch.setattr(publisher.subprocess, 'check_output', check)
    assert gpu_jobs_active(['6596064']) is expected
    assert len(calls) == 1


def test_queue_errors_are_not_treated_as_completed_jobs(monkeypatch):
    import subprocess
    from slurm import ambi_closed_loop_publish as publisher
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr='Controller unavailable')
    monkeypatch.setattr(publisher.subprocess, 'check_output', unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_jobs_active(['6596064'])


def test_finalize_requires_all_publications_before_any_wandb_mutation(tmp_path):
    import json
    from types import SimpleNamespace
    from slurm.ambi_closed_loop_publish import finalize
    campaign = {'cells': [{'name': 'missing', 'directory': str(tmp_path / 'missing')}]}
    (tmp_path / 'campaign.json').write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match='Missing completed publication'):
        finalize(SimpleNamespace(root=tmp_path))
    assert not (tmp_path / 'overview-finalization.json').exists()


def test_finalize_resumes_only_overview_without_relogging_scientific_rows(comparison, tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from slurm import ambi_closed_loop_publish as publisher
    campaign, prior, completed = comparison
    for cell in campaign['cells']:
        cell.update(training_run_id='training-'+cell['name'], performance_run_id='performance-'+cell['name'])
    campaign.update(reference=str(tmp_path / 'prior'), prior_manifest_sha256='prior-hash',
                    checkpoint_sha256='checkpoint-hash', group='group', overview_run_id='overview')
    (tmp_path / 'campaign.json').write_text(json.dumps(campaign))
    (tmp_path / 'prior').mkdir()
    (tmp_path / 'prior' / 'manifest.json').write_text(json.dumps(
        {'runs': [{'config': {'alg_params': {'inner_operator': 'none'}}, 'episodes': prior}]}))
    monkeypatch.setattr(publisher, 'completed_publications', lambda campaign: completed)
    monkeypatch.setattr(publisher, 'digest', lambda path: 'prior-hash')
    def forbidden(*args, **kwargs):
        raise AssertionError('Finalization must not regenerate native scientific history')
    monkeypatch.setattr(publisher, 'numeric_rows', forbidden)
    remote = SimpleNamespace(state='failed', config={'campaign_group': 'group', 'checkpoint_sha256': 'checkpoint-hash'})
    calls, history = [], []
    run = SimpleNamespace(summary={}, log=history.append, finish=lambda **kwargs: None)
    def init(**kwargs):
        calls.append(kwargs)
        return run
    fake_wandb = SimpleNamespace(Api=lambda **kwargs: SimpleNamespace(run=lambda path: remote),
                                 init=init, Table=lambda **kw: kw,
                                 plot=SimpleNamespace(line_series=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    publisher.finalize(SimpleNamespace(root=tmp_path))
    assert calls == [dict(entity=publisher.ENTITY, project=publisher.PROJECT, id='overview', resume='must', mode='online')]
    assert len(history) == 1 and not any(k.startswith('closed_loop/') for k in history[0])
    assert history[0]['campaign/published'] == 6
    assert run.summary['status'] == 'complete' and run.summary['failure'] is None
    assert json.loads((tmp_path / 'campaign-completion.json').read_text())['status'] == 'complete'
    publisher.finalize(SimpleNamespace(root=tmp_path))
    assert len(calls) == len(history) == 1
