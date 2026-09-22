"""Exact seed pairing and visible scientific axes for closed-loop W&B results."""
from copy import deepcopy

import pytest

from slurm.ambi_closed_loop_publish import (aggregate_results, chart_payloads,
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
