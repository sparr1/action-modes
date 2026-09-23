"""Exact-axis paired contrasts, incomplete results, and publication recovery."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_soft_return_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset=0):
    return [dict(seed=s, solver_seed=solver_seed(55, 'episode', s), length=500,
                 truncated_by_evaluator=False, **{'return': float(s - 100 + offset)})
            for s in publication.SEEDS]


def state_fixture(*, both=False):
    refs, cells = [], []
    modes = ['mean', 'policy_sample'] if both else ['policy_sample']
    for h in (1, 2, 3):
        for j in (1, 2, 4, 6, 8, 10):
            by_key = {}
            for estimator, mode, offset in [('one_step', 'mean', 5),
                                             ('one_step', 'policy_sample', 10),
                                             ('retrace', 'policy_sample', 14)]:
                reference = dict(H=h, J=j, estimator=estimator, execution=mode,
                                 episodes=episodes(j + offset), manifest_sha256='pinned',
                                 performance_run_id=f'reward-{h}-{j}-{estimator}-{mode}')
                refs.append(reference)
                by_key[estimator, mode] = reference
            for estimator in ('one_step', 'retrace'):
                for mode in modes:
                    name = f'soft_return_h{h}_j{j}_{estimator}_{mode}'
                    cells.append(dict(name=name, H=h, J=j, estimator=estimator,
                                      retrace_lambda=.9 if estimator == 'retrace' else None,
                                      execution_mode=mode, critic_kind='soft_return',
                                      mean_reference=deepcopy(by_key['one_step', 'mean']),
                                      reward_reference=deepcopy(by_key.get((estimator, mode))),
                                      performance_run_id=name + '-performance', training_run_id=name + '-training'))
    return dict(cells=cells, references=refs, checkpoint_step=575000, checkpoint_sha256='checkpoint',
                source_run='backbone', source_commit='tested', initial_alpha=.0046, target_entropy=-10.5,
                H=[1, 2, 3], J=[1, 2, 4, 6, 8, 10], estimator=['one_step', 'retrace'],
                execution_modes=modes, group='test', overview_run_id='overview', label='Test')


def returns(cell):
    return episodes(cell['H'] * 100 + cell['J'] + (8 if cell['estimator'] == 'retrace' else 0)
                    + (3 if cell['execution_mode'] == 'policy_sample' else 0))


def choose(state, estimator, mode='policy_sample', h=3, j=10):
    return next(c for c in state['cells'] if (c['H'], c['J'], c['estimator'], c['execution_mode'])
                == (h, j, estimator, mode))


def point(aggregate, cell):
    return next(p for p in aggregate['points'] if p['setting'] == cell['name'])


@pytest.mark.parametrize('both,new_count,pairs', [
    (False, 36, dict(soft_return_minus_reward=36, retrace_minus_one_step=18, sampled_minus_mean=0)),
    (True, 72, dict(soft_return_minus_reward=54, retrace_minus_one_step=36, sampled_minus_mean=36)),
])
def test_complete_scope_preserves_references_and_only_exact_comparisons(both, new_count, pairs):
    state = state_fixture(both=both)
    result = publication.aggregate_results(state, {c['name']: returns(c) for c in state['cells']})
    assert result['new_evaluated'] == new_count
    assert len(result['points']) == 54 + new_count
    assert sum(p['historical'] for p in result['points']) == 54
    assert publication.expected_pairs(state) == publication.pair_counts(result) == pairs
    for p in result['points']:
        if p['historical']:
            assert p['critic_kind'] == 'return_only'
            assert all(p[kind + '_metrics'] is None for kind in publication.CONTRASTS)
        else:
            assert p['critic_kind'] == 'soft_return'
            if p['estimator'] == 'retrace' and p['execution'] == 'mean':
                assert p['soft_return_minus_reward_difference'] is None
                assert p['baseline_performance_run_id'] is None
            else:
                assert p['soft_return_minus_reward_difference']['episodes'] == 5
                assert p['baseline_performance_run_id']
    assert all(r['retrace_minus_one_step'] == 8 for r in result['comparisons']['retrace_minus_one_step'])
    assert all(r['sampled_minus_mean'] == 3 for r in result['comparisons']['sampled_minus_mean'])


def test_late_counterparts_create_new_numeric_rows_without_imputation():
    state = state_fixture(both=True)
    retrace = choose(state, 'retrace')
    one_step = choose(state, 'one_step')
    mean = choose(state, 'retrace', 'mean')
    done = {retrace['name']: returns(retrace)}
    early = publication.aggregate_results(state, done)
    first = point(early, retrace)
    assert first['soft_return_minus_reward_difference']['mean'] == 297
    assert first['retrace_minus_one_step_difference'] is None
    assert first['sampled_minus_mean_difference'] is None
    early_ids = {identity for identity, _ in publication.numeric_rows(early)}
    done[one_step['name']] = list(reversed(returns(one_step)))
    later = publication.aggregate_results(state, done)
    assert point(later, retrace)['retrace_minus_one_step_difference']['mean'] == 8
    assert point(later, retrace)['sampled_minus_mean_difference'] is None
    later_ids = {identity for identity, _ in publication.numeric_rows(later)}
    assert retrace['name'] + '/retrace_minus_one_step' in later_ids - early_ids
    done[mean['name']] = returns(mean)
    final = publication.aggregate_results(state, done)
    assert point(final, retrace)['sampled_minus_mean_difference']['mean'] == 3
    assert point(final, mean)['soft_return_minus_reward_difference'] is None
    assert retrace['name'] + '/sampled_minus_mean' in {
        identity for identity, _ in publication.numeric_rows(final)} - later_ids


def test_other_axes_and_historical_means_do_not_supply_new_counterparts():
    state = state_fixture(both=True)
    retrace = choose(state, 'retrace')
    unmatched = [choose(state, 'one_step', h=2), choose(state, 'one_step', j=8),
                 choose(state, 'one_step', 'mean'), choose(state, 'retrace', 'mean', j=8)]
    result = publication.aggregate_results(state, {c['name']: returns(c) for c in [retrace, *unmatched]})
    p = point(result, retrace)
    assert p['sampled_minus_mean_metrics'] is p['retrace_minus_one_step_metrics'] is None


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'solver', 'short', 'nonfinite'])
def test_pairing_rejects_incomplete_or_incompatible_episodes(damage):
    left, right = episodes(7), episodes(3)
    if damage == 'missing': left.pop()
    elif damage == 'duplicate': left[0] = deepcopy(left[1])
    elif damage == 'solver': left[0]['solver_seed'] += 1
    elif damage == 'short': left[0]['length'] -= 1
    else: left[0]['return'] = float('nan')
    with pytest.raises(ValueError):
        publication.paired_delta(left, right, 'soft_return_minus_reward')


@pytest.mark.parametrize('damage', ['duplicate_cell', 'duplicate_reference', 'wrong_reward', 'missing_reward', 'unknown_completed'])
def test_aggregate_rejects_ambiguous_identity(damage):
    state = state_fixture()
    done = {}
    if damage == 'duplicate_cell': state['cells'].append(deepcopy(state['cells'][0]))
    elif damage == 'duplicate_reference': state['references'].append(deepcopy(state['references'][0]))
    elif damage == 'wrong_reward': state['cells'][0]['reward_reference']['J'] = 99
    elif damage == 'missing_reward': state['cells'][0]['reward_reference'] = None
    else: done['not_requested'] = episodes()
    with pytest.raises(ValueError):
        publication.aggregate_results(state, done)


def test_points_table_is_scalar_and_missing_results_stay_blank():
    state = state_fixture(both=True)
    mean = choose(state, 'retrace', 'mean')
    result = publication.aggregate_results(state, {mean['name']: returns(mean)})
    wb = SimpleNamespace(Table=lambda **kw: kw, plot=SimpleNamespace(line_series=lambda **kw: kw))
    output = publication.overview_log(wb, result, [])
    table = output['comparison/points']
    rows = [dict(zip(table['columns'], row)) for row in table['data']]
    assert len(rows) == 55
    assert all(not isinstance(value, (list, dict, tuple)) for row in rows for value in row.values())
    new, = [r for r in rows if not r['historical']]
    assert new['execution'] == 'mean' and new['critic_kind'] == 'soft_return'
    assert new['return_episodes'] == 5 and new['return_mean'] == 321
    assert new['baseline_performance_run_id'] is None
    assert all(new[kind + '_mean'] is None for kind in publication.CONTRASTS)
    assert 'comparison/soft_return_minus_reward_vs_J' not in output
    assert output['comparison/soft_return_minus_reward_episodes']['data'] == []


@pytest.mark.parametrize('both', [False, True])
def test_already_published_cells_still_refresh_overview_and_all_pairs(tmp_path, monkeypatch, both):
    state = state_fixture(both=both)
    for cell in state['cells']:
        directory = tmp_path / cell['name']; directory.mkdir()
        (directory / 'worker-completion.json').write_text('{}')
        cell['directory'] = str(directory)
    (tmp_path / 'campaign.json').write_text(json.dumps(state))
    logged = []
    run = SimpleNamespace(summary={}, log=lambda row: logged.append(deepcopy(row)),
                          define_metric=lambda *a, **k: None, finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **k: run))
    monkeypatch.setattr(publication, 'verify_reference', lambda *a, **k: None)
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: True)
    monkeypatch.setattr(publication, 'load_completed', lambda campaign, cell: (
        returns(cell), {'execution/sample_flag_mean': float(cell['execution_mode'] == 'policy_sample')}))
    monkeypatch.setattr(publication, 'overview_log', lambda *a: {})
    publication.watch(SimpleNamespace(root=tmp_path))
    result = publication.read(tmp_path / 'comparison-results.json')
    assert len(result['points']) == 54 + len(state['cells'])
    assert run.summary['status'] == 'complete'
    assert run.summary['evaluated'] == len(state['cells'])
    assert run.summary['pair_counts'] == publication.expected_pairs(state)
    for kind, count in publication.expected_pairs(state).items():
        assert sum(any('/' + kind + '/mean' in key for key in row) for row in logged) == count
