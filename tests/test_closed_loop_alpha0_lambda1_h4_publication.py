"""Seed-paired contrasts, distinct baselines, and complete-result publication."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_alpha0_lambda1_h4 as campaign
from slurm import ambi_closed_loop_alpha0_lambda1_h4_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset=0):
    return [dict(seed=s, solver_seed=solver_seed(55, 'episode', s), length=500,
                 truncated_by_evaluator=False, **{'return': float(s - 100 + offset)})
            for s in publication.SEEDS]


def state_fixture():
    refs = []
    for h in (1, 2, 3):
        for j in (1, 2, 4, 6, 8, 10):
            for estimator, mode, offset in [('one_step', 'mean', 0), ('one_step', 'policy_sample', 3),
                                             ('retrace', 'policy_sample', 8)]:
                refs.append(dict(H=h, J=j, estimator=estimator, execution=mode,
                    retrace_lambda=.9 if estimator == 'retrace' else None, alpha_mode='adaptive',
                    episodes=episodes(h*10+j+offset), manifest_sha256='pinned',
                    performance_run_id=f'control-{h}-{j}-{estimator}-{mode}'))
    for j in (12, 14):
        refs.append(dict(H=3, J=j, estimator='one_step', execution='mean', retrace_lambda=None,
                         episodes=episodes(30+j), manifest_sha256='pinned', performance_run_id=f'control-3-{j}-mean'))
    by_key = {(r['H'], r['J'], r['estimator'], r['execution']): r for r in refs}
    cells = campaign.cells()
    for cell in cells:
        arm, h, j = cell['experiment_arm'], cell['H'], cell['J']
        key = ((h, j, cell['estimator'], 'policy_sample') if arm == 'alpha_zero'
               else (min(h, 3), j, 'one_step', 'mean'))
        cell.update(paired_reference=deepcopy(by_key[key]),
                    performance_run_id=cell['name']+'-performance', training_run_id=cell['name']+'-training')
    return dict(cells=cells, references=refs, checkpoint_step=575000, checkpoint_sha256='checkpoint',
        source_run='backbone', source_commit='tested', inventory='inventory', group='test',
        overview_run_id='overview', label='Test', publisher_workers=3,
        h4_overview_run_id='h4-overview', h4_group='test-h4', h4_label='H4',
        prior_reference=dict(bundle='prior', manifest_sha256='prior-pin', source_commit='prior-source', episodes=episodes(-3)))


def result_episodes(cell):
    offset = {'alpha_zero': 2, 'lambda_one_mean': 7, 'h4_mean': 11}[cell['experiment_arm']]
    return [{**e, 'return': e['return']+offset} for e in cell['paired_reference']['episodes']]


def completed(state):
    return {c['name']: result_episodes(c) for c in state['cells']}


def test_full_scope_exact_controls_and_paired_bootstrap_semantics():
    state = state_fixture()
    result = publication.aggregate_results(state, completed(state))
    counts = dict(alpha_zero_minus_alpha_on=36, lambda_one_mean_minus_one_step_mean=18,
                  lambda_one_mean_minus_lambda09_sampled=18, h4_minus_h3_mean=8)
    assert result['new_evaluated'] == 62 and len(result['points']) == 118
    assert publication.expected_pairs(state) == publication.pair_counts(result) == counts
    expected = dict(alpha_zero_minus_alpha_on=2, lambda_one_mean_minus_one_step_mean=7,
                    lambda_one_mean_minus_lambda09_sampled=-1, h4_minus_h3_mean=11)
    for kind, value in expected.items():
        assert len(result['comparisons'][kind]) == counts[kind]*5
        assert all(row[kind] == value for row in result['comparisons'][kind])
        for point in result['points']:
            metrics = point[kind+'_metrics']
            if metrics:
                assert metrics['comparison/'+kind+'_ci95_low'] == metrics['comparison/'+kind+'_ci95_high'] == value
                assert metrics['comparison/'+kind+'_paired_episodes'] == 5
    rows = publication.numeric_rows(result)
    assert len(rows) == len({identity for identity, _ in rows}) == 198
    assert all('axis/inner_rounds' in row for _, row in rows)
    assert all(not isinstance(value, (list, dict)) for _, row in rows for value in row.values())


def test_missing_results_stay_absent_and_arrival_adds_only_its_measurement_and_pairs():
    state = state_fixture()
    pending = publication.aggregate_results(state, {})
    cell = next(c for c in state['cells'] if c['experiment_arm'] == 'lambda_one_mean')
    arrived = publication.aggregate_results(state, {cell['name']: list(reversed(result_episodes(cell)))})
    old = {identity for identity, _ in publication.numeric_rows(pending)}
    new = {identity for identity, _ in publication.numeric_rows(arrived)}
    assert new-old == {cell['name'], cell['name']+'/lambda_one_mean_minus_one_step_mean',
                       cell['name']+'/lambda_one_mean_minus_lambda09_sampled'}
    assert not any(key.startswith('comparison/') for key in publication.chart_payloads(pending))


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'solver', 'short', 'nonfinite'])
def test_pairing_rejects_invalid_seed_sets(damage):
    left, right = episodes(5), episodes(2)
    if damage == 'missing': left.pop()
    elif damage == 'duplicate': left[0] = deepcopy(left[1])
    elif damage == 'solver': left[0]['solver_seed'] += 1
    elif damage == 'short': left[0]['length'] -= 1
    else: left[0]['return'] = float('inf')
    with pytest.raises(ValueError):
        publication.paired_delta(left, right, 'alpha_zero_minus_alpha_on')


@pytest.mark.parametrize('damage', ['duplicate_cell', 'duplicate_reference', 'wrong_horizon', 'wrong_execution',
                                     'wrong_estimator', 'changed_episodes', 'unknown_completed'])
def test_rejects_ambiguous_or_unmatched_controls(damage):
    state = state_fixture(); done = {}
    if damage == 'duplicate_cell': state['cells'].append(deepcopy(state['cells'][0]))
    elif damage == 'duplicate_reference': state['references'].append(deepcopy(state['references'][0]))
    elif damage == 'wrong_horizon': state['cells'][0]['paired_reference']['H'] = 99
    elif damage == 'wrong_execution': state['cells'][0]['paired_reference']['execution'] = 'mean'
    elif damage == 'wrong_estimator': state['cells'][0]['paired_reference']['estimator'] = 'retrace'
    elif damage == 'changed_episodes': state['cells'][0]['paired_reference']['episodes'][0]['return'] += 1
    else: done['not_requested'] = episodes()
    with pytest.raises(ValueError): publication.aggregate_results(state, done)


def test_charts_do_not_join_different_modes_and_points_table_is_scalar():
    state = state_fixture(); aggregate = publication.aggregate_results(state, completed(state))
    charts = publication.chart_payloads(aggregate)
    assert len(charts) == 7
    assert 'Combined lambda and execution change' in charts['comparison/lambda_one_mean_minus_lambda09_sampled_vs_J']['title']
    assert len(charts['panel_a/return_vs_J']['keys']) == 12
    assert len(charts['panel_b/return_vs_J']['keys']) == 9
    assert len(charts['panel_c/return_vs_J']['keys']) == 2
    assert charts['panel_c/return_vs_J']['xs'] == [[1,2,4,6,8,10,12,14]]*2
    wb = SimpleNamespace(Table=lambda **kw: kw, plot=SimpleNamespace(line_series=lambda **kw: kw))
    output = publication.overview_log(wb, aggregate, [])
    table = output['comparison/points']
    assert len(table['data']) == 118
    assert {'experiment_arm', 'alpha_mode', 'critic_kind', 'in_panel_a', 'in_panel_b', 'in_panel_c',
            'baseline_performance_run_id'} <= set(table['columns'])
    assert all(not isinstance(value, (list, dict, tuple)) for row in table['data'] for value in row)


def test_h4_legacy_gain_uses_prior_not_h3_and_preserves_exact_legend_contract():
    state = state_fixture(); aggregate = publication.aggregate_results(state, completed(state))
    result = publication.h4_results(state, aggregate)
    assert len(result['points']) == 8 and len(result['paired_rows']) == 40
    for point in result['points']:
        assert point['paired_gain']['mean'] == 44 + point['J']
        assert point['paired_gain']['mean'] != 11  # The distinct H4-minus-H3 contrast.
    charts = publication.h4_chart_payloads(result, [1,2,4,6,8,10,12,14])
    assert charts['comparison/return_vs_J']['keys'] == ['Return-only critic', 'Frozen prior (reused)']
    assert charts['comparison/paired_gain_vs_J']['keys'] == ['Return-only critic', 'No improvement']
    assert 'comparison/return_minus_soft_vs_J' not in charts


def fake_wandb(monkeypatch):
    logged, configs = [], []
    run = SimpleNamespace(summary={}, log=lambda row: logged.append(deepcopy(row)),
                          define_metric=lambda *a, **k: None, finish=lambda **k: None)
    def init(**kwargs):
        configs.append(kwargs); return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init, Table=lambda **kw: kw,
                         plot=SimpleNamespace(line_series=lambda **kw: kw)))
    return run, logged, configs


def test_already_published_results_refresh_main_overview_without_relogging(tmp_path, monkeypatch):
    state = state_fixture()
    for cell in state['cells']:
        directory = tmp_path/cell['name']; directory.mkdir()
        (directory/'worker-completion.json').write_text('{}')
        cell['directory'] = str(directory)
    publication.write(tmp_path/'campaign.json', state)
    run, logged, configs = fake_wandb(monkeypatch)
    monkeypatch.setattr(publication, 'verify_reference', lambda *a, **k: None)
    monkeypatch.setattr(publication, 'verify_prior', lambda c: c['prior_reference'])
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: True)
    monkeypatch.setattr(publication, 'load_completed', lambda campaign, cell: (result_episodes(cell), {'protocol/alpha':0}))
    monkeypatch.setattr(publication.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(poll=lambda: 0, wait=lambda **k: 0))
    publication.write(tmp_path/'h4-overview-completion.json', dict(status='complete', run_id=state['h4_overview_run_id'], points=8))
    publication.watch(SimpleNamespace(root=tmp_path))
    assert run.summary['status'] == 'complete' and run.summary['evaluated'] == 62
    assert run.summary['pair_counts'] == publication.expected_pairs(state)
    assert len([row for row in logged if 'axis/inner_rounds' in row]) == 198
    assert publication.read(tmp_path/'campaign-completion.json')['status'] == 'complete'


def test_h4_companion_uses_scalar_horizon_and_logs_each_new_point_once(tmp_path, monkeypatch):
    state = state_fixture(); aggregate = publication.aggregate_results(state, completed(state))
    publication.write(tmp_path/'campaign.json', state)
    publication.write(tmp_path/'comparison-results.json', aggregate)
    publication.write(tmp_path/'progress.json', {'rows': [dict(setting=c['name'], experiment_arm=c['experiment_arm'],
        H=c['H'], J=c['J'], status='published', performance_url='performance', training_url='training') for c in state['cells']]})
    run, logged, configs = fake_wandb(monkeypatch)
    monkeypatch.setattr(publication, 'verify_prior', lambda c: c['prior_reference'])
    publication.h4_overview(SimpleNamespace(root=tmp_path))
    assert configs[0]['config']['H'] == 4
    assert configs[0]['config']['campaign_group'] == state['h4_group']
    assert run.summary['status'] == 'complete' and run.summary['evaluated'] == run.summary['published'] == 8
    assert len([row for row in logged if 'axis/inner_rounds' in row]) == 8
    assert publication.read(tmp_path/'h4-overview-completion.json')['status'] == 'complete'


def test_h4_companion_does_not_report_success_for_missing_measurements(tmp_path, monkeypatch):
    state = state_fixture()
    publication.write(tmp_path/'campaign.json', state)
    publication.write(tmp_path/'comparison-results.json', publication.aggregate_results(state, {}))
    publication.write(tmp_path/'progress.json', {'rows': []})
    publication.write(tmp_path/'campaign-completion.json', {'status': 'incomplete'})
    run, _, _ = fake_wandb(monkeypatch)
    monkeypatch.setattr(publication, 'verify_prior', lambda c: c['prior_reference'])
    with pytest.raises(RuntimeError, match='before all eight'):
        publication.h4_overview(SimpleNamespace(root=tmp_path))
    assert run.summary['status'] == 'failed'
    assert publication.read(tmp_path/'h4-overview-completion.json')['status'] == 'failed'
