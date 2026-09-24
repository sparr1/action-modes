"""Fresh paired value-sampling curves, publication identity and diagnostic axes."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_retrace_values as campaign
from slurm import ambi_closed_loop_retrace_values_publish as publication


def episodes(offset=0):
    return [dict(seed=seed, solver_seed=1000 + seed, episode_id=f'e{seed}',
                 return_value=offset + index, **{'return': offset + index}, length=500,
                 truncated_by_evaluator=False, control_seconds=10.)
            for index, seed in enumerate(campaign.SEEDS)]


def state_fixture():
    cells = campaign.cells()
    for index, cell in enumerate(cells):
        cell.update(performance_run_id=f'p{index}', training_run_id=f't{index}', reused=False,
                    directory=f'/unused/{cell["name"]}', bundle=f'/unused/{cell["name"]}/bundle')
    return dict(cells=cells, retrace_lambda=1., checkpoint_step=575000, checkpoint_sha256=campaign.CHECKPOINT_SHA,
                source_run=campaign.SOURCE_RUN, source_commit='tested', initial_alpha=campaign.INITIAL_ALPHA,
                target_entropy=-10.5, H=[2, 3], J=[1, 2, 4, 6, 8, 10],
                sample_pairs=[[1, 1], [4, 1], [1, 4], [4, 4]], group='test', label='Test', overview_run_id='overview')


def test_all_eight_lines_keep_horizons_counts_and_lambda_separate():
    state = state_fixture()
    done = {cell['name']: episodes(cell['J'] + cell['Ki'] + 2 * cell['Kb']) for cell in state['cells']}
    aggregate = publication.aggregate_results(state, done,
        {name: dict(control_seconds=50., control_seconds_per_decision=.02) for name in done})
    assert len(aggregate['points']) == 48 and len(aggregate['paired_comparisons']) == 240
    for payload in publication.chart_payloads(aggregate).values():
        assert len(payload['keys']) == 8 == len(set(payload['keys']))
        assert all('lambda1' in key for key in payload['keys'])
        assert payload['xs'] == [[1, 2, 4, 6, 8, 10]] * 8
    baseline = [p for p in aggregate['points'] if p['Ki'] == p['Kb'] == 1]
    assert all(p['gain_stats']['mean'] == p['gain_stats']['ci95_low'] == p['gain_stats']['ci95_high'] == 0
               for p in baseline)
    keys = [identity for identity, _ in publication.numeric_rows(aggregate)]
    assert len(keys) == len(set(keys)) == 96


def test_pairing_waits_for_matching_fresh_baseline_without_imputation():
    state = state_fixture()
    arm = next(c for c in state['cells'] if (c['H'], c['J'], c['Ki'], c['Kb']) == (3, 10, 4, 4))
    base = next(c for c in state['cells'] if (c['H'], c['J'], c['Ki'], c['Kb']) == (3, 10, 1, 1))
    unrelated = next(c for c in state['cells'] if (c['H'], c['J'], c['Ki'], c['Kb']) == (2, 10, 1, 1))
    empty = publication.aggregate_results(state, {})
    assert empty['points'] == empty['paired_comparisons'] == []
    done = {arm['name']: episodes(20), unrelated['name']: episodes(5)}
    early = publication.aggregate_results(state, done)
    point = next(p for p in early['points'] if p['setting'] == arm['name'])
    assert 'gain_stats' not in point
    assert all(r['setting'] != arm['name'] for r in early['paired_comparisons'])
    done[base['name']] = list(reversed(episodes(12)))
    late = publication.aggregate_results(state, done)
    point = next(p for p in late['points'] if p['setting'] == arm['name'])
    assert point['gain_stats']['mean'] == point['gain_stats']['ci95_low'] == point['gain_stats']['ci95_high'] == 8
    early_ids = {identity for identity, _ in publication.numeric_rows(early)}
    late_ids = {identity for identity, _ in publication.numeric_rows(late)}
    assert arm['name'] + '/paired_gain' in late_ids - early_ids
    with pytest.raises(ValueError, match='Unexpected completed'):
        publication.aggregate_results(state, {'bad': episodes()})


@pytest.mark.parametrize('change', ['solver', 'short', 'duplicate'])
def test_paired_comparison_rejects_mismatched_protocol_or_seeds(change):
    altered = episodes()
    if change == 'solver': altered[0]['solver_seed'] += 1
    if change == 'short': altered[0]['length'] = 499
    if change == 'duplicate': altered[0] = deepcopy(altered[1])
    with pytest.raises(ValueError):
        publication.paired_comparison(altered, episodes())


@pytest.mark.parametrize('field,value', [('inner_retrace_value_samples', 16),
    ('inner_retrace_boundary_value_samples', 16), ('inner_retrace_lambda', .9)])
def test_declared_counts_and_lambda_must_match_requested_config(field, value):
    state = state_fixture()
    state['cells'][0]['params'][field] = value
    with pytest.raises(ValueError, match='disagree'):
        publication.aggregate_results(state, {})


def test_duplicate_setting_and_old_lambda_are_rejected():
    state = state_fixture()
    state['cells'].append(deepcopy(state['cells'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        publication.aggregate_results(state, {})
    state = state_fixture()
    state['cells'][0]['retrace_lambda'] = .9
    with pytest.raises(ValueError, match='lambda-one'):
        publication.aggregate_results(state, {})


def test_same_j_diagnostic_panels_never_combine_different_solve_budgets():
    state = state_fixture()
    curves = {cell['name']: {'critic': [dict(index=index, metrics={'critic_loss': {'mean': float(index)}})
                                      for index in range(1, 16 * cell['J'] + 1)],
                            'probe': [dict(index=j, metrics={'togo_return_mean': {'mean': float(j)}})
                                      for j in range(cell['J'] + 1)]} for cell in state['cells']}
    charts = publication.diagnostic_chart_payloads(state, curves)
    assert len(charts) == 24
    for h in (2, 3):
        for j in state['J']:
            critic = charts[f'inner/lambda1/h{h}/j{j}/critic_loss']
            probe = charts[f'inner/lambda1/h{h}/j{j}/togo_return_mean']
            assert len(critic['keys']) == len(probe['keys']) == 4
            assert critic['xs'] == [list(range(1, 16 * j + 1))] * 4
            assert probe['xs'] == [list(range(j + 1))] * 4


def test_stream_summary_preserves_every_retrace_critic_update(tmp_path):
    cell = next(c for c in campaign.cells() if c['J'] == 1)
    events = []
    for episode in episodes():
        base = dict(episode_id=episode['episode_id'], decision_index=0, round_index=1)
        events.extend([{**base, 'phase': 'initial', 'replay_size': 0},
                       {**base, 'phase': 'collection', 'replay_size': 128 * cell['H']}])
        for index in range(1, 17):
            events.append({**base, 'phase': 'update', 'critic_updates': index, 'actor_updates': 0,
                'updated_critic': True, 'updated_actor': False, 'updated_temperature': False,
                'metrics': {'critic_loss': 1., 'critic_grad_norm': 1., 'td_error_abs_mean': 1.,
                            'q_target_mean': 1., 'retrace_trace_coefficient_mean': index / 16,
                            'retrace_value_samples': cell['Ki']}})
        for index in range(1, 5):
            events.append({**base, 'phase': 'update', 'critic_updates': 16, 'actor_updates': index,
                'updated_critic': False, 'updated_actor': True, 'updated_temperature': True,
                'metrics': {'actor_loss': 1., 'actor_grad_norm': 1., 'actor_entropy': 1., 'alpha_used': 1.}})
        events.append({**base, 'phase': 'decision', 'metrics': {}})
    with gzip.open(tmp_path / 'trace.jsonl.gz', 'wt') as file:
        for event in events: file.write(json.dumps(event) + '\n')
    campaign.write(tmp_path / 'manifest.json', {'runs': [dict(trace_files=['trace.jsonl.gz'], episodes=episodes())],
                                               'metric_catalog': {}})
    summary = publication.training_summary(tmp_path, cell, expected_steps=1)
    retrace = [row for row in summary['update_curves'] if 'retrace_trace_coefficient_mean' in row['metrics']]
    assert [row['index'] for row in retrace] == list(range(1, 17))
    assert all(row['axis'] == 'critic_update' for row in retrace)
    assert [row['metrics']['retrace_trace_coefficient_mean']['mean'] for row in retrace] == [i / 16 for i in range(1, 17)]


def test_overview_starts_with_explicit_counts_and_no_invented_results():
    aggregate = publication.aggregate_results(state_fixture(), {})
    wb = SimpleNamespace(Table=lambda **kwargs: kwargs, plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    result = publication.overview_log(wb, aggregate, [])
    assert result['campaign/evaluated'] == result['campaign/published'] == 0
    assert result['campaign/total_settings'] == 48 and result['campaign/total_episodes'] == 240
    assert not any('vs_J' in key for key in result)
    assert result['comparison/points']['data'] == []


def test_confidence_bands_publish_as_self_contained_html_without_plotting_dependencies():
    import xml.etree.ElementTree as ET
    state = state_fixture()
    done = {cell['name']: episodes(cell['J'] + cell['Ki'] + cell['Kb']) for cell in state['cells']}
    aggregate = publication.aggregate_results(state, done)
    wb = SimpleNamespace(Table=lambda **kwargs: kwargs, Html=lambda html, **kwargs: html,
                         plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    output = publication.overview_log(wb, aggregate, [])
    for field in ('return_stats', 'gain_stats'):
        html = output[f'comparison/{field}_95ci']
        svg_text = html[html.index('<svg'):html.index('</svg>') + len('</svg>')]
        svg = ET.fromstring(svg_text)
        assert len(svg.findall('{http://www.w3.org/2000/svg}polygon')) == 8
        assert len(svg.findall('{http://www.w3.org/2000/svg}circle')) == 48
        assert 'bootstrap' in html and '<script' not in html
        assert 'src=' not in html and 'href=' not in html


def test_existing_publications_populate_all_curves_and_pairs(tmp_path, monkeypatch):
    state = state_fixture()
    for cell in state['cells']:
        directory = tmp_path / cell['name']; directory.mkdir()
        (directory / 'worker-completion.json').write_text('{}')
        cell['directory'] = str(directory)
    campaign.write(tmp_path / 'campaign.json', state)
    logs = []
    run = SimpleNamespace(summary={}, log=lambda row: logs.append(deepcopy(row)),
                          define_metric=lambda *a, **k: None, finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **k: run))
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: True)
    monkeypatch.setattr(publication, 'load_completed', lambda state, cell: (episodes(cell['Ki'] + cell['Kb']), {}))
    monkeypatch.setattr(publication, 'overview_log', lambda *a, **k: {})
    publication.watch(SimpleNamespace(root=tmp_path))
    aggregate = campaign.read(tmp_path / 'comparison-results.json')
    assert len(aggregate['points']) == 48 and len(aggregate['paired_comparisons']) == 240
    assert run.summary['status'] == 'complete' and run.summary['evaluated'] == 48
    assert sum(any('/paired_gain_mean' in key for key in row) for row in logs) == 48


def test_publication_receipt_rejects_different_counts(tmp_path):
    cell = state_fixture()['cells'][0]
    cell['directory'] = str(tmp_path)
    campaign.write(tmp_path / 'training-publication.json', dict(status='complete', run_id=cell['training_run_id']))
    receipt = dict(status='complete', cell=cell['name'], setting_key=list(publication.setting_key(cell)),
                   training_run_id=cell['training_run_id'],
                   performance=dict(run_id=cell['performance_run_id'], published=1))
    campaign.write(tmp_path / 'publication-completion.json', receipt)
    assert publication.publication_complete(cell)
    receipt['setting_key'][2] = 16
    campaign.write(tmp_path / 'publication-completion.json', receipt)
    with pytest.raises(ValueError, match='identity mismatch'):
        publication.publication_complete(cell)
