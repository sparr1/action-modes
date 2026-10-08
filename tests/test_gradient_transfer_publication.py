"""The H1 gradient screen separates strengths without mixing candidate curves."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_bernoulli_probability_publication import MultiChartService
from tests.test_spectral_transfer_publication import campaign, episodes, fake_wandb
from utils import wandb_transfer_discovery_layout as layout


def gradient_campaign():
    arms = {'fresh': dict(actor_rho=0., critic_rho=0., parameter_scope='matrices')}
    for component, parts in (('actor', ('actor',)), ('critic', ('critic',)), ('joint', ('actor', 'critic'))):
        for name, key, value in (('carry', 'rho', 1.), ('blend', 'rho', .5), ('bernoulli', 'bernoulli_p', .5)):
            arms[f'{component}_{name}'] = dict(parameter_scope='matrices',
                **{f'{part}_{key}': value for part in parts})
        for method, rank in (('gradient_gate', None), ('gradient_projection', None), ('gradient', 1), ('gradient', 4)):
            for strength in (.5, 1.):
                for matched in (False, True):
                    name = f'{component}_{method}_r{rank}_s{strength:g}' + ('_norm' if matched else '')
                    spec = dict(method=method, rank=rank, strength=strength, norm_matched=matched)
                    arms[name] = dict(parameter_scope='matrices',
                        **{f'{part}_spectral': deepcopy(spec) for part in parts})
    cells = [dict(index=i, name=f'h1_j{j}_{arm}', H=1, J=j, arm=arm)
             for i, (j, arm) in enumerate((j, arm) for j in (1, 2, 4, 6) for arm in arms)]
    return dict(family='spectral_transfer', arms=arms, cells=cells, H=[1], J=[1, 2, 4, 6],
        seeds=[101, 102, 103], max_steps=500, diagnostics={'enabled': True}, publication=dict(
            gradient_alignment_view=True, slug_prefix='gradienth1575',
            view_title='575K H1 gradient transfer · SVD, gating and projection'))


def test_gradient_groups_isolate_strength_and_repeat_only_baselines():
    value = gradient_campaign()
    assert len(value['cells']) == 232
    groups = layout.campaign_curve_groups(value)
    assert [prefix for prefix, _, _ in groups] == [f'{part}/s{strength}/'
        for part in ('actor', 'critic', 'joint') for strength in ('0p5', '1')]
    for prefix, label, arms in groups:
        strength = .5 if '/s0p5/' in prefix else 1.
        metadata = [layout.spectral_arm_metadata(value, arm) for arm in arms]
        assert len(arms) == 12
        assert sum(row['method'] in ('fresh', 'carry', 'blend', 'bernoulli') for row in metadata) == 4
        assert all(row['strength'] == strength for row in metadata
                   if row['method'] not in ('fresh', 'carry', 'blend', 'bernoulli'))
        assert f'strength {strength:g}' in label
        assert len({row['color'] for row in metadata}) == 12
        styles = {(row['method'], row['requested_rank'], row['norm_matched']): row for row in metadata}
        assert styles['gradient', 1, False]['color'] != styles['gradient', 4, False]['color']
        assert styles['gradient_projection', None, True]['label'].startswith('Global norm match')
        assert styles['gradient_gate', None, True]['label'].startswith('Per-layer norm match')
        definition = layout.campaign_chart_definition(value, prefix[:-1])
        assert len(definition['encoding']['color']['scale']['domain']) == 12
        assert definition['encoding']['strokeDash']['scale']['range'] == [[], [8, 3], [2, 3]]
        assert all('_joint_' not in metric for metric, _ in layout.campaign_diagnostic_charts(value, prefix))
    assert [arms[-1] for _, _, arms in groups] == ['fresh'] * 6


def test_h1_gradient_view_and_payload_have_exact_same_keys_and_preserve_old_views(tmp_path):
    value = gradient_campaign()
    service = MultiChartService()
    before = deepcopy(service.views)
    options = dict(entity='entity', project='project', receipt_dir=tmp_path,
                   run_id='gradient123', campaign=value)
    receipt = layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), **options)
    assert receipt['url'].endswith('?nw=gradienth1575gradient123')
    assert service.views[:len(before)] == before
    assert service.chart_writes == 6 and service.writes == 1
    saved = json.loads(service.views[-1]['spec'])
    sections = layout._bank(saved)['sections']
    intro = sections[0]['panels'][0]['config']['value']
    assert 'Strengths are shown in separate panels' in intro
    assert 'Coordinate gradient gating' in intro and 'not the first SAC replay minibatch' in intro
    assert 'Gradient-line projection is signed' in intro
    assert 'their table counts are transfer decisions, not sampled heldout roots' in intro
    assert 'Projection coefficients are signed and measured before multiplying by transfer strength' in intro
    assert 'for dense norm controls, they describe the projected candidate used to set the control norm' in intro
    assert len({section['__id__'] for section in sections}) == len(sections)
    curves = [section for section in sections if '-curves-' in section['__id__']]
    assert len(curves) == 6 and all(len(section['panels']) == 2 for section in curves)
    assert all(section['flowConfig']['columnsPerPage'] == 2 for section in curves)
    selection_sections = [section for section in sections if '-selection-' in section['__id__']]
    assert len(selection_sections) == 6 and all(section['isOpen'] for section in selection_sections)
    assert sum(len(section['panels']) for section in selection_sections) == 24
    for section in sections:
        if '-proxies-' in section['__id__']:
            assert 'selection_actor_' not in json.dumps(section) and 'selection_critic_' not in json.dumps(section)
    snapshot = dict(settings=[], results=[], episodes=[], diagnostics=[], completed=0, total=232, failed=0)
    payload = publisher.payload(fake_wandb(), snapshot, value)
    keys = set(receipt['expected_chart_keys'])
    assert len(keys) == 136
    assert keys == {key for key in payload if '_vs_' in key}
    assert all('/h1_' in key for key in keys)
    assert 'discovery/joint/s0p5/h1_spectral_final_critic_loss_vs_j' in keys
    for key in keys:
        assert key in json.dumps(sections)
        assert len(payload[key]['keys']) == 12 and not any(payload[key]['ys'])
    assert layout._saved_installed(saved, 'gradient123', value, receipt['custom_chart_id'])
    assert not layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), **options)['changed']
    assert service.chart_writes == 6 and service.writes == 1


def test_gradient_payload_does_not_mix_strengths_and_shares_baseline_data():
    value = gradient_campaign()
    rows, diagnostics = [], []
    for cell in value['cells']:
        metadata = publisher._cell_metadata(cell, value)
        data = episodes()
        for episode in data:
            episode['return'] += 1000 * metadata['strength'] + cell['J']
        row, measurements = publisher._measurement_rows(metadata, data, 'current campaign', 'complete', spectral=True)
        rows.append(row)
        diagnostics.extend(measurements)
    payload = publisher.payload(fake_wandb(), dict(settings=[], results=rows, episodes=[],
        diagnostics=diagnostics, completed=232, total=232, failed=0), value)
    for part in ('actor', 'critic', 'joint'):
        charts = [payload[f'discovery/{part}/s{token}/h1_return_vs_j'] for token in ('0p5', '1')]
        for chart, strength in zip(charts, (.5, 1.)):
            for arm, xs, ys in zip(chart['keys'], chart['xs'], chart['ys']):
                metadata = layout.spectral_arm_metadata(value, arm)
                assert xs == [1, 2, 4, 6]
                assert ys == [101. + 1000 * metadata['strength'] + j for j in xs]
                if metadata['method'].startswith('gradient'):
                    assert metadata['strength'] == strength
        shared = set(charts[0]['keys']) & set(charts[1]['keys'])
        assert shared == {'fresh', f'{part}_carry', f'{part}_blend', f'{part}_bernoulli'}
        for arm in shared:
            assert charts[0]['ys'][charts[0]['keys'].index(arm)] == charts[1]['ys'][charts[1]['keys'].index(arm)]


def test_gradient_table_serialization_preserves_null_ranks_scope_and_pending_values():
    import wandb
    value = gradient_campaign()
    rows = [publisher._measurement_rows(publisher._cell_metadata(cell, value), None,
                'current campaign', 'pending', spectral=True)[0] for cell in value['cells']]
    snapshot = dict(settings=rows, results=rows, episodes=[], diagnostics=[], completed=0, total=232, failed=0)
    sdk = SimpleNamespace(Table=wandb.Table, plot=fake_wandb().plot)
    published = publisher.payload(sdk, snapshot, value)
    table = published['discovery/results']._to_table_json()
    serialized = [dict(zip(table['columns'], row)) for row in table['data']]
    assert len(serialized) == 232
    for row in serialized:
        assert row['return_mean'] is None and row['state'] == 'pending'
        if row['method'] in ('gradient_gate', 'gradient_projection'):
            assert row['requested_rank'] is None
            assert row['norm_matching_scope'] == ('component-global' if row['method'] == 'gradient_projection' else 'per-layer')
        elif row['method'] == 'gradient':
            assert row['requested_rank'] in (1, 4) and row['norm_matching_scope'] == 'per-layer'
        else:
            assert row['norm_matching_scope'] is None
    json.dumps(table, allow_nan=False)


@pytest.mark.parametrize('method', ['gradient_gate', 'gradient_projection'])
@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('known_counts', [False, True])
def test_selection_summaries_publish_means_coverage_and_separate_strength_panels(method, standalone, known_counts):
    import wandb
    value = gradient_campaign()
    arm = f'actor_{method}_rNone_s0.5'
    cell = publisher._cell_metadata(next(cell for cell in value['cells'] if cell['arm'] == arm), value)
    data = episodes()
    selected_metric = 'retained_fraction' if method == 'gradient_gate' else 'projection_coefficient'
    for index, episode in enumerate(data):
        selection = {
            f'selection_actor_{selected_metric}': (.25 * (index + 1) if method == 'gradient_gate' else float(index - 2)),
            'selection_actor_predicted_benefit': float(index - 1) * 4,
            'selection_actor_transferred_energy_fraction': .1 * (index + 1),
        }
        if method == 'gradient_projection':
            selection.update(selection_actor_gradient_squared_norm=float((index + 1) ** 2),
                             selection_actor_donor_gradient_inner_product=float(index - 1))
        counts = {key: 499 - index for key in selection} if known_counts else {}
        episode['selection_diagnostics'] = dict(summary=selection, summary_counts=counts)
        if standalone:
            episode.pop('diagnostics')
            episode['diagnostic_samples'] = 0
            episode['diagnostic_seconds'] = 0.
        else:
            episode['diagnostics']['summary'].update(selection)
            episode['diagnostics']['summary_counts'].update(counts)
    original = deepcopy(data)
    row, measurements = publisher._measurement_rows(cell, data, 'current campaign', 'complete', spectral=True)
    assert data == original
    measured = {entry['metric']: entry for entry in measurements}
    key = f'selection_actor_{selected_metric}'
    assert measured[key]['mean'] == (.5 if method == 'gradient_gate' else -1.)
    assert measured[key]['episode_std'] == (.25 if method == 'gradient_gate' else 1.)
    assert measured[key]['episodes'] == 3
    assert measured[key]['samples'] == (1494 if known_counts else None)
    assert measured[key]['sample_count_basis'] == ('applicable transfer decisions' if known_counts
                                                  else 'transfer-decision coverage unavailable')
    other_key = 'selection_actor_projection_coefficient' if method == 'gradient_gate' else 'selection_actor_retained_fraction'
    assert other_key not in measured
    if not standalone:
        assert measured['spectral_initial_actor_loss']['samples'] == 18
        assert measured['spectral_initial_actor_loss']['sample_count_basis'] == 'per-metric contributing roots'
    assert measured['selection_actor_predicted_benefit']['mean'] == 0.
    assert measured['selection_actor_predicted_benefit']['episode_std'] == 4.
    snapshot = dict(settings=[], results=[row], episodes=[], diagnostics=measurements,
                    completed=1, total=232, failed=0)
    published = publisher.payload(SimpleNamespace(Table=wandb.Table, plot=fake_wandb().plot), snapshot, value)
    chart = published[f'discovery/actor/s0p5/h1_{key}_vs_j']
    index = chart['keys'].index(arm)
    assert chart['ys'][index] == [measured[key]['mean']]
    assert 'selection bank' in chart['title']
    if method == 'gradient_projection':
        assert 'Signed projected-candidate coefficient before transfer strength' in chart['title']
    assert not any(published[f'discovery/actor/s1/h1_{key}_vs_j']['ys'])
    assert not any(published[f'discovery/actor/s0p5/h1_{other_key}_vs_j']['ys'])
    benefit = published['discovery/actor/s0p5/h1_selection_actor_predicted_benefit_vs_j']
    assert 'first-order loss reduction; selection bank' in benefit['title']
    table = published['discovery/diagnostics']._to_table_json()
    serialized = {entry['metric']: entry for entry in
                  (dict(zip(table['columns'], record)) for record in table['data'])}
    assert serialized[key]['samples'] == measured[key]['samples']
    assert serialized[key]['sample_count_basis'] == measured[key]['sample_count_basis']


def test_gradient_layout_is_opt_in_and_old_schema_stays_unchanged():
    value = campaign()
    before = deepcopy(value)
    old_charts = layout.expected_campaign_chart_keys(value)
    old_spec = layout.campaign_chart_definition(value, 'actor')
    value['publication']['gradient_alignment_view'] = False
    assert layout.expected_campaign_chart_keys(value) == old_charts
    assert layout.campaign_chart_definition(value, 'actor') == old_spec
    assert publisher.table_columns(value) == publisher.table_columns(before)
    assert all('norm_matching_scope' not in columns for columns in publisher.table_columns(value).values())


@pytest.mark.parametrize('method', ['gradient_gate', 'gradient_projection'])
def test_gradient_rankless_operators_reject_numeric_ranks_in_publication(method):
    value = gradient_campaign()
    arm = next(name for name in value['arms'] if f'_{method}_' in name)
    value['arms'][arm]['actor_spectral']['rank'] = 1
    with pytest.raises(layout.ResultsLayoutError, match='no matrix rank'):
        layout.spectral_arm_metadata(value, arm)
