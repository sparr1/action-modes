"""Rank comparisons preserve reused outcomes and publish complete diagnostic rows."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_bernoulli_probability_publication import MultiChartService
from tests.test_spectral_transfer_publication import episodes, fake_wandb
from utils import wandb_transfer_discovery_layout as layout


ROOT = Path(__file__).resolve().parents[1]


def rank_campaign():
    matrix = json.loads((ROOT / 'configs/research/ambi_spectral_transfer_575k.json').read_text())
    arms = deepcopy(matrix['arms'])
    old_cells = [dict(index=i, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
        for i, (j, h, arm) in enumerate((j, h, arm) for j in reversed(matrix['rounds'])
            for h in matrix['horizons'] for arm in matrix['arms'])]
    new_arms = {}
    for rank in (1, 4):
        for name, arm in matrix['arms'].items():
            if name.startswith('spectral_'):
                specification = deepcopy(arm)
                for component in ('actor', 'critic'):
                    if component + '_spectral' in specification:
                        specification[component + '_spectral']['rank'] = rank
                new_arms[name.replace('_r32_', f'_r{rank}_')] = specification
    arms.update(new_arms)
    cells = [dict(index=i, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
        for i, (j, h, arm) in enumerate((j, h, arm) for j in reversed(matrix['rounds'])
            for h in matrix['horizons'] for arm in new_arms)]
    reference = dict(kind='spectral', source_commit='b1f576d6-old-source',
        records=[dict(cell=cell, hashes={}) for cell in old_cells])
    return dict(family='spectral_transfer', source_commit='new-source', arms=arms, cells=cells,
        H=matrix['horizons'], J=matrix['rounds'], seeds=matrix['seeds'], max_steps=500,
        diagnostics=matrix['diagnostics'], publication=matrix['publication'],
        historical_reference=None, historical_references=[reference])


def reference_rows(value):
    reference = value['historical_references'][0]
    rows = []
    for record in reference['records']:
        cell = record['cell']
        data = episodes()
        for episode in data:
            episode['return'] += cell['H'] * 100 + cell['J'] * 10
            if cell['arm'] != 'fresh':
                episode['return'] += 7.
        rows.append((cell, data, reference))
    return rows


def test_rank_styles_include_reused_controls_without_changing_single_rank_colors():
    value = rank_campaign()
    assert len(value['cells']) == 144
    for prefix, _, arms in layout.campaign_curve_groups(value):
        assert len(arms) == 10 and arms[-1] == 'fresh'
        metadata = [layout.spectral_arm_metadata(value, arm) for arm in arms]
        assert len({item['color'] for item in metadata}) == 10
        assert sorted({item['requested_rank'] for item in metadata if item['method'] == 'svd'}) == [1, 4, 32]
        styles = {item['label']: item['color'] for item in metadata}
        assert styles['SVD r1 s1'] == '#0072b2'
        assert styles['SVD r4 s1'] == '#d55e00'
        assert styles['SVD r32 s1'] == '#009e73'
        definition = layout.campaign_chart_definition(value, prefix[:-1])
        assert definition['encoding']['strokeDash']['scale']['range'] == [[], [8, 3], [2, 3]]
    old = deepcopy(value)
    old['arms'] = {arm: specification for arm, specification in value['arms'].items()
                   if '_r1_' not in arm and '_r4_' not in arm}
    assert layout.spectral_arm_metadata(old, 'spectral_svd_r32_s1_actor')['color'] == '#0072b2'
    assert layout.spectral_arm_metadata(old, 'spectral_svd_r32_s1_actor_norm')['color'] == '#56b4e9'


def test_rank_view_preserves_existing_view_and_explains_new_and_reused_cells(tmp_path):
    value = rank_campaign()
    service = MultiChartService()
    before = deepcopy(service.views)
    receipt = layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), entity='entity',
        project='project', receipt_dir=tmp_path, run_id='rank14', campaign=value)
    assert service.views[:len(before)] == before
    spec = json.loads(service.views[-1]['spec'])
    intro = layout._bank(spec)['sections'][0]['panels'][0]['config']['value']
    assert '336 comparison configurations: 144 new and 192 reused' in intro
    assert 'SVD ranks 1, 4 and 32 are blue, orange and green' in intro
    assert 'completion counts cover only new cells' in intro
    assert 'b1f576d6-old' in intro
    assert len(receipt['expected_chart_keys']) == 168
    assert layout._saved_installed(spec, 'rank14', value, receipt['custom_chart_id'])


def test_reused_spectral_data_keep_original_sources_costs_and_pairing(tmp_path, monkeypatch):
    value = rank_campaign()
    old = reference_rows(value)
    original = deepcopy(old)
    monkeypatch.setattr(publisher, 'historical_results', lambda campaign: old)
    before = publisher.snapshot(tmp_path, value, True, {})
    assert before['total'] == 144 and before['completed'] == 0 and before['failed'] == 0
    assert before['comparison_total'] == len(before['settings']) == len(before['results']) == 336
    assert before['reused_settings'] == 192 and len(before['episodes']) == 576
    assert sum(row['state'] == 'pending' for row in before['settings']) == 144
    assert sum(row['state'] == 'historical_complete' for row in before['settings']) == 192
    assert all(row['source_commit'] == 'b1f576d6-old-source' and row['evaluation_origin'] == 'reused'
               for row in before['episodes'])
    reused = next(row for row in before['results'] if row['arm'] == 'spectral_svd_r32_s1_actor')
    assert reused['paired_vs_fresh_mean'] == 7. and reused['controller_seconds_per_decision'] == .02
    assert reused['spectral_filter_seconds_per_decision'] == .002
    assert any(row['source_commit'] == 'b1f576d6-old-source' and row['samples'] == 15
               for row in before['diagnostics'])
    new_cell = value['cells'][0]
    fresh = next(data for cell, data, _ in old
                 if (cell['H'], cell['J'], cell['arm']) == (new_cell['H'], new_cell['J'], 'fresh'))
    measured = deepcopy(fresh)
    for episode in measured:
        episode['return'] += 11.
    completed = {new_cell['name']: measured}
    after = publisher.snapshot(tmp_path, value, True, completed)
    assert after['completed'] == 1 and after['total'] == 144 and len(after['episodes']) == 579
    row = next(row for row in after['results'] if row['name'] == new_cell['name'])
    assert row['paired_vs_fresh_mean'] == 11. and row['paired_vs_fresh_std'] == 0.
    assert row['source_commit'] == 'new-source' and row['evaluation_origin'] == 'new'
    assert old == original
    payload = publisher.payload(fake_wandb(), before, value)
    for component in ('actor', 'critic', 'joint'):
        chart = payload[f'discovery/{component}/h1_return_vs_j']
        assert len(chart['keys']) == 10
        assert sum(len(points) for points in chart['ys']) == 24
        assert sum(not points for points in chart['ys']) == 4
    assert payload['campaign/total'] == 144 and payload['campaign/reused_settings'] == 192
    assert payload['campaign/comparison_total'] == 336


def test_reused_spectral_dispatches_strict_reference_verifier(monkeypatch):
    from slurm import ambi_spectral_transfer_campaign as launcher
    value = rank_campaign()
    expected = reference_rows(value)
    calls = []
    def verify(reference, current):
        calls.append((reference, current))
        return expected
    monkeypatch.setattr(launcher, 'verify_spectral_reference', verify)
    assert publisher.historical_results(value) == expected
    assert calls == [(value['historical_references'][0], value)]


def test_serialized_master_diagnostics_do_not_truncate_at_sdk_default(monkeypatch):
    import wandb
    monkeypatch.setattr(wandb.Table, 'MAX_ROWS', 10000)
    monkeypatch.setattr(wandb.Table, 'MAX_ARTIFACT_ROWS', 10000)
    value = rank_campaign()
    diagnostics = [dict(name=f'fixture-{i}', H=1, J=1, arm='fresh', metric='custom', mean=float(i))
                   for i in range(10003)]
    snapshot = dict(settings=[], results=[], episodes=[], diagnostics=diagnostics,
                    completed=0, total=144, failed=0, comparison_total=336, reused_settings=192)
    sdk = SimpleNamespace(Table=wandb.Table, plot=fake_wandb().plot)
    payload = publisher.payload(sdk, snapshot, value)
    serialized = payload['discovery/diagnostics']._to_table_json()
    assert len(serialized['data']) == 10003
    assert serialized['data'][-1][serialized['columns'].index('name')] == 'fixture-10002'
    assert wandb.Table.MAX_ROWS >= 10003 and wandb.Table.MAX_ARTIFACT_ROWS >= 10003
