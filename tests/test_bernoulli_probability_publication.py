"""The probability screen reuses complete panels with explicit provenance."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_discovery_publish as publisher
from slurm import ambi_transfer_discovery_campaign as launcher
from tests.test_bernoulli_transfer_publication import ChartService
from tests.test_transfer_discovery_publication import fixture, _save
from utils import wandb_transfer_discovery_layout as layout

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT/'configs/research/ambi_bernoulli_probability_575k.json'


def campaign():
    matrix = json.loads(MATRIX.read_text())
    cells = [dict(index=i, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
             for i, (j, h, arm) in enumerate((j,h,arm) for j in reversed(matrix['rounds'])
                 for h in matrix['horizons'] for arm in matrix['arms'])]
    return dict(H=matrix['horizons'], J=matrix['rounds'], cells=cells,
        diagnostics=matrix['diagnostics'], publication=matrix['publication'],
        historical_reference={'source_commit':'fresh-source'},
        historical_references=[{'kind':'bernoulli', 'source_commit':'half-source'}])


class MultiChartService(ChartService):
    def __init__(self):
        super().__init__()
        self.charts = {}

    def execute_graphql(self, query, variables):
        if 'query TransferChart' in query:
            return {'customChart': deepcopy(self.charts.get(variables['id']))}
        response = super().execute_graphql(query, variables)
        if 'mutation CreateTransferChart' in query:
            self.charts[self.chart['id']] = deepcopy(self.chart)
        return response


def test_separate_component_charts_have_three_probabilities_and_fresh():
    value = campaign()
    groups = layout.campaign_curve_groups(value)
    assert [prefix for prefix, _, _ in groups] == ['actor/', 'critic/', 'joint/']
    for prefix, _, arms in groups:
        assert len(arms) == 4 and arms[-1] == 'rho_a0_c0'
        component = prefix[:-1]
        assert [value['publication']['arm_probabilities'][a] for a in arms[:-1]] == [.25, .5, .75]
        spec = layout.campaign_chart_definition(value, component)
        assert len(spec['encoding']['color']['scale']['domain']) == 4
        assert spec['encoding']['color']['scale']['range'][-1] == '#000000'
        assert spec['encoding']['strokeDash']['scale']['range'][:3] == [[2,3], [], [8,3]]
        assert all(component in label.lower() for label in spec['encoding']['color']['scale']['domain'][:-1])


def test_probability_view_is_isolated_and_schema_payload_keys_match(tmp_path):
    service = MultiChartService(); before = deepcopy(service.views)
    value = campaign()
    args = dict(entity='entity', project='project', receipt_dir=tmp_path, run_id='prob123', campaign=value)
    receipt = layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), **args)
    assert receipt['url'].endswith('?nw=bernoullip575prob123')
    assert service.views[:len(before)] == before and service.chart_writes == 3 and service.writes == 1
    assert len(receipt['custom_chart_id']) == 3
    saved = json.loads(service.views[-1]['spec'])
    sections = layout._bank(saved)['sections']
    assert all(not s['isOpen'] for s in sections if 'diagnostic' in s['__id__'])
    intro = sections[0]['panels'][0]['config']['value']
    assert '72 new configurations' in intro and '50% reuses' in intro and 'half-source' in intro
    snapshot = dict(settings=[], results=[], episodes=[], diagnostics=[], completed=0, total=72, failed=0)
    fake = SimpleNamespace(Table=lambda **kwargs: kwargs, plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    payload = publisher.payload(fake, snapshot, value)
    assert len(receipt['expected_chart_keys']) == 18 + 9 * len(layout.DIAGNOSTIC_CHARTS)
    assert all(key in payload for key in receipt['expected_chart_keys'])
    assert all(len(payload[key]['ys']) == 4 for key in receipt['expected_chart_keys'])
    assert layout._saved_installed(saved, 'prob123', value, receipt['custom_chart_id'])
    assert not layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), **args)['changed']
    assert service.chart_writes == 3 and service.writes == 1


def _historical(root, arm, *, gain=0., diagnostics=False):
    old, original_cell, old_directory, manifest, result = fixture(root)
    cell = dict(original_cell, name='h1_j1_'+arm, arm=arm)
    directory = root/'settings'/cell['name']
    old['cells'] = [cell]
    old['diagnostics'] = {}
    for value in (manifest, result):
        value.update(cell_id=cell['name'], arm=arm)
    for episode in result['episodes']:
        episode['return'] += gain
        if diagnostics:
            episode.update(diagnostic_seconds=3., diagnostic_samples=6,
                diagnostics={'enabled':True, 'summary':{'initial_critic_rmse':2.}})
    _save(root/'campaign.json', old)
    _save(directory/'manifest.json', manifest); _save(directory/'results.json', result)
    _save(directory/'worker-completion.json', dict(status='complete', cell=cell, smoke=False,
        source_commit=old['source_commit'], campaign_sha256=launcher.digest(root/'campaign.json'),
        result_sha256=launcher.digest(directory/'results.json'), manifest_sha256=launcher.digest(directory/'manifest.json')))
    reference = dict(kind='fresh' if arm == 'rho_a0_c0' else 'bernoulli', probability=None if arm == 'rho_a0_c0' else .5,
        root=str(root), source_commit=old['source_commit'], campaign_sha256=launcher.digest(root/'campaign.json'),
        records=[dict(cell=cell, hashes={name:launcher.digest(directory/name)
            for name in ('manifest.json', 'results.json', 'worker-completion.json')})])
    return reference, result['episodes']


def test_reused_half_diagnostics_and_pairing_are_visible_without_new_completion(tmp_path):
    fresh, _ = _historical(tmp_path/'fresh', 'rho_a0_c0')
    half, data = _historical(tmp_path/'half', 'bernoulli_a05_c0', gain=7., diagnostics=True)
    value = campaign(); value['diagnostics'] = {}
    value['cells'] = [dict(index=0, name='new', H=1, J=1, arm='bernoulli_a025_c0')]
    value['historical_reference'] = fresh; value['historical_references'] = [fresh, half]
    snapshot = publisher.snapshot(tmp_path/'new', value, True, {})
    assert snapshot['total'] == 1 and snapshot['completed'] == 0
    assert snapshot['settings'][0]['state'] == 'pending' and len(snapshot['settings']) == 1
    assert len(snapshot['results']) == 3 and len(snapshot['episodes']) == 6
    row = next(row for row in snapshot['results'] if row['arm'] == 'bernoulli_a05_c0')
    assert row['paired_vs_fresh_mean'] == 7. and row['retention_probability'] == .5
    assert row['component'] == 'actor' and row['diagnostic_samples'] == 18
    diagnostic = snapshot['diagnostics'][0]
    assert diagnostic['mean'] == 2. and diagnostic['samples'] == 18
    assert diagnostic['provenance'].startswith('historical Bernoulli p=0.5; ')
    assert 'provenance' in publisher.COLUMNS['diagnostics']
    value['historical_references'].append(half)
    with pytest.raises(ValueError, match='duplicated'):
        publisher.snapshot(tmp_path/'new', value, True, {})


def test_historical_diagnostic_configuration_mismatch_is_rejected(tmp_path):
    half, _ = _historical(tmp_path/'half', 'bernoulli_a05_c0')
    value = campaign(); value['historical_references'] = [half]
    with pytest.raises(ValueError, match='diagnostic configuration differs'):
        publisher.historical_results(value)
