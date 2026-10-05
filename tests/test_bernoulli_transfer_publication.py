"""A separate Bernoulli campaign preserves historical provenance and visible metrics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from slurm import ambi_transfer_discovery_campaign as launcher
from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_transfer_discovery_publication import fixture, _save
from tests.test_transfer_discovery_saved_layout import Service
from utils import wandb_transfer_discovery_layout as layout

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'configs/research/ambi_bernoulli_transfer_575k.json'


def campaign():
    matrix = json.loads(MATRIX.read_text())
    cells = [dict(index=i, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
             for i, (j, h, arm) in enumerate((j,h,arm) for j in reversed(matrix['rounds'])
                 for h in matrix['horizons'] for arm in matrix['arms'])]
    return dict(H=matrix['horizons'], J=matrix['rounds'], cells=cells,
                diagnostics=matrix['diagnostics'], publication=matrix['publication'])


class ChartService(Service):
    def __init__(self, *, chart_timeout=False, chart_apply=True):
        super().__init__()
        self.chart = None
        self.chart_writes = 0
        self.chart_timeout = chart_timeout
        self.chart_apply = chart_apply

    def execute_graphql(self, query, variables):
        if 'query TransferChart' in query:
            return {'customChart': deepcopy(self.chart)}
        if 'mutation CreateTransferChart' in query:
            self.chart_writes += 1
            if self.chart_apply:
                self.chart = dict(id=variables['entity']+'/'+variables['name'],
                    name=variables['name'], type=variables['type'], spec=variables['spec'])
            if self.chart_timeout:
                raise TimeoutError('Response lost')
            return {'createCustomChart': {'chart': self.chart}}
        return super().execute_graphql(query, variables)


def test_campaign_matrix_and_six_smokes_are_bound_before_launch(tmp_path, monkeypatch):
    matrix = json.loads(MATRIX.read_text())
    assert matrix['rounds'] == [1, 2, 4, 6] and matrix['horizons'] == [1, 2, 3]
    assert matrix['seeds'] == [101, 102, 103] and matrix['max_steps'] == 500
    assert matrix['diagnostics']['decisions'] == [0, 1, 25, 100, 250, 499]
    assert matrix['diagnostics']['stationary_decisions'] == [25, 250]
    assert matrix['diagnostics']['fit_steps'] == 4
    cells = campaign()['cells']
    assert len(cells) == 36
    assert [a.get('actor_bernoulli_p', 0.) for a in matrix['arms'].values()] == [.5, 0., .5]
    assert [a.get('critic_bernoulli_p', 0.) for a in matrix['arms'].values()] == [0., .5, .5]
    checkpoint = tmp_path / 'checkpoint.pt'
    real_digest = launcher.digest
    monkeypatch.setattr(launcher, 'digest', lambda p: launcher.CHECKPOINT_SHA if Path(p) == checkpoint
        else launcher.METADATA_SHA if str(p) == str(checkpoint)+'.metadata.json' else real_digest(p))
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda args, **kwargs: json.dumps(cells))
    monkeypatch.setattr(launcher, 'source', lambda: dict(source_commit='tested', source_tree='tree', source_dir=str(ROOT)))
    out = tmp_path / 'new'
    launcher.prepare(SimpleNamespace(root=out, matrix=MATRIX, checkpoint=checkpoint, reference_root=None))
    saved = json.loads((out/'campaign.json').read_text())
    assert saved['matrix_sha256'] == real_digest(MATRIX)
    assert saved['diagnostics'] == matrix['diagnostics']
    smoke = [saved['cells'][i] for i in saved['smoke_indices']]
    assert len(smoke) == 6 and {c['arm'] for c in smoke} == set(matrix['arms'])
    assert {c['H'] for c in smoke} == {1, 2, 3} and max(c['J'] for c in smoke) == 6


def test_unique_colored_campaign_view_with_visible_diagnostics_is_idempotent(tmp_path):
    service = ChartService()
    before = deepcopy(service.views)
    value = campaign()
    value['historical_reference'] = dict(source_commit='original-source')
    options = dict(entity='entity', project='project', receipt_dir=tmp_path, run_id='new123', campaign=value)
    receipt = layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), **options)
    assert receipt['url'] == 'https://wandb.ai/entity/project?nw=bernoulli575new123'
    assert service.views[:len(before)] == before
    spec = json.loads(service.views[-1]['spec'])
    chart_id = layout.campaign_chart_id('entity', value)
    assert layout._saved_installed(spec, 'new123', value, chart_id)
    charts = [p for s in layout._bank(spec)['sections'] for p in s['panels'] if p['viewType'] == 'Vega2']
    assert len(charts) == 6 + 3 * len(layout.DIAGNOSTIC_CHARTS)
    assert {p['config']['panelDefId'] for p in charts} == {chart_id}
    definition = json.loads(service.chart['spec'])
    colors = definition['encoding']['color']['scale']['range']
    assert len(set(colors)) == 4 and colors[-1] == '#000000'
    assert 'historical' in definition['encoding']['color']['scale']['domain'][-1]
    assert not layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), **options)['changed']
    assert service.chart_writes == service.writes == 1


def test_uncertain_chart_create_is_reconciled_without_duplicate(tmp_path):
    service = ChartService(chart_timeout=True)
    identifier = layout.ensure_campaign_chart(SimpleNamespace(_service_api=service), entity='entity', campaign=campaign())
    assert identifier == service.chart['id']
    assert layout.ensure_campaign_chart(SimpleNamespace(_service_api=service), entity='entity', campaign=campaign()) == identifier
    assert service.chart_writes == 1


@pytest.mark.parametrize('failure', ['missing', 'modified'])
def test_chart_registration_failure_is_explicit_before_view_creation(tmp_path, failure):
    service = ChartService(chart_apply=False)
    if failure == 'modified':
        service.chart = dict(type='vega2', spec='{}')
    with pytest.raises(layout.ResultsLayoutError, match='chart registration'):
        layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), entity='entity', project='project',
            receipt_dir=tmp_path, run_id='new123', campaign=campaign())
    assert service.writes == 0


def test_historical_fresh_is_paired_but_not_counted_as_new_work(tmp_path):
    old_root = tmp_path/'old'
    old, old_cell, directory, _, old_result = fixture(old_root)
    new_root = tmp_path/'new'
    new, cell, _, _, result = fixture(new_root)
    # New result represents one Bernoulli cell with the same paired episode panel.
    new['cells'] = [dict(cell, name='new_cell', arm='bernoulli_a05_c0')]
    new_cell = new['cells'][0]
    new['historical_reference'] = dict(root=str(old_root), campaign_sha256=launcher.digest(old_root/'campaign.json'),
        source_commit=old['source_commit'], records=[dict(cell=old_cell, hashes={
            name: launcher.digest(directory/name) for name in ('results.json','manifest.json','worker-completion.json')})])
    values = deepcopy(result['episodes'])
    for episode in values:
        episode['return'] += 7.
    snapshot = publisher.snapshot(new_root, new, True, {new_cell['name']: values})
    assert snapshot['completed'] == snapshot['total'] == 1
    assert len(snapshot['results']) == 2 and len(snapshot['settings']) == 1
    assert snapshot['results'][0]['paired_vs_fresh_mean'] == 7.
    assert snapshot['results'][1]['state'] == 'historical_complete'
    assert snapshot['results'][1]['diagnostic_seconds_per_decision'] is None
    old_result['episodes'][0]['return'] += 1.
    _save(directory/'results.json', old_result)
    with pytest.raises(ValueError, match='reference changed'):
        publisher.snapshot(new_root, new, True, {new_cell['name']: values})


def test_diagnostic_payload_keys_match_visible_charts_and_keep_empty_values_missing():
    value = dict(settings=[], results=[], episodes=[], diagnostics=[], completed=0, total=36, failed=0)
    fake = SimpleNamespace(Table=lambda **kwargs: kwargs,
        plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    result = publisher.payload(fake, value, campaign())
    assert result['campaign/completed'] == 0 and 'discovery/diagnostics' in result
    for metric, _ in layout.DIAGNOSTIC_CHARTS:
        for h in (1, 2, 3):
            chart = result[f'discovery/h{h}_{metric}_vs_j']
            assert chart['ys'] == [[], [], []]


def test_diagnostic_smoke_requires_explicit_controller_rng_isolation(tmp_path, monkeypatch):
    config, cell, directory, manifest, result = fixture(tmp_path)
    config['diagnostics'] = {'enabled': True}
    config['smoke_seeds'] = config['seeds']
    config['smoke_steps'] = config['max_steps']
    for value in (manifest, result):
        value['smoke'] = True
    checked = []
    monkeypatch.setitem(sys.modules, 'utils.transfer_campaign_diagnostics',
        SimpleNamespace(verify_episode_diagnostics=lambda *args, **kwargs: checked.append(True)))
    _save(directory/'manifest.json', manifest)
    _save(directory/'results.json', result)
    with pytest.raises(ValueError, match='isolation verification'):
        launcher.validate_result(directory, config, cell, smoke=True)
    for episode in result['episodes']:
        episode['diagnostic_isolation_verified'] = True
    _save(directory/'results.json', result)
    launcher.validate_result(directory, config, cell, smoke=True)
    assert len(checked) == 4
