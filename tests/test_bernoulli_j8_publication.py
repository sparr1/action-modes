"""The J8 extension publishes nine new cells beside immutable lower-J evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_bernoulli_transfer_publication import ChartService
from utils import wandb_transfer_discovery_layout as layout

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT/'configs/research/ambi_bernoulli_j8_575k.json'


def campaign():
    matrix = json.loads(MATRIX.read_text())
    arms = list(matrix['arms'])
    def cells(rounds, selected_arms):
        return [dict(index=i, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
                for i, (h, j, arm) in enumerate((h,j,arm) for h in matrix['horizons']
                    for j in rounds for arm in selected_arms)]
    fresh = dict(kind='fresh', source_commit='fresh-source',
        records=[dict(cell=c) for c in cells([1,2,4,6,8], ['rho_a0_c0'])])
    half = dict(kind='bernoulli', source_commit='half-source', probability=.5,
        records=[dict(cell=c) for c in cells([1,2,4,6], arms)])
    return dict(H=matrix['horizons'], J=matrix['rounds'], cells=cells([8], arms),
        diagnostics=matrix['diagnostics'], publication=matrix['publication'],
        historical_reference=fresh, historical_references=[fresh, half])


def test_j8_view_explicitly_distinguishes_new_and_reused_rounds(tmp_path):
    value = campaign(); service = ChartService(); before = deepcopy(service.views)
    receipt = layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service), entity='entity',
        project='project', receipt_dir=tmp_path, run_id='j8new123', campaign=value)
    assert receipt['url'].endswith('?nw=bernoullij8575j8new123')
    assert service.views[:len(before)] == before
    spec = json.loads(service.views[-1]['spec'])
    sections = layout._bank(spec)['sections']
    intro = sections[0]['panels'][0]['config']['value']
    assert '9 new configurations' in intro
    assert 'H=1,2,3; new J=8; plotted J=1,2,4,6,8;' in intro
    assert 'Reused 50% results and diagnostics' in intro
    assert 'initially collapsed' not in intro
    assert all(section['isOpen'] for section in sections)
    definition = json.loads(service.chart['spec'])
    assert definition['encoding']['color']['scale']['range'] == ['#0072b2', '#d55e00', '#009e73', '#000000']
    assert 'historical' in definition['encoding']['color']['scale']['domain'][-1]
    assert len(receipt['expected_chart_keys']) == 6 + 3 * len(layout.DIAGNOSTIC_CHARTS)
    assert layout._saved_installed(spec, 'j8new123', value, receipt['custom_chart_id'])


def test_generic_payload_appends_j8_without_rerunning_or_hiding_lower_j():
    value = campaign()
    rows = [dict(c, return_mean=108., controller_seconds_per_decision=.08,
                 state='complete', provenance='current campaign') for c in value['cells']]
    rows += [dict(record['cell'], return_mean=100.+record['cell']['J'],
                  controller_seconds_per_decision=.01*record['cell']['J'],
                  state='historical_complete', provenance=reference['kind'])
             for reference in value['historical_references'] for record in reference['records']]
    assert len(rows) == 60
    snapshot = dict(settings=value['cells'], results=rows, episodes=[], diagnostics=[], completed=9, total=9, failed=0)
    fake = SimpleNamespace(Table=lambda **kwargs: kwargs, plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    result = publisher.payload(fake, snapshot, value)
    assert result['campaign/completed'] == result['campaign/total'] == 9
    for h in (1,2,3):
        chart = result[f'discovery/h{h}_return_vs_j']
        assert chart['keys'] == list(dict.fromkeys(c['arm'] for c in value['cells'])) + ['rho_a0_c0']
        assert chart['xs'] == [[1,2,4,6,8]] * 4
        assert result[f'discovery/h{h}_return_vs_compute']['xs'] == [[.01,.02,.04,.06,.08]] * 4
    # A running J8 cell must not enter the result curve; historical fresh J8 stays visible.
    for row in rows:
        if row['provenance'] == 'current campaign':
            row['return_mean'] = None
    pending = publisher.payload(fake, snapshot, value)['discovery/h1_return_vs_j']
    assert pending['xs'] == [[1,2,4,6]] * 3 + [[1,2,4,6,8]]
