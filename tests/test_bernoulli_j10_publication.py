"""J10 appends nine evaluations to two distinct, immutable p50 references."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_bernoulli_transfer_publication import ChartService
from utils import wandb_transfer_discovery_layout as layout


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT/'configs/research/ambi_bernoulli_j10_575k.json'


def campaign():
    matrix = json.loads(MATRIX.read_text())
    arms = list(matrix['arms'])
    def cells(rounds, selected_arms):
        return [dict(index=i, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
                for i, (h, j, arm) in enumerate((h,j,arm) for h in matrix['horizons']
                    for j in rounds for arm in selected_arms)]
    fresh = dict(kind='fresh', source_commit='fresh-source',
        records=[dict(cell=c) for c in cells([1,2,4,6,8,10], ['rho_a0_c0'])])
    lower = dict(kind='bernoulli', source_commit='lower-source', probability=.5,
        records=[dict(cell=c) for c in cells([1,2,4,6], arms)])
    j8 = dict(kind='bernoulli', source_commit='j8-source', probability=.5,
        records=[dict(cell=c) for c in cells([8], arms)])
    return dict(H=matrix['horizons'], J=matrix['rounds'], cells=cells([10], arms),
        diagnostics=matrix['diagnostics'], publication=matrix['publication'],
        historical_reference=fresh, historical_references=[fresh, lower, j8])


def test_j10_view_labels_all_rounds_and_both_reused_sources(tmp_path):
    value = campaign(); service = ChartService(); before = deepcopy(service.views)
    args = dict(entity='entity', project='project', receipt_dir=tmp_path,
                run_id='j10new123', campaign=value)
    api = SimpleNamespace(_service_api=service)
    receipt = layout.ensure_discovery_saved_view(api, **args)
    assert receipt['url'].endswith('?nw=bernoullij10575j10new123')
    assert service.views[:len(before)] == before
    spec = json.loads(service.views[-1]['spec'])
    sections = layout._bank(spec)['sections']
    intro = sections[0]['panels'][0]['config']['value']
    assert '9 new configurations' in intro
    assert 'H=1,2,3; new J=10; plotted J=1,2,4,6,8,10;' in intro
    assert '`lower-source`' in intro and '`j8-source`' in intro
    assert all(section['isOpen'] for section in sections)
    assert layout._saved_installed(spec, 'j10new123', value, receipt['custom_chart_id'])
    assert json.loads(service.chart['spec'])['encoding']['color']['scale']['range'] == [
        '#0072b2', '#d55e00', '#009e73', '#000000']
    assert not layout.ensure_discovery_saved_view(api, **args)['changed']
    assert service.chart_writes == service.writes == 1


def test_j10_combined_snapshot_pairs_all_rounds_and_preserves_source_provenance(tmp_path, monkeypatch):
    value = campaign()
    def episodes(gain):
        return [dict(seed=seed, solver_seed=seed+1000, **{'return':seed+gain},
                     length=500, control_seconds=10.) for seed in (101,102,103)]
    historical = [(record['cell'], episodes(0 if ref['kind'] == 'fresh' else 7), ref)
        for ref in value['historical_references'] for record in ref['records']]
    monkeypatch.setattr(publisher, 'historical_results', lambda _: historical)
    completed = {cell['name']:episodes(10) for cell in value['cells']}
    snapshot = publisher.snapshot(tmp_path/'new', value, False, completed)
    assert snapshot['completed'] == snapshot['total'] == 9 and snapshot['failed'] == 0
    assert len(snapshot['settings']) == 9 and len(snapshot['results']) == 72
    assert len({row['name'] for row in snapshot['results']}) == 72
    assert len(snapshot['episodes']) == 216
    fresh = [row for row in snapshot['results'] if row['arm'] == 'rho_a0_c0']
    lower = [row for row in snapshot['results'] if row['provenance'].endswith('; lower-source')]
    j8 = [row for row in snapshot['results'] if row['provenance'].endswith('; j8-source')]
    assert len(fresh) == 18 and len(lower) == 36 and len(j8) == 9
    assert all(row['paired_vs_fresh_mean'] == 7 for row in lower+j8)
    assert all(row['paired_vs_fresh_mean'] == 10 for row in snapshot['results'][:9])
    historical.append(historical[-1])
    with pytest.raises(ValueError, match='duplicated'):
        publisher.snapshot(tmp_path/'new', value, False, completed)


def test_j10_curves_include_six_budgets_but_never_partial_new_results():
    value = campaign()
    rows = [dict(c, return_mean=110., controller_seconds_per_decision=.10,
                 state='complete', provenance='current campaign') for c in value['cells']]
    rows += [dict(record['cell'], return_mean=100.+record['cell']['J'],
                  controller_seconds_per_decision=.01*record['cell']['J'],
                  state='historical_complete', provenance=reference['source_commit'])
             for reference in value['historical_references'] for record in reference['records']]
    snapshot = dict(settings=value['cells'], results=rows, episodes=[], diagnostics=[], completed=9, total=9, failed=0)
    fake = SimpleNamespace(Table=lambda **kwargs: kwargs, plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    result = publisher.payload(fake, snapshot, value)
    assert all(key in result for key in layout.expected_campaign_chart_keys(value))
    for h in (1,2,3):
        assert result[f'discovery/h{h}_return_vs_j']['xs'] == [[1,2,4,6,8,10]] * 4
        assert result[f'discovery/h{h}_return_vs_compute']['xs'] == [[.01,.02,.04,.06,.08,.10]] * 4
    for row in rows[:9]:
        row['return_mean'] = None
    pending = publisher.payload(fake, snapshot, value)['discovery/h1_return_vs_j']
    assert pending['xs'] == [[1,2,4,6,8]] * 3 + [[1,2,4,6,8,10]]
