"""Publish J12 beside reused evidence without inventing a fresh J12 control."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_bernoulli_transfer_publication import ChartService
from utils import wandb_transfer_discovery_layout as layout


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'configs/research/ambi_bernoulli_j12_575k.json'


def campaign():
    matrix = json.loads(MATRIX.read_text())
    arms = list(matrix['arms'])

    def cells(rounds, selected_arms):
        combinations = ((h, j, arm) for h in matrix['horizons']
                        for j in rounds for arm in selected_arms)
        return [dict(index=i, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
                for i, (h, j, arm) in enumerate(combinations)]

    fresh = dict(kind='fresh', source_commit='fresh-source',
        records=[dict(cell=c) for c in cells([1,2,4,6,8,10], ['rho_a0_c0'])])
    lower = dict(kind='bernoulli', source_commit='lower-source', probability=.5,
        records=[dict(cell=c) for c in cells([1,2,4,6], arms)])
    j8 = dict(kind='bernoulli', source_commit='j8-source', probability=.5,
        records=[dict(cell=c) for c in cells([8], arms)])
    j10 = dict(kind='bernoulli', source_commit='j10-source', probability=.5,
        records=[dict(cell=c) for c in cells([10], arms)])
    return dict(H=matrix['horizons'], J=matrix['rounds'], cells=cells([12], arms),
        diagnostics=matrix['diagnostics'], publication=matrix['publication'],
        historical_reference=fresh, historical_references=[fresh, lower, j8, j10])


def episodes(gain):
    return [dict(seed=seed, solver_seed=seed+1000, **{'return':seed+gain},
                 length=500, control_seconds=10.) for seed in (101,102,103)]


def history(value):
    return [(record['cell'], episodes(0 if ref['kind'] == 'fresh' else 7), ref)
        for ref in value['historical_references'] for record in ref['records']]


def test_j12_view_labels_new_and_historical_rounds_and_preserves_other_views(tmp_path):
    value = campaign()
    service = ChartService()
    before = deepcopy(service.views)
    args = dict(entity='entity', project='project', receipt_dir=tmp_path,
                run_id='j12new123', campaign=value)
    api = SimpleNamespace(_service_api=service)
    receipt = layout.ensure_discovery_saved_view(api, **args)
    assert receipt['url'].endswith('?nw=bernoullij12575j12new123')
    assert service.views[:len(before)] == before
    spec = json.loads(service.views[-1]['spec'])
    sections = layout._bank(spec)['sections']
    intro = sections[0]['panels'][0]['config']['value']
    assert '9 new configurations' in intro
    assert 'H=1,2,3; new J=12; plotted J=1,2,4,6,8,10,12;' in intro
    assert all(f'`{source}`' in intro for source in
               ('fresh-source', 'lower-source', 'j8-source', 'j10-source'))
    assert 'No fresh-prior controls are available at J=12' in intro
    assert 'paired gains at these rounds remain unavailable' in intro
    assert all(section['isOpen'] for section in sections)
    assert len(receipt['expected_chart_keys']) == 33
    assert sum(panel['viewType'] == 'Vega2' for section in sections
               for panel in section['panels']) == 33
    assert layout._saved_installed(spec, 'j12new123', value, receipt['custom_chart_id'])
    assert json.loads(service.chart['spec'])['encoding']['color']['scale']['range'] == [
        '#0072b2', '#d55e00', '#009e73', '#000000']
    assert not layout.ensure_discovery_saved_view(api, **args)['changed']
    assert service.chart_writes == service.writes == 1


def test_j12_complete_snapshot_preserves_references_without_fabricating_paired_gains(tmp_path, monkeypatch):
    value = campaign()
    historical = history(value)
    monkeypatch.setattr(publisher, 'historical_results', lambda _: historical)
    completed = {cell['name']:episodes(10) for cell in value['cells']}
    snapshot = publisher.snapshot(tmp_path/'new', value, False, completed)
    assert snapshot['completed'] == snapshot['total'] == 9 and snapshot['failed'] == 0
    assert len(snapshot['settings']) == 9 and len(snapshot['results']) == 81
    assert len({row['name'] for row in snapshot['results']}) == 81
    assert len(snapshot['episodes']) == 243
    fresh = [row for row in snapshot['results'] if row['arm'] == 'rho_a0_c0']
    assert len(fresh) == 18 and {row['J'] for row in fresh} == {1,2,4,6,8,10}
    for source, count in [('lower-source', 36), ('j8-source', 9), ('j10-source', 9)]:
        reused = [row for row in snapshot['results']
                  if row['provenance'].endswith('; '+source)]
        assert len(reused) == count
        assert all(row['paired_vs_fresh_mean'] == 7 for row in reused)
    current = snapshot['results'][:9]
    assert all(row['state'] == 'complete' and row['J'] == 12 for row in current)
    assert all(row['return_mean'] == 112 for row in current)
    assert all(row['paired_vs_fresh_mean'] is None and
               row['paired_vs_fresh_std'] is None for row in current)
    historical.append(historical[-1])
    with pytest.raises(ValueError, match='duplicated'):
        publisher.snapshot(tmp_path/'new', value, False, completed)


def test_j12_pending_snapshot_shows_progress_without_hiding_finished_references(tmp_path, monkeypatch):
    value = campaign()
    monkeypatch.setattr(publisher, 'historical_results', lambda _: history(value))
    snapshot = publisher.snapshot(tmp_path/'new', value, True, {})
    assert snapshot['completed'] == snapshot['failed'] == 0 and snapshot['total'] == 9
    assert all(row['state'] == 'pending' and row['completed_episodes'] == 0
               for row in snapshot['settings'])
    assert len(snapshot['results']) == 81 and len(snapshot['episodes']) == 216
    assert all(row['return_mean'] is None and row['paired_vs_fresh_mean'] is None
               for row in snapshot['results'][:9])
    assert all(row['state'] == 'historical_complete' for row in snapshot['results'][9:])


def test_j12_curves_extend_transfer_only_and_never_plot_pending_new_results():
    value = campaign()
    rows = [dict(c, return_mean=112., controller_seconds_per_decision=.12,
                 state='complete', provenance='current campaign') for c in value['cells']]
    rows += [dict(record['cell'], return_mean=100.+record['cell']['J'],
                  controller_seconds_per_decision=.01*record['cell']['J'],
                  state='historical_complete', provenance=reference['source_commit'])
             for reference in value['historical_references'] for record in reference['records']]
    snapshot = dict(settings=value['cells'], results=rows, episodes=[], diagnostics=[],
                    completed=9, total=9, failed=0)
    fake = SimpleNamespace(Table=lambda **kwargs: kwargs,
                           plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    result = publisher.payload(fake, snapshot, value)
    assert all(key in result for key in layout.expected_campaign_chart_keys(value))
    for h in (1,2,3):
        curve = result[f'discovery/h{h}_return_vs_j']
        assert curve['keys'] == list(dict.fromkeys(c['arm'] for c in value['cells'])) + ['rho_a0_c0']
        assert curve['xs'] == [[1,2,4,6,8,10,12]] * 3 + [[1,2,4,6,8,10]]
        assert result[f'discovery/h{h}_return_vs_compute']['xs'] == [
            [.01,.02,.04,.06,.08,.10,.12]] * 3 + [[.01,.02,.04,.06,.08,.10]]
    for row in rows[:9]:
        row['return_mean'] = None
    pending = publisher.payload(fake, snapshot, value)
    assert all(pending[f'discovery/h{h}_return_vs_j']['xs'] == [[1,2,4,6,8,10]] * 4
               for h in (1,2,3))


def test_fresh_coverage_caveat_is_absent_when_current_rounds_have_controls():
    value = campaign()
    value['J'] = [10]
    for cell in value['cells']:
        cell['J'] = 10
    intro = layout.campaign_sections(value, 'chart')[0]['panels'][0]['config']['value']
    assert 'No fresh-prior controls are available' not in intro
