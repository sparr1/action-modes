"""C32 comparisons reuse matched C16 observations without inventing pending data."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from slurm import ambi_closed_loop_publish as publication


@pytest.fixture
def budget_comparison():
    rounds = (1, 2, 4, 6, 8, 10, 12, 14)
    prior = [dict(seed=101+i, solver_seed=700+i, length=500,
                  truncated_by_evaluator=False, **{'return': float(100+i)}) for i in range(5)]
    cells, references, pins, completed = [], {}, {}, {}
    for j in rounds:
        name = f'return_return_alpha_h3_j{j}_c32'
        cells.append(dict(name=name, H=3, J=j, critic_kind='return_only',
            params=dict(inner_critic_updates_per_round=32, inner_actor_updates_per_round=4,
                        inner_rollouts_per_round=128, inner_batch_size=256),
            performance_run_id=f'new-performance-{j}', training_run_id=f'new-training-{j}'))
        pins[str(j)] = dict(bundle=f'/reference/j{j}', manifest_sha256=f'manifest-{j}',
            source_commit=f'commit-{j}', selector=f'critic_compare/return_return_alpha_h3_j{j}_c16',
            critic_updates_per_round=16, performance_run_id=f'old-performance-{j}',
            training_run_id=f'old-training-{j}')
        references[j] = [{**ep, 'return': ep['return']+j, 'paired_return_delta': j} for ep in prior]
        completed[name] = [{**ep, 'return': ep['return']+j*2+i, 'paired_return_delta': j*2+i}
                           for i, ep in enumerate(prior)]
    state = dict(cells=cells, comparison_references=pins, reference='/prior',
                 prior_manifest_sha256='prior-hash', checkpoint_step=575000,
                 checkpoint_sha256='checkpoint-hash', source_run='backbone')
    return state, prior, references, completed


def test_pending_budget_points_keep_c16_visible_without_c32_measurements(budget_comparison):
    campaign, prior, references, _ = budget_comparison
    result = publication.aggregate_results(campaign, prior, {}, references)
    assert len(result['budget_points']) == 16
    assert len(result['budget_differences']) == 8
    assert len(result['budget_episodes']) == 40
    assert all('mean' not in row for row in result['budget_differences'])
    for point in result['budget_points']:
        assert point['prior_return_mean'] == 102.
        if point['C'] == 16:
            assert point['return_mean'] == 102.+point['J']
            assert point['reused'] and point['status'] == 'reused'
            assert point['paired_gain_ci95_low'] == point['paired_gain_ci95_high'] == point['J']
        else:
            assert 'return_mean' not in point and not point['reused']
    rows = dict(publication.numeric_rows(result))
    assert len(rows) == 8
    assert all(key.startswith('reference_C16_J') for key in rows)
    plots = publication.chart_payloads(campaign, result)
    assert plots['comparison/critic_budget_return_mean_vs_J']['xs'][1] == []
    assert 'comparison/c32_minus_c16_vs_J' not in plots


def test_direct_difference_is_paired_c32_minus_c16_with_established_bootstrap(budget_comparison):
    campaign, prior, references, completed = budget_comparison
    name = campaign['cells'][3]['name']
    partial = {name: list(reversed(completed[name]))}
    result = publication.aggregate_results(campaign, prior, partial, references)
    point, = [p for p in result['budget_differences'] if 'mean' in p]
    assert point['J'] == 6 and point['mean'] == 8. and point['paired_episodes'] == 5
    delta = np.array([6., 7., 8., 9., 10.])
    draws = np.random.default_rng(20260912).integers(0, 5, size=(2000, 5))
    lo, hi = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
    assert point['ci95_low'] == lo and point['ci95_high'] == hi
    assert point['std'] == pytest.approx(delta.std(ddof=1))
    assert [ep['c32_minus_c16'] for ep in result['budget_difference_episodes']] == list(delta)
    rows = dict(publication.numeric_rows(result))
    assert rows['paired_C32_C16_J6']['closed_loop/c32_minus_c16/mean'] == 8.
    assert 'paired_C32_C16_J10' not in rows


def test_budget_overview_exposes_numeric_axes_confidence_intervals_and_actual_status(budget_comparison):
    campaign, prior, references, completed = budget_comparison
    result = publication.aggregate_results(campaign, prior, completed, references)
    statuses = [dict(setting=cell['name'], status='publishing') for cell in campaign['cells']]
    fake = SimpleNamespace(Table=lambda **kw: kw, plot=SimpleNamespace(line_series=lambda **kw: kw))
    payload = publication.overview_log(fake, campaign, result, statuses)
    table = payload['comparison/critic_budget_points']
    points = [dict(zip(table['columns'], row)) for row in table['data']]
    assert len(points) == 16
    assert all(point['status'] == ('reused' if point['C'] == 16 else 'publishing') for point in points)
    assert all(point['H'] == 3 and point['A'] == 4 and point['N'] == 128 and point['B'] == 256 for point in points)
    assert len(payload['comparison/c32_minus_c16']['data']) == 8
    assert len(payload['comparison/critic_budget_episodes']['data']) == 80
    assert publication.campaign_budget(campaign) == dict(C=32, A=4, N=128, B=256)


@pytest.mark.parametrize('damage', ['missing_reference', 'solver', 'duplicate', 'incomplete'])
def test_budget_aggregation_rejects_unmatched_reference_data(budget_comparison, damage):
    campaign, prior, references, completed = budget_comparison
    if damage == 'missing_reference': references.pop(1)
    if damage == 'solver': references[1][0]['solver_seed'] += 1
    if damage == 'duplicate': references[1][0] = deepcopy(references[1][1])
    if damage == 'incomplete': references[1][0]['length'] = 3
    with pytest.raises(ValueError):
        publication.aggregate_results(campaign, prior, completed, references)


@pytest.fixture
def pinned_comparison(budget_comparison, monkeypatch):
    campaign, prior, references, completed = budget_comparison
    protocol = dict(environment={'id': 'DMControl-v0', 'params': {'task': 'humanoid-walk'}},
        observation='state', seeds=[101,102,103,104,105], controller_seed=55, seed_scheme='sha256-v1',
        max_steps=500, action_rule='tanh_mean')
    manifests = {j: dict(checkpoint={'source_run': 'backbone'},
        code=dict(commit=f'commit-{j}', dirty=False, runtime={'version': 'same'}),
        protocol=protocol, reference={'manifest_sha256': 'prior-hash'},
        runs=[dict(selector=campaign['comparison_references'][str(j)]['selector'], episodes=eps)])
        for j, eps in references.items()}
    prior_manifest = dict(code={'runtime': {'version': 'same'}}, protocol=protocol, runs=[{'episodes': prior}])
    def fake_digest(path):
        return 'prior-hash' if str(path).startswith('/prior/') else 'manifest-'+path.parent.name[1:]
    def fake_read(path):
        return prior_manifest if str(path).startswith('/prior/') else manifests[int(path.parent.name[1:])]
    validated = []
    def fake_validate(bundle, expected, **kwargs):
        original = campaign['cells'][[cell['J'] for cell in campaign['cells']].index(expected['J'])]
        exact = deepcopy(original)
        exact['params']['inner_critic_updates_per_round'] = 16
        assert expected == exact
        assert kwargs == dict(checkpoint_step=575000, checkpoint_sha='checkpoint-hash')
        validated.append(expected['J'])
        return manifests[expected['J']]
    monkeypatch.setattr(publication, 'digest', fake_digest)
    monkeypatch.setattr(publication, 'read', fake_read)
    monkeypatch.setattr(publication, 'validate', fake_validate)
    return campaign, manifests, validated, references


def test_reference_loader_validates_every_j_with_only_C_changed(pinned_comparison):
    campaign, manifests, validated, references = pinned_comparison
    assert publication.load_comparison_references(campaign) == references
    assert validated == [1,2,4,6,8,10,12,14]
    assert all(cell['params']['inner_critic_updates_per_round'] == 32 for cell in campaign['cells'])


@pytest.mark.parametrize('damage', ['hash', 'source', 'dirty', 'selector', 'runtime', 'prior', 'checkpoint', 'gain', 'missing'])
def test_reference_loader_rejects_changed_provenance_before_publication(pinned_comparison, damage):
    campaign, manifests, _, _ = pinned_comparison
    manifest = manifests[1]
    if damage == 'hash': campaign['comparison_references']['1']['manifest_sha256'] = 'changed'
    if damage == 'source': manifest['code']['commit'] = 'changed'
    if damage == 'dirty': manifest['code']['dirty'] = True
    if damage == 'selector': manifest['runs'][0]['selector'] = 'changed'
    if damage == 'runtime': manifest['code']['runtime'] = {'version': 'changed'}
    if damage == 'prior': manifest['reference']['manifest_sha256'] = 'changed'
    if damage == 'checkpoint': manifest['checkpoint']['source_run'] = 'changed'
    if damage == 'gain': manifest['runs'][0]['episodes'][0]['paired_return_delta'] += 1
    if damage == 'missing': campaign['comparison_references'].pop('1')
    with pytest.raises(ValueError):
        publication.load_comparison_references(campaign)
