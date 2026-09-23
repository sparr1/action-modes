"""Sampled campaign scope, explicit reference contract and paired publication."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_sampled as campaign
from utils.ambi_benchmark import solver_seed


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_requested_scope_changes_only_final_execution(horizon):
    matrix = campaign.ROOT / f'configs/research/ambi_closed_loop_sampled_h{horizon}_575k.json'
    selected = campaign.cells(matrix)
    assert [(c['H'], c['J'], c['critic_kind']) for c in selected] == [
        (horizon, 4, 'return_only'), (horizon, 2, 'return_only'), (horizon, 1, 'return_only')]
    mean_matrix = campaign.ROOT / ('configs/research/ambi_closed_loop_critics_575k.json' if horizon == 3
        else f'configs/research/ambi_closed_loop_critics_h{horizon}_575k.json')
    old = {c['name']: c for c in campaign.mean_cells(mean_matrix)}
    for cell in selected:
        original = old[cell['mean_name']]
        assert cell['params'] == {**original['params'], campaign.EXECUTION_KEY: 'policy_sample'}
        assert cell['requested_alg_params'] == {
            **original['requested_alg_params'], campaign.EXECUTION_KEY: 'policy_sample'}
        assert cell['params']['inner_execution_action'] == 'mean'
        assert cell['params']['inner_replay_capacity'] == 3072


def test_every_horizon_has_distinct_pinned_mean_references():
    assert set(campaign.MEAN_SOURCE_COMMITS) == {1, 2, 3}
    assert len(set(campaign.MEAN_SOURCE_COMMITS.values())) == 3
    for values in (campaign.MEAN_RUN_IDS_BY_H, campaign.MEAN_MANIFEST_SHA_BY_H):
        assert set(values) == {1, 2, 3}
        assert all(set(points) == {1, 2, 4} for points in values.values())
        assert len({value for points in values.values() for value in points.values()}) == 9
    assert campaign.MEAN_RUN_IDS_BY_H[3] == campaign.MEAN_RUN_IDS
    assert campaign.MEAN_MANIFEST_SHA_BY_H[3] == campaign.MEAN_MANIFEST_SHA


def test_campaign_refuses_to_mix_horizons():
    from slurm.ambi_closed_loop_sampled_publish import aggregate_results
    one = campaign.cells(campaign.ROOT / 'configs/research/ambi_closed_loop_sampled_h1_575k.json')
    two = campaign.cells(campaign.ROOT / 'configs/research/ambi_closed_loop_sampled_h2_575k.json')
    with pytest.raises(AssertionError, match='one common horizon'):
        aggregate_results({'cells': [*one, *two]}, {})


@pytest.mark.parametrize('key,value', [('inner_rounds', 8), ('inner_actor_lr', .01),
    ('inner_entropy_enabled', False), ('inner_replay_capacity', 1024)])
def test_matrix_rejects_unrequested_scientific_changes(tmp_path, key, value):
    matrix = campaign.read(campaign.MATRIX)
    matrix['shared_alg_params'][key] = value
    path = tmp_path / 'matrix.json'
    path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError):
        campaign.cells(path)


def test_cross_action_protocol_allows_only_execution_and_smoke_duration():
    mean = {'environment': {'id': 'DMControl-v0', 'params': {'task': 'humanoid-walk'}},
            'action_rule': 'tanh_mean', 'controller_seed': 55, 'max_steps': 500,
            'seed_scheme': 'sha256-v1'}
    sampled = {**mean, 'action_rule': campaign.ACTION_RULE}
    campaign.matching_protocol(sampled, mean)
    campaign.matching_protocol({**sampled, 'max_steps': 3}, mean, steps=3)
    for key, value in [('controller_seed', 99), ('max_steps', 400),
                       ('action_rule', 'tanh_mean'), ('seed_scheme', 'different')]:
        with pytest.raises(AssertionError):
            campaign.matching_protocol({**sampled, key: value}, mean)


def test_cross_action_configuration_rejects_any_other_change():
    mean = deepcopy(campaign.cells()[0]['params'])
    mean.pop(campaign.EXECUTION_KEY)
    sampled = {**mean, campaign.EXECUTION_KEY: 'policy_sample'}
    campaign.matching_config(sampled, mean)
    campaign.matching_config(sampled, {**mean, campaign.EXECUTION_KEY: 'mean'})
    with pytest.raises(AssertionError):
        campaign.matching_config({**sampled, 'inner_actor_lr': .01}, mean)
    with pytest.raises(AssertionError):
        campaign.matching_config(sampled, sampled)


def episodes(offset=0):
    return [{'seed': s, 'solver_seed': solver_seed(55, 'episode', s),
             'return': float(s - 100 + offset), 'length': 500, 'truncated_by_evaluator': False}
            for s in campaign.SEEDS]


def test_paired_comparison_matches_seeds_not_array_order():
    from slurm.ambi_closed_loop_sampled_publish import paired_comparison
    result = paired_comparison(list(reversed(episodes(2))), episodes())
    assert [r['seed'] for r in result['rows']] == campaign.SEEDS
    assert all(r['sample_minus_mean'] == 2 for r in result['rows'])
    stats = result['metrics']
    assert stats['comparison/sample_minus_mean_mean'] == 2
    assert stats['comparison/sample_minus_mean_std'] == 0
    assert stats['comparison/sample_minus_mean_ci95_low'] == 2
    assert stats['comparison/sample_minus_mean_ci95_high'] == 2
    assert stats['comparison/sample_minus_mean_paired_episodes'] == 5
    changed = episodes(2)
    changed[0]['solver_seed'] += 1
    with pytest.raises(ValueError, match='solver seeds differ'):
        paired_comparison(changed, episodes())
    for invalid in (episodes(2)[:-1], episodes(2) + episodes(2)[:1]):
        with pytest.raises(ValueError):
            paired_comparison(invalid, episodes())


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_aggregate_keeps_missing_samples_absent_and_uses_real_j_axis(horizon):
    from slurm.ambi_closed_loop_sampled_publish import aggregate_results, numeric_rows, chart_payloads
    panel = campaign.cells(campaign.ROOT / f'configs/research/ambi_closed_loop_sampled_h{horizon}_575k.json')
    for c in panel:
        c['mean_reference'] = {'episodes': episodes(c['J'])}
    state = {'cells': panel}
    pending = aggregate_results(state, {})
    assert pending['points'] == [] and pending['episodes'] == []
    assert len(pending['means']) == 15
    assert [row['axis/inner_rounds'] for _, row in numeric_rows(pending)] == [1, 2, 4]
    assert 'comparison/sample_minus_mean_vs_J' not in chart_payloads(pending)
    j2 = next(c for c in panel if c['J'] == 2)
    aggregate = aggregate_results(state, {j2['name']: episodes(4)})
    assert [p['J'] for p in aggregate['points']] == [2]
    assert aggregate['points'][0]['difference']['mean'] == 2
    rows = dict(numeric_rows(aggregate))
    assert set(rows) == {'mean_J1', 'mean_J2', 'mean_J4', 'sampled_J2'}
    assert rows['sampled_J2']['sampled_execution/sample_minus_mean/mean'] == 2
    assert rows['sampled_J2']['axis/inner_rounds'] == 2
    chart = chart_payloads(aggregate)['comparison/return_vs_J']
    assert f'H{horizon}' in chart['title']
    assert chart['xs'] == [[1, 2, 4], [2]]
    assert chart['keys'] == ['Historical mean execution', 'Sampled execution']
    with pytest.raises(ValueError, match='Unexpected completed'):
        aggregate_results(state, {'unrequested': episodes()})
    with pytest.raises(ValueError, match='exactly one'):
        aggregate_results({'cells': [*panel, panel[0]]}, {})


def test_execution_proof_exposes_observed_sampling_metrics():
    from slurm.ambi_closed_loop_sampled_publish import execution_proof
    result = execution_proof({'runs': [{'result': {'model_metrics': {
        'inner_eval_execution_sampled': dict(mean=1, min=1, max=1),
        'inner_eval_execution_mean_action_l2': dict(mean=.2, min=0, max=.4)}}}]})
    assert result['execution/sample_flag_mean'] == 1
    assert result['execution/mean_action_l2_min'] == 0
    assert result['execution/mean_action_l2_max'] == .4


def test_immutable_receipt_requires_complete_trace_hash_coverage(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'trace.jsonl').write_text('row\n')
    (bundle / 'manifest.json').write_text(json.dumps({'runs': [{'trace_files': ['trace.jsonl']}]}))
    receipt = {'status': 'complete', 'manifest_sha256': campaign.digest(bundle / 'manifest.json'),
               'trace_sha256': {'trace.jsonl': campaign.digest(bundle / 'trace.jsonl')}}
    campaign.verify_receipt(bundle, receipt)
    with pytest.raises(AssertionError):
        campaign.verify_receipt(bundle, {**receipt, 'trace_sha256': {}})
    with pytest.raises(AssertionError):
        campaign.verify_receipt(bundle, {**receipt, 'status': 'running'})
    (bundle / 'trace.jsonl').write_text('modified\n')
    with pytest.raises(AssertionError):
        campaign.verify_receipt(bundle, receipt)


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_watcher_publishes_samples_when_cell_uploads_already_exist(tmp_path, monkeypatch, horizon):
    from slurm import ambi_closed_loop_sampled_publish as publication
    panel = campaign.cells(campaign.ROOT / f'configs/research/ambi_closed_loop_sampled_h{horizon}_575k.json')
    for cell in panel:
        directory = tmp_path / cell['name']
        directory.mkdir()
        (directory / 'worker-completion.json').write_text('{}')
        cell.update(directory=str(directory), performance_run_id='sample' + str(cell['J']),
                    training_run_id='diagnostic' + str(cell['J']), mean_reference={
                    'episodes': episodes(cell['J']), 'performance_run_id': 'mean' + str(cell['J']),
                    'manifest_sha256': 'pinned'})
    state = dict(cells=panel, checkpoint_step=575000, checkpoint_sha256=campaign.CHECKPOINT_SHA,
                 source_run=campaign.SOURCE_RUN, source_commit='tested', initial_alpha=campaign.INITIAL_ALPHA,
                 target_entropy=-10.5, mean_source_commit='verified-mean', H=horizon,
                 group='test', comparison='sample minus mean', overview_run_id='overview', label='Test')
    (tmp_path / 'campaign.json').write_text(json.dumps(state))
    logged = []
    run = SimpleNamespace(summary={}, log=lambda row: logged.append(deepcopy(row)),
                          define_metric=lambda *a, **k: None, finish=lambda **k: None)
    init_config = {}
    def initialize(**kwargs):
        init_config.update(kwargs['config'])
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=initialize))
    monkeypatch.setattr(publication, 'verify_mean_reference', lambda *a, **k: None)
    monkeypatch.setattr(publication, 'publication_complete', lambda cell: True)
    monkeypatch.setattr(publication, 'load_completed', lambda state, cell: (episodes(cell['J'] + 2),
                         {'execution/sample_flag_mean': 1, 'execution/mean_action_l2_mean': .2}))
    monkeypatch.setattr(publication, 'overview_log', lambda *a: {})
    publication.watch(SimpleNamespace(root=tmp_path))
    samples = [row for row in logged if 'sampled_execution/sample/return_mean' in row]
    assert [row['axis/inner_rounds'] for row in samples] == [1, 2, 4]
    assert all(row['sampled_execution/sample_minus_mean/mean'] == 2 for row in samples)
    assert run.summary['status'] == 'complete' and run.summary['evaluated'] == 3
    assert len(campaign.read(tmp_path / 'comparison-results.json')['points']) == 3
    assert init_config['H'] == horizon
