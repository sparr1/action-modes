from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm import warm_actor_calibration_publish as publication
from utils import wandb_results_layout as layout


def record(**overrides):
    return dict(source_cell='warm_h3_j10', H=3, J=10, seed=101, decision=75,
                branch_kind='prefix', actor_family='warm', round=0,
                action_mode='sample', replicate=0, predicted_model_return=12.,
                real_bootstrapped_return=10., real_mc_return=8., **overrides)


def row(**overrides):
    value = record()
    value.update(overrides)
    return publication.normalize_record(value, {})


def write_shard(root, rows, *, path='prefix/warm_h3_j10/seed-101/decision-75.json', status='complete'):
    destination = root/path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(schema_version=1, status=status, records=rows)))
    return destination


def campaign(root):
    value = dict(expected_prefix_shards=1, expected_replan_shards=1,
                 cells=[dict(name='warm_h3_j10', H=3, J=10)])
    (root/'campaign.json').write_text(json.dumps(value))
    return value


def test_error_decomposition_and_paired_gains_do_not_mix_modes_roots_or_seeds():
    records = [row(), row(actor_family='prior', real_mc_return=6.),
               row(round=4, predicted_model_return=19., real_bootstrapped_return=16., real_mc_return=13.),
               row(action_mode='mean', real_mc_return=100.),
               row(round=4, action_mode='mean', real_mc_return=103.),
               row(seed=102, round=4, real_mc_return=999.),
               row(decision=350, round=4, real_mc_return=777.)]
    paired = publication.derive_paired_metrics(records)
    candidate = paired[2]
    assert candidate['modeled_prefix_error'] == 3
    assert candidate['terminal_value_error'] == 3
    assert candidate['total_value_error'] == 6
    assert candidate['real_mc_return_gain_vs_inherited'] == 5
    assert candidate['real_mc_return_gain_vs_prior'] == 7
    assert paired[4]['real_mc_return_gain_vs_inherited'] == 3
    assert paired[4]['real_mc_return_mean_minus_sample'] == 90
    assert 'real_mc_return_gain_vs_inherited' not in paired[5]
    assert 'real_mc_return_gain_vs_inherited' not in paired[6]
    assert 'real_mc_return_gain_vs_prior' not in paired[4]


def test_seed_balance_is_equal_after_replicate_then_root_averaging():
    # Seed 101 has two roots and more replicates; it must not dominate seed 102.
    values = [(101, 25, 0.)]*10 + [(101, 75, 20.)] + [(102, 25, 30.)]
    summary = publication.seed_balanced(values)
    assert summary['mean'] == 20
    assert summary['seed_means'] == {101: 10, 102: 30}
    assert summary['n_roots'] == 3 and summary['n_branches'] == 12
    assert summary['ci95_low'] == 10 and summary['ci95_high'] == 30
    assert publication.seed_balanced([(101, 25, 0.)])['ci95_low'] is None


def test_terminal_reduction_calibration_keeps_same_real_tail():
    candidate = row(predicted_model_return_expected_min_pair=9.,
                    real_bootstrapped_return_expected_min_pair=7.)
    assert candidate['modeled_prefix_error_expected_min_pair'] == 2
    assert candidate['terminal_value_error_expected_min_pair'] == -1
    assert candidate['total_value_error_expected_min_pair'] == 1
    assert candidate['terminal_value_abs_error_expected_min_pair'] == 1
    result = publication.aggregate([candidate])
    assert next(r for r in result if r['metric'] == 'terminal_value_error_expected_min_pair')['mean'] == -1


def test_complete_shard_requires_all_declared_candidate_modes_and_replicates():
    first = row()
    task = {name: first[name] for name in ('source_cell', 'H', 'J', 'seed', 'decision')}
    manifest = dict(prefixes=[task], rollouts=2)
    with pytest.raises(ValueError, match='branch coverage'):
        publication.validate_shard_scope([first], manifest)
    rows = [row(actor_family=family, round=stage, action_mode=mode, replicate=replicate)
            for family, stage in [(family, stage) for family in ('warm', 'cold')
                                  for stage in (0, 1, 2, 4, 6, 8, 10)] + [('prior', 0)]
            for mode in ('sample', 'mean') for replicate in range(2)]
    publication.validate_shard_scope(rows, manifest)


def test_aggregate_preserves_replan_unit_boundary():
    rows = [row(), row(round=4, real_mc_return=13.),
            row(branch_kind='replan', action_mode='mean', real_mc_return=100.),
            row(branch_kind='replan', action_mode='mean', round=4, real_mc_return=150.)]
    result = publication.aggregate(rows)
    gain = [r for r in result if r['metric'] == 'real_mc_return_gain_vs_inherited' and r['round'] == 4]
    assert {(r['branch_kind'], r['mean']) for r in gain} == {('prefix', 5), ('replan', 50)}


def test_period_aggregation_keeps_boundaries_pairing_and_seed_balance():
    rows = []
    for decision, gain in ((99, 1.), (100, 2.), (299, 4.), (300, 7.)):
        rows.extend([row(decision=decision), row(decision=decision, round=4, real_mc_return=8.+gain)])
    periods = publication.aggregate_periods(rows)
    values = {name: next(r for r in measurements if r['metric'] == 'real_mc_return_gain_vs_inherited'
                        and r['round'] == 4) for name, measurements in periods.items()}
    assert {name: r['mean'] for name, r in values.items()} == {'early': 1., 'middle': 3., 'late': 7.}
    assert values['middle']['n_roots'] == 2 and values['middle']['n_seeds'] == 1
    assert values['middle']['ci95_low'] is None


def test_loader_rejects_duplicate_branches_nonfinite_returns_and_identity_mismatch(tmp_path):
    manifest = campaign(tmp_path)
    write_shard(tmp_path, [record(), record()])
    with pytest.raises(ValueError, match='Duplicate branch'):
        publication.load_shards(tmp_path, manifest)
    with pytest.raises(ValueError, match='finite'):
        publication.normalize_record({**record(), 'real_mc_return': float('nan')}, {})
    with pytest.raises(ValueError, match='disagree'):
        publication.normalize_record(record(), {'seed': 102})


def test_pending_report_has_no_zero_measurements_and_complete_counts_are_separate(tmp_path):
    campaign(tmp_path)
    write_shard(tmp_path, [record()], status='running')
    report, files = publication.build_report(tmp_path, render=False)
    assert report['status'] == 'pending' and report['measurements'] == [] and files == {}
    assert [r['complete'] for r in report['progress']] == [0, 0]
    write_shard(tmp_path, [record()])
    report, _ = publication.build_report(tmp_path, render=False)
    assert report['status'] == 'running'
    assert [r['complete'] for r in report['progress']] == [1, 0]
    assert report['source_shards'][0]['sha256'] and report['record_count'] == 1
    assert (tmp_path/'report/index.html').exists()


def test_new_layout_preserves_actor_transfer_sections_and_workspace_filters():
    spec = {'section': {'runSets': [{'filters': {'preserve': True}}],
                        'panelBankConfig': {'sections': layout.actor_transfer_sections()}}}
    before = deepcopy(spec)
    sections = publication.calibration_sections()
    patched = layout.patch_results_spec(spec, sections)
    assert spec == before
    assert layout.patch_results_spec(patched, sections) == patched
    assert layout._without_owned(patched, {s['__id__'] for s in sections}) == before
    assert layout._installed(patched, sections)
    panels = [p for section in sections for p in section['panels']]
    media_keys = {p['config']['mediaKeys'][0] for p in panels if p['viewType'] == 'Media Browser'}
    assert media_keys == set(publication.TABLE_KEYS)
    charts = [p for p in panels if p['viewType'] == 'Vega2']
    assert len(charts) == 4
    assert {p['config']['userQuery']['queryFields'][0]['fields'][0]['args'][0]['value'] for p in charts} == {
        name+'_table' for name in publication.IMAGE_KEYS}
    assert all(section['isOpen'] for section in sections)


def test_layout_failure_visible_without_suppressing_scientific_publication(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise layout.ResultsLayoutError('test failure')
    monkeypatch.setattr(publication, 'ensure_results_layout', fail)
    run = SimpleNamespace(summary={})
    manifest = dict(wandb_entity='entity', wandb_project='project', overview_run_id='exact-id')
    result = publication.install_layout(SimpleNamespace(Api=lambda **kwargs: object()), run, manifest, tmp_path)
    assert result['status'] == run.summary['results_layout/status'] == 'failed'
    assert run.summary['results_layout/schema_verified'] is False
    assert (tmp_path/'results-layout-failure.json').exists()


def test_layout_pins_exact_published_run_and_new_owned_ids(tmp_path, monkeypatch):
    def installed(*args, **kwargs):
        assert kwargs['run_id'] == 'exact-id'
        assert kwargs['layout_version'] == publication.LAYOUT_VERSION
        assert kwargs['sections'] == publication.calibration_sections()
        return dict(status='verified', url='https://wandb.ai/entity/project/runs/exact-id?nw=nwuserrwgao_b')
    monkeypatch.setattr(publication, 'ensure_results_layout', installed)
    run = SimpleNamespace(summary={})
    manifest = dict(wandb_entity='entity', wandb_project='project', overview_run_id='exact-id')
    result = publication.install_layout(SimpleNamespace(Api=lambda **kwargs: object()), run, manifest, tmp_path)
    assert result['url'] == run.summary['results_layout/url']
    assert run.summary['results_layout/schema_verified'] is True


def test_payload_contains_explicit_progress_and_four_native_charts_with_empty_results():
    fake = SimpleNamespace(Table=lambda **kwargs: kwargs,
                           plot=SimpleNamespace(line_series=lambda **kwargs: kwargs))
    report = dict(progress=[dict(branch_kind='prefix', complete=0, expected=120, status='pending')],
                  measurements=[], record_count=0, cells=[])
    files = {name: name+'.png' for name in publication.IMAGE_KEYS}
    values = publication.payload(fake, report, files)
    assert values['calibration/progress']['data'] == [['prefix', 0, 120, 'pending']]
    assert values['calibration/measurements']['data'] == []
    assert set(files) <= values.keys()
    assert values['calibration/gains']['ys'] == [[]]
