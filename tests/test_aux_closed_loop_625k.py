"""Exercise real adaptation, independent probe RNG and immutable seed merging."""
import copy
import json
import math
from pathlib import Path

import pytest

from evaluate_ambi_checkpoint import evaluate_matrix
from slurm.ambi_aux_closed_loop import MATRIX, MATRICES, SELECTOR, checkpoint_alpha, validate_bundle
from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params
from utils.ambi_diagnostic_series import diagnostic_history, record_from_model_bundle
from utils.ambi_seed_shards import merge_episode_bundles, seal_episode_bundle
from utils.eval_series_data import _metrics


@pytest.fixture(scope='module', params=['zero', 'inherit_outer'])
def panel(tmp_path_factory, request):
    import torch
    root = tmp_path_factory.mktemp('aux-closed-loop')
    options = dict(aux_return_mode='sac', log_std_mapping='direct_clamp',
                   sac_actor_loss_scale_mode='none', inner_operator='none',
                   inner_rounds=0, inner_rollouts_per_round=0, inner_updates_per_round=0,
                   ent_coef='auto_0.2')
    model = _tiny_model(**options)
    with torch.no_grad():
        model.agent.log_ent_coef.fill_(math.log(.037))
    checkpoint = root / 'model_625000'
    model.agent.save(checkpoint)
    model.env.close()
    assert checkpoint_alpha(checkpoint) == pytest.approx(.037)
    expected_alpha = checkpoint_alpha(checkpoint) if request.param == 'inherit_outer' else 0.
    metadata = dict(schema_version=1, checkpoint=dict(kind='periodic', step=625000, episode=50, best_score=None, best_window=100),
                    trial_run_params=dict(alg='AMBITDMPC2/AMBITDMPC2', env='Pendulum-v1', seed=55,
                        device='cpu', total_steps=2000000, alg_params=_tiny_params(**options)),
                    experiment_params=dict(env_params=dict(max_episode_steps=3)))
    Path(str(checkpoint) + '.metadata.json').write_text(json.dumps(metadata))
    matrix = json.loads(MATRICES[request.param].read_text())
    assert matrix['evaluation']['togo_return_rollouts'] == 32
    assert matrix['evaluation']['default_presets'] == [SELECTOR]
    matrix['shared_alg_params']['compile'] = False
    matrix_path = root / 'matrix.json'
    matrix_path.write_text(json.dumps(matrix))
    prior_matrix = copy.deepcopy(matrix)
    prior_matrix['evaluation']['togo_return_rollouts'] = 0
    prior_path = root / 'no-probes.json'
    prior_path.write_text(json.dumps(prior_matrix))
    evaluate_matrix(prior_path, checkpoint, selectors=['critic/prior'], seeds=[101, 102], max_steps=3,
                    device='cpu', bundle_dir=root / 'prior')
    def evaluate(path, seeds, matrix_file=matrix_path):
        return evaluate_matrix(matrix_file, checkpoint, selectors=[SELECTOR], seeds=seeds, max_steps=3,
                               device='cpu', bundle_dir=path, reference_bundle=root / 'prior')
    evaluate(root / 'serial', [101, 102])
    evaluate(root / 'without-probes', [101, 102], prior_path)
    shards = []
    for seed in [101, 102]:
        shard = root / f'shard-{seed}'
        evaluate(shard, [seed])
        seal_episode_bundle(shard)
        shards.append(shard)
    return root, shards, expected_alpha


def test_probe_rng_and_parallel_seed_execution_preserve_actual_returns(panel):
    root, shards, alpha = panel
    serial, record = validate_bundle(root / 'serial', [101, 102], 3, expected_alpha=alpha)
    plain = json.loads((root / 'without-probes/manifest.json').read_text())
    merged_path = merge_episode_bundles(shards, root / 'merged', expected_seeds=[101, 102])
    merged, merged_record = validate_bundle(merged_path, [101, 102], 3, expected_alpha=alpha)
    for source in [plain, merged]:
        for expected, actual in zip(serial['runs'][0]['episodes'], source['runs'][0]['episodes']):
            for key in ['seed', 'solver_seed', 'return', 'length', 'paired_return_delta']:
                assert actual[key] == expected[key], key
    raw_rows = [row for shard in shards for row in record_from_model_bundle(
        shard, SELECTOR, 'source-check', bootstrap_resamples=1)['rows']]
    assert merged_record['rows'] == raw_rows
    # Independently executed serial/sharded runs have different wall timings.
    def without_probe_time(items):
        items = copy.deepcopy(items)
        for item in items:
            item['metrics'].pop('probe_seconds', None)
        return items
    assert without_probe_time(merged_record['rows']) == without_probe_time(record['rows'])
    assert without_probe_time(merged_record['summaries']) == without_probe_time(record['summaries'])
    history = diagnostic_history(merged_record)
    assert [r['diagnostic/actor_updates'] for r in history] == [0, 4]
    assert [r['diagnostic/critic_updates'] for r in history] == [0, 32]
    metrics = _metrics(merged['runs'][0]['episodes'])
    assert metrics['eval/paired_episodes'] == 2
    assert 'eval/paired_gain_mean' in metrics


def test_merge_rejects_missing_overlapping_seeds_and_corrupted_traces(panel, tmp_path):
    _, shards, _ = panel
    for sources, seeds in [(shards[:1], [101, 102]), (shards, [101, 102, 103])]:
        with pytest.raises(ValueError, match='missing or unexpected seeds'):
            merge_episode_bundles(sources, tmp_path / 'missing', expected_seeds=seeds)
    import shutil
    duplicate = tmp_path / 'duplicate'
    shutil.copytree(shards[0], duplicate)
    with pytest.raises(ValueError, match='overlap'):
        merge_episode_bundles([shards[0], duplicate], tmp_path / 'overlap', expected_seeds=[101, 102])
    trace = next(duplicate.glob('*/*.jsonl.gz'))
    trace.write_bytes(trace.read_bytes() + b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        merge_episode_bundles([duplicate, shards[1]], tmp_path / 'corrupt', expected_seeds=[101, 102])
    assert not (tmp_path / 'corrupt').exists()


def test_inherited_alpha_matrix_changes_only_entropy_enablement():
    zero = json.loads(MATRIX.read_text())
    inherited = json.loads(MATRICES['inherit_outer'].read_text())
    assert inherited['evaluation'] == zero['evaluation']
    for variant in zero['comparisons']['critic']['variants']:
        assert inherited['comparisons']['critic']['variants'][variant]['alg_params'] == zero['comparisons']['critic']['variants'][variant]['alg_params']
    expected = {**zero['shared_alg_params'], 'inner_entropy_enabled': True}
    assert inherited['shared_alg_params'] == expected


def test_publication_recovery_uses_saved_panel_and_json_paths(tmp_path, monkeypatch):
    from argparse import Namespace
    from slurm.ambi_aux_closed_loop import SEEDS, publish
    import utils.ambi_benchmark as benchmark
    import utils.ambi_diagnostic_series as diagnostics
    import utils.eval_series as series
    output = tmp_path / 'production'
    output.mkdir()
    receipt = dict(status='complete', checkpoint_step=625000, seeds=SEEDS,
                   diagnostic_series_id='existing-diagnostic')
    (output / 'merge-completion.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(diagnostics, 'read_diagnostic_bundle', lambda _: dict(
        status='complete', series_id='existing-diagnostic', rows=[{}] * 5000))
    calls = []
    def stage(bundle, run_map, **kwargs):
        # Exercise the failing JSON serialization of the real helper receipt.
        json.dumps(run_map)
        calls.append((bundle, run_map))
        return {SELECTOR: dict(status='queued')}
    monkeypatch.setattr(benchmark, 'stage_completed_bundle', stage)
    monkeypatch.setattr(series, 'publish_run', lambda *a, **k: dict(status='complete'))
    monkeypatch.setattr(diagnostics, 'publish_diagnostic_bundle', lambda *a, **k: dict(status='complete'))
    publish(Namespace(root=tmp_path, run_dir=tmp_path / 'registry', inventory=tmp_path / 'inventory.json'))
    assert calls == [(output / 'bundle', {SELECTOR: str(tmp_path / 'registry')})]
    assert json.loads((output / 'publication-completion.json').read_text())['diagnostic_publication']['status'] == 'complete'
