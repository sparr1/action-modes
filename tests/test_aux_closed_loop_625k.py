"""Exercise real adaptation, independent probe RNG and immutable seed merging."""
import copy
import json
from pathlib import Path

import pytest

from evaluate_ambi_checkpoint import evaluate_matrix
from slurm.ambi_aux_closed_loop import MATRIX, SELECTOR, validate_bundle
from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params
from utils.ambi_diagnostic_series import diagnostic_history, record_from_model_bundle
from utils.ambi_seed_shards import merge_episode_bundles, seal_episode_bundle
from utils.eval_series_data import _metrics


@pytest.fixture(scope='module')
def panel(tmp_path_factory):
    root = tmp_path_factory.mktemp('aux-closed-loop')
    options = dict(aux_return_mode='sac', log_std_mapping='direct_clamp',
                   sac_actor_loss_scale_mode='none', inner_operator='none',
                   inner_rounds=0, inner_rollouts_per_round=0, inner_updates_per_round=0)
    model = _tiny_model(**options)
    checkpoint = root / 'model_625000'
    model.agent.save(checkpoint)
    model.env.close()
    metadata = dict(schema_version=1, checkpoint=dict(kind='periodic', step=625000, episode=50, best_score=None, best_window=100),
                    trial_run_params=dict(alg='AMBITDMPC2/AMBITDMPC2', env='Pendulum-v1', seed=55,
                        device='cpu', total_steps=2000000, alg_params=_tiny_params(**options)),
                    experiment_params=dict(env_params=dict(max_episode_steps=3)))
    Path(str(checkpoint) + '.metadata.json').write_text(json.dumps(metadata))
    matrix = json.loads(MATRIX.read_text())
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
    return root, shards


def test_probe_rng_and_parallel_seed_execution_preserve_actual_returns(panel):
    root, shards = panel
    serial, record = validate_bundle(root / 'serial', [101, 102], 3)
    plain = json.loads((root / 'without-probes/manifest.json').read_text())
    merged_path = merge_episode_bundles(shards, root / 'merged', expected_seeds=[101, 102])
    merged, merged_record = validate_bundle(merged_path, [101, 102], 3)
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
    _, shards = panel
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
