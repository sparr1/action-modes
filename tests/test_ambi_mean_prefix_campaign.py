"""Pin experiment 1's one-axis protocol and disjoint scheduler ownership."""
import json
from pathlib import Path

import pytest
from slurm import ambi_mean_prefix_campaign as launch

ROOT = Path(__file__).resolve().parents[1]


def test_mean_prefix_changes_only_measurement_coverage():
    old = json.loads((ROOT / 'configs/research/ambi_prior_refinement_h1_parallel.json').read_text())
    new = json.loads((ROOT / launch.MATRIX).read_text())
    assert new['shared_alg_params'] == old['shared_alg_params']
    assert new['comparisons'] == old['comparisons']
    assert new['evaluation']['seeds'] == list(range(101, 121))
    for key, value in old['evaluation'].items():
        if key != 'seeds':
            assert new['evaluation'][key] == value
    assert new['real_calibration'] == {**old['real_calibration'], 'prefix_action_rule': 'mean'}
    assert new['checkpoint_contract']['checkpoints'] == [
        r for r in old['checkpoint_contract']['checkpoints'] if r['step'] in [150000, 200000]]
    for key, value in old['checkpoint_contract'].items():
        if key != 'checkpoints':
            assert new['checkpoint_contract'][key] == value


def test_task_cells_cover_every_seed_exactly_once_per_mode_checkpoint():
    _, steps, _, shards = launch.campaign(ROOT / launch.MATRIX)
    coordinates = []
    for index in range(12):
        mode, checkpoint, shard = launch.cell(index, steps, shards)
        coordinates.extend((mode, steps[checkpoint], seed) for seed in shards[shard])
    assert len(coordinates) == len(set(coordinates)) == 80
    assert set(coordinates) == {(mode, step, seed) for mode in ['episodes', 'real']
                                for step in [150000, 200000] for seed in range(101, 121)}
    assert launch.cell(3, steps, shards) == ('episodes', 1, 0)
    assert launch.cell(9, steps, shards) == ('real', 1, 0)
    for index in [-1, 12, True, 1.5]:
        with pytest.raises(ValueError):
            launch.cell(index, steps, shards)


@pytest.mark.parametrize('mutation', ['missing', 'overlap', 'reorder', 'sampled'])
def test_campaign_rejects_invalid_shards_or_action_rule(tmp_path, mutation):
    config = json.loads((ROOT / launch.MATRIX).read_text())
    if mutation == 'missing':
        config['oscar_seed_shards'][0].pop()
    elif mutation == 'overlap':
        config['oscar_seed_shards'][0].append(108)
    elif mutation == 'reorder':
        config['oscar_seed_shards'].reverse()
    else:
        config['real_calibration']['prefix_action_rule'] = 'sampled'
    path = tmp_path / 'matrix.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        launch.campaign(path)


def test_production_cannot_finalize_partial_seed_panel(tmp_path):
    with pytest.raises(ValueError, match='every seed shard'):
        launch.finalize_checkpoint(tmp_path / 'inventory.json', tmp_path, 'attempt', 0,
                                   matrix=ROOT / launch.MATRIX, shard_indices=[0, 1])
    assert not (tmp_path / 'production').exists()


def test_workers_do_not_publish_or_stage_partial_results():
    import inspect
    body = inspect.getsource(launch.run_worker)
    assert 'eval-run-map' not in body and 'publish' not in body and 'eval_series.py' not in body
    assert 'seal-episodes' in body
    script = (ROOT / 'slurm/run_ambi_mean_prefix_oscar.sbatch').read_text()
    assert 'WANDB_MODE=offline' in script
    assert 'git status --porcelain' in script
    assert 'EXPECTED_ACTION_MODES_SHA' in script


@pytest.mark.parametrize('index,mode,seeds', [(3, 'episodes', [101]), (10, 'real', [108])])
def test_smoke_commands_preserve_learner_and_full_tail(tmp_path, monkeypatch, index, mode, seeds):
    calls = []
    monkeypatch.setattr(launch, '_run', lambda args, **kwargs: calls.append(list(map(str, args))))
    row = dict(step=200000, sha256='a' * 64)
    monkeypatch.setattr(launch, 'load_inventory', lambda *args: [row, row])
    monkeypatch.setattr(launch, 'verify_checkpoint', lambda row: tmp_path / 'checkpoint')
    launch.run_worker(tmp_path / 'inventory', tmp_path, 'attempt', index, matrix=ROOT / launch.MATRIX, smoke=True)
    target = next(c for c in calls if c[0] == ('evaluate_ambi_checkpoint.py' if mode == 'episodes' else 'evaluate_ambi_calibration.py'))
    assert target[target.index('--seeds') + 1] == str(seeds[0])
    assert '--eval-run-map' not in target
    if mode == 'real':
        assert target[target.index('--tail-steps') + 1] == '1000'
        assert target[target.index('--solver-repetitions') + 1] == '1'
        assert target[target.index('--rollout-repetitions') + 1] == '4'
    else:
        assert 'initialization/prior' in target and 'initialization/inherited' in target
        assert target[target.index('--max-steps') + 1] == '2'
        assert calls[-1][1] == 'seal-episodes'
