"""The full-episode compute sweep changes only the number of inner rounds."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('rounds', [1, 2, 4])
def test_round_scaling_preserves_per_round_learner(rounds):
    reference = json.loads((ROOT / 'configs/research/ambi_prior_mean_prefix_h1_20seeds.json').read_text())
    value = json.loads((ROOT / f'configs/research/round_scaling_j{rounds}_h1_200k.json').read_text())
    assert value['shared_alg_params'] == {**reference['shared_alg_params'], 'inner_rounds': rounds}
    assert value['evaluation'] == reference['evaluation']
    assert value['checkpoint_contract'] == {**reference['checkpoint_contract'], 'checkpoints': [
        x for x in reference['checkpoint_contract']['checkpoints'] if x['step'] == 200000]}
    for group, comparison in reference['comparisons'].items():
        assert value['comparisons'][group]['reference'] == comparison['reference']
        for name, variant in comparison['variants'].items():
            assert value['comparisons'][group]['variants'][name]['alg_params'] == variant['alg_params']
    assert value['round_scaling']['controller_seeds'] == [55, 56, 57]
    assert value['round_scaling']['rounds'] == [1, 2, 4]
    assert 'real_calibration' not in value
    assert [s for shard in value['oscar_seed_shards'] for s in shard] == list(range(101, 121))
    assert len(value['oscar_seed_shards']) == rounds
