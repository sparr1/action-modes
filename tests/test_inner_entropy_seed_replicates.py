"""Additional seeds must retain the complete approved training recipe."""

import pytest

from slurm.inner_entropy_seed_replicates import validate_recipe


@pytest.mark.parametrize("index,seed", [(0, 56), (1, 57)])
def test_seed_replication_preserves_reference_recipe(index, seed):
    _, manifest, run = validate_recipe(index)
    assert run["seed"] == seed
    assert "seed" not in manifest["overrides_alg"]
    assert len(manifest["configs"]) == 2
    assert run["total_steps"] == 1_000_000
    assert run["checkpoint_every"] == 25_000
