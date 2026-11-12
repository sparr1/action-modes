"""Seed replication retains the complete task-specific scientific protocol."""

import pytest
from slurm.posttanh_humanoid_seed_replicates import validate_recipe


@pytest.mark.parametrize("task", ["run", "stand"])
@pytest.mark.parametrize("index,seed", [(0, 56), (1, 57)])
def test_replication_retains_reference(task, index, seed):
    _, manifest, run = validate_recipe(task, index)
    assert run["seed"] == seed
    assert "seed" not in manifest["overrides_alg"]
    assert manifest["env_params"]["task"] == f"humanoid-{task}"
    assert len(manifest["configs"]) == 2
    assert run["total_steps"] == 1_000_000
    assert run["checkpoint_every"] == 25_000 and run["save_strat"] == "all"
    params = run["alg_params"]
    assert params["inner_execution_action"] == "policy_sample"
    assert params["inner_rounds"] == 8 and params["inner_solve_interval"] == 1
    assert params["inner_actor_loss_scale_scope"] == "action"
    assert params["inner_actor_loss_scale_update"] == "per_update"
    assert params["critic_geometry_diagnostics"]
    assert params["critic_geometry_every"] == 1000
    assert params["critic_geometry_batch_size"] == 64
    assert params["critic_geometry_obs_noise_std"] == .01
    assert params["critic_geometry_action_noise_std"] == .02
    assert params["compile_strict"] and params["inner_cuda_graph"]
