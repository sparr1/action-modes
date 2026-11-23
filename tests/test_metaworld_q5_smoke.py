"""The training throughput gate must distinguish intentional native actions."""

import pytest

from slurm.metaworld_campaign import read_json
from slurm.smoke_metaworld_mrsq_q5 import recipe, recipe_paths, validate_graph_usage


def row(step, used, *, evaluation=False):
    return {"step": step, "evaluation": evaluation,
            "metrics": {"inner_cuda_graph_used": used}}


def test_graph_gate_accepts_native_evaluation_and_diagnostics():
    counts = validate_graph_usage([
        row(1, 0, evaluation=True), row(1000, 0), row(1102, 1),
    ], 1000, 1102, 1201)
    assert counts == {"training_graph": 1, "training_native_diagnostic": 1,
                      "evaluation_native": 1, "measured_training_graph": 1}


@pytest.mark.parametrize("record", [
    row(1102, 0), row(1, 1, evaluation=True), row(1000, 1),
])
def test_graph_gate_rejects_unexpected_execution_path(record):
    with pytest.raises(AssertionError, match="Unexpected graph usage"):
        validate_graph_usage([record], 1000, 1102, 1201)


def test_native_diagnostic_is_not_a_valid_timing_sample():
    with pytest.raises(AssertionError, match="Timed training must use"):
        validate_graph_usage([row(2000, 0)], 1000, 1901, 2000)


@pytest.mark.parametrize("variant,num_q", [("mrsq_q5", 5), ("mrsq_q10", 10)])
def test_smoke_loads_the_matching_pinned_recipe_without_enabling_wandb(tmp_path, variant, num_q):
    path, manifest_path = recipe_paths(variant)
    run, manifest, overrides = recipe(tmp_path, variant)
    expected = read_json(path)
    assert run["alg_params"] == {**expected["alg_params"], **overrides["alg_params"]}
    assert run["alg_params"]["num_q"] == run["alg_params"]["q_pair_size"] == num_q
    assert not run["alg_params"]["wandb"] and run["alg_params"]["wandb_mode"] == "disabled"
    assert run["alg_params"]["seed_steps"] == run["alg_params"]["pretrain_steps"] == 1000
    assert manifest["env_params"] == read_json(manifest_path)["env_params"]
    assert manifest["configs"] == [f"ambi_state_mrsq_q{num_q}"]
    assert run["total_steps"] == 1201
