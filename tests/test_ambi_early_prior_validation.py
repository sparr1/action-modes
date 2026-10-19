from copy import deepcopy

import pytest

from slurm.ambi_early_prior_baselines import validate_result
from utils.ambi_benchmark import solver_seed


def artifacts(mode="mean"):
    campaign = {"seeds": [101, 102, 103, 104, 105], "controller_seed": 55}
    cell = {"checkpoint_sha256": "pinned", "execution": mode}
    episodes = [{"seed": seed, "solver_seed": solver_seed(55, "episode", seed),
        "length": 500, "truncated": True, "truncated_by_evaluator": False,
        "model_metrics": {key: 0.0 for key in (
            "inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
            "inner_temperature_optimizer_steps", "inner_model_steps")}}
        for seed in campaign["seeds"]]
    row = {"selector": "early_prior/" + mode, "execution_action": mode,
        "resolved_config": {"inner_operator": "none"}, "outer_state_unchanged": True,
        "outer_updates_before": 123, "outer_updates_after": 123,
        "episodes": episodes, "return": {"mean": 2.0},
        "model_metrics": {key: {"max": 0.0} for key in episodes[0]["model_metrics"]}}
    result = {"checkpoint_sha256": "pinned", "results": [row]}
    manifest = {"status": "complete", "protocol": {"action_rule": "tanh_" + mode}}
    return result, manifest, cell, campaign


@pytest.mark.parametrize("mode", ["mean", "sample"])
def test_complete_matched_prior_is_accepted(mode):
    assert validate_result(*artifacts(mode))["passed"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(checkpoint_sha256="other"),
    lambda r: r["results"][0].update(execution_action="sample"),
    lambda r: r["results"][0]["episodes"].pop(),
    lambda r: r["results"][0]["episodes"][0].update(solver_seed=0),
    lambda r: r["results"][0]["episodes"][0].update(truncated_by_evaluator=True),
    lambda r: r["results"][0].update(outer_state_unchanged=False),
    lambda r: r["results"][0]["episodes"][0]["model_metrics"].update(inner_model_steps=1.0),
    lambda r: r["results"][0]["model_metrics"]["inner_temperature_optimizer_steps"].update(max=1.0),
])
def test_mismatched_or_adapting_reference_is_rejected(mutation):
    result, manifest, cell, campaign = deepcopy(artifacts())
    mutation(result)
    with pytest.raises(ValueError):
        validate_result(result, manifest, cell, campaign)
