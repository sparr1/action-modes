"""Actual frozen evaluation and publication identity for sampled final actions."""
import json

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_benchmark_evaluation import checkpoint_matrix
from utils.ambi_benchmark import protocol_for, read_json
from utils.eval_series_data import planner_identity, descriptive_label


def test_mean_protocol_and_planner_identity_remain_backward_compatible():
    config = {"inner_operator": "sac", "inner_rounds": 2}
    resolved = {"environment": {"id": "Pendulum-v1"},
                "algorithm_config": {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": config}}
    before = protocol_for(resolved, 55, 3)
    planner = planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", before["action_rule"])
    config["inner_eval_execution_action"] = "mean"
    assert protocol_for(resolved, 55, 3) == before
    assert planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean") == planner
    config["inner_eval_execution_action"] = "policy_sample"
    sampled = protocol_for(resolved, 55, 3)
    assert sampled == {**before, "action_rule": "squashed_gaussian_sample"}
    sampled_planner = planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", sampled["action_rule"])
    assert sampled_planner != planner
    assert "sampled actions" in descriptive_label({"backbone": "entity/project/prior",
        "science": {"algorithm": "AMBITDMPC2/AMBITDMPC2"}, "planner": sampled_planner})
    # Evaluation draws use unit Gaussian scale; train-time controls stay inactive.
    changed = {**config, "inner_execution_action": "mean_plus_gaussian",
               "inner_execution_std_scale": 8, "inner_execution_noise_std": 9}
    assert planner_identity(changed, {}, "AMBITDMPC2/AMBITDMPC2", sampled["action_rule"]) == sampled_planner


@pytest.mark.parametrize("algorithm,operator", [("AMBITDMPC2/AMBITDMPC2", "mppi"),
    ("AMBITDMPC2/AMBITDMPC2", "td3"), ("TDAMBI/TDAMBI", "tdambi")])
def test_sampled_protocol_rejects_unsupported_controller(algorithm, operator):
    resolved = {"environment": {"id": "Pendulum-v1"}, "algorithm_config": {
        "alg": algorithm, "alg_params": {"inner_operator": operator,
        "inner_eval_execution_action": "policy_sample"}}}
    with pytest.raises(ValueError, match="requires AMBI SAC or prior-only"):
        protocol_for(resolved, 55, 3)


def test_sampled_episode_evaluation_is_frozen_seed_reproducible_and_distinct(checkpoint_matrix, tmp_path):
    checkpoint, matrix_path = checkpoint_matrix
    baseline = evaluator.evaluate_matrix(matrix_path, checkpoint, selectors=["budget/sac"],
                                         bundle_dir=tmp_path / "mean")
    matrix = read_json(matrix_path)
    matrix["shared_alg_params"]["inner_eval_execution_action"] = "policy_sample"
    matrix_path.write_text(json.dumps(matrix))
    outcomes = []
    for index, seeds in enumerate(([101, 102], [102, 101])):
        result = evaluator.evaluate_matrix(matrix_path, checkpoint, selectors=["budget/sac"],
                     seeds=seeds, bundle_dir=tmp_path / f"sampled-{index}")
        assert result["frozen_outer_learning"] and not result["deterministic_execution"]
        run, = result["results"]
        assert run["outer_state_unchanged"] and run["outer_updates_before"] == run["outer_updates_after"]
        assert run["action_rule"] == "squashed_gaussian_sample"
        assert not run["deterministic_execution"]
        manifest = read_json(tmp_path / f"sampled-{index}" / "manifest.json")
        assert manifest["protocol"]["action_rule"] == "squashed_gaussian_sample"
        outcomes.append({ep["seed"]: ep["return"] for ep in run["episodes"]})
    assert outcomes[0] == outcomes[1]
    assert outcomes[0] != {ep["seed"]: ep["return"] for ep in baseline["results"][0]["episodes"]}


def test_mean_prior_is_not_silently_relabelled_as_sampled(checkpoint_matrix, tmp_path, monkeypatch):
    checkpoint, matrix_path = checkpoint_matrix
    prior = tmp_path / "prior"
    evaluator.evaluate_matrix(matrix_path, checkpoint, bundle_dir=prior)
    matrix = read_json(matrix_path)
    matrix["shared_alg_params"]["inner_eval_execution_action"] = "policy_sample"
    matrix_path.write_text(json.dumps(matrix))
    monkeypatch.setattr(evaluator, "_make_env", lambda *_: pytest.fail("constructed model before reference rejection"))
    with pytest.raises(ValueError, match="protocol does not match"):
        evaluator.evaluate_matrix(matrix_path, checkpoint, selectors=["budget/sac"],
                                  bundle_dir=tmp_path / "sampled", reference_bundle=prior)
