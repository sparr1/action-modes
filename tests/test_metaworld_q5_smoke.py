"""The training throughput gate must distinguish intentional native actions."""

from types import SimpleNamespace

import pytest

from slurm.metaworld_campaign import read_json
from slurm.smoke_metaworld_mrsq_q5 import recipe, recipe_paths, validate_graph_usage, validate_trained_controller


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


@pytest.mark.parametrize("num_q", [5, 10])
def test_trained_resume_check_checks_graph_before_native_evaluation(monkeypatch, num_q):
    from slurm import inner_entropy_oscar
    from tests import test_inner_entropy_graph

    checked = []
    agent = SimpleNamespace(captured=True)
    buffer = object()

    def continuation(selected_agent):
        assert selected_agent is agent and agent.captured
        checked.append("continuation")

    def geometry(selected_agent, selected_buffer):
        assert selected_agent is agent and selected_buffer is buffer
        checked.append("geometry")
        return {"passed": True}

    def evaluation(step):
        assert step == 1201
        checked.append("evaluation")
        agent.captured = False

    monkeypatch.setattr(test_inner_entropy_graph, "assert_graph_resume_continuation", continuation)
    monkeypatch.setattr(inner_entropy_oscar, "check_geometry", geometry)
    learner = SimpleNamespace(agent=agent, buffer=buffer, _global_step=1201, _evaluate_policy=evaluation)
    checks = validate_trained_controller(learner, num_q)
    if num_q == 10:
        assert checked == ["continuation", "geometry", "evaluation"]
        assert checks == {"graph_native_graph_continuation": True, "critic_geometry": {"passed": True}}
    else:
        assert checked == ["evaluation"] and checks == {}


@pytest.mark.parametrize("variant,num_q,task", [
    ("mrsq_q5", 5, "assembly"),
    ("mrsq_q10", 10, "assembly"),
    ("mrsq_q10", 10, "push"),
    ("mrsq_q10", 10, "stick-pull"),
])
def test_smoke_loads_the_matching_pinned_recipe_without_enabling_wandb(tmp_path, variant, num_q, task):
    path, manifest_path = recipe_paths(variant, task)
    run, manifest, overrides = recipe(tmp_path, variant, task)
    expected = read_json(path)
    assert run["alg_params"] == {**expected["alg_params"], **overrides["alg_params"]}
    assert run["alg_params"]["num_q"] == run["alg_params"]["q_pair_size"] == num_q
    assert not run["alg_params"]["wandb"] and run["alg_params"]["wandb_mode"] == "disabled"
    assert run["alg_params"]["seed_steps"] == run["alg_params"]["pretrain_steps"] == 1000
    assert manifest["env_params"] == read_json(manifest_path)["env_params"]
    assert manifest["env_params"]["task"] == task
    assert manifest["configs"] == [f"ambi_state_mrsq_q{num_q}"]
    assert run["total_steps"] == 1201


def test_smoke_rejects_unversioned_q5_task():
    with pytest.raises(ValueError, match="Unsupported MetaWorld smoke recipe"):
        recipe_paths("mrsq_q5", "push")
