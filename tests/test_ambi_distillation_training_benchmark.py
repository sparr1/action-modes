"""Checks for the bounded timing harness and its benchmark-only graph adapter."""

from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.benchmark_ambi_distillation_training import CollectionGraph, recipe
from tests.test_ambi_teacher_graph import CUDA, _agent, _assert_close, _snapshot


@pytest.mark.parametrize("interval", [0, 10])
def test_training_timing_recipe_keeps_compute_and_logging_contract(interval):
    params = recipe(SimpleNamespace(refresh_interval=interval, device="cuda", tiny=False))
    assert params["outer_actor_objective"] == "distillation"
    assert params["inner_rounds"] == 8
    assert params["inner_rollouts_per_round"] == 128
    assert params["inner_critic_updates_per_round"] == 16
    assert params["inner_actor_updates_per_round"] == 4
    assert params["reanalyze_batch_size"] == 20
    assert params["reanalyze_parallel_roots"] == 4
    assert params["reanalyze_interval"] == interval
    assert params["wandb"] is False
    assert params["compile_strict"] is True


@CUDA
def test_collection_graph_matches_uncaptured_actions_teachers_and_current_priors():
    agent = _agent(scaled=True, compiled=True)
    root = torch.zeros(1, agent.cfg.latent_dim, device="cuda")
    control = CollectionGraph(InnerImprovementEngine(agent), root, capture=False)
    graph = CollectionGraph(InnerImprovementEngine(agent), root)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    compile_count = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])

    def copy(runner, output):
        action, metrics, lengths, teacher = output
        with torch.cuda.stream(runner.stream):
            copied = (action.clone(), {key: value.clone() for key, value in teacher.items()})
        torch.cuda.current_stream().wait_stream(runner.stream)
        return copied

    for iteration, seed in enumerate((41, 53, 41)):
        with torch.no_grad():
            for parameter in agent.model.parameters():
                parameter.add_(0.0001)
            agent.actor_loss_scale.fill_(2.0 + iteration)
            agent.fixed_ent_coef.mul_(1.1)
        expected = copy(control, control(root + iteration * .1, seed=seed))
        actual = copy(graph, graph(root + iteration * .1, seed=seed))
        torch.cuda.synchronize()
        _assert_close(actual, expected)
        _assert_close(_snapshot(graph.engine), _snapshot(control.engine))
        assert torch.isfinite(actual[0]).all()
        assert all(parameter.grad is None for parameter in agent.model.parameters())
    assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == compile_count
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
