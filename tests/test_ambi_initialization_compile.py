"""Scratch initialization stays eager while compiled SAC consumes fresh weights."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_inner_initialization import _assert_equal, _model, _snapshot


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_compiled_scratch_solves_match_eager_across_roots_and_evaluation_resets(
    monkeypatch, adaptation,
):
    original_compile = torch.compile
    regions, graphs = [], []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def compile_eager(function, **kwargs):
        regions.append(function.__name__)
        return original_compile(function, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_eager)
    options = {"inner_critic_adaptation": adaptation, "inner_updates_per_round": 2,
               "dropout": 0.2}
    eager, compiled = _model(**options), _model(**options, compile=True, compile_strict=False)
    try:
        outer_before = deepcopy(compiled.agent.model.state_dict())
        global_rng = torch.random.get_rng_state().clone()
        for index in range(3):
            if index:
                for model in (eager, compiled):
                    model.agent.inner_engine.reset_for_evaluation(900 + index, reuse_action_pool=True)
            expected = eager.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
            actual = compiled.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_equal(_snapshot(compiled.agent.inner_engine), _snapshot(eager.agent.inner_engine))
            _assert_equal(compiled.agent.model.state_dict(), outer_before)
            assert all(parameter.grad is None for parameter in compiled.agent.model.parameters())
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
            assert compiled.agent.last_inner_metrics["inner_compile_fallback"] == 0.0
            if index == 0:
                initial_graphs = len(graphs)
                assert initial_graphs > 0
            else:
                assert len(graphs) == initial_graphs
        assert {"_dense_rollout_kernel", "_sac_critic_kernel", "_sac_actor_kernel"} <= set(regions)
    finally:
        eager.env.close()
        compiled.env.close()
