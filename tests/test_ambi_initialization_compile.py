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


def _assert_vmap_floating_state_close(actual, expected, *, key=None):
    """Allow float32 kernel rounding while keeping optimizer counters exact."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            _assert_vmap_floating_state_close(actual[name], expected[name], key=name)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_vmap_floating_state_close(actual_value, expected_value, key=key)
    elif torch.is_tensor(expected) and expected.is_floating_point() and key != "step":
        torch.testing.assert_close(actual, expected, rtol=5e-7, atol=1e-10)
    else:
        _assert_equal(actual, expected)


@pytest.mark.parametrize("ensemble_backend", ["loop", "vmap"])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_compiled_scratch_solves_match_eager_across_roots_and_evaluation_resets(
    monkeypatch, adaptation, ensemble_backend,
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
               "dropout": 0.2, "critic_ensemble_backend": ensemble_backend}
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
            actual_state = _snapshot(compiled.agent.inner_engine)
            expected_state = _snapshot(eager.agent.inner_engine)
            if ensemble_backend == "loop":
                _assert_equal(actual_state, expected_state)
            else:
                # Vectorized eager/graph execution can round float32 gradients
                # differently by a few ULPs. RNG, replay and all update counters
                # retain the exact lifecycle contract of the loop backend.
                for name in ("rng", "replay"):
                    _assert_equal(actual_state[name], expected_state[name])
                _assert_vmap_floating_state_close(actual_state, expected_state)
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
