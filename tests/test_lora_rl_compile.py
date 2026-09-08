"""LoRA-RL preserves inner SAC under graph capture and explicit fallback."""

from copy import deepcopy
import warnings

import pytest
import torch

from RL.tdmpc2_core.common.lora import LoRARLLinear
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _models(**options):
    options = {"inner_critic_adaptation": "lora_rl", "inner_critic_lora_rank": 4,
               "inner_updates_per_round": 1, **options}
    return (_tiny_model(**options),
            _tiny_model(**options, compile=True, compile_strict=False))


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_state_equal(left, right)
    else:
        assert actual == expected


def _assert_private_rng_equal(left, right):
    _assert_state_equal(left.rng.training_state_dict(), right.rng.training_state_dict())


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_non_strict_graph_capture_matches_complete_lora_sac_solves(monkeypatch, dropout):
    real_compile = torch.compile
    captured_regions = []

    def eager_backend(function, **kwargs):
        captured_regions.append(function.__name__)
        return real_compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_backend)
    eager, compiled = _models(dropout=dropout)
    try:
        eager_engine, compiled_engine = eager.agent.inner_engine, compiled.agent.inner_engine
        outer_state = deepcopy(eager.agent.model.state_dict())
        global_rng = torch.random.get_rng_state().clone()
        graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        for _ in range(3):
            expected = eager.agent.act(torch.zeros(3), collect_diagnostics=False)
            actual = compiled.agent.act(torch.zeros(3), collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name in ("actor", "critic", "critic_target"):
                left = getattr(compiled_engine._action_pool, name)
                right = getattr(eager_engine._action_pool, name)
                _assert_state_equal(left.state_dict(), right.state_dict())
            for name in ("actor_optim", "critic_optim", "temperature_optim"):
                left = getattr(compiled_engine._action_pool, name)
                right = getattr(eager_engine._action_pool, name)
                _assert_state_equal(left.state_dict(), right.state_dict())
            _assert_private_rng_equal(compiled_engine, eager_engine)
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
            _assert_state_equal(compiled.agent.model.state_dict(), outer_state)
            assert all(parameter.grad is None for parameter in compiled.agent.model.parameters())
            assert compiled.agent.last_inner_metrics["inner_compile_fallback"] == 0.0
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > graphs_before
        assert {"_dense_rollout_kernel", "_sac_critic_kernel", "_sac_actor_kernel"} <= set(captured_regions)
        assert any(torch.count_nonzero(layer.lora_B) for layer in
                   compiled_engine._action_pool.critic.modules() if isinstance(layer, LoRARLLinear))
    finally:
        eager.env.close()
        compiled.env.close()


def test_detached_lora_failure_warns_once_reports_fallback_and_restores_rng(monkeypatch):
    eager, compiled = _models(dropout=0.2)
    try:
        engines = [model.agent.inner_engine for model in (eager, compiled)]
        for engine in engines:
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=True)
            with torch.no_grad():
                for layer in engine.state.critic.modules():
                    if isinstance(layer, LoRARLLinear):
                        layer.lora_B.fill_(0.01)
        compile_calls = []

        def unsupported_detached_compile(function, **kwargs):
            assert function.__name__ == "_forward_detached_eager"
            compile_calls.append(kwargs)

            def fail_after_dropout(*args, **call_kwargs):
                function(*args, **call_kwargs)
                raise RuntimeError("unsupported detached LoRA backend")

            return fail_after_dropout

        monkeypatch.setattr(torch, "compile", unsupported_detached_compile)
        global_rng = torch.random.get_rng_state().clone()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                outputs, gradients = [], []
                for engine in engines:
                    value = torch.ones(4, engine.cfg.latent_dim + engine.cfg.action_dim,
                                       requires_grad=True)
                    with engine.rng.fork("gradient_policy"):
                        output = engine.state.critic.forward_detached(value)
                    gradient, = torch.autograd.grad(output.sum(), (value,))
                    outputs.append(output)
                    gradients.append(gradient)
                    assert all(parameter.grad is None for parameter in engine.state.critic.parameters())
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
                _assert_private_rng_equal(engines[0], engines[1])
                torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert len(compile_calls) == 1
        assert compile_calls[0]["fullgraph"] is False
        fallbacks = [item for item in caught if "Falling back to eager detached critic ensemble" in str(item.message)]
        assert len(fallbacks) == 1
        metrics = engines[1]._compile_fallback_metrics()
        assert metrics["inner_compile_critic_fallback"] == 1.0
        assert metrics["inner_compile_fallback"] == 1.0
        assert not engines[1].state.critic._compile_enabled
    finally:
        eager.env.close()
        compiled.env.close()


def test_locked_runtime_strict_detached_lora_graph_fails_without_eager_fallback(monkeypatch):
    real_compile = torch.compile

    def eager_backend(function, **kwargs):
        return real_compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_backend)
    model = _tiny_model(inner_critic_adaptation="lora_rl", inner_critic_lora_rank=4)
    try:
        engine = model.agent.inner_engine
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        critic = engine.state.critic
        critic.enable_compile(strict=True)
        value = torch.ones(4, model.cfg.latent_dim + model.cfg.action_dim, requires_grad=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The locked PyTorch cannot fullgraph the custom critic's stateless
            # functional_call path. A strict request must expose that error.
            with pytest.raises(torch._dynamo.exc.Unsupported):
                critic.forward_detached(value)
        assert not critic.compile_failed
        assert not any("Falling back to eager" in str(item.message) for item in caught)
        assert engine._compile_fallback_metrics()["inner_compile_fallback"] == 0.0
    finally:
        model.env.close()
