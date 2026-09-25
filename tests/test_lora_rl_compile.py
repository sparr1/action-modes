"""LoRA-RL preserves inner SAC under graph capture and explicit fallback."""

from copy import deepcopy
import warnings

import pytest
import torch

from RL.tdmpc2_core.common.layers import Ensemble
from RL.tdmpc2_core.common.lora import LoRARLLinear, make_lora_rl_critic
from tests.test_lora_rl import _head
from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_model


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _models(*, strict=False, **options):
    options = {"inner_critic_adaptation": "lora_rl", "inner_critic_lora_rank": 4,
               "inner_updates_per_round": 1, **options}
    builder = _tiny_model
    if "inner_critic_updates_per_round" in options:
        options.pop("inner_updates_per_round")
        builder = _tiny_component_model
    return (builder(**options),
            builder(**options, compile=True, compile_strict=strict))


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
@pytest.mark.parametrize("strict", [False, True])
def test_graph_capture_matches_complete_lora_sac_solves(monkeypatch, dropout, strict):
    real_compile = torch.compile
    captured_regions = []

    def eager_backend(function, **kwargs):
        captured_regions.append(function.__name__)
        return real_compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_backend)
    eager, compiled = _models(dropout=dropout, strict=strict)
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


@pytest.mark.parametrize("placement", ["input_hidden", "hidden"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("normed", [False, True])
def test_strict_detached_lora_graph_matches_stateless_values_gradients_and_rng(
    monkeypatch, placement, training, normed,
):
    real_compile = torch.compile

    def eager_backend(function, **kwargs):
        return real_compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_backend)
    def head():
        return (_head() if normed else torch.nn.Sequential(
            torch.nn.Linear(5, 7), torch.nn.Linear(7, 6), torch.nn.Linear(6, 3)))

    eager = make_lora_rl_critic(Ensemble([head(), head()]), rank=3,
                                scale=0.75, placement=placement).train(training)
    with torch.no_grad():
        for layer in eager.modules():
            if isinstance(layer, LoRARLLinear):
                layer.lora_B.normal_(std=0.1)
                layer.base.bias.add_(0.1)
                if normed:
                    layer.base.ln.weight.mul_(1.1)
    compiled = deepcopy(eager).enable_compile(strict=True)
    sample = torch.randn(4, 5)
    rng = torch.random.get_rng_state().clone()
    expected_input = sample.clone().requires_grad_(True)
    expected = eager.forward_detached(expected_input)
    expected.square().sum().backward()
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    actual_input = sample.clone().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = compiled.forward_detached(actual_input)
        actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert torch.count_nonzero(actual_input.grad)
    assert all(parameter.grad is None for parameter in compiled.parameters())
    assert not compiled.compile_failed
    assert not any("Falling back to eager" in str(item.message) for item in caught)


@pytest.mark.parametrize("horizon", [1, 3])
def test_strict_auxiliary_return_lora_solves_preserve_dense_actor_and_reset(monkeypatch, horizon):
    real_compile = torch.compile

    def eager_backend(function, **kwargs):
        assert kwargs.get("fullgraph") is True
        return real_compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_backend)
    eager, compiled = _models(
        strict=True, aux_return_mode="sac", inner_actor_source="sac",
        inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
        inner_sac_critic_target="reward_only", inner_terminal_entropy="none",
        inner_finite_horizon=True, inner_rounds=2, inner_rollouts_per_round=4,
        inner_rollout_horizon=horizon, inner_replay_capacity=24,
        inner_updates_per_round=None, inner_critic_updates_per_round=2,
        inner_actor_updates_per_round=1, q_representation="distributional",
        num_q=5, dropout=0.01, log_std_mapping="direct_clamp",
    )
    try:
        outer_state = deepcopy(compiled.agent.model.state_dict())
        for _ in range(2):
            expected = eager.agent.act(torch.zeros(3), collect_diagnostics=False)
            actual = compiled.agent.act(torch.zeros(3), collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            left, right = compiled.agent.inner_engine, eager.agent.inner_engine
            for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim",
                         "temperature_optim"):
                _assert_state_equal(getattr(left._action_pool, name).state_dict(),
                                    getattr(right._action_pool, name).state_dict())
            _assert_private_rng_equal(left, right)
            _assert_state_equal(compiled.agent.model.state_dict(), outer_state)
            assert left._critic_base is compiled.agent.model._aux_return_Qs
            assert left._horizon_critic is compiled.agent.model._aux_return_Qs
            assert not any(isinstance(layer, LoRARLLinear) for layer in left._action_pool.actor.modules())
            assert left.state.critic_steps == 4 and left.state.actor_steps == 2
            assert all(parameter.grad is None for parameter in compiled.agent.model.parameters())
            assert compiled.agent.last_inner_metrics["inner_compile_fallback"] == 0.0
    finally:
        eager.env.close()
        compiled.env.close()
