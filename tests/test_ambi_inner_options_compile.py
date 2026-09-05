"""Real graph capture/code generation for the optional inner SAC targets."""

import pytest
import torch

from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_ambi_outer_replay import _episode
from RL.tdmpc2_core.common.layers import Ensemble, mlp


@pytest.mark.parametrize("dropout", [0., .2])
def test_compiled_detached_critic_preserves_input_gradients_and_dropout(dropout):
    torch._dynamo.reset()
    ensemble = Ensemble([mlp(3, [8, 8], 1, dropout=dropout) for _ in range(2)])
    x = torch.ones(4, 3, requires_grad=True)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        expected = ensemble.forward_detached(x)
        expected_grad, = torch.autograd.grad(expected.sum(), (x,))
        compiled = torch.compile(ensemble._forward_detached_eager, backend="eager", fullgraph=True)
        torch.manual_seed(19)
        actual = compiled(x)
        actual_grad, = torch.autograd.grad(actual.sum(), (x,))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in ensemble.parameters())
    torch._dynamo.reset()


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_inductor_horizon_targets_gradients_and_optimizer_match_eager(representation):
    torch._dynamo.reset()
    options = dict(
        inner_finite_horizon=True, q_representation=representation,
        mppi_terminal_q_reduction="mean_all", inner_q_target_reduction="min_all",
    )
    eager = _tiny_model(**options)
    compiled = _tiny_model(**options)
    try:
        engines = [model.agent.inner_engine for model in (eager, compiled)]
        for engine in engines:
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=True)
        generator = torch.Generator().manual_seed(11)
        z = torch.rand(4, eager.cfg.latent_dim, generator=generator)
        next_z = torch.rand(4, eager.cfg.latent_dim, generator=generator)
        action = torch.rand(4, eager.cfg.action_dim, generator=generator) - .5
        args = (
            z, action, torch.ones(4, 1), next_z,
            torch.tensor([[0.], [0.], [1.], [1.]]), torch.tensor(.2),
            torch.randn(4, eager.cfg.action_dim, generator=generator), None,
            torch.tensor([[0.], [1.], [0.], [1.]]),
            torch.randn(4, eager.cfg.action_dim, generator=generator),
        )
        outputs = [engines[0]._sac_critic_kernel(*args)]
        kernel = torch.compile(engines[1]._sac_critic_kernel, backend="inductor", fullgraph=True)
        outputs.append(kernel(*args))
        for actual, expected in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        for engine, output in zip(engines, outputs):
            engine.state.critic_optim.zero_grad(set_to_none=True)
            output[0].backward()
        for actual, expected in zip(engines[1].state.critic_params, engines[0].state.critic_params):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=3e-6)
        for engine in engines:
            engine.state.critic_optim.step()
            assert all(parameter.grad is None for parameter in engine.model.parameters())
        for actual, expected in zip(engines[1].state.critic_params, engines[0].state.critic_params):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-6)
    finally:
        eager.env.close()
        compiled.env.close()
        torch._dynamo.reset()


@pytest.mark.parametrize("mode", ["none", "shared_mixture", "separate_critics"])
def test_combined_options_strict_graphs_reuse_across_actions(monkeypatch, mode):
    torch._dynamo.reset()
    graphs = []
    real_compile = torch.compile
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def compile_counted(function, **kwargs):
        return real_compile(function, backend=backend, **kwargs)
    monkeypatch.setattr(torch, "compile", compile_counted)
    model = _tiny_model(
        compile=True, compile_strict=True, inner_finite_horizon=True,
        inner_steps_per_update=4, inner_updates_per_round=None,
        inner_rounds=1, inner_outer_replay_fraction=.5, inner_explorer_mode=mode,
    )
    try:
        model.buffer.add(_episode())
        global_rng = torch.random.get_rng_state().clone()
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        count = len(graphs)
        assert count > 0
        for _ in range(2):
            model.agent.act(torch.zeros(3), collect_diagnostics=False)
            assert len(graphs) == count
        assert model.agent.last_inner_metrics["inner_compile_fallback"] == 0
        assert model.agent.last_inner_metrics["inner_outer_replay_samples"] == 2
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        model.env.close()
        torch._dynamo.reset()


def test_disabled_options_preserve_actions_parameters_rng_and_replay():
    eager = _tiny_model(inner_updates_per_round=1)
    explicit = _tiny_model(
        inner_updates_per_round=1, inner_finite_horizon=False,
        inner_steps_per_update=None, inner_outer_replay_fraction=0.,
    )
    try:
        for _ in range(2):
            left = eager.agent.act(torch.zeros(3), collect_diagnostics=False)
            right = explicit.agent.act(torch.zeros(3), collect_diagnostics=False)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        left_engine, right_engine = eager.agent.inner_engine, explicit.agent.inner_engine
        for name in ("actor", "critic", "critic_target"):
            for left, right in zip(
                getattr(left_engine._action_pool, name).parameters(),
                getattr(right_engine._action_pool, name).parameters(),
            ):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        for name in left_engine.rng.STREAMS:
            torch.testing.assert_close(
                left_engine.rng.generator(name).get_state(),
                right_engine.rng.generator(name).get_state(), rtol=0, atol=0,
            )
        assert left_engine._action_pool.replay.horizon_end is None
        assert left_engine._action_pool.replay.training_state_dict()["version"] == 1
    finally:
        eager.env.close()
        explicit.env.close()
