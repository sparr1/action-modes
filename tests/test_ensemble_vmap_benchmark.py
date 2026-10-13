"""Numerical and ownership checks for the opt-in benchmark head backend."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from RL.tdmpc2_core import pave
from RL.tdmpc2_core.common.layers import Ensemble, NormedLinear, mlp
from RL.tdmpc2_core.common.q_representation import QRepresentation
from utils.ensemble_vmap_benchmark import (
    vectorized_ensemble_benchmark, vectorized_forward,
)


def _critic(representation="scalar", dropout=0.0):
    torch.manual_seed(719)
    heads = 2 if representation == "scalar" else 5
    decoder = QRepresentation(representation, num_q=heads, num_bins=7, vmin=-2, vmax=2)
    critic = Ensemble([mlp(5, [8, 8], decoder.output_dim, dropout=dropout)
                       for _ in range(heads)]).double()
    return critic, decoder


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("shape", [(5,), (6, 5), (3, 6, 5)])
def test_forward_and_input_parameter_gradients_match(representation, shape):
    critic, decoder = _critic(representation)
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    expected = critic(x)
    expected_grads = torch.autograd.grad(decoder.decode(expected).square().sum(),
                                        (x, *critic.parameters()))
    actual = vectorized_forward(critic, x)
    actual_grads = torch.autograd.grad(decoder.decode(actual).square().sum(),
                                      (x, *critic.parameters()))
    _close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        _close(actual_grad, expected_grad)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_independent_action_gradients_hvp_and_pave_parameter_gradients(representation):
    critic, decoder = _critic(representation)
    z = torch.randn(4, 3, dtype=torch.float64)
    action = torch.randn(len(critic), 4, 2, dtype=torch.float64, requires_grad=True)
    vectors = torch.randint(0, 2, action.shape).double() * 2 - 1

    def derivatives(vectorized):
        inputs = torch.cat((z.expand(len(critic), -1, -1), action), dim=-1)
        prediction = (vectorized_forward(critic, inputs, head_inputs=True) if vectorized
                      else torch.stack([head(inputs[i]) for i, head in enumerate(critic)]))
        values = decoder.decode(prediction)
        gradient = pave._action_gradient(values, action)
        hvp = pave._action_gradient((gradient * vectors).sum(), action)
        return values, gradient, hvp

    for actual, expected in zip(derivatives(True), derivatives(False)):
        _close(actual, expected)

    inputs = dict(z=z, action=action[0].detach(), noisy_z=z + .1,
                  next_z=z.flip(0), probe=vectors[0], mpr_coef=.1,
                  vfc_coef=.1, curvature_coef=.01, return_components=True)
    expected, expected_metrics, _ = pave.critic_pave_loss(critic, decoder.decode, **inputs)
    expected_grads = torch.autograd.grad(expected, tuple(critic.parameters()), allow_unused=True)
    with vectorized_ensemble_benchmark():
        actual, actual_metrics, _ = pave.critic_pave_loss(critic, decoder.decode, **inputs)
        actual_grads = torch.autograd.grad(actual, tuple(critic.parameters()), allow_unused=True)
    _close(actual, expected)
    for name in expected_metrics:
        _close(actual_metrics[name], expected_metrics[name])
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert (actual_grad is None) == (expected_grad is None)
        if expected_grad is not None:
            _close(actual_grad, expected_grad)
            assert torch.isfinite(actual_grad).all()


def test_opposing_heads_remain_independent_and_affine_curvature_is_zero():
    head = mlp(5, [8, 8], 1).double()
    other = deepcopy(head)
    with torch.no_grad():
        other[-1].weight.neg_()
        other[-1].bias.neg_()
    critic = Ensemble([head, other])
    z, action = torch.randn(4, 3).double(), torch.randn(4, 2).double()
    with vectorized_ensemble_benchmark():
        loss, metrics = pave.critic_pave_loss(
            critic, lambda x: x, z, action, noisy_z=z + 1, mpr_coef=1.,
        )
        assert loss > 0
        _close(critic(torch.cat((z, action), dim=-1)).sum(0), torch.zeros(4, 1).double())
        affine = Ensemble([nn.Sequential(nn.Linear(5, 1)).double() for _ in range(2)])
        value, stats = pave.critic_pave_loss(
            affine, lambda x: x, z, action, probe=torch.ones_like(action),
            curvature_coef=1., curvature_margin=1.,
        )
        _close(value, torch.tensor(2.).double())
        _close(stats["curvature_violation_fraction"], torch.tensor(1.))


def test_detached_forward_preserves_input_gradient_and_optimizer_parameter_ownership():
    critic, _ = _critic()
    parameters = tuple(critic.parameters())
    identities = tuple(id(p) for p in parameters)
    state = deepcopy(critic.state_dict())
    optimizer = torch.optim.Adam(parameters, lr=.01)
    x = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    expected = critic.forward_detached(x)
    expected_gradient = torch.autograd.grad(expected.square().sum(), x)[0]
    with vectorized_ensemble_benchmark() as metadata:
        actual = critic.forward_detached(x)
        gradients = torch.autograd.grad(actual.square().sum(), (x, *parameters), allow_unused=True)
        _close(actual, expected)
        _close(gradients[0], expected_gradient)
        assert all(g is None for g in gradients[1:])
        assert tuple(id(p) for p in critic.parameters()) == identities
        assert tuple(critic.state_dict()) == tuple(state)
        for name, value in critic.state_dict().items():
            _close(value, state[name])
        critic(x).square().mean().backward()
        optimizer.step()
        assert len(optimizer.state) == len(parameters)
        assert all(p.grad is not None for p in parameters)
        assert metadata["persistent_stacked_parameters"] is False
    assert any(not torch.equal(value, state[name]) for name, value in critic.state_dict().items())


def test_dropout_independence_geometry_modes_and_exception_restoration():
    critic, decoder = _critic(dropout=.5)
    critic[1].load_state_dict(critic[0].state_dict())
    x = torch.randn(128, 5).double()
    with vectorized_ensemble_benchmark():
        torch.manual_seed(31)
        output = critic(x)
        assert not torch.equal(output[0], output[1])
        torch.manual_seed(31)
        _close(critic(x), output)
        critic[0][0].dropout.eval()
        with pytest.raises(ValueError, match="homogeneous"):
            critic(x)
        modes = [(module, module.training) for module in critic.modules()]
        rng = torch.get_rng_state().clone()
        loss, _ = pave.critic_pave_loss(critic, decoder.decode, x[:, :3], x[:, 3:],
                                      next_z=x[:, :3] + .1, vfc_coef=.1)
        assert torch.isfinite(loss)
        assert all(module.training == mode for module, mode in modes)
        assert torch.equal(torch.get_rng_state(), rng)
    methods = (Ensemble._forward_eager, Ensemble._forward_detached_eager, pave._head_predictions)
    with pytest.raises(RuntimeError, match="forced"):
        with vectorized_ensemble_benchmark():
            raise RuntimeError("forced")
    assert methods == (Ensemble._forward_eager, Ensemble._forward_detached_eager,
                       pave._head_predictions)


def test_rejects_unsupported_layers_and_mismatched_head_inputs():
    bad = Ensemble([nn.Sequential(nn.Linear(5, 1), nn.Tanh()) for _ in range(2)])
    with pytest.raises(TypeError, match="Linear/NormedLinear"):
        vectorized_forward(bad, torch.zeros(4, 5))
    critic, _ = _critic()
    with pytest.raises(ValueError, match="leading row"):
        vectorized_forward(critic, torch.zeros(3, 4, 5).double(), head_inputs=True)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_complete_two_updates_compile_ordinary_with_real_aot_and_eager_pave(monkeypatch, representation):
    from benchmark_pave import _ResidentBatch
    from tests.test_ambi_root_local_sac import _tiny_model
    from tests.test_pave_integration import _batch, _nonconstant_critics, _weights

    actual_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda function, **options:
                        actual_compile(function, backend="aot_eager", **options))
    torch._dynamo.reset()
    wrapper = None
    try:
        with vectorized_ensemble_benchmark():
            wrapper = _tiny_model(compile=True, compile_strict=True,
                                  aux_return_mode="sac", q_representation=representation,
                                  num_q=5 if representation == "distributional" else 2,
                                  **_weights("outer_pave", "aux_return_pave"))
            agent = wrapper.agent
            _nonconstant_critics(agent)
            for _ in range(2):
                info = agent.update(_ResidentBatch(_batch(agent)))
                assert all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values())
            assert agent.num_updates == 2
            assert agent._outer_update_region._compiled is not None
            assert not agent._outer_update_region.failed
            for attr in ("_Qs", "_aux_return_Qs", "_target_Qs", "_target_aux_return_Qs"):
                assert not getattr(agent.model, attr).compile_failed
    finally:
        if wrapper is not None:
            wrapper.env.close()
        torch._dynamo.reset()


@pytest.mark.parametrize("detached", [False, True])
def test_aot_forward_and_parameter_input_gradient_parity(detached):
    # Importing the real agent stack also loads TensorDict's vmap monkeypatch.
    from tests.test_ambi_root_local_sac import _tiny_model  # noqa: F401

    torch._dynamo.reset()
    try:
        critic, _ = _critic("distributional")
        x = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
        parameters = tuple(critic.parameters())
        expected = critic.forward_detached(x) if detached else critic(x)
        expected_grads = torch.autograd.grad(expected.square().sum(), (x, *parameters),
                                            allow_unused=True)
        with vectorized_ensemble_benchmark():
            fn = critic._forward_detached_eager if detached else critic._forward_eager
            compiled = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=False)
            actual = compiled(x)
            actual_grads = torch.autograd.grad(actual.square().sum(), (x, *parameters),
                                              allow_unused=True)
        _close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert (actual_grad is None) == (expected_grad is None)
            if expected_grad is not None:
                _close(actual_grad, expected_grad)
    finally:
        torch._dynamo.reset()
