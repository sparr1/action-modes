"""Dense vectorized critic execution without changing parameter ownership."""

from copy import deepcopy
from collections import OrderedDict
import pickle

import pytest
import torch
from torch import nn

from RL.tdmpc2_core import pave
from RL.tdmpc2_core.common import layers
from RL.tdmpc2_core.common.layers import Ensemble, mlp
from RL.tdmpc2_core.common.q_representation import QRepresentation


def _pair(representation="scalar", dropout=0.):
    torch.manual_seed(719)
    heads = 2 if representation == "scalar" else 5
    decoder = QRepresentation(representation, num_q=heads, num_bins=7, vmin=-2, vmax=2)
    loop = Ensemble([mlp(5, [8, 8], decoder.output_dim, dropout=dropout)
                     for _ in range(heads)]).double()
    vectorized = Ensemble(deepcopy(loop.modules_list), backend="vmap")
    return loop, vectorized, decoder


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("shape", [(5,), (6, 5), (3, 6, 5)])
def test_forward_and_original_parameter_input_gradient_parity(representation, shape):
    loop, vectorized, decoder = _pair(representation)
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    expected, actual = loop(x), vectorized(x)
    expected_grads = torch.autograd.grad(decoder.decode(expected).square().sum(),
                                        (x, *loop.parameters()))
    actual_grads = torch.autograd.grad(decoder.decode(actual).square().sum(),
                                      (x, *vectorized.parameters()))
    _close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        _close(actual_grad, expected_grad)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_independent_action_gradient_hvp_and_pave_parameter_gradient_parity(representation):
    loop, vectorized, decoder = _pair(representation)
    z = torch.randn(4, 3, dtype=torch.float64)
    action = torch.randn(len(loop), 4, 2, dtype=torch.float64, requires_grad=True)
    vectors = torch.randint(0, 2, action.shape).double() * 2 - 1

    def derivatives(critic):
        inputs = torch.cat((z.expand(len(critic), -1, -1), action), dim=-1)
        values = decoder.decode(critic.forward_heads_eager(inputs))
        gradient = pave._action_gradient(values, action)
        hvp = pave._action_gradient((gradient * vectors).sum(), action)
        return values, gradient, hvp

    for actual, expected in zip(derivatives(vectorized), derivatives(loop)):
        _close(actual, expected)
    inputs = dict(z=z, action=action[0].detach(), noisy_z=z + .1, next_z=z.flip(0),
                  probe=vectors[0], mpr_coef=.1, vfc_coef=.1, curvature_coef=.01)
    expected, expected_metrics = pave.critic_pave_loss(loop, decoder.decode, **inputs)
    actual, actual_metrics = pave.critic_pave_loss(vectorized, decoder.decode, **inputs)
    expected_grads = torch.autograd.grad(expected, tuple(loop.parameters()), allow_unused=True)
    actual_grads = torch.autograd.grad(actual, tuple(vectorized.parameters()), allow_unused=True)
    _close(actual, expected)
    for name in expected_metrics:
        _close(actual_metrics[name], expected_metrics[name])
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert (actual_grad is None) == (expected_grad is None)
        if expected_grad is not None:
            _close(actual_grad, expected_grad)
            assert torch.isfinite(actual_grad).all()


def test_opposing_heads_and_affine_zero_curvature():
    _, critic, _ = _pair()
    critic[1].load_state_dict(critic[0].state_dict())
    with torch.no_grad():
        critic[1][-1].weight.neg_()
        critic[1][-1].bias.neg_()
    z, action = torch.randn(4, 3).double(), torch.randn(4, 2).double()
    loss, _ = pave.critic_pave_loss(critic, lambda x: x, z, action,
                                  noisy_z=z + 1, mpr_coef=1.)
    assert loss > 0
    _close(critic(torch.cat((z, action), dim=-1)).sum(0), torch.zeros(4, 1).double())
    affine = Ensemble([nn.Sequential(nn.Linear(5, 1)).double() for _ in range(2)], backend="vmap")
    value, stats = pave.critic_pave_loss(affine, lambda x: x, z, action,
                                       probe=torch.ones_like(action), curvature_coef=1.)
    _close(value, torch.tensor(2.).double())
    assert stats["curvature_violation_fraction"] == 1.


def test_detached_actor_gradients_parameter_identity_state_keys_and_optimizer():
    loop, critic, _ = _pair()
    parameters = tuple(critic.parameters())
    identities = tuple(id(p) for p in parameters)
    state = deepcopy(critic.state_dict())
    assert tuple(state) == tuple(loop.state_dict())
    optimizer = torch.optim.Adam(parameters, lr=.01)
    x = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    expected, actual = loop.forward_detached(x), critic.forward_detached(x)
    expected_gradient = torch.autograd.grad(expected.square().sum(), x)[0]
    gradients = torch.autograd.grad(actual.square().sum(), (x, *parameters), allow_unused=True)
    _close(actual, expected)
    _close(gradients[0], expected_gradient)
    assert all(g is None for g in gradients[1:])
    critic(x).square().mean().backward()
    optimizer.step()
    assert len(optimizer.state) == len(parameters)
    assert tuple(id(p) for p in critic.parameters()) == identities
    assert all(p.grad is not None for p in parameters)
    assert any(not torch.equal(value, state[name]) for name, value in critic.state_dict().items())
    loop.load_state_dict(critic.state_dict())
    _close(loop(x), critic(x))


def test_dropout_independence_geometry_rng_and_mixed_mode_restoration(monkeypatch):
    _, critic, decoder = _pair(dropout=.5)
    critic[1].load_state_dict(critic[0].state_dict())
    x = torch.randn(128, 5).double()
    torch.manual_seed(31)
    output = critic(x)
    assert not torch.equal(output[0], output[1])
    torch.manual_seed(31)
    _close(critic(x), output)
    critic[0][0].dropout.eval()
    modes = [(module, module.training) for module in critic.modules()]
    rng = torch.get_rng_state().clone()
    loss, _ = pave.critic_pave_loss(critic, decoder.decode, x[:, :3], x[:, 3:],
                                  next_z=x[:, :3] + .1, vfc_coef=.1)
    assert torch.isfinite(loss)
    assert all(module.training == mode for module, mode in modes)
    assert torch.equal(torch.get_rng_state(), rng)

    def failure(*args, **kwargs):
        raise RuntimeError("forced geometry failure")

    monkeypatch.setattr(critic, "forward_heads_eager", failure)
    with pytest.raises(RuntimeError, match="forced geometry"):
        pave.critic_pave_loss(critic, decoder.decode, x[:, :3], x[:, 3:],
                              next_z=x[:, :3] + .1, vfc_coef=.1)
    assert all(module.training == mode for module, mode in modes)


@pytest.mark.parametrize("reason", ["custom", "lora", "mixed_modes", "hook", "forward"])
def test_explicit_unsupported_path_keeps_module_loop(monkeypatch, reason):
    _, critic, _ = _pair(dropout=.2)
    if reason == "custom":
        critic.modules_list[0].append(nn.Tanh())
    elif reason == "lora":
        from RL.tdmpc2_core.common.lora import make_lora_rl_critic
        critic = make_lora_rl_critic(critic, rank=2)
    elif reason == "mixed_modes":
        critic[0][0].dropout.eval()
    elif reason == "hook":
        critic[0][0].register_forward_hook(lambda module, args, value: value + .1)
    else:
        original = critic[0].forward
        critic[0].forward = lambda value: original(value) + .1

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported architecture entered vectorized path")

    monkeypatch.setattr(layers, "vectorized_forward", forbidden)
    x = torch.randn(4, 5).double()
    torch.manual_seed(13)
    expected = torch.stack([head(x) for head in critic])
    torch.manual_seed(13)
    actual = critic(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    critic.forward_detached(x.requires_grad_()).sum().backward()
    assert x.grad is not None


def test_backend_validation_pickle_deepcopy_and_legacy_state():
    loop, critic, _ = _pair()
    assert loop.backend == "loop" and critic.backend == "vmap"
    assert deepcopy(critic).backend == "vmap"
    assert pickle.loads(pickle.dumps(critic)).backend == "vmap"
    state = critic.__getstate__()
    del state["_backend"]
    restored = object.__new__(Ensemble)
    restored.__setstate__(state)
    assert restored.backend == "loop"
    with pytest.raises(ValueError, match="backend"):
        Ensemble([nn.Linear(2, 1)], backend="bad")
    with pytest.raises(ValueError, match="leading row"):
        critic.forward_heads_eager(torch.zeros(3, 4, 5).double())


def test_named_sequential_and_global_hooks_use_module_loop(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Custom named/hooked heads entered vectorized path")

    monkeypatch.setattr(layers, "vectorized_forward", forbidden)
    named = Ensemble([nn.Sequential(OrderedDict([("projection", nn.Linear(5, 1))]))
                      for _ in range(2)], backend="vmap").double()
    x = torch.randn(4, 5).double()
    _close(named(x), torch.stack([head(x) for head in named]))
    _, critic, _ = _pair()
    seen = []
    handle = nn.modules.module.register_module_forward_hook(
        lambda module, args, output: seen.append(module),
    )
    try:
        actual = critic(x)
        assert critic[0] in seen and critic[0][0] in seen
        _close(actual, torch.stack([head(x) for head in critic]))
    finally:
        handle.remove()


def test_supported_vectorized_runtime_error_is_not_retried(monkeypatch):
    _, critic, _ = _pair()
    calls = []

    def failure(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(layers, "vectorized_forward", failure)
    with pytest.raises(RuntimeError, match="kernel failure"):
        critic(torch.randn(4, 5).double())
    assert calls == [1]


def test_probe_provenance_includes_backend_implementation_and_identity():
    from probe_pave import _source_identity

    files = _source_identity()["files"]
    for path in ("RL/tdmpc2_core/common/ensemble_vmap.py", "utils/ensemble_identity.py"):
        assert path in files and len(files[path]) == 64


@pytest.mark.parametrize("detached", [False, True])
def test_strict_aot_compilation_preserves_forward_and_gradient_parity(detached):
    # Exercise the same TensorDict monkeypatch present in the real agent runtime.
    from tests.test_ambi_root_local_sac import _tiny_model  # noqa: F401

    torch._dynamo.reset()
    try:
        loop, critic, _ = _pair("distributional")
        x = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
        expected = loop.forward_detached(x) if detached else loop(x)
        expected_grads = torch.autograd.grad(expected.square().sum(), (x, *loop.parameters()),
                                            allow_unused=True)
        fn = critic._forward_detached_eager if detached else critic._forward_eager
        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=False)
        actual = compiled(x)
        actual_grads = torch.autograd.grad(actual.square().sum(), (x, *critic.parameters()),
                                          allow_unused=True)
        _close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert (actual_grad is None) == (expected_grad is None)
            if expected_grad is not None:
                _close(actual_grad, expected_grad)
    finally:
        torch._dynamo.reset()
