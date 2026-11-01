"""Captured PAVE must preserve derivatives, combined updates, RNG and resumes."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core import pave
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.pave_graph import PaveGraph, captured_gradient_loss
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params
from tests.test_pave_identity import identity
from tests.test_pave_view_batching import close_tree
from tests.test_pave_view_batching_update import _batch, _nonconstant_all_critics
from utils.pave_identity import resolve_pave_controls, pave_training_signature

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU.")
TERMS = ("mpr", "action_smoothness", "both")
TARGETS = ("primary", "auxiliary", "both")


def _options(term="mpr", target="primary", **changes):
    terms = ("mpr", "action_smoothness") if term == "both" else (term,)
    prefixes = {"primary": ("outer_pave",), "auxiliary": ("aux_return_pave",),
                "both": ("outer_pave", "aux_return_pave")}[target]
    return {**{f"{prefix}_{name}_coef": .13 if prefix == "outer_pave" else .07
               for prefix in prefixes for name in terms}, **changes}


def _agent(term="mpr", target="primary", *, device="cpu", graph=False,
           representation="distributional", compiled=False, **changes):
    options = _options(
        term, target, aux_return_mode="sac", critic_ensemble_backend="vmap",
        q_representation=representation, num_q=5 if representation == "distributional" else 2,
        inner_rounds=1, inner_updates_per_round=1, ent_coef="auto_0.1",
        outer_pave_batch_size=3, grad_clip_norm=.01, dropout=.2,
        outer_pave_cuda_graph=graph, compile=compiled, compile_strict=compiled,
    )
    options.update(changes)
    cfg = _build_cfg(**_tiny_params(**options))
    cfg.device = device
    return AMBITDMPC2Agent(cfg)


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_graph_flag_requires_a_boolean(value):
    with pytest.raises(ValueError, match="must be a bool"):
        resolve_pave_controls({"outer_pave_cuda_graph": value})


def test_graph_mode_rejects_inactive_losses_and_unvalidated_terms():
    with pytest.raises(ValueError, match="requires active"):
        resolve_pave_controls({"outer_pave_cuda_graph": True})
    for key in ("outer_pave_vfc_coef", "aux_return_pave_curvature_coef"):
        with pytest.raises(ValueError, match="MPR and action smoothness only"):
            resolve_pave_controls({"outer_pave_cuda_graph": True, key: .1})
    with pytest.raises(ValueError, match="requires CUDA"):
        _agent(graph=True)


@pytest.mark.parametrize("term", TERMS)
def test_graph_execution_is_pinned_but_disabled_identity_is_unchanged(term):
    options = _options(term)
    assert identity(**options) == identity(**options, outer_pave_cuda_graph=False)
    assert identity(**options) != identity(**options, outer_pave_cuda_graph=True)
    before = pave_training_signature(options)
    after = pave_training_signature({**options, "outer_pave_cuda_graph": True})
    assert after.pop("execution") == "cuda_graph_loss_and_parameter_gradients_v1"
    assert after == before


@pytest.mark.parametrize("scale", [0., 1., -2.3])
def test_gradient_bridge_accumulates_with_other_losses_and_snapshots_inputs(scale):
    p = torch.nn.Parameter(torch.tensor([.3, -.7], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    value = (p**4).sum()
    expected = torch.autograd.grad(value, p)[0]
    supplied = expected.clone()
    bridged = captured_gradient_loss(value, (supplied, None), (p, unused))
    supplied.zero_()  # Later graph replays may overwrite the entire source.
    combined = 2 * p.square().sum() + scale * bridged
    combined.backward(retain_graph=True)
    torch.testing.assert_close(p.grad, 4 * p.detach() + scale * expected, rtol=0, atol=0)
    assert unused.grad is None
    p.grad = None
    (bridged * 3).backward()
    torch.testing.assert_close(p.grad, 3 * expected, rtol=0, atol=0)


def test_gradient_bridge_detects_parameter_changes_before_backward():
    p = torch.nn.Parameter(torch.ones(2))
    loss = captured_gradient_loss(p.sum(), (p.detach().clone(),), (p,))
    with torch.no_grad():
        p.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("term", TERMS)
@pytest.mark.parametrize("target", TARGETS)
def test_cpu_bridge_uses_actual_regularizer_derivatives(term, target):
    """Exercise the bridge algebra on real geometry without claiming CUDA coverage."""
    agent = _agent(term, target)
    _nonconstant_all_critics(agent)
    parameters = tuple(p for p in agent.model.parameters() if p.requires_grad)
    obs, action, *_ = _batch(agent)
    loss, _ = pave.outer_pave_loss(agent.model, obs, action, agent.cfg, 5)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    bridge = captured_gradient_loss(loss, gradients, parameters)
    actual = torch.autograd.grad(bridge * .37, parameters, allow_unused=True)
    for wanted, got in zip(gradients, actual):
        if wanted is None:
            assert got is None
        else:
            torch.testing.assert_close(got, .37 * wanted, rtol=0, atol=0)


def _rng(device):
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()


def _restore_rng(state, device):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1], device)


@CUDA
@pytest.mark.parametrize("term", TERMS)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("full_batch", [False, True])
def test_cuda_loss_gradients_ownership_rng_and_changed_inputs(term, target, representation, full_batch):
    agent = _agent(term, target, device="cuda", graph=True, representation=representation)
    if full_batch:
        agent.cfg.outer_pave_batch_size = None
    _nonconstant_all_critics(agent)
    runner = PaveGraph(agent.model, agent.cfg)
    all_parameters = tuple(p for p in agent.model.parameters() if p.requires_grad)
    for parameter in all_parameters:
        parameter.grad = torch.full_like(parameter, .123)
    original_grads = tuple(p.grad.clone() for p in all_parameters)
    modes = [(module, module.training) for module in agent.model.modules()]
    held_metrics = []
    for index in range(3):
        obs, action, *_ = [value.cuda() for value in _batch(agent)]
        obs = (obs + index * .2).requires_grad_()
        action = (action + index * .1).clamp(-1, 1).requires_grad_()
        before = _rng(agent.device)
        expected, expected_info = pave.outer_pave_loss(agent.model, obs, action, agent.cfg, index + 1)
        expected_gradients = torch.autograd.grad(expected, all_parameters, allow_unused=True)
        actual, info = runner(obs, action, index + 1)
        assert info.pop("outer_pave_cuda_graph_used") == 1
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
        close_tree(info, expected_info, rtol=2e-5, atol=2e-7)
        actual_gradients = torch.autograd.grad(actual, all_parameters, allow_unused=True)
        close_tree(actual_gradients, expected_gradients, rtol=3e-5, atol=2e-7)
        close_tree(_rng(agent.device), before, rtol=0, atol=0)
        assert all(module.training == mode for module, mode in modes)
        for p, grad in zip(all_parameters, original_grads):
            torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
        assert obs.grad is None and action.grad is None
        held_metrics.append((info, {key: value.clone() for key, value in info.items()}))
        with torch.no_grad():
            for _, critic, _ in runner.selected:
                for parameter in critic.parameters():
                    parameter.add_(.003)  # Capture must read current weights.
    assert runner.calls == 3
    for metrics, snapshot in held_metrics:
        close_tree(metrics, snapshot, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("term", ["mpr", "action_smoothness"])
def test_multiple_outstanding_forwards_have_independent_gradients(term):
    agent = _agent(term, device="cuda", graph=True)
    _nonconstant_all_critics(agent)
    runner = PaveGraph(agent.model, agent.cfg)
    obs, action, *_ = [value.cuda() for value in _batch(agent)]
    first, _ = runner(obs, action, 1)
    second, _ = runner(obs + .7, -action, 2)
    a, _ = pave.outer_pave_loss(agent.model, obs, action, agent.cfg, 1)
    b, _ = pave.outer_pave_loss(agent.model, obs + .7, -action, agent.cfg, 2)
    actual = torch.autograd.grad(first + 2 * second, runner.parameters, allow_unused=True)
    expected = torch.autograd.grad(a + 2 * b, runner.parameters, allow_unused=True)
    close_tree(actual, expected, rtol=3e-5, atol=2e-7)


@CUDA
@pytest.mark.parametrize("term", TERMS)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("compiled", [False, True])
def test_complete_updates_preserve_clipping_adam_targets_and_rng(term, target, compiled):
    if compiled:
        torch._dynamo.reset()
    eager = _agent(term, target, device="cuda", compiled=compiled)
    graph = _agent(term, target, device="cuda", compiled=compiled, graph=True)
    _nonconstant_all_critics(eager)
    graph.load(deepcopy(eager.checkpoint_state()))
    gradient_snapshots = [[], []]
    for agent, snapshots in zip((eager, graph), gradient_snapshots):
        step = agent.optim.step
        def observed_step(*args, _step=step, _agent=agent, _rows=snapshots, **kwargs):
            _rows.append({name: p.grad.detach().clone() for name, p in _agent.model.named_parameters()
                          if p.grad is not None})
            return _step(*args, **kwargs)
        agent.optim.step = observed_step
    for index in range(3):
        data = [value.cuda() for value in _batch(eager)]
        data[0] += index * .1
        before = _rng(eager.device)
        expected = eager._update(*data)
        expected_rng = _rng(eager.device)
        _restore_rng(before, eager.device)
        actual = graph._update(*data)
        assert actual.pop("outer_pave_cuda_graph_used") == 1
        close_tree(actual, expected, rtol=5e-5, atol=5e-7)
        assert actual["grad_norm"] > graph.cfg.grad_clip_norm
        close_tree(gradient_snapshots[1][-1], gradient_snapshots[0][-1], rtol=5e-5, atol=5e-7)
        close_tree(graph.model.state_dict(), eager.model.state_dict(), rtol=5e-5, atol=5e-7)
        for name in ("optim", "pi_optim", "ent_coef_optim"):
            close_tree(getattr(graph, name).state_dict(), getattr(eager, name).state_dict(),
                       rtol=5e-5, atol=5e-7)
        close_tree(graph.log_ent_coef, eager.log_ent_coef, rtol=5e-5, atol=5e-7)
        close_tree(graph.aux_return.checkpoint_state(), eager.aux_return.checkpoint_state(),
                   rtol=5e-5, atol=5e-7)
        assert graph.num_updates == eager.num_updates == index + 1
        close_tree(_rng(graph.device), expected_rng, rtol=0, atol=0)
    assert graph._pave_graph.calls == 3
    if compiled:
        torch._dynamo.reset()


@CUDA
@pytest.mark.parametrize("term", TERMS)
def test_exact_resume_recaptures_and_reproduces_next_update(term):
    agent = _agent(term, "both", device="cuda", graph=True)
    _nonconstant_all_critics(agent)
    data = [value.cuda() for value in _batch(agent)]
    agent._update(*data)
    agent.prepare_training_resume_boundary()
    saved = deepcopy(agent.training_state_dict())
    before = _rng(agent.device)
    expected = agent._update(*data)
    wanted_rng = _rng(agent.device)
    wanted = deepcopy(agent.checkpoint_state())
    agent.load_training_state_dict(saved)
    assert agent._pave_graph is None
    _restore_rng(before, agent.device)
    actual = agent._update(*data)
    close_tree(actual, expected, rtol=0, atol=0)
    close_tree(agent.checkpoint_state(), wanted, rtol=0, atol=0)
    close_tree(_rng(agent.device), wanted_rng, rtol=0, atol=0)
    # A different accumulation protocol must fail before touching live state.
    eager = _agent(term, "both", device="cuda")
    unchanged = deepcopy(eager.checkpoint_state())
    with pytest.raises(ValueError, match="critic target|critic-target|critic_target"):
        eager.load_training_state_dict(saved)
    close_tree(eager.checkpoint_state(), unchanged, rtol=0, atol=0)


@CUDA
def test_graph_rejects_changed_storage_hooks_shapes_and_configuration():
    agent = _agent(device="cuda", graph=True)
    _nonconstant_all_critics(agent)
    runner = PaveGraph(agent.model, agent.cfg)
    obs, action, *_ = [value.cuda() for value in _batch(agent)]
    runner(obs, action, 1)
    agent.cfg.outer_pave_mpr_coef *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        runner(obs, action, 2)
    agent.cfg.outer_pave_mpr_coef /= 2
    hook = runner.parameters[0].register_hook(lambda value: value)
    with pytest.raises(ValueError, match="gradient hooks"):
        runner(obs, action, 2)
    hook.remove()
    agent.cfg.outer_pave_batch_size = None
    with pytest.raises(ValueError, match="configuration changed"):
        runner(obs, action, 2)
    agent.cfg.outer_pave_batch_size = 3
    with pytest.raises(ValueError, match="shapes/dtypes/devices changed"):
        runner(obs[:, :1], action[:, :1], 2)
    with torch.no_grad(), pytest.raises(ValueError, match="gradient-enabled"):
        runner(obs, action, 2)
    with torch.autocast("cuda"), pytest.raises(ValueError, match="autocast"):
        runner(obs, action, 2)
    parameter = runner.parameters[0]
    parameter.data = parameter.data.clone()
    with pytest.raises(ValueError, match="storage changed"):
        runner(obs, action, 2)
