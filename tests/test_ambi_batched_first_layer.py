"""Independent-head arithmetic and SAC state gates for the wider first GEMM."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.batched_teacher import BatchedTeacherCore
from RL.tdmpc2_core.reanalyze import _preserve_global_rng
from tests.test_ambi_batched_optimizations import ALL, ReplayTape, assert_core_close
from tests.test_ambi_batched_teacher import Tape
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_reanalyze import _assert_globals, _globals
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params


FUSED = ALL | {"first_layer_fusion"}


def _agent(*, device="cpu", representation="distributional", bias=True, **overrides):
    params = _tiny_params(
        q_representation=representation, num_q=5 if representation == "distributional" else 2,
        q_num_bins=101, q_pair_size=2, inner_rounds=2,
        inner_rollouts_per_round=4, inner_rollout_horizon=2,
        inner_updates_per_round=2, inner_replay_capacity=16,
        reanalyze_backend="batched", reanalyze_parallel_roots=3,
        dropout=.2, inner_critic_dropout_enabled=True,
        target_entropy=-1., inner_target_entropy=-1.,
    )
    params.update(overrides)
    if "inner_critic_updates_per_round" in params:
        params.pop("inner_updates_per_round")
    cfg = _build_cfg(**params)
    cfg.device = device
    cfg.action_dim = 3
    cfg.action_dims = [3]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    with torch.no_grad():
        for ensemble in (agent.model._Qs, agent.model._target_Qs):
            for head in ensemble:
                # Initialization otherwise makes distributional input gradients
                # zero. Nonzero final weights exercise the complete backward.
                head[-1].weight.normal_(std=.1)
                if not bias:
                    for layer in head:
                        layer.bias = None
    return agent


def _cores(agent, width, *, reference_features=ALL):
    reference = BatchedTeacherCore(agent, width, compiled=False,
                                   optimization_features=reference_features)
    fused = BatchedTeacherCore(agent, width, compiled=False, optimization_features=FUSED)
    reference.reset()
    fused.reset()
    return reference, fused


def _pairs(core, count):
    if count is None:
        return None
    # Different nonadjacent head order in each root, including singleton pairs.
    order = torch.arange(core.heads - 1, -1, -1, device=core.device)
    return torch.stack([order.roll(root)[:count] for root in range(core.width)])


@pytest.mark.parametrize("width,selected,bias,representation,training", [
    (1, None, True, "scalar", False),
    (3, 1, False, "scalar", True),
    (3, 2, True, "distributional", True),
    (1, None, False, "distributional", True),
    (3, None, True, "distributional", False),
    (1, 1, True, "distributional", False),
])
def test_first_layer_preserves_predictions_input_and_parameter_gradients(
    width, selected, bias, representation, training,
):
    torch.manual_seed(219)
    agent = _agent(representation=representation, bias=bias)
    reference, fused = _cores(agent, width)
    before = deepcopy(agent.model.state_dict())
    specifications = fused.critic_train_spec, fused.critic_eval_spec
    z = torch.randn(width, 7, agent.cfg.latent_dim)
    actions = torch.randn(width, 7, agent.cfg.action_dim)
    pairs = _pairs(reference, selected)
    masks = reference._masks(7, reference._draw) if training else None
    expected_z, actual_z = z.clone().requires_grad_(), z.clone().requires_grad_()
    expected_action = actions.clone().requires_grad_()
    actual_action = actions.clone().requires_grad_()
    rng = _globals()
    expected = reference._predictions(reference.critic, expected_z, expected_action,
                                      masks, training=training, pairs=pairs)
    actual = fused._predictions(fused.critic, actual_z, actual_action,
                                masks, training=training, pairs=pairs)
    # Use a nonuniform, deterministic cotangent so head/root permutations or
    # accidental shared-head reductions cannot disappear inside a mean loss.
    cotangent = torch.linspace(-.7, 1.3, expected.numel()).reshape_as(expected)
    (expected * cotangent).sum().backward()
    (actual * cotangent).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-6)
    torch.testing.assert_close(actual_z.grad, expected_z.grad, rtol=2e-4, atol=3e-6)
    torch.testing.assert_close(actual_action.grad, expected_action.grad, rtol=2e-4, atol=3e-6)
    assert actual_action.grad.abs().sum() > 0
    for key, parameter in fused.critic.items():
        torch.testing.assert_close(parameter.grad, reference.critic[key].grad,
                                   rtol=2e-4, atol=3e-6)
        if pairs is not None:
            for root in range(width):
                unused = [head for head in range(fused.heads) if head not in pairs[root].tolist()]
                assert not parameter.grad[root, unused].count_nonzero()
    assert all(parameter.grad is None for parameter in fused.actor.values())
    assert all(parameter.grad is None for parameter in agent.model.parameters())
    assert (fused.critic_train_spec, fused.critic_eval_spec) == specifications
    _assert_tree_equal(agent.model.state_dict(), before)
    _assert_globals(rng)


def test_first_layer_preserves_selected_q_action_gradients_and_actor_critic_isolation():
    torch.manual_seed(517)
    agent = _agent(inner_finite_horizon=True, inner_outer_policy_kl_coef=.2,
                   inner_q_actor_reduction="mean_pair")
    reference, fused = _cores(agent, 3)
    z = torch.randn(3, 7, agent.cfg.latent_dim)
    noise = torch.randn(3, 7, agent.cfg.action_dim)
    pairs = _pairs(reference, 2)
    masks = reference._masks(7, reference._draw)
    alpha = z.new_full((3, 1, 1), .03)
    expected = reference._actor_loss(z, alpha, noise, pairs, masks, True)
    actual = fused._actor_loss(z, alpha, noise, pairs, masks, True)
    expected[0].sum().backward()
    actual[0].sum().backward()
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, rtol=2e-4, atol=3e-6)
    for key, parameter in fused.actor.items():
        torch.testing.assert_close(parameter.grad, reference.actor[key].grad,
                                   rtol=2e-4, atol=3e-6)
    assert any(parameter.grad.abs().sum() > 0 for parameter in fused.actor.values())
    assert all(parameter.grad is None for parameter in fused.critic.values())

    for core in (reference, fused):
        core.actor_optim.zero_grad(set_to_none=True)
    rollout_noise = torch.randn(3, agent.cfg.inner_rollout_horizon,
                                agent.cfg.inner_rollouts_per_round, agent.cfg.action_dim)
    with torch.no_grad():
        batch = reference._rollout(rollout_noise)[:, :7]
    prior_noise = torch.randn_like(noise)
    expected_loss = reference._critic_loss(batch, alpha, noise, pairs, masks, prior_noise, pairs)
    actual_loss = fused._critic_loss(batch, alpha, noise, pairs, masks, prior_noise, pairs)
    expected_loss.sum().backward()
    actual_loss.sum().backward()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-4, atol=3e-6)
    for key, parameter in fused.critic.items():
        torch.testing.assert_close(parameter.grad, reference.critic[key].grad,
                                   rtol=2e-4, atol=3e-6)
    assert all(parameter.grad is None for parameter in fused.actor.values())
    assert all(parameter.grad is None for parameter in agent.model.parameters())


def _compare_fresh_solves(agent, reference, fused, *, iterations=2):
    for iteration in range(iterations):
        if iteration:
            with torch.no_grad():
                for parameter in agent.model.parameters():
                    parameter.add_(.002)
                if agent.log_ent_coef is not None:
                    agent.log_ent_coef.add_(.1)
            reference.refresh_inputs()
            fused.refresh_inputs()
        roots = torch.randn_like(reference.root)
        reference.root.copy_(roots)
        fused.root.copy_(roots)
        outer = deepcopy(agent.model.state_dict())
        live = deepcopy(agent.inner_engine.training_state_dict())
        tape = Tape(reference)
        expected = reference.run(tape)
        reader = ReplayTape(tape.entries)
        rng = _globals()
        cuda_rng = torch.cuda.get_rng_state().clone() if roots.is_cuda else None
        actual = fused.run(reader)
        if roots.is_cuda:
            torch.cuda.synchronize()
        assert reader.index == len(tape.entries)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, rtol=5e-4, atol=1e-5)
        assert_core_close(fused, reference, rtol=5e-4, atol=1e-5)
        _assert_tree_equal(agent.model.state_dict(), outer)
        _assert_tree_equal(agent.inner_engine.training_state_dict(), live)
        assert all(parameter.grad is None for parameter in agent.model.parameters())
        _assert_globals(rng)
        if cuda_rng is not None:
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)


@pytest.mark.parametrize("representation,width,ring,reference_features", [
    ("scalar", 1, False, ALL),
    ("distributional", 3, True, ALL),
    ("distributional", 3, False, frozenset()),
])
def test_first_layer_full_solve_preserves_fresh_priors_replay_and_adam(
    representation, width, ring, reference_features,
):
    torch.manual_seed(916)
    agent = _agent(representation=representation, inner_rounds=3,
                   inner_critic_updates_per_round=4, inner_actor_updates_per_round=2,
                   inner_temperature_updates_per_round=2, inner_replay_capacity=24,
                   inner_finite_horizon=True, inner_actor_entropy_mode="tdmpc2_scaled",
                   inner_log_std_mapping="tdmpc2_tanh")
    if ring:
        # Exercise the low-level storage contract; public fresh-SAC recipes
        # reserve sufficient capacity for all action-local imagined transitions.
        agent.cfg.inner_replay_capacity = 11
    reference, fused = _cores(agent, width, reference_features=reference_features)
    _compare_fresh_solves(agent, reference, fused)
    if ring:
        assert fused.size == 11 and fused.position == 24 % 11


@pytest.mark.parametrize("backend", [
    "aot_eager",
    pytest.param("inductor", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires a scheduled CUDA GPU.",
    )),
])
def test_first_layer_full_solve_compiles_with_matching_gradients_and_optimizer_state(backend):
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    try:
        torch.manual_seed(812)
        agent = _agent(device="cuda" if backend == "inductor" else "cpu",
                       inner_critic_updates_per_round=3, inner_actor_updates_per_round=1,
                       inner_temperature_updates_per_round=1,
                       inner_finite_horizon=True, inner_actor_entropy_mode="tdmpc2_scaled")
        reference, fused = _cores(agent, 3)
        for region in fused.regions.values():
            region.enabled = True
            region._compiled = torch.compile(region.eager, backend=backend,
                                               fullgraph=True, dynamic=False)
        if backend == "inductor":
            # Isolate compiler/autotuner initialization just as graph setup does;
            # the following explicit-tape solves must then consume no RNG.
            with _preserve_global_rng():
                fused.run()
                torch.cuda.synchronize()
        _compare_fresh_solves(agent, reference, fused)
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(previous_precision)
