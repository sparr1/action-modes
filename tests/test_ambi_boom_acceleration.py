"""BOOM tensor compilation preserves samples, gradients, and scale ownership."""

from copy import deepcopy
import sys

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_boom import _agent, _labels
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree


@pytest.fixture(autouse=True)
def _avoid_unrelated_macos_compiler_worker_forks(monkeypatch):
    if sys.platform == "darwin":
        import torch._inductor.config

        # Dynamo's functorch guard may look up Inductor even with our callable
        # backend. Its import otherwise forks workers, which macOS can abort.
        monkeypatch.setattr(torch._inductor.config, "compile_threads", 1)


@torch.no_grad()
def _nonzero_critic(agent):
    # Fresh Q heads start with a zero output layer. Exercise a genuine action
    # gradient through the compiled critic, not only entropy/alignment gradients.
    for head in agent.model._Qs.modules_list:
        head[-1].weight.normal_(std=0.1)
        head[-1].bias.normal_(std=0.05)


def _inputs(agent, *, all_invalid=False):
    cfg = agent.cfg
    zs = torch.randn(
        cfg.train_unroll_horizon + 1, cfg.batch_size, cfg.latent_dim,
        device=agent.device,
    )
    mean, log_std, valid = (value.to(agent.device) for value in _labels(agent))
    valid[:, -1] = False
    if all_invalid:
        valid.zero_()
    mean[~valid.expand_as(mean)] = float("nan")
    return zs, mean, log_std, valid


def _trace_actor_region(agent):
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    region = agent._boom_actor_region
    region.enabled = True
    region.strict = True
    # A callable backend exercises Dynamo's actual full-graph contract without
    # starting unrelated Inductor compiler workers in the macOS test runtime.
    region._compiled = torch.compile(
        agent._boom_actor_kernel, backend=backend, fullgraph=True, dynamic=False,
    )
    return graphs


def _actor_step(agent, inputs):
    zs, mean, log_std, valid = inputs
    return agent._update_actor(
        zs, behavior_pre_tanh_mean=mean, behavior_log_std=log_std,
        behavior_policy_valid=valid,
        **({"replay_action": (mean * 1.2).tanh(), "behavior_pre_tanh_action": mean * 1.2}
           if agent._boom_replay_likelihood else {}),
    )


def _assert_tree_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _assert_tree_close(first, second)
    else:
        assert actual == expected


def _assert_actor_state_equal(actual, expected, *, compiled=False):
    compare = _assert_tree_close if compiled else _assert_tree_equal
    compare(actual.model.state_dict(), expected.model.state_dict())
    compare(actual.pi_optim.state_dict(), expected.pi_optim.state_dict())
    torch.testing.assert_close(actual.actor_loss_scale, expected.actor_loss_scale, rtol=0, atol=0)
    for first, second in zip(actual.model._pi.parameters(), expected.model._pi.parameters()):
        compare(first.grad, second.grad)


@pytest.mark.parametrize("entropy_mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("all_invalid", [False, True])
@pytest.mark.parametrize("alignment", ["sampled_action", "replay_likelihood"])
def test_boom_fullgraph_kernel_matches_eager_and_proposes_scale_without_mutation(
    entropy_mode, all_invalid, alignment,
):
    eager = _agent(
        q_representation="distributional", num_q=5, q_pair_size=2,
        outer_q_actor_reduction="min_pair", dropout=0.2,
        outer_actor_entropy_mode=entropy_mode,
        outer_boom_alignment=alignment,
    )
    _nonzero_critic(eager)
    compiled = AMBITDMPC2Agent(deepcopy(eager.cfg))
    compiled.model.load_state_dict(eager.model.state_dict())
    eager.model.train()
    compiled.model.train()
    eager.actor_loss_scale.fill_(3.2)
    compiled.actor_loss_scale.copy_(eager.actor_loss_scale)
    zs, mean, _, valid = _inputs(eager, all_invalid=all_invalid)
    mean.requires_grad_()
    graphs = _trace_actor_region(compiled)
    before = eager.actor_loss_scale.clone()
    rng = torch.get_rng_state().clone()
    replay_kwargs = (
        {"replay_action": (mean * 1.2).tanh(), "behavior_pre_tanh_action": mean * 1.2}
        if alignment == "replay_likelihood" else {}
    )
    expected = eager._boom_actor_kernel(
        zs[:-1], mean, valid, eager.alpha.detach(), eager.actor_loss_scale,
        **replay_kwargs,
    )
    expected["actor_loss"].backward()
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = compiled._boom_actor_region(
        zs[:-1], mean, valid, compiled.alpha.detach(), compiled.actor_loss_scale,
        **replay_kwargs,
    )
    actual["actor_loss"].backward()

    assert graphs
    _assert_tree_close(actual, expected)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    torch.testing.assert_close(eager.actor_loss_scale, before, rtol=0, atol=0)
    torch.testing.assert_close(compiled.actor_loss_scale, before, rtol=0, atol=0)
    independent_range = (
        torch.quantile(expected["q_policy"][0].detach(), 0.95, dim=0)
        - torch.quantile(expected["q_policy"][0].detach(), 0.05, dim=0)
    ).clamp_min(1)
    torch.testing.assert_close(
        actual["proposed_scale"], before.lerp(independent_range, eager.cfg.sac_actor_loss_scale_tau),
    )
    assert mean.grad is None
    for actual_parameter, expected_parameter in zip(
        compiled.model._pi.parameters(), eager.model._pi.parameters(),
    ):
        assert actual_parameter.grad is not None
        assert torch.isfinite(actual_parameter.grad).all()
        _assert_tree_close(actual_parameter.grad, expected_parameter.grad)
    assert all(parameter.grad is None for parameter in compiled.model._Qs.parameters())
    if all_invalid:
        assert actual["boom_metrics"]["boom_alignment_loss"] == 0


@pytest.mark.parametrize("alignment", ["sampled_action", "replay_likelihood"])
def test_boom_compiled_actor_first_and_next_adam_steps_match_eager(alignment):
    eager = _agent(
        q_representation="distributional", num_q=5, q_pair_size=2,
        outer_q_actor_reduction="min_pair", dropout=0.2,
        outer_boom_alignment=alignment,
    )
    _nonzero_critic(eager)
    compiled = AMBITDMPC2Agent(deepcopy(eager.cfg))
    compiled.load_training_state_dict(_clone_tree(eager.training_state_dict()))
    eager.model.train()
    compiled.model.train()
    eager.actor_loss_scale.fill_(2.15)
    compiled.actor_loss_scale.copy_(eager.actor_loss_scale)
    graphs = _trace_actor_region(compiled)
    gates = []
    for index, all_invalid in enumerate((False, True, False)):
        if index:
            # Outer critic learning precedes every actor step in production.
            # Compiled detached critic parameters must observe those changes.
            with torch.no_grad():
                for agent in (eager, compiled):
                    for head in agent.model._Qs.modules_list:
                        head[-1].weight[0].add_(0.002)
        inputs = _inputs(eager, all_invalid=all_invalid)
        before = eager.actor_loss_scale.clone()
        rng = torch.get_rng_state().clone()
        expected = _actor_step(eager, inputs)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        actual = _actor_step(compiled, inputs)
        _assert_tree_close(actual, expected)
        _assert_actor_state_equal(compiled, eager, compiled=True)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.testing.assert_close(
            actual["actor_loss_scale"],
            before.lerp(actual["actor_q_range_instant"], eager.cfg.sac_actor_loss_scale_tau),
        )
        gates.append(actual["boom_gate_active"].item())
    assert graphs
    # The running scale crosses the threshold while the compiled shape stays
    # fixed; a Python gate frozen during tracing would get this wrong.
    assert gates[0] == 1 and gates[-1] == 0


@pytest.mark.parametrize("alignment", ["sampled_action", "replay_likelihood"])
def test_boom_lazy_forward_failure_retries_same_sample_and_commits_scale_once(alignment):
    eager = _agent(
        q_representation="distributional", num_q=5, q_pair_size=2,
        outer_q_actor_reduction="min_pair", dropout=0.2,
        outer_boom_alignment=alignment,
    )
    _nonzero_critic(eager)
    fallback = AMBITDMPC2Agent(deepcopy(eager.cfg))
    fallback.load_training_state_dict(_clone_tree(eager.training_state_dict()))
    eager.model.train()
    fallback.model.train()
    eager.actor_loss_scale.fill_(4)
    fallback.actor_loss_scale.copy_(eager.actor_loss_scale)
    region = fallback._boom_actor_region
    region.enabled = True
    region.strict = False
    calls = []

    def lazy_failure(*args, **kwargs):
        calls.append(1)
        region.eager(*args, **kwargs)
        torch.rand(7)
        raise RuntimeError("injected BOOM backend failure after sampling")

    region._compiled = lazy_failure
    inputs = _inputs(eager)
    before = eager.actor_loss_scale.clone()
    rng = torch.get_rng_state().clone()
    expected = _actor_step(eager, inputs)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    with pytest.warns(RuntimeWarning, match="injected BOOM backend failure"):
        actual = _actor_step(fallback, inputs)
    assert actual["compile_outer_actor_fallback"] == 1
    assert expected["compile_outer_actor_fallback"] == 0
    expected["compile_outer_actor_fallback"] = 1
    _assert_tree_equal(actual, expected)
    _assert_actor_state_equal(fallback, eager)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    torch.testing.assert_close(
        fallback.actor_loss_scale,
        before.lerp(actual["actor_q_range_instant"], eager.cfg.sac_actor_loss_scale_tau),
    )
    assert calls == [1] and region.failed and not region.enabled


@pytest.mark.parametrize("objective", ["boom", "q_maximization"])
@pytest.mark.parametrize("enabled", [False, True])
def test_boom_actor_region_activation_and_cpu_mode_are_portable(objective, enabled):
    agent = _agent(
        outer_actor_objective=objective, compile=enabled,
        outer_boom_compile_mode="reduce-overhead",
    )
    assert agent.cfg.outer_boom_compile_mode == "reduce-overhead"
    assert agent._boom_actor_compile_mode == "default"
    assert agent._boom_actor_region.mode == "default"
    assert agent._boom_actor_region.enabled is (enabled and objective == "boom")


@pytest.mark.parametrize("alignment", ["sampled_action", "replay_likelihood"])
def test_boom_packed_metrics_outlive_reused_graph_output_storage(alignment):
    agent = _agent(outer_boom_alignment=alignment)
    agent.actor_loss_scale.fill_(4)
    region = agent._boom_actor_region
    # Exercise the CUDA output-ownership branch on CPU with simulated reusable
    # graph storage. The production routing test above keeps CPU mode default.
    agent._boom_actor_compile_mode = "reduce-overhead"
    region.enabled = True
    graph_outputs = []

    def simulate_graph(*args, **kwargs):
        outputs = region.eager(*args, **kwargs)
        graph_outputs.append(outputs)
        return outputs

    region._compiled = simulate_graph
    metrics = _actor_step(agent, _inputs(agent))
    snapshot = _clone_tree(metrics)
    graph_scalars = {
        "actor_loss": graph_outputs[0]["actor_loss"],
        **graph_outputs[0]["boom_metrics"],
    }
    # A single allocation owns every escaping graph scalar. Each metric is a
    # view of that allocation, with no reference to the reusable graph buffers.
    packed_storage = metrics["actor_loss"].untyped_storage().data_ptr()
    for name, graph_scalar in graph_scalars.items():
        assert metrics[name].untyped_storage().data_ptr() == packed_storage
        assert graph_scalar.untyped_storage().data_ptr() != packed_storage
    with torch.no_grad():
        for graph_scalar in graph_scalars.values():
            graph_scalar.fill_(-1234)
    _assert_tree_equal(metrics, snapshot)
    _actor_step(agent, _inputs(agent))
    _assert_tree_equal(metrics, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("alignment", ["sampled_action", "replay_likelihood"])
def test_boom_reduce_overhead_cuda_uses_fresh_samples_and_survives_restore(representation, alignment):
    base = _agent(
        q_representation=representation,
        num_q=5 if representation == "distributional" else 2,
        q_pair_size=2, outer_q_actor_reduction="min_pair", dropout=0.2,
        outer_boom_alignment=alignment,
    )
    cfg = deepcopy(base.cfg)
    cfg.device = "cuda"
    cfg.compile = True
    cfg.compile_strict = True
    cfg.outer_boom_compile_mode = "reduce-overhead"
    agent = AMBITDMPC2Agent(cfg)
    _nonzero_critic(agent)
    agent.model.train()
    agent.actor_loss_scale.fill_(4)
    # Keep weights fixed to make fresh stochastic draws observable separately
    # from learning; Adam state still advances and is included in restoration.
    for group in agent.pi_optim.param_groups:
        group["lr"] = 0.0
    inputs = _inputs(agent)
    saved = _clone_tree(agent.training_state_dict())
    held_metrics = None
    held_snapshot = None
    sample_statistics = []
    for index in range(8):
        if index == 5:
            inputs[-1].zero_()
        elif index == 6:
            inputs = _inputs(agent)
        torch.compiler.cudagraph_mark_step_begin()
        before = agent.actor_loss_scale.clone()
        metrics = _actor_step(agent, inputs)
        torch.cuda.synchronize()
        assert torch.isfinite(metrics["actor_loss"])
        assert torch.isfinite(metrics["actor_grad_norm"])
        assert metrics["boom_valid_fraction"] == (0 if index == 5 else 0.75)
        if index == 5:
            assert metrics["boom_alignment_loss"] == 0
        torch.testing.assert_close(
            agent.actor_loss_scale,
            before.lerp(metrics["actor_q_range_instant"], cfg.sac_actor_loss_scale_tau),
        )
        sample_statistics.append(metrics["actor_pre_tanh_abs_mean"].item())
        if index == 2:
            held_metrics, held_snapshot = metrics, _clone_tree(metrics)
    assert len(set(sample_statistics)) > 1
    _assert_tree_equal(held_metrics, held_snapshot)
    assert not agent._boom_actor_region.failed
    agent.load_training_state_dict(saved)
    torch.compiler.cudagraph_mark_step_begin()
    metrics = _actor_step(agent, inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        agent.actor_loss_scale,
        torch.full_like(agent.actor_loss_scale, 4).lerp(
            metrics["actor_q_range_instant"], cfg.sac_actor_loss_scale_tau,
        ),
    )
    assert torch.isfinite(metrics["actor_loss"])
