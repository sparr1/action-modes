"""TD-M(PC)^2 acceleration preserves the likelihood, gradients, and updates."""

from copy import deepcopy
import sys
import threading

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.behavior_policy import squashed_behavior_cross_entropy
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_inner_decoupling import _clone_tree
from tests.test_ambi_tdmpc_square import _agent, _identity, _recipe


@pytest.fixture(autouse=True)
def _avoid_macos_compiler_worker_forks(monkeypatch):
    if sys.platform == "darwin":
        import torch._inductor.config
        from tqdm import tqdm

        monkeypatch.setattr(torch._inductor.config, "compile_threads", 1)
        # FX creates a progress-bar lock even with progress disabled. A thread
        # lock avoids macOS resource-tracker subprocesses for this local test.
        monkeypatch.setattr(tqdm, "_lock", threading.RLock(), raising=False)


def _inputs(agent, dimension=3, mask="partial", dtype=torch.float64):
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    info = {
        "pre_tanh_mean": torch.randn(h + 1, b, dimension, dtype=dtype, requires_grad=True),
        "log_std": torch.full((h + 1, b, dimension), -0.4, dtype=dtype, requires_grad=True),
        "pre_tanh_action": torch.randn(h + 1, b, dimension, dtype=dtype, requires_grad=True),
    }
    mean = torch.randn(h, b, dimension, dtype=dtype)
    log_std = torch.full_like(mean, -0.3)
    valid = torch.ones(h, b, 1, dtype=torch.bool)
    if mask == "partial":
        valid[:, -1] = False
    elif mask == "none":
        valid.zero_()
    elif mask == "one":
        valid.zero_()
        valid[-1, 0] = True
    elif mask == "last_depth":
        valid[:-1] = False
    for value in (mean, log_std):
        value[~valid.expand_as(value)] = float("nan")
        value.requires_grad_()
    return info, mean, log_std, valid


def _assert_close(actual, expected, **tolerance):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_close(actual[key], expected[key], **tolerance)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _assert_close(first, second, **tolerance)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, **tolerance)
    elif isinstance(actual, torch.Tensor):
        assert actual.item() == pytest.approx(expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("dimension", [1, 21, 61])
@pytest.mark.parametrize("mask,rho", [
    ("all", 0.5), ("partial", 0.5), ("none", 0.5), ("one", 0.5),
    ("partial", 0.0), ("last_depth", 0.0),
])
def test_fixed_shape_ce_matches_reference_values_and_gradients(dimension, mask, rho):
    agent = _agent(train_unroll_horizon=3, rho=rho, outer_behavior_policy_kl_min_valid_count=2)
    agent.cfg.action_dim = dimension
    inputs = _inputs(agent, dimension, mask)
    info, mean, log_std, valid = inputs
    expected = agent._behavior_policy_action_ce_loss_reference(*inputs)
    expected_grads = torch.autograd.grad(expected[0], tuple(info.values()))
    actual = agent._behavior_policy_action_ce_loss(*inputs)
    actual_grads = torch.autograd.grad(actual[0], tuple(info.values()))
    _assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    _assert_close(actual_grads, expected_grads, rtol=1e-12, atol=1e-12)
    assert mean.grad is None and log_std.grad is None
    for grad in actual_grads:
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad[:-1][~valid.expand_as(mean)]) == 0
        assert torch.count_nonzero(grad[-1]) == 0


def test_ce_valid_nan_keeps_reference_quantile_nan_propagation():
    agent = _agent()
    agent.cfg.action_dim = 3
    inputs = _inputs(agent)
    with torch.no_grad():
        inputs[1][0, 0, 0] = float("nan")
    expected = agent._behavior_policy_action_ce_loss_reference(*inputs)
    actual = agent._behavior_policy_action_ce_loss(*inputs)
    for name in ("behavior_policy_action_ce_p50", "behavior_policy_action_ce_p95"):
        assert torch.isnan(actual[2][name]) and torch.isnan(expected[2][name])


def test_ce_nonfinite_temporal_support_returns_zero_loss_and_gradients():
    agent = _agent()
    agent.cfg.action_dim = 3
    agent._transition_temporal_weights.fill_(float("inf"))
    inputs = _inputs(agent)
    expected = agent._behavior_policy_action_ce_loss_reference(*inputs)
    actual = agent._behavior_policy_action_ce_loss(*inputs)
    _assert_close(actual, expected, rtol=0, atol=0)
    for grad in torch.autograd.grad(actual[0], tuple(inputs[0].values())):
        assert torch.count_nonzero(grad) == 0


def _compile_ce(agent, backend):
    region = agent._behavior_ce_region
    region.enabled = True
    region.strict = True
    region._compiled = torch.compile(
        squashed_behavior_cross_entropy, backend=backend, fullgraph=True, dynamic=False,
    )


@pytest.mark.parametrize("backend_kind", ["trace", "aot_eager"])
def test_fullgraph_forward_backward_accepts_changing_validity(backend_kind):
    agent = _agent(train_unroll_horizon=3, outer_behavior_policy_kl_min_valid_count=2)
    agent.cfg.action_dim = 3
    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    _compile_ce(agent, backend if backend_kind == "trace" else backend_kind)
    for mask in ("all", "partial", "none", "one"):
        inputs = _inputs(agent, mask=mask, dtype=torch.float32)
        expected = agent._behavior_policy_action_ce_loss_reference(*inputs)
        expected_grads = torch.autograd.grad(expected[0], tuple(inputs[0].values()))
        actual = agent._behavior_policy_action_ce_loss(*inputs)
        actual_grads = torch.autograd.grad(actual[0], tuple(inputs[0].values()))
        _assert_close(actual, expected, rtol=3e-6, atol=1e-7)
        _assert_close(actual_grads, expected_grads, rtol=3e-6, atol=1e-7)
    if backend_kind == "trace":
        assert len(graphs) == 1


def test_ce_kernel_has_no_scalar_reads_dynamic_indexing_or_duplicate_sort():
    agent = _agent(train_unroll_horizon=3)
    agent.cfg.action_dim = 3
    inputs = _inputs(agent, dtype=torch.float32)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        loss, _, _ = agent._behavior_policy_action_ce_loss(*inputs)
        loss.backward()
    counts = {event.key: event.count for event in prof.key_averages()}
    for name in ("aten::item", "aten::_local_scalar_dense", "aten::nonzero", "aten::index"):
        assert counts.get(name, 0) == 0, counts
    assert counts.get("aten::sort", 0) == 1


def _actor_inputs(agent):
    cfg = agent.cfg
    zs = torch.randn(cfg.train_unroll_horizon + 1, cfg.batch_size, cfg.latent_dim)
    _, mean, log_std, valid = _inputs(agent, dimension=cfg.action_dim, dtype=torch.float32)
    return zs, mean.detach(), log_std.detach(), valid


@pytest.mark.parametrize("compiled", [False, True])
def test_first_and_next_actor_adam_updates_match_reference(compiled):
    reference = _agent(
        q_representation="distributional", num_q=5, q_pair_size=2,
        outer_q_actor_reduction="min_pair", dropout=0.2,
    )
    with torch.no_grad():
        for head in reference.model._Qs.modules_list:
            head[-1].weight.normal_(std=0.1)
    fast = AMBITDMPC2Agent(deepcopy(reference.cfg))
    fast.load_training_state_dict(_clone_tree(reference.training_state_dict()))
    reference._behavior_policy_action_ce_loss = reference._behavior_policy_action_ce_loss_reference
    reference.actor_loss_scale.fill_(4.0)
    fast.actor_loss_scale.copy_(reference.actor_loss_scale)
    if compiled:
        _compile_ce(fast, "aot_eager")
    for all_invalid in (False, True, False):
        inputs = _actor_inputs(reference)
        if all_invalid:
            inputs[-1].zero_()
        rng = torch.get_rng_state().clone()
        expected = reference._update_actor(*inputs)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        actual = fast._update_actor(*inputs)
        _assert_close(actual, expected, rtol=3e-5, atol=1e-6)
        _assert_close(fast.model.state_dict(), reference.model.state_dict(), rtol=3e-5, atol=1e-6)
        _assert_close(fast.pi_optim.state_dict(), reference.pi_optim.state_dict(), rtol=3e-5, atol=1e-6)
        torch.testing.assert_close(fast.actor_loss_scale, reference.actor_loss_scale, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


def test_lazy_compile_failure_retries_same_sample_once():
    agent = _agent()
    inputs = _actor_inputs(agent)
    state = _clone_tree(agent.training_state_dict())
    rng = torch.get_rng_state().clone()
    expected = agent._update_actor(*inputs)
    expected_state = _clone_tree(agent.training_state_dict())
    expected_rng = torch.get_rng_state().clone()
    agent.load_training_state_dict(state)
    torch.set_rng_state(rng)
    region = agent._behavior_ce_region
    region.enabled = True

    def failed(*args):
        region.eager(*args)
        torch.rand(5)
        raise RuntimeError("injected CE backend failure")

    region._compiled = failed
    with pytest.warns(RuntimeWarning, match="injected CE backend failure"):
        actual = agent._update_actor(*inputs)
    assert actual.pop("compile_behavior_ce_fallback") == 1
    assert expected.pop("compile_behavior_ce_fallback") == 0
    _assert_close(actual, expected, rtol=0, atol=0)
    _assert_close(agent.training_state_dict(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


def test_cuda_graph_metric_storage_is_owned_outside_kernel():
    agent = _agent()
    agent.cfg.action_dim = 3
    inputs = _inputs(agent, dtype=torch.float32)
    output = agent._behavior_policy_action_ce_loss(*inputs)
    stored = {name: value.clone() for name, value in output[2].items()}
    agent._behavior_ce_region.enabled = True
    agent._behavior_ce_region.strict = True
    agent._behavior_ce_compile_mode = "reduce-overhead"
    agent._behavior_ce_region._compiled = lambda *_args: output
    _, _, metrics = agent._behavior_policy_action_ce_loss(*inputs)
    with torch.no_grad():
        for tensor in output[2].values():
            tensor.zero_()
    _assert_close(metrics, stored, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [None, True, "max-autotune", "unknown"])
def test_compile_mode_rejects_invalid_configuration(bad):
    with pytest.raises(ValueError, match="outer_behavior_policy_compile_mode"):
        _build_cfg(**_recipe(outer_behavior_policy_compile_mode=bad))


def test_compile_mode_is_opt_in_portable_and_identified_only_when_active():
    params = _recipe()
    assert _identity(params) == _identity({**params, "outer_behavior_policy_compile_mode": "default"})
    assert _identity(params) == _identity({**params, "outer_behavior_policy_compile_mode": "reduce-overhead"})
    compiled = {**params, "compile": True}
    assert _identity(compiled) != _identity({**compiled, "outer_behavior_policy_compile_mode": "reduce-overhead"})
    baseline = {"compile": True}
    assert _identity(baseline) == _identity({**baseline, "outer_behavior_policy_compile_mode": "reduce-overhead"})
    cfg = _build_cfg(**_recipe(outer_behavior_policy_compile_mode="REDUCE-OVERHEAD"))
    assert cfg.outer_behavior_policy_compile_mode == "reduce-overhead"
    agent = _agent(outer_behavior_policy_compile_mode="reduce-overhead")
    assert agent._behavior_ce_compile_mode == "default"  # portable CPU use
    assert not agent._behavior_ce_region.enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for Inductor/CUDA graph validation")
def test_cuda_compiled_ce_backward_changes_masks_and_retains_metrics():
    agent = _agent(train_unroll_horizon=3)
    cfg = deepcopy(agent.cfg)
    cfg.device = "cuda"
    agent = AMBITDMPC2Agent(cfg)
    agent.cfg.action_dim = 3
    agent._behavior_ce_region.enabled = True
    agent._behavior_ce_region.strict = True
    agent._behavior_ce_region.mode = "reduce-overhead"
    agent._behavior_ce_compile_mode = "reduce-overhead"
    retained = []
    for mask in ("all", "partial", "none", "one", "all"):
        mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if mark is not None:
            mark()
        info, mean, log_std, valid = _inputs(agent, mask=mask, dtype=torch.float32)
        info = {key: value.detach().cuda().requires_grad_() for key, value in info.items()}
        inputs = (info, mean.detach().cuda(), log_std.detach().cuda(), valid.cuda())
        expected = agent._behavior_policy_action_ce_loss_reference(*inputs)
        expected_grads = torch.autograd.grad(expected[0], tuple(info.values()))
        actual = agent._behavior_policy_action_ce_loss(*inputs)
        actual_grads = torch.autograd.grad(actual[0], tuple(info.values()))
        _assert_close(actual, expected, rtol=3e-5, atol=2e-6)
        _assert_close(actual_grads, expected_grads, rtol=3e-5, atol=2e-6)
        retained.append((actual[2], {key: value.clone() for key, value in actual[2].items()}))
        del actual, actual_grads
    for metrics, saved in retained:
        _assert_close(metrics, saved, rtol=0, atol=0)
