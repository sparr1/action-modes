"""Retrace resume semantics, private RNG ownership, and compiled solve parity."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def models():
    opened = []

    def create(**overrides):
        options = dict(
            inner_sac_return_estimator="retrace", inner_finite_horizon=True,
            inner_rollout_horizon=3, train_unroll_horizon=3,
            inner_rounds=1, inner_rollouts_per_round=2,
            inner_updates_per_round=2, inner_replay_capacity=6,
        )
        options.update(overrides)
        if options.get("inner_critic_adaptation") == "lora_rl":
            options.setdefault("inner_critic_lora_rank", 4)
        holder = _tiny_model(**options)
        opened.append(holder)
        return holder.agent

    yield create
    for holder in opened:
        holder.env.close()


def _snapshot(engine):
    pool = engine._action_pool
    result = {}
    for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim",
                 "temperature_optim", "replay"):
        value = getattr(pool, name)
        result[name] = None if value is None else deepcopy(value.state_dict())
    result["log_alpha"] = None if pool.log_alpha is None else pool.log_alpha.detach().clone()
    result["alpha_fixed"] = None if pool.alpha_fixed is None else pool.alpha_fixed.detach().clone()
    result["rng"] = engine.rng.training_state_dict()
    return result


def test_zero_work_retrace_solve_allocates_valid_empty_replay_and_returns_finite_action(models):
    agent = models(inner_rollouts_per_round=0, inner_updates_per_round=0,
                   inner_replay_capacity=None)
    before = deepcopy(agent.model.state_dict())
    action = agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    assert torch.isfinite(action).all()
    assert agent.cfg.inner_replay_capacity == agent.cfg.inner_rollout_horizon
    _assert_tree_equal(agent.model.state_dict(), before)
    replay = agent.inner_engine._action_pool.replay
    assert replay.size == 0
    assert replay.trajectory_count == 0


@pytest.mark.parametrize("options", [
    {},
    {"inner_retrace_value_samples": 4, "inner_retrace_boundary_value_samples": 3},
    {"inner_actor_initialization": "random", "inner_critic_initialization": "random",
     "inner_critic_adaptation": "lora_rl"},
    {"aux_return_mode": "return_actor", "inner_actor_source": "return_actor",
     "inner_critic_source": "aux_return", "inner_horizon_critic_source": "aux_return"},
])
def test_active_schema8_roundtrip_reproduces_next_solve(models, options):
    source = models(**options)
    source.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict())
    inner = payload["inner"]
    assert inner["version"] == 8
    assert inner["retrace_spec"] == source.inner_engine._retrace_spec()
    for key in ("actor", "critic", "critic_target", "replay"):
        assert inner["workspace"][key] is None

    direct = models(**options).inner_engine
    direct.load_training_state_dict(deepcopy(inner))
    _assert_tree_equal(direct.training_state_dict(), inner)

    restored = models(**options)
    global_rng = torch.random.get_rng_state().clone()
    restored.load_training_state_dict(payload)
    _assert_tree_equal(restored.training_state_dict(), payload)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    source.reset()
    restored.reset()
    for observation in (torch.tensor([0.2, -0.1, 0.3]), torch.zeros(3)):
        expected = source.act(observation, collect_diagnostics=False)
        actual = restored.act(observation, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_snapshot(restored.inner_engine), _snapshot(source.inner_engine))
    source.prepare_training_resume_boundary()
    restored.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("change", [
    {"inner_sac_return_estimator": "one_step"},
    {"inner_retrace_lambda": 0.5},
    {"inner_retrace_batch_trajectories": 3},
    {"inner_retrace_value_samples": 4},
    {"inner_retrace_boundary_value_samples": 3},
    {"inner_rollout_horizon": 1},
    {"inner_actor_initialization": "random"},
])
def test_changed_retrace_protocol_fails_before_state_or_rng_mutation(models, direct, change):
    source, target = models(), models(**change)
    if direct:
        source, target = source.inner_engine, target.inner_engine
    saved = deepcopy(source.training_state_dict())
    before = deepcopy(target.training_state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        target.load_training_state_dict(saved)
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing_spec", "boolean_lambda", "extra_key", "old_schema"])
def test_retrace_metadata_corruption_is_rejected_atomically(models, corruption):
    engine = models().inner_engine
    before = deepcopy(engine.training_state_dict())
    saved = deepcopy(before)
    if corruption == "missing_spec":
        del saved["retrace_spec"]
    elif corruption == "boolean_lambda":
        saved["retrace_spec"]["lambda"] = True
    elif corruption == "extra_key":
        saved["retrace_spec"]["extra"] = "unexpected"
    else:
        saved["version"] = 1
        saved.pop("retrace_spec")
    with pytest.raises((ValueError, TypeError)):
        engine.load_training_state_dict(saved)
    _assert_tree_equal(engine.training_state_dict(), before)


@pytest.mark.parametrize("source_active", [False, True])
def test_portable_loading_between_estimators_leaves_outer_frozen_during_next_solve(models, source_active):
    source = models(inner_sac_return_estimator="retrace" if source_active else "one_step")
    target = models(inner_sac_return_estimator="one_step" if source_active else "retrace")
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.add_(0.01)
    target.load(deepcopy(source.checkpoint_state()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    before = deepcopy(target.model.state_dict())
    action = target.act(torch.zeros(3), collect_diagnostics=False)
    assert torch.isfinite(action).all()
    _assert_tree_equal(target.model.state_dict(), before)
    assert all(parameter.grad is None for parameter in target.model.parameters())


@pytest.mark.parametrize("options", [
    {"q_representation": "scalar"},
    {"q_representation": "distributional", "num_q": 3},
    {"q_representation": "scalar", "inner_retrace_value_samples": 4,
     "inner_retrace_boundary_value_samples": 3},
    {"q_representation": "distributional", "num_q": 3,
     "inner_retrace_value_samples": 4, "inner_retrace_boundary_value_samples": 3},
    {"q_representation": "scalar", "inner_actor_initialization": "random",
     "inner_critic_initialization": "random", "inner_critic_adaptation": "lora_rl"},
])
def test_compiled_retrace_matches_eager_and_reuses_rollout_and_critic_graphs(models, monkeypatch, options):
    torch._dynamo.reset()
    real_compile = torch.compile
    regions, graphs = [], []
    # The locked torch version cannot fullgraph-capture the existing detached
    # LoRA actor evaluation; retain that path's established non-strict contract.
    strict = options.get("inner_critic_adaptation") != "lora_rl"

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def compile_graph(function, **kwargs):
        regions.append(function.__name__)
        assert kwargs["fullgraph"] is strict
        return real_compile(function, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_graph)
    eager = models(dropout=0.2, **options)
    compiled = models(dropout=0.2, compile=True, compile_strict=strict, **options)
    before = deepcopy(compiled.model.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    try:
        for iteration in range(2):
            if iteration:
                for agent in (eager, compiled):
                    agent.inner_engine.reset_for_evaluation(909, reuse_action_pool=True)
            expected = eager.act(torch.zeros(3), t0=True, collect_diagnostics=False)
            actual = compiled.act(torch.zeros(3), t0=True, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(compiled.inner_engine), _snapshot(eager.inner_engine))
            _assert_tree_equal(compiled.model.state_dict(), before)
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
            assert compiled.last_inner_metrics["inner_compile_fallback"] == 0.0
            assert all(parameter.grad is None for parameter in compiled.model.parameters())
            if iteration == 0:
                first_graph_count = len(graphs)
                assert first_graph_count > 0
            else:
                assert len(graphs) == first_graph_count
        assert {"_retrace_dense_rollout_kernel", "_retrace_critic_kernel", "_sac_actor_kernel"} <= set(regions)
    finally:
        torch._dynamo.reset()


def _prepare(agent):
    engine = agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def _rollout_arguments(engine):
    cfg = engine.cfg
    root = torch.linspace(-0.2, 0.3, cfg.latent_dim).reshape(1, -1)
    noise = torch.linspace(-0.7, 0.8, cfg.inner_rollouts_per_round * cfg.inner_rollout_horizon * cfg.action_dim)
    noise = noise.reshape(cfg.inner_rollout_horizon, cfg.inner_rollouts_per_round, cfg.action_dim)
    from RL.tdmpc2_core.common import math as td_math
    return root, noise, td_math.categorical_support(root, cfg)


@pytest.mark.parametrize("representation,samples,boundary_samples", [
    ("scalar", 1, 1), ("distributional", 1, 1), ("distributional", 4, 3),
])
def test_cpu_inductor_retrace_critic_targets_loss_and_gradients_match_eager(models, representation, samples, boundary_samples):
    from torch._inductor import config as inductor_config

    torch._dynamo.reset()
    options = dict(q_representation=representation,
                   num_q=3 if representation == "distributional" else 2,
                   mppi_terminal_q_reduction="mean_all",
                   inner_retrace_value_samples=samples,
                   inner_retrace_boundary_value_samples=boundary_samples)
    agents = [models(**options) for _ in range(2)]
    for agent in agents:
        with torch.no_grad():
            for index, head in enumerate(agent.model._Qs):
                # Nonzero heads exercise gradients through the hidden critic
                # as well as the final prediction layer.
                head[-1].weight.copy_(torch.linspace(
                    -0.1, 0.15, head[-1].weight.numel(),
                ).reshape_as(head[-1].weight) + index * 0.01)
    eager, compiled = [_prepare(agent) for agent in agents]
    with torch.no_grad():
        batch = eager._retrace_dense_rollout_kernel(*_rollout_arguments(eager))
    noise_shape = (*batch["action"].shape[:2], samples, batch["action"].shape[-1])
    noise = torch.linspace(-0.9, 0.4, batch["action"].numel() * samples).reshape(noise_shape)
    prior_noise = torch.linspace(-0.5, 0.3, batch["action"].shape[0] * boundary_samples * batch["action"].shape[-1])
    prior_noise = prior_noise.reshape(batch["action"].shape[0], boundary_samples, batch["action"].shape[-1])
    if samples == 1:
        noise = noise.squeeze(2)
    if boundary_samples == 1:
        prior_noise = prior_noise.squeeze(1)
    pair = torch.tensor([0, 2]) if representation == "distributional" else None
    alpha = torch.tensor(0.2, requires_grad=True)
    arguments = (batch, alpha, noise, prior_noise, pair)
    try:
        expected = eager._retrace_critic_kernel(*arguments)
        # A single compile worker avoids unsafe subprocess forks in a
        # multithreaded macOS pytest process; no CUDA behavior is asserted.
        with inductor_config.patch(compile_threads=1):
            kernel = torch.compile(compiled._retrace_critic_kernel, backend="inductor",
                                   fullgraph=True, dynamic=False)
            actual = kernel(*arguments)
            expected[0].backward()
            actual[0].backward()
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=5e-5, atol=5e-6)
        for value, reference in zip(compiled.state.critic_params, eager.state.critic_params):
            torch.testing.assert_close(value.grad, reference.grad, rtol=1e-4, atol=5e-6)
        assert alpha.grad is None
        assert all(parameter.grad is None for parameter in compiled.model.parameters())
        assert all(parameter.grad is None for parameter in compiled.state.actor.parameters())
        assert all(parameter.grad is None for parameter in compiled.state.critic_target.parameters())
    finally:
        torch._dynamo.reset()


def test_cpu_inductor_dense_retrace_rollout_matches_eager_stored_densities(models):
    from torch._inductor import config as inductor_config

    torch._dynamo.reset()
    eager, compiled = [_prepare(models(inner_behavior_std_scale=0.4)) for _ in range(2)]
    arguments = _rollout_arguments(eager)
    before = torch.random.get_rng_state().clone()
    try:
        with torch.no_grad(), inductor_config.patch(compile_threads=1):
            expected = eager._retrace_dense_rollout_kernel(*arguments)
            kernel = torch.compile(compiled._retrace_dense_rollout_kernel, backend="inductor",
                                   fullgraph=True, dynamic=False)
            actual = kernel(*arguments)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=3e-5, atol=3e-6)
        assert {"pre_tanh_action", "behavior_log_prob", "valid", "horizon_end"} <= actual.keys()
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("options,version", [
    ({}, 1),
    ({"inner_actor_initialization": "random", "inner_critic_initialization": "random",
      "inner_critic_adaptation": "lora_rl"}, 4),
    ({"aux_return_mode": "return_actor", "inner_critic_source": "aux_return"}, 6),
])
def test_disabled_retrace_preserves_legacy_schema_draws_and_optimizer_results(models, options, version):
    implicit = models(inner_sac_return_estimator="one_step", **options)
    explicit = models(inner_sac_return_estimator="ONE_STEP", inner_retrace_lambda=0.2,
                      inner_retrace_batch_trajectories=17, **options)
    for agent in (implicit, explicit):
        state = agent.inner_engine.training_state_dict()
        assert state["version"] == version
        assert "retrace_spec" not in state
        assert agent.inner_engine._compile_regions["rollout"].eager.__name__ == "_dense_rollout_kernel"
        assert agent.inner_engine._compile_regions["critic"].eager.__name__ == "_sac_critic_kernel"
    _assert_tree_equal(implicit.training_state_dict(), explicit.training_state_dict())
    rng = torch.random.get_rng_state().clone()
    for observation in (torch.zeros(3), torch.tensor([0.1, -0.2, 0.3])):
        expected = implicit.act(observation, collect_diagnostics=False)
        actual = explicit.act(observation, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_snapshot(explicit.inner_engine), _snapshot(implicit.inner_engine))
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for agent in (implicit, explicit):
        agent.prepare_training_resume_boundary()
    _assert_tree_equal(implicit.training_state_dict(), explicit.training_state_dict())
