"""Finite-horizon SAC routing, lifecycle, and strict compilation."""

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
            inner_horizon_conditioning="one_hot", inner_finite_horizon=True,
            inner_rollout_horizon=3, train_unroll_horizon=3,
            inner_rounds=1, inner_updates_per_round=2, inner_replay_capacity=24,
            inner_q_target_reduction="min_all", mppi_terminal_q_reduction="mean_all",
        )
        options.update(overrides)
        model = _tiny_model(**options)
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.env.close()


def _prepare(model):
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("entropy", ["reward_only", "entropy_augmented"])
def test_self_loop_bellman_targets_use_remaining_time(models, monkeypatch, representation, entropy):
    model = models(q_representation=representation, inner_sac_critic_target=entropy)
    engine = _prepare(model)
    gamma, reward, tail, alpha = float(model.agent.discount), 2., 7., .25
    h = torch.tensor([[3], [2], [1], [3], [1]])
    z = torch.zeros(5, model.cfg.latent_dim)
    terminal = torch.tensor([[0.], [0.], [0.], [1.], [1.]])
    calls = []

    def pi(next_z, **kwargs):
        steps = kwargs.get("remaining_horizon")
        calls.append((kwargs.get("policy"), steps))
        action = next_z.new_zeros(5, model.cfg.action_dim)
        if steps is not None:
            action += steps / 10.
        return action, {"log_prob": next_z.new_full((5, 1), -1.)}

    def q(next_z, action, **kwargs):
        steps = kwargs.get("remaining_horizon")
        if steps is None:
            assert "qs" not in kwargs  # Frozen outer boundary.
            return next_z.new_full((5, 1), tail)
        assert kwargs["qs"] is engine.state.critic_target
        torch.testing.assert_close(action[:, :1], steps / 10.)
        # Exact reward-only finite-horizon critic for a deterministic self-loop.
        result = reward * (1. - gamma ** steps.float()) / (1. - gamma) + gamma ** steps.float() * tail
        result[3:] = float("nan")  # True terminals must discard unused values.
        return result

    monkeypatch.setattr(engine.model, "pi", pi)
    monkeypatch.setattr(engine.model, "Q", q)
    original_predictions = engine.model.q_predictions
    seen = []

    def predictions(z, action, **kwargs):
        seen.append(kwargs["remaining_horizon"].clone())
        return original_predictions(z, action, **kwargs)

    monkeypatch.setattr(engine.model, "q_predictions", predictions)
    outputs = engine._sac_critic_kernel(
        z, z.new_zeros(5, model.cfg.action_dim), z.new_full((5, 1), reward), z,
        terminal, torch.tensor(alpha), z.new_zeros(5, model.cfg.action_dim), None,
        (h == 1).float(), z.new_zeros(5, model.cfg.action_dim), remaining_horizon=h,
    )
    expected = reward * (1. - gamma ** h.float()) / (1. - gamma) + gamma ** h.float() * tail
    if entropy == "entropy_augmented":
        expected += gamma * alpha * (h > 1)
    expected[3:] = reward
    torch.testing.assert_close(outputs[2], expected, atol=2e-5, rtol=2e-5)
    assert len(torch.unique(outputs[2][:3])) == 3
    torch.testing.assert_close(seen[0], h)
    assert calls[0][0] is engine.state.actor
    torch.testing.assert_close(calls[0][1], (h - 1).clamp_min(1))
    assert calls[1] == (None, None)


@pytest.mark.parametrize("timing", ["round", "step"])
@pytest.mark.parametrize("episodic", [False, True])
@pytest.mark.parametrize("horizon", [1, 3])
def test_collection_updates_and_root_execution_route_horizon(models, monkeypatch, timing, episodic, horizon):
    options = dict(inner_update_timing=timing, episodic=episodic, inner_rollout_horizon=horizon)
    if timing == "step":
        options.update(inner_steps_per_update=2, inner_updates_per_round=None)
    model = models(**options)
    engine = model.agent.inner_engine
    if episodic:
        def termination(z):
            done = z.new_zeros(len(z), 1)
            if len(z) == 2:
                done[0] = 1.
            return done
        monkeypatch.setattr(engine.model, "termination", termination)
    calls = []
    original = engine._policy_action

    def record(z, policy, **kwargs):
        calls.append(kwargs.get("remaining_horizon").clone())
        return original(z, policy, **kwargs)

    monkeypatch.setattr(engine, "_policy_action", record)
    before = deepcopy(engine.model.state_dict())
    action = model.agent.act(torch.zeros(3))  # Includes root diagnostics.
    assert torch.isfinite(action).all()
    _assert_tree_equal(before, engine.model.state_dict())
    assert calls[-1].tolist() == [[horizon]]
    replay = engine._action_pool.replay
    expected = [horizon, horizon]
    for h in range(horizon - 1, 0, -1):
        expected.extend([h] if episodic else [h, h])
    assert replay.remaining_horizon[:replay.size, 0].tolist() == expected
    torch.testing.assert_close(replay.horizon_end[:replay.size].bool(), replay.remaining_horizon[:replay.size] == 1)
    assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] > 0


@pytest.mark.parametrize("scaled", [False, True])
def test_actor_loss_routes_same_h_and_keeps_outer_anchor_unconditioned(models, monkeypatch, scaled):
    options = dict(sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                   ent_coef=.2, inner_temperature_mode="inherit_outer") if scaled else {}
    model = models(inner_outer_policy_kl_coef=.1, **options)
    engine = _prepare(model)
    h = torch.tensor([[3], [1], [2], [3]])
    z = torch.randn(4, model.cfg.latent_dim)
    calls = []
    pi, q = engine.model.pi, engine.model.Q

    def record_pi(z, **kwargs):
        calls.append(("pi", kwargs.get("policy"), kwargs.get("remaining_horizon")))
        return pi(z, **kwargs)

    def record_q(z, action, **kwargs):
        calls.append(("q", kwargs.get("qs"), kwargs.get("remaining_horizon")))
        return q(z, action, **kwargs)

    monkeypatch.setattr(engine.model, "pi", record_pi)
    monkeypatch.setattr(engine.model, "Q", record_q)
    engine._sac_policy_step(
        {"z": z, "remaining_horizon": h}, update_temperature=not scaled, update_actor=True,
        alpha=torch.tensor(.2), actor_loss_scale=torch.tensor([2.]) if scaled else None,
    )
    assert [(kind, module is engine.state.actor, module is engine.state.critic) for kind, module, _ in calls[:2]] == [
        ("pi", True, False), ("q", False, True),
    ]
    for _, _, steps in calls[:2]:
        torch.testing.assert_close(steps, h)
    assert calls[2][1] is engine.state.actor_anchor and calls[2][2] is None
    assert all(p.grad is None for p in engine.state.critic.parameters())
    assert engine.state.actor[0].weight.grad[:, -3:].abs().sum() > 0


@pytest.mark.parametrize("target_init", ["online", "outer_target"])
def test_fresh_solve_restores_prior_zeroes_columns_and_reuses_parameters(models, target_init):
    model = models(inner_critic_target_initialization=target_init)
    engine = _prepare(model)
    initial = {name: deepcopy(getattr(engine.state, name).state_dict())
               for name in ("actor", "critic", "critic_target")}
    ids = [id(p) for p in engine.state.actor.parameters()]
    engine._collect_round(torch.zeros(1, model.cfg.latent_dim))
    engine._sac_policy_step(engine.state.replay.sample(4), update_actor=True,
                            update_temperature=True, alpha=torch.tensor(.2))
    assert engine.state.actor[0].weight[:, -3:].abs().sum() > 0
    engine._prepare_workspace(t0=False)
    assert ids == [id(p) for p in engine.state.actor.parameters()]
    for name, expected in initial.items():
        _assert_tree_equal(getattr(engine.state, name).state_dict(), expected)
    engine.state.critic[0][0].weight.data[:, -3:].fill_(2.)
    from RL.tdmpc2_core.inner_improvement import polyak_update
    polyak_update(engine.state.critic, engine.state.critic_target, .25)
    torch.testing.assert_close(engine.state.critic_target[0][0].weight[:, -3:], torch.full_like(engine.state.critic_target[0][0].weight[:, -3:], .5))


def test_conditioned_checkpoint_restores_next_solve_and_portable_prior(models):
    source, restored = models(), models()
    source.agent.act(torch.zeros(3), collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    payload = deepcopy(source.agent.training_state_dict())
    assert payload["inner"]["version"] == 7
    restored.agent.load_training_state_dict(payload)
    source.agent.reset()
    restored.agent.reset()
    torch.testing.assert_close(
        source.agent.act(torch.zeros(3), collect_diagnostics=False),
        restored.agent.act(torch.zeros(3), collect_diagnostics=False), rtol=0, atol=0,
    )
    prior = models(inner_horizon_conditioning="none")
    restored.agent.load(deepcopy(prior.agent.checkpoint_state()))
    _assert_tree_equal(restored.agent.model.state_dict(), prior.agent.model.state_dict())
    assert torch.isfinite(restored.agent.act(torch.zeros(3), collect_diagnostics=False)).all()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("change", ["horizon", "off", "from_off"])
def test_incompatible_resume_is_rejected_before_mutation(models, direct, change):
    source = models(inner_horizon_conditioning="none" if change == "from_off" else "one_hot")
    target = models(inner_horizon_conditioning="none" if change == "off" else "one_hot",
                    inner_rollout_horizon=2 if change == "horizon" else 3)
    a = source.agent.inner_engine if direct else source.agent
    b = target.agent.inner_engine if direct else target.agent
    before = deepcopy(b.training_state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        b.load_training_state_dict(deepcopy(a.training_state_dict()))
    _assert_tree_equal(before, b.training_state_dict())
    torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("conditioning", ["none", "one_hot"])
@pytest.mark.parametrize("timing", ["round", "step"])
def test_strict_compilation_reuses_graphs_across_fresh_solves(models, monkeypatch, timing, diagnostics, conditioning):
    torch._dynamo.reset()
    real_compile, graphs = torch.compile, []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(torch, "compile", lambda fn, **kw: real_compile(fn, backend=backend, **kw))
    options = dict(compile=True, compile_strict=True, inner_update_timing=timing,
                   inner_horizon_diagnostics=diagnostics, inner_horizon_conditioning=conditioning)
    if timing == "step":
        options.update(inner_steps_per_update=2, inner_updates_per_round=None)
    model = models(**options)
    try:
        rng = torch.random.get_rng_state().clone()
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        count = len(graphs)
        assert count > 0
        for _ in range(2):
            model.agent.act(torch.zeros(3), collect_diagnostics=False)
            assert len(graphs) == count
        assert model.agent.last_inner_metrics["inner_compile_fallback"] == 0
        torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_inductor_conditioned_critic_loss_and_gradients_match_eager(models, representation):
    torch._dynamo.reset()
    eager, compiled = [models(q_representation=representation) for _ in range(2)]
    engines = [_prepare(m) for m in (eager, compiled)]
    z = torch.randn(4, eager.cfg.latent_dim)
    h = torch.tensor([[3], [2], [1], [1]])
    args = (z, torch.zeros(4, eager.cfg.action_dim), torch.ones(4, 1), z,
            torch.tensor([[0.], [0.], [0.], [1.]]), torch.tensor(.2),
            torch.zeros(4, eager.cfg.action_dim), None, (h == 1).float(),
            torch.zeros(4, eager.cfg.action_dim))
    try:
        expected = engines[0]._sac_critic_kernel(*args, remaining_horizon=h)
        kernel = torch.compile(engines[1]._sac_critic_kernel, backend="inductor", fullgraph=True)
        actual = kernel(*args, remaining_horizon=h)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        expected[0].backward()
        actual[0].backward()
        for a, b in zip(engines[1].state.critic_params, engines[0].state.critic_params):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=3e-6)
    finally:
        torch._dynamo.reset()
