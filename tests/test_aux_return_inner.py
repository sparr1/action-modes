"""Independent actor, initialization value, and continuation value routing."""

from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common.control_sources import actor_module, critic_module, source_metadata
from RL.tdmpc2_core.inner_trace import InnerActionTrace, evaluate_outer_tail
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params, _tiny_params


@contextmanager
def _prepared(**overrides):
    settings = dict(
        aux_return_mode="return_actor", inner_rounds=1, inner_updates_per_round=1,
        inner_temperature_mode="inherit_outer", ent_coef=.2, aux_return_ent_coef=.07,
        inner_actor_source="return_actor", inner_critic_source="aux_return",
        inner_horizon_actor_source="sac", inner_horizon_critic_source="sac",
        aux_return_log_std_min=-3., aux_return_log_std_max=-1.,
    )
    settings.update(overrides)
    if settings.get("inner_critic_adaptation") == "lora_rl":
        settings.setdefault("inner_critic_lora_rank", 2)
    settings = _tiny_params(**settings)
    if settings["inner_operator"] in {"none", "mppi"}:
        for key in ("inner_rounds", "inner_rollouts_per_round", "inner_updates_per_round"):
            settings.pop(key, None)
        if settings["inner_operator"] == "mppi":
            settings["inner_model_step_budget"] = (
                12 + settings["inner_mppi_num_pi_trajs"] * (settings["inner_rollout_horizon"] - 1)
            )
        settings["inner_temperature_mode"] = "fixed"
        settings["inner_temperature"] = 1.
    holder = _model_from_params(settings)
    try:
        agent = holder.agent
        engine = agent.inner_engine
        with torch.no_grad():
            actors = [(agent.model._pi, -.3)]
            if hasattr(agent.model, "_return_pi"):
                actors.append((agent.model._return_pi, .6))
            for actor, mean in actors:
                actor[-1].weight.zero_()
                actor[-1].bias.copy_(torch.tensor([mean, -2.]))
            for critic, value in ((agent.model._Qs, 2.), (agent.model._aux_return_Qs, 9.)):
                for head in critic:
                    head[-1].weight.zero_()
                    head[-1].bias.fill_(value)
            agent.model.soft_update_target_Q(1.)
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        yield holder, engine
    finally:
        holder.close()


@pytest.mark.parametrize("actor_source", ["sac", "return_actor"])
@pytest.mark.parametrize("critic_source", ["sac", "aux_return"])
@pytest.mark.parametrize("adaptation", ["clone", "frozen", "lora_rl"])
def test_initialization_and_repeated_action_reset_follow_selected_sources(
    actor_source, critic_source, adaptation,
):
    with _prepared(inner_actor_source=actor_source, inner_critic_source=critic_source,
                   inner_critic_adaptation=adaptation) as (holder, engine):
        z, action = torch.zeros(4, holder.cfg.latent_dim), torch.zeros(4, 1)
        expected_actor = actor_module(holder.agent, actor_source)
        expected_critic = critic_module(holder.agent, critic_source)
        for _ in range(2):
            torch.testing.assert_close(engine.state.actor(z), expected_actor(z), rtol=0, atol=0)
            actual = engine.model.Q(z, action, qs=engine.state.critic, reduction="all")
            expected = engine.model.Q(z, action, qs=expected_critic, reduction="all")
            torch.testing.assert_close(actual, expected)
            with torch.no_grad():
                next(engine.state.actor.parameters()).add_(1.)
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=False)
        expected_alpha = .2 if actor_source == "sac" else .07
        assert engine.alpha.item() == pytest.approx(expected_alpha)
        assert engine.state.actor is not expected_actor
        assert engine.state.critic is not expected_critic


@pytest.mark.parametrize("horizon_actor", ["sac", "return_actor"])
@pytest.mark.parametrize("horizon_critic", ["sac", "aux_return"])
def test_horizon_pair_is_independent_from_inner_pair_and_never_adds_entropy(
    horizon_actor, horizon_critic, monkeypatch,
):
    with _prepared(inner_horizon_actor_source=horizon_actor,
                   inner_horizon_critic_source=horizon_critic) as (_, engine):
        seen = []
        original = engine.model.Q
        def q(z, action, **kwargs):
            seen.append((action.clone(), kwargs["qs"]))
            return original(z, action, **kwargs)
        monkeypatch.setattr(engine.model, "Q", q)
        z = torch.zeros(4, engine.cfg.latent_dim)
        value = engine._prior_bootstrap(z, torch.zeros(4, 1))
        assert seen[0][1] is critic_module(engine.agent, horizon_critic)
        expected_action = torch.tensor(-.3 if horizon_actor == "sac" else .6).tanh()
        torch.testing.assert_close(seen[0][0], expected_action.expand(4, 1))
        torch.testing.assert_close(value, torch.full((4, 1), 2. if horizon_critic == "sac" else 9.))


def test_inner_target_uses_adapted_actor_and_only_boundary_uses_frozen_pair(monkeypatch):
    with _prepared(inner_horizon_critic_source="aux_return",
                   inner_sac_critic_target="reward_only") as (_, engine):
        with torch.no_grad():
            for q in engine.state.critic_target:
                q[-1].bias.fill_(5.)
        actors = []
        original = engine.model.pi
        def pi(z, **kwargs):
            actors.append(kwargs.get("policy"))
            return original(z, **kwargs)
        monkeypatch.setattr(engine.model, "pi", pi)
        z = torch.zeros(4, engine.cfg.latent_dim)
        output = engine._sac_critic_kernel(
            z, torch.zeros(4, 1), torch.ones(4, 1), z,
            torch.tensor([[0.], [0.], [1.], [1.]]), torch.tensor(.3),
            torch.zeros(4, 1), None, torch.tensor([[0.], [1.], [0.], [1.]]),
            torch.zeros(4, 1),
        )
        assert actors == [engine.state.actor, engine.model._pi]
        expected = torch.tensor([[1.+.99*5], [1.+.99*9], [1.], [1.]])
        torch.testing.assert_close(output[2], expected)
        output[0].backward()
        assert all(p.grad is None for p in engine.model.parameters())


@pytest.mark.parametrize("operator,extra", [
    ("none", {}), ("sac", {"inner_rounds": 0}),
    ("mppi", {"inner_mppi_iterations": 1}),
])
def test_prior_only_and_zero_update_fallback_execute_selected_actor(operator, extra):
    with _prepared(inner_operator=operator, **extra) as (holder, engine):
        if operator == "mppi":
            # The engine's zero-iteration compatibility fallback is also used
            # by callers that turn off planning after constructing a controller.
            engine.cfg.inner_mppi_iterations = 0
        before = _clone_tree(engine.model.state_dict())
        rng = torch.random.get_rng_state().clone()
        action = holder.agent.act(torch.zeros(3), eval_mode=True, collect_diagnostics=False)
        torch.testing.assert_close(action, torch.tensor([.6]).tanh())
        _assert_tree_equal(engine.model.state_dict(), before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_selected_actor_temperature_and_selected_critic_scale_are_independent():
    with _prepared(inner_actor_source="sac", aux_return_sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                   sac_actor_loss_scale_mode="none", aux_return_ent_coef=.01) as (holder, engine):
        holder.agent.aux_return.actor_loss_scale.fill_(7.)
        holder.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert holder.agent.last_inner_metrics["inner_actor_loss_scale"] == 7.
        assert holder.agent.last_inner_metrics["inner_alpha_initial"] == pytest.approx(.2)
        assert holder.agent.aux_return.actor_loss_scale.item() == 7.


def test_sac_bootstrap_auxiliary_critic_needs_no_return_actor():
    with _prepared(aux_return_mode="sac", inner_actor_source="sac",
                   inner_horizon_critic_source="aux_return") as (holder, engine):
        assert not hasattr(engine.model, "_return_pi")
        holder.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert holder.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 1
        assert engine._critic_base is engine.model._aux_return_Qs


def test_auxiliary_target_initialization_uses_its_own_outer_target():
    with _prepared(inner_critic_target_initialization="outer_target") as (_, engine):
        with torch.no_grad():
            for q in engine.model._target_aux_return_Qs:
                q[-1].bias.fill_(13.)
        for _ in range(2):
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=False)
            z, a = torch.zeros(2, engine.cfg.latent_dim), torch.zeros(2, 1)
            actual = engine.model.Q(z, a, qs=engine.state.critic_target, reduction="all")
            torch.testing.assert_close(actual, torch.full((2, 2, 1), 13.))


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_inner_actor_improves_through_selected_critic_without_outer_gradients(representation):
    with _prepared(q_representation=representation, inner_entropy_enabled=False) as (_, engine):
        with torch.no_grad():
            for q in engine.state.critic:
                q[-1].weight.normal_(std=.1)
        output = engine._sac_actor_kernel(
            torch.randn(4, engine.cfg.latent_dim), torch.tensor(.07),
            torch.zeros(4, 1), None, True,
        )
        output[2].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in engine.state.actor_params)
        assert all(p.grad is None for p in engine.state.critic.parameters())
        assert all(p.grad is None for p in engine.model.parameters())


def test_inner_scaled_temperature_inherits_selected_actor_state_and_numeric_target():
    with _prepared(
        aux_return_outer_actor_entropy_mode="tdmpc2_scaled",
        aux_return_ent_coef="auto_0.3", aux_return_target_entropy=-.7,
        inner_temperature_mode="auto", inner_temperature_initialization="inherit_outer",
        inner_target_entropy="inherit_outer",
    ) as (holder, engine):
        assert engine.alpha.item() == pytest.approx(.3)
        assert engine._resolved_inner_target_entropy() == pytest.approx(-.7)
        before = holder.agent.aux_return.alpha.detach().clone()
        assert torch.isfinite(holder.agent.act(torch.zeros(3))).all()
        torch.testing.assert_close(holder.agent.aux_return.alpha, before, rtol=0, atol=0)


@pytest.mark.parametrize("probe_mode", ["legacy", "outer_tail"])
def test_routed_trace_and_action_preserve_outer_weights_scales_and_rng(probe_mode):
    with _prepared(inner_horizon_actor_source="return_actor",
                   inner_horizon_critic_source="aux_return") as (holder, engine):
        before = _clone_tree(engine.model.state_dict())
        rng = torch.random.get_rng_state().clone()
        trace = InnerActionTrace(probes=True, probe_mode=probe_mode, probe_rollouts=2,
                                 probe_horizon=2, capture_actors=True)
        holder.agent.act(torch.zeros(3), trace=trace)
        assert any(item["phase"] == "probe" for item in trace.events)
        for event in trace.events:
            routing = event["value_routing"]
            assert routing["inner_horizon_actor_source"] == "return_actor"
            assert routing["horizon_critic_continuation_actor"] == "return_actor"
            assert routing["horizon_critic_target"] == "reward_only"
            assert routing["terminal_entropy_bonus"] is False
        _assert_tree_equal(engine.model.state_dict(), before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        assert trace.actor_snapshots[0].policy_bounds["log_std_max"] == -1.


def test_outer_tail_probe_uses_selected_horizon_critic():
    with _prepared(inner_horizon_critic_source="aux_return") as (_, engine):
        result = evaluate_outer_tail(
            engine, torch.zeros(1, engine.cfg.latent_dim), engine.state.actor,
            torch.zeros(3, 2, 1), policy_bounds=engine._inner_policy_kwargs(),
        )
        torch.testing.assert_close(result["bootstrap"], torch.full((2, 1), 9.*.99**2))


def test_inner_resume_roundtrip_and_source_mismatch_are_transactional():
    with _prepared() as (holder, engine):
        holder.agent.act(torch.zeros(3), collect_diagnostics=False)
        engine.prepare_training_resume_boundary()
        before = _clone_tree(engine.training_state_dict())
        assert before["version"] == 6
        assert before["control_sources"] == source_metadata(engine.cfg)
        engine.load_training_state_dict(before)
        _assert_tree_equal(engine.training_state_dict(), before)
        invalid = deepcopy(before)
        invalid["control_sources"]["inner_critic_source"] = "sac"
        with pytest.raises(ValueError, match="control sources"):
            engine.load_training_state_dict(invalid)
        _assert_tree_equal(engine.training_state_dict(), before)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("component", ["actor", "critic"])
def test_auxiliary_inner_compilation_matches_values_and_gradients(backend, component):
    with _prepared() as (_, engine):
        z = torch.randn(4, engine.cfg.latent_dim)
        args = (z, torch.zeros(4, 1), torch.ones(4, 1), z,
                torch.zeros(4, 1), torch.tensor(.07), torch.zeros(4, 1), None,
                torch.tensor([[0.], [1.], [0.], [1.]]), torch.zeros(4, 1))
        kernel, params, loss_index = engine._sac_critic_kernel, engine.state.critic_params, 0
        if component == "actor":
            kernel, params, loss_index = engine._sac_actor_kernel, engine.state.actor_params, 2
            args = (z, torch.tensor(.07), torch.zeros(4, 1), None, True)
        eager = kernel(*args)
        gradients = torch.autograd.grad(eager[loss_index], params)
        compiled = torch.compile(kernel, backend=backend, fullgraph=True)(*args)
        torch.testing.assert_close(compiled, eager)
        actual = torch.autograd.grad(compiled[loss_index], params)
        for expected, value in zip(gradients, actual):
            torch.testing.assert_close(value, expected)
