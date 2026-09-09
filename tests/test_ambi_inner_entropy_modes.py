"""Consistent actor, temperature, and critic entropy selection in inner SAC."""

import math

import pytest
import torch

from tests.test_ambi_inner_decoupling import _model
from tests.test_ambi_random_explorer_engine_invariants import _explorer_model


def test_inner_entropy_metric_catalog_distinguishes_statistic_and_objective():
    from RL.tdmpc2_core.inner_trace import metric_catalog

    catalog = metric_catalog()
    assert catalog["actor_entropy"]["unit"] == "nats"
    for prefix in ("", "explorer_"):
        entropy = catalog[f"{prefix}actor_scaled_entropy"]
        contribution = catalog[f"{prefix}actor_entropy_bonus"]
        assert entropy["unit"] == "scaled_entropy_statistic"
        assert contribution["unit"] == "objective"
        for entry in (entropy, contribution):
            assert entry["sampling_phase"] == "pre_update_minibatch"
            assert entry["preferred_axis"] == "actor_updates"
            assert not entry["definition"].startswith("Raw")


def _prepare(model):
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def _controlled_policy(engine, parameters, requests):
    def policy(z, **kwargs):
        requested = kwargs.get("include_scaled_entropy", False)
        requests.append(requested)
        action = parameters[0].expand(z.shape[0], engine.cfg.action_dim)
        log_prob = parameters[1].expand(z.shape[0], 1)
        info = {"log_prob": log_prob, "entropy": -log_prob}
        if requested:
            info["scaled_entropy"] = parameters[2].expand_as(log_prob)
        info["kl"] = parameters[3].expand_as(log_prob)
        return action, info

    return policy


def _action_q(engine):
    def critic(z, action, **kwargs):
        return action[:, :1].unsqueeze(0).expand(engine.cfg.num_q, -1, -1)

    return critic


def test_default_inner_kernel_preserves_legacy_loss_gradients_and_rng():
    model = _model(inner_outer_policy_kl_coef=0.0)
    try:
        engine = _prepare(model)
        z = torch.linspace(-1, 1, 4 * model.cfg.latent_dim).reshape(4, -1)
        noise = torch.linspace(-1, 1, 4).reshape(4, 1)
        pair = torch.tensor([0, 1])
        alpha = torch.tensor(0.17)
        rng = torch.random.get_rng_state().clone()
        outputs = engine._sac_actor_kernel(z, alpha, noise, pair, True)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(outputs[2], engine.state.actor_params)

        action, info = engine.model.pi(
            z, policy=engine.state.actor, noise=noise, **engine._inner_policy_kwargs()
        )
        q_all = engine.model.Q(
            z, action, qs=engine.state.critic, detach=True, reduction="all"
        )
        q_value = engine.model.q_backend.reduce(
            q_all, model.cfg.inner_q_actor_reduction,
            pair_indices=pair, trusted_pair_indices=True,
        )
        expected_loss = (alpha * info["log_prob"] - q_value).mean()
        expected_grad = torch.autograd.grad(expected_loss, engine.state.actor_params)
        assert len(outputs) == 10
        torch.testing.assert_close(outputs[0], info["log_prob"], rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], info["entropy"], rtol=0, atol=0)
        torch.testing.assert_close(outputs[2], expected_loss, rtol=0, atol=0)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        model.env.close()


@pytest.mark.parametrize("saturated", [False, True])
def test_scaled_inner_kernel_matches_literal_entropy_and_gradients(saturated):
    model = _model(inner_actor_entropy_mode="tdmpc2_scaled")
    try:
        engine = _prepare(model)
        if saturated:
            with torch.no_grad():
                engine.state.actor[-1].bias[0] = 20.0
        z = torch.linspace(-1, 1, 4 * model.cfg.latent_dim).reshape(4, -1)
        noise = torch.linspace(-1.3, 0.8, 4).reshape(4, 1)
        pair = torch.tensor([0, 1])
        alpha = torch.tensor(0.17)
        scale = torch.tensor(3.0, requires_grad=True)
        rng = torch.random.get_rng_state().clone()
        outputs = engine._sac_actor_kernel(z, alpha, noise, pair, True, q_scale=scale)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(outputs[2], engine.state.actor_params)

        action, info = engine.model.pi(
            z, policy=engine.state.actor, noise=noise, **engine._inner_policy_kwargs()
        )
        gaussian = (-0.5 * noise.square() - info["log_std"] - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        upstream_log_prob = gaussian - torch.log(torch.relu(1 - action.square()) + 1e-6).sum(-1, keepdim=True)
        expected_entropy = -upstream_log_prob * (
            model.cfg.action_dim * gaussian / (upstream_log_prob + 1e-8)
        )
        q_all = engine.model.Q(z, action, qs=engine.state.critic, detach=True, reduction="all")
        q_value = engine.model.q_backend.reduce(
            q_all, model.cfg.inner_q_actor_reduction,
            pair_indices=pair, trusted_pair_indices=True,
        )
        expected_loss = (-q_value / scale.detach() - alpha * expected_entropy).mean()
        expected_grad = torch.autograd.grad(expected_loss, engine.state.actor_params)
        assert len(outputs) == 11
        torch.testing.assert_close(outputs[1], info["entropy"])
        torch.testing.assert_close(outputs[8], expected_entropy)
        torch.testing.assert_close(outputs[2], expected_loss)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert scale.grad is None
    finally:
        model.env.close()


@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_scaled_inner_entropy_and_kl_are_not_q_normalized(monkeypatch, scale):
    model = _model(
        inner_actor_entropy_mode="tdmpc2_scaled",
        inner_outer_policy_kl_coef=0.5,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        ent_coef=0.25, outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
    )
    try:
        engine = _prepare(model)
        parameters = torch.tensor([2.0, -5.0, 3.0, 4.0], requires_grad=True)
        requests = []
        monkeypatch.setattr(engine.model, "pi", _controlled_policy(engine, parameters, requests))
        monkeypatch.setattr(engine.model, "Q", _action_q(engine))
        monkeypatch.setattr(engine, "_gaussian_kl", lambda info, anchor: info["kl"])
        outputs = engine._scaled_sac_actor_kernel(
            torch.zeros(4, model.cfg.latent_dim), torch.tensor(0.25),
            torch.tensor(scale), None, None, True,
        )
        outputs[2].backward()
        torch.testing.assert_close(parameters.grad, torch.tensor([-1.0 / scale, 0.0, -0.25, 0.5]))
        assert outputs[2].item() == pytest.approx(-2.0 / scale - 0.75 + 2.0)
        assert requests == [True, False]
    finally:
        model.env.close()


@pytest.mark.parametrize("entropy_mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("selected_entropy", [0.0, 2.0])
@pytest.mark.parametrize("update_actor", [False, True])
def test_inner_temperature_uses_selected_entropy_and_correct_direction(
    monkeypatch, entropy_mode, selected_entropy, update_actor
):
    model = _model(
        inner_actor_entropy_mode=entropy_mode,
        inner_temperature_mode="auto",
        inner_temperature_initialization="fixed",
        inner_temperature=0.25,
        inner_target_entropy=1.0,
        inner_temperature_updates_per_action=1,
    )
    try:
        engine = _prepare(model)
        log_prob = -selected_entropy if entropy_mode == "squashed" else -5.0
        parameters = torch.nn.Parameter(torch.tensor([2.0, log_prob, selected_entropy, 0.0]))
        engine.state.actor_params = [parameters]
        engine.state.actor_optim = torch.optim.SGD([parameters], lr=0.0)
        requests = []
        monkeypatch.setattr(engine.model, "pi", _controlled_policy(engine, parameters, requests))
        monkeypatch.setattr(engine.model, "Q", _action_q(engine))
        alpha_before = engine.alpha.detach().clone()
        expected_temperature_loss = engine.state.log_alpha.detach() * (selected_entropy - 1.0)
        metrics = engine._sac_policy_step(
            {"z": torch.zeros(4, model.cfg.latent_dim)},
            update_temperature=True, update_actor=update_actor, alpha=alpha_before,
        )
        torch.testing.assert_close(metrics["temperature_loss"], expected_temperature_loss)
        assert (engine.alpha > alpha_before).item() == (selected_entropy < 1.0)
        assert engine.state.temperature_steps == 1
        assert engine.state.actor_steps == int(update_actor)
        assert requests == [entropy_mode == "tdmpc2_scaled"]
        if entropy_mode == "tdmpc2_scaled":
            assert metrics["actor_scaled_entropy"].item() == selected_entropy
            assert metrics["actor_entropy_bonus"].item() == pytest.approx(0.25 * selected_entropy)
        else:
            assert "actor_scaled_entropy" not in metrics
            assert "actor_entropy_bonus" not in metrics
        if update_actor:
            assert metrics["actor_entropy"].item() == -log_prob
        else:
            assert parameters.grad is None
    finally:
        model.env.close()


@pytest.mark.parametrize("update_actors", [False, True])
def test_separate_critics_selects_entropy_for_both_actors_and_temperatures(monkeypatch, update_actors):
    model = _explorer_model(
        "separate_critics", inner_actor_entropy_mode="tdmpc2_scaled",
        inner_temperature_mode="auto", inner_temperature_initialization="fixed",
        inner_temperature=0.25, inner_target_entropy=1.0,
    )
    try:
        engine = _prepare(model)
        state = engine.state
        primary = torch.nn.Parameter(torch.tensor([2.0, -5.0, 0.0, 0.0]))
        explorer = torch.nn.Parameter(torch.tensor([3.0, -6.0, 2.0, 0.0]))
        requests = []
        policies = {
            id(state.actor): _controlled_policy(engine, primary, requests),
            id(state.explorer_actor): _controlled_policy(engine, explorer, requests),
        }
        monkeypatch.setattr(engine.model, "pi", lambda z, policy, **kw: policies[id(policy)](z, **kw))
        monkeypatch.setattr(engine.model, "Q", _action_q(engine))
        state.actor_params, state.explorer_actor_params = [primary], [explorer]
        state.actor_optim = torch.optim.SGD([primary], lr=0.0)
        state.explorer_actor_optim = torch.optim.SGD([explorer], lr=0.0)
        initial_alphas = [engine.alpha.detach().clone(), engine.explorer_alpha.detach().clone()]
        metrics = engine._separate_policy_step(
            {"z": torch.zeros(4, model.cfg.latent_dim)},
            update_primary_actor=update_actors, update_explorer_actor=update_actors,
            update_primary_temperature=True, update_explorer_temperature=True,
        )
        assert requests == [True, True]
        assert engine.alpha > initial_alphas[0]
        assert engine.explorer_alpha < initial_alphas[1]
        for prefix, parameter, initial_alpha in zip(("", "explorer_"), (primary, explorer), initial_alphas):
            assert metrics[f"{prefix}actor_scaled_entropy"].item() == parameter[2].item()
            assert metrics[f"{prefix}actor_entropy_bonus"].item() == pytest.approx(float(initial_alpha * parameter[2]))
            if update_actors:
                torch.testing.assert_close(parameter.grad, torch.tensor([-1.0, 0.0, -float(initial_alpha), 0.0]))
                assert metrics[f"{prefix}actor_entropy"].item() == -parameter[1].item()
            else:
                assert parameter.grad is None
    finally:
        model.env.close()


@pytest.mark.parametrize("target_mode", ["entropy_augmented", "reward_only"])
def test_inner_critic_target_uses_selected_entropy(monkeypatch, target_mode):
    model = _model(inner_actor_entropy_mode="tdmpc2_scaled", inner_sac_critic_target=target_mode)
    try:
        engine = _prepare(model)
        requests = []
        parameters = torch.tensor([0.0, -5.0, 2.0, 0.0])
        monkeypatch.setattr(engine.model, "pi", _controlled_policy(engine, parameters, requests))
        monkeypatch.setattr(engine, "_bootstrap_q", lambda z, action, **kw: torch.full((z.shape[0], 1), 3.0))
        z = torch.zeros(4, model.cfg.latent_dim)
        reward = torch.ones(4, 1)
        terminated = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        args = (z, torch.zeros(4, 1), reward, z, terminated, torch.tensor(0.25), torch.zeros(4, 1), None)
        scaled_outputs = engine._sac_critic_kernel(*args)
        model.cfg.inner_actor_entropy_mode = "squashed"
        ordinary_outputs = engine._sac_critic_kernel(*args)
        if target_mode == "reward_only":
            for actual, expected in zip(scaled_outputs, ordinary_outputs):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected = reward + model.agent.discount * (1 - terminated) * (
            3.0 + (0.5 if target_mode == "entropy_augmented" else 0.0)
        )
        torch.testing.assert_close(scaled_outputs[2], expected)
        expected_ordinary = reward + model.agent.discount * (1 - terminated) * (
            3.0 + (1.25 if target_mode == "entropy_augmented" else 0.0)
        )
        torch.testing.assert_close(ordinary_outputs[2], expected_ordinary)
        assert not scaled_outputs[2].requires_grad
        assert requests == [target_mode == "entropy_augmented", False]
    finally:
        model.env.close()


@pytest.mark.parametrize("explorer_mode", ["none", "frozen_random", "adaptive_param_noise", "separate_critics"])
def test_scaled_inner_entropy_runs_gaussian_explorer_variants(explorer_mode):
    kwargs = {"inner_actor_entropy_mode": "tdmpc2_scaled", "inner_target_entropy": 1.0}
    if explorer_mode == "none":
        model = _model(**kwargs)
    elif explorer_mode == "adaptive_param_noise":
        from tests.test_ambi_parameter_noise_runtime import _parameter_noise_model
        model = _parameter_noise_model(**kwargs)
    else:
        model = _explorer_model(explorer_mode, **kwargs)
    try:
        rng = torch.random.get_rng_state().clone()
        action = model.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        assert torch.isfinite(action).all()
        metrics = model.agent.last_inner_metrics
        for prefix in (["inner_", "inner_explorer_"] if explorer_mode == "separate_critics" else ["inner_"]):
            assert math.isfinite(metrics[f"{prefix}actor_entropy"])
            assert math.isfinite(metrics[f"{prefix}actor_scaled_entropy"])
            assert math.isfinite(metrics[f"{prefix}actor_entropy_bonus"])
    finally:
        model.env.close()


@pytest.mark.parametrize("target_mode", ["entropy_augmented", "reward_only"])
@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("temperature", ["fixed", "auto"])
def test_separate_critic_targets_use_selected_entropy_and_own_alpha(monkeypatch, target_mode, mode, temperature):
    model = _explorer_model(
        "separate_critics", inner_actor_entropy_mode=mode,
        inner_temperature_mode=temperature, inner_temperature=0.25,
        inner_temperature_initialization="fixed", inner_target_entropy=1.0,
        inner_sac_critic_target=target_mode,
    )
    try:
        engine = _prepare(model)
        state = engine.state
        with torch.no_grad():
            if temperature == "auto":
                state.explorer_log_alpha.fill_(math.log(0.5))
            else:
                state.explorer_alpha_fixed.fill_(0.5)
        requests = []
        policies = {
            id(state.actor): _controlled_policy(
                engine, torch.tensor([0.0, -5.0, 20.0, 0.0]), requests
            ),
            id(state.explorer_actor): _controlled_policy(
                engine, torch.tensor([0.0, -6.0, 30.0, 0.0]), requests
            ),
        }
        monkeypatch.setattr(
            engine.model, "pi", lambda z, policy, **kw: policies[id(policy)](z, **kw)
        )
        monkeypatch.setattr(
            engine, "_q_with", lambda z, action, critic: torch.full((z.shape[0], 1), 3.0)
        )
        targets = []
        original_loss = engine.model.critic_loss

        def capture_target(predictions, target):
            assert not target.requires_grad
            targets.append(target.detach().clone())
            return original_loss(predictions, target)

        monkeypatch.setattr(engine.model, "critic_loss", capture_target)
        batch = {
            "z": torch.zeros(4, model.cfg.latent_dim),
            "action": torch.zeros(4, 1),
            "next_z": torch.zeros(4, model.cfg.latent_dim),
            "reward": torch.ones(4, 1),
            "terminated": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
        }
        engine._separate_critics_step(batch, update_primary=True, update_explorer=True)
        assert requests == [mode == "tdmpc2_scaled" and target_mode == "entropy_augmented"] * 2
        entropies = (20.0, 30.0) if mode == "tdmpc2_scaled" else (5.0, 6.0)
        for target, entropy, alpha in zip(targets, entropies, (0.25, 0.5)):
            expected = batch["reward"] + model.agent.discount * (
                1 - batch["terminated"]
            ) * (3.0 + (alpha * entropy if target_mode == "entropy_augmented" else 0.0))
            torch.testing.assert_close(target, expected)
        assert len(targets) == 2
    finally:
        model.env.close()


@pytest.mark.parametrize("update_actor", [False, True])
@pytest.mark.parametrize("q_scale", [None, 3.0])
def test_scaled_inner_actor_compiled_eager_outputs_and_gradients_match(update_actor, q_scale):
    model = _model(
        inner_actor_entropy_mode="tdmpc2_scaled",
        sac_actor_loss_scale_mode="none" if q_scale is None else "tdmpc2_percentile_range",
        ent_coef=0.25, outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
    )
    try:
        engine = _prepare(model)
        args = (
            torch.zeros(4, model.cfg.latent_dim), torch.tensor(0.25),
            *((torch.tensor(q_scale),) if q_scale is not None else ()),
            torch.ones(4, model.cfg.action_dim), torch.tensor([0, 1]), update_actor,
        )
        kernel = engine._compile_regions["actor"].eager
        eager = kernel(*args)
        # Dynamo capture with a real AOTAutograd backend also exercises the
        # conditional payload and backward graph, without platform C++ setup.
        compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True, dynamic=False)
        actual = compiled(*args)
        assert len(actual) == len(eager) == 11
        for result, expected in zip(actual, eager):
            torch.testing.assert_close(result, expected)
        if update_actor:
            actual_grad = torch.autograd.grad(actual[2], engine.state.actor_params)
            eager_grad = torch.autograd.grad(eager[2], engine.state.actor_params)
            for result, expected in zip(actual_grad, eager_grad):
                torch.testing.assert_close(result, expected)
    finally:
        model.env.close()


@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("target_mode", ["reward_only", "entropy_augmented"])
def test_inner_critic_entropy_compiled_targets_gradients_and_horizon_boundaries(
    mode, representation, target_mode,
):
    from tests.test_ambi_root_local_sac import _tiny_model

    torch._dynamo.reset()
    model = _tiny_model(
        inner_actor_entropy_mode=mode, inner_target_entropy=1.0,
        q_representation=representation, inner_sac_critic_target=target_mode,
        inner_finite_horizon=True, mppi_terminal_q_reduction="mean_all",
    )
    try:
        engine = _prepare(model)
        z = torch.randn(4, model.cfg.latent_dim)
        next_z = torch.randn_like(z, requires_grad=True)
        alpha = torch.tensor(0.25, requires_grad=True)
        args = (
            z, torch.zeros(4, model.cfg.action_dim), torch.ones(4, 1), next_z,
            torch.tensor([[0.0], [0.0], [1.0], [1.0]]), alpha,
            torch.ones(4, model.cfg.action_dim), torch.tensor([0, 1]),
            torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
            torch.zeros(4, model.cfg.action_dim),
        )
        kernel = engine._sac_critic_kernel
        rng = torch.random.get_rng_state()
        eager = kernel(*args)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True, dynamic=False)
        actual = compiled(*args)
        for result, expected in zip(actual, eager):
            torch.testing.assert_close(result, expected)
        for result, expected in zip(
            torch.autograd.grad(actual[0], engine.state.critic_params),
            torch.autograd.grad(eager[0], engine.state.critic_params),
        ):
            torch.testing.assert_close(result, expected)
        assert not actual[2].requires_grad
        assert next_z.grad is alpha.grad is None
        prior = engine._prior_bootstrap(next_z, args[-1])
        torch.testing.assert_close(actual[2][1], 1.0 + model.agent.discount * prior[1])
        torch.testing.assert_close(actual[2][2:], torch.ones(2, 1))
    finally:
        model.env.close()
        torch._dynamo.reset()
