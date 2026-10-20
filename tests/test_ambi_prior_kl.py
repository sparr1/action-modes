"""Scientific contract for replacing inner entropy with KL to the frozen prior.

These tests exercise the existing production objective and Adam update rather
than a second implementation of the proposed controller.
"""

from copy import deepcopy

import pytest
import torch
from torch.distributions import Normal, TransformedDistribution, kl_divergence
from torch.distributions.transforms import TanhTransform

from evaluate_ambi_checkpoint import _outer_state_digest
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_component_model


def _model(**overrides):
    settings = dict(
        aux_return_mode="sac",
        inner_actor_source="sac",
        inner_critic_source="aux_return",
        inner_horizon_actor_source="sac",
        inner_horizon_critic_source="aux_return",
        inner_sac_critic_target="reward_only",
        inner_entropy_enabled=False,
        inner_outer_policy_kl_coef=1.0,  # Resolved from loaded outer alpha below.
        ent_coef="auto_0.0033654973",
        inner_finite_horizon=True,
        inner_rollout_horizon=3,
        train_unroll_horizon=3,
        inner_rounds=2,
        inner_rollouts_per_round=4,
        inner_replay_capacity=24,
        inner_critic_updates_per_round=2,
        inner_actor_updates_per_round=2,
        inner_batch_size=8,
        inner_actor_adam_eps=1e-5,
        sac_actor_loss_scale_mode="none",
        critic_ensemble_backend="vmap",
    )
    settings.update(overrides)
    wrapper = _tiny_component_model(**settings)
    # A frozen checkpoint's learned alpha is available only after loading it.
    # Store a number, not a tensor alias or a temperature-controlled coefficient.
    wrapper.cfg.inner_outer_policy_kl_coef = float(wrapper.agent.alpha.detach())
    return wrapper


def _prepare(wrapper):
    engine = wrapper.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def _batch(engine):
    return {"z": torch.linspace(-0.7, 0.7, 8 * engine.cfg.latent_dim).reshape(8, -1)}


def _actor_vector(engine):
    return torch.cat([parameter.detach().flatten() for parameter in engine.state.actor_params])


def _analytic_kl(engine, z):
    _, adapted = engine.model.pi(z, policy=engine.state.actor, deterministic=True,
                                 **engine._inner_policy_kwargs())
    with torch.no_grad():
        _, prior = engine.model.pi(z, policy=engine.state.actor_anchor,
                                  deterministic=True, **engine._actor_options)
    return engine._gaussian_kl(adapted, prior).mean()


def _controlled_q(engine, *, slope):
    def q(z, action, **kwargs):
        assert kwargs["reduction"] == "all"
        assert kwargs["detach"] is True
        # Preserve the action path even when its derivative is exactly zero.
        return (0.7 + slope * action[..., :1]).unsqueeze(0).expand(engine.cfg.num_q, -1, -1)
    return q


def test_analytic_kl_is_inner_to_prior_and_tanh_preserves_log_ratio():
    mu = torch.tensor([[0.3, -0.4], [0.9, 0.1]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-0.2, 0.1], [-0.5, -0.8]], dtype=torch.float64, requires_grad=True)
    prior_mu = torch.tensor([[-0.1, 0.2], [0.2, -0.7]], dtype=torch.float64)
    prior_log_std = torch.tensor([[0.2, -0.6], [-0.2, -0.4]], dtype=torch.float64)
    inner, prior = Normal(mu, log_std.exp()), Normal(prior_mu, prior_log_std.exp())
    actual = InnerImprovementEngine._gaussian_kl(
        {"pre_tanh_mean": mu, "log_std": log_std},
        {"pre_tanh_mean": prior_mu, "log_std": prior_log_std},
    )
    expected = kl_divergence(inner, prior).sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, kl_divergence(prior, inner).sum(-1, keepdim=True))

    # The shared bijection cancels pointwise in log p_inner - log p_prior.
    raw = torch.tensor([[0.5, -1.1], [1.3, 0.7]], dtype=torch.float64)
    action = raw.tanh()
    squashed_inner = TransformedDistribution(inner, [TanhTransform()])
    squashed_prior = TransformedDistribution(prior, [TanhTransform()])
    torch.testing.assert_close(
        squashed_inner.log_prob(action) - squashed_prior.log_prob(action),
        inner.log_prob(raw) - prior.log_prob(raw),
    )
    actual.sum().backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(log_std.grad).all()


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_flat_q_keeps_prior_after_32_real_adam_updates(monkeypatch, representation):
    wrapper = _model(q_representation=representation, num_q=2 if representation == "scalar" else 3)
    try:
        engine = _prepare(wrapper)
        monkeypatch.setattr(engine.model, "Q", _controlled_q(engine, slope=0.0))
        batch = _batch(engine)
        initial = _actor_vector(engine)
        outer = _outer_state_digest(wrapper)
        beta = wrapper.cfg.inner_outer_policy_kl_coef
        assert engine.state.actor_anchor is wrapper.agent.model._pi
        assert engine.state.actor_anchor is not engine.state.actor
        assert engine.alpha.item() == 0.0

        for _ in range(32):
            metrics = engine._sac_policy_step(
                batch, update_actor=True, update_temperature=True,
                alpha=torch.tensor(100.0),  # Entropy off must override this too.
            )
            assert "temperature_loss" not in metrics
            assert abs(metrics["outer_policy_kl"].item()) < 1e-7

        # Roundoff in Gaussian KL derivatives can cause sub-ulp Adam movement.
        torch.testing.assert_close(_actor_vector(engine), initial, rtol=0, atol=2e-7)
        assert abs(_analytic_kl(engine, batch["z"]).item()) < 1e-7
        assert engine.state.actor_steps == 32
        assert engine.state.temperature_steps == 0
        assert engine.state.temperature_optim is None
        assert engine.state.log_alpha is None and engine.state.alpha_fixed is None
        assert wrapper.cfg.inner_outer_policy_kl_coef == beta
        assert _outer_state_digest(wrapper) == outer
    finally:
        wrapper.env.close()


def test_return_gradient_can_move_actor_despite_prior_kl(monkeypatch):
    wrapper = _model()
    try:
        engine = _prepare(wrapper)
        monkeypatch.setattr(engine.model, "Q", _controlled_q(engine, slope=1.0))
        batch = _batch(engine)
        noise = torch.linspace(-1.5, 1.5, 8).reshape(8, 1)
        initial = _actor_vector(engine)
        outer = _outer_state_digest(wrapper)

        def expected_q():
            with torch.no_grad():
                action, _ = engine.model.pi(batch["z"], policy=engine.state.actor,
                                           noise=noise, **engine._inner_policy_kwargs())
                return (0.7 + action[..., :1]).mean()

        before = expected_q()
        for _ in range(32):
            engine._sac_policy_step(batch, update_actor=True, update_temperature=True,
                                    alpha=engine.alpha)
        assert expected_q() > before + 0.01
        assert (_actor_vector(engine) - initial).norm().item() > 0.01
        assert _analytic_kl(engine, batch["z"]).item() > 1e-4
        assert engine.state.temperature_steps == 0
        assert _outer_state_digest(wrapper) == outer
    finally:
        wrapper.env.close()


def test_kernel_uses_raw_q_minus_fixed_kl_without_entropy(monkeypatch):
    wrapper = _model()
    try:
        engine = _prepare(wrapper)
        monkeypatch.setattr(engine.model, "Q", _controlled_q(engine, slope=1.0))
        z = _batch(engine)["z"]
        noise = torch.linspace(-1.5, 1.5, 8).reshape(8, 1)
        with torch.no_grad():
            engine.state.actor[-1].bias.add_(0.15)
        output = engine._sac_actor_kernel(z, torch.tensor(100.0), noise, None, True)
        actual_grads = torch.autograd.grad(output[2], engine.state.actor_params)
        action, info = engine.model.pi(z, policy=engine.state.actor, noise=noise,
                                      **engine._inner_policy_kwargs())
        with torch.no_grad():
            _, prior = engine.model.pi(z, policy=engine.state.actor_anchor,
                                      deterministic=True, **engine._actor_options)
        expected_kl = kl_divergence(
            Normal(info["pre_tanh_mean"], info["log_std"].exp()),
            Normal(prior["pre_tanh_mean"], prior["log_std"].exp()),
        ).sum(-1, keepdim=True)
        expected = (-(0.7 + action[..., :1])
                    + wrapper.cfg.inner_outer_policy_kl_coef * expected_kl).mean()
        expected_grads = torch.autograd.grad(expected, engine.state.actor_params)
        torch.testing.assert_close(output[2], expected)
        torch.testing.assert_close(output[4], expected_kl.mean())
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)
        assert all(parameter.grad is None for parameter in engine.state.actor_anchor.parameters())
    finally:
        wrapper.env.close()


def test_prior_kl_kernel_matches_aot_compilation_including_gradients():
    wrapper = _model()
    try:
        engine = _prepare(wrapper)
        z = _batch(engine)["z"]
        with torch.no_grad():
            engine.state.actor[-1].bias.add_(0.15)
            for parameter in engine.state.critic.parameters():
                parameter.add_(0.03 * torch.randn_like(parameter))
        args = (z, torch.tensor(100.0), torch.linspace(-1.5, 1.5, 8).reshape(8, 1),
                None, True)
        eager = engine._sac_actor_kernel(*args)
        compiled = torch.compile(engine._sac_actor_kernel, backend="aot_eager",
                                 fullgraph=True, dynamic=False)(*args)
        for actual, reference in zip(compiled, eager):
            torch.testing.assert_close(actual, reference)
        actual_grads = torch.autograd.grad(compiled[2], engine.state.actor_params)
        expected_grads = torch.autograd.grad(eager[2], engine.state.actor_params)
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference)
    finally:
        wrapper.env.close()


def test_replacement_preserves_reward_only_critic_targets_and_auxiliary_tail():
    wrapper = _model()
    try:
        engine = _prepare(wrapper)
        z = _batch(engine)["z"]
        # Nontrivial frozen targets ensure a mistaken terminal source is visible.
        with torch.no_grad():
            for module in (engine.state.critic_target, engine._horizon_critic):
                for parameter in module.parameters():
                    parameter.add_(0.03 * torch.randn_like(parameter))
        action = torch.linspace(-0.7, 0.7, 8).reshape(8, 1)
        reward = torch.linspace(0.1, 0.8, 8).reshape(8, 1)
        terminal = torch.zeros(8, 1)
        terminal[1] = 1
        horizon_end = torch.tensor([[0], [0], [1], [0], [1], [0], [1], [1]], dtype=torch.bool)
        args = (z, action, reward, z + 0.1, terminal, torch.tensor(0.7),
                action.flip(0), None, horizon_end, action)
        no_entropy = engine._sac_critic_kernel(*args)
        wrapper.cfg.inner_entropy_enabled = True
        wrapper.cfg.inner_outer_policy_kl_coef = 0.0
        original = engine._sac_critic_kernel(*args)
        for actual, reference in zip(no_entropy, original):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        prior_value = engine._prior_bootstrap(z + 0.1, action)
        torch.testing.assert_close(no_entropy[2][horizon_end],
                                   (reward + wrapper.agent.discount * prior_value)[horizon_end])
        torch.testing.assert_close(no_entropy[2][1], reward[1])
        assert engine._horizon_critic is wrapper.agent.model._aux_return_Qs
    finally:
        wrapper.env.close()


def test_full_solves_have_no_temperature_updates_and_preserve_outer_state(monkeypatch):
    wrapper = _model(dropout=0.1, inner_critic_dropout_enabled=True)
    try:
        # Preserve pre-existing outer gradients as well as serialized state.
        parameters = list(wrapper.agent.model.parameters())
        parameters[0].grad = torch.ones_like(parameters[0]) * 0.125
        gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
        outer = _outer_state_digest(wrapper)
        optimizer = deepcopy(wrapper.agent.ent_coef_optim.state_dict())
        rng = torch.random.get_rng_state().clone()
        engine = wrapper.agent.inner_engine
        anchors = []
        original_prepare = engine._prepare_workspace

        def capture_prepare(**kwargs):
            original_prepare(**kwargs)
            anchors.append(engine.state.actor_anchor is wrapper.agent.model._pi)
            _assert_tree_equal(engine.state.actor.state_dict(), wrapper.agent.model._pi.state_dict())
            assert engine.state.temperature_optim is None

        monkeypatch.setattr(engine, "_prepare_workspace", capture_prepare)
        for decision in range(2):
            action = wrapper.agent.act(torch.tensor([0.2, -0.1, 0.7]), t0=decision == 0,
                                       eval_mode=True, collect_diagnostics=True)
            assert torch.isfinite(action).all()
            assert engine.state.actor_steps == 4
            assert engine.state.temperature_steps == 0
            assert wrapper.agent.last_inner_metrics["inner_outer_policy_kl"] >= -1e-7
        assert anchors == [True, True]
        assert _outer_state_digest(wrapper) == outer
        _assert_tree_equal(wrapper.agent.ent_coef_optim.state_dict(), optimizer)
        _assert_tree_equal([p.grad for p in parameters], gradients)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    finally:
        wrapper.env.close()
