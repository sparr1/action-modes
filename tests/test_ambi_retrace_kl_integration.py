"""Independent equations and real CPU execution for Retrace with a prior KL."""

from contextlib import contextmanager

import pytest
import torch
from torch.distributions import Normal, TanhTransform, kl_divergence

from RL.tdmpc2_core.common.retrace import retrace_targets
from tests.test_ambi_entropy_probe_invariants import _fixture
from tests.test_ambi_root_local_sac import _assert_finite_metrics


@contextmanager
def _model(*, entropy=True, **overrides):
    options = dict(
        inner_sac_return_estimator="retrace",
        inner_retrace_lambda=1.0,
        inner_outer_policy_kl_coef=0.1,
        inner_entropy_enabled=entropy,
        inner_actor_updates_per_round=1,
        ent_coef="auto_0.03",
        dropout=0.0,
    )
    options.update(overrides)
    model = _fixture(**options)
    try:
        yield model
    finally:
        model.env.close()


def _prepare(model):
    engine = model.agent.inner_engine
    engine._prepare_workspace(t0=True)
    root = torch.zeros(1, engine.cfg.latent_dim)
    engine._collect_retrace_round(root)
    batch = engine._sample_retrace_batch(torch.tensor([0, 1]))
    return engine, batch


def _stats(engine, z, *, policy=None, std_scale=1.0):
    return engine.model.policy_stats(
        z, policy=engine.state.actor if policy is None else policy,
        std_scale=std_scale,
        log_std_mapping=engine.cfg.inner_log_std_mapping,
        log_std_min=engine.cfg.inner_log_std_min,
        log_std_max=engine.cfg.inner_log_std_max,
    )


def _normal(stats):
    return Normal(stats["pre_tanh_mean"], stats["log_std"].exp())


def _kernel(engine, batch, alpha):
    count, horizon = batch["valid"].shape[:2]
    return engine._retrace_critic_kernel(
        batch, torch.as_tensor(alpha),
        torch.zeros(count, horizon, engine.cfg.action_dim),
        torch.zeros(count, engine.cfg.action_dim), torch.tensor([0, 1]),
    )


def test_retrace_known_return_uses_successor_ratio_and_all_suffixes():
    # The first action's Q and ratio cannot affect its own return. Hand-worked
    # three-step example: G2=30, G1=20+.9*.25*(30-5), G0=10+.9*.5*(G1-4).
    column = lambda x: torch.tensor(x, dtype=torch.float64).reshape(1, 3, 1)
    reward, discount = column([1, 2, 3]), column([0.9, 0.9, 0.9])
    value, q = column([10, 20, 30]), column([float("nan"), 4, 5])
    coefficients = column([float("nan"), 0.5, 0.25])
    valid = torch.ones_like(reward, dtype=torch.bool)
    targets, lengths, corrections = retrace_targets(
        reward, discount, value, q, coefficients, valid,
    )
    torch.testing.assert_close(targets, column([19.73125, 25.625, 30]))
    torch.testing.assert_close(lengths, column([1.625, 1.25, 1]))
    torch.testing.assert_close(corrections, column([9.73125, 5.625, 0]))
    # A true termination cuts both the bootstrap and the remaining trace.
    discount[:, 1] = 0
    value[:, 1] = float("nan")
    targets, lengths, _ = retrace_targets(reward, discount, value, q, coefficients, valid)
    torch.testing.assert_close(targets, column([9.1, 2, 30]))
    torch.testing.assert_close(lengths, column([1.5, 1, 1]))


def test_retrace_uses_stored_collection_density_and_current_actor_density():
    # Nonunit collection scale distinguishes behavior density from target pi.
    with _model(inner_behavior_std_scale=1.3) as model:
        engine, batch = _prepare(model)
        z = batch["z"].reshape(-1, engine.cfg.latent_dim)
        u = batch["pre_tanh_action"].reshape(-1, engine.cfg.action_dim)
        behavior = _normal(_stats(engine, z, std_scale=1.3))
        jacobian = TanhTransform().log_abs_det_jacobian(u, u.tanh())
        expected_mu = (behavior.log_prob(u) - jacobian).sum(-1, keepdim=True)
        saved_mu = batch["behavior_log_prob"].clone()
        torch.testing.assert_close(saved_mu.reshape_as(expected_mu), expected_mu, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(batch["action"].reshape_as(u), u.tanh())

        # Change the local actor after collection; the denominator must remain
        # the actual historical behavior, not recomputed under the new actor.
        with torch.no_grad():
            engine.state.actor[-1].bias[0].add_(0.7)
        current = _normal(_stats(engine, z))
        expected = (current.log_prob(u) - behavior.log_prob(u)).sum(-1, keepdim=True).exp().clamp(max=1)
        outputs = _kernel(engine, batch, 0.03)
        torch.testing.assert_close(outputs[4].reshape_as(expected), expected, atol=2e-5, rtol=2e-5)
        assert (expected < 0.9).any(), "The example must exercise an off-policy ratio."
        torch.testing.assert_close(batch["behavior_log_prob"], saved_mu, rtol=0, atol=0)
        assert not outputs[2].requires_grad


@pytest.mark.parametrize("entropy", [False, True], ids=["kl_only", "entropy_and_kl"])
def test_reward_only_retrace_target_excludes_alpha_and_beta(entropy):
    with _model(entropy=entropy) as model:
        engine, batch = _prepare(model)
        baseline = _kernel(engine, batch, 0.0)
        engine.cfg.inner_outer_policy_kl_coef = 8.0
        changed = _kernel(engine, batch, 3.0)
        torch.testing.assert_close(changed[2], baseline[2], rtol=0, atol=0)
        torch.testing.assert_close(changed[4], baseline[4], rtol=0, atol=0)
        # Positive control: entropy belongs in the target only when explicitly
        # requested, and even then never at the frozen outer horizon boundary.
        engine.cfg.inner_sac_critic_target = "entropy_augmented"
        soft = _kernel(engine, batch, 3.0)[2].reshape(2, 3, 1)
        reward_only = baseline[2].reshape_as(soft)
        torch.testing.assert_close(soft[:, -1], reward_only[:, -1], rtol=0, atol=0)
        assert not torch.allclose(soft[:, :-1], reward_only[:, :-1])


@pytest.mark.parametrize("entropy", [False, True], ids=["kl_only", "entropy_and_kl"])
def test_retrace_actor_has_exact_prior_kl_loss_and_gradient(entropy):
    with _model(entropy=entropy) as model:
        engine, batch = _prepare(model)
        with torch.no_grad():
            engine.state.actor[-1].bias.add_(torch.tensor([0.2, 0.05]))
        z = batch["z"].reshape(-1, engine.cfg.latent_dim)
        noise = torch.linspace(-0.8, 0.8, len(z)).reshape(-1, 1)
        alpha = torch.tensor(0.03)
        outputs = engine._sac_actor_kernel(z, alpha, noise, torch.tensor([0, 1]), True)
        current = _normal(_stats(engine, z))
        prior = _normal(_stats(engine, z, policy=engine.state.actor_anchor))
        analytic_kl = kl_divergence(current, prior).sum(-1).mean()
        assert analytic_kl > 0
        torch.testing.assert_close(outputs[4], analytic_kl)
        beta = engine.cfg.inner_outer_policy_kl_coef
        expected_loss = -outputs[3] + beta * analytic_kl
        if entropy:
            expected_loss = expected_loss + alpha * outputs[0].mean()
        torch.testing.assert_close(outputs[2], expected_loss)

        engine.cfg.inner_outer_policy_kl_coef = 0.0
        unregularized = engine._sac_actor_kernel(z, alpha, noise, torch.tensor([0, 1]), True)
        parameters = tuple(engine.state.actor.parameters())
        observed_gradient = torch.autograd.grad(outputs[2] - unregularized[2], parameters, retain_graph=True)
        expected_gradient = torch.autograd.grad(beta * analytic_kl, parameters)
        for observed, expected in zip(observed_gradient, expected_gradient):
            torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("entropy", [False, True], ids=["kl_only", "entropy_and_kl"])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_complete_retrace_kl_solve_updates_expected_components_and_full_trajectories(entropy, representation):
    # Exercise the real collector, trajectory replay, optimizer dispatch and
    # target updates; online-critic dropout matches the production capability.
    with _model(entropy=entropy, representation=representation, dropout=0.01) as model:
        engine = model.agent.inner_engine
        outer_before = {name: value.clone() for name, value in model.agent.model.state_dict().items()}
        actor_before = [p.detach().clone() for p in engine._actor_base.parameters()]
        critic_before = [p.detach().clone() for p in engine._critic_base.parameters()]
        model.agent.act(torch.tensor([0.2, -0.1, 0.7]), t0=True, eval_mode=True)
        metrics = model.agent.last_inner_metrics
        _assert_finite_metrics(metrics)
        assert metrics["inner_model_steps"] == 24
        assert metrics["inner_critic_optimizer_steps"] == 4
        assert metrics["inner_actor_optimizer_steps"] == 2
        assert metrics["inner_temperature_optimizer_steps"] == (2 if entropy else 0)
        assert metrics["inner_retrace_replay_trajectories"] == 8
        assert metrics["inner_retrace_trajectory_draws"] == 8
        # Actor B=4 stays unchanged; critic ceil(B/H)=2 trajectories means six
        # rows per update, including every suffix, not four truncated rows.
        assert engine.cfg.inner_batch_size == 4
        assert engine.cfg.inner_retrace_batch_trajectories == 2
        assert metrics["inner_retrace_critic_rows"] == 24
        pool = engine._action_pool
        assert any(not torch.equal(a, b) for a, b in zip(actor_before, pool.actor.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(critic_before, pool.critic.parameters()))
        if entropy:
            assert pool.log_alpha is not None
            assert float(metrics["inner_alpha_delta"]) != 0
        else:
            assert pool.log_alpha is None
            assert metrics["inner_alpha_final"] == 0
        for name, value in model.agent.model.state_dict().items():
            torch.testing.assert_close(value, outer_before[name], rtol=0, atol=0)
