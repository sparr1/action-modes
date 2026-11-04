"""Corrected TD-M(PC)^2 weighting of AMBI's true squashed behavior density."""

import itertools

import pytest
import torch

import RL.tdmpc2_core.ambi_agent as agent_module
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_behavior_policy_kl import _behavior_batch, _kl_model, _stub_actor
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from utils.resume_identity import scientific_trial_parameters


def _recipe(**overrides):
    return {
        "outer_behavior_policy_objective": "action_space_cross_entropy",
        "outer_behavior_policy_weighting": "tdmpc_square",
        "outer_behavior_policy_kl_schedule": "quantile_gate",
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "ent_coef": 0.0001,
        "inner_temperature_mode": "inherit_outer",
        **overrides,
    }


def _agent(**overrides):
    params = _recipe(**overrides)
    schedule = params.pop("outer_behavior_policy_kl_schedule")
    wrapped = _kl_model(schedule, **params)
    wrapped.env.close()
    return wrapped.agent


def _identity(params):
    return scientific_trial_parameters({
        "alg": "AMBITDMPC2/AMBITDMPC2", "seed": 7, "alg_params": params,
    })


def test_recipe_is_opt_in_and_stores_behavior_without_reanalysis():
    default = _build_cfg()
    assert default.outer_behavior_policy_weighting == "constant"
    assert not default.store_behavior_policy
    cfg = _build_cfg(**_recipe(outer_behavior_policy_weighting="TDMPC_SQUARE"))
    assert cfg.outer_behavior_policy_weighting == "tdmpc_square"
    assert cfg.outer_behavior_policy_kl_coef == 1.0
    assert cfg.outer_behavior_policy_kl_q_threshold == 2.0
    assert cfg.store_behavior_policy and not cfg.store_teacher_policy
    assert cfg.outer_actor_objective == "q_maximization"
    assert cfg.outer_actor_entropy_mode == "squashed"


@pytest.mark.parametrize("overrides,match", [
    ({"outer_behavior_policy_weighting": "unknown"}, "outer_behavior_policy_weighting"),
    ({"outer_behavior_policy_weighting": None}, "outer_behavior_policy_weighting"),
    ({"outer_behavior_policy_weighting": True}, "outer_behavior_policy_weighting"),
    ({"outer_behavior_policy_objective": "reverse_kl"}, "outer_behavior_policy_objective"),
    ({"outer_behavior_policy_kl_schedule": "smooth"}, "outer_behavior_policy_kl_schedule"),
    ({"outer_behavior_policy_kl_schedule": "none"}, "outer_behavior_policy_kl_schedule"),
    ({"outer_behavior_policy_kl_schedule": "dual"}, "outer_behavior_policy_kl_schedule"),
    ({"sac_actor_loss_scale_mode": "none"}, "sac_actor_loss_scale_mode"),
    ({"outer_actor_objective": "boom"}, "outer_actor_objective"),
    ({"outer_actor_objective": "distillation"}, "outer_actor_objective"),
    ({"critic_value_mode": "return_entropy"}, "critic_value_mode"),
    ({"rho": -0.1}, "rho >= 0"),
    ({"ent_coef": "auto"}, "fixed temperatures"),
    ({"inner_execution_action": "mean"}, "stochastic inner execution"),
])
def test_recipe_rejects_inconsistent_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        _build_cfg(**_recipe(**overrides))


def test_direct_agent_construction_also_validates_weighting():
    cfg = _agent().cfg
    cfg.outer_behavior_policy_objective = "reverse_kl"
    with pytest.raises(ValueError, match="TD-M.*weighting requires"):
        AMBITDMPC2Agent(cfg)


@pytest.mark.parametrize("dimension", [1, 3, 21, 61])
def test_coefficient_uses_joint_density_once_and_the_just_updated_shared_scale(
    monkeypatch, dimension,
):
    agent = _agent(sac_actor_loss_scale_tau=1.0, outer_behavior_policy_kl_coef=0.05)
    # Policy/Q are stubbed, so change the coordinate count without changing
    # the Pendulum fixture. This checks dimension scaling independently.
    agent.cfg.action_dim = dimension
    zs = _stub_actor(monkeypatch, agent, current_mean=0.0)
    agent.actor_loss_scale.fill_(99.0)
    scales = iter([2.0, 4.0, 1.5, 3.0])
    calls = []

    def next_range(*_args, **_kwargs):
        calls.append(1)
        return torch.tensor([next(scales)])

    monkeypatch.setattr(agent_module, "percentile_range", next_range)
    for scale in [2.0, 4.0, 1.5, 3.0]:
        metrics = agent._update_actor(zs, *_behavior_batch(agent))
        joint_coefficient = 0.05 * (dimension / 61.0) / scale if scale > 2 else 0.0
        assert metrics["actor_loss_scale"] == pytest.approx(scale)
        assert metrics["behavior_policy_action_ce_gate_active"] == float(scale > 2)
        assert metrics["behavior_policy_action_ce_joint_coefficient"] == pytest.approx(joint_coefficient)
        assert metrics["behavior_policy_action_ce_base_coefficient"] == 0.05
        per_dimension_ce = metrics["behavior_policy_action_ce"]
        expected = joint_coefficient * (per_dimension_ce * dimension)
        torch.testing.assert_close(metrics["behavior_policy_action_ce_weighted_loss"], expected)
        torch.testing.assert_close(metrics["actor_loss"], expected)
    assert len(calls) == 4


def test_actor_gradient_matches_true_squashed_behavior_log_likelihood(monkeypatch):
    agent = _agent(batch_size=8, sac_actor_loss_scale_tau=1.0, grad_clip_norm=1e10)
    agent.cfg.action_dim = 3
    zs = _stub_actor(monkeypatch, agent)
    noise = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=3)))
    offsets = torch.tensor([0.2, -0.3, 0.4])
    parameter = next(agent.model._pi.parameters())
    theta = parameter.flatten()[0].detach().clone().requires_grad_()
    calls = []

    def policy(z):
        calls.append(1)
        anchor = parameter.flatten()[0]
        mean = (anchor + offsets).expand(*z.shape[:-1], 3)
        log_std = (0.1 * anchor - 0.4 + offsets * 0.2).expand_as(mean)
        u = mean + log_std.exp() * noise
        zero = anchor * torch.zeros(*z.shape[:-1], 1)
        return u.tanh(), {
            "pre_tanh_mean": mean, "log_std": log_std, "pre_tanh_action": u,
            "log_prob": zero, "entropy": -zero,
        }

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.pi_optim, "step", lambda: None)
    monkeypatch.setattr(agent_module, "percentile_range", lambda *_a, **_kw: torch.tensor([4.0]))
    behavior_mean, behavior_log_std, valid = _behavior_batch(agent)
    behavior_mean[:] = torch.tensor([-0.1, 0.15, 0.3])
    behavior_log_std[:] = torch.tensor([-0.2, 0.1, -0.6])
    behavior_mean.requires_grad_()
    behavior_log_std.requires_grad_()

    # Antithetic cubature integrates the quadratic Gaussian term exactly.
    # The same samples evaluate the non-polynomial tanh Jacobian in both
    # estimators. This independently checks value AND actor gradient against
    # PyTorch's actual transformed distribution, including variance gradients.
    reference_u = (theta + offsets) + (0.1 * theta - 0.4 + offsets * 0.2).exp() * noise
    distribution = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(behavior_mean.detach(), behavior_log_std.detach().exp()),
        [torch.distributions.TanhTransform()],
    )
    joint_nll = -distribution.log_prob(reference_u.tanh()).sum(-1)
    weights = agent.cfg.rho ** torch.arange(agent.cfg.train_unroll_horizon)
    expected = ((joint_nll.mean(1) * weights).sum() / weights.sum()) * (3.0 / 61.0) / 4.0
    expected_gradient, = torch.autograd.grad(expected, theta)

    metrics = agent._update_actor(zs, behavior_mean, behavior_log_std, valid)
    torch.testing.assert_close(metrics["actor_loss"], expected.detach(), rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(parameter.grad.flatten()[0], expected_gradient, rtol=2e-5, atol=1e-7)
    assert behavior_mean.grad is behavior_log_std.grad is None
    assert all(p.grad is None for p in agent.model._Qs.parameters())
    assert len(calls) == 1


@pytest.mark.parametrize("valid_count", [0, 1, 3])
def test_missing_nan_labels_are_masked_and_readiness_is_respected(monkeypatch, valid_count):
    agent = _agent(sac_actor_loss_scale_tau=1.0, outer_behavior_policy_kl_min_valid_count=2)
    zs = torch.zeros(agent.cfg.train_unroll_horizon + 1, agent.cfg.batch_size, agent.cfg.latent_dim)
    mean, log_std, valid = _behavior_batch(agent, valid=False)
    mean.fill_(float("nan"))
    log_std.fill_(float("nan"))
    valid.reshape(-1)[:valid_count] = True
    mean[valid.squeeze(-1)] = 0.0
    log_std[valid.squeeze(-1)] = -0.5
    monkeypatch.setattr(agent_module, "percentile_range", lambda *_a, **_kw: torch.tensor([4.0]))
    metrics = agent._update_actor(zs, mean, log_std, valid)
    assert torch.isfinite(metrics["actor_loss"])
    assert metrics["behavior_policy_action_ce_ready"] == float(valid_count >= 2)
    if valid_count < 2:
        assert metrics["behavior_policy_action_ce_weighted_loss"] == 0
        assert metrics["behavior_policy_action_ce_joint_coefficient"] == 0
    assert all(torch.isfinite(p.grad).all() for p in agent.model._pi.parameters() if p.grad is not None)


def test_weighting_has_distinct_identity_but_constant_preserves_old_identity():
    assert _identity({}) == _identity({"outer_behavior_policy_weighting": "constant"})
    assert _identity({}) == _identity({"outer_behavior_policy_weighting": "CONSTANT"})
    active = _recipe()
    assert _identity(active) != _identity({**active, "outer_behavior_policy_weighting": "constant"})
    assert _identity(active) == _identity({**active, "outer_behavior_policy_weighting": "TDMPC_SQUARE"})


@pytest.mark.parametrize("exact", [False, True])
def test_checkpoint_roundtrip_and_atomic_rejection_of_different_weighting(exact):
    source = _agent()
    source.actor_loss_scale.fill_(3.25)
    state = source.training_state_dict() if exact else source.checkpoint_state()
    restored = _agent()
    load = restored.load_training_state_dict if exact else restored.load
    load(state)
    torch.testing.assert_close(restored.actor_loss_scale, source.actor_loss_scale)
    assert restored._behavior_policy_kl_spec() == source._behavior_policy_kl_spec()
    assert restored._actor_loss_scale_spec()["application"] == "q_and_behavior_cross_entropy"
    assert not restored._behavior_policy_kl_spec()["extra_logprob_action_dim_multiplier"]

    for incoming, target in [
        (source, _agent(outer_behavior_policy_weighting="constant")),
        (_agent(outer_behavior_policy_weighting="constant"), _agent()),
    ]:
        before = _clone_tree(target.training_state_dict())
        incoming_state = incoming.training_state_dict() if exact else incoming.checkpoint_state()
        with pytest.raises(ValueError, match="specification"):
            (target.load_training_state_dict if exact else target.load)(incoming_state)
        _assert_tree_equal(target.training_state_dict(), before)


def test_full_outer_training_resumes_with_identical_next_update():
    source = _agent()
    horizon, batch, dimension = (
        source.cfg.train_unroll_horizon, source.cfg.batch_size, source.cfg.action_dim,
    )
    data = (
        torch.randn(horizon + 1, batch, 3),
        torch.randn(horizon, batch, dimension).tanh(),
        torch.randn(horizon, batch, 1),
        torch.zeros(horizon, batch, 1),
    )
    mean, log_std, valid = _behavior_batch(source)
    labels = dict(
        behavior_pre_tanh_mean=mean,
        behavior_log_std=log_std,
        behavior_policy_valid=valid,
    )
    source.actor_loss_scale.fill_(3.2)
    first = source._update(*data, **labels)
    assert first["behavior_policy_action_ce_gate_active"] == 1
    assert first["behavior_policy_action_ce_joint_coefficient"] > 0
    assert torch.isfinite(torch.as_tensor(first["actor_loss"]))

    restored = _agent()
    restored.load_training_state_dict(_clone_tree(source.training_state_dict()))
    rng = torch.random.get_rng_state().clone()
    expected_metrics = source._update(*data, **labels)
    expected_state = _clone_tree(source.training_state_dict())
    torch.random.set_rng_state(rng)
    actual_metrics = restored._update(*data, **labels)
    _assert_tree_equal(restored.training_state_dict(), expected_state)
    torch.testing.assert_close(actual_metrics["actor_loss"], expected_metrics["actor_loss"])
