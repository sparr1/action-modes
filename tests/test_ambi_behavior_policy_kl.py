import math

import pytest
import torch

import RL.tdmpc2_core.ambi_agent as ambi_agent_module
from RL.tdmpc2_core.common import math as td_math
from tests.test_ambi_inner_decoupling import _model


def _kl_model(schedule, **overrides):
    params = {
        "outer_behavior_policy_kl_schedule": schedule,
        "outer_behavior_policy_kl_min_valid_count": 1,
        "outer_behavior_policy_kl_ramp_updates": 4,
        "ent_coef": 0.5,
    }
    params.update(overrides)
    return _model(**params)


def _stub_actor(monkeypatch, agent, *, current_mean=1.0, q_values=None):
    steps = int(agent.cfg.train_unroll_horizon) + 1
    batch = int(agent.cfg.batch_size)
    if q_values is None:
        q_values = torch.zeros(steps, batch, 1, device=agent.device)

    def policy(z):
        anchor = next(agent.model._pi.parameters()).reshape(-1)[0]
        shape = (*z.shape[:-1], agent.cfg.action_dim)
        mean = torch.full(shape, current_mean, device=z.device, dtype=z.dtype)
        mean = mean + anchor * 0.0
        log_std = torch.zeros_like(mean) + anchor * 0.0
        action = torch.tanh(mean)
        log_prob = torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype)
        log_prob = log_prob + anchor * 0.0
        return action, {
            "pre_tanh_mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": -log_prob,
        }

    def critic(z, action, **kwargs):
        values = q_values.to(device=z.device, dtype=z.dtype)
        values = values + action[..., :1] * 0.0
        if kwargs.get("reduction") == "all":
            return values.unsqueeze(0).expand(int(agent.cfg.num_q), *values.shape)
        return values

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", critic)
    return torch.zeros(steps, batch, agent.cfg.latent_dim, device=agent.device)


def _behavior_batch(agent, *, valid=True, mean=0.0):
    shape = (
        int(agent.cfg.train_unroll_horizon),
        int(agent.cfg.batch_size),
        int(agent.cfg.action_dim),
    )
    behavior_mean = torch.full(shape, mean, device=agent.device)
    behavior_log_std = torch.zeros(shape, device=agent.device)
    mask = torch.full(
        shape[:-1] + (1,), valid, dtype=torch.bool, device=agent.device
    )
    return behavior_mean, behavior_log_std, mask


def test_diagonal_gaussian_reverse_kl_matches_torch_and_optional_action_sum():
    current_mean = torch.tensor(
        [[[0.2, -0.3, 0.7]], [[-0.5, 0.1, 0.4]]], dtype=torch.float64
    )
    current_log_std = torch.tensor(
        [[[-1.2, 0.3, -0.7]], [[0.2, -0.4, 0.8]]], dtype=torch.float64
    )
    behavior_mean = torch.tensor(
        [
            [[0.0, 0.5, -0.2], [0.4, -0.1, 0.8]],
            [[-0.2, 0.3, 0.1], [0.9, -0.7, 0.0]],
        ],
        dtype=torch.float64,
    )
    behavior_log_std = torch.tensor(
        [
            [[-0.5, 0.1, -1.0], [0.6, -0.8, 0.2]],
            [[-1.1, 0.7, -0.3], [0.0, -0.2, 0.4]],
        ],
        dtype=torch.float64,
    )

    elementwise = td_math.diagonal_gaussian_reverse_kl(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
        sum_action_dim=False,
    )
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(
            current_mean, current_log_std.exp()
        ),
        torch.distributions.Normal(
            behavior_mean, behavior_log_std.exp()
        ),
    )
    torch.testing.assert_close(elementwise, expected)
    assert elementwise.shape == (2, 2, 3)

    summed = td_math.diagonal_gaussian_reverse_kl(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
    )
    torch.testing.assert_close(summed, expected.sum(dim=-1, keepdim=True))
    assert summed.shape == (2, 2, 1)


def test_diagonal_gaussian_reverse_kl_detaches_behavior_and_has_current_gradients():
    current_mean = torch.tensor(
        [[0.4, -0.2, 0.7]], dtype=torch.float64, requires_grad=True
    )
    current_log_std = torch.tensor(
        [[-0.3, 0.5, -1.1]], dtype=torch.float64, requires_grad=True
    )
    behavior_mean = torch.tensor(
        [[-0.1, 0.6, 0.2]], dtype=torch.float64, requires_grad=True
    )
    behavior_log_std = torch.tensor(
        [[0.2, -0.4, -0.7]], dtype=torch.float64, requires_grad=True
    )

    loss = td_math.diagonal_gaussian_reverse_kl(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
    ).sum()
    loss.backward()

    expected_mean_grad = (current_mean.detach() - behavior_mean.detach()) * torch.exp(
        -2.0 * behavior_log_std.detach()
    )
    expected_log_std_grad = torch.exp(
        2.0 * (current_log_std.detach() - behavior_log_std.detach())
    ) - 1.0
    torch.testing.assert_close(current_mean.grad, expected_mean_grad)
    torch.testing.assert_close(current_log_std.grad, expected_log_std_grad)
    assert behavior_mean.grad is None
    assert behavior_log_std.grad is None


def test_diagonal_gaussian_reverse_kl_is_finite_at_policy_log_std_extremes():
    current_mean = torch.tensor(
        [[0.25, -0.5, 0.0, 1.0]], dtype=torch.float32, requires_grad=True
    )
    current_log_std = torch.tensor(
        [[2.0, -20.0, 2.0, -20.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    behavior_mean = torch.tensor(
        [[0.0, 0.25, 0.0, 1.0]], dtype=torch.float32, requires_grad=True
    )
    behavior_log_std = torch.tensor(
        [[-20.0, 2.0, 2.0, -20.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    elementwise = td_math.diagonal_gaussian_reverse_kl(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
        sum_action_dim=False,
    )
    current_mean_64 = current_mean.detach().double()
    current_log_std_64 = current_log_std.detach().double()
    behavior_mean_64 = behavior_mean.detach().double()
    behavior_log_std_64 = behavior_log_std.detach().double()
    expected_64 = (
        behavior_log_std_64
        - current_log_std_64
        + 0.5
        * torch.exp(2.0 * (current_log_std_64 - behavior_log_std_64))
        + 0.5
        * (current_mean_64 - behavior_mean_64).square()
        * torch.exp(-2.0 * behavior_log_std_64)
        - 0.5
    )

    assert torch.isfinite(elementwise).all()
    assert (elementwise >= 0.0).all()
    assert elementwise[0, 0] > 1e18
    torch.testing.assert_close(
        elementwise.double(), expected_64, rtol=2e-6, atol=1e-6
    )

    elementwise.sum().backward()
    assert torch.isfinite(current_mean.grad).all()
    assert torch.isfinite(current_log_std.grad).all()
    assert behavior_mean.grad is None
    assert behavior_log_std.grad is None


def test_diagonal_gaussian_reverse_kl_rejects_non_boolean_reduction_flag():
    value = torch.zeros(1, 2)
    with pytest.raises(TypeError, match="sum_action_dim must be bool"):
        td_math.diagonal_gaussian_reverse_kl(
            value,
            value,
            value,
            value,
            sum_action_dim="yes",
        )


def test_shared_tanh_leaves_the_samplewise_log_density_ratio_unchanged():
    current = torch.distributions.Normal(
        torch.tensor([0.4, -0.7]), torch.tensor([0.8, 1.3])
    )
    behavior = torch.distributions.Normal(
        torch.tensor([-0.2, 0.1]), torch.tensor([1.1, 0.6])
    )
    pre_tanh = torch.tensor([0.25, -1.2])
    action = torch.tanh(pre_tanh)
    log_jacobian = torch.log1p(-action.square())

    unsquashed_ratio = current.log_prob(pre_tanh) - behavior.log_prob(pre_tanh)
    squashed_ratio = (
        current.log_prob(pre_tanh) - log_jacobian
    ) - (behavior.log_prob(pre_tanh) - log_jacobian)
    torch.testing.assert_close(squashed_ratio, unsquashed_ratio)


def test_behavior_kl_masks_invalid_rows_and_excludes_terminal_actor_state():
    agent = _kl_model("smooth").agent
    # The loss helper is independent of the environment actor head; use two
    # coordinates here to make the per-action-dimension normalization visible.
    agent.cfg.action_dim = 2
    steps = int(agent.cfg.train_unroll_horizon) + 1
    batch = int(agent.cfg.batch_size)
    current_mean = torch.zeros(steps, batch, agent.cfg.action_dim)
    current_mean[0, 0] = 1.0
    current_mean[1, 0] = 2.0
    current_mean[-1, 1:] = 1_000.0
    policy_info = {
        "pre_tanh_mean": current_mean.requires_grad_(),
        "log_std": torch.zeros_like(current_mean, requires_grad=True),
    }
    behavior_mean, behavior_log_std, valid = _behavior_batch(agent, valid=False)
    valid[0, 0] = True
    valid[1, 0] = True

    kl, ready, metrics = agent._behavior_policy_kl_loss(
        policy_info, behavior_mean, behavior_log_std, valid
    )

    weights = agent._transition_temporal_weights
    expected = (weights[0] * 0.5 + weights[1] * 2.0) / weights.sum()
    torch.testing.assert_close(kl, expected)
    assert ready
    assert metrics["behavior_policy_kl_valid_count"] == 2.0
    kl.backward()
    assert policy_info["pre_tanh_mean"].grad[-1].abs().sum() == 0


def test_smooth_schedule_uses_smoothstep_and_pauses_on_unready_batches(monkeypatch):
    agent = _kl_model(
        "smooth",
        outer_behavior_policy_kl_coef=2.0,
        outer_behavior_policy_kl_min_valid_count=2,
    ).agent
    zs = _stub_actor(monkeypatch, agent)
    behavior = _behavior_batch(agent)

    first = agent._update_actor(zs, *behavior)
    assert first["behavior_policy_kl_ramp_progress"] == pytest.approx(0.25)
    assert first["behavior_policy_kl_coefficient"] == pytest.approx(0.3125)
    assert agent.behavior_policy_kl_eligible_updates == 1

    invalid = _behavior_batch(agent, valid=False)
    paused = agent._update_actor(zs, *invalid)
    assert paused["behavior_policy_kl_coefficient"] == 0.0
    assert paused["behavior_policy_kl_ramp_progress"] == pytest.approx(0.25)
    assert agent.behavior_policy_kl_eligible_updates == 1

    resumed = agent._update_actor(zs, *behavior)
    assert resumed["behavior_policy_kl_ramp_progress"] == pytest.approx(0.5)
    assert resumed["behavior_policy_kl_coefficient"] == pytest.approx(1.0)
    assert agent.behavior_policy_kl_eligible_updates == 2


def test_quantile_gate_is_strict_reversible_and_does_not_enable_loss_scaling(
    monkeypatch,
):
    agent = _kl_model(
        "quantile_gate",
        sac_actor_loss_scale_tau=1.0,
        outer_behavior_policy_kl_q_threshold=2.0,
    ).agent
    zs = _stub_actor(monkeypatch, agent)
    behavior = _behavior_batch(agent)
    ranges = iter((2.0, 2.01, 2.0))
    monkeypatch.setattr(
        ambi_agent_module,
        "percentile_range",
        lambda *_args, **_kwargs: torch.tensor(
            [next(ranges)], device=agent.device
        ),
    )

    equal = agent._update_actor(zs, *behavior)
    above = agent._update_actor(zs, *behavior)
    back_equal = agent._update_actor(zs, *behavior)

    assert not agent.actor_loss_scale_enabled
    assert "actor_loss_scale" not in equal
    assert equal["behavior_policy_kl_gate_active"] == 0.0
    assert equal["behavior_policy_kl_coefficient"] == 0.0
    assert above["behavior_policy_kl_gate_active"] == 1.0
    assert above["behavior_policy_kl_coefficient"] == pytest.approx(1.0)
    assert back_equal["behavior_policy_kl_gate_active"] == 0.0
    assert back_equal["behavior_policy_kl_coefficient"] == 0.0


@pytest.mark.parametrize(("behavior_mean", "direction"), [(0.0, "up"), (1.0, "down")])
def test_log_dual_moves_in_the_violation_direction_with_one_step_lag(
    monkeypatch, behavior_mean, direction
):
    agent = _kl_model(
        "dual",
        outer_behavior_policy_kl_target=0.1,
        outer_behavior_policy_kl_dual_init=0.1,
        outer_behavior_policy_kl_dual_lr=0.05,
    ).agent
    zs = _stub_actor(monkeypatch, agent)
    behavior = _behavior_batch(agent, mean=behavior_mean)

    metrics = agent._update_actor(zs, *behavior)

    assert metrics["behavior_policy_kl_coefficient"] == pytest.approx(0.1)
    updated = float(metrics["behavior_policy_kl_dual_coefficient"])
    assert (updated > 0.1) if direction == "up" else (updated < 0.1)
    assert metrics["behavior_policy_kl_dual_updated"] == 1.0
    assert agent.behavior_policy_kl_dual_updates == 1


def test_log_dual_freezes_on_unready_data_and_clamps_at_its_cap(monkeypatch):
    agent = _kl_model(
        "dual",
        outer_behavior_policy_kl_min_valid_count=2,
        outer_behavior_policy_kl_dual_init=0.1,
        outer_behavior_policy_kl_dual_lr=1.0,
        outer_behavior_policy_kl_dual_max=0.1,
    ).agent
    zs = _stub_actor(monkeypatch, agent)

    frozen = agent._update_actor(zs, *_behavior_batch(agent, valid=False))
    assert frozen["behavior_policy_kl_coefficient"] == 0.0
    assert frozen["behavior_policy_kl_dual_updated"] == 0.0
    assert agent.behavior_policy_kl_dual_updates == 0

    capped = agent._update_actor(zs, *_behavior_batch(agent))
    assert capped["behavior_policy_kl_coefficient"] == pytest.approx(0.1)
    assert capped["behavior_policy_kl_dual_coefficient"] == pytest.approx(0.1)
    assert capped["behavior_policy_kl_dual_cap_hit"] == 1.0


@pytest.mark.parametrize("schedule", ["smooth", "quantile_gate", "dual"])
def test_active_scheduler_portable_state_roundtrips(schedule):
    source = _kl_model(schedule).agent
    if schedule == "smooth":
        source.behavior_policy_kl_eligible_updates = 7
    elif schedule == "quantile_gate":
        source._actor_loss_scale_value.fill_(3.25)
    else:
        source.log_behavior_policy_kl_coef.data.fill_(math.log(0.7))
        source.behavior_policy_kl_dual_updates = 0
    state = source.checkpoint_state()
    assert state["checkpoint_version"] == 5

    restored = _kl_model(schedule).agent
    restored.load(state)
    assert restored._behavior_policy_kl_spec() == source._behavior_policy_kl_spec()
    if schedule == "smooth":
        assert restored.behavior_policy_kl_eligible_updates == 7
    elif schedule == "quantile_gate":
        torch.testing.assert_close(restored.actor_q_range, torch.tensor([3.25]))
    else:
        assert float(restored.log_behavior_policy_kl_coef.exp().detach()) == pytest.approx(
            0.7
        )


def test_behavior_kl_and_actor_scaling_share_one_q_tracker_and_scale_full_loss(
    monkeypatch,
):
    agent = _kl_model(
        "quantile_gate",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        sac_actor_loss_scale_tau=1.0,
        outer_behavior_policy_kl_q_threshold=2.0,
    ).agent
    zs = _stub_actor(monkeypatch, agent)
    calls = 0

    def fixed_range(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return torch.tensor([4.0], device=agent.device)

    monkeypatch.setattr(ambi_agent_module, "percentile_range", fixed_range)
    metrics = agent._update_actor(zs, *_behavior_batch(agent))

    assert calls == 1
    assert metrics["actor_loss_scale"] == pytest.approx(4.0)
    assert metrics["behavior_policy_kl"] == pytest.approx(0.5)
    assert metrics["behavior_policy_kl_weighted_loss"] == pytest.approx(0.125)
    assert metrics["actor_loss"] == pytest.approx(0.125)
    assert agent.checkpoint_state()["checkpoint_version"] == 6


def test_inner_execution_capture_uses_the_action_call_and_resolved_std_scale(
    monkeypatch,
):
    model = _kl_model(
        "smooth",
        inner_execution_std_scale=0.4,
        outer_behavior_policy_kl_min_valid_count=1,
    )
    agent = model.agent
    original_pi = agent.model.pi
    execution_calls = []

    def tracked_pi(*args, **kwargs):
        result = original_pi(*args, **kwargs)
        if kwargs.get("std_scale") == pytest.approx(0.4):
            execution_calls.append(result)
        return result

    monkeypatch.setattr(agent.model, "pi", tracked_pi)
    action, behavior = agent.act(
        torch.zeros(3),
        t0=True,
        eval_mode=False,
        collect_diagnostics=False,
        return_behavior_policy=True,
    )

    assert len(execution_calls) == 1
    executed_action, executed_info = execution_calls[0]
    torch.testing.assert_close(action, executed_action[0].cpu())
    torch.testing.assert_close(
        behavior["pre_tanh_mean"], executed_info["pre_tanh_mean"][0].cpu()
    )
    torch.testing.assert_close(
        behavior["log_std"], executed_info["log_std"][0].cpu()
    )


def test_outer_intervention_capture_comes_from_its_action_producing_call(
    monkeypatch,
):
    agent = _kl_model("smooth").agent
    original_pi = agent.model.pi
    calls = []

    def tracked_pi(*args, **kwargs):
        result = original_pi(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(agent.model, "pi", tracked_pi)
    action, behavior = agent.act_outer_policy(
        torch.zeros(3),
        generator=torch.Generator(device="cpu").manual_seed(9),
        return_behavior_policy=True,
    )

    assert len(calls) == 1
    sampled_action, sampled_info = calls[0]
    torch.testing.assert_close(action, sampled_action[0].cpu())
    torch.testing.assert_close(
        behavior["pre_tanh_mean"], sampled_info["pre_tanh_mean"][0].cpu()
    )
    torch.testing.assert_close(
        behavior["log_std"], sampled_info["log_std"][0].cpu()
    )


def test_inactive_inner_sac_still_captures_its_exact_outer_gaussian():
    model = _kl_model(
        "smooth",
        inner_rounds=0,
        inner_model_step_budget=0,
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
    )
    action, behavior = model.agent.act(
        torch.zeros(3),
        t0=True,
        eval_mode=False,
        collect_diagnostics=False,
        return_behavior_policy=True,
    )
    assert action.shape == (model.cfg.action_dim,)
    assert behavior is not None
    assert set(behavior) == {"pre_tanh_mean", "log_std"}

    _, eval_behavior = model.agent.act(
        torch.zeros(3),
        t0=False,
        eval_mode=True,
        collect_diagnostics=False,
        return_behavior_policy=True,
    )
    assert eval_behavior is None


def test_behavior_metadata_does_not_extend_tensor_rollout_lengths():
    action = torch.tensor([0.25])
    behavior = {
        "pre_tanh_mean": torch.tensor([0.75]),
        "log_std": torch.tensor([-1.25]),
    }
    _, _, lengths, materialized_behavior = (
        _kl_model("smooth").agent._materialize_action_metrics(
            action,
            {},
            torch.tensor([2, 1, 2]),
            behavior_policy=behavior,
        )
    )
    assert lengths == [2, 1, 2]
    torch.testing.assert_close(
        materialized_behavior["pre_tanh_mean"], behavior["pre_tanh_mean"]
    )


def test_episode_staging_aligns_captured_policy_with_action_row_and_masks_random():
    model = _kl_model("smooth")
    obs = torch.zeros(3)
    assert model._start_episode_staging(obs) == 1
    assert not bool(model._episode_staging["behavior_policy_valid"][0])

    model._pending_behavior_policy = {
        "pre_tanh_mean": torch.tensor([0.75]),
        "log_std": torch.tensor([-1.25]),
    }
    model._stage_transition(1, obs, torch.tensor([0.2]), 1.0, False)
    assert bool(model._episode_staging["behavior_policy_valid"][1])
    torch.testing.assert_close(
        model._episode_staging["behavior_pre_tanh_mean"][1], torch.tensor([0.75])
    )

    model._random_action_norm()
    model._stage_transition(2, obs, torch.tensor([0.1]), 0.0, False)
    assert not bool(model._episode_staging["behavior_policy_valid"][2])
    torch.testing.assert_close(
        model._episode_staging["behavior_log_std"][2], torch.zeros(1)
    )
