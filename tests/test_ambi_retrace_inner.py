"""Finite-horizon Retrace uses real SAC networks and action-local replay."""

from copy import deepcopy
import math

import pytest
import torch
import torch.nn.functional as F

from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_aux_return_inner import _prepared as _prepared_aux


@pytest.fixture
def make_model():
    opened = []

    def create(**overrides):
        options = dict(
            inner_sac_return_estimator="retrace", inner_finite_horizon=True,
            inner_update_timing="round", inner_rollout_horizon=3,
            train_unroll_horizon=3, inner_rounds=1,
            inner_rollouts_per_round=2, inner_updates_per_round=1,
            inner_replay_capacity=6,
        )
        options.update(overrides)
        if options.get("inner_critic_adaptation") == "lora_rl":
            options.setdefault("inner_critic_lora_rank", 2)
        holder = _tiny_model(**options)
        opened.append(holder)
        return holder

    yield create
    for holder in opened:
        holder.close()


def _prepare(holder):
    engine = holder.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def _collect(engine):
    result = engine._collect_round(torch.zeros(1, engine.cfg.latent_dim))
    count = int(engine.cfg.inner_rollouts_per_round)
    batch = engine.state.replay.sample_trajectories(
        count, indices=torch.arange(count), include_ids=True,
    )
    return batch, result


def _kernel(engine, batch, alpha=0.2):
    batch_size, horizon = batch["valid"].shape[:2]
    return engine._retrace_critic_kernel(
        batch, torch.tensor(alpha),
        torch.zeros(batch_size, horizon, engine.cfg.action_dim),
        torch.zeros(batch_size, engine.cfg.action_dim), torch.tensor([0, 1]),
    )


def _normal_log_prob(value, mean, log_std):
    return (-0.5 * ((value - mean) / math.exp(log_std)).square()
            - log_std - 0.5 * math.log(2.0 * math.pi)).sum(-1, keepdim=True)


@pytest.mark.parametrize("std_scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("mean", [0.2, 20.0])
def test_collection_density_and_changed_actor_ratio_use_saved_pre_tanh(
    make_model, std_scale, mean,
):
    engine = _prepare(make_model(
        inner_behavior_std_scale=std_scale, inner_retrace_lambda=0.8,
        inner_log_std_mapping="direct_clamp", inner_log_std_min=-4.0,
        inner_log_std_max=1.0,
    ))
    log_std = -0.7
    with torch.no_grad():
        engine.state.actor[-1].weight.zero_()
        engine.state.actor[-1].bias.copy_(torch.tensor([mean, log_std]))
    batch, result = _collect(engine)
    assert result["transition_count"] == 6
    pre_tanh = batch["pre_tanh_action"]
    torch.testing.assert_close(pre_tanh.tanh(), batch["action"], rtol=0, atol=0)
    if mean == 20.0:
        assert (batch["action"] == 1).all()
    assert torch.isfinite(batch["behavior_log_prob"]).all()
    log_mu_normal = _normal_log_prob(pre_tanh, mean, log_std + math.log(std_scale))
    log_jacobian = (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(-1, keepdim=True)
    torch.testing.assert_close(batch["behavior_log_prob"], log_mu_normal - log_jacobian)
    stored_density = batch["behavior_log_prob"].clone()
    old_targets = _kernel(engine, batch)[2]

    changed_mean = mean + 0.4
    with torch.no_grad():
        engine.state.actor[-1].bias[0] = changed_mean
    outputs = _kernel(engine, batch)
    # The same tanh Jacobian cancels. The target actor uses its unscaled std,
    # while mu retains the collection-time std even after actor adaptation.
    log_pi_normal = _normal_log_prob(pre_tanh, changed_mean, log_std)
    expected_c = 0.8 * (log_pi_normal - log_mu_normal).clamp_max(0).exp()
    torch.testing.assert_close(outputs[4], expected_c, rtol=2e-5, atol=2e-6)
    assert not torch.allclose(outputs[2], old_targets)
    torch.testing.assert_close(batch["behavior_log_prob"], stored_density, rtol=0, atol=0)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("target_semantics", ["reward_only", "entropy_augmented"])
@pytest.mark.parametrize("horizon,trace_lambda", [(1, 1.0), (3, 0.0)])
def test_h1_and_lambda_zero_match_existing_one_step_critic_kernel(
    make_model, representation, target_semantics, horizon, trace_lambda,
):
    engine = _prepare(make_model(
        q_representation=representation, inner_rollout_horizon=horizon,
        inner_retrace_lambda=trace_lambda, inner_sac_critic_target=target_semantics,
    ))
    batch, _ = _collect(engine)
    count = batch["valid"].shape[0]
    noise = torch.linspace(-0.7, 0.7, count * horizon).reshape(count, horizon, 1)
    prior_noise = torch.tensor([[0.3], [-0.4]])
    pair, alpha = torch.tensor([0, 1]), torch.tensor(0.31)
    actual = engine._retrace_critic_kernel(batch, alpha, noise, prior_noise, pair)
    expected = engine._sac_critic_kernel(
        *[batch[key].flatten(0, 1) for key in ("z", "action", "reward", "next_z", "terminated")],
        alpha, noise.flatten(0, 1), pair,
        horizon_end=batch["horizon_end"].flatten(0, 1),
        prior_noise=prior_noise[:, None, :].expand(-1, horizon, -1).reshape(-1, 1),
    )
    for retrace_value, one_step_value in zip(actual[:4], expected):
        torch.testing.assert_close(retrace_value, one_step_value)
    assert not actual[2].requires_grad
    assert torch.count_nonzero(actual[6]) == 0
    actual[0].backward()
    actual_gradients = [parameter.grad.clone() for parameter in engine.state.critic_params]
    engine.state.critic_optim.zero_grad(set_to_none=True)
    expected[0].backward()
    for parameter, gradient in zip(engine.state.critic_params, actual_gradients):
        torch.testing.assert_close(parameter.grad, gradient)


@pytest.mark.parametrize("horizon_critic", ["sac", "aux_return"])
def test_outer_boundary_replaces_soft_continuation_and_true_terminal_stops_it(
    horizon_critic,
):
    with _prepared_aux(
        inner_sac_return_estimator="retrace", inner_finite_horizon=True,
        inner_rollout_horizon=3, train_unroll_horizon=3,
        inner_retrace_lambda=0.0, inner_horizon_critic_source=horizon_critic,
    ) as (_, engine):
        batch, _ = _collect(engine)
        batch["reward"][:] = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1)
        batch["terminated"][1, 1] = 1.0
        batch["next_z"][1, 1] = float("nan")
        batch["valid"][1, 2] = False
        for key, value in batch.items():
            if value.is_floating_point() and value.ndim == 3:
                value[1, 2] = float("nan")
        cold, hot = _kernel(engine, batch, alpha=0.0), _kernel(engine, batch, alpha=100.0)
        cold_target, hot_target = cold[2].reshape(2, 3), hot[2].reshape(2, 3)
        frozen_value = 2.0 if horizon_critic == "sac" else 9.0
        expected_boundary = 3.0 + float(engine.agent.discount) * frozen_value
        assert hot_target[0, 2].item() == pytest.approx(expected_boundary)
        torch.testing.assert_close(hot_target[0, 2], cold_target[0, 2], rtol=0, atol=0)
        assert not torch.isclose(hot_target[0, 0], cold_target[0, 0])
        assert hot_target[1, 1].item() == 2.0
        assert hot_target[1, 2].item() == 0.0
        assert torch.isfinite(hot[0])
        assert torch.isfinite(hot[2]).all()


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_every_valid_suffix_receives_a_critic_gradient_and_padding_does_not(
    make_model, representation, monkeypatch,
):
    engine = _prepare(make_model(
        q_representation=representation, inner_sac_critic_target="reward_only",
    ))
    batch, _ = _collect(engine)
    batch["reward"][:] = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1)
    batch["terminated"][1, 1] = 1.0
    batch["valid"][1, 2] = False
    monkeypatch.setattr(engine, "_bootstrap_q", lambda z, action, **kwargs: z.new_zeros(z.shape[0], 1))
    monkeypatch.setattr(engine, "_prior_bootstrap", lambda z, noise: z.new_zeros(z.shape[0], 1))
    width = 1 if representation == "scalar" else engine.cfg.q_num_bins
    predictions = torch.zeros(engine.cfg.num_q, 6, width, requires_grad=True)
    monkeypatch.setattr(engine.model, "q_predictions", lambda *args, **kwargs: predictions)
    outputs = _kernel(engine, batch, alpha=0.0)
    gamma = float(engine.agent.discount)
    expected = torch.tensor([
        [1 + gamma * 2 + gamma ** 2 * 3, 2 + gamma * 3, 3],
        [1 + gamma * 2, 2, 0],
    ])
    torch.testing.assert_close(outputs[2].reshape(2, 3), expected)
    outputs[0].backward()
    per_row_gradient = predictions.grad.abs().sum((0, 2)).reshape(2, 3)
    valid = batch["valid"].squeeze(-1)
    assert (per_row_gradient[valid] > 0).all()
    assert (per_row_gradient[~valid] == 0).all()
    assert all(parameter.grad is None for parameter in engine.state.actor.parameters())
    assert all(parameter.grad is None for parameter in engine.state.critic_target.parameters())
    assert all(parameter.grad is None for parameter in engine.model.parameters())


def test_episodic_collection_keeps_only_valid_prefix_at_true_termination(make_model, monkeypatch):
    engine = _prepare(make_model(episodic=True))
    monkeypatch.setattr(engine.model, "termination", lambda z: z.new_ones(z.shape[0], 1))
    batch, result = _collect(engine)
    assert result["transition_count"] == 2
    assert result["lengths"].tolist() == [1, 1]
    assert batch["valid"].squeeze(-1).tolist() == [[True, False, False], [True, False, False]]
    assert batch["terminated"][:, 0].all()
    assert not batch["horizon_end"].any()
    target = _kernel(engine, batch)[2].reshape(2, 3, 1)
    torch.testing.assert_close(target[:, 0], batch["reward"][:, 0], rtol=0, atol=0)
    assert (target[:, 1:] == 0).all()


def test_episodic_branches_keep_identity_when_they_terminate_at_different_depths(
    make_model, monkeypatch,
):
    engine = _prepare(make_model(episodic=True))
    dynamics_calls = []

    def next_state(joint):
        dynamics_calls.append(joint.shape[0])
        if len(dynamics_calls) == 1:
            return torch.tensor([10.0, 20.0])[:, None].expand(-1, engine.cfg.latent_dim).clone()
        return joint.new_full((joint.shape[0], engine.cfg.latent_dim), 19 + len(dynamics_calls))

    def termination(z):
        return ((z[:, :1] == 10) | (z[:, :1] >= 22)).to(z.dtype)

    monkeypatch.setattr(engine.model, "next_from_joint", next_state)
    monkeypatch.setattr(engine.model, "termination", termination)
    batch, result = _collect(engine)
    assert dynamics_calls == [2, 1, 1]
    assert result["transition_count"] == 4
    assert result["lengths"].tolist() == [1, 3]
    assert batch["valid"].squeeze(-1).tolist() == [[True, False, False], [True, True, True]]
    assert batch["terminated"].squeeze(-1).tolist() == [[1, 0, 0], [0, 0, 1]]
    assert batch["horizon_end"].squeeze(-1).tolist() == [[0, 0, 0], [0, 0, 1]]
    torch.testing.assert_close(batch["z"][1, :, 0], torch.tensor([0.0, 20.0, 21.0]))
    torch.testing.assert_close(batch["next_z"][1, :, 0], torch.tensor([20.0, 21.0, 22.0]))
    target = _kernel(engine, batch)[2].reshape(2, 3, 1)
    torch.testing.assert_close(target[0, 0], batch["reward"][0, 0], rtol=0, atol=0)
    torch.testing.assert_close(target[1, 2], batch["reward"][1, 2], rtol=0, atol=0)
    assert (target[0, 1:] == 0).all()




def test_target_value_and_stored_action_baseline_share_pair_but_outer_has_own_reduction(
    make_model, monkeypatch,
):
    engine = _prepare(make_model(
        q_representation="distributional", num_q=5,
        mppi_terminal_q_reduction="mean_all",
    ))
    batch, _ = _collect(engine)
    original_q, calls = engine.model.Q, []

    def record_q(z, action, **kwargs):
        calls.append(kwargs)
        return original_q(z, action, **kwargs)

    monkeypatch.setattr(engine.model, "Q", record_q)
    pair = torch.tensor([1, 3])
    engine._retrace_critic_kernel(
        batch, torch.tensor(0.2), torch.zeros(2, 3, 1), torch.zeros(2, 1), pair,
    )
    inner_calls = [call for call in calls if call.get("qs") is engine.state.critic_target]
    assert len(inner_calls) == 2
    for call in inner_calls:
        assert call["pair_indices"] is pair
        assert call["trusted_pair_indices"] is True
        assert call["reduction"] == "min_pair"
    outer_calls = [call for call in calls if call.get("qs") is not engine.state.critic_target]
    assert len(outer_calls) == 1
    assert outer_calls[0]["reduction"] == "mean_all"
    assert outer_calls[0].get("pair_indices") is None


def test_actor_keeps_transition_batch_and_critic_diagnostics_sum_all_suffix_rows(
    make_model, monkeypatch,
):
    holder = make_model(
        inner_batch_size=5, inner_replay_capacity=14,
        inner_rounds=2, inner_updates_per_round=2,
    )
    engine = holder.agent.inner_engine
    critic_batches, policy_batches = [], []
    original_critic, original_policy = engine._retrace_critic_step, engine._sac_policy_step

    def record_critic(batch, *args, **kwargs):
        critic_batches.append((batch["valid"].shape[0], int(batch["valid"].sum())))
        return original_critic(batch, *args, **kwargs)

    def record_policy(batch, *args, **kwargs):
        policy_batches.append(batch["action"].shape[0])
        assert batch["action"].ndim == 2
        return original_policy(batch, *args, **kwargs)

    monkeypatch.setattr(engine, "_retrace_critic_step", record_critic)
    monkeypatch.setattr(engine, "_sac_policy_step", record_policy)
    holder.agent.act(torch.zeros(3), t0=True, collect_diagnostics=True)
    assert critic_batches == [(2, 6)] * 4
    assert policy_batches == [5] * 4
    metrics = holder.agent.last_inner_metrics
    assert metrics["inner_retrace_trajectory_draws"] == 8
    assert metrics["inner_retrace_critic_rows"] == 24
    assert metrics["inner_retrace_requested_capacity"] == 14
    assert metrics["inner_retrace_effective_capacity"] == 12
    assert metrics["inner_retrace_replay_trajectories"] == 4


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("options", [
    {},
    {"inner_actor_initialization": "random", "inner_critic_initialization": "random"},
    {"inner_critic_adaptation": "lora_rl"},
    {"aux_return_mode": "return_actor", "inner_actor_source": "return_actor",
     "inner_critic_source": "aux_return", "inner_horizon_critic_source": "aux_return"},
])
def test_real_retrace_solve_updates_private_critic_and_preserves_outer_state(
    make_model, representation, options,
):
    holder = make_model(q_representation=representation, **options)
    agent = holder.agent
    outer = deepcopy(agent.model.state_dict())
    optimizers = [agent.optim, agent.pi_optim, agent.ent_coef_optim]
    optimizer_states = [deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    alpha, counters = agent.alpha.detach().clone(), (agent.num_updates, agent.outer_version)
    action = agent.act(torch.zeros(3), t0=True, collect_diagnostics=True)
    metrics = agent.last_inner_metrics
    assert torch.isfinite(action).all()
    assert metrics["inner_model_steps"] == 6
    assert metrics["inner_critic_optimizer_steps"] == 1
    assert metrics["inner_actor_optimizer_steps"] == 1
    pool = agent.inner_engine._action_pool
    assert pool.critic_optim.state
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in pool.critic.parameters())
    _assert_tree_equal(agent.model.state_dict(), outer)
    for optimizer, previous in zip(optimizers, optimizer_states):
        _assert_tree_equal(optimizer.state_dict(), previous)
    torch.testing.assert_close(agent.alpha, alpha, rtol=0, atol=0)
    assert (agent.num_updates, agent.outer_version) == counters
    assert all(parameter.grad is None for parameter in agent.model.parameters())


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_sampled_reward_retrace_keeps_critic_first_order_and_transition_batch_budget(horizon, monkeypatch):
    from tests.test_ambi_root_local_sac import _tiny_component_model
    from tests.test_ambi_eval_execution import _capture_execution

    holder = _tiny_component_model(
        inner_sac_return_estimator='retrace', inner_retrace_lambda=.9,
        inner_finite_horizon=True, inner_rollout_horizon=horizon, train_unroll_horizon=3,
        inner_rounds=2, inner_rollouts_per_round=128, inner_batch_size=256,
        inner_critic_updates_per_round=2, inner_actor_updates_per_round=1,
        inner_replay_capacity=3840, inner_temperature_mode='auto',
        aux_return_mode='sac', inner_critic_source='aux_return',
        inner_horizon_critic_source='aux_return', inner_sac_critic_target='reward_only',
        inner_terminal_entropy='none', inner_eval_execution_action='policy_sample',
    )
    try:
        engine = holder.agent.inner_engine
        captured = _capture_execution(monkeypatch, engine)
        phases = []
        critic_step, policy_step = engine._retrace_critic_step, engine._sac_policy_step
        trajectories = (256 + horizon - 1) // horizon

        def critic(batch, *args, **kwargs):
            phases.append('critic')
            assert batch['valid'].shape[:2] == (trajectories, horizon)
            return critic_step(batch, *args, **kwargs)

        def policy(batch, *args, **kwargs):
            phases.append('actor')
            assert batch['z'].shape[0] == 256
            return policy_step(batch, *args, **kwargs)

        monkeypatch.setattr(engine, '_retrace_critic_step', critic)
        monkeypatch.setattr(engine, '_sac_policy_step', policy)
        outer = deepcopy(holder.agent.model.state_dict())
        global_rng = torch.random.get_rng_state().clone()
        action, _ = holder.predict(torch.zeros(3).numpy(), deterministic=True, episode_start=True)
        assert phases == ['critic', 'critic', 'actor'] * 2
        expected = holder._unscale_action(captured[0]['sample'].numpy())
        torch.testing.assert_close(torch.as_tensor(action), torch.as_tensor(expected), rtol=0, atol=0)
        assert captured[0]['eval_mode'] is True
        metrics = holder.agent.last_inner_metrics
        assert metrics['inner_eval_execution_sampled'] == 1
        assert metrics['inner_model_steps'] == metrics['inner_buffer_size'] == 256 * horizon
        assert metrics['inner_retrace_trajectory_draws'] == 4 * trajectories
        assert metrics['inner_retrace_critic_rows'] == 4 * trajectories * horizon
        assert metrics['inner_critic_optimizer_steps'] == 4
        assert metrics['inner_actor_optimizer_steps'] == metrics['inner_temperature_optimizer_steps'] == 2
        if horizon == 1:
            assert metrics['inner_retrace_effective_trace_length'] == 1
            assert metrics['inner_retrace_correction_abs_mean'] == 0
        _assert_tree_equal(holder.agent.model.state_dict(), outer)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        holder.close()
