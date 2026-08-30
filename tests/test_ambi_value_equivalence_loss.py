import inspect
from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common import math as td_math


def _params(horizon=1, **overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "num_enc_layers": 2,
        "simnorm_dim": 8,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": horizon,
        "outer_planning_horizon": 2,
        "buffer_size": 100,
        "seed_steps": 4,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "wandb": False,
        "dropout": 0.0,
        "q_representation": "scalar",
        "num_q": 2,
        "q_pair_size": 2,
        "q_num_bins": 11,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 1,
        "inner_updates_per_round": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 8,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "inherit_outer",
        "inner_temperature_initialization": "inherit_outer",
        "inner_critic_target_tau": 1.0,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "value_equivalence_diagnostics": False,
        "value_equivalence_loss_coef": 0.0,
        "value_equivalence_loss_mc_samples": 1,
    }
    params.update(overrides)
    return params


def _model(horizon=1, **overrides):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    params = _params(horizon, **overrides)
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        params,
        {
            "seed": 29,
            "device": params["device"],
            "env": "test",
            "total_steps": 10,
        },
        {},
    )


def _loss_inputs(agent, horizon=1, batch_size=2, *, requires_grad=False):
    latent_states = torch.zeros(
        horizon + 1,
        batch_size,
        int(agent.cfg.latent_dim),
        device=agent.device,
    )
    if requires_grad:
        latent_states.requires_grad_()
    next_z_targets = torch.zeros_like(latent_states[1:])
    return latent_states, next_z_targets


def _update_batch(agent):
    horizon = int(agent.cfg.train_unroll_horizon)
    batch_size = int(agent.cfg.batch_size)
    obs_dim = int(agent.cfg.obs_shape["state"][0])
    obs = torch.randn(horizon + 1, batch_size, obs_dim)
    action = torch.randn(horizon, batch_size, agent.cfg.action_dim).tanh()
    reward = torch.randn(horizon, batch_size, 1)
    terminated = torch.zeros_like(reward)
    return obs, action, reward, terminated


def _patch_probe(monkeypatch, agent, *, value, log_prob=None):
    def fake_pi(z, *args, **kwargs):
        assert kwargs.get("detach_policy") is True
        action = z.new_zeros(*z.shape[:-1], int(agent.cfg.action_dim))
        resolved_log_prob = (
            z.new_zeros(*z.shape[:-1], 1)
            if log_prob is None
            else log_prob(z)
        )
        return action, {"log_prob": resolved_log_prob}

    def fake_q(critic, z, action, reduction, pair_indices, *args, **kwargs):
        return value(z, action)

    monkeypatch.setattr(agent.model, "pi", fake_pi)
    monkeypatch.setattr(agent, "_value_equivalence_q_with_input_grad", fake_q)


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return deepcopy(value)


def _assert_tree_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _metric(metrics, key):
    return float(torch.as_tensor(metrics[key]))


def test_value_equivalence_loss_is_exact_value_only_raw_mse(monkeypatch):
    model = _model(
        inner_sac_critic_target="reward_only",
        temporal_loss_normalization="divide_horizon",
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=1,
    )
    agent = model.agent
    agent.discount = 0.5
    _patch_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z[..., :1],
    )
    latent_states, next_z_targets = _loss_inputs(agent)
    # Model/reference successor-value differences are [4, -2], so the
    # discounted residuals are [2, -1] and their raw MSE is 5/2.
    latent_states[1, :, 0] = torch.tensor([5.0, 4.0])
    next_z_targets[0, :, 0] = torch.tensor([1.0, 6.0])

    raw_loss, per_depth = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=7,
    )

    assert tuple(per_depth.shape) == (1,)
    assert float(per_depth[0]) == pytest.approx(2.5)
    assert float(raw_loss) == pytest.approx(2.5)


@pytest.mark.parametrize("horizon", [1, 6])
def test_value_equivalence_loss_uses_told_temporal_reduction(
    monkeypatch,
    horizon,
):
    model = _model(
        horizon=horizon,
        rho=0.5,
        temporal_loss_normalization="reference_weighted_mean",
        temporal_loss_reference_horizon=3,
        inner_sac_critic_target="reward_only",
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=1,
    )
    agent = model.agent
    agent.discount = 1.0
    _patch_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z[..., :1],
    )
    latent_states, next_z_targets = _loss_inputs(agent, horizon=horizon)
    residual = torch.arange(
        1,
        horizon + 1,
        dtype=latent_states.dtype,
        device=latent_states.device,
    )
    latent_states[1:, :, 0] = residual[:, None]

    raw_loss, per_depth = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=11,
    )
    expected_per_depth = residual.square()
    expected_raw = td_math.reduce_temporal_loss(
        expected_per_depth,
        agent.cfg.rho,
        normalization=agent.cfg.temporal_loss_normalization,
        reference_horizon=agent.cfg.temporal_loss_reference_horizon,
        legacy_order="vector_sum_divide",
        weights=agent._transition_temporal_weights,
    )

    torch.testing.assert_close(per_depth, expected_per_depth)
    torch.testing.assert_close(raw_loss, expected_raw)


def test_value_equivalence_loss_averages_shared_mc_probes_before_square(
    monkeypatch,
):
    model = _model(
        q_pair_size=1,
        inner_sac_critic_target="reward_only",
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=4,
    )
    agent = model.agent
    agent.discount = 1.0
    latent_states, next_z_targets = _loss_inputs(agent)
    latent_states[1, :, 0] = 1.0

    coefficients = (-3.0, -1.0, 1.0, 3.0)
    policy_noises = []
    q_pairs = []
    sampled_pairs = []

    def fake_pi(z, *args, **kwargs):
        assert kwargs.get("detach_policy") is True
        noise = kwargs.get("noise")
        assert noise is not None
        assert z.shape[0] == 2
        torch.testing.assert_close(noise[0], noise[1], rtol=0, atol=0)
        policy_noises.append(noise.detach().clone())
        coefficient = coefficients[len(policy_noises) - 1]
        # Common noise does not imply a common action: each branch applies the
        # same actor/noise coupling to its own successor latent.
        action = coefficient * z[..., : int(agent.cfg.action_dim)]
        return action, {"log_prob": z.new_zeros(*z.shape[:-1], 1)}

    def fake_q(critic, z, action, reduction, pair_indices, *args, **kwargs):
        assert pair_indices is not None
        q_pairs.append(pair_indices.detach().clone())
        return action[..., :1]

    def fake_sample_pair_indices(device, *, generator=None):
        assert generator is not None
        pair = torch.tensor(
            [len(sampled_pairs) % 2],
            dtype=torch.long,
            device=device,
        )
        sampled_pairs.append(pair.detach().clone())
        return pair

    monkeypatch.setattr(agent.model, "pi", fake_pi)
    monkeypatch.setattr(agent, "_value_equivalence_q_with_input_grad", fake_q)
    monkeypatch.setattr(
        agent.model.q_backend,
        "sample_pair_indices",
        fake_sample_pair_indices,
    )

    raw_loss, per_depth = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=13,
    )

    assert len(policy_noises) == len(q_pairs) == len(sampled_pairs) == 4
    for actual, expected in zip(q_pairs, sampled_pairs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Paired residuals are [-3, -1, 1, 3]. Their mean is zero although their
    # mean square is five, distinguishing Bellman-operator error from target
    # sampling variance.
    torch.testing.assert_close(per_depth, torch.zeros_like(per_depth))
    torch.testing.assert_close(raw_loss, torch.zeros_like(raw_loss))


@pytest.mark.parametrize(
    ("critic_target", "expected"),
    [("reward_only", 0.0), ("entropy_augmented", 1.0)],
)
def test_value_equivalence_loss_uses_configured_soft_value_probe(
    monkeypatch,
    critic_target,
    expected,
):
    model = _model(
        ent_coef=2.0,
        inner_temperature_mode="inherit_outer",
        inner_sac_critic_target=critic_target,
        temporal_loss_normalization="divide_horizon",
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=1,
    )
    agent = model.agent
    agent.discount = 0.5
    _patch_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z.new_zeros(*z.shape[:-1], 1),
        log_prob=lambda z: z[..., :1],
    )
    latent_states, next_z_targets = _loss_inputs(agent)
    latent_states[1, :, 0] = 2.0
    next_z_targets[0, :, 0] = 1.0

    raw_loss, per_depth = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=17,
    )

    assert float(per_depth[0]) == pytest.approx(expected)
    assert float(raw_loss) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("temperature_mode", "initialization", "expected_alpha"),
    [
        ("inherit_outer", "fixed", 2.0),
        ("fixed", "inherit_outer", 0.25),
        ("auto", "inherit_outer", 2.0),
        ("auto", "fixed", 0.25),
    ],
)
def test_value_equivalence_loss_uses_fresh_inner_alpha_modes(
    monkeypatch,
    temperature_mode,
    initialization,
    expected_alpha,
):
    model = _model(
        ent_coef=2.0,
        inner_temperature_mode=temperature_mode,
        inner_temperature_initialization=initialization,
        inner_temperature=0.25,
        inner_sac_critic_target="entropy_augmented",
        temporal_loss_normalization="divide_horizon",
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=1,
    )
    agent = model.agent
    agent.discount = 0.5
    _patch_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z.new_zeros(*z.shape[:-1], 1),
        log_prob=lambda z: z[..., :1],
    )
    latent_states, next_z_targets = _loss_inputs(agent)
    latent_states[1, :, 0] = 2.0
    next_z_targets[0, :, 0] = 1.0

    raw_loss, per_depth = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=18,
    )

    expected_loss = (0.5 * expected_alpha) ** 2
    assert float(per_depth[0]) == pytest.approx(expected_loss)
    assert float(raw_loss) == pytest.approx(expected_loss)


@pytest.mark.parametrize(
    ("bootstrap_source", "expected_critic"),
    [
        ("inner_target", "_Qs"),
        ("outer_online", "_Qs"),
        ("outer_target", "_target_Qs"),
    ],
)
@pytest.mark.parametrize("reduction", ["min_pair", "mean_pair", "min_all", "mean_all"])
def test_value_equivalence_loss_uses_configured_source_and_reduction(
    monkeypatch,
    bootstrap_source,
    expected_critic,
    reduction,
):
    model = _model(
        inner_bootstrap_source=bootstrap_source,
        inner_q_target_reduction=reduction,
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=1,
    )
    agent = model.agent
    latent_states, next_z_targets = _loss_inputs(agent)
    calls = []

    def fake_pi(z, *args, **kwargs):
        assert kwargs.get("detach_policy") is True
        return (
            z.new_zeros(*z.shape[:-1], int(agent.cfg.action_dim)),
            {"log_prob": z.new_zeros(*z.shape[:-1], 1)},
        )

    def fake_q(critic, z, action, actual_reduction, pair_indices, *args, **kwargs):
        calls.append((critic, actual_reduction, pair_indices))
        return z.new_zeros(*z.shape[:-1], 1)

    monkeypatch.setattr(agent.model, "pi", fake_pi)
    monkeypatch.setattr(agent, "_value_equivalence_q_with_input_grad", fake_q)

    agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=19,
    )

    assert len(calls) == 1
    critic, actual_reduction, pair_indices = calls[0]
    assert critic is getattr(agent.model, expected_critic)
    assert actual_reduction == reduction
    assert (pair_indices is not None) is (
        reduction.endswith("_pair")
        and int(agent.cfg.q_pair_size) < int(agent.cfg.num_q)
    )


@pytest.mark.parametrize("q_representation", ["scalar", "distributional"])
def test_value_equivalence_loss_gradients_belong_only_to_told_latents(
    q_representation,
):
    model = _model(
        horizon=2,
        q_representation=q_representation,
        inner_q_target_reduction="mean_all",
        inner_sac_critic_target="entropy_augmented",
        ent_coef="auto_0.5",
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=2,
    )
    agent = model.agent
    # The TD-MPC2 initialization zeroes Q output weights. Give the frozen
    # probe a deterministic state derivative so gradient ownership is tested
    # rather than accidentally hidden behind a constant initial critic.
    with torch.no_grad():
        for critic in agent.model._Qs:
            if q_representation == "scalar":
                critic[-1].weight.fill_(0.05)
            else:
                bin_slopes = torch.linspace(
                    -0.05,
                    0.05,
                    critic[-1].out_features,
                    device=critic[-1].weight.device,
                    dtype=critic[-1].weight.dtype,
                )
                critic[-1].weight.copy_(
                    bin_slopes[:, None].expand_as(critic[-1].weight)
                )
            critic[-1].bias.zero_()

    batch_size = int(agent.cfg.batch_size)
    obs_dim = int(agent.cfg.obs_shape["state"][0])
    initial_obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(
        2,
        batch_size,
        int(agent.cfg.action_dim),
    ).tanh()
    z = agent.model.encode(initial_obs)
    latent_states = [z]
    for action in actions:
        z = agent.model.next(z, action)
        latent_states.append(z)
    latent_states = torch.stack(latent_states)
    next_z_targets = torch.randn_like(latent_states[1:], requires_grad=True)

    for parameter in agent.model.parameters():
        parameter.grad = None
    if agent.log_ent_coef is not None:
        agent.log_ent_coef.grad = None

    raw_loss, _ = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=23,
    )
    raw_loss.backward()

    encoder_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in agent.model._encoder.parameters()
        if parameter.grad is not None
    )
    dynamics_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in agent.model._dynamics.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad > 0.0
    assert dynamics_grad > 0.0
    assert next_z_targets.grad is None
    assert all(parameter.grad is None for parameter in agent.model._reward.parameters())
    assert all(parameter.grad is None for parameter in agent.model._pi.parameters())
    assert all(parameter.grad is None for parameter in agent.model._Qs.parameters())
    assert all(parameter.grad is None for parameter in agent.model._target_Qs.parameters())
    assert agent.log_ent_coef is not None
    assert agent.log_ent_coef.grad is None


def test_value_equivalence_loss_has_zero_residual_and_gradient_for_equal_successors(
    monkeypatch,
):
    model = _model(
        inner_sac_critic_target="reward_only",
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=4,
    )
    agent = model.agent
    _patch_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z[..., :1].square(),
    )
    latent_states, next_z_targets = _loss_inputs(agent, requires_grad=True)
    equal_successors = torch.randn_like(next_z_targets)
    with torch.no_grad():
        latent_states[1:].copy_(equal_successors)
        next_z_targets.copy_(equal_successors)

    raw_loss, per_depth = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=29,
    )
    raw_loss.backward()

    torch.testing.assert_close(per_depth, torch.zeros_like(per_depth), rtol=0, atol=0)
    torch.testing.assert_close(raw_loss, torch.zeros_like(raw_loss), rtol=0, atol=0)
    torch.testing.assert_close(
        latent_states.grad,
        torch.zeros_like(latent_states.grad),
        rtol=0,
        atol=0,
    )


def test_zero_value_equivalence_loss_coefficient_is_full_update_invariant():
    first_model = _model(
        dropout=0.25,
        q_pair_size=1,
        value_equivalence_loss_coef=0.0,
        value_equivalence_loss_mc_samples=1,
    )
    second_model = _model(
        dropout=0.25,
        q_pair_size=1,
        value_equivalence_loss_coef=0.0,
        value_equivalence_loss_mc_samples=4,
    )
    first = first_model.agent
    second = second_model.agent
    _assert_tree_equal(first.model.state_dict(), second.model.state_dict())

    batch = _update_batch(first)
    torch.manual_seed(20260823)
    update_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(update_rng)
    first_metrics = first._update(*batch)
    first_rng_after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(update_rng)
    second_metrics = second._update(*batch)
    second_rng_after = torch.random.get_rng_state().clone()

    torch.testing.assert_close(first_rng_after, second_rng_after, rtol=0, atol=0)
    _assert_tree_equal(first.model.state_dict(), second.model.state_dict())
    _assert_tree_equal(first.optim.state_dict(), second.optim.state_dict())
    _assert_tree_equal(first.pi_optim.state_dict(), second.pi_optim.state_dict())
    _assert_tree_equal(
        first.ent_coef_optim.state_dict(),
        second.ent_coef_optim.state_dict(),
    )
    _assert_tree_equal(first_metrics, second_metrics)
    assert not any(key.startswith("value_equivalence_") for key in first_metrics)


def test_positive_coefficient_adds_weighted_loss_and_enabled_metrics(monkeypatch):
    baseline_model = _model(
        value_equivalence_loss_coef=0.0,
        value_equivalence_loss_mc_samples=1,
    )
    enabled_model = _model(
        value_equivalence_loss_coef=0.25,
        value_equivalence_loss_mc_samples=1,
    )
    baseline = baseline_model.agent
    enabled = enabled_model.agent
    batch = _update_batch(baseline)

    def fake_loss(latent_states, next_z_targets, loss_update):
        del next_z_targets, loss_update
        raw = latent_states.sum() * 0.0 + 2.0
        return raw, raw.reshape(1)

    monkeypatch.setattr(enabled, "_value_equivalence_loss", fake_loss)
    torch.manual_seed(20260824)
    update_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(update_rng)
    baseline_metrics = baseline._update(*batch)
    torch.random.set_rng_state(update_rng)
    enabled_metrics = enabled._update(*batch)

    assert not any(
        key.startswith("value_equivalence_") for key in baseline_metrics
    )
    assert _metric(enabled_metrics, "value_equivalence_loss") == pytest.approx(2.0)
    assert _metric(
        enabled_metrics,
        "value_equivalence_weighted_loss",
    ) == pytest.approx(0.5)
    assert _metric(
        enabled_metrics,
        "value_equivalence_loss_depth_1",
    ) == pytest.approx(2.0)
    assert _metric(enabled_metrics, "total_loss") == pytest.approx(
        _metric(baseline_metrics, "total_loss") + 0.5
    )


def test_value_equivalence_loss_and_sparse_monitor_are_independent():
    model = _model(
        value_equivalence_loss_coef=0.25,
        value_equivalence_loss_mc_samples=1,
        value_equivalence_diagnostics=True,
        value_equivalence_every_updates=1,
        value_equivalence_mc_samples=2,
    )

    metrics = model.agent._update(*_update_batch(model.agent))

    assert "value_equivalence_loss" in metrics
    assert "value_equivalence_weighted_loss" in metrics
    assert "ve_prior_target_mae" in metrics
    assert "ve_prior_target_rmse" in metrics


def test_value_equivalence_loss_preserves_modes_and_rng_and_repeats_on_resume():
    source_model = _model(
        q_pair_size=1,
        dropout=0.25,
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=4,
    )
    source = source_model.agent
    latent_states, next_z_targets = _loss_inputs(source, requires_grad=True)
    with torch.no_grad():
        latent_states.normal_()
        next_z_targets.normal_()

    source.model.train(True)
    next(iter(source.model._pi.children())).eval()
    next(iter(source.model._Qs.children())).eval()
    module_modes = [module.training for module in source.model.modules()]
    global_rng = torch.random.get_rng_state().clone()
    inner_rng = _clone_tree(source.inner_engine.rng.training_state_dict())

    first_loss, first_depths = source._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=31,
    )
    second_loss, second_depths = source._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=31,
    )

    assert [module.training for module in source.model.modules()] == module_modes
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    _assert_tree_equal(source.inner_engine.rng.training_state_dict(), inner_rng)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    torch.testing.assert_close(first_depths, second_depths, rtol=0, atol=0)

    saved = _clone_tree(source.training_state_dict())
    restored_model = _model(
        q_pair_size=1,
        dropout=0.25,
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=4,
    )
    restored = restored_model.agent
    restored.load_training_state_dict(saved)
    restored_loss, restored_depths = restored._value_equivalence_loss(
        latent_states.detach().clone().requires_grad_(),
        next_z_targets,
        loss_update=31,
    )

    torch.testing.assert_close(first_loss, restored_loss, rtol=0, atol=0)
    torch.testing.assert_close(first_depths, restored_depths, rtol=0, atol=0)


def test_enabled_value_equivalence_loss_resumes_exact_outer_updates():
    source_model = _model(
        q_pair_size=1,
        dropout=0.25,
        value_equivalence_loss_coef=0.125,
        value_equivalence_loss_mc_samples=3,
    )
    source = source_model.agent
    restored_model = _model(
        q_pair_size=1,
        dropout=0.25,
        value_equivalence_loss_coef=0.125,
        value_equivalence_loss_mc_samples=3,
    )
    restored = restored_model.agent
    restored.load_training_state_dict(_clone_tree(source.training_state_dict()))
    batch = _update_batch(source)

    torch.manual_seed(20260825)
    update_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(update_rng)
    source_metrics = source._update(*batch)
    source_rng_after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(update_rng)
    restored_metrics = restored._update(*batch)
    restored_rng_after = torch.random.get_rng_state().clone()

    torch.testing.assert_close(source_rng_after, restored_rng_after, rtol=0, atol=0)
    _assert_tree_equal(source.model.state_dict(), restored.model.state_dict())
    _assert_tree_equal(source.optim.state_dict(), restored.optim.state_dict())
    _assert_tree_equal(source.pi_optim.state_dict(), restored.pi_optim.state_dict())
    _assert_tree_equal(
        source.ent_coef_optim.state_dict(),
        restored.ent_coef_optim.state_dict(),
    )
    _assert_tree_equal(source_metrics, restored_metrics)
    assert "value_equivalence_loss" in source_metrics


def test_value_equivalence_loss_metrics_accumulate_under_train_namespace():
    model = _model(value_equivalence_loss_coef=0.25)
    model._accumulate_train_metrics(
        {
            "value_equivalence_loss": torch.tensor(1.0),
            "value_equivalence_weighted_loss": torch.tensor(0.25),
            "value_equivalence_loss_depth_1": torch.tensor(1.0),
        }
    )
    model._accumulate_train_metrics(
        {
            "value_equivalence_loss": torch.tensor(3.0),
            "value_equivalence_weighted_loss": torch.tensor(0.75),
            "value_equivalence_loss_depth_1": torch.tensor(3.0),
        }
    )

    payload = model._wandb_update_window.pop_floats(include_stats=True)

    assert payload["train/value_equivalence_loss"] == pytest.approx(2.0)
    assert payload["train/value_equivalence_loss_count"] == pytest.approx(2.0)
    assert payload["train/value_equivalence_loss_min"] == pytest.approx(1.0)
    assert payload["train/value_equivalence_loss_max"] == pytest.approx(3.0)
    assert payload["train/value_equivalence_weighted_loss"] == pytest.approx(0.5)
    assert payload["train/value_equivalence_loss_depth_1"] == pytest.approx(2.0)


def test_value_equivalence_loss_runs_outside_compiled_told_region(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda function, **_kwargs: function)
    model = _model(
        compile=True,
        compile_strict=True,
        value_equivalence_loss_coef=0.25,
        value_equivalence_loss_mc_samples=2,
    )

    metrics = model.agent._update(*_update_batch(model.agent))

    assert "value_equivalence_loss" in metrics
    assert _metric(metrics, "compile_outer_update_fallback") == pytest.approx(0.0)
    assert not model.agent._outer_update_region.failed


def test_value_equivalence_loss_contract_is_continuing_and_non_terminating():
    model = _model(
        episodic=False,
        value_equivalence_loss_coef=1.0,
        value_equivalence_loss_mc_samples=1,
    )
    agent = model.agent
    signature = inspect.signature(agent._value_equivalence_loss)

    assert tuple(signature.parameters) == (
        "latent_states",
        "next_z_targets",
        "loss_update",
    )
    assert agent.model._termination is None
    latent_states, next_z_targets = _loss_inputs(agent)
    raw_loss, per_depth = agent._value_equivalence_loss(
        latent_states,
        next_z_targets,
        loss_update=37,
    )
    assert torch.isfinite(raw_loss)
    assert torch.isfinite(per_depth).all()


def test_positive_value_equivalence_loss_rejects_episodic_models():
    with pytest.raises(ValueError, match="value_equivalence_loss.*episodic"):
        _model(
            episodic=True,
            value_equivalence_loss_coef=0.1,
            value_equivalence_loss_mc_samples=1,
        )
