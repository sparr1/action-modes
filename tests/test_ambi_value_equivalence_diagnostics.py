import math
from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core import ambi_agent as ambi_agent_module


_AGGREGATE_KEYS = {
    "ve_prior_target_mae",
    "ve_prior_target_rmse",
    "ve_prior_target_bias",
    "ve_prior_target_nrmse",
    "ve_prior_target_abs_p95",
    "ve_prior_reference_target_rms",
    "ve_prior_reward_rmse",
    "ve_prior_bootstrap_rmse",
    "ve_prior_cancellation_fraction",
}
_DEPTH_STEMS = {
    "ve_prior_target_mae_depth_{}",
    "ve_prior_target_rmse_depth_{}",
    "ve_prior_target_bias_depth_{}",
    "ve_prior_reward_rmse_depth_{}",
    "ve_prior_bootstrap_rmse_depth_{}",
}


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
        "value_equivalence_diagnostics": True,
        "value_equivalence_every_updates": 1,
        "value_equivalence_mc_samples": 1,
    }
    params.update(overrides)
    return params


def _model(horizon=1, **overrides):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    params = _params(horizon, **overrides)
    device = params["device"]
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        params,
        {"seed": 19, "device": device, "env": "test", "total_steps": 10},
        {},
    )


def _inputs(agent, horizon=1, batch_size=2):
    latent_dim = int(agent.cfg.latent_dim)
    latent_states = torch.zeros(
        horizon + 1, batch_size, latent_dim, device=agent.device
    )
    reward_predictions = torch.zeros(
        horizon, batch_size, 1, device=agent.device
    )
    next_z_targets = torch.zeros(
        horizon, batch_size, latent_dim, device=agent.device
    )
    reward = torch.zeros(horizon, batch_size, 1, device=agent.device)
    terminated = torch.zeros_like(reward)
    return (
        latent_states,
        reward_predictions,
        None,
        next_z_targets,
        reward,
        terminated,
    )


def _update_batch(agent):
    horizon = int(agent.cfg.train_unroll_horizon)
    batch_size = int(agent.cfg.batch_size)
    obs_dim = int(agent.cfg.obs_shape["state"][0])
    obs = torch.randn(horizon + 1, batch_size, obs_dim)
    action = torch.randn(horizon, batch_size, agent.cfg.action_dim).tanh()
    reward = torch.randn(horizon, batch_size, 1)
    terminated = torch.zeros_like(reward)
    return obs, action, reward, terminated


def _metric(metrics, key):
    return float(torch.as_tensor(metrics[key]))


def _identity_reward_decoder(monkeypatch):
    monkeypatch.setattr(
        ambi_agent_module.td_math,
        "two_hot_inv",
        lambda prediction, cfg: prediction,
    )


def _patch_value_probe(monkeypatch, agent, *, value, log_prob=None):
    def fake_pi(z, *args, **kwargs):
        assert not torch.is_grad_enabled()
        action = z.new_zeros(*z.shape[:-1], int(agent.cfg.action_dim))
        resolved_log_prob = (
            z.new_zeros(*z.shape[:-1], 1)
            if log_prob is None
            else log_prob(z)
        )
        return action, {"log_prob": resolved_log_prob}

    def fake_q(critic, z, action, reduction, pair_indices):
        assert not torch.is_grad_enabled()
        return value(z, action)

    monkeypatch.setattr(agent.model, "pi", fake_pi)
    monkeypatch.setattr(agent, "_value_equivalence_q", fake_q)


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


def test_value_equivalence_matches_analytic_target_and_decomposition(monkeypatch):
    model = _model(
        inner_sac_critic_target="reward_only",
        value_equivalence_mc_samples=1,
    )
    agent = model.agent
    agent.discount = 0.5
    _identity_reward_decoder(monkeypatch)
    _patch_value_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z[..., :1],
    )

    inputs = list(_inputs(agent))
    # Real targets are [1 + .5*1, 2 + .5*6] = [1.5, 5].
    inputs[3][0, :, 0] = torch.tensor([1.0, 6.0])
    inputs[4][0, :, 0] = torch.tensor([1.0, 2.0])
    # Model targets are [3 + .5*5, -1 + .5*4] = [5.5, 1].
    inputs[0][1, :, 0] = torch.tensor([5.0, 4.0])
    inputs[1][0, :, 0] = torch.tensor([3.0, -1.0])

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=7,
    )

    assert _AGGREGATE_KEYS <= set(metrics)
    assert _metric(metrics, "ve_prior_target_mae") == pytest.approx(4.0)
    assert _metric(metrics, "ve_prior_target_rmse") == pytest.approx(4.0)
    assert _metric(metrics, "ve_prior_target_bias") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_target_abs_p95") == pytest.approx(4.0)
    reference_rms = math.sqrt((1.5**2 + 5.0**2) / 2.0)
    assert _metric(metrics, "ve_prior_reference_target_rms") == pytest.approx(
        reference_rms
    )
    assert _metric(metrics, "ve_prior_target_nrmse") == pytest.approx(
        4.0 / reference_rms
    )
    assert _metric(metrics, "ve_prior_reward_rmse") == pytest.approx(math.sqrt(6.5))
    assert _metric(metrics, "ve_prior_bootstrap_rmse") == pytest.approx(
        math.sqrt(2.5)
    )
    assert _metric(metrics, "ve_prior_cancellation_fraction") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_target_mae_depth_1") == pytest.approx(4.0)
    assert _metric(metrics, "ve_prior_target_rmse_depth_1") == pytest.approx(4.0)
    assert _metric(metrics, "ve_prior_target_bias_depth_1") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_reward_rmse_depth_1") == pytest.approx(
        math.sqrt(6.5)
    )
    assert _metric(metrics, "ve_prior_bootstrap_rmse_depth_1") == pytest.approx(
        math.sqrt(2.5)
    )


def test_value_equivalence_reports_analytic_reward_bootstrap_cancellation(
    monkeypatch,
):
    model = _model(
        inner_sac_critic_target="reward_only",
        value_equivalence_mc_samples=1,
    )
    agent = model.agent
    agent.discount = 0.5
    _identity_reward_decoder(monkeypatch)
    _patch_value_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z[..., :1],
    )
    inputs = list(_inputs(agent))
    # e_r=+2 and e_b=-1, so e_y=+1. The approved bounded cancellation
    # fraction is -2 E[e_r e_b] / (E[e_r^2] + E[e_b^2]) = 4/5.
    inputs[1].fill_(2.0)
    inputs[3][..., 0] = 2.0

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=9,
    )

    assert _metric(metrics, "ve_prior_target_rmse") == pytest.approx(1.0)
    assert _metric(metrics, "ve_prior_reward_rmse") == pytest.approx(2.0)
    assert _metric(metrics, "ve_prior_bootstrap_rmse") == pytest.approx(1.0)
    assert _metric(metrics, "ve_prior_cancellation_fraction") == pytest.approx(0.8)


@pytest.mark.parametrize("q_representation", ["scalar", "distributional"])
def test_value_equivalence_reports_zero_for_identical_model_and_real_targets(
    monkeypatch,
    q_representation,
):
    model = _model(
        q_representation=q_representation,
        inner_sac_critic_target="entropy_augmented",
        q_pair_size=1,
        value_equivalence_mc_samples=4,
    )
    agent = model.agent
    _identity_reward_decoder(monkeypatch)
    inputs = list(_inputs(agent))
    next_states = torch.randn_like(inputs[3])
    rewards = torch.randn_like(inputs[4])
    inputs[0][1:] = next_states
    inputs[3].copy_(next_states)
    inputs[1].copy_(rewards)
    inputs[4].copy_(rewards)

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=11,
    )

    zero_keys = {
        "ve_prior_target_mae",
        "ve_prior_target_rmse",
        "ve_prior_target_bias",
        "ve_prior_target_nrmse",
        "ve_prior_target_abs_p95",
        "ve_prior_reward_rmse",
        "ve_prior_bootstrap_rmse",
        "ve_prior_cancellation_fraction",
        "ve_prior_target_mae_depth_1",
        "ve_prior_target_rmse_depth_1",
        "ve_prior_target_bias_depth_1",
        "ve_prior_reward_rmse_depth_1",
        "ve_prior_bootstrap_rmse_depth_1",
    }
    for key in zero_keys:
        assert _metric(metrics, key) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    ("critic_target", "expected_error"),
    [("reward_only", 0.0), ("entropy_augmented", -1.0)],
)
def test_value_equivalence_uses_the_configured_inner_entropy_target(
    monkeypatch,
    critic_target,
    expected_error,
):
    model = _model(
        ent_coef=2.0,
        inner_temperature_initialization="inherit_outer",
        inner_sac_critic_target=critic_target,
    )
    agent = model.agent
    agent.discount = 0.5
    _identity_reward_decoder(monkeypatch)
    _patch_value_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z.new_zeros(*z.shape[:-1], 1),
        log_prob=lambda z: z[..., :1],
    )
    inputs = list(_inputs(agent))
    inputs[0][1, :, 0] = 2.0
    inputs[3][0, :, 0] = 1.0

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=3,
    )

    assert _metric(metrics, "ve_prior_target_bias") == pytest.approx(expected_error)
    assert _metric(metrics, "ve_prior_target_rmse") == pytest.approx(
        abs(expected_error)
    )
    assert _metric(metrics, "ve_prior_bootstrap_rmse") == pytest.approx(
        abs(expected_error)
    )
    assert _metric(metrics, "ve_prior_reward_rmse") == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("temperature_mode", "initialization", "expected"),
    [
        ("inherit_outer", "fixed", 2.0),
        ("fixed", "inherit_outer", 0.25),
        ("auto", "inherit_outer", 2.0),
        ("auto", "fixed", 0.25),
    ],
)
def test_value_equivalence_uses_fresh_inner_temperature_semantics(
    temperature_mode,
    initialization,
    expected,
):
    model = _model(
        ent_coef=2.0,
        inner_temperature_mode=temperature_mode,
        inner_temperature_initialization=initialization,
        inner_temperature=0.25,
    )

    assert float(model.agent._initial_inner_diagnostic_alpha()) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("bootstrap_source", "expected_critic"),
    [
        ("inner_target", "_Qs"),
        ("outer_online", "_Qs"),
        ("outer_target", "_target_Qs"),
    ],
)
def test_value_equivalence_uses_configured_bootstrap_source(
    bootstrap_source,
    expected_critic,
):
    model = _model(inner_bootstrap_source=bootstrap_source)

    assert model.agent._value_equivalence_reference_critic() is getattr(
        model.agent.model, expected_critic
    )


def test_value_equivalence_masks_terminal_bootstraps_and_reports_disagreement(
    monkeypatch,
):
    model = _model(
        episodic=True,
        inner_sac_critic_target="reward_only",
        inner_termination_threshold=0.5,
    )
    agent = model.agent
    agent.discount = 0.5
    _identity_reward_decoder(monkeypatch)
    _patch_value_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z.new_full((*z.shape[:-1], 1), 2.0),
    )
    inputs = list(_inputs(agent))
    inputs[2] = torch.tensor([[[100.0], [-100.0]]])
    inputs[5].fill_(1.0)

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=5,
    )

    # The first model transition and both real transitions terminate. The
    # second model transition remains alive, contributing a bootstrap of one.
    assert _metric(metrics, "ve_prior_target_mae") == pytest.approx(0.5)
    assert _metric(metrics, "ve_prior_target_rmse") == pytest.approx(
        math.sqrt(0.5)
    )
    assert _metric(metrics, "ve_prior_target_bias") == pytest.approx(0.5)
    assert _metric(metrics, "ve_prior_bootstrap_rmse") == pytest.approx(
        math.sqrt(0.5)
    )
    assert _metric(metrics, "ve_prior_reward_rmse") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_termination_disagreement") == pytest.approx(
        0.5
    )
    assert _metric(
        metrics, "ve_prior_termination_disagreement_depth_1"
    ) == pytest.approx(0.5)


def test_episodic_value_equivalence_omits_forced_continuation_depths(
    monkeypatch,
):
    model = _model(
        horizon=3,
        episodic=True,
        inner_sac_critic_target="reward_only",
        inner_termination_threshold=0.5,
    )
    agent = model.agent
    _identity_reward_decoder(monkeypatch)
    _patch_value_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z.new_zeros(*z.shape[:-1], 1),
    )
    inputs = list(_inputs(agent, horizon=3))
    # Branch zero terminates at depth one; branch one terminates at depth two.
    inputs[2] = torch.tensor(
        [
            [[100.0], [-100.0]],
            [[-100.0], [100.0]],
            [[-100.0], [-100.0]],
        ]
    )
    # Large errors after termination must not enter aggregate or depth metrics.
    inputs[1][1, 0] = 50.0
    inputs[1][2] = 100.0

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=9,
    )

    assert _metric(metrics, "ve_prior_target_rmse") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_reward_rmse") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_termination_disagreement") == pytest.approx(
        2.0 / 3.0
    )
    assert _metric(
        metrics, "ve_prior_termination_disagreement_depth_1"
    ) == pytest.approx(0.5)
    assert _metric(
        metrics, "ve_prior_termination_disagreement_depth_2"
    ) == pytest.approx(1.0)
    assert "ve_prior_target_rmse_depth_3" not in metrics
    assert "ve_prior_termination_disagreement_depth_3" not in metrics


def test_value_equivalence_averages_four_shared_noise_and_pair_samples(
    monkeypatch,
):
    model = _model(
        q_pair_size=1,
        inner_sac_critic_target="reward_only",
        value_equivalence_mc_samples=4,
    )
    agent = model.agent
    agent.discount = 1.0
    _identity_reward_decoder(monkeypatch)
    inputs = list(_inputs(agent))
    inputs[0][1, :, 0] = 1.0
    inputs[3][0, :, 0] = 0.0

    coefficients = (-3.0, -1.0, 1.0, 3.0)
    policy_noises = []
    q_pairs = []
    sampled_pairs = []

    def fake_pi(z, *args, **kwargs):
        assert not torch.is_grad_enabled()
        assert z.shape[0] == 2
        noise = kwargs.get("noise")
        assert noise is not None
        torch.testing.assert_close(noise[0], noise[1], rtol=0, atol=0)
        policy_noises.append(noise.detach().clone())
        coefficient = coefficients[len(policy_noises) - 1]
        action = z.new_full((*z.shape[:-1], int(agent.cfg.action_dim)), coefficient)
        return action, {"log_prob": z.new_zeros(*z.shape[:-1], 1)}

    def fake_q(critic, z, action, reduction, pair_indices):
        assert pair_indices is not None
        q_pairs.append(pair_indices.detach().clone())
        return z[..., :1] * action[..., :1]

    def fake_sample_pair_indices(device, *, generator=None):
        assert generator is not None
        pair = torch.tensor(
            [len(sampled_pairs) % 2], dtype=torch.long, device=device
        )
        sampled_pairs.append(pair.detach().clone())
        return pair

    monkeypatch.setattr(agent.model, "pi", fake_pi)
    monkeypatch.setattr(agent, "_value_equivalence_q", fake_q)
    monkeypatch.setattr(
        agent.model.q_backend,
        "sample_pair_indices",
        fake_sample_pair_indices,
    )

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=13,
    )

    assert len(policy_noises) == len(q_pairs) == len(sampled_pairs) == 4
    for actual, expected in zip(q_pairs, sampled_pairs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The paired residuals are [-3, -1, 1, 3]. Bellman equivalence concerns
    # their conditional mean, so statistics are computed after MC averaging.
    assert _metric(metrics, "ve_prior_target_mae") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_target_rmse") == pytest.approx(0.0)
    assert _metric(metrics, "ve_prior_bootstrap_rmse") == pytest.approx(0.0)


def test_value_equivalence_cadence_and_skipped_update_keys(monkeypatch):
    model = _model(
        value_equivalence_diagnostics=True,
        value_equivalence_every_updates=2,
        value_equivalence_mc_samples=1,
    )
    agent = model.agent

    for num_updates, expected in ((0, False), (1, True), (2, False), (3, True)):
        agent.num_updates = num_updates
        assert agent._should_run_value_equivalence_diagnostics() is expected
    agent.cfg.value_equivalence_diagnostics = False
    assert agent._should_run_value_equivalence_diagnostics() is False
    agent.cfg.value_equivalence_diagnostics = True
    agent.num_updates = 0

    diagnostic_updates = []

    def fake_diagnostics(*args, **kwargs):
        diagnostic_updates.append(kwargs.get("diagnostic_update", args[-1]))
        return {"ve_prior_target_mae": torch.tensor(7.0)}

    monkeypatch.setattr(agent, "_value_equivalence_diagnostics", fake_diagnostics)
    obs = torch.randn(2, agent.cfg.batch_size, agent.cfg.obs_shape["state"][0])
    action = torch.randn(1, agent.cfg.batch_size, agent.cfg.action_dim).tanh()
    reward = torch.randn(1, agent.cfg.batch_size, 1)
    terminated = torch.zeros_like(reward)

    first = agent._update(obs, action, reward, terminated)
    second = agent._update(obs, action, reward, terminated)

    assert not any(key.startswith("ve_prior_") for key in first)
    assert _metric(second, "ve_prior_target_mae") == pytest.approx(7.0)
    assert diagnostic_updates == [2]


def test_sparse_value_equivalence_metrics_do_not_accumulate_missing_updates():
    model = _model(value_equivalence_diagnostics=True)

    model._accumulate_train_metrics({"ve_prior_target_rmse": torch.tensor(1.0)})
    model._accumulate_train_metrics({"critic_loss": torch.tensor(9.0)})
    model._accumulate_train_metrics({"ve_prior_target_rmse": torch.tensor(3.0)})
    payload = model._wandb_update_window.pop_floats(include_stats=True)

    assert payload["train/ve_prior_target_rmse"] == pytest.approx(2.0)
    assert payload["train/ve_prior_target_rmse_count"] == pytest.approx(2.0)
    assert payload["train/ve_prior_target_rmse_min"] == pytest.approx(1.0)
    assert payload["train/ve_prior_target_rmse_max"] == pytest.approx(3.0)


def test_value_equivalence_cadence_resumes_on_the_same_n_and_2n_updates():
    source = _model(
        value_equivalence_diagnostics=True,
        value_equivalence_every_updates=2,
        value_equivalence_mc_samples=1,
    )
    first_batch = _update_batch(source.agent)
    first = source.agent._update(*first_batch)
    assert not any(key.startswith("ve_prior_") for key in first)
    assert source.agent.num_updates == source.agent.outer_version == 1
    assert source.agent._should_run_value_equivalence_diagnostics() is True

    saved = _clone_tree(source.agent.training_state_dict())
    restored = _model(
        value_equivalence_diagnostics=True,
        value_equivalence_every_updates=2,
        value_equivalence_mc_samples=1,
    )
    restored.agent.load_training_state_dict(saved)

    assert restored.agent.num_updates == restored.agent.outer_version == 1
    assert restored.agent._should_run_value_equivalence_diagnostics() is True
    second = restored.agent._update(*_update_batch(restored.agent))
    assert _AGGREGATE_KEYS <= set(second)
    assert restored.agent._should_run_value_equivalence_diagnostics() is False
    third = restored.agent._update(*_update_batch(restored.agent))
    assert not any(key.startswith("ve_prior_") for key in third)
    assert restored.agent.num_updates == restored.agent.outer_version == 3
    assert restored.agent._should_run_value_equivalence_diagnostics() is True


def test_due_value_equivalence_diagnostic_is_observational_for_full_update():
    without_diagnostics = _model(
        dropout=0.25,
        q_pair_size=1,
        value_equivalence_diagnostics=False,
        value_equivalence_every_updates=1,
        value_equivalence_mc_samples=4,
    )
    with_diagnostics = _model(
        dropout=0.25,
        q_pair_size=1,
        value_equivalence_diagnostics=True,
        value_equivalence_every_updates=1,
        value_equivalence_mc_samples=4,
    )
    first = without_diagnostics.agent
    second = with_diagnostics.agent
    _assert_tree_equal(first.model.state_dict(), second.model.state_dict())
    _assert_tree_equal(first.optim.state_dict(), second.optim.state_dict())
    _assert_tree_equal(first.pi_optim.state_dict(), second.pi_optim.state_dict())
    _assert_tree_equal(
        first.ent_coef_optim.state_dict(), second.ent_coef_optim.state_dict()
    )

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
        first.ent_coef_optim.state_dict(), second.ent_coef_optim.state_dict()
    )
    torch.testing.assert_close(first.alpha, second.alpha, rtol=0, atol=0)
    assert (first.num_updates, first.outer_version) == (
        second.num_updates,
        second.outer_version,
    )
    _assert_tree_equal(
        first.inner_engine.training_state_dict(),
        second.inner_engine.training_state_dict(),
    )
    assert first.model.training == second.model.training

    assert not any(key.startswith("ve_prior_") for key in first_metrics)
    assert _AGGREGATE_KEYS <= set(second_metrics)
    second_non_ve = {
        key: value
        for key, value in second_metrics.items()
        if not key.startswith("ve_prior_")
    }
    _assert_tree_equal(first_metrics, second_non_ve)


@pytest.mark.parametrize("horizon", [1, 6])
def test_value_equivalence_emits_every_and_only_available_depth(monkeypatch, horizon):
    model = _model(horizon=horizon, value_equivalence_mc_samples=1)
    agent = model.agent
    _identity_reward_decoder(monkeypatch)
    _patch_value_probe(
        monkeypatch,
        agent,
        value=lambda z, action: z[..., :1],
    )
    inputs = list(_inputs(agent, horizon=horizon))
    inputs[0][1:].copy_(inputs[3])
    inputs[1].copy_(inputs[4])

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=17,
    )

    assert _AGGREGATE_KEYS <= set(metrics)
    for depth in range(1, horizon + 1):
        assert {stem.format(depth) for stem in _DEPTH_STEMS} <= set(metrics)
    assert not any(
        key.endswith(f"_depth_{horizon + 1}") for key in metrics
    )


def test_value_equivalence_is_no_grad_and_preserves_module_modes_and_rng(
    monkeypatch,
):
    model = _model(
        dropout=0.25,
        q_pair_size=1,
        value_equivalence_mc_samples=4,
    )
    agent = model.agent
    _identity_reward_decoder(monkeypatch)
    inputs = list(_inputs(agent))
    inputs[0].normal_().requires_grad_()
    inputs[1].normal_().requires_grad_()
    inputs[3].normal_().requires_grad_()
    inputs[4].normal_().requires_grad_()

    agent.model.train(True)
    next(iter(agent.model._pi.children())).eval()
    next(iter(agent.model._Qs.children())).eval()
    module_modes = [module.training for module in agent.model.modules()]
    global_rng = torch.random.get_rng_state().clone()
    inner_rng = _clone_tree(agent.inner_engine.rng.training_state_dict())

    first = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=23,
    )
    second = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=23,
    )

    assert [module.training for module in agent.model.modules()] == module_modes
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    _assert_tree_equal(agent.inner_engine.rng.training_state_dict(), inner_rng)
    assert first.keys() == second.keys()
    for key in first:
        first_value = torch.as_tensor(first[key])
        second_value = torch.as_tensor(second[key])
        assert first_value.numel() == 1
        assert not first_value.requires_grad
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_value_equivalence_preserves_all_cuda_rng_states():
    model = _model(
        device="cuda",
        dropout=0.25,
        q_pair_size=1,
        value_equivalence_mc_samples=4,
    )
    agent = model.agent
    inputs = _inputs(agent)
    torch.manual_seed(20260823)
    torch.cuda.manual_seed_all(20260823)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    metrics = agent._value_equivalence_diagnostics(
        *inputs,
        diagnostic_update=29,
    )

    torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng, rtol=0, atol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
