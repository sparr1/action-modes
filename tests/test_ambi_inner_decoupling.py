import math
from collections import OrderedDict
from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common.inner_utils import lora_uses_shared_bases
from RL.tdmpc2_core.common.lora import LoRALinear, LoRANormedLinear


def _params(**overrides):
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
        "batch_size": 4,
        "train_unroll_horizon": 2,
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
        "q_num_bins": 11,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_model_step_budget": 16,
        "inner_rounds": 2,
        "inner_rollout_horizon": 2,
        "inner_critic_updates_per_action": 2,
        "inner_actor_updates_per_action": 2,
        "inner_temperature_updates_per_action": 0,
        "inner_batch_size": 8,
        "inner_replay_capacity": 32,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_target_tau": 1.0,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "inner_mppi_num_elites": 2,
        "inner_mppi_num_pi_trajs": 0,
    }
    params.update(overrides)
    return params


def _model(**overrides):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        _params(**overrides),
        {"seed": 13, "device": "cpu", "env": "test", "total_steps": 10},
        {},
    )


def _assert_finite(metrics):
    for value in metrics.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value))


@pytest.mark.parametrize("inner_operator", ["sac", "td3"])
def test_online_evaluation_keeps_inner_optimizer_gradients_enabled(
    inner_operator, monkeypatch
):
    model = _model(inner_operator=inner_operator)
    model._eval_episodes = 1
    model._record_evaluation = lambda step, reward: None
    obs, _ = model._reset_env(seed=13)
    update_counts = []
    original_updates = model.agent.inner_engine._run_update_counts

    def tracked_updates(**kwargs):
        result = original_updates(**kwargs)
        update_counts.append(
            (kwargs["critic_count"], kwargs["actor_count"], len(result))
        )
        return result

    monkeypatch.setattr(
        model.agent.inner_engine, "_run_update_counts", tracked_updates
    )
    outer_before = {
        key: value.detach().clone()
        for key, value in model.agent.model.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()

    model._evaluate_policy(0, initial_obs=obs)

    assert update_counts
    assert all(critic == actor == slots == 1 for critic, actor, slots in update_counts)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    for key, value in model.agent.model.state_dict().items():
        torch.testing.assert_close(value, outer_before[key], rtol=0, atol=0)


def test_online_evaluation_restores_run_scoped_inner_state_and_rng():
    model = _model(
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_temperature_scope="run",
        inner_replay_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_temperature_optimizer_scope="run",
        inner_rebase_persistent=True,
    )
    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
    model.agent.prepare_training_resume_boundary()
    before = _clone_tree(model.agent.inner_engine.training_state_dict())
    metrics_before = _clone_tree(model.agent.last_inner_metrics)
    lengths_before = list(model.agent.last_inner_rollout_lengths)
    model._eval_episodes = 2
    model._record_evaluation = lambda step, reward: None
    obs, _ = model.env.reset(seed=13)

    model._evaluate_policy(5, initial_obs=obs)

    _assert_tree_equal(model.agent.inner_engine.training_state_dict(), before)
    _assert_tree_equal(model.agent.last_inner_metrics, metrics_before)
    assert model.agent.last_inner_rollout_lengths == lengths_before
    assert model.agent._resume_boundary_prepared is True


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


def _optimizer_group_options(optimizer):
    return _clone_tree(
        [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ]
    )


def _set_foreign_optimizer_group_options(optimizer):
    """Make checkpoint-only Adam options differ from receiving defaults."""

    for group in optimizer.param_groups:
        group.update(
            weight_decay=0.25,
            maximize=True,
            foreach=False,
            capturable=True,
            differentiable=True,
            fused=False,
        )


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


@pytest.mark.parametrize(
    ("representation", "num_q"),
    [("scalar", 2), ("distributional", 5)],
)
def test_sac_inner_and_outer_updates_support_both_q_representations(
    representation, num_q
):
    model = _model(q_representation=representation, num_q=num_q)
    agent = model.agent
    outer_before = {
        key: value.detach().clone() for key, value in agent.model.state_dict().items()
    }
    optim_before = _clone_tree(agent.optim.state_dict())
    pi_optim_before = _clone_tree(agent.pi_optim.state_dict())
    entropy_optim_before = _clone_tree(agent.ent_coef_optim.state_dict())
    alpha_before = agent.alpha.detach().clone()
    updates_before = (agent.num_updates, agent.outer_version)
    rng_before = torch.random.get_rng_state().clone()
    action = agent.act(torch.zeros(model.cfg.obs_shape["state"]), eval_mode=False)

    assert action.shape == (model.cfg.action_dim,)
    assert agent.last_inner_metrics["inner_model_steps_budget"] == 16
    assert agent.last_inner_metrics["inner_model_steps"] == 16
    assert agent.last_inner_metrics["inner_critic_optimizer_steps"] == 2
    assert agent.last_inner_metrics["inner_actor_optimizer_steps"] == 2
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    for key, value in agent.model.state_dict().items():
        torch.testing.assert_close(value, outer_before[key], rtol=0, atol=0)
    _assert_tree_equal(agent.optim.state_dict(), optim_before)
    _assert_tree_equal(agent.pi_optim.state_dict(), pi_optim_before)
    _assert_tree_equal(agent.ent_coef_optim.state_dict(), entropy_optim_before)
    torch.testing.assert_close(agent.alpha, alpha_before, rtol=0, atol=0)
    assert (agent.num_updates, agent.outer_version) == updates_before
    _assert_finite(agent.last_inner_metrics)

    obs = torch.randn(model.cfg.train_unroll_horizon + 1, model.cfg.batch_size, 3)
    actions = torch.randn(model.cfg.train_unroll_horizon, model.cfg.batch_size, 1).tanh()
    rewards = torch.randn(model.cfg.train_unroll_horizon, model.cfg.batch_size, 1)
    update = agent._update(obs, actions, rewards, torch.zeros_like(rewards))
    _assert_finite(update)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {
                "inner_actor_adaptation": "frozen",
                "inner_actor_updates_per_action": 0,
            },
            (2, 0, 0),
        ),
        (
            {
                "inner_critic_adaptation": "frozen",
                "inner_critic_updates_per_action": 0,
            },
            (0, 2, 0),
        ),
        (
            {
                "inner_actor_adaptation": "frozen",
                "inner_critic_adaptation": "frozen",
                "inner_actor_updates_per_action": 0,
                "inner_critic_updates_per_action": 0,
                "inner_temperature_mode": "auto",
                "inner_temperature_updates_per_action": 2,
            },
            (0, 0, 2),
        ),
        (
            {
                "inner_actor_adaptation": "frozen",
                "inner_critic_adaptation": "frozen",
                "inner_actor_updates_per_action": 0,
                "inner_critic_updates_per_action": 0,
            },
            (0, 0, 0),
        ),
    ],
)
def test_component_toggles_have_independent_step_counts(overrides, expected):
    model = _model(**overrides)
    model.agent.act(torch.zeros(3), eval_mode=False)
    metrics = model.agent.last_inner_metrics
    critic, actor, temperature = expected
    assert metrics["inner_critic_optimizer_steps"] == critic
    assert metrics["inner_actor_optimizer_steps"] == actor
    assert metrics["inner_temperature_optimizer_steps"] == temperature


def test_td3_and_mppi_and_none_share_the_action_contract():
    variants = (
        _model(
            inner_operator="none",
            inner_model_step_budget=0,
            inner_rounds=0,
            inner_critic_updates_per_action=0,
            inner_actor_updates_per_action=0,
            inner_temperature_updates_per_action=0,
        ),
        _model(inner_operator="td3", inner_temperature_mode="inherit_outer"),
        _model(
            inner_operator="mppi",
            inner_critic_updates_per_action=0,
            inner_actor_updates_per_action=0,
            inner_temperature_updates_per_action=0,
        ),
    )
    for model in variants:
        action = model.agent.act(torch.zeros(3), t0=True, eval_mode=True)
        assert action.shape == (1,)
        assert torch.isfinite(action).all()
        assert (action.abs() <= 1).all()
        _assert_finite(model.agent.last_inner_metrics)
    assert variants[0].agent.last_inner_metrics["inner_model_steps"] == 0
    assert variants[1].agent.last_inner_metrics["inner_critic_optimizer_steps"] == 2
    assert variants[2].agent.last_inner_metrics["inner_model_steps"] == 16


def test_episode_scoped_replay_persists_but_is_invalidated_after_outer_update():
    model = _model(
        inner_actor_scope="episode",
        inner_critic_scope="episode",
        inner_actor_optimizer_scope="episode",
        inner_critic_optimizer_scope="episode",
        inner_replay_scope="episode",
        inner_model_step_budget=8,
        inner_replay_capacity=32,
    )
    agent = model.agent
    agent.act(torch.zeros(3), t0=True)
    actor_id = id(agent.inner_engine.state.actor)
    assert agent.inner_engine.state.replay.size == 8
    agent.act(torch.zeros(3), t0=False)
    assert id(agent.inner_engine.state.actor) == actor_id
    assert agent.inner_engine.state.replay.size == 16

    obs = torch.randn(model.cfg.train_unroll_horizon + 1, model.cfg.batch_size, 3)
    actions = torch.randn(model.cfg.train_unroll_horizon, model.cfg.batch_size, 1).tanh()
    rewards = torch.randn(model.cfg.train_unroll_horizon, model.cfg.batch_size, 1)
    agent._update(obs, actions, rewards, torch.zeros_like(rewards))
    agent.act(torch.zeros(3), t0=False)
    assert id(agent.inner_engine.state.actor) == actor_id
    assert agent.inner_engine.state.replay.size == 8

    agent.act(torch.zeros(3), t0=True)
    assert id(agent.inner_engine.state.actor) != actor_id


def test_checkpoint_preflight_rejects_q_architecture_mismatch(tmp_path):
    scalar = _model()
    checkpoint = tmp_path / "scalar.pt"
    scalar.agent.save(checkpoint)
    restored = _model()
    restored.agent.load(checkpoint)
    assert restored.agent.model.critic_signature == scalar.agent.model.critic_signature

    distributional = _model(q_representation="distributional", num_q=5)
    with pytest.raises(ValueError, match="critic specification"):
        distributional.agent.load(checkpoint)


def test_inner_policy_sampling_uses_its_mapping_and_bounds():
    model = _model(
        log_std_mapping="direct_clamp",
        log_std_min=-20.0,
        log_std_max=2.0,
        inner_log_std_mapping="tdmpc2_tanh",
        inner_log_std_min=-10.0,
        inner_log_std_max=2.0,
    )
    with torch.no_grad():
        for parameter in model.agent.model._pi.parameters():
            parameter.zero_()
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    latent = torch.zeros(2, model.cfg.latent_dim)
    noise = torch.ones(2, model.cfg.action_dim)

    action, _ = engine._policy_action(
        latent,
        engine.state.actor,
        mode="policy_sample",
        generator=None,
        noise=noise,
    )

    expected = torch.tanh(torch.full_like(action, float(torch.exp(torch.tensor(-4.0)))))
    torch.testing.assert_close(action, expected, rtol=0, atol=1e-7)


def test_none_operator_executes_outer_policy_with_outer_log_std_mapping(monkeypatch):
    model = _model(
        inner_operator="none",
        inner_model_step_budget=0,
        inner_rounds=0,
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
        inner_temperature_updates_per_action=0,
        log_std_mapping="direct_clamp",
        inner_log_std_mapping="tdmpc2_tanh",
    )
    engine = model.agent.inner_engine
    inner_bounds = []
    original = engine._policy_action

    def tracked_policy_action(*args, **kwargs):
        inner_bounds.append(kwargs.get("inner_bounds"))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_policy_action", tracked_policy_action)
    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

    assert inner_bounds == [False]


def test_diagnostics_keep_outer_and_inner_log_std_semantics_separate(monkeypatch):
    model = _model(
        inner_diagnostic_rollouts=1,
        log_std_mapping="direct_clamp",
        log_std_min=-20.0,
        log_std_max=2.0,
        inner_log_std_mapping="tdmpc2_tanh",
        inner_log_std_min=-10.0,
        inner_log_std_max=2.0,
    )
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    calls = []
    original = model.agent.model.pi

    def tracked_pi(z, *args, **kwargs):
        calls.append(
            (
                kwargs.get("policy"),
                kwargs.get("log_std_mapping"),
                kwargs.get("log_std_min"),
                kwargs.get("log_std_max"),
            )
        )
        return original(z, *args, **kwargs)

    monkeypatch.setattr(model.agent.model, "pi", tracked_pi)
    engine._diagnostics(
        torch.zeros(1, model.cfg.latent_dim),
        engine.state.actor,
    )

    outer_calls = [call for call in calls if call[0] is model.agent.model._pi]
    inner_calls = [call for call in calls if call[0] is engine.state.actor]
    assert outer_calls
    assert inner_calls
    assert all(call[1:] == (None, None, None) for call in outer_calls)
    assert all(
        call[1:] == ("tdmpc2_tanh", -10.0, 2.0) for call in inner_calls
    )


def test_final_outer_policy_kl_is_observational_closed_form_and_restores_modes(
    monkeypatch,
):
    model = _model(inner_diagnostic_rollouts=0, inner_outer_policy_kl_coef=0.0)
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)

    outer_policy = model.agent.model._pi
    improved_policy = engine.state.actor
    outer_policy.eval()
    improved_policy.train()
    observed_modes = []

    def known_gaussian_policy(z, *args, **kwargs):
        del args
        policy = kwargs["policy"]
        observed_modes.append(bool(policy.training))
        is_improved = policy is improved_policy
        pre_tanh_mean = torch.full(
            (*z.shape[:-1], model.cfg.action_dim),
            1.0 if is_improved else 0.0,
            dtype=z.dtype,
            device=z.device,
        )
        log_std = torch.full_like(
            pre_tanh_mean,
            math.log(2.0) if is_improved else 0.0,
        )
        return torch.tanh(pre_tanh_mean), {
            "pre_tanh_mean": pre_tanh_mean,
            "log_std": log_std,
        }

    monkeypatch.setattr(model.agent.model, "pi", known_gaussian_policy)
    metrics = engine._diagnostics(
        torch.zeros(1, model.cfg.latent_dim),
        improved_policy,
    )

    # Per action dimension: -log(2) + 0.5 * (2**2 + 1**2) - 0.5.
    expected = model.cfg.action_dim * (2.0 - math.log(2.0))
    assert metrics["inner_final_outer_policy_kl"] == pytest.approx(expected)
    assert observed_modes == [False, False]
    assert outer_policy.training is False
    assert improved_policy.training is True


def test_persistent_target_intervals_count_optimizer_steps_across_actions():
    model = _model(
        inner_rounds=1,
        inner_model_step_budget=8,
        inner_actor_adaptation="frozen",
        inner_actor_updates_per_action=0,
        inner_critic_updates_per_action=1,
        inner_critic_target_update_interval=2,
        inner_critic_target_tau=1.0,
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_replay_scope="run",
    )
    agent = model.agent
    agent.act(torch.zeros(3), t0=True)
    state = agent.inner_engine.state
    assert state.critic_lifetime_steps == 1
    assert agent.last_inner_metrics["inner_critic_target_updates"] == 0

    agent.act(torch.zeros(3), t0=False)
    assert state.critic_lifetime_steps == 2
    assert agent.last_inner_metrics["inner_critic_target_updates"] == 1
    for online, target in zip(state.critic.parameters(), state.critic_target.parameters()):
        torch.testing.assert_close(online, target, rtol=0, atol=0)


def test_td3_actor_target_has_independent_step_cadence():
    model = _model(
        inner_operator="td3",
        inner_temperature_mode="inherit_outer",
        inner_rounds=1,
        inner_model_step_budget=8,
        inner_critic_adaptation="frozen",
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=1,
        inner_actor_target_update_interval=2,
        inner_actor_target_tau=1.0,
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_replay_scope="run",
    )
    agent = model.agent
    agent.act(torch.zeros(3), t0=True)
    state = agent.inner_engine.state
    assert state.actor_lifetime_steps == 1
    assert agent.last_inner_metrics["inner_actor_target_updates"] == 0
    agent.act(torch.zeros(3), t0=False)
    assert state.actor_lifetime_steps == 2
    assert agent.last_inner_metrics["inner_actor_target_updates"] == 1
    for online, target in zip(state.actor.parameters(), state.actor_target.parameters()):
        torch.testing.assert_close(online, target, rtol=0, atol=0)


def test_persistent_clone_rebase_preserves_online_target_lag():
    model = _model(
        inner_model_step_budget=0,
        inner_actor_adaptation="frozen",
        inner_critic_adaptation="clone",
        inner_actor_updates_per_action=0,
        inner_critic_updates_per_action=0,
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_replay_scope="run",
    )
    agent = model.agent
    agent.act(torch.zeros(3), t0=True)
    state = agent.inner_engine.state
    with torch.no_grad():
        for parameter in state.critic.parameters():
            parameter.add_(0.3)
        for parameter in state.critic_target.parameters():
            parameter.add_(0.1)
        lag_before = [
            online.detach().clone() - target.detach().clone()
            for online, target in zip(
                state.critic.parameters(), state.critic_target.parameters()
            )
        ]
        for parameter in agent.model._Qs.parameters():
            parameter.add_(0.5)
    agent.outer_version += 1
    with agent.inner_engine.rng.fork("initialization"):
        agent.inner_engine._prepare_workspace(t0=False)
    for online, target, expected_lag in zip(
        state.critic.parameters(), state.critic_target.parameters(), lag_before
    ):
        torch.testing.assert_close(online - target, expected_lag, rtol=0, atol=1e-7)


def test_diagnostics_and_unrelated_inner_compute_have_independent_rng_streams():
    no_diagnostics = _model(inner_diagnostic_rollouts=0)
    with_diagnostics = _model(inner_diagnostic_rollouts=3)
    first = no_diagnostics.agent.act(torch.zeros(3), eval_mode=False)
    second = with_diagnostics.agent.act(torch.zeros(3), eval_mode=False)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    collect_only = _model(
        inner_actor_adaptation="frozen",
        inner_actor_updates_per_action=0,
        inner_critic_updates_per_action=0,
    )
    critic_only = _model(
        inner_actor_adaptation="frozen",
        inner_actor_updates_per_action=0,
        inner_critic_updates_per_action=2,
    )
    collect_action = collect_only.agent.act(torch.zeros(3), eval_mode=False)
    critic_action = critic_only.agent.act(torch.zeros(3), eval_mode=False)
    torch.testing.assert_close(collect_action, critic_action, rtol=0, atol=0)


def test_lora_dropout_diagnostics_leave_cpu_rng_bitwise_unchanged():
    model = _model(
        q_representation="distributional",
        num_q=5,
        dropout=0.2,
        inner_actor_adaptation="lora",
        inner_critic_adaptation="lora",
        inner_actor_lora_dropout=0.3,
        inner_critic_lora_dropout=0.3,
        inner_diagnostic_rollouts=2,
    )
    before = torch.random.get_rng_state().clone()
    model.agent.act(torch.zeros(3), eval_mode=False)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_act_restores_outer_model_training_mode():
    model = _model()
    model.agent.model.train()
    assert model.agent.model.training
    assert not model.agent.model._target_Qs.training
    model.agent.act(torch.zeros(3))
    assert model.agent.model.training
    assert not model.agent.model._target_Qs.training


def test_checkpoint_load_resets_discarded_inner_rng_state(tmp_path):
    source = _model()
    checkpoint = tmp_path / "source.pt"
    source.agent.save(checkpoint)

    used = _model()
    fresh = _model()
    used.agent.act(torch.zeros(3), eval_mode=False)
    used.agent.load(checkpoint)
    fresh.agent.load(checkpoint)
    used_action = used.agent.act(torch.zeros(3), eval_mode=False)
    fresh_action = fresh.agent.act(torch.zeros(3), eval_mode=False)
    torch.testing.assert_close(used_action, fresh_action, rtol=0, atol=0)


def test_checkpoint_preflight_rejects_entropy_mode_before_mutation(tmp_path):
    automatic = _model(ent_coef="auto_0.7")
    checkpoint = tmp_path / "automatic.pt"
    automatic.agent.save(checkpoint)
    fixed = _model(ent_coef=0.123)
    before = _clone_tree(fixed.agent.model.state_dict())
    with pytest.raises(ValueError, match="entropy"):
        fixed.agent.load(checkpoint)
    _assert_tree_equal(fixed.agent.model.state_dict(), before)


def test_checkpoint_policy_spec_records_and_rejects_mapping_mismatch():
    source = _model(
        log_std_mapping="direct_clamp",
        log_std_min=-10.0,
        log_std_max=2.0,
    )
    checkpoint = _clone_tree(source.agent.checkpoint_state())
    assert checkpoint["policy_spec"] == {
        "log_std_mapping": "direct_clamp",
        "log_std_min": -10.0,
        "log_std_max": 2.0,
    }

    incompatible = _model(
        log_std_mapping="tdmpc2_tanh",
        log_std_min=-10.0,
        log_std_max=2.0,
    )
    incompatible.agent.num_updates = 7
    incompatible.agent.outer_version = 7
    before = _clone_tree(incompatible.agent.model.state_dict())
    with pytest.raises(ValueError, match="policy specification"):
        incompatible.agent.load(checkpoint)
    _assert_tree_equal(incompatible.agent.model.state_dict(), before)
    assert incompatible.agent.num_updates == 7
    assert incompatible.agent.outer_version == 7


@pytest.mark.parametrize(
    ("actor_loss_scale_mode", "checkpoint_version"),
    [("none", 3), ("tdmpc2_percentile_range", 4)],
)
def test_pre_mapping_policy_spec_preserves_direct_clamp_checkpoint_compatibility(
    actor_loss_scale_mode, checkpoint_version
):
    source = _model(
        log_std_mapping="direct_clamp",
        sac_actor_loss_scale_mode=actor_loss_scale_mode,
    )
    checkpoint = _clone_tree(source.agent.checkpoint_state())
    assert checkpoint["checkpoint_version"] == checkpoint_version
    checkpoint["policy_spec"].pop("log_std_mapping")

    restored = _model(
        log_std_mapping="direct_clamp",
        sac_actor_loss_scale_mode=actor_loss_scale_mode,
    )
    restored.agent.load(checkpoint)
    _assert_tree_equal(
        restored.agent.model.state_dict(), source.agent.model.state_dict()
    )

    incompatible = _model(
        log_std_mapping="tdmpc2_tanh",
        sac_actor_loss_scale_mode=actor_loss_scale_mode,
    )
    with pytest.raises(ValueError, match="policy specification"):
        incompatible.agent.load(checkpoint)

    exact = _clone_tree(source.agent.training_state_dict())
    exact["outer"]["policy_spec"].pop("log_std_mapping")
    exact_restored = _model(
        log_std_mapping="direct_clamp",
        sac_actor_loss_scale_mode=actor_loss_scale_mode,
    )
    exact_restored.agent.load_training_state_dict(exact)
    _assert_tree_equal(
        exact_restored.agent.model.state_dict(), source.agent.model.state_dict()
    )


@pytest.mark.parametrize(
    ("representation", "num_q"),
    [("scalar", 2), ("distributional", 5)],
)
def test_native_checkpoint_roundtrips_all_outer_training_state(
    tmp_path, representation, num_q
):
    source = _model(q_representation=representation, num_q=num_q)
    obs = torch.randn(source.cfg.train_unroll_horizon + 1, source.cfg.batch_size, 3)
    actions = torch.randn(source.cfg.train_unroll_horizon, source.cfg.batch_size, 1).tanh()
    rewards = torch.randn(source.cfg.train_unroll_horizon, source.cfg.batch_size, 1)
    source.agent._update(obs, actions, rewards, torch.zeros_like(rewards))
    checkpoint = tmp_path / f"{representation}.pt"
    source.agent.save(checkpoint)

    restored = _model(q_representation=representation, num_q=num_q)
    restored.agent.act(torch.zeros(3))
    assert restored.agent.inner_engine.state.outer_version >= 0
    restored.agent.load(checkpoint)
    _assert_tree_equal(
        restored.agent.model.state_dict(), source.agent.model.state_dict()
    )
    _assert_tree_equal(restored.agent.optim.state_dict(), source.agent.optim.state_dict())
    _assert_tree_equal(
        restored.agent.pi_optim.state_dict(), source.agent.pi_optim.state_dict()
    )
    _assert_tree_equal(
        restored.agent.ent_coef_optim.state_dict(),
        source.agent.ent_coef_optim.state_dict(),
    )
    torch.testing.assert_close(restored.agent.alpha, source.agent.alpha, rtol=0, atol=0)
    assert restored.agent.num_updates == source.agent.num_updates
    assert restored.agent.outer_version == source.agent.outer_version
    assert restored.agent.inner_engine.state.outer_version == -1


def test_portable_checkpoint_keeps_receiving_optimizer_hyperparameters():
    source = _model(
        lr=2e-3,
        enc_lr_scale=0.2,
        critic_lr=3e-3,
        actor_lr=4e-3,
        ent_coef_lr=5e-3,
        adam_eps=2e-8,
        actor_adam_eps=2e-5,
    )
    obs = torch.randn(source.cfg.train_unroll_horizon + 1, source.cfg.batch_size, 3)
    actions = torch.randn(
        source.cfg.train_unroll_horizon, source.cfg.batch_size, 1
    ).tanh()
    rewards = torch.randn(source.cfg.train_unroll_horizon, source.cfg.batch_size, 1)
    source.agent._update(obs, actions, rewards, torch.zeros_like(rewards))

    restored = _model(
        lr=6e-3,
        enc_lr_scale=0.4,
        critic_lr=7e-3,
        actor_lr=8e-3,
        ent_coef_lr=9e-3,
        adam_eps=3e-8,
        actor_adam_eps=3e-5,
    )
    optimizers = (
        (source.agent.optim, restored.agent.optim),
        (source.agent.pi_optim, restored.agent.pi_optim),
        (source.agent.ent_coef_optim, restored.agent.ent_coef_optim),
    )
    for saved, _ in optimizers:
        _set_foreign_optimizer_group_options(saved)
    configured_options = [
        _optimizer_group_options(target) for _, target in optimizers
    ]

    restored.agent.load(source.agent.checkpoint_state())

    for (saved, target), expected_options in zip(optimizers, configured_options):
        _assert_tree_equal(_optimizer_group_options(target), expected_options)
        assert target.state_dict()["state"]
        _assert_tree_equal(target.state_dict()["state"], saved.state_dict()["state"])


def test_portable_checkpoint_keeps_receiving_fixed_entropy_coefficient():
    source = _model(ent_coef=0.2)
    restored = _model(ent_coef=0.7)

    restored.agent.load(source.agent.checkpoint_state())

    torch.testing.assert_close(
        restored.agent.fixed_ent_coef,
        torch.tensor(0.7, device=restored.agent.device),
        rtol=0,
        atol=0,
    )


def test_legacy_scalar_checkpoint_and_vectorized_distributional_model_import():
    scalar = _model()
    legacy = {
        "model": _clone_tree(scalar.agent.model.state_dict()),
        "optim": _clone_tree(scalar.agent.optim.state_dict()),
        "pi_optim": _clone_tree(scalar.agent.pi_optim.state_dict()),
        "num_updates": 0,
        "log_ent_coef": scalar.agent.log_ent_coef.detach().cpu().clone(),
        "ent_coef_optim": _clone_tree(scalar.agent.ent_coef_optim.state_dict()),
    }
    restored_scalar = _model()
    restored_scalar.agent.load(legacy)
    _assert_tree_equal(
        restored_scalar.agent.model.state_dict(), scalar.agent.model.state_dict()
    )

    distributional = _model(q_representation="distributional", num_q=5)
    port_state = distributional.agent.model.state_dict()
    official = OrderedDict()
    prefixes = ("_Qs.modules_list.", "_target_Qs.modules_list.")
    for key, value in port_state.items():
        if not key.startswith(prefixes):
            official[key] = value.clone()

    def pack(port_prefix, official_prefix):
        grouped = {}
        for key, value in port_state.items():
            if not key.startswith(port_prefix):
                continue
            remainder = key[len(port_prefix):]
            critic_index, layer_and_field = remainder.split(".", 1)
            grouped.setdefault(layer_and_field, {})[int(critic_index)] = value
        for layer_and_field, values in grouped.items():
            official[official_prefix + layer_and_field] = torch.stack(
                [values[index] for index in sorted(values)], dim=0
            )

    pack("_Qs.modules_list.", "_Qs.params.")
    pack("_Qs.modules_list.", "_detach_Qs_params.")
    pack("_target_Qs.modules_list.", "_target_Qs_params.")
    imported = _model(q_representation="distributional", num_q=5)
    imported.agent.load(official)
    _assert_tree_equal(imported.agent.model.state_dict(), port_state)


def test_unknown_checkpoint_version_fails_without_partial_mutation():
    model = _model()
    before = _clone_tree(model.agent.model.state_dict())
    invalid = {
        "checkpoint_version": 999,
        "model": _clone_tree(model.agent.model.state_dict()),
    }
    with pytest.raises(ValueError, match="Unsupported"):
        model.agent.load(invalid)
    _assert_tree_equal(model.agent.model.state_dict(), before)


def test_portable_checkpoint_optimizer_preflight_prevents_partial_mutation():
    source = _model()
    obs = torch.randn(source.cfg.train_unroll_horizon + 1, source.cfg.batch_size, 3)
    actions = torch.randn(
        source.cfg.train_unroll_horizon, source.cfg.batch_size, 1
    ).tanh()
    rewards = torch.randn(source.cfg.train_unroll_horizon, source.cfg.batch_size, 1)
    source.agent._update(obs, actions, rewards, torch.zeros_like(rewards))
    invalid = _clone_tree(source.agent.checkpoint_state())
    first_state = next(iter(invalid["optim"]["state"].values()))
    first_state["exp_avg"] = torch.zeros(999)

    target = _model()
    before_model = _clone_tree(target.agent.model.state_dict())
    before_optimizer = _clone_tree(target.agent.optim.state_dict())
    before_counters = (target.agent.num_updates, target.agent.outer_version)

    with pytest.raises(ValueError, match="exp_avg"):
        target.agent.load(invalid)

    _assert_tree_equal(target.agent.model.state_dict(), before_model)
    _assert_tree_equal(target.agent.optim.state_dict(), before_optimizer)
    assert (target.agent.num_updates, target.agent.outer_version) == before_counters


def test_version_three_checkpoint_records_observation_contract():
    model = _model()
    checkpoint = model.agent.checkpoint_state()
    assert checkpoint["checkpoint_version"] == 3
    assert checkpoint["observation_spec"] == {
        "mode": "state",
        "shape": [3],
        "dtype": "float32",
    }
    assert checkpoint["critic_target_spec"] == {
        "outer_critic_target": "entropy_augmented",
        "inner_sac_critic_target": "entropy_augmented",
    }


def test_portable_checkpoint_allows_intentional_critic_target_ablation():
    source = _model(
        outer_critic_target="entropy_augmented",
        inner_sac_critic_target="entropy_augmented",
    )
    checkpoint = _clone_tree(source.agent.checkpoint_state())

    reward_return = _model(
        outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
    )
    reward_return.agent.load(checkpoint)

    assert reward_return.agent._critic_target_spec() == {
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
    }
    _assert_tree_equal(
        reward_return.agent.model.state_dict(), source.agent.model.state_dict()
    )


@pytest.mark.parametrize(
    ("actor_loss_scale_mode", "checkpoint_version"),
    [("none", 3), ("tdmpc2_percentile_range", 4)],
)
def test_pre_target_spec_portable_checkpoint_still_loads(
    actor_loss_scale_mode, checkpoint_version
):
    source = _model(sac_actor_loss_scale_mode=actor_loss_scale_mode)
    checkpoint = _clone_tree(source.agent.checkpoint_state())
    assert checkpoint["checkpoint_version"] == checkpoint_version
    checkpoint.pop("critic_target_spec")

    restored = _model(
        sac_actor_loss_scale_mode=actor_loss_scale_mode,
        outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
    )
    restored.agent.load(checkpoint)
    _assert_tree_equal(
        restored.agent.model.state_dict(), source.agent.model.state_dict()
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("outer_critic_target", "reward_only"),
        ("inner_sac_critic_target", "reward_only"),
    ],
)
def test_exact_training_state_rejects_critic_target_semantic_change(key, value):
    source = _model()
    checkpoint = _clone_tree(source.agent.training_state_dict())
    configured = _model(**{key: value})
    pristine = _clone_tree(configured.agent.training_state_dict())

    with pytest.raises(ValueError, match="critic-target specification"):
        configured.agent.load_training_state_dict(checkpoint)

    _assert_tree_equal(configured.agent.training_state_dict(), pristine)


@pytest.mark.parametrize("legacy_version", [1, 2])
def test_versioned_legacy_checkpoint_without_new_metadata_still_loads(
    legacy_version,
):
    source = _model()
    checkpoint = _clone_tree(source.agent.checkpoint_state())
    checkpoint["checkpoint_version"] = legacy_version
    checkpoint.pop("observation_spec")
    checkpoint.pop("critic_target_spec")

    restored = _model(
        outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
    )
    restored.agent.load(checkpoint)
    _assert_tree_equal(
        restored.agent.model.state_dict(), source.agent.model.state_dict()
    )


def test_ambi_observation_mismatch_fails_before_outer_state_mutation():
    model = _model()
    model.agent.num_updates = 7
    model.agent.outer_version = 11
    before_model = _clone_tree(model.agent.model.state_dict())
    before_optim = _clone_tree(model.agent.optim.state_dict())
    checkpoint = _clone_tree(model.agent.checkpoint_state())
    checkpoint["observation_spec"] = {
        "mode": "rgb",
        "shape": [9, 64, 64],
        "dtype": "uint8",
    }
    checkpoint["num_updates"] = 99
    checkpoint["outer_version"] = 99

    with pytest.raises(ValueError, match="observation specification"):
        model.agent.load(checkpoint)

    _assert_tree_equal(model.agent.model.state_dict(), before_model)
    _assert_tree_equal(model.agent.optim.state_dict(), before_optim)
    assert model.agent.num_updates == 7
    assert model.agent.outer_version == 11


@pytest.mark.parametrize(
    ("representation", "num_q", "adaptation"),
    [
        ("scalar", 2, "frozen"),
        ("scalar", 2, "clone"),
        ("scalar", 2, "lora"),
        ("distributional", 5, "frozen"),
        ("distributional", 5, "clone"),
        ("distributional", 5, "lora"),
    ],
)
def test_tiny_pendulum_end_to_end_across_q_and_adaptation_modes(
    representation, num_q, adaptation
):
    update_count = 0 if adaptation == "frozen" else 2
    model = _model(
        q_representation=representation,
        num_q=num_q,
        inner_actor_adaptation=adaptation,
        inner_critic_adaptation=adaptation,
        inner_actor_updates_per_action=update_count,
        inner_critic_updates_per_action=update_count,
    )
    model.learn(total_timesteps=6)
    assert model._global_step == 6
    assert model.agent.num_updates > 0
    assert model.agent.last_inner_metrics["inner_model_steps"] == 16
    _assert_finite(model.agent.last_inner_metrics)
    _assert_finite(model._last_train_metrics)


def test_predicted_termination_reports_realized_compute_and_strict_underfill():
    model = _model(episodic=True)

    def terminate_immediately(z, task=None, unnormalized=False):
        del task
        value = torch.ones(z.shape[0], 1, device=z.device)
        return value if not unnormalized else torch.full_like(value, 20.0)

    model.agent.model.termination = terminate_immediately
    model.agent.act(torch.zeros(3))
    metrics = model.agent.last_inner_metrics
    assert metrics["inner_model_steps_budget"] == 16
    assert metrics["inner_model_steps"] == 8
    assert metrics["inner_requested_rollouts"] == 8
    assert metrics["inner_rollouts"] == 8
    assert metrics["inner_termination_rate"] == 1.0
    assert model.agent.last_inner_rollout_lengths == [1] * 8

    strict = _model(episodic=True, inner_replay_sampling="without_replacement")
    strict.agent.model.termination = terminate_immediately
    with pytest.raises(ValueError, match="without replacement"):
        strict.agent.act(torch.zeros(3))


def _toy_policy(z, task=None, *, policy=None, deterministic=False, **kwargs):
    del task, deterministic, kwargs
    raw = policy(z)
    mean, log_std = raw.chunk(2, dim=-1)
    action = mean.tanh()
    log_prob = mean.sum(dim=-1, keepdim=True) * 0.0 - 0.5
    return action, {
        "mean": action,
        "pre_tanh_mean": mean,
        "log_std": log_std * 0.0,
        "log_prob": log_prob,
        "entropy": -log_prob,
    }


def _actor_q_heads(world_model, values):
    values = tuple(float(value) for value in values)
    assert len(values) == int(world_model.q_backend.num_q)

    def critic(z, action, **kwargs):
        q_base = action.sum(dim=-1, keepdim=True) * 0.0
        q_all = torch.stack(tuple(q_base + value for value in values))
        if kwargs.get("reduction") == "all":
            return q_all
        return world_model.q_backend.reduce(
            q_all,
            kwargs.get("reduction", "min_pair"),
            pair_indices=kwargs.get("pair_indices"),
            trusted_pair_indices=kwargs.get("trusted_pair_indices", False),
        )

    return critic


def _constant_actor_q(world_model, value):
    return _actor_q_heads(
        world_model,
        (value,) * int(world_model.q_backend.num_q),
    )


@pytest.mark.parametrize(
    ("target_mode", "entropy_bonus"),
    [("entropy_augmented", 0.5), ("reward_only", 0.0)],
)
def test_sac_inner_target_and_actor_objective_match_hand_computation(
    monkeypatch, target_mode, entropy_bonus
):
    model = _model(
        inner_rounds=1,
        inner_model_step_budget=8,
        inner_sac_critic_target=target_mode,
    )
    engine = model.agent.inner_engine
    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    engine._compile_regions["critic"].enabled = True
    assert (
        engine._compile_regions["actor"].eager.__func__
        is engine._sac_actor_kernel.__func__
    )
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    batch = {
        "z": torch.zeros(4, model.cfg.latent_dim),
        "action": torch.zeros(4, model.cfg.action_dim),
        "reward": torch.full((4, 1), 3.0),
        "next_z": torch.zeros(4, model.cfg.latent_dim),
        "terminated": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    }
    monkeypatch.setattr(model.agent.model, "pi", _toy_policy)
    monkeypatch.setattr(
        engine,
        "_bootstrap_q",
        lambda z, action: torch.full((z.shape[0], 1), 5.0),
    )
    captured = {}
    original_loss = model.agent.model.critic_loss

    def capture_loss(predictions, scalar_target, **kwargs):
        captured["target"] = scalar_target.detach().clone()
        return original_loss(predictions, scalar_target, **kwargs)

    monkeypatch.setattr(model.agent.model, "critic_loss", capture_loss)
    alpha = torch.tensor(0.25)
    with engine.rng.fork("bootstrap"):
        engine._sac_critic_step(batch, alpha)
    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {"fullgraph": False, "dynamic": False}
    expected_target = batch["reward"] + model.agent.discount * (
        1.0 - batch["terminated"]
    ) * (5.0 + entropy_bonus * alpha)
    torch.testing.assert_close(captured["target"], expected_target)

    actor_q_calls = []
    actor_q = _actor_q_heads(model.agent.model, (2.0, 6.0))

    def tracked_actor_q(z, action, **kwargs):
        actor_q_calls.append(dict(kwargs))
        return actor_q(z, action, **kwargs)

    monkeypatch.setattr(model.agent.model, "Q", tracked_actor_q)
    engine._compile_regions["actor"].enabled = True
    actor_metrics = engine._sac_policy_step(
        batch,
        update_temperature=False,
        update_actor=True,
        alpha=alpha,
    )
    assert len(compile_calls) == 2
    assert compile_calls[1][1] == {"fullgraph": False, "dynamic": False}
    assert len(actor_q_calls) == 1
    assert actor_q_calls[0]["reduction"] == "all"
    assert actor_q_calls[0]["detach"] is True
    assert actor_q_calls[0]["qs"] is engine.state.critic
    assert actor_metrics["actor_loss"] == pytest.approx(-2.125)
    assert actor_metrics["actor_q_mean_all"] == pytest.approx(4.0)
    assert actor_metrics["actor_q_min_all"] == pytest.approx(2.0)
    assert actor_metrics["actor_q_mean_all_minus_min_all"] == pytest.approx(2.0)


def test_scaled_sac_inner_actor_divides_full_objective_but_not_temperature(
    monkeypatch,
):
    model = _model(
        inner_rounds=1,
        inner_model_step_budget=8,
        inner_outer_policy_kl_coef=0.5,
        inner_temperature_mode="auto",
        inner_temperature_updates_per_action=1,
        inner_temperature_initialization="fixed",
        inner_temperature=0.25,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
    )
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    batch = {
        "z": torch.zeros(4, model.cfg.latent_dim),
        "action": torch.zeros(4, model.cfg.action_dim),
        "reward": torch.zeros(4, 1),
        "next_z": torch.zeros(4, model.cfg.latent_dim),
        "terminated": torch.zeros(4, 1),
    }
    monkeypatch.setattr(model.agent.model, "pi", _toy_policy)
    monkeypatch.setattr(
        model.agent.model,
        "Q",
        _constant_actor_q(model.agent.model, 2.0),
    )
    monkeypatch.setattr(
        engine,
        "_gaussian_kl",
        lambda inner_info, outer_info: inner_info["log_prob"].new_full(
            inner_info["log_prob"].shape, 4.0
        ),
    )
    assert (
        engine._compile_regions["actor"].eager.__func__
        is engine._scaled_sac_actor_kernel.__func__
    )

    alpha = torch.tensor(0.25)
    expected_temperature_loss = -(
        engine.state.log_alpha.detach()
        * (-0.5 + engine._resolved_inner_target_entropy())
    ).mean()
    metrics = engine._sac_policy_step(
        batch,
        update_temperature=True,
        update_actor=True,
        alpha=alpha,
        actor_loss_scale=torch.tensor(2.0),
    )

    # The raw full objective is 0.25*(-0.5) - 2 + 0.5*4 = -0.125.
    assert metrics["actor_loss"] == pytest.approx(-0.125 / 2.0)
    assert metrics["outer_policy_kl"] == pytest.approx(4.0)
    torch.testing.assert_close(
        metrics["temperature_loss"], expected_temperature_loss
    )


def test_scaled_sac_actor_compile_region_caches_explicit_scale_argument(monkeypatch):
    model = _model(
        inner_rounds=1,
        inner_model_step_budget=8,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
    )
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    batch = {
        "z": torch.zeros(4, model.cfg.latent_dim),
        "action": torch.zeros(4, model.cfg.action_dim),
        "reward": torch.zeros(4, 1),
        "next_z": torch.zeros(4, model.cfg.latent_dim),
        "terminated": torch.zeros(4, 1),
    }
    compile_calls = []
    observed_scales = []

    def fake_compile(function, **kwargs):
        compile_calls.append((function, kwargs))

        def compiled(*args, **call_kwargs):
            observed_scales.append(args[2].detach().clone())
            return function(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    engine._compile_regions["actor"].enabled = True
    for scale in (2.0, 3.0):
        engine._sac_policy_step(
            batch,
            update_temperature=False,
            update_actor=False,
            alpha=torch.tensor(0.25),
            actor_loss_scale=torch.tensor(scale),
        )

    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {"fullgraph": False, "dynamic": False}
    torch.testing.assert_close(observed_scales[0], torch.tensor(2.0))
    torch.testing.assert_close(observed_scales[1], torch.tensor(3.0))


@pytest.mark.parametrize(
    ("mode", "actor_loss_scale", "message"),
    [
        ("tdmpc2_percentile_range", None, "requires an action-frozen"),
        ("none", torch.tensor(2.0), "mode='none'"),
    ],
)
def test_sac_policy_step_rejects_actor_loss_scale_mode_mismatch(
    mode,
    actor_loss_scale,
    message,
):
    model = _model(sac_actor_loss_scale_mode=mode)
    engine = model.agent.inner_engine

    with pytest.raises(RuntimeError, match=message):
        engine._sac_policy_step(
            {"z": torch.zeros(1, model.cfg.latent_dim)},
            update_temperature=False,
            update_actor=False,
            alpha=torch.tensor(0.25),
            actor_loss_scale=actor_loss_scale,
        )


def test_sac_actor_loss_scale_is_snapshotted_once_per_real_action(monkeypatch):
    model = _model(sac_actor_loss_scale_mode="tdmpc2_percentile_range")
    agent = model.agent
    engine = agent.inner_engine
    scale_reads = 0
    action_scales = (2.0, 3.0)

    def read_scale(_agent):
        nonlocal scale_reads
        scale = action_scales[scale_reads]
        scale_reads += 1
        return torch.tensor([scale])

    monkeypatch.setattr(
        type(agent),
        "actor_loss_scale",
        property(read_scale),
        raising=False,
    )
    observed_scales = []
    scaled_kernel = engine._compile_regions["actor"].eager

    def record_scale(*args, **kwargs):
        observed_scales.append(args[2].detach().clone())
        return scaled_kernel(*args, **kwargs)

    engine._compile_regions["actor"].eager = record_scale
    action_metrics = []
    for _ in action_scales:
        agent.act(torch.zeros(3), collect_diagnostics=False)
        action_metrics.append(dict(agent.last_inner_metrics))

    assert scale_reads == len(action_scales)
    slots_per_action = model.cfg.inner_actor_updates_per_action
    assert len(observed_scales) == len(action_scales) * slots_per_action
    for action_index, expected_scale in enumerate(action_scales):
        start = action_index * slots_per_action
        stop = start + slots_per_action
        for scale in observed_scales[start:stop]:
            torch.testing.assert_close(scale, torch.tensor([expected_scale]))
        metrics = action_metrics[action_index]
        assert metrics["inner_actor_loss_scale"] == pytest.approx(expected_scale)
        assert metrics["inner_effective_alpha"] == pytest.approx(
            metrics["inner_alpha_final"] / expected_scale
        )


def test_td3_inner_target_and_actor_objective_match_hand_computation(monkeypatch):
    model = _model(
        inner_operator="td3",
        inner_temperature_mode="inherit_outer",
        inner_rounds=1,
        inner_model_step_budget=8,
        inner_td3_target_noise_std=0.0,
    )
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    batch = {
        "z": torch.zeros(4, model.cfg.latent_dim),
        "action": torch.zeros(4, model.cfg.action_dim),
        "reward": torch.full((4, 1), 1.5),
        "next_z": torch.zeros(4, model.cfg.latent_dim),
        "terminated": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    }
    monkeypatch.setattr(model.agent.model, "pi", _toy_policy)
    monkeypatch.setattr(
        engine,
        "_bootstrap_q",
        lambda z, action: torch.full((z.shape[0], 1), 4.0),
    )
    captured = {}
    original_loss = model.agent.model.critic_loss

    def capture_loss(predictions, scalar_target, **kwargs):
        captured["target"] = scalar_target.detach().clone()
        return original_loss(predictions, scalar_target, **kwargs)

    monkeypatch.setattr(model.agent.model, "critic_loss", capture_loss)
    with engine.rng.fork("bootstrap"):
        engine._td3_critic_step(batch)
    expected_target = batch["reward"] + model.agent.discount * (
        1.0 - batch["terminated"]
    ) * 4.0
    torch.testing.assert_close(captured["target"], expected_target)

    monkeypatch.setattr(
        model.agent.model,
        "Q",
        _constant_actor_q(model.agent.model, 3.0),
    )
    with engine.rng.fork("gradient_policy"):
        actor_metrics = engine._td3_actor_step(batch)
    assert actor_metrics["actor_loss"] == pytest.approx(-3.0)
    assert actor_metrics["actor_q_mean_all"] == pytest.approx(3.0)
    assert actor_metrics["actor_q_min_all"] == pytest.approx(3.0)
    assert actor_metrics["actor_q_mean_all_minus_min_all"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("target_mode", "entropy_bonus"),
    [("entropy_augmented", 0.4), ("reward_only", 0.0)],
)
def test_outer_scalar_target_and_optimizer_partition_are_preserved(
    monkeypatch, target_mode, entropy_bonus
):
    model = _model(
        q_representation="scalar",
        num_q=2,
        outer_critic_target=target_mode,
    )
    agent = model.agent

    def outer_policy(z, *args, **kwargs):
        action = z.new_zeros(z.shape[0], model.cfg.action_dim)
        return action, {"log_prob": z.new_full((z.shape[0], 1), -0.4)}

    monkeypatch.setattr(agent.model, "pi", outer_policy)
    monkeypatch.setattr(
        agent.model,
        "Q",
        lambda z, action, **kwargs: z.new_full((z.shape[0], 1), 6.0),
    )
    reward = torch.tensor([[2.0], [2.0]])
    terminated = torch.tensor([[0.0], [1.0]])
    target = agent._soft_td_target(
        torch.zeros(2, model.cfg.latent_dim), reward, terminated
    )
    expected = reward + agent.discount * (1.0 - terminated) * (
        6.0 + entropy_bonus * agent.alpha.detach()
    )
    torch.testing.assert_close(target, expected)

    groups = agent.optim.param_groups
    assert len(groups) == 4
    assert groups[0]["lr"] == pytest.approx(model.cfg.lr * model.cfg.enc_lr_scale)
    assert groups[-1]["lr"] == pytest.approx(model.cfg.critic_lr)
    critic_ids = {id(parameter) for parameter in agent.model._Qs.parameters()}
    actor_ids = {id(parameter) for parameter in agent.model._pi.parameters()}
    assert {id(parameter) for parameter in groups[-1]["params"]} == critic_ids
    assert not actor_ids.intersection(
        id(parameter)
        for group in groups
        for parameter in group["params"]
    )


def test_outer_actor_and_inner_sac_adam_epsilons_are_wired_independently():
    model = _model(
        adam_eps=1e-8,
        actor_adam_eps=1e-5,
        inner_adam_eps=3e-8,
        inner_temperature_mode="auto",
        inner_temperature_updates_per_action=2,
    )
    agent = model.agent

    assert {group["eps"] for group in agent.optim.param_groups} == {1e-8}
    assert {group["eps"] for group in agent.pi_optim.param_groups} == {1e-5}
    assert {group["eps"] for group in agent.ent_coef_optim.param_groups} == {1e-8}

    engine = agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    inner_optimizers = (
        engine.state.actor_optim,
        engine.state.critic_optim,
        engine.state.temperature_optim,
    )
    assert all(optimizer is not None for optimizer in inner_optimizers)
    for optimizer in inner_optimizers:
        assert {group["eps"] for group in optimizer.param_groups} == {3e-8}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_act_preserves_all_global_rng_streams_and_outer_state():
    model = _model(
        device="cuda",
        q_representation="distributional",
        num_q=5,
        dropout=0.1,
        inner_actor_adaptation="lora",
        inner_critic_adaptation="lora",
        inner_actor_lora_dropout=0.2,
        inner_critic_lora_dropout=0.2,
        inner_diagnostic_rollouts=2,
    )
    agent = model.agent
    outer_before = _clone_tree(agent.model.state_dict())
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    agent.act(torch.zeros(3))
    _assert_tree_equal(agent.model.state_dict(), outer_before)
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng_before, rtol=0, atol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng_before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_action_lora_workspace_reuses_adapters_and_shares_outer_bases():
    model = _model(
        inner_actor_adaptation="lora",
        inner_critic_adaptation="lora",
        inner_actor_updates_per_action=0,
        inner_critic_updates_per_action=0,
    )
    engine = model.agent.inner_engine
    outer = model.agent.model
    outer_keys = tuple(outer.state_dict())
    outer_parameter_ids = {id(parameter) for parameter in outer.parameters()}
    outer_modes = {
        name: module.training for name, module in outer.named_modules()
    }

    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    actor = engine.state.actor
    critic = engine.state.critic
    critic_target = engine.state.critic_target
    assert lora_uses_shared_bases(actor)
    assert lora_uses_shared_bases(critic)
    assert lora_uses_shared_bases(critic_target)
    assert engine.state.actor_anchor is None
    assert engine.state.critic_anchor is None
    assert outer_parameter_ids.isdisjoint(id(parameter) for parameter in actor.parameters())
    assert outer_parameter_ids.isdisjoint(id(parameter) for parameter in critic.parameters())

    for component, source in ((actor, outer._pi), (critic, outer._Qs)):
        for path, adapter in component.named_modules():
            if isinstance(adapter, (LoRALinear, LoRANormedLinear)):
                assert adapter.base is source.get_submodule(path)

    engine._clear_expired(t0=False, include_action=True)
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=False)
    assert engine.state.actor is actor
    assert engine.state.critic is critic
    assert tuple(outer.state_dict()) == outer_keys
    assert {
        name: module.training for name, module in outer.named_modules()
    } == outer_modes
    assert all(parameter.requires_grad for parameter in outer._pi.parameters())
    assert all(parameter.requires_grad for parameter in outer._Qs.parameters())


@pytest.mark.parametrize("rebase", [False, True])
def test_persistent_lora_shares_only_when_live_rebasing_preserves_semantics(rebase):
    model = _model(
        inner_actor_adaptation="lora",
        inner_critic_adaptation="lora",
        inner_actor_updates_per_action=0,
        inner_critic_updates_per_action=0,
        inner_actor_scope="run",
        inner_critic_scope="run",
        inner_actor_optimizer_scope="run",
        inner_critic_optimizer_scope="run",
        inner_rebase_persistent=rebase,
    )
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)

    assert lora_uses_shared_bases(engine.state.actor) is rebase
    assert lora_uses_shared_bases(engine.state.critic) is rebase
    assert lora_uses_shared_bases(engine.state.critic_target) is rebase
    if rebase:
        assert engine.state.actor_anchor is None
        assert engine.state.critic_anchor is None
    else:
        assert engine.state.actor_anchor is not None
        assert engine.state.critic_anchor is not None


def test_shared_lora_outer_regularizer_cannot_accumulate_outer_gradients():
    model = _model(
        inner_actor_adaptation="lora",
        inner_critic_adaptation="lora",
        inner_outer_policy_kl_coef=0.3,
    )
    outer = model.agent.model
    before = _clone_tree(outer.state_dict())
    assert all(parameter.grad is None for parameter in outer.parameters())

    model.agent.act(torch.zeros(3), collect_diagnostics=False)

    assert all(parameter.grad is None for parameter in outer.parameters())
    _assert_tree_equal(outer.state_dict(), before)
