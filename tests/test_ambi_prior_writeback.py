from copy import deepcopy
import random

import gymnasium as gym
import numpy as np
import pytest
import torch
import utils.resume_identity as resume_identity

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core import inner_improvement


def _build_cfg(**params):
    """Resolve a small configuration without allocating the learned modules."""

    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


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
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 2,
        "inner_updates_per_round": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 4,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_target_tau": 1.0,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "inner_mppi_num_elites": 2,
        "inner_mppi_num_pi_trajs": 0,
        "outer_policy_episode_probability": 0.0,
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


def _parameter_clones(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _assert_parameters_equal(module, expected):
    actual = list(module.parameters())
    assert len(actual) == len(expected)
    for parameter, expected_parameter in zip(actual, expected):
        torch.testing.assert_close(
            parameter.detach(), expected_parameter, rtol=0, atol=0
        )


def _assert_numpy_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def _initialize_adam_state(optimizer):
    """Populate Adam moments so a no-op empty state cannot mask a mutation."""

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _install_synthetic_inner_update(
    monkeypatch,
    agent,
    *,
    actor_delta=2.0,
    critic_delta=-3.0,
    prepared=None,
):
    """Replace expensive rollouts with a known final action-local solution."""

    engine = agent.inner_engine

    def synthetic_act_rl(
        root_z,
        *,
        t0,
        eval_mode,
        start,
        return_behavior_policy=False,
    ):
        del eval_mode, start
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=t0)

        if prepared is not None:
            prepared.append(
                {
                    "actor_id": id(engine.state.actor),
                    "critic_id": id(engine.state.critic),
                    "critic_target_id": id(engine.state.critic_target),
                    "actor_optim_id": id(engine.state.actor_optim),
                    "critic_optim_id": id(engine.state.critic_optim),
                    "replay_id": id(engine.state.replay),
                    "actor": _parameter_clones(engine.state.actor),
                    "critic": _parameter_clones(engine.state.critic),
                    "critic_target": _parameter_clones(
                        engine.state.critic_target
                    ),
                }
            )

        with torch.no_grad():
            for parameter in engine.state.actor.parameters():
                parameter.add_(actor_delta)
            for parameter in engine.state.critic.parameters():
                parameter.add_(critic_delta)

        action = root_z.new_zeros(int(engine.cfg.action_dim))
        metrics = engine._base_metrics(active=True)
        metrics["inner_actor_optimizer_steps"] = 1.0
        metrics["inner_critic_optimizer_steps"] = 1.0
        if return_behavior_policy:
            behavior_policy = {
                "pre_tanh_mean": action.detach().clone(),
                "log_std": action.detach().clone(),
            }
            return action, metrics, [], behavior_policy
        return action, metrics, []

    monkeypatch.setattr(engine, "_act_rl", synthetic_act_rl)


def _outer_parameter_snapshot(agent):
    return {
        "actor": _parameter_clones(agent.model._pi),
        "critic": _parameter_clones(agent.model._Qs),
    }


def _assert_outer_snapshot(agent, snapshot):
    _assert_parameters_equal(agent.model._pi, snapshot["actor"])
    _assert_parameters_equal(agent.model._Qs, snapshot["critic"])


def test_prior_writeback_defaults_and_valid_endpoints():
    default = _build_cfg()
    assert default.inner_actor_writeback_coef == pytest.approx(0.0)
    assert default.inner_critic_writeback_coef == pytest.approx(0.0)

    explicit = _build_cfg(
        inner_actor_writeback_coef=0.25,
        inner_critic_writeback_coef=1,
    )
    assert explicit.inner_actor_writeback_coef == pytest.approx(0.25)
    assert explicit.inner_critic_writeback_coef == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value",
    [None, True, False, "0.5", -0.01, 1.01, float("nan"), float("inf")],
)
@pytest.mark.parametrize(
    "key", ["inner_actor_writeback_coef", "inner_critic_writeback_coef"]
)
def test_prior_writeback_coefficients_are_strict_probabilities(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: value})


def test_zero_writeback_does_not_restrict_existing_inner_modes():
    no_inner = _build_cfg(
        inner_operator="none",
        inner_actor_writeback_coef=0.0,
        inner_critic_writeback_coef=0.0,
    )
    assert no_inner.inner_operator == "none"

    lora = _build_cfg(
        inner_actor_adaptation="lora",
        inner_critic_adaptation="lora",
        inner_actor_writeback_coef=0.0,
        inner_critic_writeback_coef=0.0,
    )
    assert lora.inner_actor_adaptation == "lora"
    assert lora.inner_critic_adaptation == "lora"


@pytest.mark.parametrize(
    ("actor_coef", "critic_coef", "expected_target"),
    [(0.0, 0.0, None), (0.25, 0.0, "actor"), (0.0, 0.25, "critic")],
)
def test_each_zero_coefficient_skips_its_foreach_update(
    monkeypatch,
    actor_coef,
    critic_coef,
    expected_target,
):
    model = _model(
        inner_actor_writeback_coef=actor_coef,
        inner_critic_writeback_coef=critic_coef,
    )
    _install_synthetic_inner_update(monkeypatch, model.agent)
    calls = []
    original = inner_improvement.polyak_update

    def tracked(source, target, tau, *, adapters_only=False):
        calls.append(target)
        return original(source, target, tau, adapters_only=adapters_only)

    monkeypatch.setattr(inner_improvement, "polyak_update", tracked)
    model.agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )

    expected = {
        None: [],
        "actor": [model.agent.model._pi],
        "critic": [model.agent.model._Qs],
    }[expected_target]
    assert calls == expected


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"inner_operator": "td3", "inner_actor_writeback_coef": 0.1},
            "inner_operator",
        ),
        (
            {
                "inner_model_step_budget": 3,
                "inner_actor_updates_per_action": 1,
                "inner_critic_updates_per_action": 1,
                "inner_temperature_updates_per_action": 0,
                "inner_actor_writeback_coef": 0.1,
            },
            "canonical J/N/H/G",
        ),
        (
            {
                "inner_actor_adaptation": "lora",
                "inner_actor_writeback_coef": 0.1,
            },
            "inner_actor_adaptation",
        ),
        (
            {
                "inner_actor_adaptation": "frozen",
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_actor_adaptation",
        ),
        (
            {
                "inner_critic_adaptation": "frozen",
                "inner_actor_writeback_coef": 0.1,
            },
            "inner_critic_adaptation",
        ),
        (
            {
                "inner_critic_adaptation": "lora",
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_critic_adaptation",
        ),
        (
            {
                "inner_actor_scope": "episode",
                "inner_actor_writeback_coef": 0.1,
            },
            "inner_actor_scope",
        ),
        (
            {
                "inner_critic_scope": "episode",
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_critic_scope",
        ),
        (
            {
                "inner_temperature_scope": "episode",
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_temperature_scope",
        ),
        (
            {
                "inner_replay_scope": "run",
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_replay_scope",
        ),
        (
            {
                "inner_actor_optimizer_scope": "episode",
                "inner_actor_writeback_coef": 0.1,
            },
            "inner_actor_optimizer_scope",
        ),
        (
            {
                "inner_critic_optimizer_scope": "episode",
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_critic_optimizer_scope",
        ),
        (
            {
                "inner_temperature_optimizer_scope": "episode",
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_temperature_optimizer_scope",
        ),
        (
            {
                "outer_policy_episode_probability": 0.1,
                "inner_actor_writeback_coef": 0.1,
            },
            "outer_policy_episode_probability",
        ),
        (
            {
                "inner_updates_per_round": 0,
                "inner_actor_writeback_coef": 0.1,
            },
            "inner_actor_writeback_coef",
        ),
        (
            {
                "inner_updates_per_round": 0,
                "inner_critic_writeback_coef": 0.1,
            },
            "inner_critic_writeback_coef",
        ),
        (
            {
                "inner_log_std_mapping": "tdmpc2_tanh",
                "inner_actor_writeback_coef": 0.1,
            },
            "prior write-back",
        ),
        (
            {
                "inner_log_std_min": -19.0,
                "inner_actor_writeback_coef": 0.1,
            },
            "prior write-back",
        ),
    ],
)
def test_active_prior_writeback_rejects_incompatible_configs(overrides, message):
    with pytest.raises(ValueError, match=message):
        _build_cfg(**overrides)


def test_any_active_writeback_requires_actor_distribution_equivalence():
    for mismatch in (
        {"inner_log_std_mapping": "tdmpc2_tanh"},
        {"inner_log_std_min": -7.0},
        {"inner_log_std_max": 0.5},
    ):
        with pytest.raises(ValueError, match="prior write-back"):
            _build_cfg(
                inner_actor_writeback_coef=0.0,
                inner_critic_writeback_coef=0.4,
                **mismatch,
            )


@pytest.mark.parametrize(
    ("representation", "num_q", "actor_coef", "critic_coef"),
    [
        ("scalar", 2, 0.25, 0.0),
        ("scalar", 2, 0.0, 0.4),
        ("scalar", 2, 0.25, 0.75),
        ("scalar", 2, 1.0, 0.0),
        ("scalar", 2, 0.0, 1.0),
        ("scalar", 2, 1.0, 1.0),
        ("distributional", 3, 0.25, 0.75),
    ],
)
def test_authorized_prior_writeback_exactly_lerps_online_heads_only(
    monkeypatch,
    representation,
    num_q,
    actor_coef,
    critic_coef,
):
    model = _model(
        q_representation=representation,
        num_q=num_q,
        inner_actor_writeback_coef=actor_coef,
        inner_critic_writeback_coef=critic_coef,
    )
    agent = model.agent
    actor_delta = 2.0
    critic_delta = -3.0
    _install_synthetic_inner_update(
        monkeypatch,
        agent,
        actor_delta=actor_delta,
        critic_delta=critic_delta,
    )

    for optimizer in (agent.optim, agent.pi_optim, agent.ent_coef_optim):
        _initialize_adam_state(optimizer)
    actor_before = _parameter_clones(agent.model._pi)
    critic_before = _parameter_clones(agent.model._Qs)
    target_before = _clone_tree(agent.model._target_Qs.state_dict())
    world_before = {
        name: _clone_tree(module.state_dict())
        for name, module in (
            ("encoder", agent.model._encoder),
            ("dynamics", agent.model._dynamics),
            ("reward", agent.model._reward),
            ("termination", agent.model._termination),
        )
        if module is not None
    }
    policy_buffers_before = {
        name: buffer.detach().clone()
        for name, buffer in agent.model.named_buffers()
        if not name.startswith("_Qs.") and not name.startswith("_target_Qs.")
    }
    optim_before = _clone_tree(agent.optim.state_dict())
    pi_optim_before = _clone_tree(agent.pi_optim.state_dict())
    entropy_optim_before = _clone_tree(agent.ent_coef_optim.state_dict())
    ids_before = {
        "actor": tuple(id(parameter) for parameter in agent.model._pi.parameters()),
        "critic": tuple(id(parameter) for parameter in agent.model._Qs.parameters()),
        "target": tuple(
            id(parameter) for parameter in agent.model._target_Qs.parameters()
        ),
    }
    counters_before = (agent.num_updates, agent.outer_version)
    alpha_before = agent.alpha.detach().clone()
    replay_before = (
        model.buffer.num_eps,
        model.buffer.num_transitions,
        model.buffer.total_transitions,
        model.buffer.size,
    )
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()

    action = agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        eval_mode=False,
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )

    assert action.shape == (model.cfg.action_dim,)
    for actual, before in zip(agent.model._pi.parameters(), actor_before):
        expected = torch.lerp(before, before + actor_delta, actor_coef)
        torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
    for actual, before in zip(agent.model._Qs.parameters(), critic_before):
        expected = torch.lerp(before, before + critic_delta, critic_coef)
        torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)

    _assert_tree_equal(agent.model._target_Qs.state_dict(), target_before)
    for name, module in (
        ("encoder", agent.model._encoder),
        ("dynamics", agent.model._dynamics),
        ("reward", agent.model._reward),
        ("termination", agent.model._termination),
    ):
        if module is not None:
            _assert_tree_equal(module.state_dict(), world_before[name])
    assert {
        name: buffer.detach().clone()
        for name, buffer in agent.model.named_buffers()
        if not name.startswith("_Qs.") and not name.startswith("_target_Qs.")
    }.keys() == policy_buffers_before.keys()
    for name, buffer in agent.model.named_buffers():
        if name in policy_buffers_before:
            torch.testing.assert_close(
                buffer, policy_buffers_before[name], rtol=0, atol=0
            )
    _assert_tree_equal(agent.optim.state_dict(), optim_before)
    _assert_tree_equal(agent.pi_optim.state_dict(), pi_optim_before)
    _assert_tree_equal(agent.ent_coef_optim.state_dict(), entropy_optim_before)
    assert {
        "actor": tuple(id(parameter) for parameter in agent.model._pi.parameters()),
        "critic": tuple(id(parameter) for parameter in agent.model._Qs.parameters()),
        "target": tuple(
            id(parameter) for parameter in agent.model._target_Qs.parameters()
        ),
    } == ids_before
    assert (agent.num_updates, agent.outer_version) == counters_before
    torch.testing.assert_close(agent.alpha, alpha_before, rtol=0, atol=0)
    assert (
        model.buffer.num_eps,
        model.buffer.num_transitions,
        model.buffer.total_transitions,
        model.buffer.size,
    ) == replay_before
    assert random.getstate() == python_rng_before
    _assert_numpy_rng_equal(np.random.get_state(), numpy_rng_before)
    torch.testing.assert_close(
        torch.random.get_rng_state(), torch_rng_before, rtol=0, atol=0
    )

    metrics = agent.last_inner_metrics
    assert metrics["inner_actor_writeback_coef"] == pytest.approx(actor_coef)
    assert metrics["inner_critic_writeback_coef"] == pytest.approx(critic_coef)
    assert metrics["inner_actor_writeback_applied"] == float(actor_coef > 0.0)
    assert metrics["inner_critic_writeback_applied"] == float(critic_coef > 0.0)


def test_default_and_direct_action_paths_do_not_write_back(monkeypatch):
    default = _model()
    _install_synthetic_inner_update(monkeypatch, default.agent)
    default_before = _outer_parameter_snapshot(default.agent)

    default.agent.act(
        torch.zeros(default.cfg.obs_shape["state"]),
        apply_inner_writeback=True,
        collect_diagnostics=False,
    )

    _assert_outer_snapshot(default.agent, default_before)
    assert "inner_actor_writeback_applied" not in default.agent.last_inner_metrics
    assert "inner_critic_writeback_applied" not in default.agent.last_inner_metrics

    active = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    _install_synthetic_inner_update(monkeypatch, active.agent)
    active_before = _outer_parameter_snapshot(active.agent)
    active_checkpoint_before = _clone_tree(active.agent.checkpoint_state())

    active.agent.act(
        torch.zeros(active.cfg.obs_shape["state"]),
        eval_mode=False,
        collect_diagnostics=False,
    )
    _assert_outer_snapshot(active.agent, active_before)
    _assert_tree_equal(active.agent.checkpoint_state(), active_checkpoint_before)
    assert active.agent.last_inner_metrics["inner_actor_writeback_applied"] == 0.0
    assert active.agent.last_inner_metrics["inner_critic_writeback_applied"] == 0.0

    active.agent.act(
        torch.zeros(active.cfg.obs_shape["state"]),
        eval_mode=True,
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )
    _assert_outer_snapshot(active.agent, active_before)
    _assert_tree_equal(active.agent.checkpoint_state(), active_checkpoint_before)
    assert active.agent.last_inner_metrics["inner_actor_writeback_applied"] == 0.0
    assert active.agent.last_inner_metrics["inner_critic_writeback_applied"] == 0.0

    for deterministic in (False, True):
        active.predict(
            np.zeros(active.cfg.obs_shape["state"], dtype=np.float32),
            deterministic=deterministic,
            collect_diagnostics=False,
        )
        _assert_outer_snapshot(active.agent, active_before)
        _assert_tree_equal(active.agent.checkpoint_state(), active_checkpoint_before)
        assert active.agent.last_inner_metrics["inner_actor_writeback_applied"] == 0.0
        assert active.agent.last_inner_metrics["inner_critic_writeback_applied"] == 0.0


def test_full_online_evaluation_cannot_write_back(monkeypatch):
    model = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    _install_synthetic_inner_update(monkeypatch, model.agent)
    outer_before = _clone_tree(model.agent.model.state_dict())
    replay_before = (
        model.buffer.num_eps,
        model.buffer.num_transitions,
        model.buffer.total_transitions,
        model.buffer.size,
    )
    model._eval_episodes = 1
    model._record_evaluation = lambda step, reward: None
    obs, _ = model.env.reset(seed=13)

    model._evaluate_policy(0, initial_obs=obs)

    _assert_tree_equal(model.agent.model.state_dict(), outer_before)
    assert (
        model.buffer.num_eps,
        model.buffer.num_transitions,
        model.buffer.total_transitions,
        model.buffer.size,
    ) == replay_before


def test_seed_action_bypasses_inner_writeback(monkeypatch):
    model = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
        seed_steps=4,
    )
    outer_before = _clone_tree(model.agent.model.state_dict())

    def unexpected_planned_action(*args, **kwargs):
        raise AssertionError("seed collection must bypass the planned-action path")

    monkeypatch.setattr(model, "_act_agent", unexpected_planned_action)
    obs, _ = model.env.reset(seed=13)

    model._run_training_episode(obs, total_timesteps=1, eval_pending=False)

    _assert_tree_equal(model.agent.model.state_dict(), outer_before)


def test_wrapper_authorizes_training_but_not_evaluation_writeback(monkeypatch):
    model = _model(
        inner_actor_writeback_coef=0.5,
        inner_critic_writeback_coef=0.25,
    )
    agent = model.agent
    actor_delta = 2.0
    critic_delta = -3.0
    _install_synthetic_inner_update(
        monkeypatch,
        agent,
        actor_delta=actor_delta,
        critic_delta=critic_delta,
    )
    obs = torch.zeros(model.cfg.obs_shape["state"])
    actor_before = _parameter_clones(agent.model._pi)
    critic_before = _parameter_clones(agent.model._Qs)

    model._act_agent(obs, t0=True, eval_mode=True)

    _assert_parameters_equal(agent.model._pi, actor_before)
    _assert_parameters_equal(agent.model._Qs, critic_before)
    assert agent.last_inner_metrics["inner_actor_writeback_applied"] == 0.0
    assert agent.last_inner_metrics["inner_critic_writeback_applied"] == 0.0

    model._act_agent(obs, t0=False, eval_mode=False)

    for actual, before in zip(agent.model._pi.parameters(), actor_before):
        torch.testing.assert_close(
            actual.detach(), torch.lerp(before, before + actor_delta, 0.5)
        )
    for actual, before in zip(agent.model._Qs.parameters(), critic_before):
        torch.testing.assert_close(
            actual.detach(), torch.lerp(before, before + critic_delta, 0.25)
        )
    assert agent.last_inner_metrics["inner_actor_writeback_applied"] == 1.0
    assert agent.last_inner_metrics["inner_critic_writeback_applied"] == 1.0

    model._reset_wandb_window()
    model._record_action_metrics(planned=True, action_seconds=0.0)
    payload = model._wandb_train_window.pop()
    assert payload["train/inner_actor_writeback_applied"] == pytest.approx(1.0)
    assert payload["train/inner_critic_writeback_applied"] == pytest.approx(1.0)
    assert payload["train/inner_actor_writeback_coef"] == pytest.approx(0.5)
    assert payload["train/inner_critic_writeback_coef"] == pytest.approx(0.25)


def test_action_pool_is_reused_and_reinitialized_from_written_back_outer(
    monkeypatch,
):
    model = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    agent = model.agent
    engine = agent.inner_engine
    prepared = []
    actor_delta = 0.75
    critic_delta = -0.5
    _install_synthetic_inner_update(
        monkeypatch,
        agent,
        actor_delta=actor_delta,
        critic_delta=critic_delta,
        prepared=prepared,
    )
    obs = torch.zeros(model.cfg.obs_shape["state"])
    actor_initial = _parameter_clones(agent.model._pi)
    critic_initial = _parameter_clones(agent.model._Qs)

    agent.act(
        obs,
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )
    actor_after_first = _parameter_clones(agent.model._pi)
    critic_after_first = _parameter_clones(agent.model._Qs)

    assert engine.state.actor is None
    assert engine.state.critic is None
    assert engine._action_pool.actor is not None
    assert engine._action_pool.critic is not None
    assert id(engine._action_pool.actor) == prepared[0]["actor_id"]
    assert id(engine._action_pool.critic) == prepared[0]["critic_id"]
    assert id(engine._action_pool.critic_target) == prepared[0]["critic_target_id"]
    assert id(engine._action_pool.actor_optim) == prepared[0]["actor_optim_id"]
    assert id(engine._action_pool.critic_optim) == prepared[0]["critic_optim_id"]
    assert id(engine._action_pool.replay) == prepared[0]["replay_id"]

    agent.act(
        obs,
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )

    assert prepared[1]["actor_id"] == prepared[0]["actor_id"]
    assert prepared[1]["critic_id"] == prepared[0]["critic_id"]
    assert prepared[1]["critic_target_id"] == prepared[0]["critic_target_id"]
    assert prepared[1]["actor_optim_id"] == prepared[0]["actor_optim_id"]
    assert prepared[1]["critic_optim_id"] == prepared[0]["critic_optim_id"]
    assert prepared[1]["replay_id"] == prepared[0]["replay_id"]
    for prepared_parameter, outer_parameter in zip(
        prepared[1]["actor"], actor_after_first
    ):
        torch.testing.assert_close(
            prepared_parameter, outer_parameter, rtol=0, atol=0
        )
    for prepared_parameter, outer_parameter in zip(
        prepared[1]["critic"], critic_after_first
    ):
        torch.testing.assert_close(
            prepared_parameter, outer_parameter, rtol=0, atol=0
        )
    for target_parameter, critic_parameter in zip(
        prepared[1]["critic_target"], prepared[1]["critic"]
    ):
        torch.testing.assert_close(
            target_parameter, critic_parameter, rtol=0, atol=0
        )

    for actual, initial in zip(agent.model._pi.parameters(), actor_initial):
        torch.testing.assert_close(
            actual.detach(), initial + 2.0 * actor_delta
        )
    for actual, initial in zip(agent.model._Qs.parameters(), critic_initial):
        torch.testing.assert_close(
            actual.detach(), initial + 2.0 * critic_delta
        )
    assert engine.state.actor is None
    assert engine.state.critic is None
    assert id(engine._action_pool.actor) == prepared[0]["actor_id"]
    assert id(engine._action_pool.critic) == prepared[0]["critic_id"]


def test_first_action_diagnostics_and_behavior_capture_are_pre_writeback():
    overrides = {
        "inner_actor_writeback_coef": 0.4,
        "inner_critic_writeback_coef": 0.6,
    }
    control = _model(**overrides)
    written = _model(**overrides)
    written.agent.model.load_state_dict(control.agent.model.state_dict())
    obs = torch.zeros(control.cfg.obs_shape["state"])

    control_action, control_behavior = control.agent.act(
        obs,
        eval_mode=False,
        collect_diagnostics=True,
        return_behavior_policy=True,
        apply_inner_writeback=False,
    )
    written_action, written_behavior = written.agent.act(
        obs,
        eval_mode=False,
        collect_diagnostics=True,
        return_behavior_policy=True,
        apply_inner_writeback=True,
    )

    torch.testing.assert_close(written_action, control_action, rtol=0, atol=0)
    for key in ("pre_tanh_mean", "log_std"):
        torch.testing.assert_close(
            written_behavior[key], control_behavior[key], rtol=0, atol=0
        )
    for key in (
        "inner_final_outer_policy_kl",
        "inner_policy_mean_delta_l2",
        "inner_outer_q_gain",
    ):
        torch.testing.assert_close(
            torch.as_tensor(written.agent.last_inner_metrics[key]),
            torch.as_tensor(control.agent.last_inner_metrics[key]),
            rtol=0,
            atol=0,
        )
    _assert_tree_equal(
        written.agent.inner_engine.rng.training_state_dict(),
        control.agent.inner_engine.rng.training_state_dict(),
    )
    assert written.agent.last_inner_metrics["inner_actor_writeback_applied"] == 1.0
    assert written.agent.last_inner_metrics["inner_critic_writeback_applied"] == 1.0
    assert control.agent.last_inner_metrics["inner_actor_writeback_applied"] == 0.0
    assert control.agent.last_inner_metrics["inner_critic_writeback_applied"] == 0.0


def test_agent_reset_does_not_undo_written_outer_weights(monkeypatch):
    model = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    _install_synthetic_inner_update(
        monkeypatch,
        model.agent,
        actor_delta=0.25,
        critic_delta=-0.5,
    )
    obs = torch.zeros(model.cfg.obs_shape["state"])

    model.agent.act(
        obs,
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )
    written = _outer_parameter_snapshot(model.agent)

    model.agent.reset()

    _assert_outer_snapshot(model.agent, written)
    assert model.agent.last_inner_metrics == {}
    assert model.agent.last_inner_rollout_lengths == []


def test_outer_target_changes_only_at_ordinary_update_cadence(monkeypatch):
    model = _model(
        inner_actor_writeback_coef=0.5,
        inner_critic_writeback_coef=1.0,
    )
    agent = model.agent
    _install_synthetic_inner_update(
        monkeypatch,
        agent,
        actor_delta=0.01,
        critic_delta=-0.02,
    )
    target_before_action = _parameter_clones(agent.model._target_Qs)
    target_calls = []
    original_target_update = agent.model.soft_update_target_Q

    def tracked_target_update(tau=None):
        target_before = _parameter_clones(agent.model._target_Qs)
        online_before = _parameter_clones(agent.model._Qs)
        original_target_update(tau=tau)
        effective_tau = float(agent.cfg.tau if tau is None else tau)
        target_calls.append(
            [
                torch.lerp(target, online, effective_tau)
                for target, online in zip(target_before, online_before)
            ]
        )

    monkeypatch.setattr(agent.model, "soft_update_target_Q", tracked_target_update)

    agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )

    assert target_calls == []
    _assert_parameters_equal(agent.model._target_Qs, target_before_action)

    horizon = int(model.cfg.train_unroll_horizon)
    batch_size = int(model.cfg.batch_size)
    obs = torch.zeros(horizon + 1, batch_size, model.cfg.obs_shape["state"][0])
    action = torch.zeros(horizon, batch_size, model.cfg.action_dim)
    reward = torch.zeros(horizon, batch_size, 1)
    terminated = torch.zeros_like(reward)
    agent._update(obs, action, reward, terminated)

    assert len(target_calls) == 1
    _assert_parameters_equal(agent.model._target_Qs, target_calls[0])
    assert agent.num_updates == 1
    assert agent.outer_version == 1


def test_written_weights_roundtrip_through_portable_checkpoint_and_raw_model(
    monkeypatch,
):
    model = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    _install_synthetic_inner_update(
        monkeypatch,
        model.agent,
        actor_delta=0.25,
        critic_delta=-0.5,
    )
    model.agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )
    checkpoint = _clone_tree(model.agent.checkpoint_state())

    feature_off = _model()
    feature_off_checkpoint = feature_off.agent.checkpoint_state()
    assert checkpoint.keys() == feature_off_checkpoint.keys()
    assert checkpoint["model"].keys() == feature_off_checkpoint["model"].keys()
    assert checkpoint["checkpoint_version"] == feature_off_checkpoint[
        "checkpoint_version"
    ]

    structured_receiver = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    structured_receiver.agent.load(checkpoint)
    _assert_tree_equal(
        structured_receiver.agent.model.state_dict(), checkpoint["model"]
    )

    raw_receiver = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    raw_receiver.agent.load(checkpoint["model"])
    _assert_tree_equal(raw_receiver.agent.model.state_dict(), checkpoint["model"])


def test_written_weights_roundtrip_through_exact_training_state(monkeypatch):
    source = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    _install_synthetic_inner_update(
        monkeypatch,
        source.agent,
        actor_delta=0.25,
        critic_delta=-0.5,
    )
    source.agent.act(
        torch.zeros(source.cfg.obs_shape["state"]),
        collect_diagnostics=False,
        apply_inner_writeback=True,
    )
    source.agent.prepare_training_resume_boundary()
    saved = _clone_tree(source.agent.training_state_dict())

    feature_off = _model()
    feature_off.agent.prepare_training_resume_boundary()
    feature_off_state = feature_off.agent.training_state_dict()
    assert saved.keys() == feature_off_state.keys()
    assert saved["outer"].keys() == feature_off_state["outer"].keys()
    assert saved["inner"].keys() == feature_off_state["inner"].keys()
    assert saved["version"] == feature_off_state["version"]

    restored = _model(
        inner_actor_writeback_coef=1.0,
        inner_critic_writeback_coef=1.0,
    )
    restored.agent.load_training_state_dict(saved)

    _assert_tree_equal(restored.agent.training_state_dict(), saved)


def test_writeback_coefficients_change_lineage_fingerprint(monkeypatch):
    monkeypatch.setattr(
        resume_identity,
        "source_identity",
        lambda _repo_root: {"commit": "test", "dirty": False},
    )
    monkeypatch.setattr(
        resume_identity,
        "dependency_identity",
        lambda: {"python": "test"},
    )
    base = {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "seed": 7,
        "alg_params": {
            "inner_actor_writeback_coef": 0.0,
            "inner_critic_writeback_coef": 0.0,
        },
    }

    def fingerprint(params):
        return resume_identity.lineage_identity(
            trial_run_params=params,
            experiment_params={"exp_name": "writeback-test"},
            repo_root=".",
        )["fingerprint"]

    actor_changed = deepcopy(base)
    actor_changed["alg_params"]["inner_actor_writeback_coef"] = 0.25
    critic_changed = deepcopy(base)
    critic_changed["alg_params"]["inner_critic_writeback_coef"] = 0.25

    assert fingerprint(base) != fingerprint(actor_changed)
    assert fingerprint(base) != fingerprint(critic_changed)
    assert fingerprint(actor_changed) != fingerprint(critic_changed)

    omitted = deepcopy(base)
    omitted["alg_params"] = {}
    integer_zero = deepcopy(base)
    integer_zero["alg_params"] = {
        "inner_actor_writeback_coef": 0,
        "inner_critic_writeback_coef": 0,
    }
    signed_zero = deepcopy(base)
    signed_zero["alg_params"] = {
        "inner_actor_writeback_coef": -0.0,
        "inner_critic_writeback_coef": -0.0,
    }
    assert fingerprint(omitted) == fingerprint(base)
    assert fingerprint(integer_zero) == fingerprint(base)
    assert fingerprint(signed_zero) == fingerprint(base)
