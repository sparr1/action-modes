from copy import deepcopy
from contextlib import contextmanager

import gymnasium as gym
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core import inner_improvement


_EXPENSIVE_DIAGNOSTIC_KEYS = {
    "inner_replay_unique_fraction",
    "inner_policy_mean_delta_l2",
    "inner_fixed_target_q_action_gain",
    "inner_outer_q_gain",
    "inner_fixed_target_q_outer",
    "inner_fixed_target_q_improved",
    "inner_fixed_target_q_abs_mean",
    "inner_fixed_evaluator_alpha",
    "inner_predicted_j_outer",
    "inner_predicted_j_improved",
    "inner_predicted_j_gain",
    "inner_predicted_soft_j_outer",
    "inner_predicted_soft_j_improved",
    "inner_predicted_soft_j_gain",
    "inner_fixed_alpha_soft_j_outer",
    "inner_fixed_alpha_soft_j_improved",
    "inner_fixed_alpha_soft_j_gain",
    "inner_diagnostic_model_steps",
    "inner_distributional_q_entropy",
    "inner_distributional_q_edge_mass",
    "inner_diagnostics_step",
}


def _params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 5,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 2,
        "outer_planning_horizon": 2,
        "buffer_size": 32,
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
        "q_num_bins": 5,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_model_step_budget": 4,
        "inner_rounds": 1,
        "inner_rollout_horizon": 2,
        "inner_critic_updates_per_action": 1,
        "inner_actor_updates_per_action": 1,
        "inner_temperature_updates_per_action": 0,
        "inner_batch_size": 4,
        "inner_replay_capacity": 8,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_target_tau": 1.0,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 2,
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
        {"seed": 17, "device": "cpu", "env": "test", "total_steps": 10},
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
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_equal(actual_item, expected_item)
        return
    assert actual == expected


def _pool_snapshot(agent):
    pool = agent.inner_engine._action_pool
    return {
        "actor": _clone_tree(pool.actor.state_dict()),
        "critic": _clone_tree(pool.critic.state_dict()),
        "critic_target": _clone_tree(pool.critic_target.state_dict()),
        "actor_optim": _clone_tree(pool.actor_optim.state_dict()),
        "critic_optim": _clone_tree(pool.critic_optim.state_dict()),
        "replay": _clone_tree(pool.replay.state_dict()),
    }


def _optimizer_tensor_pointers(optimizer):
    return tuple(
        value.data_ptr()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def _optimizer_steps(optimizer):
    return [float(state["step"]) for state in optimizer.state.values()]


def _optimizer_parameter_ids(optimizer):
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_collect_diagnostics_false_omits_expensive_metrics_and_marks_unsampled():
    model = _model(q_representation="distributional", num_q=3)

    model.agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        eval_mode=False,
        collect_diagnostics=False,
    )
    metrics = model.agent.last_inner_metrics

    assert metrics["inner_diagnostics_sampled"] == 0.0
    assert metrics["inner_diagnostics_sample_count"] == 0.0
    assert _EXPENSIVE_DIAGNOSTIC_KEYS.isdisjoint(metrics)
    assert "inner_diagnostic_seconds" not in metrics
    assert metrics["inner_model_steps"] == 4.0
    assert metrics["inner_total_model_steps"] == 4.0


def test_collect_diagnostics_true_includes_values_and_sampling_metadata():
    model = _model(q_representation="distributional", num_q=3)

    model.agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        eval_mode=False,
        collect_diagnostics=True,
    )
    metrics = model.agent.last_inner_metrics

    assert metrics["inner_diagnostics_sampled"] == 1.0
    assert metrics["inner_diagnostics_sample_count"] == 1.0
    assert metrics["inner_diagnostics_step"] == 1.0
    assert _EXPENSIVE_DIAGNOSTIC_KEYS.issubset(metrics)
    assert "inner_diagnostic_seconds" in metrics
    assert metrics["inner_diagnostic_model_steps"] == 8.0
    assert metrics["inner_total_model_steps"] == 12.0


def test_diagnostic_toggle_preserves_action_updates_and_training_rng_streams():
    without_diagnostics = _model(q_representation="distributional", num_q=3)
    with_diagnostics = _model(q_representation="distributional", num_q=3)
    observation = torch.zeros(without_diagnostics.cfg.obs_shape["state"])
    global_rng_before = torch.random.get_rng_state().clone()

    first_action = without_diagnostics.agent.act(
        observation,
        eval_mode=False,
        collect_diagnostics=False,
    )
    torch.testing.assert_close(
        torch.random.get_rng_state(), global_rng_before, rtol=0, atol=0
    )
    second_action = with_diagnostics.agent.act(
        observation,
        eval_mode=False,
        collect_diagnostics=True,
    )

    torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.random.get_rng_state(), global_rng_before, rtol=0, atol=0
    )
    _assert_tree_equal(
        _pool_snapshot(with_diagnostics.agent),
        _pool_snapshot(without_diagnostics.agent),
    )

    first_rng = without_diagnostics.agent.inner_engine.rng
    second_rng = with_diagnostics.agent.inner_engine.rng
    for stream in first_rng.STREAMS:
        if stream == "diagnostics":
            continue
        torch.testing.assert_close(
            second_rng.generators[stream].get_state(),
            first_rng.generators[stream].get_state(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            second_rng.phase_generators[stream].get_state(),
            first_rng.phase_generators[stream].get_state(),
            rtol=0,
            atol=0,
        )


def test_inner_action_saves_default_rng_state_only_once(monkeypatch):
    model = _model(inner_rounds=2, inner_model_step_budget=8)
    original = torch.random.fork_rng
    calls = 0

    @contextmanager
    def counted_fork_rng(*args, **kwargs):
        nonlocal calls
        calls += 1
        with original(*args, **kwargs):
            yield

    monkeypatch.setattr(torch.random, "fork_rng", counted_fork_rng)
    model.agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        collect_diagnostics=True,
    )

    assert calls == 1


def test_action_scoped_workspace_reuses_allocations_but_resets_logical_state():
    model = _model()
    agent = model.agent
    observation = torch.zeros(model.cfg.obs_shape["state"])

    agent.act(observation, collect_diagnostics=False)
    pool = agent.inner_engine._action_pool
    object_ids = {
        name: id(getattr(pool, name))
        for name in (
            "actor",
            "critic",
            "critic_target",
            "actor_optim",
            "critic_optim",
            "replay",
        )
    }
    replay_storage_ptr = pool.replay._storage.data_ptr()
    actor_state_ptrs = _optimizer_tensor_pointers(pool.actor_optim)
    critic_state_ptrs = _optimizer_tensor_pointers(pool.critic_optim)
    actor_params_id = id(pool.actor_params)
    critic_params_id = id(pool.critic_params)
    assert pool.replay.size == 4
    assert pool.replay.next_sample_id == 4
    assert set(_optimizer_steps(pool.actor_optim)) == {1.0}
    assert set(_optimizer_steps(pool.critic_optim)) == {1.0}

    agent.act(observation, collect_diagnostics=False)
    pool = agent.inner_engine._action_pool

    assert {
        name: id(getattr(pool, name)) for name in object_ids
    } == object_ids
    assert pool.replay._storage.data_ptr() == replay_storage_ptr
    assert _optimizer_tensor_pointers(pool.actor_optim) == actor_state_ptrs
    assert _optimizer_tensor_pointers(pool.critic_optim) == critic_state_ptrs
    assert id(pool.actor_params) == actor_params_id
    assert id(pool.critic_params) == critic_params_id
    assert pool.replay.size == 4
    assert pool.replay.next_sample_id == 4
    assert set(_optimizer_steps(pool.actor_optim)) == {1.0}
    assert set(_optimizer_steps(pool.critic_optim)) == {1.0}


def test_action_optimizer_pool_never_outlives_episode_parameters():
    model = _model(
        inner_actor_scope="episode",
        inner_critic_scope="episode",
        inner_actor_optimizer_scope="action",
        inner_critic_optimizer_scope="action",
    )
    agent = model.agent
    observation = torch.zeros(model.cfg.obs_shape["state"])

    agent.act(observation, collect_diagnostics=False)
    engine = agent.inner_engine
    old_actor = engine.state.actor
    old_critic = engine.state.critic
    old_actor_optim = engine._action_pool.actor_optim
    old_critic_optim = engine._action_pool.critic_optim
    old_actor_state = _clone_tree(old_actor.state_dict())
    old_critic_state = _clone_tree(old_critic.state_dict())

    agent.reset()
    agent.act(observation, collect_diagnostics=False)

    assert engine.state.actor is not old_actor
    assert engine.state.critic is not old_critic
    assert engine._action_pool.actor_optim is not old_actor_optim
    assert engine._action_pool.critic_optim is not old_critic_optim
    assert _optimizer_parameter_ids(engine._action_pool.actor_optim) == {
        id(parameter) for parameter in engine.state.actor_params
    }
    assert _optimizer_parameter_ids(engine._action_pool.critic_optim) == {
        id(parameter) for parameter in engine.state.critic_params
    }
    _assert_tree_equal(old_actor.state_dict(), old_actor_state)
    _assert_tree_equal(old_critic.state_dict(), old_critic_state)


def test_unsampled_mppi_counts_only_work_that_executed():
    without_diagnostics = _model(
        inner_operator="mppi",
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
        inner_temperature_updates_per_action=0,
    )
    with_diagnostics = _model(
        inner_operator="mppi",
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
        inner_temperature_updates_per_action=0,
    )
    observation = torch.zeros(without_diagnostics.cfg.obs_shape["state"])

    without_diagnostics.agent.act(observation, collect_diagnostics=False)
    with_diagnostics.agent.act(observation, collect_diagnostics=True)
    unsampled = without_diagnostics.agent.last_inner_metrics
    sampled = with_diagnostics.agent.last_inner_metrics

    assert sampled["inner_policy_evaluations"] == unsampled["inner_policy_evaluations"] + 1
    assert sampled["inner_q_evaluations"] == unsampled["inner_q_evaluations"] + 2


def test_direct_predict_keeps_full_diagnostics_unless_explicitly_disabled():
    model = _model(inner_diagnostics_every=1000)
    observation = model.env.observation_space.sample()

    model.predict(observation, deterministic=True)
    assert model.agent.last_inner_metrics["inner_diagnostics_sampled"] == 1.0

    model.predict(
        observation,
        deterministic=True,
        episode_start=False,
        collect_diagnostics=False,
    )
    metrics = model.agent.last_inner_metrics
    assert metrics["inner_diagnostics_sampled"] == 0.0
    assert _EXPENSIVE_DIAGNOSTIC_KEYS.isdisjoint(metrics)


def test_trainer_diagnostic_cadence_is_keyed_to_resulting_environment_step(
    monkeypatch,
):
    model = _model(inner_diagnostics_every=1000)
    sampled = []
    recorded_steps = []

    def fake_act(obs, **kwargs):
        del obs
        collect = kwargs["collect_diagnostics"]
        sampled.append(collect)
        model.agent.last_inner_metrics = {
            "inner_diagnostics_sampled": float(collect),
        }
        if collect:
            # The trainer must replace this engine-local marker.
            model.agent.last_inner_metrics["inner_diagnostics_step"] = -1.0
        return torch.zeros(model.cfg.action_dim)

    monkeypatch.setattr(model.agent, "act", fake_act)
    observation = torch.zeros(model.cfg.obs_shape["state"])
    for current_step in (0, 998, 999, 1000):
        model._global_step = current_step
        model._act_agent(observation, t0=False, eval_mode=False)
        recorded_steps.append(
            model.agent.last_inner_metrics.get("inner_diagnostics_step")
        )

    assert sampled == [False, False, True, False]
    assert recorded_steps == [None, None, 1000.0, None]


def test_diagnostic_sampling_step_uses_last_event_semantics_in_wandb_window():
    model = _model()
    model._reset_wandb_window()

    for sampling_step in (1000.0, 2000.0):
        model.agent.last_inner_rollout_lengths = []
        model.agent.last_inner_metrics = {
            "inner_active": 1.0,
            "inner_rollouts": 0.0,
            "inner_steps": 0.0,
            "inner_updates": 0.0,
            "inner_diagnostics_sampled": 1.0,
            "inner_diagnostics_sample_count": 1.0,
            "inner_diagnostics_step": sampling_step,
        }
        model._record_action_metrics(planned=True, action_seconds=0.0)

    payload = model._wandb_train_window.pop()
    assert payload["train/inner_diagnostics_step"] == 2000.0
    assert "train/inner_diagnostics_step_mean" not in payload


def test_zero_rollout_actions_do_not_create_termination_population_samples():
    model = _model()
    model._reset_wandb_window()
    model.agent.last_inner_rollout_lengths = []
    model.agent.last_inner_metrics = {
        "inner_active": 0.0,
        "inner_rollouts": 0.0,
        "inner_steps": 0.0,
        "inner_updates": 0.0,
        "inner_termination_rate": 0.0,
    }

    model._record_action_metrics(planned=True, action_seconds=0.0)

    payload = model._wandb_train_window.pop()
    assert "train/inner_termination_rate" not in payload
    assert "train/inner_termination_rate_count" not in payload

    model._reset_wandb_window()
    model.agent.last_inner_metrics = {
        "inner_active": 1.0,
        "inner_rollouts": 0.0,
        "inner_steps": 0.0,
        "inner_updates": 0.0,
        "inner_termination_rate": 0.0,
    }
    model._record_action_metrics(planned=True, action_seconds=0.0)
    model.agent.last_inner_metrics = {
        "inner_active": 1.0,
        "inner_rollouts": 2.0,
        "inner_steps": 2.0,
        "inner_updates": 0.0,
        "inner_termination_rate": 1.0,
        "inner_termination_rate_std": 0.0,
        "inner_termination_rate_min": 1.0,
        "inner_termination_rate_max": 1.0,
    }
    model._record_action_metrics(planned=True, action_seconds=0.0)

    payload = model._wandb_train_window.pop()
    assert payload["train/inner_termination_rate"] == 1.0
    assert payload["train/inner_termination_rate_mean"] == 1.0
    assert payload["train/inner_termination_rate_count"] == 2.0


def test_no_q_diagnostic_is_fabricated_when_no_q_evaluation_ran():
    model = _model(
        inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0,
        inner_temperature_updates_per_action=0,
    )
    model.agent.act(
        torch.zeros(model.cfg.obs_shape["state"]),
        collect_diagnostics=False,
    )
    metrics = model.agent.last_inner_metrics
    assert "inner_q_abs_mean" not in metrics
    assert "inner_alpha_to_abs_q" not in metrics


def test_dense_non_episodic_rollout_counts_and_horizon_major_replay(monkeypatch):
    model = _model()
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)

    call_index = 0

    def fake_policy_action(z, policy, **kwargs):
        nonlocal call_index
        del policy, kwargs
        action = (
            torch.arange(z.shape[0], device=z.device, dtype=z.dtype)
            + 10 * call_index
        ).unsqueeze(-1)
        call_index += 1
        return action, None

    def fake_reward_from_joint(joint):
        return joint[..., -model.cfg.action_dim :] + 100.0

    def fake_next_from_joint(joint):
        z = joint[..., : model.cfg.latent_dim]
        action = joint[..., -model.cfg.action_dim :]
        return z + action

    def forbidden_termination(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dense non-episodic rollout queried termination")

    monkeypatch.setattr(engine, "_policy_action", fake_policy_action)
    monkeypatch.setattr(engine.model, "reward_from_joint", fake_reward_from_joint)
    monkeypatch.setattr(engine.model, "next_from_joint", fake_next_from_joint)
    monkeypatch.setattr(engine.model, "termination", forbidden_termination)
    monkeypatch.setattr(
        inner_improvement.td_math, "two_hot_inv", lambda prediction, cfg: prediction
    )

    root_z = torch.zeros(1, model.cfg.latent_dim)
    rollout = engine._collect_round(root_z)
    replay = engine.state.replay
    count = model.cfg.inner_rollouts_per_round
    horizon = model.cfg.inner_rollout_horizon

    assert count == 2
    assert horizon == 2
    assert replay.size == count * horizon == 4
    assert replay.next_sample_id == 4
    torch.testing.assert_close(replay.sample_id[:4], torch.arange(4), rtol=0, atol=0)
    torch.testing.assert_close(
        replay.action[:4, 0], torch.tensor([0.0, 1.0, 10.0, 11.0])
    )
    torch.testing.assert_close(
        replay.z[:4],
        torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(-1).expand(-1, 8),
    )
    torch.testing.assert_close(
        replay.next_z[:4],
        torch.tensor([0.0, 1.0, 10.0, 12.0]).unsqueeze(-1).expand(-1, 8),
    )
    torch.testing.assert_close(
        replay.reward[:4, 0], torch.tensor([100.0, 101.0, 110.0, 111.0])
    )
    torch.testing.assert_close(replay.terminated[:4], torch.zeros(4, 1))
    torch.testing.assert_close(rollout["lengths"], torch.full((2,), 2))
    torch.testing.assert_close(rollout["terminated"], torch.zeros(2, dtype=torch.bool))
    assert engine.state.policy_evaluations == 4
