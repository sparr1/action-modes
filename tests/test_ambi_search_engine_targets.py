"""White-box contracts for finite-search target dispatch.

The broad search smoke matrix lives in ``test_ambi_search_engine.py``.  These
tests instead replace learned values with hand-written scalars so a passing
action smoke cannot hide a wrong depth, leaf, entropy, or importance weight.
"""

import pytest
import torch

from tests.test_ambi_root_local_sac import _tiny_component_model


def _finite_model(estimator="td0", **overrides):
    params = {
        "discount": 0.5,
        "train_unroll_horizon": 3,
        "inner_q_objective": "finite_horizon",
        "inner_critic_horizon_mode": "stage_heads",
        "inner_return_estimator": estimator,
        "inner_search_replay_retention": "action",
        "inner_offpolicy_mode": "none",
        "inner_search_bootstrap_critic": "target",
        "inner_target_update_event": "optimizer_step",
        "inner_depth_update_order": "mixed",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 3,
        "inner_replay_capacity": 6,
        "inner_batch_size": 2,
        "inner_critic_updates_per_round": 1,
        "inner_actor_updates_per_round": 1,
        "inner_behavior_action": "policy_sample",
        "inner_behavior_std_scale": 1.0,
        "inner_behavior_noise_std": 0.0,
    }
    if estimator == "n_step":
        params["inner_return_steps"] = 2
        params["inner_search_replay_retention"] = "round"
    elif estimator == "lambda_return":
        params["inner_return_lambda"] = 0.5
        params["inner_search_replay_retention"] = "round"
    elif estimator == "full_suffix":
        params.update(
            inner_search_replay_retention="round",
            inner_search_bootstrap_critic="none",
            inner_target_update_event="none",
        )
    elif estimator == "retrace":
        params.update(
            inner_return_lambda=0.8,
            inner_offpolicy_mode="per_decision_is",
        )
    params.update(overrides)
    return _tiny_component_model(**params)


def _vtrace_model(**overrides):
    params = {
        "discount": 0.5,
        "train_unroll_horizon": 3,
        "inner_operator": "vtrace",
        "inner_q_objective": "finite_horizon",
        "inner_critic_horizon_mode": "stage_heads",
        "inner_return_estimator": "td0",
        "inner_return_lambda": 0.8,
        "inner_search_replay_retention": "action",
        "inner_offpolicy_mode": "per_decision_is",
        "inner_search_bootstrap_critic": "target",
        "inner_target_update_event": "optimizer_step",
        "inner_depth_update_order": "mixed",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 3,
        "inner_replay_capacity": 6,
        "inner_batch_size": 2,
        "inner_critic_updates_per_round": 1,
        "inner_actor_updates_per_round": 1,
        "inner_behavior_action": "policy_sample",
        "inner_behavior_std_scale": 1.0,
        "inner_behavior_noise_std": 0.0,
        "inner_vtrace_distill_updates": 1,
        "inner_vtrace_distill_action_samples": 1,
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
    }
    params.update(overrides)
    return _tiny_component_model(**params)


def _prepare(model):
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def _batch(model, horizons, rewards, *, pre_tanh=None, behavior_log_prob=None):
    batch_size = len(horizons)
    time = int(model.cfg.inner_rollout_horizon)
    latent_dim, action_dim = int(model.cfg.latent_dim), int(model.cfg.action_dim)
    z = torch.zeros(batch_size, time, latent_dim)
    next_z = torch.zeros_like(z)
    action = torch.zeros(batch_size, time, action_dim)
    pre_tanh_action = torch.zeros_like(action)
    reward = torch.zeros(batch_size, time, 1)
    valid = torch.zeros(batch_size, time, 1, dtype=torch.bool)
    remaining = torch.zeros(batch_size, time, 1, dtype=torch.long)
    for row, horizon in enumerate(horizons):
        reward[row, :horizon, 0] = torch.as_tensor(rewards[row])[:horizon]
        valid[row, :horizon] = True
        remaining[row, :horizon, 0] = torch.arange(horizon, 0, -1)
    if pre_tanh is not None:
        pre_tanh_action[..., 0] = torch.as_tensor(pre_tanh)
    if behavior_log_prob is None:
        behavior_log_prob = torch.zeros(batch_size, time, 1)
    return {
        "z": z,
        "action": action,
        "pre_tanh_action": pre_tanh_action,
        "reward": reward,
        "next_z": next_z,
        "terminated": torch.zeros_like(valid),
        "valid": valid,
        "behavior_log_prob": behavior_log_prob,
        "round_id": torch.zeros(batch_size, time, 1, dtype=torch.long),
        "remaining_horizon": remaining,
    }


def test_td0_dispatch_uses_outer_leaf_only_at_depth_one(monkeypatch):
    model = _finite_model(
        outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
    )
    try:
        engine = _prepare(model)
        batch = _batch(model, [1, 2, 3], [[1], [1, 2], [1, 2, 3]])
        calls = []

        def successor(next_z, next_depth, **_):
            calls.append(next_depth.detach().clone())
            return 10.0 + next_depth.reshape(-1, 1).to(next_z.dtype)

        monkeypatch.setattr(engine, "_search_successor_value", successor)
        _, target, depth, diagnostics = engine._build_search_q_target(batch)

        torch.testing.assert_close(depth, torch.tensor([1, 2, 3]))
        torch.testing.assert_close(calls[0], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(target, torch.tensor([[6.0], [6.5], [7.0]]))
        # Only depth one reaches V_0; deeper TD(0) rows recursively bootstrap
        # from an inner critic and must not be logged as leaf contribution.
        torch.testing.assert_close(
            diagnostics["leaf_contribution"],
            torch.tensor([[5.0], [0.0], [0.0]]),
        )
    finally:
        model.env.close()


def test_full_suffix_dispatch_places_entropy_only_on_future_actions(monkeypatch):
    model = _finite_model("full_suffix")
    try:
        engine = _prepare(model)
        with torch.no_grad():
            if engine.state.log_alpha is not None:
                engine.state.log_alpha.fill_(torch.log(torch.tensor(0.5)))
            else:
                engine.state.alpha_fixed.fill_(0.5)
        batch = _batch(
            model,
            [3],
            [[1.0, 2.0, 3.0]],
            pre_tanh=[[-9.0, 0.2, 0.4]],
        )
        depths = []

        def successor(next_z, next_depth, **_):
            depths.append(next_depth.detach().clone())
            return next_z.new_full((next_z.shape[0], 1), 10.0)

        monkeypatch.setattr(engine, "_search_successor_value", successor)
        monkeypatch.setattr(
            engine,
            "_search_behavior_log_prob",
            lambda _z, pre_tanh: pre_tanh[:, :1],
        )
        _, target, _, diagnostics = engine._build_search_q_target(batch)

        # 1 + .5*(2-.5*.2) + .25*(3-.5*.4) + .125*10 = 3.9.
        torch.testing.assert_close(target, torch.tensor([[3.9]]))
        torch.testing.assert_close(depths[0], torch.tensor([0]))
        torch.testing.assert_close(
            diagnostics["leaf_contribution"], torch.tensor([[1.25]])
        )
    finally:
        model.env.close()


def test_n_step_pdis_dispatch_uses_w_n_minus_one_bootstrap(monkeypatch):
    model = _finite_model(
        "n_step",
        inner_search_replay_retention="action",
        inner_offpolicy_mode="per_decision_is",
        outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
    )
    try:
        engine = _prepare(model)
        batch = _batch(
            model,
            [3],
            [[1.0, 2.0, 3.0]],
            pre_tanh=[
                [
                    torch.log(torch.tensor(100.0)),
                    torch.log(torch.tensor(2.0)),
                    torch.log(torch.tensor(3.0)),
                ]
            ],
        )
        depths = []

        def successor(next_z, next_depth, **_):
            depths.append(next_depth.detach().clone())
            return next_z.new_full((next_z.shape[0], 1), 8.0)

        monkeypatch.setattr(engine, "_search_successor_value", successor)
        monkeypatch.setattr(
            engine,
            "_search_behavior_log_prob",
            lambda _z, pre_tanh: pre_tanh[:, :1],
        )
        _, target, _, diagnostics = engine._build_search_q_target(batch)

        # 1 + .5*2*2 + .25*W1*8 = 7.  rho0 is conditioned away and rho2
        # is not part of the two-step bootstrap's W_(n-1).
        torch.testing.assert_close(target, torch.tensor([[7.0]]))
        torch.testing.assert_close(
            diagnostics["bootstrap_importance_weight"], torch.tensor([[2.0]])
        )
        torch.testing.assert_close(depths[0], torch.tensor([1]))
        torch.testing.assert_close(
            diagnostics["leaf_contribution"], torch.tensor([[0.0]])
        )
    finally:
        model.env.close()


def test_retrace_dispatch_uses_depth_aware_td_error_control_variate(monkeypatch):
    model = _finite_model("retrace")
    try:
        engine = _prepare(model)
        batch = _batch(
            model,
            [3],
            [[1.0, 2.0, 3.0]],
            pre_tanh=[
                [
                    torch.log(torch.tensor(100.0)),
                    torch.log(torch.tensor(0.5)),
                    torch.log(torch.tensor(2.0)),
                ]
            ],
        )
        batch["z"][0, :, 0] = torch.tensor([4.0, 5.0, 6.0])
        batch["next_z"][0, :, 0] = torch.tensor([5.0, 6.0, 10.0])
        prediction_depths, successor_depths = [], []

        def predictions(z, _action, remaining_horizon, **_):
            prediction_depths.append(remaining_horizon.detach().clone())
            q = z[:, :1]
            return torch.stack((q, q), dim=0)

        def successor(next_z, next_depth, **_):
            successor_depths.append(next_depth.detach().clone())
            return next_z[:, :1]

        monkeypatch.setattr(engine, "_search_q_predictions", predictions)
        monkeypatch.setattr(engine, "_search_successor_value", successor)
        monkeypatch.setattr(
            engine,
            "_search_behavior_log_prob",
            lambda _z, pre_tanh: pre_tanh[:, :1],
        )
        _, target, _, diagnostics = engine._build_search_q_target(batch)

        torch.testing.assert_close(target, torch.tensor([[3.66]]))
        torch.testing.assert_close(prediction_depths[0], torch.tensor([3, 2, 1]))
        torch.testing.assert_close(successor_depths[0], torch.tensor([2, 1, 0]))
        torch.testing.assert_close(
            diagnostics["trace_c"].reshape(-1),
            torch.tensor([0.8, 0.4, 0.8]),
        )
    finally:
        model.env.close()


def test_resimulation_starts_after_anchor_and_counts_only_valid_target_steps():
    model = _finite_model(
        "full_suffix",
        inner_search_replay_retention="action",
        inner_offpolicy_mode="resimulate",
    )
    try:
        engine = _prepare(model)
        batch = _batch(model, [3, 2], [[1, 2, 3], [4, 5]])
        first = {
            name: batch[name][:, 0].clone()
            for name in (
                "z",
                "action",
                "pre_tanh_action",
                "reward",
                "next_z",
                "terminated",
                "valid",
                "behavior_log_prob",
                "remaining_horizon",
            )
        }

        rebuilt = engine._resimulate_search_batch(batch)

        assert engine.state.target_model_steps == 3  # (h=3 -> 2) + (h=2 -> 1)
        assert rebuilt["valid"][:, :, 0].tolist() == [
            [True, True, True],
            [True, True, False],
        ]
        torch.testing.assert_close(
            rebuilt["remaining_horizon"][:, :, 0],
            torch.tensor([[3, 2, 1], [2, 1, 0]]),
        )
        for name, value in first.items():
            torch.testing.assert_close(rebuilt[name][:, 0], value)
    finally:
        model.env.close()


@pytest.mark.parametrize(("source", "target_flag"), [("outer_target", True), ("outer_online", False)])
def test_outer_leaf_uses_configured_frozen_q_and_outer_alpha(
    monkeypatch, source, target_flag
):
    model = _finite_model(inner_leaf_q_source=source, inner_leaf_value_samples=2)
    try:
        engine = _prepare(model)
        engine._search_outer_alpha = torch.tensor(0.25)
        calls = []

        def pi(z, *, policy, **_):
            calls.append(("pi", policy, z.shape[0]))
            return z.new_zeros(z.shape[0], model.cfg.action_dim), {
                "log_prob": z.new_full((z.shape[0], 1), -2.0)
            }

        def q(z, action, *, target, **_):
            calls.append(("q", target, z.shape[0], action.shape[0]))
            return z.new_full((z.shape[0], 1), 4.0)

        monkeypatch.setattr(engine.model, "pi", pi)
        monkeypatch.setattr(engine.model, "Q", q)
        value = engine._search_outer_leaf_value(
            torch.zeros(3, model.cfg.latent_dim),
            generator=engine.rng.generator("bootstrap"),
            pair_indices=torch.tensor([0, 1]),
        )

        torch.testing.assert_close(value, torch.full((3, 1), 4.5))
        assert calls[0] == ("pi", engine.model._pi, 6)
        assert calls[1] == ("q", target_flag, 6, 6)
    finally:
        model.env.close()


def test_vtrace_dispatch_routes_successor_depths_and_canonical_targets(monkeypatch):
    model = _vtrace_model()
    try:
        engine = _prepare(model)
        batch = _batch(
            model,
            [3],
            [[1.0, 2.0, 3.0]],
            pre_tanh=[
                [
                    torch.log(torch.tensor(2.0)),
                    torch.log(torch.tensor(0.5)),
                    torch.log(torch.tensor(3.0)),
                ]
            ],
        )
        batch["z"][0, :, 0] = torch.tensor([4.0, 5.0, 6.0])
        batch["next_z"][0, :, 0] = torch.tensor([5.0, 6.0, 10.0])
        successor_depths = []

        class FixedValue(torch.nn.Module):
            def forward(self, z, _remaining_horizon):
                return z[:, :1]

        def successor(next_z, next_depth, **_):
            successor_depths.append(next_depth.detach().clone())
            return next_z[:, :1]

        engine.state.critic_target = FixedValue()
        monkeypatch.setattr(engine, "_search_successor_value", successor)
        monkeypatch.setattr(
            engine,
            "_search_behavior_log_prob",
            lambda _z, pre_tanh: pre_tanh[:, :1],
        )
        result, _ = engine._build_vtrace_target(batch)

        torch.testing.assert_close(
            result["value_target"].reshape(-1),
            torch.tensor([3.66, 5.4, 8.0]),
        )
        torch.testing.assert_close(
            result["pg_advantage"].reshape(-1),
            torch.tensor([-0.3, 0.5, 2.0]),
        )
        torch.testing.assert_close(successor_depths[0], torch.tensor([2, 1, 0]))
    finally:
        model.env.close()
