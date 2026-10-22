"""Temperature-only dose and episode carry preserve the other inner budgets."""

import math

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_component_model


def _model(**overrides):
    return _tiny_component_model(**{
        "inner_rounds": 2,
        "inner_critic_updates_per_round": 2,
        "inner_actor_updates_per_round": 2,
        "inner_temperature_updates_per_actor": 1,
        "inner_temperature_lr": 0.003,
        "inner_target_entropy": -2.0,
        **overrides,
    })


@pytest.mark.parametrize("repeats,learning_rate", [(1, .003), (3, .003), (100, .003), (100, 1.)])
def test_repeated_temperature_matches_scalar_adam_and_reuses_policy_sample(repeats, learning_rate):
    model = _model(
        inner_temperature_updates_per_actor=repeats,
        inner_temperature_lr=learning_rate,
    )
    try:
        engine = model.agent.inner_engine
        engine._prepare_workspace(t0=True)
        state = engine.state
        reference = torch.nn.Parameter(state.log_alpha.detach().clone())
        optimizer = torch.optim.Adam(
            [reference], lr=model.cfg.inner_temperature_lr,
            eps=model.cfg.inner_adam_eps, foreach=False,
        )
        calls = []
        values = torch.tensor([[-0.5], [-1.0], [-1.5], [-2.0]])

        def actor_region(z, alpha, policy_noise, pair_indices, update_actor):
            calls.append(policy_noise.detach().clone())
            loss = next(state.actor.parameters()).sum() * .01
            zero = z.new_zeros(())
            return values, -values, loss, zero, zero, zero, zero, zero

        engine._compile_regions["actor"] = actor_region
        outer_before = _clone_tree(model.agent.model.state_dict())
        metrics = engine._sac_policy_step(
            {"z": torch.zeros(4, model.cfg.latent_dim)},
            update_actor=True, update_temperature=True, alpha=engine.alpha.detach(),
        )
        losses = []
        for _ in range(repeats):
            loss = -(reference * (values + model.cfg.inner_target_entropy)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [reference], engine._temperature_grad_clip_maximum,
            )
            optimizer.step()
            with torch.no_grad():
                reference.clamp_(min=math.log(1e-8))
            losses.append(loss.detach())
        torch.testing.assert_close(state.log_alpha, reference, rtol=0, atol=0)
        _assert_tree_equal(state.temperature_optim.state_dict(), optimizer.state_dict())
        torch.testing.assert_close(metrics["temperature_loss"], torch.stack(losses).mean())
        assert metrics["temperature_entropy_residual"].item() == pytest.approx(3.25)
        assert metrics["temperature_updates"] == repeats
        assert len(calls) == state.actor_steps == 1
        assert state.temperature_steps == state.temperature_lifetime_steps == repeats
        assert state.policy_evaluations == 4
        _assert_tree_equal(model.agent.model.state_dict(), outer_before)
    finally:
        model.close()


def test_repetition_keeps_actor_critic_and_rng_budgets_and_trace_counts():
    observations = []
    for repeats in (1, 100):
        model = _model(inner_temperature_updates_per_actor=repeats)
        try:
            outer_before = _clone_tree(model.agent.model.state_dict())
            outer_alpha = model.agent.alpha.detach().clone()
            trace = InnerActionTrace()
            model.agent.act(torch.zeros(3), t0=True, eval_mode=True, trace=trace)
            metrics = model.agent.last_inner_metrics
            for name in ("inner_actor_optimizer_steps", "inner_critic_optimizer_steps"):
                assert metrics[name] == 4
            assert metrics["inner_temperature_optimizer_steps"] == 4 * repeats
            assert model.cfg.inner_temperature_updates_per_action == 4 * repeats
            assert model.cfg.inner_expected_update_slots == 8
            assert metrics["inner_temperature_optimizer_step_initial"] == 0
            assert metrics["inner_temperature_optimizer_step_final"] == 4 * repeats
            assert metrics["inner_temperature_lifetime_steps_initial"] == 0
            assert metrics["inner_temperature_lifetime_steps_final"] == 4 * repeats
            updates = [row for row in trace.events if row["phase"] == "update"]
            assert len(updates) == 8
            actor_rows = [row for row in updates if row["updated_actor"]]
            assert [row["temperature_updates"] for row in actor_rows] == [
                repeats * step for step in (1, 2, 3, 4)
            ]
            assert all(row["metrics"]["temperature_updates"] == repeats for row in actor_rows)
            assert actor_rows[1]["metrics"]["alpha_used"] != actor_rows[0]["metrics"]["alpha_used"]
            observations.append((
                {key: metrics[key] for key in (
                    "inner_policy_evaluations", "inner_q_evaluations",
                    "inner_replay_draws", "inner_buffer_size",
                )},
                _clone_tree(model.agent.inner_engine.rng.training_state_dict()),
            ))
            _assert_tree_equal(model.agent.model.state_dict(), outer_before)
            torch.testing.assert_close(model.agent.alpha, outer_alpha, rtol=0, atol=0)
        finally:
            model.close()
    _assert_tree_equal(observations[0], observations[1])


@pytest.mark.parametrize("scope", ["action", "episode"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_only_temperature_and_adam_carry_then_reset_at_episode_boundary(scope, auxiliary):
    extra = ({
        "aux_return_mode": "sac",
        "inner_actor_source": "sac", "inner_critic_source": "aux_return",
        "inner_horizon_actor_source": "sac", "inner_horizon_critic_source": "aux_return",
        "inner_sac_critic_target": "reward_only",
    } if auxiliary else {})
    model = _model(
        inner_temperature_scope=scope, inner_temperature_optimizer_scope=scope,
        **extra,
    )
    try:
        engine = model.agent.inner_engine
        outer_before = _clone_tree(model.agent.model.state_dict())
        outer_alpha = model.agent.alpha.detach().clone()
        previous_alpha = None
        previous_optimizer = None
        initial_records = []
        prepare = engine._prepare_workspace

        def record_prepared(*, t0):
            prepare(t0=t0)
            state = engine.state
            # Fresh actor, critic and empty replay for every root, even when
            # alpha/Adam survive. Allocation pooling does not imply transfer.
            _assert_tree_equal(state.actor.state_dict(), engine._actor_owner.model._pi.state_dict())
            expected_critic = (model.agent.model._aux_return_Qs if auxiliary else model.agent.model._Qs)
            _assert_tree_equal(state.critic.state_dict(), expected_critic.state_dict())
            assert state.replay.size == 0
            assert state.actor_lifetime_steps == state.critic_lifetime_steps == 0
            initial_records.append(_clone_tree(state.temperature_optim.state_dict()))

        engine._prepare_workspace = record_prepared
        for decision, t0 in enumerate((True, False, False, True)):
            model.agent.act(torch.zeros(3), t0=t0, eval_mode=True)
            metrics = model.agent.last_inner_metrics
            carries = scope == "episode" and not t0
            expected_start = decision * 4 if carries else 0
            assert metrics["inner_temperature_optimizer_step_initial"] == expected_start
            assert metrics["inner_temperature_optimizer_step_final"] == expected_start + 4
            assert metrics["inner_temperature_lifetime_steps_initial"] == expected_start
            assert metrics["inner_temperature_lifetime_steps_final"] == expected_start + 4
            assert metrics["inner_temperature_episode_reset"] == float(t0)
            if carries:
                assert metrics["inner_alpha_initial"] == previous_alpha
                _assert_tree_equal(initial_records[-1], previous_optimizer)
            else:
                assert metrics["inner_alpha_initial"] == pytest.approx(outer_alpha.item())
                for slots in initial_records[-1]["state"].values():
                    assert all(torch.count_nonzero(value) == 0 for value in slots.values())
            previous_alpha = metrics["inner_alpha_final"]
            if scope == "episode":
                previous_optimizer = _clone_tree(engine.state.temperature_optim.state_dict())
        _assert_tree_equal(model.agent.model.state_dict(), outer_before)
        torch.testing.assert_close(model.agent.alpha, outer_alpha, rtol=0, atol=0)
    finally:
        model.close()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100", None])
def test_repeat_count_requires_a_strict_positive_integer(value):
    with pytest.raises(ValueError, match="inner_temperature_updates_per_actor"):
        _build_cfg(inner_temperature_updates_per_actor=value)


@pytest.mark.parametrize("override", [
    {"inner_temperature_mode": "fixed"},
    {"inner_actor_adaptation": "frozen"},
    {"inner_actor_updates_per_round": 0, "inner_critic_updates_per_round": 1},
    {"inner_updates_per_round": 1},
])
def test_repeated_temperature_rejects_inactive_or_unpaired_actor(override):
    with pytest.raises(ValueError, match="inner_temperature_updates_per_actor"):
        _build_cfg(inner_temperature_updates_per_actor=100, **override)
