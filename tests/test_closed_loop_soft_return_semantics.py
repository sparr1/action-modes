"""Mixed soft inner learning with a frozen reward-only horizon boundary."""

from copy import deepcopy
import math

import pytest
import torch

from tests.test_ambi_eval_execution import _capture_execution
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_component_model
from tests.test_aux_return_inner import _prepared


MIXED_SOURCES = dict(
    aux_return_mode="sac", inner_actor_source="sac", inner_critic_source="sac",
    inner_horizon_actor_source="sac", inner_horizon_critic_source="aux_return",
    inner_sac_critic_target="entropy_augmented", inner_terminal_entropy="none",
    inner_finite_horizon=True,
)


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("estimator", ["one_step", "retrace"])
def test_mixed_targets_use_soft_interior_and_unaugmented_return_boundary(
    horizon, estimator, monkeypatch,
):
    """A numerical oracle distinguishes all three Q roles and the entropy edge."""
    with _prepared(
        **MIXED_SOURCES, inner_rollout_horizon=horizon, train_unroll_horizon=3,
        inner_sac_return_estimator=estimator, inner_retrace_lambda=.9,
        inner_log_std_mapping="direct_clamp", inner_log_std_min=-10.,
        inner_log_std_max=2.,
    ) as (holder, engine):
        # The fixture's soft critic is 2, return critic is 9. Only the adapted
        # target is then changed to 5, separating initialization from bootstrap.
        z = torch.zeros(2 * horizon, holder.cfg.latent_dim)
        action = torch.zeros(2 * horizon, 1)
        initial = engine.model.Q(z, action, qs=engine.state.critic, reduction="all")
        torch.testing.assert_close(initial, torch.full_like(initial, 2.))
        assert engine._critic_base is engine.model._Qs
        assert engine._horizon_critic is engine.model._aux_return_Qs
        assert engine._horizon_actor is engine.model._pi
        with torch.no_grad():
            for head in engine.state.critic_target:
                head[-1].bias.fill_(5.)
        outer_before = deepcopy(holder.agent.checkpoint_state())

        # The actor is N(-.3, exp(-2)^2) before tanh. At zero noise the
        # differential action entropy sample is log(sigma)+log(sqrt(2*pi))
        # plus log(1-tanh(mu)^2); no implementation entropy helper is used.
        entropy = -2. + .5 * math.log(2. * math.pi) + math.log(1. - math.tanh(-.3)**2)
        shape = (2, horizon, 1)
        batch = dict(
            z=z.reshape(2, horizon, -1), next_z=z.reshape(2, horizon, -1),
            action=torch.full(shape, math.tanh(-.3)),
            pre_tanh_action=torch.full(shape, -.3),
            behavior_log_prob=torch.full(shape, -entropy),
            reward=torch.arange(1., horizon + 1.).view(1, horizon, 1).expand(2, -1, -1),
            terminated=torch.zeros(shape), horizon_end=torch.zeros(shape),
            valid=torch.ones(shape, dtype=torch.bool),
        )
        batch["horizon_end"][:, -1] = 1.
        batch["terminated"][1, -1] = 1.
        policy_calls, q_calls = [], []
        original_pi, original_q = engine.model.pi, engine.model.Q

        def pi(*args, **kwargs):
            policy_calls.append(kwargs.get("policy"))
            return original_pi(*args, **kwargs)

        def q(*args, **kwargs):
            q_calls.append(kwargs.get("qs"))
            return original_q(*args, **kwargs)

        monkeypatch.setattr(engine.model, "pi", pi)
        monkeypatch.setattr(engine.model, "Q", q)

        def kernel(alpha):
            if estimator == "retrace":
                return engine._retrace_critic_kernel(
                    batch, torch.tensor(alpha), torch.zeros(shape),
                    torch.zeros(2, 1), torch.tensor([0, 1]),
                )
            return engine._sac_critic_kernel(
                *[batch[key].flatten(0, 1) for key in
                  ("z", "action", "reward", "next_z", "terminated")],
                torch.tensor(alpha), torch.zeros(2 * horizon, 1), torch.tensor([0, 1]),
                batch["horizon_end"].flatten(0, 1), torch.zeros(2 * horizon, 1),
            )

        cold, hot = kernel(0.), kernel(.31)
        targets = hot[2].reshape(2, horizon)
        gamma = float(holder.agent.discount)
        expected = torch.empty_like(targets)
        for trajectory in range(2):
            for step in reversed(range(horizon)):
                if step == horizon - 1:
                    expected[trajectory, step] = step + 1. + (gamma * 9. if trajectory == 0 else 0.)
                else:
                    continuation = 5. + .31 * entropy
                    if estimator == "retrace":
                        continuation += .9 * (expected[trajectory, step + 1] - 5.)
                    expected[trajectory, step] = step + 1. + gamma * continuation
        torch.testing.assert_close(targets, expected)
        torch.testing.assert_close(targets[:, -1], cold[2].reshape(2, horizon)[:, -1], rtol=0, atol=0)
        if horizon > 1:
            assert not torch.allclose(targets[:, :-1], cold[2].reshape(2, horizon)[:, :-1])
        else:
            torch.testing.assert_close(hot[2], cold[2], rtol=0, atol=0)
            if estimator == "retrace":
                assert torch.count_nonzero(hot[6]) == 0
        assert policy_calls == [engine.state.actor, engine.model._pi] * 2
        assert sum(qs is engine.model._aux_return_Qs for qs in q_calls) == 2
        assert not hot[2].requires_grad
        hot[0].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in engine.state.critic_params)
        assert all(p.grad is None for p in engine.model.parameters())
        _assert_tree_equal(holder.agent.checkpoint_state(), outer_before)


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("estimator", ["one_step", "retrace"])
def test_mixed_solve_mean_and_sample_share_adaptation_and_preserve_outer(
    horizon, estimator, monkeypatch,
):
    holder = _tiny_component_model(
        **MIXED_SOURCES, inner_rollout_horizon=horizon, train_unroll_horizon=3,
        inner_sac_return_estimator=estimator, inner_retrace_lambda=.9,
        inner_rounds=2, inner_rollouts_per_round=128, inner_batch_size=256,
        inner_critic_updates_per_round=2, inner_actor_updates_per_round=1,
        inner_replay_capacity=3840, inner_temperature_mode="auto",
    )
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        captured = _capture_execution(monkeypatch, engine)
        outer_before = deepcopy(agent.checkpoint_state())
        global_before = torch.random.get_rng_state().clone()
        for execution in ("mean", "policy_sample"):
            holder.cfg.inner_eval_execution_action = execution
            engine.reset_for_evaluation(919)
            action = agent.act(torch.zeros(3), t0=True, eval_mode=True)
            final = captured[-1]
            expected = final["mean" if execution == "mean" else "sample"]
            torch.testing.assert_close(action, expected, rtol=0, atol=0)
            metrics = agent.last_inner_metrics
            assert final["eval_mode"] is True
            assert metrics["inner_eval_execution_sampled"] == int(execution == "policy_sample")
            assert metrics["inner_model_steps"] == metrics["inner_buffer_size"] == 256 * horizon
            assert metrics["inner_critic_optimizer_steps"] == 4
            assert metrics["inner_actor_optimizer_steps"] == metrics["inner_temperature_optimizer_steps"] == 2
            if estimator == "retrace":
                assert metrics["inner_retrace_trajectory_draws"] == 4 * math.ceil(256 / horizon)
                assert metrics["inner_retrace_critic_rows"] == 4 * math.ceil(256 / horizon) * horizon
                if horizon == 1:
                    assert metrics["inner_retrace_effective_trace_length"] == 1
                    assert metrics["inner_retrace_correction_abs_mean"] == 0
            assert all(math.isfinite(float(value)) for value in metrics.values())
            _assert_tree_equal(agent.checkpoint_state(), outer_before)
            torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
        _assert_tree_equal(captured[0]["adaptation"], captured[1]["adaptation"])
        assert not torch.equal(captured[1]["sample"], captured[1]["mean"])
    finally:
        holder.close()
