"""Execution, entropy and full-replay contracts for the three new panels."""

from copy import deepcopy
import math

import pytest
import torch

from tests.test_ambi_eval_execution import _capture_execution
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_component_model
from tests.test_aux_return_inner import _prepared


RETURN_SOURCES = dict(
    aux_return_mode="sac", inner_actor_source="sac", inner_critic_source="aux_return",
    inner_horizon_actor_source="sac", inner_horizon_critic_source="aux_return",
    inner_sac_critic_target="reward_only", inner_terminal_entropy="none",
    inner_finite_horizon=True,
)
CASES = [
    ("alpha0", h, e, 2, 2, 1, 3072, "policy_sample", .9, False)
    for h in (1, 2, 3) for e in ("one_step", "retrace")
] + [
    ("lambda1", h, "retrace", 2, 2, 1, 3072, "mean", 1., True)
    for h in (1, 2, 3)
] + [("h4", 4, "one_step", 14, 16, 4, 7168, "mean", 1., True)]


@pytest.mark.parametrize("panel,horizon,estimator,rounds,c,a,capacity,execution,lam,entropy", CASES)
def test_panel_solve_executes_selected_policy_with_exact_work_and_frozen_outer(
    panel, horizon, estimator, rounds, c, a, capacity, execution, lam, entropy, monkeypatch,
):
    holder = _tiny_component_model(
        **RETURN_SOURCES, inner_rollout_horizon=horizon, train_unroll_horizon=3,
        inner_sac_return_estimator=estimator, inner_retrace_lambda=lam,
        inner_rounds=rounds, inner_rollouts_per_round=128, inner_batch_size=256,
        inner_critic_updates_per_round=c, inner_actor_updates_per_round=a,
        inner_replay_capacity=capacity, inner_entropy_enabled=entropy,
        inner_temperature_mode="auto" if entropy else "inherit_outer",
        inner_eval_execution_action=execution,
    )
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        captured = _capture_execution(monkeypatch, engine)
        outer = deepcopy(agent.checkpoint_state())
        global_rng = torch.random.get_rng_state().clone()
        engine.reset_for_evaluation(919)
        action = agent.act(torch.zeros(3), t0=True, eval_mode=True)
        result = captured[-1]
        torch.testing.assert_close(action, result["sample" if execution == "policy_sample" else "mean"], rtol=0, atol=0)
        metrics = agent.last_inner_metrics
        assert engine._critic_base is engine.model._aux_return_Qs
        assert engine._horizon_critic is engine.model._aux_return_Qs
        assert metrics["inner_model_steps"] == metrics["inner_buffer_size"] == 128 * horizon * rounds
        assert metrics["inner_critic_optimizer_steps"] == c * rounds
        assert metrics["inner_actor_optimizer_steps"] == a * rounds
        assert metrics["inner_temperature_optimizer_steps"] == (a * rounds if entropy else 0)
        assert metrics["inner_eval_execution_sampled"] == int(execution == "policy_sample")
        if not entropy:
            assert metrics["inner_alpha_initial"] == metrics["inner_alpha_final"] == 0
            assert not torch.equal(result["sample"], result["mean"])
        else:
            assert metrics["inner_alpha_initial"] > 0 and metrics["inner_alpha_final"] > 0
            assert metrics["inner_eval_execution_mean_action_l2"] == 0
        if estimator == "retrace":
            trajectories = math.ceil(256 / horizon)
            assert metrics["inner_retrace_trajectory_draws"] == c * rounds * trajectories
            assert metrics["inner_retrace_critic_rows"] == c * rounds * trajectories * horizon
            if horizon == 1:
                assert metrics["inner_retrace_correction_abs_mean"] == 0
        assert all(math.isfinite(float(v)) for v in metrics.values())
        _assert_tree_equal(agent.checkpoint_state(), outer)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        holder.close()


def test_lambda_one_keeps_clipped_off_policy_corrections_and_return_boundary():
    """Lambda=1 does not force importance coefficients to one."""
    with _prepared(
        **RETURN_SOURCES, inner_rollout_horizon=3, train_unroll_horizon=3,
        inner_sac_return_estimator="retrace", inner_retrace_lambda=1.,
        inner_log_std_mapping="direct_clamp", inner_log_std_min=-10., inner_log_std_max=2.,
    ) as (holder, engine):
        with torch.no_grad():
            for head in engine.state.critic_target:
                head[-1].bias.fill_(5.)
        entropy = -2. + .5 * math.log(2. * math.pi) + math.log(1. - math.tanh(-.3)**2)
        shape = (1, 3, 1)
        z = torch.zeros(1, 3, holder.cfg.latent_dim)
        batch = dict(z=z, next_z=z, action=torch.full(shape, math.tanh(-.3)),
            pre_tanh_action=torch.full(shape, -.3),
            behavior_log_prob=torch.full(shape, -entropy + math.log(2.)),
            reward=torch.tensor([[[1.], [2.], [3.]]]), terminated=torch.zeros(shape),
            horizon_end=torch.tensor([[[0.], [0.], [1.]]]), valid=torch.ones(shape, dtype=torch.bool))
        output = engine._retrace_critic_kernel(
            batch, torch.tensor(.3), torch.zeros(shape), torch.zeros(1, 1), torch.tensor([0, 1]),
        )
        gamma = float(holder.agent.discount)
        g2 = 3. + gamma * 9.
        g1 = 2. + gamma * (5. + .5 * (g2 - 5.))
        g0 = 1. + gamma * (5. + .5 * (g1 - 5.))
        torch.testing.assert_close(output[2].reshape(shape), torch.tensor([[[g0], [g1], [g2]]]))
        torch.testing.assert_close(output[4], torch.full(shape, .5))
        assert not output[2].requires_grad
