"""MPPI candidate policies and horizon policies have separate identities."""

from dataclasses import replace

import pytest
import torch

from RL.tdmpc2_core.mppi import MPPIModelCallbacks, mppi_plan
from tests.test_aux_return_inner import _prepared
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree


def _callbacks(seen):
    def policy(z, *, generator):
        seen.append(("candidate", len(z)))
        return z.new_full((len(z), 1), .25)
    def terminal(z, *, generator):
        seen.append(("terminal", len(z)))
        return z.new_full((len(z), 1), -.8)
    def q(z, action, *, reduction, generator):
        torch.testing.assert_close(action, action.new_full(action.shape, -.8))
        return z.new_full((len(z), 1), 7.)
    return MPPIModelCallbacks(
        action_dim=1, dynamics=lambda z, a: z,
        reward=lambda z, a: a, policy=policy, terminal_policy=terminal,
        terminal_q=q,
    )


def test_mppi_uses_candidate_actor_only_for_policy_trajectories_and_separate_tail():
    seen = []
    result = mppi_plan(
        torch.zeros(1, 2), callbacks=_callbacks(seen), horizon=2, iterations=1,
        num_samples=2, num_pi_trajs=2, num_elites=2, temperature=1.,
        min_std=.01, max_std=1., discount=.9, q_reduction="mean_all",
        generator=torch.Generator().manual_seed(3), eval_mode=True,
    )
    assert seen == [("candidate", 2), ("candidate", 2), ("terminal", 2)]
    assert result.metrics["planner_value_mean"] == pytest.approx(.25+.9*.25+.9**2*7)
    assert result.model_steps == 6


def test_no_policy_trajectories_still_uses_terminal_actor():
    seen = []
    mppi_plan(
        torch.zeros(1, 2), callbacks=_callbacks(seen), horizon=2, iterations=2,
        num_samples=2, num_pi_trajs=0, num_elites=2, temperature=1.,
        min_std=.01, max_std=1., discount=.9, q_reduction="mean_all",
        generator=torch.Generator().manual_seed(3), eval_mode=True,
    )
    assert seen == [("terminal", 2), ("terminal", 2)]


def test_unspecified_terminal_policy_preserves_legacy_samples_and_generator_state():
    policy = lambda z, *, generator: torch.randn((len(z), 1), generator=generator).tanh()
    callbacks = MPPIModelCallbacks(
        action_dim=1, dynamics=lambda z, a: z, reward=lambda z, a: a,
        policy=policy, terminal_q=lambda z, a, **kwargs: a,
    )
    options = dict(horizon=2, iterations=2, num_samples=4, num_pi_trajs=1,
                   num_elites=2, temperature=1., min_std=.01, max_std=1.,
                   discount=.9, q_reduction="mean_all", eval_mode=True)
    legacy_rng, explicit_rng = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    legacy = mppi_plan(torch.zeros(1, 2), callbacks=callbacks, generator=legacy_rng, **options)
    explicit = mppi_plan(torch.zeros(1, 2), callbacks=replace(callbacks, terminal_policy=policy),
                         generator=explicit_rng, **options)
    torch.testing.assert_close(legacy.action, explicit.action, rtol=0, atol=0)
    torch.testing.assert_close(legacy.next_mean, explicit.next_mean, rtol=0, atol=0)
    torch.testing.assert_close(legacy_rng.get_state(), explicit_rng.get_state(), rtol=0, atol=0)


@pytest.mark.parametrize("horizon_actor", ["sac", "return_actor"])
@pytest.mark.parametrize("horizon_critic", ["sac", "aux_return"])
def test_engine_mppi_routes_selected_pair_without_mutating_outer(horizon_actor, horizon_critic, monkeypatch):
    with _prepared(inner_operator="mppi", inner_mppi_iterations=1,
                   inner_mppi_num_pi_trajs=1,
                   inner_horizon_actor_source=horizon_actor,
                   inner_horizon_critic_source=horizon_critic) as (holder, engine):
        policies, critics = [], []
        pi_action, q = engine.model.pi_action, engine.model.Q
        def sample(z, **kwargs):
            policies.append(kwargs.get("policy"))
            return pi_action(z, **kwargs)
        def value(z, action, **kwargs):
            critics.append(kwargs.get("qs"))
            return q(z, action, **kwargs)
        monkeypatch.setattr(engine.model, "pi_action", sample)
        monkeypatch.setattr(engine.model, "Q", value)
        before = _clone_tree(engine.model.state_dict())
        rng = torch.random.get_rng_state().clone()
        holder.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert policies[:2] == [engine.model._return_pi, engine.model._return_pi]
        assert policies[2:] == [engine._horizon_actor]
        assert critics == [engine._horizon_critic]
        assert engine.state.actor_steps == engine.state.critic_steps == 0
        _assert_tree_equal(engine.model.state_dict(), before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
