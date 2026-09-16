"""Fresh-inner diagnostics use the same sources and boundary as control."""

import pytest
import torch

from evaluate_ambi_checkpoint import _outer_state_digest
from RL.tdmpc2_core.common.control_sources import actor_module, critic_module
from tests.test_ambi_value_equivalence_diagnostics import _inputs, _model


@pytest.mark.parametrize("horizon", [1, 2, 4])
@pytest.mark.parametrize("tail_actor,tail_critic", [
    ("sac", "aux_return"), ("return_actor", "sac"),
])
def test_diagnostic_values_follow_sources_scale_and_horizon(
    monkeypatch, horizon, tail_actor, tail_critic,
):
    model = _model(
        horizon=3, aux_return_mode="return_actor", inner_rollout_horizon=horizon,
        inner_actor_source="return_actor", inner_critic_source="aux_return",
        inner_horizon_actor_source=tail_actor, inner_horizon_critic_source=tail_critic,
        ent_coef=.2, aux_return_ent_coef=.07,
        aux_return_sac_actor_loss_scale_mode="tdmpc2_percentile_range",
    )
    try:
        agent = model.agent
        agent.aux_return.actor_loss_scale.fill_(4.)
        assert agent._initial_inner_diagnostic_entropy_coefficient().item() == pytest.approx(.28)
        assert agent._value_equivalence_reference_critic() is agent.model._aux_return_Qs

        def policy(z, *, policy, **kwargs):
            action_value = 1. if policy is agent.model._pi else 2.
            return z.new_full((*z.shape[:-1], 1), action_value), {"log_prob": -z[..., :1]}

        def value(critic, z, action, reduction, pair_indices):
            gain = 3. if critic is agent.model._Qs else 7.
            return (gain + action) * z[..., :1]

        monkeypatch.setattr(agent.model, "pi", policy)
        monkeypatch.setattr(agent.model, "decode_reward", lambda pred: pred)
        monkeypatch.setattr(agent, "_value_equivalence_q", value)
        inputs = _inputs(agent, horizon=3)
        inputs[0][1:, :, 0] = 1.
        before_rng = torch.get_rng_state().clone()
        before = _outer_state_digest(model)
        metrics = agent._value_equivalence_diagnostics(*inputs, diagnostic_update=1)
        for depth in range(1, 4):
            expected = 9. + .28
            if depth >= horizon:
                expected = (1. if tail_actor == "sac" else 2.) + (
                    3. if tail_critic == "sac" else 7.)
            assert metrics[f"ve_prior_target_bias_depth_{depth}"].item() == pytest.approx(
                agent.discount * expected,
            )
        assert _outer_state_digest(model) == before
        assert torch.equal(torch.get_rng_state(), before_rng)
    finally:
        model.env.close()


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_real_diagnostic_is_observational_with_selected_sources(mode):
    actor = "return_actor" if mode == "return_actor" else "sac"
    model = _model(
        horizon=2, aux_return_mode=mode, inner_rollout_horizon=2,
        inner_actor_source=actor, inner_critic_source="aux_return",
        inner_horizon_actor_source=actor, inner_horizon_critic_source="aux_return",
        inner_critic_target_initialization="outer_target", inner_entropy_enabled=False,
    )
    try:
        agent = model.agent
        assert agent._initial_inner_diagnostic_alpha().item() == 0.
        assert agent._value_equivalence_reference_critic() is critic_module(agent, "aux_return", target=True)
        inputs = list(_inputs(agent, horizon=2))
        inputs[1] = torch.zeros(2, 2, agent.cfg.num_bins)
        actor_module(agent, actor).train()
        before_modes = tuple(module.training for module in agent.model.modules())
        before_rng = torch.get_rng_state().clone()
        before = _outer_state_digest(model)
        metrics = agent._value_equivalence_diagnostics(*inputs, diagnostic_update=1)
        assert all(torch.isfinite(value).all() for value in metrics.values())
        assert _outer_state_digest(model) == before
        assert tuple(module.training for module in agent.model.modules()) == before_modes
        assert torch.equal(torch.get_rng_state(), before_rng)
    finally:
        model.env.close()
