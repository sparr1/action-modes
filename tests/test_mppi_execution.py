"""MPPI execution matches TD-MPC2 without changing historical mean evaluation."""

import json
from pathlib import Path

import pytest
import torch

from RL.tdmpc2_core.mppi import MPPIModelCallbacks, mppi_plan
from tests.test_ambi_inner_decoupling import _model
from tests.test_ambi_root_local_sac import _build_cfg


def _fixed_population(generator, *, eval_mode, eval_action="mean"):
    # All candidates are specified policy trajectories: planning consumes no
    # randomness, so the execution RNG contract can be checked independently.
    callbacks = MPPIModelCallbacks(
        action_dim=1,
        dynamics=lambda z, a: z,
        reward=lambda z, a: a,
        policy=lambda z, **kw: z.new_tensor([[-.6], [.1], [.8]]),
        terminal_q=lambda z, a, **kw: torch.zeros_like(a),
    )
    return mppi_plan(
        torch.zeros(1, 2), callbacks=callbacks, horizon=1, iterations=1,
        num_samples=3, num_elites=3, num_pi_trajs=3, temperature=.5,
        min_std=.05, max_std=2., discount=.99, q_reduction="mean_pair",
        generator=generator, eval_mode=eval_mode, eval_action=eval_action,
    )


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_tdmpc2_eval_samples_weighted_elite_without_gaussian_noise(seed):
    generator = torch.Generator().manual_seed(seed)
    expected_rng = torch.Generator().manual_seed(seed)
    elites = torch.tensor([.8, .1, -.6])
    weights = torch.softmax(.5 * elites, dim=0)
    expected = elites[torch.multinomial(weights, 1, generator=expected_rng)]
    global_rng = torch.random.get_rng_state().clone()

    result = _fixed_population(generator, eval_mode=True, eval_action="weighted_elite")

    torch.testing.assert_close(result.action, expected, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), expected_rng.get_state(), rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    assert not torch.allclose(result.action, result.next_mean[0])


def test_historical_mean_eval_keeps_execution_rng_untouched():
    rng = torch.Generator().manual_seed(17)
    before = rng.get_state().clone()
    result = _fixed_population(rng, eval_mode=True)
    torch.testing.assert_close(result.action, result.next_mean[0], rtol=0, atol=0)
    torch.testing.assert_close(rng.get_state(), before, rtol=0, atol=0)


def test_training_uses_weighted_elite_plus_noise_independent_of_eval_mode_setting():
    first_rng = torch.Generator().manual_seed(17)
    second_rng = torch.Generator().manual_seed(17)
    expected_rng = torch.Generator().manual_seed(17)
    elites = torch.tensor([.8, .1, -.6])
    weights = torch.softmax(.5 * elites, dim=0)
    mean = (weights * elites).sum()
    std = (weights * (elites - mean).square()).sum().sqrt().clamp(.05, 2.)
    selected = elites[torch.multinomial(weights, 1, generator=expected_rng)]
    expected = (selected + std * torch.randn(1, generator=expected_rng)).clamp(-1, 1)
    first = _fixed_population(first_rng, eval_mode=False)
    second = _fixed_population(second_rng, eval_mode=False, eval_action="weighted_elite")
    torch.testing.assert_close(first.action, expected)
    torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)
    torch.testing.assert_close(first_rng.get_state(), second_rng.get_state(), rtol=0, atol=0)
    torch.testing.assert_close(first_rng.get_state(), expected_rng.get_state(), rtol=0, atol=0)


def test_invalid_eval_selection_is_rejected_at_config_and_helper_boundaries():
    with pytest.raises(ValueError, match="inner_mppi_eval_action"):
        _build_cfg(inner_mppi_eval_action="best")
    with pytest.raises(ValueError, match="eval_action"):
        _fixed_population(torch.Generator(), eval_mode=True, eval_action="best")


def test_frozen_protocol_records_stochastic_elite_selection_truthfully():
    from utils.ambi_benchmark import protocol_for
    resolved = {"algorithm_config": {"alg_params": {"inner_operator": "mppi"}},
                "environment": {"id": "test"}}
    historical = protocol_for(resolved, 55, 500)
    assert historical["action_rule"] == "tanh_mean"
    resolved["algorithm_config"]["alg_params"]["inner_mppi_eval_action"] = "mean"
    assert protocol_for(resolved, 55, 500) == historical
    resolved["algorithm_config"]["alg_params"]["inner_mppi_eval_action"] = "weighted_elite"
    assert protocol_for(resolved, 55, 500)["action_rule"] == "mppi_weighted_elite"


def test_engine_routes_tdmpc2_eval_to_auxiliary_online_critic(monkeypatch):
    from RL.tdmpc2_core import mppi
    original = mppi.mppi_plan
    calls = []
    learner = _model(
        inner_operator="mppi", aux_return_mode="sac",
        inner_actor_source="sac", inner_horizon_actor_source="sac",
        inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
        inner_mppi_eval_action="weighted_elite", mppi_terminal_q_reduction="mean_pair",
        inner_critic_updates_per_action=0, inner_actor_updates_per_action=0,
    )
    agent = learner.agent
    engine = agent.inner_engine
    model_before = {k: v.clone() for k, v in agent.model.state_dict().items()}
    q_calls = []
    original_q = agent.model.Q

    def record_q(*args, **kwargs):
        q_calls.append(kwargs)
        return original_q(*args, **kwargs)

    def record_plan(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(mppi, "mppi_plan", record_plan)
    monkeypatch.setattr(agent.model, "Q", record_q)
    try:
        action = agent.act(torch.zeros(3), t0=True, eval_mode=True, collect_diagnostics=False)
        assert calls[0]["eval_action"] == "weighted_elite" and calls[0]["eval_mode"]
        assert q_calls and all(c["qs"] is engine._horizon_critic for c in q_calls)
        assert all(c["reduction"] == "mean_pair" and not c.get("target", False) for c in q_calls)
        assert torch.isfinite(action).all()
        assert engine.state.actor_steps == engine.state.critic_steps == 0
        for key, value in agent.model.state_dict().items():
            torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)
    finally:
        learner.env.close()


def test_single_h3_recipe_preserves_outer_training_and_pins_planner_budget():
    root = Path(__file__).resolve().parents[1]
    read = lambda path: json.loads((root / path).read_text())
    baseline = read("configs/dmcontrol/algs/ambi_full_h3_j8_fresh.json")
    recipe = read("configs/dmcontrol/algs/ambi_full_h3_mppi_aux_return.json")
    manifest = read("configs/dmcontrol/experiments/ambi_mppi_training.json")
    assert manifest["configs"] == ["ambi_full_h3_mppi_aux_return", "ambi_full_h3_mppi_soft"]
    assert "alg_params" not in manifest["overrides_alg"]
    for key in ("seed", "total_steps", "checkpoint_every", "save_strat"):
        assert recipe[key] == baseline[key]
    def outer(params):
        settings = {k: v for k, v in params.items()
                    if not k.startswith(("inner_", "wandb_", "mppi_"))}
        # The launched J6/J8 release predates joint ensemble execution.
        settings.setdefault("aux_return_joint_vmap", False)
        return settings
    assert outer(recipe["alg_params"]) == outer(baseline["alg_params"])
    cfg = _build_cfg(**recipe["alg_params"])
    assert cfg.inner_operator == "mppi"
    assert cfg.aux_return_joint_vmap is False
    assert cfg.inner_rollout_horizon == cfg.train_unroll_horizon == 3
    assert cfg.inner_mppi_iterations == 8 and cfg.inner_mppi_num_samples == 512
    assert cfg.inner_mppi_num_elites == 64 and cfg.inner_mppi_num_pi_trajs == 24
    assert cfg.inner_model_step_budget == 12336
    assert cfg.inner_mppi_eval_action == "weighted_elite"
    assert cfg.inner_mppi_warm_start_scope == "episode"
    assert cfg.inner_horizon_critic_source == "aux_return"
    assert cfg.inner_horizon_actor_source == cfg.inner_actor_source == "sac"
    assert cfg.mppi_terminal_q_reduction == "mean_pair"
    assert cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 0
