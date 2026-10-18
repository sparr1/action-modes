"""The two MPPI controls differ only in terminal critic and auxiliary training."""

import json
from pathlib import Path

import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_finite, _params
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params


ROOT = Path(__file__).resolve().parents[1]


def _recipe(name):
    return json.loads((ROOT / f"configs/dmcontrol/algs/{name}.json").read_text())


def test_mppi_critic_comparison_preserves_primary_training_and_planner():
    aux = _recipe("ambi_full_h3_mppi_aux_return")
    soft = _recipe("ambi_full_h3_mppi_soft")
    assert {k: v for k, v in aux.items() if k != "alg_params"} == {
        k: v for k, v in soft.items() if k != "alg_params"
    }
    changed = {k for k in aux["alg_params"]
               if aux["alg_params"][k] != soft["alg_params"][k]}
    assert changed == {
        "aux_return_mode", "inner_critic_source", "inner_horizon_critic_source",
        "wandb_tags", "wandb_run_name",
    }
    assert set(aux["alg_params"]) == set(soft["alg_params"])
    cfg = _build_cfg(**soft["alg_params"])
    assert cfg.aux_return_mode == "off"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "sac"
    assert cfg.outer_critic_target == "entropy_augmented"
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_mppi_iterations == 8 and cfg.inner_mppi_num_samples == 512
    assert cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 0
    assert cfg.inner_mppi_eval_action == "weighted_elite"
    assert cfg.mppi_terminal_q_reduction == "mean_pair"


@pytest.mark.parametrize("variant", ["aux_return", "soft"])
def test_training_recipe_allocates_and_trains_only_requested_critic(variant, monkeypatch):
    recipe = _recipe(f"ambi_full_h3_mppi_{variant}")["alg_params"]
    # Keep the scientific source selectors and learner losses, with a small CPU
    # model/population for the allocation, routing and real-update checks.
    params = {k: v for k, v in recipe.items() if not k.startswith("wandb_")}
    params.update(
        device="cpu", model_size=None, compile=False, compile_strict=False,
        wandb=False, batch_size=2, seed_steps=4, pretrain_steps=1,
        inner_model_step_budget=28, inner_mppi_iterations=2,
        inner_mppi_num_elites=2, inner_mppi_num_pi_trajs=2,
        inner_critic_updates_per_action=0, inner_actor_updates_per_action=0,
    )
    params = _params(**params)
    # The shared SAC fixture carries an old schedule alias; MPPI uses its own
    # explicit iteration setting, and does not use LoRA adaptation.
    params.pop("inner_rounds")
    params.pop("inner_critic_lora_rank")
    learner = _model_from_params(params)
    agent = learner.agent
    expected_aux = variant == "aux_return"
    try:
        assert (agent.aux_return is not None) is expected_aux
        assert hasattr(agent.model, "_aux_return_Qs") is expected_aux
        assert hasattr(agent.model, "_target_aux_return_Qs") is expected_aux
        assert any("aux_return" in k for k in agent.model.state_dict()) is expected_aux
        expected_q = agent.model._aux_return_Qs if expected_aux else agent.model._Qs
        calls = []
        original_q = agent.model.Q

        def record_q(*args, **kwargs):
            calls.append(kwargs)
            return original_q(*args, **kwargs)

        monkeypatch.setattr(agent.model, "Q", record_q)
        action = agent.act(torch.zeros(3), t0=True, eval_mode=False,
                           collect_diagnostics=False)
        assert torch.isfinite(action).all()
        assert calls and all(call.get("qs", agent.model._Qs) is expected_q for call in calls)
        assert all(call["reduction"] == "mean_pair" and not call.get("target", False)
                   for call in calls)
        monkeypatch.setattr(agent.model, "Q", original_q)
        horizon, batch = learner.cfg.train_unroll_horizon, learner.cfg.batch_size
        obs = torch.randn(horizon + 1, batch, 3)
        actions = torch.randn(horizon, batch, 1).tanh()
        rewards = torch.randn(horizon, batch, 1)
        before = agent.model._Qs[0][-1].weight.detach().clone()
        info = agent._update(obs, actions, rewards, torch.zeros_like(rewards))
        _assert_finite(info)
        assert ("aux_return_critic_loss" in info) is expected_aux
        assert not torch.equal(agent.model._Qs[0][-1].weight, before)
    finally:
        learner.env.close()
