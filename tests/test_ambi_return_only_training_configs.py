"""Reward-return critics retain learned entropy regularization in the actors."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_finite
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params
from tests.test_pave_integration import _batch


ROOT = Path(__file__).resolve().parents[1]
GROUP = "ambi-return-only-training-20261019"
CONFIGS = {"mppi": "ambi_full_h3_mppi_reward_only", "sac": "ambi_full_h3_j8_k3_reward_only"}


def recipe(method):
    return json.loads((ROOT / f"configs/dmcontrol/algs/{CONFIGS[method]}.json").read_text())


@pytest.mark.parametrize("method", CONFIGS)
def test_recipe_changes_only_requested_objective_sources_and_publication(method):
    base_name = "ambi_full_h3_mppi_soft" if method == "mppi" else "ambi_full_h3_j8_k3_fresh"
    base = json.loads((ROOT / f"configs/dmcontrol/algs/{base_name}.json").read_text())
    current = recipe(method)
    expected = deepcopy(base)
    p = expected["alg_params"]
    p.update(outer_critic_target="reward_only", inner_sac_critic_target="reward_only")
    if method == "mppi":
        p.update(inner_solve_interval=1)
    else:
        p.update(aux_return_mode="off", inner_critic_source="sac", inner_horizon_critic_source="sac")
    for key in ("wandb_tags", "wandb_group", "wandb_run_name"):
        p[key] = current["alg_params"][key]
    assert current == expected
    cfg = _build_cfg(**current["alg_params"])
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert cfg.aux_return_mode == "off" and cfg.aux_return_joint_vmap is False
    assert cfg.critic_value_mode == "single" and cfg.q_representation == "distributional"
    assert (cfg.num_q, cfg.q_num_bins, cfg.q_vmin, cfg.q_vmax, cfg.q_pair_size) == (5, 101, -10, 10, 2)
    assert cfg.outer_q_target_reduction == "min_pair" and cfg.outer_q_actor_reduction == "mean_pair"
    assert cfg.ent_coef == "auto_1.0" and cfg.target_entropy == -10.5
    assert cfg.outer_actor_entropy_mode == "squashed" and cfg.sac_actor_loss_scale_mode == "none"
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "sac"
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
    assert cfg.inner_critic_bernoulli_p == 0
    assert cfg.inner_rollout_horizon == cfg.train_unroll_horizon == cfg.outer_planning_horizon == 3
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1
    if method == "mppi":
        assert cfg.inner_operator == "mppi" and cfg.inner_solve_interval == 1
        assert (cfg.inner_mppi_iterations, cfg.inner_mppi_num_samples,
                cfg.inner_mppi_num_elites, cfg.inner_mppi_num_pi_trajs) == (8, 512, 64, 24)
        assert cfg.inner_model_step_budget == 12336
        assert (cfg.inner_mppi_temperature, cfg.inner_mppi_min_std, cfg.inner_mppi_max_std) == (.5, .05, 2.)
        assert cfg.inner_mppi_eval_action == "weighted_elite" and cfg.inner_mppi_warm_start_scope == "episode"
        assert cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 0
    else:
        assert cfg.inner_operator == "sac" and cfg.inner_solve_interval == 3
        assert cfg.inner_hold_encoder == "solve_snapshot"
        assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (8, 128, 256, 3072)
        assert (cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (16, 4)
        assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
        assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "prior"
        assert cfg.inner_temperature_mode == "auto" and cfg.inner_entropy_enabled
        assert cfg.inner_temperature_initialization == cfg.inner_target_entropy == "inherit_outer"
        assert cfg.inner_execution_action == "policy_sample" and cfg.inner_finite_horizon
    assert cfg.mppi_terminal_q_reduction == "mean_pair"


@pytest.mark.parametrize("method", CONFIGS)
def test_single_cell_manifest_and_truthful_wandb_identity(method):
    manifest = json.loads((ROOT / f"configs/dmcontrol/experiments/ambi_return_only_{method}_training.json").read_text())
    run = recipe(method)
    assert manifest["configs"] == [CONFIGS[method]] and manifest["trials"] == 1
    assert manifest["overrides_alg"]["total_steps"] == run["total_steps"] == 500000
    assert manifest["overrides_alg"]["seed"] == run["seed"] == 55
    assert manifest["checkpoint_every"] == run["checkpoint_every"] == 25000
    assert manifest["save_strat"] == run["save_strat"] == "all"
    assert "alg_params" not in manifest["overrides_alg"]
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    p = run["alg_params"]
    assert p["wandb_project"] == "ambi" and p["wandb_group"] == GROUP
    assert "reward-only Q" in p["wandb_run_name"] and "H3" in p["wandb_run_name"]
    assert {"no-auxiliary-critic", "reward-only-primary-critic", "automatic-actor-entropy", GROUP} <= set(p["wandb_tags"])
    assert not {"soft-Q-bootstrap", "soft-soft", "aux-return-representation-shared"} & set(p["wandb_tags"])
    assert p["eval_freq"] is None and p["eval_inner_comparison"] is False


@pytest.mark.parametrize("method", CONFIGS)
def test_no_auxiliary_allocation_reward_only_bootstrap_and_live_temperature(method, monkeypatch):
    p = recipe(method)["alg_params"]
    p.update(device="cpu", compile=False, compile_strict=False, wandb=False,
             model_size=None, enc_dim=32, mlp_dim=32, latent_dim=16,
             num_enc_layers=2, simnorm_dim=8, batch_size=2,
             seed_steps=4, pretrain_steps=1, buffer_size=32)
    if method == "mppi":
        p.update(inner_model_step_budget=28, inner_mppi_iterations=2,
                 inner_mppi_num_elites=2, inner_mppi_num_pi_trajs=2)
    else:
        p.update(inner_rounds=1, inner_rollouts_per_round=4, inner_batch_size=4,
                 inner_replay_capacity=12, inner_critic_updates_per_round=1,
                 inner_actor_updates_per_round=1)
    learner = _model_from_params(p)
    agent, cfg = learner.agent, learner.cfg
    try:
        assert agent.aux_return is None
        assert not hasattr(agent.model, "_aux_return_Qs")
        assert not hasattr(agent.model, "_target_aux_return_Qs")
        assert not any("aux_return" in key for key in agent.model.state_dict())
        assert agent.inner_engine._critic_base is agent.inner_engine._horizon_critic is agent.model._Qs
        assert agent.ent_coef_optim is not None and agent.log_ent_coef.requires_grad
        assert agent.alpha.item() == pytest.approx(1.) and agent.target_entropy == -10.5
        calls = []

        def policy(z, **kwargs):
            return z.new_zeros(*z.shape[:-1], 1), {
                "log_prob": z.new_full((*z.shape[:-1], 1), -999.),
                "entropy": z.new_full((*z.shape[:-1], 1), 999.),
            }

        def q(z, action, **kwargs):
            calls.append(kwargs)
            return z.new_full((*z.shape[:-1], 1), 7.)

        with monkeypatch.context() as patch:
            patch.setattr(agent.model, "pi", policy)
            patch.setattr(agent.model, "Q", q)
            z = torch.zeros(2, cfg.latent_dim)
            target = agent._soft_td_target(z, torch.full((2, 1), 2.), torch.tensor([[0.], [1.]]))
            torch.testing.assert_close(target, torch.tensor([[2. + agent.discount * 7.], [2.]]))
            tail = agent.inner_engine._prior_bootstrap(z, torch.zeros(2, 1))
            torch.testing.assert_close(tail, torch.full((2, 1), 7.))
        assert calls == [{"target": True, "reduction": "min_pair"}, {"reduction": "mean_pair"}]
        action = agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        assert torch.isfinite(action).all()
        if method == "sac":
            assert agent.last_inner_metrics["inner_solve_performed"] == 1
            assert agent.last_inner_metrics["inner_temperature_optimizer_steps"] == 1
            agent.act(torch.ones(3), collect_diagnostics=False)
            assert agent.last_inner_metrics["inner_policy_held"] == 1
            assert agent.last_inner_metrics["inner_critic_optimizer_steps"] == 0
        before_q = agent.model._Qs[0][-1].weight.detach().clone()
        before_alpha = agent.log_ent_coef.detach().clone()
        info = agent._update(*_batch(agent))
        _assert_finite(info)
        assert not any(key.startswith("aux_return_") for key in info)
        assert not torch.equal(agent.model._Qs[0][-1].weight, before_q)
        assert not torch.equal(agent.log_ent_coef, before_alpha)
    finally:
        learner.close()
        learner.env.close()
