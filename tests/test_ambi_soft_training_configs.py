"""Keep the soft-only inner-SAC control matched to the launched H3 J8 recipe."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from tests.test_ambi_inner_decoupling import _assert_finite
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads((ROOT / path).read_text())


def _recipe():
    return _read("configs/dmcontrol/algs/ambi_full_h3_j8_soft.json")


def test_soft_only_changes_critic_sources_target_and_auxiliary_learner():
    baseline = _read("configs/dmcontrol/algs/ambi_full_h3_j8_fresh.json")
    recipe = _recipe()
    assert {k: v for k, v in recipe.items() if k != "alg_params"} == {
        k: v for k, v in baseline.items() if k != "alg_params"
    }
    expected = deepcopy(baseline["alg_params"])
    expected.update(
        aux_return_mode="off", inner_critic_source="sac",
        inner_horizon_critic_source="sac",
        inner_sac_critic_target="entropy_augmented", aux_return_joint_vmap=False,
    )
    for key in ("wandb_tags", "wandb_group", "wandb_run_name"):
        expected[key] = recipe["alg_params"][key]
    assert recipe["alg_params"] == expected
    cfg = _build_cfg(**recipe["alg_params"])
    assert cfg.aux_return_mode == "off" and cfg.aux_return_joint_vmap is False
    assert cfg.inner_operator == "sac" and cfg.inner_rollout_horizon == 3
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_batch_size) == (8, 128, 256)
    assert (cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (16, 4)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
    assert cfg.inner_replay_capacity == 3072 and cfg.inner_finite_horizon
    assert cfg.inner_sac_critic_target == cfg.outer_critic_target == "entropy_augmented"
    assert cfg.mppi_terminal_q_reduction == "mean_pair"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "sac"
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.inner_critic_bernoulli_p == 0.
    assert cfg.inner_actor_scope == cfg.inner_critic_scope == "action"
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_entropy_enabled
    assert cfg.critic_ensemble_backend == "vmap" and cfg.compile and cfg.compile_strict
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1


def test_soft_training_manifest_has_one_cell_and_full_checkpoint_bank():
    manifest = _read("configs/dmcontrol/experiments/ambi_soft_training.json")
    recipe = _recipe()
    assert manifest["configs"] == ["ambi_full_h3_j8_soft"]
    assert manifest["trials"] == 1
    assert "alg_params" not in manifest["overrides_alg"]
    assert manifest["overrides_alg"]["total_steps"] == recipe["total_steps"] == 500000
    assert manifest["overrides_alg"]["seed"] == recipe["seed"] == 55
    assert manifest["checkpoint_every"] == recipe["checkpoint_every"] == 25000
    assert manifest["save_strat"] == recipe["save_strat"] == "all"
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    params = recipe["alg_params"]
    assert params["wandb_run_name"] == "H3 | Fresh | J8 | soft-only | seed55"
    assert params["wandb_project"] == "ambi"
    assert params["wandb_group"] == "ambi-soft-training-20261017"
    assert {"soft-soft", "no-auxiliary-critic"} <= set(params["wandb_tags"])
    assert not {"return-return", "aux-return-representation-shared"} & set(params["wandb_tags"])


def test_soft_only_inner_and_outer_updates_without_auxiliary_allocation(monkeypatch):
    params = _recipe()["alg_params"]
    params.update(
        device="cpu", compile=False, compile_strict=False, wandb=False,
        model_size=None, enc_dim=32, mlp_dim=32, latent_dim=16,
        num_enc_layers=2, simnorm_dim=8, batch_size=2,
        seed_steps=4, pretrain_steps=1, buffer_size=32,
        inner_rounds=1, inner_rollouts_per_round=4, inner_batch_size=4,
        inner_replay_capacity=12, inner_critic_updates_per_round=1,
        inner_actor_updates_per_round=1,
    )
    learner = _model_from_params(params)
    agent = learner.agent
    try:
        assert agent.aux_return is None
        assert not hasattr(agent.model, "_aux_return_Qs")
        assert not hasattr(agent.model, "_target_aux_return_Qs")
        assert not any("aux_return" in key for key in agent.model.state_dict())
        assert agent.inner_engine._critic_base is agent.model._Qs
        assert agent.inner_engine._horizon_critic is agent.model._Qs
        # The retained finite-horizon boundary uses the online primary Q,
        # averaged over a sampled pair, without another terminal entropy bonus.
        calls = []

        def terminal_policy(z, **kwargs):
            assert "policy" not in kwargs
            return z.new_zeros(z.shape[0], 1), {
                "log_prob": z.new_full((z.shape[0], 1), -99.),
                "entropy": z.new_full((z.shape[0], 1), 99.),
            }

        def terminal_q(z, action, **kwargs):
            calls.append(kwargs)
            return z.new_full((z.shape[0], 1), 7.)

        with monkeypatch.context() as patch:
            patch.setattr(agent.model, "pi", terminal_policy)
            patch.setattr(agent.model, "Q", terminal_q)
            value = agent.inner_engine._prior_bootstrap(
                torch.zeros(2, learner.cfg.latent_dim), torch.zeros(2, 1),
            )
        assert calls == [{"reduction": "mean_pair"}]
        torch.testing.assert_close(value, torch.full((2, 1), 7.))
        action = agent.act(torch.zeros(3), t0=True, eval_mode=False,
                           collect_diagnostics=False)
        assert torch.isfinite(action).all()
        assert agent.last_inner_metrics["inner_critic_optimizer_steps"] == 1
        assert agent.last_inner_metrics["inner_actor_optimizer_steps"] == 1
        horizon, batch = learner.cfg.train_unroll_horizon, learner.cfg.batch_size
        obs = torch.randn(horizon + 1, batch, 3)
        actions = torch.randn(horizon, batch, 1).tanh()
        rewards = torch.randn(horizon, batch, 1)
        before = agent.model._Qs[0][-1].weight.detach().clone()
        metrics = agent._update(obs, actions, rewards, torch.zeros_like(rewards))
        _assert_finite(metrics)
        assert "aux_return_critic_loss" not in metrics
        assert not torch.equal(agent.model._Qs[0][-1].weight, before)
    finally:
        learner.env.close()
