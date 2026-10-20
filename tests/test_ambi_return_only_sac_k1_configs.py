"""The K1 control changes the cadence and training budget of the K3 reference."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "ambi_full_h3_j8_k1_reward_only"


def recipe():
    return json.loads((ROOT / f"configs/dmcontrol/algs/{CONFIG}.json").read_text())


def test_k1_changes_only_solve_cadence_budget_and_publication():
    reference = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_h3_j8_k3_reward_only.json").read_text())
    assert reference["alg_params"]["inner_solve_interval"] == 3
    assert reference["total_steps"] == 500000
    current = recipe()
    expected = deepcopy(reference)
    expected["alg_params"]["inner_solve_interval"] = 1
    expected["total_steps"] = 1000000
    for key in ("wandb_run_name", "wandb_tags"):
        expected["alg_params"][key] = current["alg_params"][key]
    assert current == expected

    cfg = _build_cfg(**current["alg_params"])
    assert cfg.inner_solve_interval == 1 and not hasattr(cfg, "inner_hold_encoder")
    assert cfg.inner_operator == "sac"
    assert cfg.inner_rollout_horizon == cfg.train_unroll_horizon == cfg.outer_planning_horizon == 3
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (8, 128, 256, 3072)
    assert (cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (16, 4)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action, cfg.inner_temperature_updates_per_action) == (128, 32, 32)
    assert cfg.inner_model_step_budget == 3072
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "prior"
    assert cfg.inner_critic_bernoulli_p == 0.
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0.
    assert cfg.inner_execution_action == cfg.inner_behavior_action == "policy_sample"
    assert cfg.inner_actor_source == cfg.inner_critic_source == "sac"
    assert cfg.inner_horizon_actor_source == cfg.inner_horizon_critic_source == "sac"
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert cfg.aux_return_mode == "off" and cfg.aux_return_joint_vmap is False
    assert cfg.ent_coef == "auto_1.0" and cfg.target_entropy == -10.5
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_entropy_enabled
    assert cfg.inner_temperature_initialization == cfg.inner_target_entropy == "inherit_outer"
    assert cfg.model_size == 5 and cfg.num_q == 5 and cfg.q_representation == "distributional"
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1


def test_single_cell_manifest_and_labels_match_requested_million_decisions():
    current = recipe()
    manifest = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_return_only_sac_k1_training.json").read_text())
    expected = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_return_only_sac_training.json").read_text())
    expected["configs"] = [CONFIG]
    expected["overrides_alg"]["total_steps"] = 1000000
    expected["study_note"] = manifest["study_note"]
    assert manifest == expected
    assert manifest["trials"] == 1 and "alg_params" not in manifest["overrides_alg"]
    assert manifest["overrides_alg"]["total_steps"] == current["total_steps"] == 1000000
    assert manifest["overrides_alg"]["seed"] == current["seed"] == 55
    assert manifest["checkpoint_every"] == current["checkpoint_every"] == 25000
    assert current["total_steps"] // current["checkpoint_every"] == 40
    assert manifest["save_strat"] == current["save_strat"] == "all"
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    params = current["alg_params"]
    assert params["wandb_project"] == "ambi"
    assert params["wandb_group"] == "ambi-return-only-training-20261019"
    assert params["wandb_run_name"] == "H3 | Fresh SAC | J8 | K1 | reward-only Q | 1M decisions | seed55"
    tags = set(params["wandb_tags"])
    assert {"K1", "solve-every-decision", "H3", "J8", "fresh", "no-auxiliary-critic",
            "reward-only-primary-critic", "automatic-actor-entropy", "target-entropy-10p5",
            "1m-decisions", "2m-environment-steps"} <= tags
    assert not {"K3", "held-feedback-policy", "solve-snapshot-encoder", "500k-decisions",
                "1m-environment-steps"} & tags


def test_k1_runs_fresh_inner_learning_at_each_decision(monkeypatch):
    params = recipe()["alg_params"]
    params.update(device="cpu", compile=False, compile_strict=False, wandb=False,
                  model_size=None, enc_dim=32, mlp_dim=32, latent_dim=16,
                  num_enc_layers=2, simnorm_dim=8, batch_size=2,
                  seed_steps=4, pretrain_steps=1, buffer_size=32,
                  inner_rounds=1, inner_rollouts_per_round=4, inner_batch_size=4,
                  inner_replay_capacity=12, inner_critic_updates_per_round=1,
                  inner_actor_updates_per_round=1)
    learner = _model_from_params(params)
    agent, engine = learner.agent, learner.agent.inner_engine
    initializations = []
    prepare = engine._prepare_workspace

    def checked_prepare(*args, **kwargs):
        prepare(*args, **kwargs)
        for current, prior in zip(engine.state.actor.parameters(), agent.model._pi.parameters()):
            torch.testing.assert_close(current, prior, rtol=0, atol=0)
        for current, prior in zip(engine.state.critic.parameters(), agent.model._Qs.parameters()):
            torch.testing.assert_close(current, prior, rtol=0, atol=0)
        torch.testing.assert_close(engine.alpha.reshape(()), agent.alpha.reshape(()))
        assert engine.state.replay.size == 0
        assert engine.state.critic_steps == engine.state.actor_steps == engine.state.temperature_steps == 0
        initializations.append(True)

    monkeypatch.setattr(engine, "_prepare_workspace", checked_prepare)
    try:
        assert agent.aux_return is None
        assert not any("aux_return" in key for key in agent.model.state_dict())
        for decision in range(3):
            # The next solve must start from the current prior, even after the
            # outer learner changes it between real decisions.
            if decision:
                with torch.no_grad():
                    next(agent.model._pi.parameters()).add_(.01)
                    agent.log_ent_coef.add_(.02)
            action = agent.act(torch.full((3,), float(decision)),
                               t0=decision == 0, eval_mode=False, collect_diagnostics=False)
            assert torch.isfinite(action).all()
            metrics = agent.last_inner_metrics
            assert metrics["inner_active"] == 1 and metrics["inner_model_steps"] == 12
            assert metrics["inner_critic_optimizer_steps"] == 1
            assert metrics["inner_actor_optimizer_steps"] == 1
            assert metrics["inner_temperature_optimizer_steps"] == 1
            assert metrics.get("inner_policy_held", 0) == 0
            assert engine._held_actor is None and engine._held_encoder is None
        assert len(initializations) == 3
    finally:
        learner.close()
        learner.env.close()
