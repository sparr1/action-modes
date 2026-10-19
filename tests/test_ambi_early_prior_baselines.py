"""Matched early-checkpoint priors execute saved actors without imagined work."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_early_h3_j8 import _tiny_checkpoint
from tests.test_ambi_root_local_sac import _build_cfg
from utils.ambi_benchmark import frozen_execution_action, protocol_for, solver_seed
from utils.ambi_research import load_preset_matrix, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_early_prior_baselines.json"


def _resolve(config, mode="mean", *, matrix=MATRIX):
    context = SimpleNamespace(
        source="checkpoint.metadata.json", trial_run_params=config,
        experiment_params={"env_params": {"domain_name": "humanoid", "task_name": "walk"}},
    )
    prefix = "early_prior" if matrix == MATRIX else "early_h3_j8"
    return resolve_preset(matrix, f"{prefix}/{mode}", checkpoint_context=context)


@pytest.mark.parametrize("source_name", ["ambi_full_h3_j8_fresh", "ambi_full_h3_mppi_aux_return"])
def test_both_backbones_disable_inherited_planners_and_preserve_outer_actor(source_name):
    source = json.loads((ROOT / f"configs/dmcontrol/algs/{source_name}.json").read_text())
    before = deepcopy(source)
    for mode in ("mean", "sample"):
        resolved = _resolve(source, mode)
        params = resolved["algorithm_config"]["alg_params"]
        cfg = _build_cfg(**params)
        assert cfg.inner_operator == "none"
        for key in ("inner_rounds", "inner_model_step_budget", "inner_mppi_iterations",
                    "inner_critic_updates_per_action", "inner_actor_updates_per_action",
                    "inner_temperature_updates_per_action"):
            assert getattr(cfg, key) == 0, key
        assert cfg.inner_actor_source == "sac"
        assert cfg.inner_actor_adaptation == cfg.inner_critic_adaptation == "frozen"
        assert cfg.inner_execution_action == "policy_sample"
        assert cfg.inner_execution_std_scale == 1.0
        assert cfg.inner_execution_noise_std == 0.0
        assert cfg.inner_critic_bernoulli_p == 0.0
        assert not cfg.inner_finite_horizon
        assert not cfg.inner_outer_replay_fraction
        assert not cfg.inner_actor_writeback_coef and not cfg.inner_critic_writeback_coef
        assert not any(key.startswith("inner_mppi_") for key in params)
        assert "inner_critic_updates_per_round" not in params
        assert "inner_actor_updates_per_round" not in params
        assert "inner_steps_per_update" not in params
        for key, value in source["alg_params"].items():
            if not key.startswith("inner_") and key not in {"compile", "compile_strict", "wandb"}:
                assert params[key] == value, key
        assert frozen_execution_action(resolved) == mode
        assert resolved["evaluation"]["togo_return_rollouts"] == 0
    assert source == before


def test_each_baseline_has_exact_matched_h3_j8_environment_action_seed_protocol():
    matrix = load_preset_matrix(MATRIX)
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 55
    assert matrix["evaluation"]["max_steps"] == 500
    source = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_h3_j8_fresh.json").read_text())
    for mode in ("mean", "sample"):
        prior = _resolve(source, mode)
        adapted = _resolve(source, mode, matrix=ROOT / "configs/research/ambi_early_h3_j8.json")
        assert protocol_for(prior, 55, 500) == protocol_for(adapted, 55, 500)
        assert prior["evaluation"]["seeds"] == adapted["evaluation"]["seeds"]


@pytest.mark.parametrize("source_operator", ["sac", "mppi"])
@pytest.mark.parametrize("mode", ["mean", "sample"])
def test_loaded_prior_action_matches_saved_actor_and_performs_zero_inner_work(
        tmp_path, monkeypatch, source_operator, mode):
    checkpoint, tiny = _tiny_checkpoint(tmp_path, source_operator)
    resolved = _resolve(tiny["algorithm_config"], mode)
    env = gym.make("Pendulum-v1", max_episode_steps=3)
    model, _ = evaluator._initialize_frozen_model(resolved, env, checkpoint, 55, device="cpu")
    try:
        engine = model.agent.inner_engine
        assert engine._actor_base is model.agent.model._pi
        observation, _ = env.reset(seed=101)
        before = evaluator._outer_state_digest(model)
        with torch.no_grad():
            latent = model.agent.model.encode(model._obs_to_tensor(observation).unsqueeze(0))
        seed = solver_seed(55, "episode", 101)
        engine.reset_for_evaluation(seed)
        # Independently call the saved outer actor with exactly the isolated
        # execution stream. No inner distribution bounds or actor clones apply.
        with torch.no_grad(), engine.rng.action_fork(), engine.rng.fork("execution") as generator:
            expected = model.agent.model.pi_action(
                latent, deterministic=(mode == "mean"), generator=generator,
            )[0].numpy()
        expected = model._unscale_action(expected)
        def forbidden(*args, **kwargs):
            pytest.fail("Prior-only evaluation invoked imagined work or an outer update")
        monkeypatch.setattr(model.agent.model, "next_from_joint", forbidden)
        monkeypatch.setattr(model.agent.model, "reward_from_joint", forbidden)
        monkeypatch.setattr(model.agent, "update", forbidden)
        engine.reset_for_evaluation(seed)
        action, _ = model.predict(
            observation, deterministic=(frozen_execution_action(resolved) == "mean"),
            episode_start=True,
        )
        np.testing.assert_array_equal(action, expected)
        assert evaluator._outer_state_digest(model) == before
        for key in ("inner_active", "inner_model_steps", "inner_q_evaluations",
                    "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                    "inner_temperature_optimizer_steps", "inner_replay_draws"):
            assert model.agent.last_inner_metrics[key] == 0, key
        assert model.agent.last_inner_metrics["inner_policy_evaluations"] == 1
        assert engine.state.actor is None and engine.state.critic is None
        assert engine.state.actor_optim is None and engine.state.critic_optim is None
        engine.reset_for_evaluation(seed)
        repeated, _ = model.predict(observation, deterministic=(mode == "mean"), episode_start=True)
        np.testing.assert_array_equal(repeated, action)
    finally:
        model.close()
        env.close()
