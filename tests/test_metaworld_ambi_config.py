"""Scientific contract of the four-action post-tanh sampled AMBI recipe.

The source is configs/dmcontrol/algs/ambi_inner_entropy_posttanh_outer_adapt.json
at 5f6089fe70441b968ed814d1b1e73ddbd3f08a71. Tests resolve the local recipe;
they require neither that Git history nor a saved external experiment snapshot.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/metaworld/algs/ambi_state.json"


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate configuration key: {key}"
        result[key] = value
    return result


def _recipe():
    return json.loads(RECIPE.read_text(), object_pairs_hook=_unique_keys)


class MetaWorldSpaces(gym.Env):
    observation_type = "state"

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(39,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.spec = SimpleNamespace(max_episode_steps=100)


def _resolve(env):
    recipe = _recipe()
    learner = object.__new__(AMBITDMPC2)
    learner.env = env
    learner.run_params = {**recipe, "device": "cpu"}
    learner.custom_params = deepcopy(recipe["alg_params"])
    # Only select a CPU resolver. Production compilation and CUDA-graph flags
    # remain enabled and undergo their full configuration validation.
    return learner._build_cfg({**learner.custom_params, "device": "cpu"})


@pytest.fixture
def cfg():
    env = MetaWorldSpaces()
    try:
        yield _resolve(env)
    finally:
        env.close()


def test_recipe_preserves_seed_budget_and_production_execution_flags(cfg):
    recipe = _recipe()
    assert (recipe["seed"], recipe["total_steps"], recipe["episodes"]) == (
        55, 1_000_000, None,
    )
    assert recipe["env"] == "MetaWorld-v0"
    assert recipe["alg"] == "AMBITDMPC2/AMBITDMPC2"
    assert recipe["device"] == "cuda"
    assert recipe["alg_params"]["episode_length"] == 100
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
    assert cfg.critic_ensemble_backend == "vmap"
    assert cfg.model_size == 5 and cfg.latent_dim == cfg.mlp_dim == 512
    assert cfg.batch_size == 256 and cfg.buffer_size == 1_000_000 and cfg.utd == 1
    assert cfg.obs == "state" and cfg.obs_shape == {"state": (39,)}
    assert cfg.action_dim == 4 and cfg.episode_length == 100
    assert cfg.seed == 55 and cfg.seed_steps == cfg.pretrain_steps == 1000
    assert cfg.discount == cfg.discount_min == cfg.discount_max == 0.95
    assert not cfg.episodic and not cfg.mpc


def test_post_tanh_entropy_retains_dimension_matching_and_fixed_temperature(cfg):
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.log_std_mapping == cfg.inner_log_std_mapping == "tdmpc2_tanh"
    assert (cfg.log_std_min, cfg.log_std_max) == (-10, 2)
    assert (cfg.inner_log_std_min, cfg.inner_log_std_max) == (-10, 2)
    # The original 21-action coefficient was 21 * 1e-4, matching TD-MPC2's
    # dimension multiplier. Keeping .0021 here would change that protocol.
    assert cfg.ent_coef == pytest.approx(cfg.action_dim * 1e-4)
    assert cfg.inner_temperature == pytest.approx(cfg.ent_coef)
    assert cfg.inner_temperature_mode == "inherit_outer"
    assert cfg.inner_temperature_initialization == "inherit_outer"
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_entropy_enabled


def test_sampled_fresh_inner_sac_preserves_complete_solve_budget(cfg):
    assert cfg.inner_operator == "sac"
    assert cfg.inner_behavior_action == cfg.inner_execution_action == "policy_sample"
    assert cfg.inner_behavior_std_scale == cfg.inner_execution_std_scale == 1.0
    assert cfg.inner_behavior_noise_std == cfg.inner_execution_noise_std == 0.0
    assert cfg.outer_policy_episode_probability == 0.0
    assert cfg.train_unroll_horizon == cfg.outer_planning_horizon == cfg.inner_rollout_horizon == 3
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_solve_interval) == (8, 128, 1)
    assert (cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (16, 4)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
    assert cfg.inner_model_step_budget == 8 * 128 * 3 == cfg.inner_replay_capacity
    assert cfg.inner_batch_size == 256 and cfg.inner_replay_sampling == "with_replacement"
    assert cfg.inner_finite_horizon and cfg.inner_bootstrap_source == "inner_target"
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "prior"
    assert cfg.inner_actor_adaptation == cfg.inner_critic_adaptation == "clone"
    for component in ("actor", "critic", "temperature", "replay"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"
    for component in ("actor", "critic", "temperature"):
        assert getattr(cfg, f"inner_{component}_optimizer_scope") == "action"
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0.0


def test_reward_only_q_and_action_local_outer_adapt_scale_are_retained(cfg):
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert cfg.q_representation == "distributional" and cfg.critic_value_mode == "single"
    assert (cfg.num_q, cfg.q_pair_size, cfg.q_num_bins) == (5, 2, 101)
    assert (cfg.q_vmin, cfg.q_vmax) == (-10, 10)
    assert cfg.outer_q_actor_reduction == cfg.inner_q_actor_reduction == "mean_pair"
    assert cfg.outer_q_target_reduction == cfg.inner_q_target_reduction == "min_pair"
    assert cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"
    assert cfg.sac_actor_loss_scale_tau == 0.01
    assert cfg.inner_actor_loss_scale_scope == "action"
    assert cfg.inner_actor_loss_scale_update == "per_update"
    # This run backbone predates teacher-policy replay and only maximizes Q.
    assert cfg.outer_actor_objective == "q_maximization"
    assert cfg.aux_return_mode == "off" and cfg.inner_critic_bernoulli_p == 0.0
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "sac"
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.inner_explorer_mode == "none" and cfg.outer_behavior_policy_kl_schedule == "none"


def test_evaluation_and_required_geometry_are_enabled_without_publication(cfg):
    assert cfg.eval_freq == 25_000 and cfg.eval_episodes == 10
    assert not cfg.eval_inner_comparison and not cfg.eval_value
    assert cfg.eval_csv_path is None  # campaign supplies a distinct file per cell
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.critic_geometry_batch_size == 64
    assert cfg.critic_geometry_obs_noise_std == 0.01
    assert cfg.critic_geometry_action_noise_std == 0.02
    assert not cfg.wandb and cfg.wandb_mode == "disabled"


@pytest.mark.skipif(
    os.environ.get("AMBI_RUN_REAL_METAWORLD_TESTS") != "1",
    reason="requires the isolated Meta-World runtime; opt in explicitly",
)
@pytest.mark.parametrize("task", ["assembly", "hand-insert", "push", "stick-pull"])
def test_full_recipe_resolves_on_actual_metaworld_without_allocating_agent(task):
    import domains  # noqa: F401

    env = gym.make("MetaWorld-v0", task=task)
    try:
        cfg = _resolve(env)
        assert cfg.obs_shape == {"state": (39,)} and cfg.action_dim == 4
        assert cfg.episode_length == 100 and cfg.discount == 0.95
        assert cfg.ent_coef == pytest.approx(4 * 1e-4)
        assert cfg.inner_cuda_graph and cfg.compile_strict
        assert env.unwrapped.evaluation_success_key == "success"
    finally:
        env.close()
