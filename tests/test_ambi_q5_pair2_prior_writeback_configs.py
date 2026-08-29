import copy
import json
from pathlib import Path

import gymnasium as gym

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
REFERENCE_NAME = (
    "ambi_humanoid_walk_q5_pair2_prior_writeback_reference_1m_seed55"
)
VARIANTS = {
    REFERENCE_NAME: (
        0.0,
        0.0,
    ),
    "ambi_humanoid_walk_q5_pair2_prior_writeback_actor_001_1m_seed55": (
        0.01,
        0.0,
    ),
    "ambi_humanoid_walk_q5_pair2_prior_writeback_actor_01_1m_seed55": (
        0.1,
        0.0,
    ),
    "ambi_humanoid_walk_q5_pair2_prior_writeback_critic_001_1m_seed55": (
        0.0,
        0.01,
    ),
    "ambi_humanoid_walk_q5_pair2_prior_writeback_critic_01_1m_seed55": (
        0.0,
        0.1,
    ),
}
Q_REDUCTION_KEYS = (
    "outer_q_target_reduction",
    "outer_q_actor_reduction",
    "inner_q_target_reduction",
    "inner_q_actor_reduction",
)
ACTION_LOCAL_KEYS = (
    "inner_actor_scope",
    "inner_critic_scope",
    "inner_temperature_scope",
    "inner_replay_scope",
    "inner_actor_optimizer_scope",
    "inner_critic_optimizer_scope",
    "inner_temperature_optimizer_scope",
)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path):
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def _algorithm(name):
    return _load(ALGORITHM_ROOT / f"{name}.json")


def _manifest(name):
    return _load(EXPERIMENT_ROOT / f"{name}.json")


def _resolve_ambi_config(params):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 55,
        "device": "cpu",
        "env": "DMControl-v0",
        "total_steps": 1_000_000,
    }
    algorithm.custom_params = copy.deepcopy(params)
    try:
        return algorithm._build_cfg({"device": "cpu", **copy.deepcopy(params)})
    finally:
        algorithm.env.close()


def test_writeback_cells_change_only_budget_identity_and_writeback_controls():
    expected = copy.deepcopy(_algorithm(REFERENCE_NAME))
    expected["alg_params"].pop("wandb_tags")
    expected["alg_params"].pop("inner_actor_writeback_coef")
    expected["alg_params"].pop("inner_critic_writeback_coef")

    for name, (actor_coef, critic_coef) in VARIANTS.items():
        actual = copy.deepcopy(_algorithm(name))
        params = actual["alg_params"]
        tags = params.pop("wandb_tags")
        assert params.pop("inner_actor_writeback_coef") == actor_coef
        assert params.pop("inner_critic_writeback_coef") == critic_coef
        assert actual == expected

        assert len(tags) == len(set(tags))
        assert "14m-decisions" not in tags
        assert {
            "1m-decisions",
            "prior-writeback-phase1",
            "inner-final-outer-policy-kl-every-1k",
            "single-seed",
            "seed55",
        } <= set(tags)


def test_writeback_cells_freeze_q_temperature_eval_and_lifecycle_contracts():
    for name, (actor_coef, critic_coef) in VARIANTS.items():
        algorithm = _algorithm(name)
        params = algorithm["alg_params"]

        assert algorithm["seed"] == 55
        assert algorithm["total_steps"] == 1_000_000
        assert params["q_representation"] == "distributional"
        assert params["num_q"] == 5
        assert params["q_pair_size"] == 2
        assert {params[key] for key in Q_REDUCTION_KEYS} == {"min_pair"}
        assert params["outer_critic_target"] == "reward_only"
        assert params["inner_sac_critic_target"] == "reward_only"

        assert params["rho"] == 0.5
        assert params["train_unroll_horizon"] == 3
        assert params["temporal_loss_normalization"] == (
            "reference_weighted_mean"
        )
        assert params["temporal_loss_reference_horizon"] == 3
        assert params["ent_coef"] == "auto"
        assert params["inner_temperature_mode"] == "auto"

        assert params["inner_operator"] == "sac"
        assert params["inner_rounds"] == 2
        assert params["inner_rollouts_per_round"] == 32
        assert params["inner_rollout_horizon"] == 3
        assert params["inner_updates_per_round"] == 4
        assert params["inner_actor_adaptation"] == "clone"
        assert params["inner_critic_adaptation"] == "clone"
        assert {params[key] for key in ACTION_LOCAL_KEYS} == {"action"}

        assert params["inner_actor_writeback_coef"] == actor_coef
        assert params["inner_critic_writeback_coef"] == critic_coef
        assert params["eval_freq"] == 50_000
        assert params["eval_inner_comparison"] is True
        assert params["eval_inner_comparison_episodes"] == 5
        assert params["eval_inner_comparison_seed"] == 12_345
        assert params["eval_value"] is True
        assert params["eval_value_samples"] == 100
        assert params["inner_diagnostic_rollouts"] == 0
        assert params["inner_diagnostics_every"] == 1_000
        assert params["wandb"] is True
        assert params["wandb_mode"] == "online"
        assert params["wandb_step_every"] == 1_000

        # KL is logged observationally as train/inner_final_outer_policy_kl.
        # These learning regularizers remain disabled through their defaults.
        assert "inner_outer_policy_kl_coef" not in params
        assert "outer_behavior_policy_kl_schedule" not in params


def test_every_writeback_cell_passes_the_active_ambi_config_preflight():
    for name, (actor_coef, critic_coef) in VARIANTS.items():
        cfg = _resolve_ambi_config(_algorithm(name)["alg_params"])
        assert cfg.inner_actor_writeback_coef == actor_coef
        assert cfg.inner_critic_writeback_coef == critic_coef
        assert cfg.eval_inner_comparison is True
        assert cfg.inner_schedule_mode == "canonical"


def test_writeback_manifests_are_independent_seed55_screening_cells():
    for name in VARIANTS:
        manifest = _manifest(name)
        assert manifest["study_type"] == (
            "single_seed_1m_prior_writeback_phase1"
        )
        assert manifest["overrides_alg"] == {
            "seed": 55,
            "device": "cuda",
            "env": "DMControl-v0",
            "total_steps": 1_000_000,
            "episodes": None,
        }
        assert manifest["env_params"] == {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        }
        assert manifest["trials"] == 1
        assert manifest["configs"] == [name]
        assert manifest["checkpoint_every"] == 50_000
        assert manifest["save_strat"] == ["best", "latest"]

        note = manifest["study_note"]
        assert "eval/paired_fresh_inner_minus_outer" in note
        assert "train/inner_final_outer_policy_kl" in note
        assert "no KL regularization loss is enabled" in note
        assert "normalized rho-weighted actor-depth mixture" in note
