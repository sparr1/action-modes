import copy
import json
from pathlib import Path

import gymnasium as gym

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
REFERENCE = "ambi_humanoid_walk_q5_pair2_prior_writeback_reference_1m_seed55"
VARIANT = "ambi_humanoid_walk_q5_pair2_ve_loss_c0p1_mc4_1m_seed55"


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


def test_value_equivalence_screen_changes_only_approved_controls_and_identity():
    reference = copy.deepcopy(_algorithm(REFERENCE))
    variant = copy.deepcopy(_algorithm(VARIANT))
    assert {key: value for key, value in variant.items() if key != "alg_params"} == {
        key: value for key, value in reference.items() if key != "alg_params"
    }

    reference_params = reference["alg_params"]
    params = variant["alg_params"]
    reference_params.pop("wandb_tags")
    tags = params.pop("wandb_tags")
    for key, expected in {
        "value_equivalence_loss_coef": 0.1,
        "value_equivalence_loss_mc_samples": 4,
        "wandb_group": "ambi-hw-q5-pair2-ve-loss-screen-1m",
        "wandb_run_name": (
            "AMBITDMPC2-humanoid-walk-q5-pair2-ve-loss-c0p1-mc4-1m-seed55"
        ),
    }.items():
        assert params.pop(key) == expected
        reference_params.pop(key, None)
    assert params == reference_params

    assert len(tags) == len(set(tags))
    assert {
        "value-equivalence-loss-screen",
        "value-equivalence-loss-active",
        "value-equivalence-loss-coef-0p1",
        "value-equivalence-loss-mc-samples-4",
        "matched-q5-pair2-base",
        "continuing-task",
        "actor-writeback-beta-0",
        "critic-writeback-beta-0",
    } <= set(tags)
    assert {
        "prior-writeback-phase1",
        "reference",
        "prior-writeback-reference",
    }.isdisjoint(tags)


def test_value_equivalence_screen_freezes_the_matched_training_contract():
    algorithm = _algorithm(VARIANT)
    params = algorithm["alg_params"]

    assert algorithm["seed"] == 55
    assert algorithm["total_steps"] == 1_000_000
    assert params["episodic"] is False
    assert params["inner_operator"] == "sac"
    assert params["q_representation"] == "distributional"
    assert params["num_q"] == 5
    assert params["q_pair_size"] == 2
    assert {
        params[key]
        for key in (
            "outer_q_target_reduction",
            "outer_q_actor_reduction",
            "inner_q_target_reduction",
            "inner_q_actor_reduction",
        )
    } == {"min_pair"}
    assert params["outer_critic_target"] == "reward_only"
    assert params["inner_sac_critic_target"] == "reward_only"
    assert params["inner_actor_writeback_coef"] == 0
    assert params["inner_critic_writeback_coef"] == 0
    assert "value_equivalence_diagnostics" not in params


def test_value_equivalence_screen_passes_active_ambi_config_preflight():
    cfg = _resolve_ambi_config(_algorithm(VARIANT)["alg_params"])
    assert cfg.value_equivalence_loss_coef == 0.1
    assert cfg.value_equivalence_loss_mc_samples == 4
    assert cfg.value_equivalence_diagnostics is False
    assert cfg.episodic is False
    assert cfg.inner_operator == "sac"


def test_value_equivalence_screen_manifest_is_single_seed_and_resumable():
    manifest = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")
    assert manifest["study_type"] == "single_seed_1m_value_equivalence_loss_screen"
    assert manifest["configs"] == [VARIANT]
    assert manifest["trials"] == 1
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
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 50_000
    assert manifest["save_strat"] == ["best", "latest"]

    note = manifest["study_note"]
    assert "one seed does not support a confidence or significance claim" in note
    assert "only learning-axis change" in note
    assert "coefficient 0.1 and four Monte Carlo samples" in note
    assert "explicit value-equivalence diagnostics remain disabled" in note
    assert "c6857707751946929e27be0514a502a5" in note
