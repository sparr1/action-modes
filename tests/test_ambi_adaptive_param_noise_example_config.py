"""Runnable D512 example for the adaptive parameter-noise population."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym

import main as training_main
from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_v2_d512_1"
PARAM_NOISE = "ambi_humanoid_walk_base_v2_adaptive_param_noise_d512_k4"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _resolve(params):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 55,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 14_000_000,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def test_d512_k4_example_changes_only_parameter_noise_contract_and_identity():
    base = _load(ALGORITHM_ROOT / f"{BASE}.json")
    actual = _load(ALGORITHM_ROOT / f"{PARAM_NOISE}.json")
    expected = copy.deepcopy(base)
    expected["alg_params"].update(
        {
            "inner_explorer_mode": "adaptive_param_noise",
            "inner_prior_rollout_weight": 0.5,
            "inner_param_noise_actor_count": 4,
            "inner_param_noise_target_action_rms": 0.1,
            "inner_param_noise_sigma_init": 0.001,
            "inner_param_noise_sigma_min": 0.000001,
            "inner_param_noise_sigma_max": 0.1,
            "inner_param_noise_calibration_directions": 8,
            "inner_param_noise_calibration_batch_size": 32,
            "inner_param_noise_calibration_max_probes": 8,
            "inner_execution_policy_source": "primary",
            "wandb_run_name": actual["alg_params"]["wandb_run_name"],
            "wandb_tags": actual["alg_params"]["wandb_tags"],
        }
    )

    assert actual == expected


def test_d512_k4_example_resolves_exact_equal_population_budget():
    params = _load(ALGORITHM_ROOT / f"{PARAM_NOISE}.json")["alg_params"]
    cfg = _resolve(params)

    assert cfg.inner_explorer_mode == "adaptive_param_noise"
    assert cfg.inner_explorer_active is True
    assert cfg.inner_param_noise_active is True
    assert cfg.inner_primary_rollouts_per_round == 256
    assert cfg.inner_explorer_rollouts_per_round == 256
    assert cfg.inner_param_noise_actor_count == 4
    assert cfg.inner_param_noise_rollouts_per_actor == 64
    assert cfg.inner_behavior_action == "policy_sample"
    assert cfg.inner_execution_policy_source == "primary"
    assert cfg.inner_model_step_budget == 8 * 512 * 3 == 12_288
    assert cfg.inner_replay_capacity == 12_288

    metadata = training_main._resolved_runtime_metadata(
        SimpleNamespace(
            cfg=cfg,
            agent=SimpleNamespace(
                model=SimpleNamespace(critic_signature={})
            ),
            env=SimpleNamespace(),
        ),
        trial_run_params={
            "alg": "AMBITDMPC2/AMBITDMPC2",
            "seed": 55,
        },
    )
    inner = metadata["inner_budget"]
    assert inner["inner_param_noise_active"] is True
    assert inner["inner_param_noise_actor_count"] == 4
    assert inner["inner_param_noise_rollouts_per_actor"] == 64
    assert inner["inner_param_noise_target_action_rms"] == 0.1
    assert inner["inner_behavior_action"] == "policy_sample"
    assert inner["inner_behavior_std_scale"] == 1.0
    assert inner["inner_log_std_mapping"] == cfg.inner_log_std_mapping
    assert inner["inner_log_std_min"] == cfg.inner_log_std_min
    assert inner["inner_log_std_max"] == cfg.inner_log_std_max


def test_d512_k4_manifest_links_only_the_example_and_submits_nothing_itself():
    manifest = _load(EXPERIMENT_ROOT / f"{PARAM_NOISE}.json")

    assert manifest["configs"] == [PARAM_NOISE]
    assert manifest["trials"] == 1
    assert manifest["checkpoint_every"] is None
    assert manifest["save_strat"] == "none"
    assert manifest["overrides_alg"]["total_steps"] == 14_000_000
    note = manifest["study_note"].lower()
    assert "256 rollouts to the clean" in note
    assert "64 rollouts per perturbed actor" in note
    assert "12,288 imagined transitions" in note
