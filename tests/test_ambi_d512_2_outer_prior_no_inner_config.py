import copy
import json
import os
from pathlib import Path
import subprocess

import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
LAUNCHER = (
    ROOT
    / "slurm/run_ambi_humanoid_walk_outer_prior_no_inner_d512_2_hydra.sbatch"
)
PARENT = "ambi_humanoid_walk_base_v2_d512_2"
VARIANT = "ambi_humanoid_walk_outer_prior_no_inner_d512_2_14m"
VARIANT_TAGS = [
    "ambi",
    "dmcontrol",
    "humanoid-walk",
    "state",
    "base-v2",
    "tdmpc2-aligned-recipe",
    "training-focused",
    "14m-decisions",
    "d512-2-parent-control",
    "outer-prior-execution",
    "no-action-local-improvement",
    "inner-operator-none",
    "zero-inner-model-steps",
    "system-level-training-ablation",
    "policy-sample-training",
    "outer-critic-lr3e-4",
    "outer-actor-lr3e-4",
    "q-min-pair",
    "q-heads-5",
    "q-pair-size-2",
    "critic-target-entropy-augmented",
    "outer-alpha-auto",
    "outer-alpha-lr3e-4",
    "actor-loss-scale-none",
    "eval-off",
    "no-model-checkpoints",
    "single-seed",
    "seed55",
]


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


def test_outer_prior_control_is_an_exact_d512_2_derivative():
    parent = _load(ALGORITHM_ROOT / f"{PARENT}.json")
    actual = _load(ALGORITHM_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(parent)
    params = expected["alg_params"]
    params.update(
        {
            "inner_operator": "none",
            "inner_rounds": None,
            "inner_rollouts_per_round": None,
            "inner_temperature_mode": "inherit_outer",
            "wandb_run_name": (
                "AMBITDMPC2-humanoid-walk-base-v2-d512-2-"
                "outer-prior-no-inner-seed55"
            ),
            "wandb_tags": VARIANT_TAGS,
        }
    )
    params.pop("inner_critic_updates_per_round")
    params.pop("inner_actor_updates_per_round")

    assert actual == expected


def test_outer_prior_control_preserves_parent_outer_learning_contract():
    parent_params = _load(ALGORITHM_ROOT / f"{PARENT}.json")["alg_params"]
    variant_params = _load(ALGORITHM_ROOT / f"{VARIANT}.json")["alg_params"]

    keys = (
        "obs",
        "model_size",
        "episodic",
        "discount",
        "buffer_size",
        "batch_size",
        "utd",
        "train_unroll_horizon",
        "temporal_loss_normalization",
        "temporal_loss_reference_horizon",
        "rho",
        "compile",
        "compile_strict",
        "mpc",
        "q_representation",
        "num_q",
        "q_num_bins",
        "q_vmin",
        "q_vmax",
        "q_pair_size",
        "outer_q_target_reduction",
        "outer_q_actor_reduction",
        "outer_critic_target",
        "dropout",
        "actor_lr",
        "critic_lr",
        "critic_coef",
        "ent_coef",
        "ent_coef_lr",
        "target_entropy",
        "tau",
        "target_update_interval",
        "outer_policy_episode_probability",
    )
    assert {key: variant_params[key] for key in keys} == {
        key: parent_params[key] for key in keys
    }


def test_outer_prior_control_resolves_zero_inner_work_and_prior_sampling():
    params = _load(ALGORITHM_ROOT / f"{VARIANT}.json")["alg_params"]
    cfg = _resolve(params)

    assert cfg.inner_operator == "none"
    assert cfg.inner_component_update_schedule is False
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.inner_execution_std_scale == pytest.approx(1.0)
    assert cfg.inner_execution_noise_std == pytest.approx(0.0)
    assert cfg.outer_policy_episode_probability == pytest.approx(0.0)
    assert cfg.inner_temperature_mode == "inherit_outer"
    assert cfg.inner_rounds == 0
    assert cfg.inner_rollouts_per_round == 0
    assert cfg.inner_updates_per_round == 0
    assert cfg.inner_mppi_iterations == 0
    assert cfg.inner_model_step_budget == 0
    assert cfg.inner_expected_update_slots == 0
    assert cfg.inner_critic_updates_per_action == 0
    assert cfg.inner_actor_updates_per_action == 0
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_critic_updates_per_round is None
    assert cfg.inner_actor_updates_per_round is None
    assert cfg.inner_batch_size == 512
    assert cfg.inner_replay_capacity == 12_288


def test_outer_prior_control_manifest_preserves_d512_2_run_contract():
    parent = _load(EXPERIMENT_ROOT / f"{PARENT}.json")
    actual = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(parent)
    expected.update(
        {
            "study_type": (
                "single_seed_exploratory_no_action_local_improvement_d512_2"
            ),
            "study_note": actual["study_note"],
            "configs": [VARIANT],
        }
    )

    assert actual == expected
    assert actual["overrides_alg"]["seed"] == 55
    assert actual["overrides_alg"]["total_steps"] == 14_000_000
    assert actual["trials"] == 1
    assert actual["checkpoint_every"] is None
    assert actual["save_strat"] == "none"

    note = actual["study_note"].lower()
    assert "configuration and outer-learning recipe derive from" in note
    assert "it trains from scratch" in note
    assert "planned post-seed training decision" in note
    assert "persistent stochastic outer policy prior" in note
    assert "j=8, n=512, c=3, and a=1 controls are intentionally removed" in note
    assert "system-level training ablation" in note
    assert "learned weights and replay contents evolve" in note
    assert "single-seed exploratory" in note


def test_hydra_launcher_pins_one_seed55_job_to_gpu2201():
    contents = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --nodelist=gpu2201" in contents
    assert 'readonly EXPECTED_NODE="gpu2201"' in contents
    assert 'if "RTX A6000" not in device_name:' in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --no-requeue" in contents
    assert "#SBATCH --array" not in contents
    assert f'readonly CONFIG_STEM="{VARIANT}"' in contents
    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "AMBI_NO_INNER_RESULTS_ROOT" in contents
    assert "AMBI_NO_INNER_PYTHON" in contents
    assert "environments/dmcontrol/uv.lock" in contents
    assert "PARENT_CONFIG" in contents
    assert "config != expected" in contents
    assert '"inner_model_step_budget": 0' in contents
    assert 'cfg.inner_execution_action != "policy_sample"' in contents
    assert "CUDA_LAUNCH_BLOCKING" in contents
    assert 'task="humanoid-walk"' in contents
    assert "observation.shape != (67,)" in contents
    assert "--alg-index 0" in contents
    assert "--trial-index 0" in contents
    assert "--num-runs 1" in contents


def test_hydra_launcher_is_executable_and_has_valid_bash_syntax():
    assert os.access(LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
