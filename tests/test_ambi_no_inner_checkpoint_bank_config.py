import copy
import json
from pathlib import Path

import gymnasium as gym

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
PARENT = "ambi_humanoid_walk_outer_prior_no_inner_d512_2_14m"
BANK = "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"
LAUNCHER = (
    ROOT
    / "slurm/run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_hydra.sbatch"
)


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
        "total_steps": 1_500_000,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def test_bank_is_exact_no_inner_parent_derivative():
    parent = _load(ALGORITHM_ROOT / f"{PARENT}.json")
    actual = _load(ALGORITHM_ROOT / f"{BANK}.json")
    expected = copy.deepcopy(parent)
    expected["total_steps"] = 1_500_000
    expected_params = expected["alg_params"]
    actual_params = actual["alg_params"]
    for key in ("wandb_group", "wandb_run_name", "wandb_tags"):
        expected_params[key] = actual_params[key]

    assert actual == expected
    assert actual["seed"] == 55
    assert actual["episodes"] is None
    assert "checkpoint_every" not in actual
    assert "save_strat" not in actual


def test_bank_locks_entropy_augmented_outer_critic_and_auto_alpha():
    params = _load(ALGORITHM_ROOT / f"{BANK}.json")["alg_params"]

    assert params["outer_critic_target"] == "entropy_augmented"
    assert params["inner_sac_critic_target"] == "entropy_augmented"
    assert params["ent_coef"] == "auto"
    assert params["ent_coef_lr"] == 0.0003
    assert params["target_entropy"] == "auto"
    assert "outer-critic-entropy-augmented" in params["wandb_tags"]
    assert "outer-alpha-auto" in params["wandb_tags"]

    cfg = _resolve(params)
    assert cfg.outer_critic_target == "entropy_augmented"
    assert cfg.inner_sac_critic_target == "entropy_augmented"
    assert cfg.ent_coef == "auto"
    assert cfg.target_entropy == "auto"


def test_bank_resolves_no_inner_training_or_diagnostics():
    params = _load(ALGORITHM_ROOT / f"{BANK}.json")["alg_params"]
    cfg = _resolve(params)

    assert cfg.inner_operator == "none"
    assert cfg.mpc is False
    assert cfg.inner_component_update_schedule is False
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.outer_policy_episode_probability == 0.0
    assert cfg.inner_rounds == 0
    assert cfg.inner_rollouts_per_round == 0
    assert cfg.inner_updates_per_round == 0
    assert cfg.inner_mppi_iterations == 0
    assert cfg.inner_model_step_budget == 0
    assert cfg.inner_expected_update_slots == 0
    assert cfg.inner_critic_updates_per_action == 0
    assert cfg.inner_actor_updates_per_action == 0
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_diagnostic_rollouts == 0
    assert cfg.eval_freq is None
    assert cfg.eval_inner_comparison is False
    assert cfg.eval_value is False
    assert cfg.value_equivalence_diagnostics is False


def test_bank_manifest_matches_tdmpc2_checkpoint_pattern():
    algorithm = _load(ALGORITHM_ROOT / f"{BANK}.json")
    manifest = _load(EXPERIMENT_ROOT / f"{BANK}.json")

    assert manifest["overrides_alg"] == {
        "seed": 55,
        "device": "cuda",
        "env": "DMControl-v0",
        "total_steps": 1_500_000,
        "episodes": None,
    }
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 1
    assert manifest["configs"] == [BANK]
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 25_000
    assert manifest["save_strat"] == ["all"]
    assert algorithm["total_steps"] % manifest["checkpoint_every"] == 0
    assert algorithm["total_steps"] // manifest["checkpoint_every"] == 60

    note = manifest["study_note"].lower()
    assert "zero inner model steps and zero inner optimizer updates" in note
    assert "entropy-augmented outer" in note
    assert "25,000-decision intervals" in note
    assert "no step-zero checkpoint" in note


def test_bank_launcher_pins_one_gpu2501_l40s_job_and_rechecks_contract():
    contents = LAUNCHER.read_text(encoding="utf-8")

    for contract in (
        "#SBATCH --partition=gpus",
        "#SBATCH --nodelist=gpu2501",
        "#SBATCH --constraint=l40s",
        "#SBATCH --gres=gpu:nvidia_l40s:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=5-00:00:00",
        "#SBATCH --no-requeue",
        "git status --porcelain --untracked-files=normal",
        '"outer_critic_target": "entropy_augmented"',
        '"ent_coef": "auto"',
        '"inner_operator": "none"',
        "Agent decisions: 1500000",
        "Raw simulator control steps: 3000000",
        "Inner model steps per action: 0",
        "Checkpoint cadence: 25000",
        "Expected numbered checkpoints: 60",
        "--alg-index 0",
        "--trial-index 0",
        "--num-runs 1",
    ):
        assert contract in contents
    assert "#SBATCH --array" not in contents
    assert "SLURM_ARRAY_" not in contents
