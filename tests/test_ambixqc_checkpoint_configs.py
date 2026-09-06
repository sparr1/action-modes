"""Scientific configuration contracts for the no-inner XQC checkpoint bank."""

import json
from copy import deepcopy
from pathlib import Path

import gymnasium as gym

import main as training_main
from RL.AMBIXQC import AMBIXQC
from utils.ambi_research import load_preset_matrix, normalize_selectors


ROOT = Path(__file__).resolve().parents[1]
STEM = "ambixqc_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"


def _load(relative):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"Duplicate configuration key: {key}"
            result[key] = value
        return result

    return json.loads((ROOT / relative).read_text(), object_pairs_hook=unique)


def test_prior_bank_preserves_standard_xqc_learning_and_dormant_inner_budget():
    standard = _load("configs/dmcontrol/algs/ambixqc_humanoid_walk_state.json")
    bank = _load(f"configs/dmcontrol/algs/{STEM}.json")
    expected = deepcopy(standard)
    expected["total_steps"] = 1_500_000
    expected["alg_params"]["inner_operator"] = "none"
    expected["alg_params"]["eval_freq"] = None
    # Run identity is intentionally different; all other parameters must match.
    for key in ("wandb_group", "wandb_tags"):
        assert bank["alg_params"][key] != standard["alg_params"][key]
        expected["alg_params"][key] = bank["alg_params"][key]
    assert bank == expected
    assert bank["seed"] == 55
    assert bank["alg"] == "AMBIXQC/AMBIXQC"
    assert bank["alg_params"]["inner_rounds"] == 2
    assert bank["alg_params"]["inner_rollouts_per_round"] == 32
    assert bank["alg_params"]["inner_rollout_horizon"] == 3
    assert bank["alg_params"]["inner_updates_per_round"] == 4
    assert bank["alg_params"]["inner_batch_size"] == 64


def test_prior_bank_keeps_sixty_checkpoints_and_no_online_evaluation():
    bank = _load(f"configs/dmcontrol/algs/{STEM}.json")
    manifest = _load(f"configs/dmcontrol/experiments/{STEM}.json")
    assert manifest["configs"] == [STEM]
    assert manifest["trials"] == 1
    assert manifest["save_strat"] == ["all"]
    assert manifest["checkpoint_every"] == 25_000
    assert bank["total_steps"] // manifest["checkpoint_every"] == 60
    assert bank["alg_params"]["eval_freq"] is None
    assert manifest["logs"] != "overwrite"
    assert manifest["env_params"] == {
        "task": "humanoid-walk", "obs": "state", "render_mode": None,
    }
    # An algorithm-level cadence would override the manifest, even if null.
    assert "checkpoint_every" not in bank
    assert "alg_params" not in manifest.get("overrides_alg", {})


def test_real_wrapper_resolves_prior_bank_with_zero_active_inner_work():
    config = _load(f"configs/dmcontrol/algs/{STEM}.json")
    wrapper = object.__new__(AMBIXQC)
    wrapper.env = gym.make("Pendulum-v1")
    wrapper.run_params = {**config, "device": "cpu"}
    wrapper.experiment_params = {}
    wrapper.custom_params = config["alg_params"]
    try:
        cfg = wrapper._build_cfg({**config["alg_params"], "device": "cpu"})
    finally:
        wrapper.env.close()
    assert cfg.steps == cfg.xqc_lr_transition_steps == 1_500_000
    assert cfg.inner_operator == "none"
    assert cfg.inner_rounds == 2
    assert cfg.inner_updates_per_round == 4
    assert cfg.inner_reward_normalization == "frozen_real_scale"
    assert cfg.inner_model_step_budget == 0
    assert cfg.inner_expected_update_slots == 0
    assert cfg.inner_critic_updates_per_action == 0
    assert cfg.inner_actor_updates_per_action == 0
    assert cfg.inner_temperature_updates_per_action == 0
    wrapper.cfg = cfg
    metadata = training_main._resolved_runtime_metadata(wrapper, trial_run_params=config)
    inner = metadata["inner_budget"]
    assert inner["inner_rounds"] == 2
    assert inner["inner_rollouts_per_round"] == 32
    assert inner["branches_per_action"] == 0
    assert inner["transitions_per_round"] == 0
    assert inner["transitions_per_action"] == 0
    assert inner["replay_rows_drawn_per_action"] == 0


def test_xqc_matrix_uses_saved_problem_and_requires_explicit_inner_selection():
    matrix = load_preset_matrix(ROOT / "configs/research/ambixqc_humanoid_inner_benchmark.json")
    assert matrix["base_alg_config"] == "checkpoint"
    assert "source_run" not in matrix
    assert "environment" not in matrix
    assert matrix["shared_alg_params"] == {}
    assert normalize_selectors(matrix) == ["controller/prior"]
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["max_steps"] == 500
    variants = matrix["comparisons"]["controller"]["variants"]
    assert variants["prior"]["alg_params"] == {"inner_operator": "none"}
    assert variants["xqc"]["alg_params"] == {"inner_operator": "xqc"}
