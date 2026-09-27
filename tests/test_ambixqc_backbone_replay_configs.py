"""Scientific contract for the three 1M-decision replay-preserving backbones."""

import json
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import pytest

from RL.AMBIXQC import AMBIXQC
from utils.checkpointing import resolve_checkpoint_config


ROOT = Path(__file__).resolve().parents[1]
STEM = "ambixqc_humanoid_walk_backbone_replay_1m"
ARMS = {"baseline": ("off", True), "aux_shared": ("xqc", False),
        "aux_detached": ("xqc", True)}
IDENTITY_KEYS = {"wandb_group", "wandb_run_name", "wandb_tags"}
AUX_KEYS = {"aux_return_mode", "aux_return_detach_representation", "aux_return_critic_coef"}
INNER_KEYS = {"inner_reward_normalization", "inner_critic_source",
              "inner_horizon_critic_source", "inner_critic_target"}


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"Duplicate key {key} in {path}"
            result[key] = value
        return result
    return json.loads((ROOT / path).read_text(), object_pairs_hook=unique)


def _arm_config(arm, folder="algs"):
    return _read(f"configs/dmcontrol/{folder}/{STEM}_{arm}.json")


@pytest.mark.parametrize("arm", ARMS)
def test_backbone_arms_preserve_prior_bank_architecture_and_training_recipe(arm):
    source = _read("configs/dmcontrol/algs/ambixqc_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m.json")
    config = _arm_config(arm)
    assert source["total_steps"] == 1_500_000
    assert config["total_steps"] == 1_000_000
    assert config["seed"] == 55
    mode, detach = ARMS[arm]
    params = config["alg_params"]
    assert params["aux_return_mode"] == mode
    assert params["aux_return_detach_representation"] is detach
    assert params["aux_return_critic_coef"] == 0.1
    assert params["inner_operator"] == "none"
    assert params["inner_reward_normalization"] == "frozen_real_scale"
    assert params["inner_critic_source"] == params["inner_horizon_critic_source"] == "xqc"
    assert params["inner_critic_target"] == "entropy_augmented"
    assert params["compile"] is params["compile_strict"] is False
    assert params["eval_freq"] is None
    assert params["seed_steps"] == params["pretrain_steps"] == 2500
    assert params["wandb_group"] == "ambixqc-humanoid-walk-backbone-replay-1m"
    assert arm.replace("_", "-") in params["wandb_tags"]
    assert "replay-archive" in params["wandb_tags"]
    assert "1p5m-decisions" not in params["wandb_tags"]

    # Only budget, explicit new semantic defaults, and run identity differ
    # from the established no-inner source recipe.
    comparable = deepcopy(config)
    comparable["total_steps"] = source["total_steps"]
    for key in IDENTITY_KEYS | AUX_KEYS | INNER_KEYS:
        comparable["alg_params"].pop(key, None)
        source["alg_params"].pop(key, None)
    assert comparable == source


def test_backbone_arms_differ_only_in_auxiliary_training_and_identity():
    baselines = []
    names = set()
    for arm in ARMS:
        config = _arm_config(arm)
        names.add(config["alg_params"]["wandb_run_name"])
        for key in IDENTITY_KEYS | AUX_KEYS:
            config["alg_params"].pop(key)
        baselines.append(config)
    assert len(names) == 3
    assert baselines[0] == baselines[1] == baselines[2]


@pytest.mark.parametrize("arm", ARMS)
def test_manifest_retains_forty_checkpoint_matched_replays_per_arm(arm):
    algorithm = _arm_config(arm)
    manifest = _arm_config(arm, "experiments")
    assert manifest["configs"] == [f"{STEM}_{arm}"]
    assert manifest["trials"] == 1
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    assert manifest["overrides_alg"] == {"env": "DMControl-v0"}
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["log_type"] == "summary"
    assert manifest["log_info"] is False
    checkpoint = resolve_checkpoint_config(algorithm, manifest)
    assert checkpoint.enabled
    assert checkpoint.every == 25_000
    assert checkpoint.strategies == ("all",)
    assert checkpoint.save_replay_buffer is True
    assert algorithm["total_steps"] // checkpoint.every == 40
    assert "save_replay_buffer" not in algorithm["alg_params"]


@pytest.mark.parametrize("arm", ARMS)
def test_real_wrapper_resolves_backbone_arms_with_no_inner_work(arm):
    config = _arm_config(arm)
    wrapper = object.__new__(AMBIXQC)
    wrapper.env = gym.make("Pendulum-v1")
    wrapper.run_params = {**config, "device": "cpu"}
    wrapper.experiment_params = {}
    wrapper.custom_params = config["alg_params"]
    try:
        cfg = wrapper._build_cfg({**config["alg_params"], "device": "cpu"})
    finally:
        wrapper.env.close()
    mode, detach = ARMS[arm]
    assert cfg.steps == cfg.xqc_lr_transition_steps == 1_000_000
    assert cfg.aux_return_mode == mode
    assert cfg.aux_return_detach_representation is detach
    assert cfg.inner_operator == "none"
    assert cfg.inner_model_step_budget == cfg.inner_expected_update_slots == 0
    assert cfg.inner_critic_updates_per_action == cfg.inner_actor_updates_per_action == 0
    assert cfg.inner_rounds == 2 and cfg.inner_rollouts_per_round == 32
    assert cfg.inner_rollout_horizon == 3 and cfg.inner_updates_per_round == 4
    assert cfg.inner_replay_capacity == 192
