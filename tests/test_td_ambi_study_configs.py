"""Protocol and one-axis contrasts for the TD-AMBI prior/full training study."""

import json
from pathlib import Path

import pytest

from tests.test_ambi_config_decoupling import _build_cfg


ROOT = Path(__file__).resolve().parents[1] / "configs/dmcontrol"
CASES = {
    "td_ambi_prior_reward_qscale": ("prior", "reward_only", True, False, None),
    "td_ambi_prior_entropy_qscale": ("prior", "entropy_augmented", True, False, None),
    "td_ambi_prior_entropy_autotemp": ("prior", "entropy_augmented", False, True, None),
    "td_ambi_prior_reward_autotemp": ("prior", "reward_only", False, True, None),
    "td_ambi_full_reward_qscale_frozen": ("full", "reward_only", True, False, "per_action"),
    "td_ambi_full_reward_qscale_adaptive": ("full", "reward_only", True, False, "per_update"),
    "td_ambi_full_entropy_qscale_frozen": ("full", "entropy_augmented", True, False, "per_action"),
    "td_ambi_full_entropy_qscale_adaptive": ("full", "entropy_augmented", True, False, "per_update"),
    "td_ambi_full_entropy_autotemp": ("full", "entropy_augmented", False, True, "per_action"),
    "td_ambi_full_reward_autotemp": ("full", "reward_only", False, True, "per_action"),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate JSON key: {key}"
        result[key] = value
    return result


def _load(kind, name):
    return json.loads(
        (ROOT / kind / f"{name}.json").read_text(), object_pairs_hook=_unique_object
    )


@pytest.mark.parametrize("name", CASES)
def test_study_objectives_and_inner_schedule_resolve(name):
    category, target, scaling, auto_alpha, scale_update = CASES[name]
    config = _load("algs", name)
    params = config["alg_params"]
    cfg = _build_cfg(**params)

    assert config["alg"] == "AMBITDMPC2/AMBITDMPC2"
    assert cfg.mpc is False
    assert cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.outer_critic_target == target
    assert cfg.sac_actor_loss_scale_mode == (
        "tdmpc2_percentile_range" if scaling else "none"
    )
    assert cfg.ent_coef == ("auto_0.0001" if auto_alpha else 0.0001)
    if auto_alpha:
        assert cfg.target_entropy == -441

    if category == "prior":
        assert cfg.inner_operator == "none"
        assert cfg.inner_model_step_budget == 0
        assert cfg.inner_actor_updates_per_action == 0
        assert cfg.inner_critic_updates_per_action == 0
        assert cfg.inner_temperature_updates_per_action == 0
        return

    assert cfg.inner_operator == "sac"
    assert cfg.inner_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.inner_sac_critic_target == target
    assert cfg.inner_rounds == 6
    assert cfg.inner_rollouts_per_round == 512
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_batch_size == 512
    assert cfg.inner_update_timing == "step"
    assert cfg.inner_steps_per_update == 512
    assert cfg.inner_updates_per_round is None
    assert cfg.inner_model_step_budget == 9216
    assert cfg.inner_replay_capacity == 9216
    assert cfg.inner_replay_sampling == "with_replacement"
    assert cfg.inner_actor_updates_per_action == 18
    assert cfg.inner_critic_updates_per_action == 18
    assert cfg.inner_actor_loss_scale_update == scale_update
    assert cfg.inner_critic_target_initialization == "outer_target"
    assert cfg.inner_bootstrap_source == "inner_target"

    inner_auto = auto_alpha and target == "entropy_augmented"
    assert cfg.inner_temperature_mode == ("auto" if inner_auto else "inherit_outer")
    assert cfg.inner_temperature_initialization == "inherit_outer"
    assert cfg.inner_temperature_updates_per_action == (18 if inner_auto else 0)
    if inner_auto:
        assert cfg.inner_target_entropy == -441
        assert cfg.inner_temperature_grad_clip_norm is None
        assert cfg.inner_temperature_lr == cfg.ent_coef_lr

    assert cfg.inner_actor_lr == cfg.actor_lr == 3e-4
    assert cfg.inner_critic_lr == cfg.critic_lr == 3e-4
    assert cfg.inner_actor_adam_eps == cfg.actor_adam_eps == 1e-5
    assert cfg.inner_adam_eps == cfg.adam_eps == 1e-8
    assert cfg.inner_critic_loss_coef == cfg.critic_coef == 0.1
    assert cfg.inner_actor_grad_clip_norm == cfg.inner_critic_grad_clip_norm == cfg.grad_clip_norm == 20
    assert cfg.inner_critic_target_tau == cfg.tau == 0.01
    assert cfg.inner_critic_target_update_interval == cfg.target_update_interval == 1
    assert cfg.inner_critic_dropout_enabled is True
    assert cfg.inner_actor_adaptation == cfg.inner_critic_adaptation == "clone"
    for component in ("actor", "critic", "temperature", "replay", "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"


@pytest.mark.parametrize("name", CASES)
def test_study_preserves_td_ambi_backbone_and_has_explicit_save_policy(name):
    base = _load("algs", "TD-AMBI")
    config = _load("algs", name)
    manifest = _load("experiments", name)
    for key in ("seed", "env", "device", "episodes"):
        assert config[key] == base[key]
    assert base["total_steps"] == 14_000_000
    assert config["total_steps"] == 2_000_000
    assert manifest["overrides_alg"]["total_steps"] == 2_000_000
    params, base_params = config["alg_params"], base["alg_params"]
    for key in (
        "obs", "model_size", "episodic", "discount", "discount_denom",
        "discount_min", "discount_max", "buffer_size", "batch_size", "utd",
        "train_unroll_horizon", "outer_planning_horizon", "rho", "compile",
        "q_representation", "num_q", "q_num_bins", "q_vmin", "q_vmax",
        "q_pair_size", "outer_q_target_reduction", "outer_q_actor_reduction",
        "dropout", "actor_lr", "critic_lr", "adam_eps", "actor_adam_eps",
        "critic_coef", "outer_actor_entropy_mode", "log_std_mapping",
        "log_std_min", "log_std_max", "tau", "target_update_interval",
        "reward_coef", "consistency_coef", "termination_coef", "lr", "enc_lr_scale",
        "outer_behavior_policy_kl_schedule", "value_equivalence_loss_coef",
        "inner_actor_writeback_coef", "inner_critic_writeback_coef", "inner_explorer_mode",
    ):
        assert params[key] == base_params[key], key
    assert params["eval_freq"] is None
    assert params["eval_inner_comparison"] is False
    assert params["eval_value"] is False
    assert manifest["configs"] == [name]
    assert manifest["trials"] == 1
    assert manifest["env_params"]["task"] == "humanoid-walk"
    assert manifest["env_params"]["obs"] == "state"
    assert manifest["logs"] == "timestamp"
    assert "alg_params" not in manifest.get("overrides_alg", {})
    assert config["checkpoint_every"] == manifest["checkpoint_every"] == 25_000
    assert config["save_strat"] == manifest["save_strat"] == "all"
    assert manifest["save_trials"] == "none"
    assert config["total_steps"] // config["checkpoint_every"] == 80


@pytest.mark.parametrize("target", ("reward", "entropy"))
def test_freeze_comparison_changes_only_scale_updates_and_labels(target):
    frozen = _load("algs", f"td_ambi_full_{target}_qscale_frozen")["alg_params"]
    adaptive = _load("algs", f"td_ambi_full_{target}_qscale_adaptive")["alg_params"]
    ignored = {"wandb_run_name", "wandb_tags", "inner_actor_loss_scale_update"}
    assert {k: v for k, v in frozen.items() if k not in ignored} == {
        k: v for k, v in adaptive.items() if k not in ignored
    }


@pytest.mark.parametrize("category", ("prior", "full"))
def test_suite_manifests_cover_each_requested_run_once(category):
    suite = _load("experiments", f"td_ambi_{category}_study")
    expected = {name for name, case in CASES.items() if case[0] == category}
    assert set(suite["configs"]) == expected
    assert len(suite["configs"]) == len(expected)
    assert suite["trials"] == 1
    assert "alg_params" not in suite.get("overrides_alg", {})
    assert suite["overrides_alg"]["total_steps"] == 2_000_000
    assert suite["checkpoint_every"] == 25_000
    assert suite["save_strat"] == "all"
    assert suite["save_trials"] == "none"
    scheduled_checkpoints = len(expected) * suite["overrides_alg"]["total_steps"] // suite["checkpoint_every"]
    assert scheduled_checkpoints == (320 if category == "prior" else 480)


def test_study_run_names_are_distinct_and_have_current_budget_tags():
    configs = [_load("algs", name) for name in CASES]
    names = [config["alg_params"]["wandb_run_name"] for config in configs]
    assert len(names) == len(set(names))
    stale_tags = {"j8", "n32", "g1", "fixed-update-slots-8", "no-model-checkpoints", "14m-decisions"}
    for name, config in zip(CASES, configs):
        tags = set(config["alg_params"]["wandb_tags"])
        # No inherited description of the old eight-round / 32-branch recipe.
        assert not tags.intersection(stale_tags)
        assert {"2m-decisions", "checkpoints-every25k", "retain-all-checkpoints"} <= tags
        assert config["alg_params"]["wandb_run_name"].endswith("-2m-seed55")
