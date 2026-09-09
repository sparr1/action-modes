"""TD-AMBI objective selection and retained Humanoid base-v2 protocol."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_config_decoupling import _build_cfg


ROOT = Path(__file__).resolve().parents[1] / "configs/dmcontrol"


def _load(kind, name):
    return json.loads((ROOT / kind / f"{name}.json").read_text())


def test_td_ambi_resolves_tdmpc2_actor_and_critic_objectives():
    config = _load("algs", "TD-AMBI")
    assert config["alg"] == "AMBITDMPC2/AMBITDMPC2"
    cfg = _build_cfg(**config["alg_params"])
    expected = {
        "q_representation": "distributional",
        "num_q": 5,
        "q_num_bins": 101,
        "q_vmin": -10,
        "q_vmax": 10,
        "q_pair_size": 2,
        "outer_q_actor_reduction": "mean_pair",
        "inner_q_actor_reduction": "mean_pair",
        "outer_q_target_reduction": "min_pair",
        "inner_q_target_reduction": "min_pair",
        "outer_actor_entropy_mode": "tdmpc2_scaled",
        "inner_actor_entropy_mode": "tdmpc2_scaled",
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
        "ent_coef": 1e-4,
        "inner_temperature_mode": "inherit_outer",
        "log_std_mapping": "tdmpc2_tanh",
        "inner_log_std_mapping": "tdmpc2_tanh",
        "log_std_min": -10,
        "inner_log_std_min": -10,
        "log_std_max": 2,
        "inner_log_std_max": 2,
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "sac_actor_loss_scale_tau": 0.01,
        "inner_actor_loss_scale_update": "per_update",
        "critic_coef": 0.1,
        "inner_critic_loss_coef": 0.1,
        "inner_critic_target_initialization": "outer_target",
        "rho": 0.5,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "inner_actor_lr": 3e-4,
        "inner_critic_lr": 3e-4,
        "adam_eps": 1e-8,
        "inner_adam_eps": 1e-8,
        "actor_adam_eps": 1e-5,
        "inner_actor_adam_eps": 1e-5,
        "tau": 0.01,
        "inner_critic_target_tau": 0.01,
        "target_update_interval": 1,
        "inner_critic_target_update_interval": 1,
        "inner_temperature_updates_per_action": 0,
        "outer_behavior_policy_kl_schedule": "none",
        "inner_outer_policy_kl_coef": 0.0,
        "inner_outer_action_l2_coef": 0.0,
        "inner_actor_writeback_coef": 0.0,
        "inner_critic_writeback_coef": 0.0,
        "value_equivalence_loss_coef": 0.0,
    }
    assert {key: getattr(cfg, key) for key in expected} == expected
    assert cfg.inner_actor_updates_per_action == 8
    assert cfg.inner_critic_updates_per_action == 8


def test_td_ambi_manifest_preserves_base_v2_training_protocol():
    config = _load("algs", "TD-AMBI")
    baseline = _load("algs", "ambi_humanoid_walk_base_v2")
    manifest = _load("experiments", "TD-AMBI")
    baseline_manifest = _load("experiments", "ambi_humanoid_walk_base_v2")
    for key in ("seed", "env", "device", "total_steps", "episodes"):
        assert config[key] == baseline[key]
    for key in (
        "obs", "model_size", "episodic", "discount", "buffer_size",
        "batch_size", "utd", "train_unroll_horizon", "outer_planning_horizon",
        "inner_rollout_horizon", "compile", "compile_strict", "mpc",
        "eval_freq", "eval_inner_comparison", "eval_value",
        "value_equivalence_diagnostics", "outer_policy_episode_probability",
        "inner_operator", "inner_rounds", "inner_rollouts_per_round",
        "inner_updates_per_round", "inner_batch_size", "inner_replay_capacity",
        "inner_replay_sampling", "inner_diagnostic_rollouts",
        "inner_actor_adaptation", "inner_critic_adaptation",
        "inner_behavior_action", "inner_execution_action",
        "inner_actor_scope", "inner_critic_scope", "inner_replay_scope",
        "inner_actor_optimizer_scope", "inner_critic_optimizer_scope",
    ):
        assert config["alg_params"][key] == baseline["alg_params"][key], key
    expected_manifest = deepcopy(baseline_manifest)
    for key in ("study_type", "study_note", "configs"):
        expected_manifest[key] = manifest[key]
    assert manifest == expected_manifest
    assert manifest["configs"] == ["TD-AMBI"]
    assert "alg_params" not in manifest["overrides_alg"]
    assert "transition minibatches" in manifest["study_note"]
    assert "without sequence-level rho weighting" in manifest["study_note"]
    assert config["alg_params"]["wandb_run_name"] == "TD-AMBI-humanoid-walk-seed55"
    assert "outer-alpha-auto" not in config["alg_params"]["wandb_tags"]
    assert "inner-alpha-auto" not in config["alg_params"]["wandb_tags"]


def test_td_ambi_controls_preserve_existing_default_configuration():
    cfg = _build_cfg()
    assert cfg.inner_actor_adam_eps == cfg.inner_adam_eps == 1e-8
    assert cfg.inner_critic_loss_coef == 1.0
    assert cfg.inner_actor_loss_scale_update == "per_action"
    assert cfg.inner_critic_target_initialization == "online"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"inner_critic_loss_coef": 0.0}, "inner_critic_loss_coef"),
        ({"inner_actor_adam_eps": 0.0}, "inner_actor_adam_eps"),
        ({"inner_actor_loss_scale_update": "always"}, "inner_actor_loss_scale_update"),
        ({"sac_actor_loss_scale_mode": "none"}, "per_update.*requires"),
        ({"inner_critic_scope": "run"}, "action-local cloned critic"),
    ],
)
def test_td_ambi_rejects_incompatible_inner_objective_controls(overrides, message):
    params = _load("algs", "TD-AMBI")["alg_params"]
    with pytest.raises(ValueError, match=message):
        _build_cfg(**{**params, **overrides})


def test_td_ambi_runtime_has_fixed_temperatures_and_actor_adam_epsilon():
    params = _load("algs", "TD-AMBI")["alg_params"]
    # Resolve the real preset, then shrink networks for a CPU runtime check.
    cfg = _build_cfg(**{**params, "compile": False, "wandb": False})
    cfg.enc_dim = cfg.mlp_dim = 32
    cfg.latent_dim = 16
    agent = AMBITDMPC2Agent(cfg)
    assert agent.log_ent_coef is None
    assert agent.ent_coef_optim is None
    assert agent.alpha.item() == pytest.approx(1e-4)
    engine = agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    assert engine.state.log_alpha is None
    assert engine.state.temperature_optim is None
    assert engine.alpha.item() == pytest.approx(1e-4)
    assert agent.pi_optim.param_groups[0]["eps"] == 1e-5
    assert engine.state.actor_optim.param_groups[0]["eps"] == 1e-5
    assert engine.state.critic_optim.param_groups[0]["eps"] == 1e-8
