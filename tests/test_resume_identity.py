import hashlib

import pytest

import utils.resume_identity as resume_identity
from utils.resume_identity import (
    ResumeConfigurationError,
    canonical_json,
    scientific_trial_parameters,
    validate_resume_selection,
)


def test_canonical_json_is_mapping_order_independent():
    left = {"b": [2, 3], "a": {"z": 1}}
    right = {"a": {"z": 1}, "b": [2, 3]}
    assert canonical_json(left) == canonical_json(right)


def test_scientific_projection_excludes_only_the_segment_eval_destination():
    left = {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "seed": 7,
        "alg_params": {
            "eval_csv_path": "/allocation-a/segment-0/eval.csv",
            "pretrained_model_path": "/scientific/input/model.pt",
            "batch_size": 256,
        },
    }
    right = {
        **left,
        "alg_params": {
            **left["alg_params"],
            "eval_csv_path": "/allocation-b/segment-4/eval.csv",
        },
    }
    assert scientific_trial_parameters(left) == scientific_trial_parameters(right)
    changed_input = {
        **right,
        "alg_params": {
            **right["alg_params"],
            "pretrained_model_path": "/different/scientific/model.pt",
        },
    }
    assert scientific_trial_parameters(left) != scientific_trial_parameters(
        changed_input
    )


def test_ambi_inner_comparison_defaults_canonicalize_scientific_lineage():
    omitted = {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "seed": 7,
        "alg_params": {"batch_size": 256},
    }
    explicit = {
        **omitted,
        "alg_params": {
            **omitted["alg_params"],
            "eval_inner_comparison": False,
            "eval_inner_comparison_episodes": 5,
            "eval_inner_comparison_seed": 12345,
        },
    }

    omitted_projection = scientific_trial_parameters(omitted)
    assert omitted_projection == scientific_trial_parameters(explicit)
    assert omitted_projection["alg_params"] == {
        "batch_size": 256,
        "inner_finite_horizon": False,
        "inner_steps_per_update": None,
        "inner_outer_replay_fraction": 0.0,
        "eval_inner_comparison": False,
        "eval_inner_comparison_episodes": 5,
        "eval_inner_comparison_seed": 12345,
        "inner_actor_writeback_coef": 0.0,
        "inner_critic_writeback_coef": 0.0,
        "inner_explorer_mode": "none",
        "inner_prior_rollout_weight": 0.5,
        "inner_behavior_action": "policy_sample",
        "inner_behavior_std_scale": 1.0,
        "inner_log_std_mapping": "direct_clamp",
        "inner_log_std_min": -20.0,
        "inner_log_std_max": 2.0,
        "inner_mixture_target_estimator": "stratified",
        "inner_explorer_actor_updates_per_round": None,
        "inner_explorer_critic_updates_per_round": None,
        "inner_explorer_temperature_updates_per_round": None,
        "inner_param_noise_actor_count": None,
        "inner_param_noise_target_action_rms": 0.1,
        "inner_param_noise_sigma_init": 1e-3,
        "inner_param_noise_sigma_min": 1e-6,
        "inner_param_noise_sigma_max": 0.1,
        "inner_param_noise_calibration_directions": 8,
        "inner_param_noise_calibration_batch_size": 32,
        "inner_param_noise_calibration_max_probes": 8,
        "inner_execution_policy_source": "primary",
        "inner_execution_handoff_samples": 8,
    }

    for field, value in (
        ("eval_inner_comparison", True),
        ("eval_inner_comparison_episodes", 6),
        ("eval_inner_comparison_seed", 12346),
    ):
        changed = {
            **explicit,
            "alg_params": {**explicit["alg_params"], field: value},
        }
        assert scientific_trial_parameters(changed) != omitted_projection


def test_sac_actor_loss_scale_fields_change_lineage_fingerprint(monkeypatch):
    monkeypatch.setattr(
        resume_identity,
        "source_identity",
        lambda _repo_root: {"commit": "test", "dirty": False},
    )
    monkeypatch.setattr(
        resume_identity,
        "dependency_identity",
        lambda: {"python": "test"},
    )
    base = {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "seed": 7,
        "alg_params": {
            "sac_actor_loss_scale_mode": "none",
            "sac_actor_loss_scale_tau": 0.01,
        },
    }
    changed_mode = {
        **base,
        "alg_params": {
            **base["alg_params"],
            "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        },
    }
    changed_tau = {
        **base,
        "alg_params": {
            **base["alg_params"],
            "sac_actor_loss_scale_tau": 0.02,
        },
    }

    def fingerprint(trial_run_params):
        return resume_identity.lineage_identity(
            trial_run_params=trial_run_params,
            experiment_params={"exp_name": "test"},
            repo_root=".",
        )["fingerprint"]

    assert fingerprint(base) != fingerprint(changed_mode)
    assert fingerprint(base) != fingerprint(changed_tau)


def test_separate_inner_update_counts_change_lineage_fingerprint(monkeypatch):
    monkeypatch.setattr(
        resume_identity,
        "source_identity",
        lambda _repo_root: {"commit": "test", "dirty": False},
    )
    monkeypatch.setattr(
        resume_identity,
        "dependency_identity",
        lambda: {"python": "test"},
    )
    base = {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "seed": 7,
        "alg_params": {
            "inner_rounds": 8,
            "inner_rollouts_per_round": 512,
            "inner_rollout_horizon": 3,
            "inner_critic_updates_per_round": 3,
            "inner_actor_updates_per_round": 1,
        },
    }
    changed_critic = {
        **base,
        "alg_params": {
            **base["alg_params"],
            "inner_critic_updates_per_round": 4,
        },
    }
    changed_actor = {
        **base,
        "alg_params": {
            **base["alg_params"],
            "inner_actor_updates_per_round": 2,
        },
    }

    def fingerprint(trial_run_params):
        return resume_identity.lineage_identity(
            trial_run_params=trial_run_params,
            experiment_params={"exp_name": "test"},
            repo_root=".",
        )["fingerprint"]

    assert fingerprint(base) != fingerprint(changed_critic)
    assert fingerprint(base) != fingerprint(changed_actor)


def test_resume_selection_is_strict_and_resource_neutral():
    validate_resume_selection(
        algorithm="AMBITDMPC2/AMBITDMPC2",
        observation_mode="state",
        num_runs=1,
        save_trials="none",
        checkpoint_minutes=60,
        drain_after_seconds=100,
    )

    with pytest.raises(ResumeConfigurationError, match="--num-runs 1"):
        validate_resume_selection(
            algorithm="AMBITDMPC2/AMBITDMPC2",
            observation_mode="state",
            num_runs=-1,
            save_trials="none",
            checkpoint_minutes=60,
            drain_after_seconds=None,
        )
    with pytest.raises(ResumeConfigurationError, match="state observations"):
        validate_resume_selection(
            algorithm="TDMPC2/TDMPC2Baseline",
            observation_mode="rgb",
            num_runs=1,
            save_trials="none",
            checkpoint_minutes=60,
            drain_after_seconds=None,
        )
    with pytest.raises(ResumeConfigurationError, match="does not implement"):
        validate_resume_selection(
            algorithm="SAC/SAC",
            observation_mode="state",
            num_runs=1,
            save_trials="none",
            checkpoint_minutes=60,
            drain_after_seconds=None,
        )
    for policy in ("first", "all", "best"):
        with pytest.raises(ResumeConfigurationError, match="save_trials='none'"):
            validate_resume_selection(
                algorithm="AMBITDMPC2/AMBITDMPC2",
                observation_mode="state",
                num_runs=1,
                save_trials=policy,
                checkpoint_minutes=60,
                drain_after_seconds=None,
            )


def _lora_trial(params):
    return {"alg": "AMBITDMPC2/AMBITDMPC2", "seed": 7, "alg_params": params}


@pytest.mark.parametrize("adaptation,actor_scale,critic_rank,expected", [
    ("clone", 1.0, 8, "82d9ca518cd512c7d324cf761e0daa783408d648f1db588badaf7aa181cedbcc"),
    ("lora", 2.0, 16, "01ada266adb16675cb559dd0461dbc6e54bf7c7e53d52c69023c2ede7238f95d"),
])
def test_pre_lora_rl_lineage_projection_hashes_are_unchanged(
    adaptation, actor_scale, critic_rank, expected
):
    # These hashes were captured before adding lora_rl. Raw dense configurations
    # sometimes contained inactive adapter defaults and must keep their lineage.
    trial = _lora_trial({
        "inner_operator": "sac",
        "inner_actor_adaptation": adaptation,
        "inner_critic_adaptation": adaptation,
        "inner_actor_lora_rank": 8,
        "inner_actor_lora_scale": actor_scale,
        "inner_actor_lora_dropout": 0.0,
        "inner_critic_lora_rank": critic_rank,
        "inner_critic_lora_scale": actor_scale,
        "inner_critic_lora_dropout": 0.0,
    })
    projected = scientific_trial_parameters(trial)
    assert hashlib.sha256(canonical_json(projected).encode()).hexdigest() == expected


def test_lora_rl_lineage_omitted_and_explicit_defaults_match():
    params = {"inner_operator": "sac", "inner_critic_adaptation": "lora_rl"}
    explicit = {
        **params,
        "inner_actor_adaptation": "CLONE",
        "inner_critic_adaptation": "LORA_RL",
        "inner_critic_lora_layers": "INPUT_HIDDEN",
        "inner_critic_lora_rank": 96,
        "inner_critic_lora_scale": 1,
        "inner_critic_lora_weight_decay": 0.0002,
        "inner_actor_lora_rank": 8,
        "inner_actor_lora_scale": 1.0,
        "inner_actor_lora_dropout": 0.0,
        "inner_critic_lora_dropout": 0.0,
        "lora_alpha": 8.0,
    }
    assert scientific_trial_parameters(_lora_trial(params)) == scientific_trial_parameters(
        _lora_trial(explicit)
    )
    assert params == {"inner_operator": "sac", "inner_critic_adaptation": "lora_rl"}


@pytest.mark.parametrize("field,value", [
    ("inner_critic_adaptation", "lora"),
    ("inner_critic_lora_layers", "hidden"),
    ("inner_critic_lora_rank", 48),
    ("inner_critic_lora_scale", 2.0),
    ("inner_critic_lora_weight_decay", 0.0),
])
def test_lora_rl_scientific_choices_split_lineages(field, value):
    params = {"inner_operator": "sac", "inner_critic_adaptation": "lora_rl"}
    baseline = scientific_trial_parameters(_lora_trial(params))
    changed = scientific_trial_parameters(_lora_trial({**params, field: value}))
    assert canonical_json(baseline) != canonical_json(changed)
