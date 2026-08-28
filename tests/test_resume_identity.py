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
        "eval_inner_comparison": False,
        "eval_inner_comparison_episodes": 5,
        "eval_inner_comparison_seed": 12345,
        "inner_actor_writeback_coef": 0.0,
        "inner_critic_writeback_coef": 0.0,
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
