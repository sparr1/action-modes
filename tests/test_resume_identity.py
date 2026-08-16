import pytest

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
