import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
REFERENCE_NAME = "ambi_humanoid_walk_base_min_all_reward_only_value_calibration"
BASE_NAME = "ambi_humanoid_walk_q5_pair2_base_14m"
VARIANTS = {
    "ambi_humanoid_walk_q5_pair2_rkl_smooth_b0p03_14m": {
        "objective": "reverse_kl",
        "coef": 0.03,
        "objective_tag": "behavior-policy-reverse-kl",
        "coef_tag": "behavior-policy-beta-0p03",
    },
    "ambi_humanoid_walk_q5_pair2_ce_smooth_b0p03_14m": {
        "objective": "action_space_cross_entropy",
        "coef": 0.03,
        "objective_tag": "behavior-policy-action-space-ce",
        "coef_tag": "behavior-policy-beta-0p03",
    },
    "ambi_humanoid_walk_q5_pair2_ce_smooth_b0p1_14m": {
        "objective": "action_space_cross_entropy",
        "coef": 0.1,
        "objective_tag": "behavior-policy-action-space-ce",
        "coef_tag": "behavior-policy-beta-0p1",
    },
}
Q_REDUCTION_KEYS = (
    "outer_q_target_reduction",
    "outer_q_actor_reduction",
    "inner_q_target_reduction",
    "inner_q_actor_reduction",
)
REGULARIZER_DEFAULTS = {
    "outer_behavior_policy_kl_schedule": "smooth",
    "outer_behavior_policy_kl_min_valid_count": "auto",
    "outer_behavior_policy_kl_ramp_updates": 500_000,
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path):
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def _algorithm(name):
    return _load(ALGORITHM_ROOT / f"{name}.json")


def _manifest(name):
    return _load(EXPERIMENT_ROOT / f"{name}.json")


def test_q5_pair2_base_changes_only_the_declared_anchor_controls():
    reference = _algorithm(REFERENCE_NAME)
    actual = _algorithm(BASE_NAME)
    expected = copy.deepcopy(reference)
    expected["total_steps"] = 14_000_000
    expected_params = expected["alg_params"]
    expected_params.update(
        {
            "eval_inner_comparison": True,
            "eval_inner_comparison_episodes": 5,
            "eval_inner_comparison_seed": 12_345,
            "inner_diagnostic_rollouts": 0,
        }
    )
    for key in Q_REDUCTION_KEYS:
        expected_params[key] = "min_pair"

    actual_tags = actual["alg_params"].pop("wandb_tags")
    expected["alg_params"].pop("wandb_tags")
    try:
        assert actual == expected
    finally:
        actual["alg_params"]["wandb_tags"] = actual_tags

    assert len(actual_tags) == len(set(actual_tags))
    assert {
        "q5-pair2-base-v1",
        "14m-decisions",
        "q-min-pair",
        "q-heads-5",
        "q-pair-size-2",
        "outer-critic-lr3e-4",
        "outer-actor-lr3e-4",
        "inner-critic-lr1e-4",
        "inner-actor-lr5e-5",
        "outer-alpha-rho-weighted",
        "figure1-value-calibration",
        "figure2-left-paired-controller",
        "single-seed",
        "seed55",
    } <= set(actual_tags)
    assert not {
        "base-v1",
        "1m-decisions",
        "q-min-all",
        "three-seed",
    }.intersection(actual_tags)


def test_q5_pair2_base_freezes_critic_alpha_and_evaluation_contracts():
    algorithm = _algorithm(BASE_NAME)
    params = algorithm["alg_params"]

    assert algorithm["seed"] == 55
    assert algorithm["total_steps"] == 14_000_000
    assert params["num_q"] == 5
    assert params["q_pair_size"] == 2
    assert {params[key] for key in Q_REDUCTION_KEYS} == {"min_pair"}
    assert params["outer_critic_target"] == "reward_only"
    assert params["inner_sac_critic_target"] == "reward_only"
    assert params["sac_actor_loss_scale_mode"] == "none"
    assert params["ent_coef"] == "auto"
    assert params["inner_temperature_mode"] == "auto"
    assert params["rho"] == 0.5
    assert params["temporal_loss_normalization"] == "reference_weighted_mean"
    assert params["temporal_loss_reference_horizon"] == 3
    assert params["eval_freq"] == 50_000
    assert params["eval_value"] is True
    assert params["eval_value_samples"] == 100
    assert params["eval_value_seed"] == 12_345
    assert params["eval_value_protocols"] == [
        "paper_deterministic",
        "stochastic_bellman",
    ]
    assert params["eval_inner_comparison"] is True
    assert params["eval_inner_comparison_episodes"] == 5
    assert params["eval_inner_comparison_seed"] == 12_345
    assert params["inner_diagnostic_rollouts"] == 0
    assert params["inner_diagnostics_every"] == 1_000
    assert params["wandb"] is True
    assert params["wandb_mode"] == "online"
    assert params["wandb_step_every"] == 1_000
    assert "outer_behavior_policy_kl_schedule" not in params


def test_behavior_regularizer_arms_change_only_objective_and_dose_controls():
    base = _algorithm(BASE_NAME)
    base_params = dict(base["alg_params"])
    base_tags = base_params.pop("wandb_tags")

    for name, expected in VARIANTS.items():
        variant = _algorithm(name)
        assert {
            key: value for key, value in variant.items() if key != "alg_params"
        } == {key: value for key, value in base.items() if key != "alg_params"}

        params = dict(variant["alg_params"])
        tags = params.pop("wandb_tags")
        assert params.pop("outer_behavior_policy_objective") == expected["objective"]
        assert params.pop("outer_behavior_policy_kl_coef") == expected["coef"]
        for key, value in REGULARIZER_DEFAULTS.items():
            assert params.pop(key) == value
        assert not any(
            key in params
            for key in (
                "outer_behavior_policy_kl_q_threshold",
                "outer_behavior_policy_kl_target",
                "outer_behavior_policy_kl_dual_init",
                "outer_behavior_policy_kl_dual_lr",
                "outer_behavior_policy_kl_dual_max",
            )
        )
        assert params == base_params

        added_tags = {
            "behavior-policy-regularizer",
            expected["objective_tag"],
            "behavior-policy-smooth",
            expected["coef_tag"],
            "behavior-policy-ramp-500k",
        }
        assert len(tags) == len(set(tags))
        assert set(tags).difference(base_tags) == added_tags
        assert [tag for tag in tags if tag not in added_tags] == base_tags


def test_behavior_screen_manifests_are_four_independent_seed55_cells():
    names = [BASE_NAME, *VARIANTS]
    for name in names:
        manifest = _manifest(name)
        assert manifest["study_type"] == "single_seed_14m_behavior_regularizer_screen"
        assert manifest["overrides_alg"] == {
            "seed": 55,
            "device": "cuda",
            "env": "DMControl-v0",
            "total_steps": 14_000_000,
            "episodes": None,
        }
        assert manifest["env_params"] == {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        }
        assert manifest["trials"] == 1
        assert manifest["configs"] == [name]
        assert manifest["logs"] == "none"
        assert manifest["save_trials"] == "none"
        assert manifest["checkpoint_every"] == 100_000
        assert manifest["save_strat"] == ["best", "latest"]
        assert manifest["checkpoint_best_window"] == 100
        assert manifest["log_info"] is False
        assert manifest["log_type"] == "summary"
        note = manifest["study_note"].lower()
        assert "seed-55" in note
        assert "one training seed" in note
        assert "confidence or significance claim" in note
        assert "every 50,000 decisions" in note
        assert "100-sample" in note
        assert "seed-12345" in note
        assert "five" in note and "paired real-environment episodes" in note
        assert "q-versus-mc" in note
        assert "fresh action-local inner sac solve" in note
        assert "eval/paired_fresh_inner_minus_outer" in note
        assert "primary improvement signal" in note
        assert "inner_diagnostic_rollouts=0" in note
        assert "fixed-target root-q action-gain" in note
        assert "critic/model optimism or compounded distribution shift" in note
        assert "evaluation-only compute" in note
        assert "timed separately" in note
        assert "do not alter training or consume its decision budget" in note
        if name in VARIANTS:
            expected = VARIANTS[name]
            assert "500,000-eligible-update smoothstep ramp" in note
            assert f"terminal coefficient of {expected['coef']}" in note
            objective_label = (
                "normalized analytic reverse kl"
                if expected["objective"] == "reverse_kl"
                else "exact normalized squashed-action behavior cross-entropy"
            )
            assert objective_label in note
        assert (ALGORITHM_ROOT / f"{name}.json").is_file()
