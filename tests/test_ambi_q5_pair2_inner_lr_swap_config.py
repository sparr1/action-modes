import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGS = ROOT / "configs" / "dmcontrol" / "algs"
EXPERIMENTS = ROOT / "configs" / "dmcontrol" / "experiments"
BASE = "ambi_humanoid_walk_q5_pair2_base_14m"
VARIANT = "ambi_humanoid_walk_q5_pair2_inner_lr_swap_14m_seed55"


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load(path):
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_pairs)


def test_inner_lr_swap_preserves_q5_pair2_training_contract():
    base = _load(ALGS / f"{BASE}.json")
    variant = _load(ALGS / f"{VARIANT}.json")

    assert {key: value for key, value in variant.items() if key != "alg_params"} == {
        key: value for key, value in base.items() if key != "alg_params"
    }

    base_params = dict(base["alg_params"])
    params = dict(variant["alg_params"])
    base_params.pop("wandb_tags")
    params.pop("wandb_tags")

    expected_changes = {
        "inner_actor_lr": 1e-4,
        "inner_critic_lr": 5e-5,
        "value_equivalence_diagnostics": True,
        "value_equivalence_every_updates": 1000,
        "value_equivalence_mc_samples": 4,
    }
    for key, expected in expected_changes.items():
        assert params.pop(key) == expected
        base_params.pop(key, None)
    assert params == base_params

    variant_params = variant["alg_params"]
    assert variant_params["num_q"] == 5
    assert variant_params["q_pair_size"] == 2
    assert {
        variant_params[key]
        for key in (
            "outer_q_target_reduction",
            "outer_q_actor_reduction",
            "inner_q_target_reduction",
            "inner_q_actor_reduction",
        )
    } == {"min_pair"}
    assert variant_params["actor_lr"] == 3e-4
    assert variant_params["critic_lr"] == 3e-4
    assert variant_params["inner_temperature_lr"] == 5e-5


def test_inner_lr_swap_enables_all_observational_logging():
    params = _load(ALGS / f"{VARIANT}.json")["alg_params"]

    assert params["wandb"] is True
    assert params["wandb_mode"] == "online"
    assert params["wandb_step_every"] == 1000
    assert params["inner_diagnostics_every"] == 1000
    assert params["inner_diagnostic_rollouts"] == 0
    assert params["value_equivalence_diagnostics"] is True
    assert params["value_equivalence_every_updates"] == 1000
    assert params["value_equivalence_mc_samples"] == 4
    assert "value_equivalence_loss_coef" not in params
    assert params["eval_freq"] == 50_000
    assert params["eval_value"] is True
    assert params["eval_value_samples"] == 100
    assert params["eval_value_protocols"] == [
        "paper_deterministic",
        "stochastic_bellman",
    ]
    assert params["eval_inner_comparison"] is True
    assert params["eval_inner_comparison_episodes"] == 5

    tags = params["wandb_tags"]
    assert len(tags) == len(set(tags))
    for tag in (
        "inner-critic-lr5e-5",
        "inner-actor-lr1e-4",
        "inner-final-outer-policy-kl-every-1k",
        "actor-ensemble-gap",
        "value-equivalence-diagnostics",
        "figure1-value-calibration",
        "figure2-left-paired-controller",
    ):
        assert tag in tags


def test_inner_lr_swap_manifest_is_one_trial_resumable_14m_run():
    manifest = _load(EXPERIMENTS / f"{VARIANT}.json")

    assert manifest["configs"] == [VARIANT]
    assert manifest["trials"] == 1
    assert manifest["overrides_alg"]["seed"] == 55
    assert manifest["overrides_alg"]["total_steps"] == 14_000_000
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 100_000
    assert manifest["save_strat"] == ["best", "latest"]
    assert manifest["log_info"] is False
    assert manifest["log_type"] == "summary"
