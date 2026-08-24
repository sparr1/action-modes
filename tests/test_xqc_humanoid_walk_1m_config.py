import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_PATH = ROOT / "configs/dmcontrol/algs/xqc_humanoid_walk_state_1m.json"
MANIFEST_PATH = (
    ROOT / "configs/dmcontrol/experiments/xqc_humanoid_walk_state_1m.json"
)
FROZEN_ALGORITHM_PATH = (
    ROOT / "configs/dmcontrol/algs/xqc_humanoid_walk_state.json"
)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def test_one_million_decision_baseline_preserves_vanilla_xqc():
    config = _load_json(ALGORITHM_PATH)
    frozen = _load_json(FROZEN_ALGORITHM_PATH)
    params = config["alg_params"]

    assert config["seed"] == 55
    assert config["env"] == "DMControl-v0"
    assert config["alg"] == "XQC/XQC"
    assert config["device"] == "cuda"
    assert config["episodes"] is None
    assert config["total_steps"] == params["num_interactions"] == 1_000_000

    scientific_keys = {
        "obs",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_freq",
        "gradient_steps",
        "updates_per_step",
        "learning_rate",
        "actor_lr",
        "critic_lr",
        "lr_end",
        "gamma",
        "tau",
        "target_update_interval",
        "policy_delay",
        "actor_net_arch",
        "critic_net_arch",
        "num_atoms",
        "vmin",
        "vmax",
        "init_temperature",
        "target_entropy",
        "adam_eps",
        "weight_decay",
        "debug_checks",
        "compile",
        "compile_strict",
        "optimizer_backend",
        "eval_freq",
        "eval_episodes",
    }
    assert {key: params[key] for key in scientific_keys} == {
        key: frozen["alg_params"][key] for key in scientific_keys
    }
    assert params["learning_starts"] == 5_000
    assert params["gradient_steps"] == params["updates_per_step"] == 2
    assert params["eval_freq"] == 50_000
    assert params["eval_episodes"] == 10


def test_one_million_decision_baseline_shares_the_ambi_wandb_workspace():
    params = _load_json(ALGORITHM_PATH)["alg_params"]

    assert params["wandb"] is True
    assert params["wandb_mode"] == "online"
    assert params["wandb_entity"] == "rwgao_b-brown-university"
    assert params["wandb_project"] == "ambi"
    assert params["wandb_group"] == "ambixqc-humanoid-walk-state-1m"
    assert params["wandb_env_step_unit"] == "decision"
    assert "wandb_run_name" not in params
    assert {
        "vanilla-xqc",
        "1m-decisions",
        "2m-raw-frames",
        "extended-budget",
        "not-paper-reproduction",
    }.issubset(params["wandb_tags"])


def test_one_million_decision_manifest_is_an_explicit_extended_baseline():
    manifest = _load_json(MANIFEST_PATH)

    assert (
        manifest["study_type"]
        == "xqc_humanoid_walk_single_seed_extended_budget_baseline"
    )
    assert "21 evaluation rows" in manifest["study_note"]
    assert "extends the released 500k-decision budget" in manifest["study_note"]
    assert "not a paper reproduction" in manifest["study_note"]
    assert "Exact trainer resume is not supported" in manifest["study_note"]
    assert manifest["overrides_alg"] == {"env": "DMControl-v0"}
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 1
    assert manifest["configs"] == ["xqc_humanoid_walk_state_1m"]
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 100_000
    assert manifest["save_strat"] == ["best", "latest"]
    assert manifest["checkpoint_best_window"] == 100


def test_one_million_decision_evaluation_grid_keeps_xqc_cadence():
    config = _load_json(ALGORITHM_PATH)
    params = config["alg_params"]
    decisions = [
        1,
        *range(params["eval_freq"], config["total_steps"] + 1, params["eval_freq"]),
    ]

    assert len(decisions) == 21
    assert decisions == [1, *range(50_000, 1_000_001, 50_000)]
    assert [2 * decision for decision in decisions] == [
        2,
        *range(100_000, 2_000_001, 100_000),
    ]
