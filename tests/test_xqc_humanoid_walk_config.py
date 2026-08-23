import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_PATH = (
    ROOT / "configs/dmcontrol/algs/xqc_humanoid_walk_state.json"
)
MANIFEST_PATH = (
    ROOT / "configs/dmcontrol/experiments/xqc_humanoid_walk_state.json"
)
WALKER_ALGORITHM_PATH = (
    ROOT / "configs/dmcontrol/algs/xqc_walker_walk_state.json"
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


def test_humanoid_walk_uses_the_faithful_full_xqc_configuration():
    config = _load_json(ALGORITHM_PATH)
    walker = _load_json(WALKER_ALGORITHM_PATH)

    # Task selection belongs to the manifest. The scientific learner profile
    # is the already-validated full XQC profile; only the paper seed and the
    # experiment's online logging destination differ.
    expected = dict(walker)
    expected["seed"] = 0
    expected["alg_params"] = dict(walker["alg_params"])
    expected["alg_params"].update(
        {
            "wandb": True,
            "wandb_entity": "rwgao_b-brown-university",
            "wandb_project": "ambi_humanoid",
            "wandb_mode": "online",
        }
    )
    assert config == expected

    params = config["alg_params"]
    assert config["alg"] == "XQC/XQC"
    assert config["device"] == "cuda"
    assert config["total_steps"] == params["num_interactions"] == 500_000
    assert params["learning_starts"] == 5_000
    assert params["gradient_steps"] == params["updates_per_step"] == 2
    assert params["eval_freq"] == 50_000
    assert params["eval_episodes"] == 10
    assert params["wandb"] is True
    assert params["wandb_mode"] == "online"
    assert params["wandb_project"] == "ambi_humanoid"
    assert params["wandb_entity"] == "rwgao_b-brown-university"
    assert "wandb_group" not in params
    assert "wandb_tags" not in params


def test_humanoid_walk_manifest_is_an_artifact_free_two_seed_parity_check():
    config = _load_json(ALGORITHM_PATH)
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["study_type"] == "xqc_humanoid_walk_two_seed_parity"
    assert "seeds 0 and 1" in manifest["study_note"]
    assert "ten-seed paper curve" in manifest["study_note"]
    assert "not a statistical reproduction" in manifest["study_note"]
    assert manifest["overrides_alg"] == {"env": "DMControl-v0"}
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 2
    assert [config["seed"] + trial for trial in range(manifest["trials"])] == [
        0,
        1,
    ]
    assert manifest["configs"] == ["xqc_humanoid_walk_state"]
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] is None
    assert manifest["save_strat"] == "none"
    assert "checkpoint_best_window" not in manifest
    assert manifest["log_info"] is False
    assert manifest["log_type"] == "summary"


def test_humanoid_walk_curve_uses_the_paper_frame_grid():
    config = _load_json(ALGORITHM_PATH)
    params = config["alg_params"]

    decision_steps = [1, *range(params["eval_freq"], config["total_steps"] + 1, params["eval_freq"])]
    paper_frame_steps = [
        0 if decision == 1 else 2 * decision for decision in decision_steps
    ]

    assert paper_frame_steps == list(range(0, 1_000_001, 100_000))
