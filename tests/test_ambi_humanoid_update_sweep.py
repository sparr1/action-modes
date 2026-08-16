import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "configs/ambi/algs/ambi_anchor.json"
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
UPDATES = (2, 4, 8, 16)
BASE_G4 = "ambi_humanoid_walk_updates_g4"
LR_CELLS = {
    "ambi_humanoid_walk_updates_g4_critic_lr1e4": {
        "inner_actor_lr": 5e-5,
        "inner_critic_lr": 1e-4,
        "wandb_run_name": (
            "AMBITDMPC2-humanoid-walk-g4-critic-lr1e-4-seed55"
        ),
        "wandb_tags": [
            "ambi",
            "dmcontrol",
            "humanoid-walk",
            "state",
            "inner-learning-rate-screen",
            "g4",
            "critic-lr1e-4",
            "actor-lr5e-5",
        ],
        "study_note": (
            "Single-seed exploratory Humanoid Walk G4 run doubling only the "
            "AMBI inner critic learning rate; make no confidence, significance, "
            "or confirmatory claims."
        ),
    },
    "ambi_humanoid_walk_updates_g4_both_lr1e4": {
        "inner_actor_lr": 1e-4,
        "inner_critic_lr": 1e-4,
        "wandb_run_name": "AMBITDMPC2-humanoid-walk-g4-both-lr1e-4-seed55",
        "wandb_tags": [
            "ambi",
            "dmcontrol",
            "humanoid-walk",
            "state",
            "inner-learning-rate-screen",
            "g4",
            "critic-lr1e-4",
            "actor-lr1e-4",
        ],
        "study_note": (
            "Single-seed exploratory Humanoid Walk G4 run doubling both AMBI "
            "inner actor and critic learning rates; make no confidence, "
            "significance, or confirmatory claims."
        ),
    },
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _name(updates: int) -> str:
    return f"ambi_humanoid_walk_updates_g{updates}"


def test_humanoid_update_sweep_changes_only_task_and_update_dose_from_anchor():
    anchor = _load(ANCHOR)
    run_names = set()

    for updates in UPDATES:
        name = _name(updates)
        actual = _load(ALGORITHM_ROOT / f"{name}.json")
        expected = copy.deepcopy(anchor)
        expected["env"] = "DMControl-v0"
        expected_params = expected["alg_params"]
        expected_params["obs"] = "state"
        expected_params["inner_updates_per_round"] = updates
        expected_params["wandb_mode"] = "online"
        expected_params["wandb_run_name"] = (
            f"AMBITDMPC2-humanoid-walk-g{updates}-seed55"
        )
        expected_params["wandb_tags"] = [
            "ambi",
            "dmcontrol",
            "humanoid-walk",
            "state",
            "inner-updates-screen",
            f"g{updates}",
        ]

        assert actual == expected
        assert "wandb_group" not in actual["alg_params"]
        run_names.add(actual["alg_params"]["wandb_run_name"])

    assert len(run_names) == len(UPDATES)


def test_humanoid_update_sweep_is_log_spaced_and_keeps_anchor_work_budget():
    total_slots = []
    for updates in UPDATES:
        params = _load(ALGORITHM_ROOT / f"{_name(updates)}.json")["alg_params"]
        assert (
            params["inner_rounds"],
            params["inner_rollouts_per_round"],
            params["inner_rollout_horizon"],
            params["inner_batch_size"],
            params["inner_replay_capacity"],
        ) == (2, 32, 3, 64, 192)
        assert params["inner_temperature_mode"] == "inherit_outer"
        assert params["q_representation"] == "distributional"
        assert params["q_pair_size"] == 2
        assert params["compile"] is True
        total_slots.append(params["inner_rounds"] * updates)

    assert total_slots == [4, 8, 16, 32]


def test_each_humanoid_update_cell_is_an_independent_resumable_screen():
    for updates in UPDATES:
        name = _name(updates)
        algorithm = _load(ALGORITHM_ROOT / f"{name}.json")
        experiment = _load(EXPERIMENT_ROOT / f"{name}.json")

        assert algorithm["seed"] == 55
        assert algorithm["total_steps"] == 1_000_000
        assert algorithm["total_steps"] % 500 == 0
        assert experiment["study_type"] == (
            "single_seed_exploratory_inner_update_screening"
        )
        assert "no confidence" in experiment["study_note"].lower()
        assert experiment["overrides_alg"] == {
            "seed": 55,
            "device": "cuda",
            "env": "DMControl-v0",
            "total_steps": 1_000_000,
            "episodes": None,
        }
        assert experiment["env_params"] == {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        }
        assert experiment["trials"] == 1
        assert experiment["configs"] == [name]
        assert experiment["logs"] == "timestamp"
        assert experiment["save_trials"] == "none"
        assert experiment["checkpoint_every"] == 100_000
        assert experiment["save_strat"] == ["best", "latest"]
        assert experiment["checkpoint_best_window"] == 100
        assert experiment["log_type"] == "summary"


def test_g4_learning_rate_cells_change_only_the_selected_inner_optimizers():
    baseline = _load(ALGORITHM_ROOT / f"{BASE_G4}.json")

    for name, cell in LR_CELLS.items():
        actual = _load(ALGORITHM_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected["alg_params"].update(
            {
                "inner_actor_lr": cell["inner_actor_lr"],
                "inner_critic_lr": cell["inner_critic_lr"],
                "wandb_run_name": cell["wandb_run_name"],
                "wandb_tags": cell["wandb_tags"],
            }
        )

        # This full equality keeps the cell a deep copy of G4 except for the
        # two selected optimizer rates and its explicit W&B identity.
        assert actual == expected
        assert actual["alg_params"]["actor_lr"] == 3e-4
        assert actual["alg_params"]["critic_lr"] == 3e-4
        assert actual["alg_params"]["inner_temperature_lr"] == 5e-5
        assert actual["alg_params"]["inner_updates_per_round"] == 4


def test_g4_learning_rate_cells_have_distinct_single_run_wandb_identities():
    algorithms = [
        _load(ALGORITHM_ROOT / f"{name}.json")
        for name in (BASE_G4, *LR_CELLS)
    ]
    run_names = [algorithm["alg_params"]["wandb_run_name"] for algorithm in algorithms]

    assert len(run_names) == len(set(run_names))
    for algorithm in algorithms[1:]:
        params = algorithm["alg_params"]
        tags = set(params["wandb_tags"])
        assert params["wandb"] is True
        assert params["wandb_mode"] == "online"
        assert "wandb_group" not in params
        assert {"ambi", "dmcontrol", "humanoid-walk", "state", "g4"} <= tags
        assert any("lr" in tag.lower() for tag in tags)


def test_g4_learning_rate_cells_reuse_the_resumable_humanoid_protocol():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE_G4}.json")

    for name, cell in LR_CELLS.items():
        actual = _load(EXPERIMENT_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected["study_type"] = (
            "single_seed_exploratory_inner_learning_rate_screening"
        )
        expected["study_note"] = cell["study_note"]
        expected["configs"] = [name]

        assert actual == expected
