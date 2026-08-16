import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "configs/ambi/algs/ambi_anchor.json"
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
UPDATES = (2, 4, 8, 16)


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
