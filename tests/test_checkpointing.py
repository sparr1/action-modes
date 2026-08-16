import json
from pathlib import Path

import pytest

from utils.checkpointing import (
    CheckpointTracker,
    normalize_save_strat,
    publish_checkpoint,
    resolve_checkpoint_config,
)


def test_strategy_normalization_and_validation():
    assert normalize_save_strat(None) == ("all",)
    assert normalize_save_strat("last") == ("latest",)
    assert normalize_save_strat(["latest", "best", "last", "best"]) == (
        "best",
        "latest",
    )
    assert normalize_save_strat("none") == ("none",)

    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_save_strat([])
    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_save_strat(["none", "best"])
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_save_strat("newest")
    with pytest.raises(TypeError, match="must be a string"):
        normalize_save_strat(["best", 1])


def test_checkpoint_config_uses_per_algorithm_values_before_experiment():
    config = resolve_checkpoint_config(
        {
            "checkpoint_every": None,
            "save_strat": ["best", "latest"],
            "checkpoint_best_window": 7,
        },
        {
            "checkpoint_every": 100,
            "save_strat": "all",
            "checkpoint_best_window": 20,
        },
    )
    assert config.every is None
    assert config.strategies == ("best", "latest")
    assert config.best_window == 7
    assert not config.enabled

    legacy = resolve_checkpoint_config({}, {"checkpoint_every": 25})
    assert legacy.every == 25
    assert legacy.strategies == ("all",)
    assert legacy.enabled


@pytest.mark.parametrize("key", ["checkpoint_every", "checkpoint_best_window"])
def test_checkpoint_config_rejects_non_positive_or_fractional_numbers(key):
    values = {"checkpoint_every": 10, "checkpoint_best_window": 10}
    for invalid in (0, -1, 1.5, True, "not-a-number"):
        with pytest.raises(ValueError):
            resolve_checkpoint_config({**values, key: invalid}, {})


def test_tracker_uses_partial_finite_window_strict_improvement_and_clean_final(tmp_path):
    tracker = CheckpointTracker(
        10,
        tmp_path,
        "model:trial",
        save_strat=["best", "latest"],
        best_window=2,
    )
    tracker.record_episode_return(2.0)
    first = tracker.targets(10)
    assert [target.kind for target in first] == ["best", "latest"]
    assert tracker.best_score == 2.0

    tracker.record_episode_return(float("nan"))
    tied = tracker.targets(20)
    assert [target.kind for target in tied] == ["latest"]
    assert tracker.episode_count == 2
    assert tracker.recent_returns == (2.0,)

    tracker.record_episode_return(6.0)
    final = tracker.targets(23, final=True)
    assert [target.kind for target in final] == ["best", "latest"]
    assert tracker.best_score == 4.0
    assert final[0].metadata["checkpoint"] == {
        "kind": "best",
        "step": 23,
        "episode": 3,
        "best_score": 4.0,
        "best_window": 2,
    }


def test_numbered_checkpoint_names_are_backend_parameterized(tmp_path):
    default = CheckpointTracker(5, tmp_path, "sac", save_strat="all")
    tdmpc = CheckpointTracker(
        5,
        tmp_path,
        "tdmpc",
        save_strat="all",
        periodic_step_suffix="",
    )
    assert default.targets(5)[0].path.name == "sac_5_steps"
    assert tdmpc.targets(5)[0].path.name == "tdmpc_5"


def test_one_serialization_publishes_multiple_aliases_and_portable_sidecars(tmp_path):
    tracker = CheckpointTracker(
        5,
        tmp_path,
        "model:cfg_0",
        save_strat=["best", "latest"],
        trial_run_params={"seed": 12, "alg": "SAC/SAC"},
        experiment_params={"trials": 1},
    )
    tracker.record_episode_return(3.5)
    targets = tracker.targets(5)
    calls = []

    def serialize(path):
        calls.append(Path(path))
        Path(path).write_bytes(b"same immutable state")

    published = publish_checkpoint(targets, serialize, extension=".pt")
    assert len(calls) == 1
    assert [Path(path).name for path in published] == [
        "model:cfg_0_best.pt",
        "model:cfg_0_latest.pt",
    ]
    assert {Path(path).read_bytes() for path in published} == {b"same immutable state"}
    sidecars = [json.loads(Path(path + ".metadata.json").read_text()) for path in published]
    assert [metadata["checkpoint"]["kind"] for metadata in sidecars] == [
        "best",
        "latest",
    ]
    assert all(metadata["schema_version"] == 1 for metadata in sidecars)
    assert all(metadata["trial_run_params"]["seed"] == 12 for metadata in sidecars)
    assert all(metadata["experiment_params"] == {"trials": 1} for metadata in sidecars)
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.staging.pt"))


def test_fixed_alias_checkpoint_and_sidecar_are_both_replaced(tmp_path):
    tracker = CheckpointTracker(5, tmp_path, "model", save_strat="latest")

    def write_payload(payload):
        def serialize(path):
            Path(path).write_bytes(payload)

        return serialize

    first = tracker.targets(5)
    publish_checkpoint(
        first,
        write_payload(b"step-five"),
        extension=".pt",
    )
    latest = tmp_path / "model_latest.pt"
    assert latest.read_bytes() == b"step-five"

    second = tracker.targets(10)
    publish_checkpoint(
        second,
        write_payload(b"step-ten"),
        extension=".pt",
    )
    assert latest.read_bytes() == b"step-ten"
    metadata = json.loads(Path(f"{latest}.metadata.json").read_text())
    assert metadata["checkpoint"]["kind"] == "latest"
    assert metadata["checkpoint"]["step"] == 10


def test_none_strategy_never_requests_periodic_or_final_publication(tmp_path):
    tracker = CheckpointTracker(5, tmp_path, "model", save_strat="none")
    tracker.record_episode_return(100.0)
    assert tracker.targets(5) == ()
    assert tracker.targets(7, final=True) == ()


def test_checkpoint_tracker_resume_round_trip_and_state_validation(tmp_path):
    tracker = CheckpointTracker(
        5,
        tmp_path,
        "model",
        save_strat=["best", "latest"],
        best_window=2,
    )
    tracker.record_episode_return(1.0)
    tracker.record_episode_return(3.0)
    tracker.targets(5)
    state = tracker.state_dict()

    restored = CheckpointTracker(
        5,
        tmp_path,
        "model",
        save_strat=["best", "latest"],
        best_window=2,
    )
    restored.load_state_dict(state)

    assert restored.episode_count == 2
    assert restored.recent_returns == (1.0, 3.0)
    assert restored.best_score == 2.0

    with pytest.raises(ValueError, match="schema"):
        restored.load_state_dict({**state, "future": 1})
    with pytest.raises(ValueError, match="recent_returns"):
        restored.load_state_dict({**state, "episode_count": 1})


def test_sb3_callback_tracks_returns_and_refreshes_latest_at_clean_end(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("stable_baselines3")
    from RL.baselines import ComposableCheckpointCallback

    tracker = CheckpointTracker(
        5,
        tmp_path,
        "model:sb3_0",
        save_strat=["best", "latest"],
        trial_run_params={"alg": "baselines/PPO"},
        experiment_params={"trials": 1},
    )
    save_calls = []

    class FakeModel:
        n_envs = 1

        def save(self, path):
            save_calls.append(Path(path))
            Path(path).write_bytes(f"save-{len(save_calls)}".encode())

    callback = ComposableCheckpointCallback(tracker)
    callback.model = FakeModel()
    callback.locals = {"reset_num_timesteps": True}
    callback.num_timesteps = 0
    callback._on_training_start()
    callback.locals = {
        "rewards": np.array([3.0]),
        "dones": np.array([True]),
    }
    callback.num_timesteps = 5
    assert callback._on_step()

    assert len(save_calls) == 1
    best = tmp_path / "model:sb3_0_best.zip"
    latest = tmp_path / "model:sb3_0_latest.zip"
    assert best.read_bytes() == b"save-1"
    assert latest.read_bytes() == b"save-1"

    callback.num_timesteps = 6
    callback._on_training_end()
    assert len(save_calls) == 2
    assert best.read_bytes() == b"save-1"
    assert latest.read_bytes() == b"save-2"
    latest_metadata = json.loads(Path(f"{latest}.metadata.json").read_text())
    assert latest_metadata["checkpoint"]["step"] == 6
    assert latest_metadata["checkpoint"]["episode"] == 1
