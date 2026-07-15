import csv
import json

import numpy as np
import pytest

import log as log_module
from log import AMBITrainingLogger, TrainingLogger
from utils.utils import setup_logs


def test_summary_logger_does_not_materialize_or_retain_trajectories(tmp_path, monkeypatch):
    logger = AMBITrainingLogger(
        tmp_path,
        log_info=False,
        log_type="summary",
    )
    step_writer = logger._step_writer

    def fail_conversion(_value):
        raise AssertionError("summary logging materialized trajectory data")

    monkeypatch.setattr(log_module, "_convert_arrays_recursively", fail_conversion)
    monkeypatch.setattr(log_module, "_trajectory_list", fail_conversion)

    logger.on_step(setup_logs(
        reward=1.5,
        obs=np.ones((1, 1024), dtype=np.float32),
        action=np.ones((1, 16), dtype=np.float32),
        dones=[False],
        info=None,
        inner_steps=np.arange(8, dtype=np.int64)[None, :],
        materialize=False,
    ))

    assert logger._step_writer is step_writer
    assert not step_writer._stream.closed
    assert logger.episode_observations == []
    assert logger.episode_actions == []
    assert logger.episode_inner_steps == []
    assert logger.episode_return == 1.5
    assert logger.episode_inner_step_count == 28

    logger.on_step(setup_logs(
        reward=2.5,
        obs=np.zeros((1, 1024), dtype=np.float32),
        action=np.zeros((1, 16), dtype=np.float32),
        dones=[True],
        info=None,
        inner_steps=np.array([[2, 3]], dtype=np.int64),
        materialize=False,
    ))
    logger.close()

    with open(tmp_path / "step_stats.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[-1]["episode_return"] == "4.0"
    assert rows[-1]["inner_steps"] == "5.0"
    assert logger.inner_step_count == 33
    assert list((tmp_path / "train_episodes").iterdir()) == []
    assert "Total Reward = 4.0" in (tmp_path / "stats.txt").read_text()


def test_detailed_logger_preserves_episode_json_schema_and_flushes_worker(tmp_path):
    logger = AMBITrainingLogger(
        tmp_path,
        log_info=True,
        log_type="detailed",
    )
    logger.on_step(setup_logs(
        reward=1.25,
        obs=np.array([[1.0, 2.0]], dtype=np.float32),
        action=np.array([[0.5]], dtype=np.float32),
        dones=[True],
        info=[{"terminated": False, "truncated": True}],
        inner_steps=[[2, 3]],
    ))
    logger.close()

    with open(tmp_path / "train_episodes" / "episode_1.json") as stream:
        payload = json.load(stream)
    assert payload == {
        "rewards": [1.25],
        "observations": [[1.0, 2.0]],
        "actions": [[0.5]],
        "info": [[{"terminated": False, "truncated": True}]],
        "inner_steps": [[2, 3]],
        "cumulative_step": 1,
    }


def test_setup_logs_materializes_by_default_and_native_values_are_opt_in():
    obs = np.array([[1.0, 2.0]], dtype=np.float32)
    action = np.array([[0.5]], dtype=np.float32)
    inner_steps = np.array([[2, 3]], dtype=np.int64)

    materialized = setup_logs(
        reward=np.float32(1.25),
        obs=obs,
        action=action,
        dones=np.array([False]),
        info={"reward_info": {"control": np.float32(-0.5)}},
        inner_steps=inner_steps,
    )

    assert json.loads(json.dumps(materialized)) == materialized
    assert materialized["obs"] == [[1.0, 2.0]]
    assert materialized["actions"] == [[0.5]]
    assert materialized["inner_steps"] == [[2, 3]]

    native = setup_logs(
        reward=1.25,
        obs=obs,
        action=action,
        dones=[False],
        inner_steps=inner_steps,
        materialize=False,
    )
    assert native["obs"] is obs
    assert native["actions"] is action
    assert native["inner_steps"] is inner_steps


def test_basic_logger_keeps_csv_schema_and_close_is_idempotent(tmp_path):
    logger = TrainingLogger(tmp_path, log_info=False, log_type="summary")
    logger.on_step(setup_logs(
        reward=3.0,
        obs=np.array([[1.0]], dtype=np.float32),
        action=np.array([[0.0]], dtype=np.float32),
        dones=[True],
        info=[{"terminated": True, "truncated": False}],
    ))
    logger.close()
    logger.close()

    with open(tmp_path / "step_stats.csv", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == [
        "global_step",
        "episode",
        "episode_step",
        "reward",
        "episode_return",
        "done",
        "terminated",
        "truncated",
        "inner_steps",
    ]
    assert rows[0]["inner_steps"] == ""


def test_logger_closes_trajectory_writer_after_csv_close_failure():
    class FailingStepWriter:
        def __init__(self):
            self.close_calls = 0

        def flush(self):
            raise OSError("first CSV failure")

        def close(self):
            self.close_calls += 1
            raise OSError("second CSV failure")

    class TrackingTrajectoryWriter:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    logger = TrainingLogger.__new__(TrainingLogger)
    logger._closed = False
    step_writer = FailingStepWriter()
    trajectory_writer = TrackingTrajectoryWriter()
    logger._step_writer = step_writer
    logger._trajectory_writer = trajectory_writer

    with pytest.raises(OSError, match="first CSV failure"):
        logger.close()

    assert step_writer.close_calls == 1
    assert trajectory_writer.close_calls == 1
    assert logger._step_writer is None
    assert logger._trajectory_writer is None
    assert logger._closed


def test_buffered_step_writer_closes_stream_after_flush_failure():
    class FailingStream:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def flush(self):
            raise OSError("first flush failure")

        def close(self):
            self.close_calls += 1
            self.closed = True
            raise OSError("second close failure")

    writer = log_module._BufferedStepWriter.__new__(
        log_module._BufferedStepWriter
    )
    stream = FailingStream()
    writer._stream = stream

    with pytest.raises(OSError, match="first flush failure"):
        writer.close()

    assert stream.close_calls == 1
    assert stream.closed
