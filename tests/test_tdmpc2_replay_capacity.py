"""Explicit replay rows are independent of the training decision budget."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.common.buffer import Buffer
from tests.test_ambixqc_core import _tiny_model
from tests.test_ambixqc_replay_archive import _assert_equal, _resident_rows, _view_rows
from tests.test_tdmpc2_pixels import PixelEnv, _make_model, _tiny_params
from tests.test_tdmpc2_training_state import _episode, _replay_cfg
from utils.replay_archive import load_checkpoint_replay


def _cfg(**overrides):
    cfg = _replay_cfg()
    cfg.buffer_size = 10
    cfg.steps = 5
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, 5),
        ({"replay_capacity": None}, 5),
        ({"steps": 20}, 10),
        ({"replay_capacity": 17}, 17),
        ({"replay_capacity": 3}, 3),
        ({"replay_capacity": np.int64(12)}, 12),
    ],
)
def test_replay_capacity_preserves_default_and_overrides_both_limits(fields, expected):
    replay = Buffer(_cfg(**fields))
    assert replay.capacity == expected
    assert not replay.resumable_storage
    assert not hasattr(replay, "_buffer")


def test_legacy_buffer_without_training_budget_keeps_buffer_size():
    cfg = _cfg()
    del cfg.steps
    assert Buffer(cfg).capacity == cfg.buffer_size


@pytest.mark.parametrize(
    "value", [True, False, np.bool_(True), 0, -1, 1.5, 8.0, float("nan"), float("inf"), "20", []]
)
def test_invalid_capacity_is_rejected_by_buffer_and_wrapper_before_agent_creation(value):
    with pytest.raises(ValueError, match="replay_capacity must be a positive integer"):
        Buffer(_cfg(replay_capacity=value))
    wrapper = object.__new__(TDMPC2Baseline)
    wrapper.run_params = {"device": "cpu", "total_steps": 5}
    # Invalid capacity must fail during config resolution before the environment
    # or model is inspected, not much later when the first replay batch arrives.
    with pytest.raises(ValueError, match="replay_capacity must be a positive integer"):
        wrapper._build_cfg({"replay_capacity": value})


def test_explicit_capacity_controls_allocation_ring_eviction_and_metrics(capsys):
    replay = Buffer(_cfg(replay_capacity=13))
    for episode in range(3):
        replay.add(_episode(6, offset=episode * 100))
    assert replay._buffer._storage.max_size == 13
    assert replay._buffer._storage._storage["obs"].shape == (13, 3)
    assert replay.size == 13
    assert replay.num_eps == 3
    assert replay.total_transitions == 15
    assert replay.num_transitions == 11
    assert replay.fill_fraction == 1
    assert _resident_rows(replay)["episode"].tolist() == [0] + [1] * 6 + [2] * 6
    output = capsys.readouterr().out
    assert "Buffer capacity: 13" in output
    assert "Using CPU memory for storage." in output


def test_rgb_footprint_uses_explicit_capacity_without_allocating_replay():
    with pytest.warns(UserWarning, match="36.9 GB.*1,000,000 uint8 rows"):
        model = _make_model(
            TDMPC2Baseline,
            PixelEnv(),
            _tiny_params(buffer_size=10, replay_capacity=1_000_000),
            total_steps=5,
        )
    try:
        assert model.cfg.replay_capacity == 1_000_000
        assert model.cfg.buffer_size == 10
        assert model.buffer.capacity == 1_000_000
        assert not hasattr(model.buffer, "_buffer")
    finally:
        model.flush_checkpoints()
        model.env.close()


def test_explicit_capacity_survives_baseline_budget_rebuild():
    model = _make_model(
        TDMPC2Baseline,
        PixelEnv(),
        _tiny_params(buffer_size=10, replay_capacity=23, seed_steps=100),
        total_steps=5,
    )
    try:
        model.learn(total_timesteps=2)
        assert model.cfg.steps == 2
        assert model.cfg.replay_capacity == 23
        assert model.buffer.capacity == 23
    finally:
        model.flush_checkpoints()
        model.env.close()


def test_exact_replay_resume_validates_resolved_capacity_and_remains_opt_in():
    source = Buffer(_cfg(replay_capacity=13), resumable=True)
    for episode in range(3):
        source.add(_episode(6, offset=episode * 100))
    metadata = source.training_state_metadata()
    shards = list(source.iter_training_state_shards(max_rows=4))
    assert metadata["signature"]["capacity"] == 13

    restored = Buffer(_cfg(buffer_size=999, steps=1000, replay_capacity=13), resumable=True)
    restored.load_training_state_shards(deepcopy(metadata), iter(deepcopy(shards)))
    _assert_equal(_resident_rows(restored), _resident_rows(source))
    random_state = torch.get_rng_state()
    expected = source.sample()
    torch.set_rng_state(random_state)
    _assert_equal(restored.sample(), expected)

    incompatible = Buffer(_cfg(replay_capacity=14), resumable=True)
    with pytest.raises(ValueError, match="signature"):
        incompatible.load_training_state_shards(metadata, iter(shards))
    assert incompatible.size == 0
    assert not hasattr(incompatible, "_buffer")
    ordinary = Buffer(_cfg(replay_capacity=13))
    with pytest.raises(RuntimeError, match="enable_resumable_storage"):
        ordinary.training_state_metadata()


def test_ambixqc_archive_records_explicit_capacity_and_complete_episodes(tmp_path):
    model = _tiny_model(
        replay_capacity=20, buffer_size=3, inner_operator="none", xqc_utd=2
    )
    try:
        assert model.cfg.steps == 10
        assert model.cfg.replay_capacity == model.buffer.capacity == 20
        model.enable_replay_archive(tmp_path, "explicit_capacity")
        model.learn(total_timesteps=10)
        checkpoint = model.save(tmp_path, "explicit_capacity")
        replay = load_checkpoint_replay(checkpoint)
        assert replay.capacity == 20
        assert replay.num_rows == 12
        assert replay.num_transitions == 10
        assert replay.metadata["resident_start"] == 0
        _assert_equal(_view_rows(replay), _resident_rows(model.buffer))
        rows = _view_rows(replay)
        assert rows["episode"].tolist() == [0] * 6 + [1] * 6
        assert torch.isnan(rows["reward"][[0, 6]]).all()
        assert not model.buffer.resumable_storage
    finally:
        model.close_replay_archive()
        model.flush_checkpoints()
        model.env.close()
