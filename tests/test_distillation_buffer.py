from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from RL.tdmpc2_core.common.buffer import Buffer


def _cfg(*, teacher=True, behavior=False, capacity=12, batch_size=3):
    return SimpleNamespace(
        device="cpu", buffer_size=capacity, steps=100, batch_size=batch_size,
        train_unroll_horizon=2, multitask=False, obs="state",
        obs_shape={"state": (3,)}, obs_dtype="float32", action_dim=2,
        store_teacher_policy=teacher, store_behavior_policy=behavior,
    )


def _episode(start=0, rows=3, *, labels=False, behavior=False):
    ids = torch.arange(start, start + rows, dtype=torch.float32)
    td = TensorDict({
        "obs": ids[:, None].expand(-1, 3).clone(),
        "action": ids[:, None].expand(-1, 2).clone(),
        "reward": ids.clone(), "terminated": torch.zeros(rows),
    }, batch_size=[rows])
    if labels:
        td["teacher_pre_tanh_mean"] = torch.zeros(rows, 2)
        td["teacher_pre_tanh_mean"][1:] = ids[:-1, None]
        td["teacher_log_std"] = torch.full((rows, 2), -1.0)
        td["teacher_valid"] = torch.arange(rows) > 0
        td["teacher_outer_version"] = torch.full((rows,), 7, dtype=torch.int64)
        td["teacher_outer_version"][0] = -1
    if behavior:
        td["behavior_pre_tanh_mean"] = 1000 + td["action"]
        td["behavior_log_std"] = torch.full((rows, 2), -2.0)
        td["behavior_policy_valid"] = torch.arange(rows) > 0
    return td


def _assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for actual, expected in zip(left, right):
            _assert_equal(actual, expected)
    elif torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        assert left == right


def _unique_positions(teacher):
    seen, positions = set(), []
    for b in range(teacher["index"].shape[1]):
        for t in range(teacher["index"].shape[0]):
            index = int(teacher["index"][t, b])
            if index not in seen:
                seen.add(index)
                positions.append((t, b))
    return torch.tensor(positions, dtype=torch.int64)


@pytest.mark.parametrize("behavior", [False, True])
def test_teacher_disabled_retains_old_schema_and_samples(behavior):
    replay = Buffer(_cfg(teacher=False, behavior=behavior), resumable=True)
    replay.add(_episode(behavior=behavior))
    assert len(replay.sample(include_behavior_policy=behavior)) == (8 if behavior else 5)
    metadata = replay.training_state_metadata()
    assert metadata["version"] == (2 if behavior else 1)
    assert "teacher_policy_replay" not in metadata["signature"]
    assert not any(spec["name"].startswith("teacher_") for spec in metadata["field_specs"])
    with pytest.raises(RuntimeError, match="not enabled"):
        replay.sample_distillation()


@pytest.mark.parametrize("bulk", [False, True])
def test_missing_teacher_labels_initialize_invalid_on_add_and_load(bulk):
    replay = Buffer(_cfg(), resumable=True)
    if bulk:
        replay.load(torch.stack((_episode(), _episode(100))))
    else:
        replay.add(_episode())
    batch, teacher = replay.sample_distillation()
    assert len(batch) == 5
    assert teacher["mean"].shape == teacher["log_std"].shape == (2, 3, 2)
    assert teacher["valid"].shape == teacher["version"].shape == (2, 3, 1)
    assert teacher["index"].shape == teacher["episode"].shape == (2, 3)
    assert not teacher["valid"].any()
    assert (teacher["version"] == -1).all()
    assert (teacher["mean"] == 0).all()


def test_collection_labels_align_with_roots_and_behavior_stays_independent():
    replay = Buffer(_cfg(behavior=True), resumable=True)
    replay.add(_episode(10, labels=True, behavior=True))
    batch, teacher = replay.sample_distillation(include_behavior_policy=True)
    assert len(batch) == 8
    torch.testing.assert_close(teacher["mean"], batch[0][:-1, :, :2])
    torch.testing.assert_close(batch[5], 1000 + batch[1])
    assert teacher["valid"].all() and (teacher["version"] == 7).all()
    storage = replay._buffer._storage._storage
    torch.testing.assert_close(storage["obs"][teacher["index"] - 1], batch[0][:-1])


def test_writeback_refreshes_duplicates_without_changing_replay_or_rng():
    replay = Buffer(_cfg(behavior=True), resumable=True)
    replay.add(_episode(behavior=True))
    _, teacher = replay.sample_distillation(include_behavior_policy=True)
    positions = _unique_positions(teacher)
    assert len(positions) == 2 < teacher["index"].numel()
    storage = replay._buffer._storage._storage
    unchanged = {
        name: value.clone() for name, value in storage.items()
        if not name.startswith("teacher_")
    }
    accounting = replay._accounting_state()
    writer = deepcopy(replay._buffer._writer.state_dict())
    sampler = deepcopy(replay._buffer._sampler.state_dict())
    rng = torch.random.get_rng_state().clone()
    mean = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
    updated = replay.write_teacher_targets(teacher, positions, mean, -mean, 9)
    assert updated["valid"].all() and (updated["version"] == 9).all()
    for row, position in enumerate(positions):
        selected = teacher["index"] == teacher["index"][position[0], position[1]]
        torch.testing.assert_close(updated["mean"][selected], mean[row].expand(selected.sum(), -1))
        torch.testing.assert_close(updated["log_std"][selected], -mean[row].expand(selected.sum(), -1))
    assert not teacher["valid"].any()  # Original sample remains a snapshot.
    for name, value in unchanged.items():
        torch.testing.assert_close(storage[name], value, rtol=0, atol=0, equal_nan=True)
    _assert_equal(replay._accounting_state(), accounting)
    _assert_equal(replay._buffer._writer.state_dict(), writer)
    _assert_equal(replay._buffer._sampler.state_dict(), sampler)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_wrapped_sample_writeback_and_sharded_exact_roundtrip():
    replay = Buffer(_cfg(capacity=7, batch_size=8), resumable=True)
    replay.add(_episode(rows=4))
    replay.add(_episode(100, rows=5, labels=True))
    torch.manual_seed(23)
    batch, teacher = replay.sample_distillation()
    positions = _unique_positions(teacher)
    roots = batch[0][:-1][positions[:, 0], positions[:, 1], :2]
    teacher = replay.write_teacher_targets(teacher, positions, roots + 20, -torch.ones_like(roots), 11)
    assert (teacher["index"] == 0).any()  # Successor crosses physical ring end.
    torch.testing.assert_close(teacher["mean"], batch[0][:-1, :, :2] + 20)
    metadata = replay.training_state_metadata()
    assert metadata["version"] == 3
    assert metadata["signature"]["teacher_policy_replay"] == "successor-row-unscaled-pre-tanh-gaussian-v1"
    shards = list(replay.iter_training_state_shards(max_rows=2))
    restored = Buffer(_cfg(capacity=7, batch_size=8), resumable=True)
    restored.load_training_state_shards(metadata, shards)
    rng = torch.random.get_rng_state()
    expected = replay.sample_distillation()
    torch.random.set_rng_state(rng)
    actual = restored.sample_distillation()
    _assert_equal(actual, expected)


@pytest.mark.parametrize("kind", ["duplicate", "nan", "version", "position"])
def test_invalid_writeback_is_transactional(kind):
    replay = Buffer(_cfg(), resumable=True)
    replay.add(_episode())
    _, teacher = replay.sample_distillation()
    positions = _unique_positions(teacher)
    mean, log_std, version = torch.ones(2, 2), torch.zeros(2, 2), 4
    if kind == "duplicate":
        positions[1] = positions[0]
    elif kind == "nan":
        mean[1, 1] = float("nan")
    elif kind == "version":
        version = -1
    else:
        positions[1, 0] = 100
    before = replay._buffer._storage._storage.clone()
    with pytest.raises(ValueError):
        replay.write_teacher_targets(teacher, positions, mean, log_std, version)
    for key in before.keys():
        torch.testing.assert_close(replay._buffer._storage._storage[key], before[key], rtol=0, atol=0, equal_nan=True)


def test_writeback_rejects_overwritten_successors_and_evicted_predecessors():
    replay = Buffer(_cfg(capacity=6), resumable=True)
    replay.add(_episode(rows=3))
    _, teacher = replay.sample_distillation()
    replay.add(_episode(100, rows=4))  # Evicts row 0 but leaves successor 1.
    with pytest.raises(ValueError, match="predecessor/destination"):
        replay.write_teacher_targets(teacher, _unique_positions(teacher), torch.zeros(2, 2), torch.zeros(2, 2), 1)
    replay.add(_episode(200, rows=3))  # Now evicts the destinations too.
    with pytest.raises(ValueError, match="predecessor/destination"):
        replay.write_teacher_targets(teacher, _unique_positions(teacher), torch.zeros(2, 2), torch.zeros(2, 2), 1)


def test_oldest_row_is_not_a_root_successor_even_when_entire_ring_has_one_episode():
    replay = Buffer(_cfg(capacity=6), resumable=True)
    replay.add(_episode(rows=7))
    _, teacher = replay.sample_distillation()
    oldest = replay._buffer._writer._cursor
    teacher["index"][0, 0] = oldest
    teacher["episode"][0, 0] = 0
    positions = torch.tensor([[0, 0]], dtype=torch.int64)
    with pytest.raises(ValueError, match="predecessor/destination"):
        replay.write_teacher_targets(teacher, positions, torch.zeros(1, 2), torch.zeros(1, 2), 1)


@pytest.mark.parametrize("kind", ["partial", "nan", "invalid_version", "initial"])
def test_collection_label_validation_precedes_replay_initialization(kind):
    replay = Buffer(_cfg(), resumable=True)
    episode = _episode(labels=True)
    if kind == "partial":
        del episode["teacher_log_std"]
    elif kind == "nan":
        episode["teacher_pre_tanh_mean"][1, 0] = float("nan")
    elif kind == "invalid_version":
        episode["teacher_outer_version"][1] = -1
    else:
        episode["teacher_valid"][0] = True
        episode["teacher_outer_version"][0] = 0
    with pytest.raises(ValueError):
        replay.add(episode)
    assert replay.size == 0 and not hasattr(replay, "_buffer")


def test_corrupt_teacher_shard_and_cross_mode_restore_reject_before_mutation():
    source = Buffer(_cfg(), resumable=True)
    source.add(_episode(labels=True))
    metadata = source.training_state_metadata()
    shards = list(source.iter_training_state_shards(max_rows=2))
    disabled = Buffer(_cfg(teacher=False), resumable=True)
    with pytest.raises(ValueError, match="version"):
        disabled.load_training_state_shards(metadata, shards)
    target = Buffer(_cfg(), resumable=True)
    target.add(_episode(100))
    before = target._buffer._storage._storage.clone()
    shards[-1]["fields"]["teacher_log_std"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        target.load_training_state_shards(metadata, shards)
    for key in before.keys():
        torch.testing.assert_close(target._buffer._storage._storage[key], before[key], rtol=0, atol=0, equal_nan=True)
