import pytest
import torch

from RL.tdmpc2_core.common.search_replay import SearchTrajectoryReplayBuffer


def _trajectories(start, count, *, horizon=3, valid=None, round_id=0):
    base = torch.arange(
        start,
        start + count * horizon,
        dtype=torch.float32,
    ).reshape(count, horizon, 1)
    z = torch.cat((base, base + 100), dim=-1)
    if valid is None:
        valid = torch.ones(count, horizon, 1, dtype=torch.bool)
    return {
        "z": z,
        "action": base + 10,
        "pre_tanh_action": base + 20,
        "reward": base + 30,
        "next_z": z + 1,
        "terminated": torch.zeros(count, horizon, 1, dtype=torch.bool),
        "valid": valid,
        "behavior_log_prob": -(base + 1) / 10,
        "round_id": round_id,
    }


def _buffer(capacity=6):
    return SearchTrajectoryReplayBuffer(
        capacity,
        horizon=3,
        latent_dim=2,
        action_dim=1,
        device="cpu",
    )


def test_search_replay_capacity_is_an_integer_number_of_trajectories():
    with pytest.raises(ValueError, match="multiple of the horizon"):
        SearchTrajectoryReplayBuffer(7, 3, 2, 1, "cpu")


def test_search_replay_wrap_evicts_one_complete_trajectory():
    replay = _buffer()
    replay.add_trajectories(**_trajectories(0, 2, round_id=4))
    replay.add_trajectories(**_trajectories(100, 1, round_id=5))

    assert replay.full
    assert replay.pos == 1
    assert replay.size == 2
    torch.testing.assert_close(replay.trajectory_id, torch.tensor([2, 1]))
    # Physical slot zero is entirely the new trajectory; slot one is entirely
    # the second old trajectory.  No horizon row was independently evicted.
    torch.testing.assert_close(
        replay.reward[0, :, 0], torch.tensor([130.0, 131.0, 132.0])
    )
    torch.testing.assert_close(
        replay.reward[1, :, 0], torch.tensor([33.0, 34.0, 35.0])
    )
    assert torch.equal(replay.round_id[0], torch.full((3, 1), 5))
    assert torch.equal(replay.round_id[1], torch.full((3, 1), 4))


def test_search_replay_bulk_overflow_matches_sequential_trajectory_appends():
    bulk, sequential = _buffer(), _buffer()
    fields = _trajectories(0, 5, round_id=7)
    bulk.add_trajectories(**fields)
    for index in range(5):
        one = {
            name: value[index : index + 1] if torch.is_tensor(value) else value
            for name, value in fields.items()
        }
        sequential.add_trajectories(**one)

    assert bulk.pos == sequential.pos == 1
    assert bulk.next_trajectory_id == sequential.next_trajectory_id == 5
    for name in (
        "z",
        "action",
        "pre_tanh_action",
        "reward",
        "next_z",
        "terminated",
        "valid",
        "behavior_log_prob",
        "round_id",
        "remaining_horizon",
    ):
        torch.testing.assert_close(getattr(bulk, name), getattr(sequential, name))


def test_search_replay_bulk_overflow_from_nonzero_cursor_matches_sequential():
    # Exercise the tricky case where a single incoming batch is larger than
    # the entire ring and the write cursor is already nonzero.  Retaining the
    # newest trajectories is not enough: they must land in the same physical
    # slots, chronological order, and cursor position as one-at-a-time writes.
    bulk, sequential = _buffer(capacity=9), _buffer(capacity=9)
    initial = _trajectories(0, 2, round_id=6)
    incoming = _trajectories(100, 5, round_id=7)
    bulk.add_trajectories(**initial)
    sequential.add_trajectories(**initial)

    bulk.add_trajectories(**incoming)
    for index in range(5):
        one = {
            name: value[index : index + 1] if torch.is_tensor(value) else value
            for name, value in incoming.items()
        }
        sequential.add_trajectories(**one)

    assert bulk.full and sequential.full
    assert bulk.pos == sequential.pos == 1
    assert bulk.next_trajectory_id == sequential.next_trajectory_id == 7
    torch.testing.assert_close(bulk.trajectory_id, torch.tensor([6, 4, 5]))
    for name in (
        "z",
        "action",
        "pre_tanh_action",
        "reward",
        "next_z",
        "terminated",
        "valid",
        "behavior_log_prob",
        "round_id",
        "remaining_horizon",
    ):
        torch.testing.assert_close(getattr(bulk, name), getattr(sequential, name))


def test_search_replay_clear_preserves_globally_unique_sample_identities():
    replay = _buffer()
    replay.add_trajectories(**_trajectories(0, 2, round_id=11))
    before = replay.sample_trajectories(
        2, replacement=False, indices=torch.tensor([0, 1])
    )

    replay.clear()
    replay.add_trajectories(**_trajectories(100, 2, round_id=12))
    after = replay.sample_trajectories(
        2, replacement=False, indices=torch.tensor([0, 1])
    )

    assert before["trajectory_ids"].tolist() == [0, 1]
    assert after["trajectory_ids"].tolist() == [2, 3]
    assert set(before["trajectory_ids"].tolist()).isdisjoint(
        after["trajectory_ids"].tolist()
    )
    assert before["round_id"][:, 0, 0].tolist() == [11, 11]
    assert after["round_id"][:, 0, 0].tolist() == [12, 12]

    # Identity continuity is training state, not merely a logging detail.
    restored = _buffer()
    restored.load_training_state_dict(replay.training_state_dict())
    restored.clear()
    restored.add_trajectories(**_trajectories(200, 1, round_id=13))
    assert restored.sample_trajectories(
        1, indices=torch.tensor([0])
    )["trajectory_ids"].item() == 4


def test_search_replay_anchor_sampling_returns_masked_fixed_shape_suffix():
    replay = _buffer()
    valid = torch.tensor(
        [[[True], [True], [False]], [[True], [True], [True]]]
    )
    replay.add_trajectories(**_trajectories(0, 2, valid=valid, round_id=11))

    # Valid candidates are (trajectory,time)=(0,0),(0,1),(1,0),(1,1),(1,2).
    batch = replay.sample_anchors(
        2,
        candidate_indices=torch.tensor([1, 3]),
        include_ids=True,
    )

    assert batch["z"].shape == (2, 3, 2)
    assert batch["anchor_time"].tolist() == [1, 1]
    assert batch["valid"][:, :, 0].tolist() == [
        [True, False, False],
        [True, True, False],
    ]
    torch.testing.assert_close(batch["reward"][0, :, 0], torch.tensor([31.0, 0.0, 0.0]))
    torch.testing.assert_close(batch["reward"][1, :, 0], torch.tensor([34.0, 35.0, 0.0]))
    torch.testing.assert_close(
        batch["remaining_horizon"][:, :, 0],
        torch.tensor([[2, 0, 0], [2, 1, 0]]),
    )
    assert batch["trajectory_ids"].tolist() == [0, 1]


def test_search_replay_horizon_major_round_and_seeded_sampling():
    replay = _buffer(capacity=9)
    fields = _trajectories(0, 3, round_id=2)
    horizon_major = {
        name: value.transpose(0, 1) if torch.is_tensor(value) else value
        for name, value in fields.items()
    }
    replay.add_round(horizon_major=True, **horizon_major)

    first = replay.sample_trajectories(
        8, generator=torch.Generator().manual_seed(13)
    )
    second = replay.sample_trajectories(
        8, generator=torch.Generator().manual_seed(13)
    )
    torch.testing.assert_close(first["trajectory_indices"], second["trajectory_indices"])
    torch.testing.assert_close(first["z"], second["z"])
    assert replay.transition_size == 9


def test_search_replay_clear_reuses_storage_and_state_round_trips_wrapped_ring():
    replay = _buffer()
    replay.add_trajectories(**_trajectories(0, 3, round_id=1))
    state = replay.training_state_dict()
    storage_pointer = replay.z.data_ptr()
    replay.clear()
    assert replay.size == 0
    assert replay.z.data_ptr() == storage_pointer

    replay.load_training_state_dict(state)
    restored = _buffer()
    restored.load_training_state_dict(state)
    assert replay.pos == restored.pos == 1
    assert replay.full and restored.full
    assert replay.next_trajectory_id == restored.next_trajectory_id == 3
    for name in (
        "z",
        "action",
        "pre_tanh_action",
        "reward",
        "next_z",
        "terminated",
        "valid",
        "behavior_log_prob",
        "round_id",
        "remaining_horizon",
        "trajectory_id",
    ):
        torch.testing.assert_close(getattr(replay, name), getattr(restored, name))


def test_search_replay_rejects_nonprefix_valid_mask_and_padded_termination():
    replay = _buffer()
    gap = torch.tensor([[[True], [False], [True]]])
    with pytest.raises(ValueError, match="prefix masks"):
        replay.add_trajectories(**_trajectories(0, 1, valid=gap))

    fields = _trajectories(
        0, 1, valid=torch.tensor([[[True], [False], [False]]])
    )
    fields["terminated"][0, 1] = True
    with pytest.raises(ValueError, match="padded.*terminated"):
        replay.add_trajectories(**fields)

    fields = _trajectories(0, 1)
    fields["terminated"][0, 1] = True
    with pytest.raises(ValueError, match="after termination"):
        replay.add_trajectories(**fields)
