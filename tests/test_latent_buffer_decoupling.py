import pytest
import torch

from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer


def _buffer(capacity=8):
    replay = LatentReplayBuffer(capacity, latent_dim=2, action_dim=1, device="cpu")
    z = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    replay.add_batch(
        z,
        torch.arange(6, dtype=torch.float32).reshape(6, 1),
        torch.ones(6, 1),
        z + 1,
        torch.zeros(6, 1),
    )
    return replay


def test_latent_replay_without_replacement_returns_unique_indices():
    replay = _buffer()
    generator = torch.Generator().manual_seed(7)
    batch = replay.sample(6, replacement=False, generator=generator)
    assert batch["indices"].unique().numel() == 6
    torch.testing.assert_close(batch["z"][:, 0] / 2, batch["indices"].float())


def test_latent_replay_replacement_is_seeded_and_reports_indices():
    first = _buffer().sample(
        12, replacement=True, generator=torch.Generator().manual_seed(9)
    )
    second = _buffer().sample(
        12, replacement=True, generator=torch.Generator().manual_seed(9)
    )
    torch.testing.assert_close(first["indices"], second["indices"])
    assert first["indices"].numel() == 12


def test_latent_replay_without_replacement_rejects_underfill():
    replay = _buffer()
    with pytest.raises(ValueError, match="without replacement"):
        replay.sample(7, replacement=False)


def test_latent_replay_clear_and_restore():
    replay = _buffer()
    storage_ptr = replay._storage.data_ptr()
    state = replay.state_dict()
    replay.clear()
    assert replay.size == 0
    assert replay._storage.data_ptr() == storage_ptr
    replay.load_state_dict(state)
    assert replay.size == 6
    assert replay._storage.data_ptr() == storage_ptr
    torch.testing.assert_close(replay.z[:6], state["z"])


def test_wrapped_replay_restore_preserves_write_position_and_sample_identity():
    replay = _buffer(capacity=4)
    replay.add_batch(
        torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
        torch.tensor([[10.0], [11.0]]),
        torch.ones(2, 1),
        torch.tensor([[21.0, 22.0], [23.0, 24.0]]),
        torch.zeros(2, 1),
    )
    state = replay.state_dict()
    restored = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    restored.load_state_dict(state)
    assert restored.full is replay.full
    assert restored.pos == replay.pos
    assert restored.next_sample_id == replay.next_sample_id
    torch.testing.assert_close(restored.sample_id, replay.sample_id)

    first = replay.sample(
        8, replacement=True, generator=torch.Generator().manual_seed(17)
    )
    second = restored.sample(
        8, replacement=True, generator=torch.Generator().manual_seed(17)
    )
    torch.testing.assert_close(first["sample_ids"], second["sample_ids"])


def test_dense_round_append_is_horizon_major_and_assigns_stable_ids():
    replay = LatentReplayBuffer(8, latent_dim=2, action_dim=1, device="cpu")
    z = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    action = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    reward = (100 + torch.arange(6, dtype=torch.float32)).reshape(2, 3, 1)
    next_z = z + 50
    terminated = torch.zeros(2, 3, 1)

    replay.add_round(z, action, reward, next_z, terminated)

    assert replay.size == 6
    torch.testing.assert_close(replay.z[:6], z.reshape(6, 2))
    torch.testing.assert_close(replay.action[:6], action.reshape(6, 1))
    torch.testing.assert_close(replay.reward[:6], reward.reshape(6, 1))
    torch.testing.assert_close(replay.sample_id[:6], torch.arange(6))


def test_bulk_round_overflow_matches_sequential_horizon_appends_physically():
    bulk = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    sequential = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    z = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    action = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1)
    reward = (100 + torch.arange(6, dtype=torch.float32)).reshape(3, 2, 1)
    next_z = z + 50
    terminated = torch.zeros(3, 2, 1)

    bulk.add_round(z, action, reward, next_z, terminated)
    for step in range(3):
        sequential.add_batch(
            z[step],
            action[step],
            reward[step],
            next_z[step],
            terminated[step],
        )

    assert bulk.pos == sequential.pos == 2
    assert bulk.full and sequential.full
    assert bulk.next_sample_id == sequential.next_sample_id == 6
    torch.testing.assert_close(bulk._storage, sequential._storage, rtol=0, atol=0)
    torch.testing.assert_close(bulk.sample_id, sequential.sample_id, rtol=0, atol=0)


def test_packed_sample_can_skip_ids_and_use_indices_without_advancing_rng():
    replay = _buffer()
    indices = torch.tensor([5, 1, 3, 1], dtype=torch.long)
    generator = torch.Generator().manual_seed(23)
    rng_state = generator.get_state().clone()

    batch = replay.sample(
        len(indices),
        generator=generator,
        include_ids=False,
        indices=indices,
    )

    assert set(batch) == {"z", "action", "reward", "next_z", "terminated"}
    torch.testing.assert_close(batch["z"], replay.z[indices])
    torch.testing.assert_close(batch["action"], replay.action[indices])
    torch.testing.assert_close(generator.get_state(), rng_state)


def test_lazy_sample_ids_follow_wrapped_physical_slots():
    replay = LatentReplayBuffer(5, latent_dim=2, action_dim=1, device="cpu")
    for start, count in ((0, 3), (3, 4)):
        z = torch.arange(start * 2, (start + count) * 2, dtype=torch.float32).reshape(
            count, 2
        )
        replay.add_batch(
            z,
            torch.arange(start, start + count, dtype=torch.float32).reshape(-1, 1),
            torch.ones(count, 1),
            z + 1,
            torch.zeros(count, 1),
        )

    assert replay.pos == 2
    torch.testing.assert_close(replay.sample_id, torch.tensor([5, 6, 2, 3, 4]))
    physical = torch.arange(replay.capacity)
    batch = replay.sample(replay.capacity, indices=physical)
    torch.testing.assert_close(batch["sample_ids"], replay.sample_id)
