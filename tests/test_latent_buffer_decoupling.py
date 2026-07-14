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
    state = replay.state_dict()
    replay.clear()
    assert replay.size == 0
    replay.load_state_dict(state)
    assert replay.size == 6
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
