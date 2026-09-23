"""Whole-trajectory ownership, densities, and exact replay lifecycle for Retrace."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common.latent_trajectory_buffer import LatentTrajectoryReplayBuffer


def _replay(capacity=10, horizon=3, conditioned=True):
    return LatentTrajectoryReplayBuffer(
        capacity, 2, 1, "cpu", horizon=horizon,
        horizon_conditioning_horizon=horizon if conditioned else None,
    )

def _batch(labels, lengths=None, *, horizon=3, conditioned=True):
    n = len(labels)
    lengths = [horizon] * n if lengths is None else lengths
    steps = torch.arange(horizon).reshape(1, horizon, 1)
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(n, 1, 1)
    base = labels * 100 + steps
    z = torch.cat((base, base + 0.5), dim=-1)
    valid = steps < torch.tensor(lengths).reshape(n, 1, 1)
    unsquashed = labels.expand(n, horizon, 1) / 10 + steps / 20
    terminated = torch.zeros(n, horizon, 1)
    for i, length in enumerate(lengths):
        if length < horizon:
            terminated[i, length - 1] = 1
    result = {
        "z": z, "action": unsquashed.tanh(), "reward": base + 10,
        "next_z": z + 1, "terminated": terminated,
        "horizon_end": ((steps == horizon - 1) & valid).float(),
        "pre_tanh_action": unsquashed,
        "behavior_log_prob": (-1 - labels).expand(n, horizon, 1).clone(),
        "valid": valid,
    }
    if conditioned:
        result["remaining_horizon"] = (horizon - steps).expand(n, horizon, 1).float()
    return result


def _assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for name, value in first.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, second[name])
        else:
            assert value == second[name], name


@pytest.mark.parametrize("conditioned", [False, True])
def test_trajectory_sampling_keeps_branches_densities_and_all_valid_suffixes(conditioned):
    replay = _replay(conditioned=conditioned)
    source = _batch([1, 2, 3], [3, 1, 2], conditioned=conditioned)
    replay.add_trajectories(source)
    assert replay.requested_capacity == 10
    assert replay.capacity == 9 and replay.trajectory_capacity == 3
    assert replay.size == 6 and replay.trajectory_count == 3
    batch = replay.sample_trajectories(3, indices=torch.tensor([2, 0, 1]), include_ids=True)
    assert batch["valid"].sum(dim=1).flatten().tolist() == [2, 3, 1]
    assert batch["trajectory_ids"].tolist() == [2, 0, 1]
    assert batch["sample_ids"].tolist() == [[4, 5, -1], [0, 1, 2], [3, -1, -1]]
    for sample, original in enumerate([2, 0, 1]):
        valid = source["valid"][original, :, 0]
        for field in ("z", "action", "reward", "next_z", "pre_tanh_action", "behavior_log_prob"):
            torch.testing.assert_close(batch[field][sample, valid], source[field][original, valid])
        # Every real step is available for its own suffix target, including the last.
        assert batch["reward"][sample, valid].shape[0] == [2, 3, 1][sample]


def test_transition_sampling_is_over_valid_rows_and_never_padding():
    replay = _replay()
    replay.add_trajectories(_batch([1, 2, 3], [3, 1, 2]))
    batch = replay.sample(6, replacement=False, indices=torch.arange(6))
    assert batch["sample_ids"].tolist() == list(range(6))
    assert batch["trajectory_ids"].tolist() == [0, 0, 0, 1, 2, 2]
    assert batch["z"][:, 0].tolist() == [100, 101, 102, 200, 300, 301]
    assert batch["remaining_horizon"][:, 0].tolist() == [3, 2, 1, 3, 3, 2]
    assert batch["terminated"][:, 0].tolist() == [0, 0, 0, 1, 0, 1]
    assert batch["horizon_end"][:, 0].tolist() == [0, 0, 1, 0, 0, 0]
    assert "valid" not in batch


def test_padding_is_finite_neutral_and_cannot_enter_actor_data():
    replay = _replay()
    source = _batch([1], [1])
    for name, value in source.items():
        if name != "valid":
            value[:, 1:] = float("nan")
    replay.add_trajectories(source)
    result = replay.sample_trajectories(1)
    for name, value in result.items():
        if name != "valid":
            assert torch.isfinite(value).all()
            expected = 1 if name == "remaining_horizon" else 0
            assert (value[:, 1:] == expected).all()
    assert replay.sample(20)["z"].shape == (20, 2)
    assert replay.sample(20)["sample_ids"].unique().tolist() == [0]


def test_ring_eviction_is_atomic_and_bulk_matches_sequential_appends():
    bulk, sequential = _replay(capacity=6), _replay(capacity=6)
    for replay in (bulk, sequential):
        replay.add_trajectories(_batch([0], [2]))
    bulk.add_trajectories(_batch([1, 2, 3, 4], [3, 1, 2, 3]))
    for label, length in zip([1, 2, 3, 4], [3, 1, 2, 3]):
        sequential.add_trajectories(_batch([label], [length]))
    _assert_state_equal(bulk.state_dict(), sequential.state_dict())
    assert bulk.size == 5 and bulk.trajectory_count == 2
    retained = bulk.sample_trajectories(2, indices=torch.arange(2), include_ids=True)
    assert set(retained["trajectory_ids"].tolist()) == {3, 4}
    assert set(retained["sample_ids"][retained["valid"].squeeze(-1)].tolist()) == set(range(6, 11))
    assert bulk.next_sample_id == 11 and bulk.next_trajectory_id == 5
    assert bulk.pos == 1


@pytest.mark.parametrize("method", ["sample", "sample_trajectories"])
def test_seeded_sampling_and_explicit_indices_preserve_rng_contract(method):
    replay = _replay()
    replay.add_trajectories(_batch([1, 2, 3]))
    fn = getattr(replay, method)
    first = fn(12, generator=torch.Generator().manual_seed(7), include_ids=True)
    second = fn(12, generator=torch.Generator().manual_seed(7), include_ids=True)
    for field in first:
        torch.testing.assert_close(first[field], second[field])
    generator = torch.Generator().manual_seed(22)
    before = generator.get_state().clone()
    fn(2, indices=torch.tensor([0, 1]), generator=generator)
    torch.testing.assert_close(before, generator.get_state())
    assert "indices" not in fn(2, include_ids=False)
    size = replay.size if method == "sample" else replay.trajectory_count
    with pytest.raises(ValueError, match="without replacement"):
        fn(size + 1, replacement=False)
    with pytest.raises((IndexError, RuntimeError)):
        fn(1, indices=torch.tensor([size]))


@pytest.mark.parametrize("corruption", [
    "hole", "empty", "short_without_terminal", "interior_terminal", "fractional_terminal",
    "boundary", "horizon", "density_nan", "density_inf", "action", "shape", "missing",
])
def test_invalid_append_is_rejected_before_live_state_changes(corruption):
    replay = _replay()
    replay.add_trajectories(_batch([1, 2]))
    before = replay.state_dict()
    batch = _batch([3], [2])
    if corruption == "hole":
        batch["valid"][0, :, 0] = torch.tensor([True, False, True])
    elif corruption == "empty":
        batch["valid"].zero_()
    elif corruption == "short_without_terminal":
        batch["terminated"].zero_()
    elif corruption == "interior_terminal":
        batch["terminated"][0, 0] = 1
    elif corruption == "fractional_terminal":
        batch["terminated"][0, 1] = 0.5
    elif corruption == "boundary":
        batch["horizon_end"][0, 1] = 1
    elif corruption == "horizon":
        batch["remaining_horizon"][0, 1] = 1
    elif corruption.startswith("density"):
        batch["behavior_log_prob"][0, 0] = float("nan" if corruption == "density_nan" else "inf")
    elif corruption == "action":
        batch["action"][0, 0] = -0.99
    elif corruption == "shape":
        batch["next_z"] = batch["next_z"][:, :1]
    else:
        del batch["pre_tanh_action"]
    with pytest.raises((ValueError, TypeError)):
        replay.add_trajectories(batch)
    _assert_state_equal(replay.state_dict(), before)


@pytest.mark.parametrize("training", [False, True])
def test_roundtrip_clear_reuse_and_preflight_are_independent(training):
    replay = _replay(capacity=6)
    replay.add_trajectories(_batch([1, 2, 3], [1, 3, 2]))
    saved = replay.training_state_dict() if training else replay.state_dict()
    restored = _replay(capacity=6)
    pointer = restored._storage.data_ptr()
    (restored.load_training_state_dict if training else restored.load_state_dict)(saved)
    _assert_state_equal(replay.state_dict(), restored.state_dict())
    for method in ("sample", "sample_trajectories"):
        first = getattr(replay, method)(8, generator=torch.Generator().manual_seed(19), include_ids=True)
        second = getattr(restored, method)(8, generator=torch.Generator().manual_seed(19), include_ids=True)
        for name in first:
            torch.testing.assert_close(first[name], second[name])
    replay.add_trajectories(_batch([4], [3]))
    restored.add_trajectories(_batch([4], [3]))
    _assert_state_equal(replay.state_dict(), restored.state_dict())
    before = restored.state_dict()
    candidate = restored.preflight_state_dict(before)
    before["reward"].fill_(12345)
    restored._commit_state_candidate(candidate)
    _assert_state_equal(replay.state_dict(), restored.state_dict())
    restored.clear()
    assert restored._storage.data_ptr() == pointer
    assert restored.size == restored.trajectory_count == 0
    assert restored.next_sample_id == restored.next_trajectory_id == 0
    with pytest.raises(ValueError, match="empty"):
        restored.sample_trajectories(1)
    restored.add_trajectories(_batch([9], [1]))
    assert restored.sample(1)["sample_ids"].item() == 0


@pytest.mark.parametrize("corruption", [
    "capacity", "requested_capacity", "conditioning", "pos", "full", "counter",
    "trajectory_id", "sample_id", "padding_id", "mask", "dtype", "density",
])
def test_malformed_restore_fails_atomically(corruption):
    replay = _replay(capacity=6)
    replay.add_trajectories(_batch([1, 2, 3], [1, 3, 2]))
    before = replay.state_dict()
    bad = deepcopy(before)
    if corruption in ("capacity", "requested_capacity"):
        bad[corruption] += 1
    elif corruption == "conditioning":
        bad["horizon_conditioning_horizon"] = None
    elif corruption == "pos":
        bad["pos"] = 0
    elif corruption == "full":
        bad["full"] = False
    elif corruption == "counter":
        bad["next_sample_id"] += 1
    elif corruption == "trajectory_id":
        bad["trajectory_id"][0] += 1
    elif corruption == "sample_id":
        bad["sample_id"][0, 0] += 1
    elif corruption == "padding_id":
        bad["sample_id"][~bad["valid"].squeeze(-1)] = 0
    elif corruption == "mask":
        bad["valid"][0, 0] = False
    elif corruption == "dtype":
        bad["reward"] = bad["reward"].double()
    else:
        bad["behavior_log_prob"][0, 0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        replay.load_state_dict(bad)
    _assert_state_equal(before, replay.state_dict())


def test_trusted_append_matches_validated_and_empty_roundtrip():
    first, second = _replay(), _replay()
    first.load_state_dict(second.state_dict())
    source = _batch([1, 2], [1, 3])
    source["remaining_horizon"] = source["remaining_horizon"].long()
    first.add_trajectories(source)
    second.add_trajectories(source, validate=False)
    _assert_state_equal(first.state_dict(), second.state_dict())
    first.add_trajectories(_batch([]))
    _assert_state_equal(first.state_dict(), second.state_dict())
    source["reward"].fill_(999)
    assert not (first.reward == 999).any()


def test_restore_cannot_claim_evicted_trajectories_had_zero_transitions():
    replay = _replay(capacity=6)
    replay.add_trajectories(_batch([1, 2, 3]))
    saved = replay.state_dict()
    # The retained two trajectories account for six steps. The third, evicted
    # trajectory must have contributed at least one step to the lifetime IDs.
    saved["next_sample_id"] = 6
    saved["sample_id"] -= 3
    with pytest.raises(ValueError, match="evicted trajectories"):
        replay.load_state_dict(saved)


def test_exact_training_restore_cannot_clear_buffer_with_empty_payload():
    replay = _replay()
    replay.add_trajectories(_batch([1]))
    before = replay.state_dict()
    payload = replay.training_state_dict()
    payload["state"] = {}
    with pytest.raises(ValueError):
        replay.load_training_state_dict(payload)
    _assert_state_equal(before, replay.state_dict())


@pytest.mark.parametrize("overrides", [
    {"capacity": 2}, {"capacity": 0}, {"capacity": True}, {"horizon": 0},
    {"horizon": 1.5}, {"latent_dim": 0}, {"horizon_conditioning_horizon": 2},
])
def test_invalid_constructor_contract(overrides):
    kwargs = dict(capacity=9, latent_dim=2, action_dim=1, device="cpu", horizon=3)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        LatentTrajectoryReplayBuffer(**kwargs)
