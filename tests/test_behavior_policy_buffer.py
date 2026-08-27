from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from RL.tdmpc2_core.common.buffer import Buffer


def _cfg(*, store_behavior_policy=None):
    cfg = SimpleNamespace(
        device="cpu",
        buffer_size=12,
        steps=20,
        batch_size=1,
        train_unroll_horizon=2,
        multitask=False,
        obs="state",
        obs_shape={"state": (3,)},
        obs_dtype="float32",
        action_dim=2,
    )
    if store_behavior_policy is not None:
        cfg.store_behavior_policy = store_behavior_policy
    return cfg


def _episode(*, include_behavior_policy):
    rows = 4
    fields = {
        "obs": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
        "action": torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
        "reward": torch.arange(rows, dtype=torch.float32),
        "terminated": torch.zeros(rows, dtype=torch.float32),
    }
    if include_behavior_policy:
        fields.update(
            behavior_pre_tanh_mean=(
                10.0 + torch.arange(rows * 2, dtype=torch.float32)
            ).reshape(rows, 2),
            behavior_log_std=(
                -1.0 - torch.arange(rows * 2, dtype=torch.float32) / 10.0
            ).reshape(rows, 2),
            behavior_policy_valid=torch.tensor(
                [False, True, False, True], dtype=torch.bool
            ),
        )
    return TensorDict(fields, batch_size=[rows])


def _assert_batches_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        if actual_item is None:
            assert expected_item is None
        else:
            torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


def test_disabled_behavior_replay_preserves_legacy_sample_and_exact_schema():
    replay = Buffer(_cfg(), resumable=True)
    replay.add(_episode(include_behavior_policy=False))

    assert len(replay.sample()) == 5
    with pytest.raises(RuntimeError, match="not enabled"):
        replay.sample(include_behavior_policy=True)

    metadata = replay.training_state_metadata()
    assert metadata["version"] == 1
    assert "behavior_policy_replay" not in metadata["signature"]
    assert {spec["name"] for spec in metadata["field_specs"]} == {
        "obs",
        "action",
        "reward",
        "terminated",
        "episode",
    }
    assert all(
        shard["version"] == 1
        for shard in replay.iter_training_state_shards(max_rows=2)
    )


def test_enabled_behavior_replay_is_aligned_and_requires_complete_fields():
    replay = Buffer(_cfg(store_behavior_policy=True), resumable=True)
    episode = _episode(include_behavior_policy=True)

    prepared = replay._prepare_batch(
        episode.unsqueeze(1), include_behavior_policy=True
    )
    assert len(prepared) == 8
    assert prepared[5].shape == (3, 1, 2)
    assert prepared[6].shape == (3, 1, 2)
    assert prepared[7].shape == (3, 1, 1)
    torch.testing.assert_close(
        prepared[5].squeeze(1), episode["behavior_pre_tanh_mean"][1:]
    )
    torch.testing.assert_close(
        prepared[6].squeeze(1), episode["behavior_log_std"][1:]
    )
    torch.testing.assert_close(
        prepared[7].squeeze(1).squeeze(-1),
        episode["behavior_policy_valid"][1:],
    )

    with pytest.raises(RuntimeError, match="lacks behavior-policy fields"):
        replay._prepare_batch(
            _episode(include_behavior_policy=False).unsqueeze(1),
            include_behavior_policy=True,
        )


def test_enabled_behavior_replay_v2_roundtrips_exact_sample():
    cfg = _cfg(store_behavior_policy=True)
    source = Buffer(cfg, resumable=True)
    source.add(_episode(include_behavior_policy=True))

    metadata = source.training_state_metadata()
    assert metadata["version"] == 2
    assert metadata["signature"]["behavior_policy_replay"] == (
        "pre-tanh-diagonal-gaussian-v1"
    )
    assert {spec["name"] for spec in metadata["field_specs"]}.issuperset(
        {
            "behavior_pre_tanh_mean",
            "behavior_log_std",
            "behavior_policy_valid",
        }
    )
    shards = list(source.iter_training_state_shards(max_rows=2))
    assert shards and all(shard["version"] == 2 for shard in shards)

    restored = Buffer(_cfg(store_behavior_policy=True), resumable=True)
    restored.load_training_state_shards(metadata, iter(shards))

    assert len(source.sample()) == 5
    rng_state = torch.random.get_rng_state()
    source_sample = source.sample(include_behavior_policy=True)
    torch.random.set_rng_state(rng_state)
    restored_sample = restored.sample(include_behavior_policy=True)
    _assert_batches_equal(source_sample, restored_sample)


@pytest.mark.parametrize(
    ("source_enabled", "target_enabled"),
    [(False, True), (True, False)],
)
def test_behavior_replay_exact_versions_reject_cross_mode_restore(
    source_enabled, target_enabled
):
    source = Buffer(
        _cfg(store_behavior_policy=source_enabled), resumable=True
    )
    source.add(_episode(include_behavior_policy=source_enabled))
    metadata = source.training_state_metadata()
    shards = list(source.iter_training_state_shards(max_rows=2))

    target = Buffer(
        _cfg(store_behavior_policy=target_enabled), resumable=True
    )
    with pytest.raises(ValueError, match="Unsupported TD-MPC2 sharded replay version"):
        target.load_training_state_shards(metadata, iter(shards))
    assert target.size == 0
    assert not hasattr(target, "_buffer")
