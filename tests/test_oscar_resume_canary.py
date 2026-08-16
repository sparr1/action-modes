import json

import pytest

from utils.oscar_resume_canary import CanaryError, benchmark_full_replay


def test_full_replay_benchmark_uses_production_shards_and_exact_next_sample(tmp_path):
    output = tmp_path / "replay.json"
    metrics = benchmark_full_replay(
        output_path=output,
        durability_root=tmp_path,
        capacity=16,
        observation_dim=2,
        action_dim=1,
        episode_rows=5,
        batch_size=1,
        train_unroll_horizon=1,
        shard_rows=5,
        maximum_estimated_bytes=100_000,
    )

    assert metrics["capacity_rows"] == 16
    assert metrics["shard_count"] == 4
    assert metrics["bytes_per_row"] == 28
    assert metrics["checkpoint_bytes"] > 0
    assert len(metrics["checkpoint_sha256"]) == 64
    assert metrics["verified"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == metrics
    assert not any(path.name.startswith(".replay-canary.") for path in tmp_path.iterdir())


def test_benchmark_refuses_unsafe_estimate_before_allocating(tmp_path):
    output = tmp_path / "replay.json"
    with pytest.raises(CanaryError, match="estimated peak"):
        benchmark_full_replay(
            output_path=output,
            durability_root=tmp_path,
            capacity=16,
            observation_dim=2,
            action_dim=1,
            episode_rows=5,
            batch_size=1,
            train_unroll_horizon=1,
            shard_rows=5,
            maximum_estimated_bytes=1,
        )
    assert not output.exists()


def test_benchmark_result_is_durable_and_never_overwritten(tmp_path):
    output = tmp_path / "replay.json"
    arguments = dict(
        output_path=output,
        durability_root=tmp_path,
        capacity=8,
        observation_dim=2,
        action_dim=1,
        episode_rows=4,
        batch_size=1,
        train_unroll_horizon=1,
        shard_rows=4,
        maximum_estimated_bytes=100_000,
    )
    benchmark_full_replay(**arguments)
    with pytest.raises(CanaryError, match="already exists"):
        benchmark_full_replay(**arguments)
