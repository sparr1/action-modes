import json
from pathlib import Path

import pytest

from utils.oscar_resume_canary import (
    CanaryError,
    benchmark_full_replay,
    verify_resumed_lineage,
)
from utils.resume_lineage import GenerationFile, LineageStore, write_bytes


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_canary_is_bounded_but_replay_spec_remains_production_scale():
    manifest = json.loads(
        (ROOT / "configs/experiments/AntAMBITDMPC2ResumeCanary.json").read_text()
    )
    algorithm = json.loads(
        (ROOT / "configs/algs/AntAMBITDMPC2ResumeCanary.json").read_text()
    )
    params = algorithm["alg_params"]
    assert manifest["configs"] == ["AntAMBITDMPC2ResumeCanary"]
    assert manifest["save_trials"] == "none"
    assert manifest["env_params"]["max_episode_steps"] == 100
    assert manifest["overrides_alg"]["total_steps"] == 1_000_000
    assert params["wandb"] is True
    assert params["seed_steps"] == 500
    assert params["pretrain_steps"] == 10
    assert params["model_size"] == 1
    assert params["inner_rollouts_per_round"] == 4

    production_manifest = json.loads(
        (ROOT / "configs/ambi/experiments/ambi_anchor.json").read_text()
    )
    production_params = json.loads(
        (ROOT / "configs/ambi/algs/ambi_anchor.json").read_text()
    )["alg_params"]
    assert production_manifest["overrides_alg"]["total_steps"] == 1_000_000
    assert production_manifest["env_params"]["max_episode_steps"] == 1000
    assert production_params["buffer_size"] == 1_000_000
    assert production_params["batch_size"] == 256
    assert production_params["train_unroll_horizon"] == 3


def _canary_lineage(tmp_path, *, first=None, second=None):
    first_metadata = {
        "segment_id": "job.canary.0",
        "global_step": 600,
        "num_updates": 100,
        "wandb_run_id": "stable-run",
    }
    second_metadata = {
        "segment_id": "job.canary.1",
        "global_step": 700,
        "num_updates": 200,
        "wandb_run_id": "stable-run",
    }
    first_metadata.update(first or {})
    second_metadata.update(second or {})
    root = tmp_path / "lineage"
    payload = [GenerationFile("trainer.pt", "trainer", write_bytes(b"state"))]
    with LineageStore.open(root, mode="new", lineage_metadata={"test": True}) as store:
        store.publish("first", files=payload, metadata=first_metadata)
        store.publish("second", files=payload, metadata=second_metadata)
    return root


def test_resumed_lineage_verifier_requires_real_progress_and_one_wandb_run(tmp_path):
    root = _canary_lineage(tmp_path)
    result = verify_resumed_lineage(
        lineage_dir=root,
        first_segment="job.canary.0",
        second_segment="job.canary.1",
        minimum_first_step=500,
    )
    assert result["first_step"] == 600
    assert result["second_step"] == 700
    assert result["wandb_run_id"] == "stable-run"
    assert result["verified"] is True


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ({"global_step": 500}, {}, "learned-state"),
        ({"num_updates": 0}, {}, "learned-state"),
        ({}, {"global_step": 600}, "environment steps"),
        ({}, {"num_updates": 100}, "optimizer updates"),
        ({}, {"wandb_run_id": "other"}, "different W&B run"),
        ({}, {"segment_id": "wrong"}, "segment ID"),
    ],
)
def test_resumed_lineage_verifier_rejects_false_smokes(
    tmp_path, first, second, message
):
    root = _canary_lineage(tmp_path, first=first, second=second)
    with pytest.raises(CanaryError, match=message):
        verify_resumed_lineage(
            lineage_dir=root,
            first_segment="job.canary.0",
            second_segment="job.canary.1",
            minimum_first_step=500,
        )


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
