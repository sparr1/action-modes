import os
import json

import pytest
import torch

from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.common import checkpoint as checkpoint_io
from utils.checkpointing import CheckpointTarget, CheckpointTracker


def test_async_snapshot_is_immutable_and_atomically_published(tmp_path):
    live = torch.tensor([1.0, 2.0])
    target = tmp_path / "periodic.pt"
    writer = checkpoint_io.AsyncCheckpointWriter()

    writer.enqueue({"model": {"weight": live}}, target, signature=(10, 3, 3))
    live.add_(100.0)
    writer.flush()

    saved = torch.load(target, weights_only=False)
    torch.testing.assert_close(saved["model"]["weight"], torch.tensor([1.0, 2.0]))
    assert not list(tmp_path.glob(".periodic.pt.*.tmp"))
    writer.shutdown()


def test_async_multi_alias_checkpoint_freezes_once_and_writes_sidecars(
    tmp_path, monkeypatch
):
    tracker = CheckpointTracker(
        5,
        tmp_path,
        "model",
        save_strat=["best", "latest"],
        periodic_step_suffix="",
        trial_run_params={"seed": 7},
        experiment_params={"trials": 1},
    )
    tracker.record_episode_return(4.0)
    targets = tracker.targets(5)
    live = torch.tensor([2.0])
    original_freeze = checkpoint_io.freeze_checkpoint
    freeze_calls = []

    def counted_freeze(state):
        freeze_calls.append(state)
        return original_freeze(state)

    monkeypatch.setattr(checkpoint_io, "freeze_checkpoint", counted_freeze)
    writer = checkpoint_io.AsyncCheckpointWriter()
    writer.enqueue_many({"model": {"weight": live}}, targets, signature=(5, 1, 1))
    live.fill_(9.0)
    writer.flush()

    assert len(freeze_calls) == 1
    for target in targets:
        saved = torch.load(target.path, weights_only=False)
        torch.testing.assert_close(saved["model"]["weight"], torch.tensor([2.0]))
        sidecar = json.loads((tmp_path / f"{target.path.name}.metadata.json").read_text())
        assert sidecar["checkpoint"]["kind"] == target.kind
        assert sidecar["trial_run_params"] == {"seed": 7}
    writer.shutdown()


def test_async_metadata_failure_removes_old_sidecar_before_checkpoint_replace(
    tmp_path, monkeypatch
):
    tracker = CheckpointTracker(
        5,
        tmp_path,
        "model",
        save_strat="latest",
        periodic_step_suffix="",
    )
    writer = checkpoint_io.AsyncCheckpointWriter()
    checkpoint = tmp_path / "model_latest"
    sidecar = tmp_path / "model_latest.metadata.json"
    try:
        writer.save_many(
            {"value": torch.tensor([1.0])},
            tracker.targets(5),
            signature=(5, 1, 1),
        )
        assert sidecar.is_file()

        monkeypatch.setattr(
            checkpoint_io,
            "write_metadata_atomic",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected sidecar failure")
            ),
        )
        with pytest.raises(OSError, match="injected sidecar failure"):
            writer.save_many(
                {"value": torch.tensor([2.0])},
                tracker.targets(10),
                signature=(10, 2, 2),
            )

        torch.testing.assert_close(
            torch.load(checkpoint, weights_only=False)["value"],
            torch.tensor([2.0]),
        )
        assert not sidecar.exists()
    finally:
        writer.shutdown()


def test_reused_snapshot_invalidates_target_sidecar_before_alias_publication(
    tmp_path, monkeypatch
):
    writer = checkpoint_io.AsyncCheckpointWriter()
    state = {"value": torch.tensor([4.0])}
    signature = (20, 7, 7)
    source = tmp_path / "periodic.pt"
    target = tmp_path / "final.pt"
    source_publication = CheckpointTarget(source, "periodic", {"step": 20})
    target_publication = CheckpointTarget(target, "final", {"step": 20})
    try:
        writer.save_many(state, (source_publication,), signature=signature)
        target.write_bytes(b"stale checkpoint")
        target_sidecar = tmp_path / "final.pt.metadata.json"
        target_sidecar.write_text('{"step": 1}', encoding="utf-8")

        monkeypatch.setattr(
            checkpoint_io,
            "write_metadata_atomic",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected reused-sidecar failure")
            ),
        )
        with pytest.raises(OSError, match="injected reused-sidecar failure"):
            writer.save_many(state, (target_publication,), signature=signature)

        torch.testing.assert_close(
            torch.load(target, weights_only=False)["value"],
            torch.tensor([4.0]),
        )
        assert not target_sidecar.exists()
        assert (tmp_path / "periodic.pt.metadata.json").is_file()
    finally:
        writer.shutdown()


def test_async_publication_fsyncs_files_then_directories_before_metadata(
    tmp_path, monkeypatch
):
    writer = checkpoint_io.AsyncCheckpointWriter()
    target = CheckpointTarget(tmp_path / "model.pt", "latest", {"step": 5})
    events = []
    real_fsync_files = checkpoint_io.fsync_checkpoint_files
    real_fsync_directories = checkpoint_io.fsync_checkpoint_directories
    real_write_metadata = checkpoint_io.write_metadata_atomic

    def record_files(paths):
        paths = tuple(paths)
        events.append("files")
        return real_fsync_files(paths)

    def record_directories(paths):
        paths = tuple(paths)
        events.append("directories")
        return real_fsync_directories(paths)

    def record_metadata(*args, **kwargs):
        events.append("metadata")
        return real_write_metadata(*args, **kwargs)

    monkeypatch.setattr(checkpoint_io, "fsync_checkpoint_files", record_files)
    monkeypatch.setattr(
        checkpoint_io, "fsync_checkpoint_directories", record_directories
    )
    monkeypatch.setattr(checkpoint_io, "write_metadata_atomic", record_metadata)
    try:
        writer.save_many(
            {"value": torch.tensor([3.0])},
            (target,),
            signature=(5, 1, 1),
        )
    finally:
        writer.shutdown()

    assert events == ["files", "directories", "metadata"]


def test_async_file_fsync_failure_removes_old_sidecar_before_publication_error(
    tmp_path, monkeypatch
):
    tracker = CheckpointTracker(
        5,
        tmp_path,
        "model",
        save_strat="latest",
        periodic_step_suffix="",
    )
    writer = checkpoint_io.AsyncCheckpointWriter()
    checkpoint = tmp_path / "model_latest"
    sidecar = tmp_path / "model_latest.metadata.json"
    try:
        writer.save_many(
            {"value": torch.tensor([1.0])},
            tracker.targets(5),
            signature=(5, 1, 1),
        )
        assert sidecar.is_file()
        monkeypatch.setattr(
            checkpoint_io,
            "fsync_checkpoint_files",
            lambda _paths: (_ for _ in ()).throw(
                OSError("injected async file fsync failure")
            ),
        )

        with pytest.raises(OSError, match="injected async file fsync failure"):
            writer.save_many(
                {"value": torch.tensor([2.0])},
                tracker.targets(10),
                signature=(10, 2, 2),
            )

        torch.testing.assert_close(
            torch.load(checkpoint, weights_only=False)["value"],
            torch.tensor([2.0]),
        )
        assert not sidecar.exists()
    finally:
        writer.shutdown()


def test_standalone_torch_save_fsyncs_data_before_directory_entry(
    tmp_path, monkeypatch
):
    events = []
    real_fsync_files = checkpoint_io.fsync_checkpoint_files
    real_fsync_directories = checkpoint_io.fsync_checkpoint_directories

    def record_files(paths):
        paths = tuple(paths)
        events.append("files")
        return real_fsync_files(paths)

    def record_directories(paths):
        paths = tuple(paths)
        events.append("directories")
        return real_fsync_directories(paths)

    monkeypatch.setattr(checkpoint_io, "fsync_checkpoint_files", record_files)
    monkeypatch.setattr(
        checkpoint_io, "fsync_checkpoint_directories", record_directories
    )

    checkpoint_io.save_checkpoint(
        {"value": torch.tensor([9.0])},
        tmp_path / "standalone.pt",
    )

    assert events == ["files", "directories"]


def test_identical_final_checkpoint_reuses_completed_snapshot(tmp_path, monkeypatch):
    state = {"model": {"weight": torch.tensor([4.0])}, "num_updates": 7}
    periodic = tmp_path / "periodic.pt"
    final = tmp_path / "final.pt"
    signature = (20, 7, 7)
    writer = checkpoint_io.AsyncCheckpointWriter()
    writer.enqueue(state, periodic, signature=signature)
    writer.flush()

    def fail_if_refrozen(_state):
        raise AssertionError("identical final state was snapshotted a second time")

    monkeypatch.setattr(checkpoint_io, "freeze_checkpoint", fail_if_refrozen)
    writer.save(state, final, signature=signature)

    assert final.read_bytes() == periodic.read_bytes()
    if os.name != "nt":
        assert final.stat().st_ino == periodic.stat().st_ino
    writer.shutdown()


def test_invalidation_prevents_stale_reuse_for_matching_signature(
    tmp_path, monkeypatch
):
    periodic = tmp_path / "periodic.pt"
    final = tmp_path / "final.pt"
    signature = (20, 7, 7)
    writer = checkpoint_io.AsyncCheckpointWriter()
    writer.enqueue(
        {"model": {"weight": torch.tensor([4.0])}, "num_updates": 7},
        periodic,
        signature=signature,
    )
    writer.flush()
    writer.invalidate()

    def fail_if_reused(_source, _target):
        raise AssertionError("invalidated checkpoint cache was reused")

    monkeypatch.setattr(checkpoint_io, "_atomic_clone", fail_if_reused)
    writer.save(
        {"model": {"weight": torch.tensor([9.0])}, "num_updates": 7},
        final,
        signature=signature,
    )

    saved = torch.load(final, weights_only=False)
    torch.testing.assert_close(saved["model"]["weight"], torch.tensor([9.0]))
    writer.shutdown()


def test_background_write_failures_surface_on_flush(tmp_path, monkeypatch):
    writer = checkpoint_io.AsyncCheckpointWriter()

    def fail_write(_state, _path):
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_io, "_atomic_torch_save", fail_write)
    writer.enqueue({"value": torch.tensor(1)}, tmp_path / "failed.pt")
    with pytest.raises(OSError, match="disk full"):
        writer.flush()
    writer.shutdown()


def test_baseline_periodic_checkpoint_captures_exact_step(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.weight = torch.tensor([3.0])
            self.num_updates = 2
            self.outer_version = 2

        def checkpoint_state(self):
            return {"model": {"weight": self.weight}, "num_updates": self.num_updates}

    learner = object.__new__(TDMPC2Baseline)
    learner.agent = FakeAgent()
    learner.alg_logger = None
    learner._checkpoint_writer = checkpoint_io.AsyncCheckpointWriter()
    learner._checkpointing = (5, tmp_path, "model")
    learner._global_step = 5
    learner._num_updates = 2

    learner._maybe_checkpoint()
    learner.agent.weight.fill_(9.0)
    learner.flush_checkpoints()

    saved = torch.load(tmp_path / "model_5", weights_only=False)
    torch.testing.assert_close(saved["model"]["weight"], torch.tensor([3.0]))
    learner._checkpoint_writer.shutdown()


def test_baseline_load_invalidates_matching_completed_snapshot(tmp_path):
    class FakeAgent:
        def __init__(self):
            self.weight = torch.tensor([4.0])
            self.num_updates = 7
            self.outer_version = 7

        def checkpoint_state(self):
            return {
                "model": {"weight": self.weight},
                "num_updates": self.num_updates,
            }

        def load(self, state):
            self.weight = state["model"]["weight"].clone()
            self.num_updates = int(state["num_updates"])
            self.outer_version = int(state["outer_version"])

    learner = object.__new__(TDMPC2Baseline)
    learner.agent = FakeAgent()
    learner._checkpoint_writer = checkpoint_io.AsyncCheckpointWriter()
    learner._global_step = 20
    learner._num_updates = 7
    signature = learner._checkpoint_signature()
    learner._checkpoint_writer.enqueue(
        learner.agent.checkpoint_state(),
        tmp_path / "before-load.pt",
        signature=signature,
    )
    learner.flush_checkpoints()

    learner.load(
        {
            "model": {"weight": torch.tensor([9.0])},
            "num_updates": 7,
            "outer_version": 7,
        }
    )
    learner.save(tmp_path, "after-load.pt")

    saved = torch.load(tmp_path / "after-load.pt", weights_only=False)
    torch.testing.assert_close(saved["model"]["weight"], torch.tensor([9.0]))
    learner._checkpoint_writer.shutdown()


def test_baseline_failed_load_cannot_reuse_preload_snapshot(tmp_path):
    class PartiallyMutatingAgent:
        def __init__(self):
            self.weight = torch.tensor([4.0])
            self.num_updates = 7
            self.outer_version = 7

        def checkpoint_state(self):
            return {
                "model": {"weight": self.weight},
                "num_updates": self.num_updates,
            }

        def load(self, state):
            self.weight.copy_(state["model"]["weight"])
            raise ValueError("incompatible checkpoint after partial mutation")

    learner = object.__new__(TDMPC2Baseline)
    learner.agent = PartiallyMutatingAgent()
    learner._checkpoint_writer = checkpoint_io.AsyncCheckpointWriter()
    learner._global_step = 20
    learner._num_updates = 7
    signature = learner._checkpoint_signature()
    learner._checkpoint_writer.enqueue(
        learner.agent.checkpoint_state(),
        tmp_path / "before-failed-load.pt",
        signature=signature,
    )
    learner.flush_checkpoints()

    with pytest.raises(ValueError, match="after partial mutation"):
        learner.load({"model": {"weight": torch.tensor([9.0])}})

    learner.save(tmp_path, "after-failed-load.pt")
    saved = torch.load(tmp_path / "after-failed-load.pt", weights_only=False)
    torch.testing.assert_close(saved["model"]["weight"], learner.agent.weight)
    torch.testing.assert_close(saved["model"]["weight"], torch.tensor([9.0]))
    learner._checkpoint_writer.shutdown()
