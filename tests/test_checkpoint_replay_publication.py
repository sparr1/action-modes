"""Replay references bind to actual model bytes across checkpoint aliases."""

from copy import deepcopy
import hashlib
import json

import pytest
import torch

from RL.tdmpc2_core.common import checkpoint as checkpoint_io
from utils.checkpointing import CheckpointTarget


def _target(path, *, step=5, kind="periodic"):
    return CheckpointTarget(path, kind, {
        "checkpoint": {"step": step, "kind": kind},
        "replay": {"step": step, "archive_path": "replay", "manifest_path": "5.json"},
    })


def _metadata(path):
    return json.loads(path.with_name(path.name + ".metadata.json").read_text())


def test_async_aliases_bind_replay_without_mutating_caller_metadata(tmp_path):
    targets = (_target(tmp_path / "all.pt"), _target(tmp_path / "latest.pt"))
    before = deepcopy([target.metadata for target in targets])
    writer = checkpoint_io.AsyncCheckpointWriter()
    live = torch.tensor([1.0])
    try:
        writer.enqueue_many({"weight": live}, targets, signature=(5, 1, 1))
        live.fill_(100.0)
        writer.flush()
        for target in targets:
            expected = hashlib.sha256(target.path.read_bytes()).hexdigest()
            assert _metadata(target.path)["replay"]["checkpoint_sha256"] == expected
            torch.testing.assert_close(
                torch.load(target.path, weights_only=True)["weight"], torch.tensor([1.0])
            )
        assert [target.metadata for target in targets] == before
    finally:
        writer.shutdown()


def test_reused_snapshot_binds_new_sidecar_to_same_model_bytes(tmp_path, monkeypatch):
    writer = checkpoint_io.AsyncCheckpointWriter()
    state = {"weight": torch.tensor([2.0])}
    original = _target(tmp_path / "periodic.pt")
    final = _target(tmp_path / "final.pt", kind="trial_final")
    final.metadata["replay"]["checkpoint_sha256"] = "stale hash"
    try:
        writer.save_many(state, (original,), signature=(5, 1, 1))

        def unexpected_snapshot(_state):
            raise AssertionError("Equivalent final checkpoint must reuse the model file")

        monkeypatch.setattr(checkpoint_io, "freeze_checkpoint", unexpected_snapshot)
        writer.save_many(state, (final,), signature=(5, 1, 1))
        assert original.path.read_bytes() == final.path.read_bytes()
        assert _metadata(final.path)["replay"]["checkpoint_sha256"] == (
            _metadata(original.path)["replay"]["checkpoint_sha256"]
        )
        assert final.metadata["replay"]["checkpoint_sha256"] == "stale hash"
    finally:
        writer.shutdown()


def test_wrong_replay_step_is_rejected_before_checkpoint_replacement(tmp_path):
    target = _target(tmp_path / "model.pt")
    target.metadata["replay"]["step"] = 4
    target.path.write_bytes(b"existing model")
    writer = checkpoint_io.AsyncCheckpointWriter()
    try:
        with pytest.raises(ValueError, match="steps do not match"):
            writer.save_many({"weight": torch.tensor([3.0])}, (target,))
        assert target.path.read_bytes() == b"existing model"
    finally:
        writer.shutdown()


def test_hash_failure_never_leaves_an_old_replay_sidecar(tmp_path, monkeypatch):
    target = _target(tmp_path / "latest.pt")
    writer = checkpoint_io.AsyncCheckpointWriter()
    try:
        writer.save_many({"weight": torch.tensor([1.0])}, (target,))

        def fail_hash(_path):
            raise OSError("injected checksum read failure")

        monkeypatch.setattr(
            checkpoint_io.AsyncCheckpointWriter, "_checkpoint_sha256", staticmethod(fail_hash)
        )
        with pytest.raises(OSError, match="injected checksum"):
            writer.save_many({"weight": torch.tensor([2.0])}, (target,))
        assert not target.path.with_name(target.path.name + ".metadata.json").exists()
    finally:
        writer.shutdown()


def test_disabled_saves_do_not_hash_model_files(tmp_path, monkeypatch):
    def unexpected_hash(_path):
        raise AssertionError("Replay-disabled saves should not compute extra checksums")

    monkeypatch.setattr(
        checkpoint_io.AsyncCheckpointWriter, "_checkpoint_sha256", staticmethod(unexpected_hash)
    )
    writer = checkpoint_io.AsyncCheckpointWriter()
    target = CheckpointTarget(tmp_path / "model.pt", "periodic", {"checkpoint": {"step": 5}})
    try:
        writer.save_many({"weight": torch.tensor([1.0])}, (target,))
        assert _metadata(target.path) == target.metadata
    finally:
        writer.shutdown()
