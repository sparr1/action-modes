"""Portable checkpoint banks may live outside durable resume generations."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import main as training_main
import utils.resume_identity as identity_module
import utils.resume_training as resume_training_module
from tests.test_main_resume import FakeSession, ResumeEnv, ResumeModel, _write_manifest
from utils.checkpointing import CheckpointTracker


@pytest.mark.parametrize("extra,match", [
    ([], "requires exact resume"),
    (["--resume-mode", "new"], "requires exact resume"),
    (["--lineage-dir", "/unused/lineage"], "requires exact resume"),
    (["--resume-mode", "new", "--lineage-dir", "/unused/lineage"], "absolute"),
])
def test_snapshot_root_cli_rejects_nonresume_or_relative_before_loading_config(monkeypatch, extra, match):
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--run", "unread-config.json",
        "--resume-model-snapshot-root", "relative-bank", *extra,
    ])
    with pytest.raises(ValueError, match=match):
        training_main.main()


@pytest.mark.parametrize("location", ["source", "source/bank", "lineage", "lineage/bank", "."])
def test_snapshot_root_rejects_overlap_with_source_or_lineage(tmp_path, location):
    with pytest.raises(ValueError, match="must not overlap"):
        training_main._validate_resume_model_snapshot_root(
            tmp_path / location, lineage_dir=tmp_path / "lineage", repo_root=tmp_path / "source",
        )
    assert not (tmp_path / "source").exists()
    assert not (tmp_path / "lineage").exists()


def test_snapshot_root_resolves_symlinks_before_ownership_check(tmp_path):
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(lineage, target_is_directory=True)
    with pytest.raises(ValueError, match="lineage"):
        training_main._validate_resume_model_snapshot_root(
            alias / "bank", lineage_dir=lineage, repo_root=tmp_path / "source",
        )


def test_snapshot_root_rejects_file_and_accepts_separate_absent_directory(tmp_path):
    file = tmp_path / "file"
    file.write_text("preserve")
    options = dict(lineage_dir=tmp_path / "lineage", repo_root=tmp_path / "source")
    with pytest.raises(ValueError, match="must be a directory"):
        training_main._validate_resume_model_snapshot_root(file, **options)
    root = tmp_path / "bank"
    assert training_main._validate_resume_model_snapshot_root(root, **options) == root.resolve()
    assert not root.exists()
    assert file.read_text() == "preserve"


def _session(tmp_path, segment_id="segment-one"):
    segment = tmp_path / "lineage" / "segments" / segment_id
    segment.mkdir(parents=True)
    return SimpleNamespace(segment_id=segment_id, segment_dir=segment)


def test_default_snapshot_location_stays_in_segment_without_external_pointer(tmp_path):
    session = _session(tmp_path)
    path = training_main._prepare_resume_model_snapshot_directory(session)
    assert path == session.segment_dir / "model_snapshots"
    assert path.is_dir()
    assert not (session.segment_dir / "model_snapshots.json").exists()
    with pytest.raises(FileExistsError):
        training_main._prepare_resume_model_snapshot_directory(session)


def test_external_snapshot_bank_records_pointer_and_never_reuses_a_segment(tmp_path):
    session = _session(tmp_path)
    root = tmp_path / "external" / "bank"
    path = training_main._prepare_resume_model_snapshot_directory(session, external_root=root)
    assert path == root / session.segment_id / "model_snapshots"
    assert path.is_dir()
    pointer = session.segment_dir / "model_snapshots.json"
    assert json.loads(pointer.read_text()) == {
        "schema_version": 1, "segment_id": session.segment_id, "path": str(path),
    }
    assert not (session.segment_dir / "model_snapshots").exists()
    preserved = path / "model"
    preserved.write_text("retained checkpoint")
    original_pointer = pointer.read_bytes()
    with pytest.raises(FileExistsError):
        training_main._prepare_resume_model_snapshot_directory(session, external_root=root)
    assert preserved.read_text() == "retained checkpoint"
    assert pointer.read_bytes() == original_pointer
    later = _session(tmp_path, "segment-two")
    assert training_main._prepare_resume_model_snapshot_directory(later, external_root=root).parent.name == "segment-two"


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_main_routes_bank_and_replay_without_changing_lineage_or_scientific_parameters(
    monkeypatch, tmp_path, external, replay,
):
    manifest, alg_dir = _write_manifest(tmp_path)
    recipe = json.loads(manifest.read_text())
    recipe.update(checkpoint_every=2, save_strat="all", checkpoint_replay=replay)
    manifest.write_text(json.dumps(recipe))
    lineage, bank = tmp_path / "durable-lineage", tmp_path / "external-bank"
    model, env = ResumeModel(), ResumeEnv()
    model.supports_checkpoint_replay = True
    identities = []
    FakeSession.calls = []

    def record_identity(**kwargs):
        identities.append(deepcopy(kwargs))
        return {"schema_version": 1, "fingerprint": "same-scientific-run"}

    monkeypatch.setattr(identity_module, "lineage_identity", record_identity)
    monkeypatch.setattr(resume_training_module, "TrainingResumeSession", FakeSession)
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(training_main, "initialize_alg", lambda *_args, **_kwargs: (model, False, "TDMPC2Baseline"))
    command = [
        "main.py", "--run", str(manifest), "--alg-dir", str(alg_dir),
        "--num-runs", "1", "--lineage-dir", str(lineage), "--resume-mode", "new",
    ]
    if external:
        command += ["--resume-model-snapshot-root", str(bank)]
    monkeypatch.setattr(sys, "argv", command)
    assert training_main.main() == 75
    session = model.learn_calls[0]["resume_session"]
    expected = (bank / session.segment_dir.name if external else session.segment_dir) / "model_snapshots"
    checkpoint = model.checkpoint_calls[0]
    assert checkpoint["save_path"] == str(expected)
    assert checkpoint.get("checkpoint_replay", False) is replay
    assert ("checkpoint_replay" in checkpoint) is replay
    assert expected.is_dir()
    assert FakeSession.calls[0][0] == lineage
    assert session.root == lineage
    assert "resume_model_snapshot_root" not in identities[0]["trial_run_params"]
    assert str(bank) not in json.dumps(identities[0]["trial_run_params"])
    assert identities[0]["experiment_params"] == recipe
    assert (session.segment_dir / "model_snapshots.json").exists() is external
    assert not (lineage / "generations").exists()  # The fake session never publishes.
    assert env.closed


def test_checkpoint_tracker_resume_preserves_new_segment_destination(tmp_path):
    original = CheckpointTracker(25_000, tmp_path / "old-bank", "model", save_strat="all")
    restored = CheckpointTracker(25_000, tmp_path / "new-bank", "model", save_strat="all")
    restored.load_state_dict(original.state_dict())
    assert restored.save_path == tmp_path / "new-bank"
    assert restored.state_dict() == original.state_dict()
