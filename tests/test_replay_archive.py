import hashlib
import json
import shutil
import threading

import pytest
import torch

import utils.replay_archive as archive_module
from utils.replay_archive import (
    ReplayArchiveError,
    ReplayArchiveWriter,
    load_checkpoint_replay,
)


def _episode(index, transitions=3):
    rows = transitions + 1
    values = torch.arange(rows, dtype=torch.float32) + index * 100
    episode = {
        "obs": torch.stack((values, values + 0.5), dim=1),
        "action": (values / 1000).unsqueeze(1),
        "reward": values + 10,
        "terminated": torch.zeros(rows),
    }
    for name in ("action", "reward", "terminated"):
        episode[name][0] = float("nan")
    return episode


def _writer(tmp_path, *, capacity=7, chunk_rows=3):
    return ReplayArchiveWriter(
        tmp_path / "replay", capacity=capacity, observation_shape=(2,),
        action_dim=1, chunk_rows=chunk_rows,
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(writer, path, *, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"checkpoint at {step}".encode())
    reference = writer.checkpoint_reference(path, step=step)
    reference["checkpoint_sha256"] = _sha(path)
    sidecar = {"checkpoint": {"step": step}, "replay": reference}
    path.with_name(f"{path.name}.metadata.json").write_text(json.dumps(sidecar))
    return reference


def _concat(replay, name):
    return torch.cat([part["fields"][name] for part in replay.fragments])


def test_incremental_chunks_preserve_prior_snapshot_and_partial_ring_membership(tmp_path):
    writer = _writer(tmp_path)
    episode0, episode1 = _episode(0), _episode(1, transitions=4)
    original0 = {key: value.clone() for key, value in episode0.items()}
    writer.append_episode(episode0, episode_id=0)
    episode0["obs"].fill_(-999)  # The training staging tensor may be reused now.
    path0 = tmp_path / "first.pt"
    reference0 = _publish(writer, path0, step=4)  # An unfinished real step is excluded.
    manifest0_path = writer.root / reference0["manifest_path"]
    manifest0_bytes = manifest0_path.read_bytes()
    writer.append_episode(episode1, episode_id=1)
    path1 = tmp_path / "second.pt"
    _publish(writer, path1, step=7)
    writer.close()

    earlier, later = load_checkpoint_replay(path0), load_checkpoint_replay(path1)
    assert earlier.num_rows == 4
    assert earlier.num_transitions == 3
    assert later.num_rows == later.capacity == 7
    assert later.num_transitions == 6
    assert later.num_episodes == 2
    assert later.fragments[0]["episode_offset"] == 2
    assert later.fragments[0]["global_start"] == 2
    for name in original0:
        torch.testing.assert_close(_concat(earlier, name), original0[name], equal_nan=True)
        expected = torch.cat((original0[name][2:], episode1[name]))
        torch.testing.assert_close(_concat(later, name), expected, equal_nan=True)
    assert manifest0_path.read_bytes() == manifest0_bytes
    chunks = [torch.load(path, weights_only=True) for path in (writer.root / "chunks").glob("*.pt")]
    assert sum(chunk["stop"] - chunk["start"] for chunk in chunks) == 9
    assert all(chunk["stop"] - chunk["start"] <= 3 for chunk in chunks)


def test_sampler_uses_only_valid_sequences_and_does_not_touch_global_rng(tmp_path):
    writer = _writer(tmp_path)
    writer.append_episode(_episode(0), episode_id=0)
    writer.append_episode(_episode(1, transitions=4), episode_id=1)
    path = tmp_path / "model.pt"
    _publish(writer, path, step=7)
    writer.close()
    replay = load_checkpoint_replay(path)
    before = torch.get_rng_state().clone()
    obs, action, reward, terminated, task = replay.sample_sequences(2000, 1, seed=37)
    assert torch.equal(before, torch.get_rng_state())
    assert obs.shape == (2, 2000, 2)
    assert action.shape == reward.shape == terminated.shape == (1, 2000, 1)
    assert task is None
    assert torch.isfinite(action).all() and torch.isfinite(reward).all()
    torch.testing.assert_close(obs[1, :, 0] - obs[0, :, 0], torch.ones(2000))
    # Equal weight per valid start, including the short partial oldest episode.
    starts, counts = obs[0, :, 0].unique(return_counts=True)
    assert starts.tolist() == [2, 100, 101, 102, 103]
    assert bool(((counts - 400).abs() < 75).all())
    again = replay.sample_sequences(2000, 1, seed=37)
    assert torch.equal(obs, again[0])
    obs.fill_(-999)
    fragments = replay.fragments
    fragments[0]["fields"]["obs"].fill_(-999)
    assert torch.equal(again[0], replay.sample_sequences(2000, 1, seed=37)[0])
    long_obs = replay.sample_sequences(12, 3, seed=5)[0]
    assert bool((long_obs[:, :, 0] >= 100).all())
    with pytest.raises(ReplayArchiveError, match="no sequences"):
        replay.sample_sequences(1, 9)


def test_empty_archive_and_final_reference_after_idempotent_close(tmp_path):
    writer = _writer(tmp_path)
    writer.close()
    writer.close()
    path = tmp_path / "empty.pt"
    first = _publish(writer, path, step=0)
    second = writer.checkpoint_reference(tmp_path / "alias.pt", step=0)
    assert first["manifest_path"] == second["manifest_path"]
    replay = load_checkpoint_replay(path)
    assert replay.num_rows == replay.num_transitions == replay.num_episodes == 0
    assert replay.fragments == ()
    with pytest.raises(ReplayArchiveError, match="no sequences"):
        replay.sample_sequences(1, 1)
    with pytest.raises(ReplayArchiveError, match="closed"):
        writer.append_episode(_episode(0), episode_id=0)


@pytest.mark.parametrize("collect", [False, True])
def test_zero_capacity_archive_has_no_resident_rows(tmp_path, collect):
    writer = _writer(tmp_path, capacity=0)
    if collect:
        writer.append_episode(_episode(0), episode_id=0)
    path = tmp_path / "model.pt"
    _publish(writer, path, step=3 if collect else 0)
    writer.close()
    replay = load_checkpoint_replay(path)
    assert replay.capacity == replay.num_rows == replay.num_transitions == 0
    assert replay.fragments == ()
    assert replay.metadata["total_rows"] == (4 if collect else 0)


def test_new_directory_entries_are_durable_before_reference_publication(tmp_path, monkeypatch):
    calls = []
    original = archive_module._fsync_directory

    def record(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(archive_module, "_fsync_directory", record)
    container = tmp_path / "first" / "second"
    writer = ReplayArchiveWriter(
        container, capacity=7, observation_shape=(2,), action_dim=1, chunk_rows=3,
    )
    assert calls == [tmp_path, tmp_path / "first", container]
    writer.append_episode(_episode(0, transitions=2), episode_id=0)
    _publish(writer, tmp_path / "model.pt", step=2)
    writer.close()
    assert calls[3:] == [
        writer.root, writer.root / "chunks", writer.root, writer.root / "snapshots",
    ]


def test_failed_directory_sync_prevents_manifest_publication(tmp_path, monkeypatch):
    writer = _writer(tmp_path)
    writer.append_episode(_episode(0), episode_id=0)
    writer.flush()
    original = archive_module._fsync_directory

    def fail_snapshot_directory(path):
        if path == writer.root and (writer.root / "snapshots").exists():
            raise OSError("directory sync failed")
        return original(path)

    monkeypatch.setattr(archive_module, "_fsync_directory", fail_snapshot_directory)
    with pytest.raises(ReplayArchiveError, match="snapshot manifest"):
        writer.checkpoint_reference(tmp_path / "model.pt", step=3)
    assert not list((writer.root / "snapshots").glob("*.json"))
    with pytest.raises(ReplayArchiveError, match="previously failed"):
        writer.close()


def test_archive_relocation_override_and_independent_writer_roots(tmp_path):
    writer, other = _writer(tmp_path), _writer(tmp_path)
    assert writer.root != other.root
    other.close()
    writer.append_episode(_episode(0), episode_id=0)
    path = tmp_path / "original" / "model.pt"
    _publish(writer, path, step=3)
    writer.close()
    relocated = tmp_path / "relocated"
    copied_archive = relocated / writer.root.name
    shutil.copytree(writer.root, copied_archive)
    copied_model = relocated / "model.pt"
    shutil.copy2(path, copied_model)
    shutil.copy2(f"{path}.metadata.json", f"{copied_model}.metadata.json")
    shutil.rmtree(writer.root)
    with pytest.raises(ReplayArchiveError):
        load_checkpoint_replay(copied_model)
    replay = load_checkpoint_replay(copied_model, archive_root=copied_archive)
    assert replay.num_rows == 4


def test_moving_complete_artifact_tree_preserves_relative_replay_reference(tmp_path):
    original = tmp_path / "original"
    writer = _writer(original)
    writer.append_episode(_episode(0), episode_id=0)
    model = original / "models" / "model.pt"
    _publish(writer, model, step=3)
    writer.close()
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)

    replay = load_checkpoint_replay(relocated / "models" / "model.pt")
    assert replay.num_rows == 4
    assert replay.num_transitions == 3
    assert _concat(replay, "obs")[:, 0].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("corrupt", ["checkpoint", "manifest", "chunk", "missing_chunk"])
def test_reader_rejects_missing_or_corrupted_artifacts(tmp_path, corrupt):
    writer = _writer(tmp_path)
    writer.append_episode(_episode(0), episode_id=0)
    path = tmp_path / "model.pt"
    reference = _publish(writer, path, step=3)
    writer.close()
    manifest_path = writer.root / reference["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    artifact = {
        "checkpoint": path,
        "manifest": manifest_path,
        "chunk": writer.root / manifest["chunks"][0]["path"],
        "missing_chunk": writer.root / manifest["chunks"][0]["path"],
    }[corrupt]
    if corrupt == "missing_chunk":
        artifact.unlink()
    else:
        artifact.write_bytes(artifact.read_bytes() + b"corruption")
    with pytest.raises(ReplayArchiveError, match="SHA-256 mismatch" if corrupt != "missing_chunk" else "No such file"):
        load_checkpoint_replay(path)


def test_failed_background_write_cannot_publish_snapshot(tmp_path, monkeypatch):
    writer = _writer(tmp_path)

    def fail(*args):
        raise OSError("disk is full")

    monkeypatch.setattr(writer, "_write_chunk", fail)
    writer.append_episode(_episode(0, transitions=2), episode_id=0)
    with pytest.raises(ReplayArchiveError, match="Could not publish"):
        writer.checkpoint_reference(tmp_path / "model.pt", step=2)
    assert not (writer.root / "snapshots").exists()
    with pytest.raises(ReplayArchiveError, match="previously failed"):
        writer.close()
    assert writer._executor is None


def test_incomplete_chunk_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    writer = _writer(tmp_path)
    writer.append_episode(_episode(0), episode_id=0)
    previous = tmp_path / "previous.pt"
    _publish(writer, previous, step=3)
    chunks_before = set((writer.root / "chunks").iterdir())
    snapshots_before = set((writer.root / "snapshots").iterdir())

    def incomplete_save(payload, stream):
        stream.write(b"incomplete serialized tensor data")
        raise OSError("injected failure after partial chunk write")

    monkeypatch.setattr(archive_module.torch, "save", incomplete_save)
    writer.append_episode(_episode(1), episode_id=1)
    with pytest.raises(ReplayArchiveError, match="Could not publish replay archive chunk"):
        writer.checkpoint_reference(tmp_path / "failed.pt", step=6)
    assert set((writer.root / "chunks").iterdir()) == chunks_before
    assert set((writer.root / "snapshots").iterdir()) == snapshots_before
    previous_replay = load_checkpoint_replay(previous)
    assert previous_replay.num_rows == 4
    assert _concat(previous_replay, "obs")[:, 0].tolist() == [0, 1, 2, 3]
    with pytest.raises(ReplayArchiveError, match="previously failed"):
        writer.close()
    assert writer._executor is None


def test_one_chunk_in_flight_and_owned_capture(tmp_path, monkeypatch):
    writer = _writer(tmp_path, chunk_rows=3)
    entered, release = threading.Event(), threading.Event()
    original = writer._write_chunk

    def delayed(*args):
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("Test did not release background writer.")
        return original(*args)

    monkeypatch.setattr(writer, "_write_chunk", delayed)
    episode = _episode(0, transitions=2)
    writer.append_episode(episode, episode_id=0)
    assert entered.wait(timeout=10)
    episode["obs"].fill_(999)
    release.set()
    path = tmp_path / "model.pt"
    _publish(writer, path, step=2)
    writer.close()
    assert _concat(load_checkpoint_replay(path), "obs")[:, 0].tolist() == [0, 1, 2]


@pytest.mark.parametrize("problem", ["episode_id", "shape", "dtype", "initial", "finite", "extra", "terminal"])
def test_invalid_episode_is_rejected_before_recording_rows(tmp_path, problem):
    writer = _writer(tmp_path)
    episode = _episode(0)
    episode_id = 0
    if problem == "episode_id":
        episode_id = 2
    elif problem == "shape":
        episode["obs"] = episode["obs"][:, :1]
    elif problem == "dtype":
        episode["reward"] = episode["reward"].double()
    elif problem == "initial":
        episode["reward"][0] = 0
    elif problem == "finite":
        episode["reward"][2] = float("nan")
    elif problem == "terminal":
        episode["terminated"][1] = 1
    else:
        episode["task"] = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError):
        writer.append_episode(episode, episode_id=episode_id)
    path = tmp_path / "empty.pt"
    _publish(writer, path, step=0)
    writer.close()
    assert load_checkpoint_replay(path).num_rows == 0


@pytest.mark.parametrize("problem", ["inventory", "path_escape", "episode", "step"])
def test_manifest_semantics_checked_even_when_hash_matches(tmp_path, problem):
    writer = _writer(tmp_path)
    writer.append_episode(_episode(0), episode_id=0)
    path = tmp_path / "model.pt"
    reference = _publish(writer, path, step=3)
    writer.close()
    manifest_path = writer.root / reference["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    if problem == "inventory":
        manifest["chunks"].pop()
    elif problem == "path_escape":
        manifest["chunks"][0]["path"] = "../../outside.pt"
    elif problem == "episode":
        manifest["episodes"][0]["episode_id"] = 1
    else:
        manifest["step"] = 4
    manifest_path.write_text(json.dumps(manifest))
    sidecar_path = path.with_name(f"{path.name}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["replay"]["manifest_sha256"] = _sha(manifest_path)
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ReplayArchiveError):
        load_checkpoint_replay(path)


@pytest.mark.parametrize("problem", ["shape", "sentinel", "serialization", "sidecar"])
def test_reader_validates_payloads_even_when_hashes_match(tmp_path, problem):
    writer = _writer(tmp_path)
    writer.append_episode(_episode(0), episode_id=0)
    path = tmp_path / "model.pt"
    reference = _publish(writer, path, step=3)
    writer.close()
    manifest_path = writer.root / reference["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = path.with_name(f"{path.name}.metadata.json")
    sidecar = json.loads(sidecar_path.read_text())
    if problem == "sidecar":
        sidecar["checkpoint"] = None
    else:
        record = manifest["chunks"][0]
        chunk_path = writer.root / record["path"]
        if problem == "serialization":
            chunk_path.write_bytes(b"invalid torch payload")
        else:
            chunk = torch.load(chunk_path, weights_only=True)
            if problem == "shape":
                chunk["fields"]["obs"] = chunk["fields"]["obs"][:, :1]
            else:
                chunk["fields"]["reward"][0] = 0
            torch.save(chunk, chunk_path)
        record["sha256"] = _sha(chunk_path)
        manifest_path.write_text(json.dumps(manifest))
        sidecar["replay"]["manifest_sha256"] = _sha(manifest_path)
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ReplayArchiveError, match="Cannot load checkpoint replay"):
        load_checkpoint_replay(path)
