"""Retained, hash-bound replay companions for portable training checkpoints.

Replay is streamed synchronously while training is paused. Only one owned CPU
shard is materialized at a time. The model writer later publishes the immutable
companion before exposing the model's metadata sidecar.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from utils.checkpointing import fsync_checkpoint_directories, fsync_checkpoint_files


REPLAY_SHARD_ROWS = 32768


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_directory(staging, target):
    """Replace an alias while leaving incomplete publication detectable."""
    backup = None
    if target.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.old-", dir=target.parent))
        backup.rmdir()
        os.replace(target, backup)
    try:
        os.replace(staging, target)
        fsync_checkpoint_directories((target,))
    except BaseException:
        if backup is not None and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if backup is not None and target.exists():
            shutil.rmtree(backup)


class PreparedCheckpointReplay:
    """Disk-owned replay snapshot, safe for the asynchronous model writer."""

    def __init__(self, staging, *, step, metadata, files):
        self.staging = Path(staging)
        self.step = int(step)
        self.metadata = metadata
        self.files = files

    def close(self):
        shutil.rmtree(self.staging, ignore_errors=True)

    def __call__(self, publications):
        try:
            for model_path, sidecar in publications:
                model_path = Path(model_path)
                if sidecar is None or sidecar["checkpoint"]["step"] != self.step:
                    raise ValueError("Replay publication requires matching model step metadata.")
                target = Path(f"{model_path}.replay")
                staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
                try:
                    for record in self.files:
                        source, output = self.staging / record["path"], staging / record["path"]
                        try:
                            os.link(source, output)
                        except OSError:
                            shutil.copyfile(source, output)
                    model_hash = _sha256(model_path)
                    manifest = {
                        "schema": "tdmpc2-checkpoint-replay", "version": 1,
                        "step": self.step,
                        "model": {"path": model_path.name, "sha256": model_hash},
                        "storage_rows": self.metadata["storage_rows"],
                        "resident_transitions": self.metadata["num_transitions"],
                        "total_transitions": self.metadata["total_transitions"],
                        "files": self.files,
                    }
                    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    fsync_checkpoint_files(staging.iterdir())
                    fsync_checkpoint_directories((staging / "manifest.json",))
                    _install_directory(staging, target)
                    sidecar["replay"] = {
                        "schema_version": 1, "path": target.name,
                        "manifest": "manifest.json", "step": self.step,
                        "model_sha256": model_hash,
                    }
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
        finally:
            self.close()


def prepare_checkpoint_replay(buffer, publications, *, step, pending_episode=None,
                              max_rows=REPLAY_SHARD_ROWS):
    """Capture every resident row and sampler/writer state without RNG draws."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("Replay checkpoint shard rows must be a positive integer.")
    publications = tuple(publications)
    if not publications:
        raise ValueError("Replay checkpoint requires a model publication.")
    first = Path(publications[0].path)
    first.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{first.name}.replay-capture-", dir=first.parent))
    files = []

    def save(name, value):
        path = staging / name
        torch.save(value, path)
        files.append({"path": name, "size": path.stat().st_size, "sha256": _sha256(path)})

    try:
        metadata = buffer.training_state_metadata()
        save("metadata.pt", metadata)
        for index in range(math.ceil(metadata["storage_rows"] / max_rows)):
            # The native sharder clones only this slice, so torch.save cannot
            # serialize a view's full replay-capacity backing storage.
            save(f"shard-{index:06d}.pt", buffer.training_state_shard(index, max_rows=max_rows))
        save("pending-episode.pt", pending_episode)
        fsync_checkpoint_files(staging.iterdir())
        fsync_checkpoint_directories((staging / "metadata.pt",))
        return PreparedCheckpointReplay(staging, step=step, metadata=metadata, files=files)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_checkpoint_replay(model_path, buffer):
    """Verify the complete model/replay pair, then restore its resident buffer.

    The returned ``pending_episode`` contains collected rows not yet inserted
    into replay at a mid-episode save. It is kept separate to reproduce the
    exact sampling population. This portable pair is not a full trainer resume;
    use the native training lineage for environment/optimizer continuation.
    """
    model_path = Path(model_path)
    root = Path(f"{model_path}.replay")
    manifest = json.loads((root / "manifest.json").read_text())
    sidecar = json.loads(Path(f"{model_path}.metadata.json").read_text())
    if (manifest.get("schema") != "tdmpc2-checkpoint-replay" or manifest.get("version") != 1
            or manifest.get("model", {}).get("path") != model_path.name
            or manifest.get("step") != sidecar.get("checkpoint", {}).get("step")
            or sidecar.get("replay", {}).get("path") != root.name
            or sidecar.get("replay", {}).get("step") != manifest.get("step")):
        raise ValueError("Checkpoint replay manifest and model metadata do not match.")
    model_hash = _sha256(model_path)
    if (model_hash != manifest["model"]["sha256"]
            or model_hash != sidecar["replay"].get("model_sha256")):
        raise ValueError("Checkpoint model hash does not match retained replay.")
    names = []
    for record in manifest["files"]:
        name = record["path"]
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ValueError("Invalid or duplicate replay payload path.")
        path = root / name
        if path.stat().st_size != record["size"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Checkpoint replay payload failed integrity verification: {name}.")
        names.append(name)
    if "metadata.pt" not in names or "pending-episode.pt" not in names:
        raise ValueError("Checkpoint replay payload is incomplete.")
    shard_names = sorted(name for name in names if name.startswith("shard-"))
    if len(names) != len(shard_names) + 2:
        raise ValueError("Unexpected checkpoint replay payload.")
    metadata = torch.load(root / "metadata.pt", map_location="cpu")
    pending = torch.load(root / "pending-episode.pt", map_location="cpu")
    buffer.load_training_state_shards(metadata, (
        torch.load(root / name, map_location="cpu") for name in shard_names
    ))
    return {"step": manifest["step"], "pending_episode": pending, "manifest": manifest}
