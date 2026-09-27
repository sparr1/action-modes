"""Incremental, checkpoint-matched raw replay for frozen-model evaluation.

Completed episodes are copied once into immutable, bounded CPU chunks. Snapshot
manifests select the chronological rows resident in the training ring; they are
not trainer-resume state. Readers return owned copies and never consume global
training or evaluation RNG streams.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


_VERSION = 1
_FIELDS = ("obs", "action", "reward", "terminated", "episode")


class ReplayArchiveError(RuntimeError):
    """Replay was unavailable, inconsistent, or could not be published."""


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path, writer):
    path = Path(path)
    _mkdir_durable(path.parent)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".replay-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fsync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _mkdir_durable(path):
    """Persist every newly created directory entry, including nested roots."""
    missing = []
    current = Path(path)
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _fsync_directory(directory.parent)


def _write_json(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(data))


def _validate_fields(fields, rows, observation_shape, action_dim, *, episode=True):
    expected = set(_FIELDS if episode else _FIELDS[:-1])
    if set(fields) != expected:
        raise ValueError(f"Replay fields must be {sorted(expected)}.")
    for name in expected:
        value = fields[name]
        trailing = observation_shape if name == "obs" else (
            (action_dim,) if name == "action" else ()
        )
        dtype = torch.int64 if name == "episode" else torch.float32
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or tuple(value.shape) != (rows, *trailing)
            or value.dtype != dtype
        ):
            raise ValueError(
                f"Replay field {name!r} must be a CPU {dtype} tensor "
                f"with shape {(rows, *trailing)}."
            )


def _validate_episode_values(fields, *, includes_initial):
    if not bool(torch.isfinite(fields["obs"]).all()):
        raise ValueError("Replay observations must be finite.")
    start = int(includes_initial)
    for name in ("action", "reward", "terminated"):
        value = fields[name]
        if includes_initial and not bool(torch.isnan(value[0]).all()):
            raise ValueError(f"Initial replay {name} must retain its NaN sentinel.")
        if not bool(torch.isfinite(value[start:]).all()):
            raise ValueError(f"Replay transition {name} must be finite.")
    terminated = fields["terminated"][start:]
    if not bool(((terminated == 0) | (terminated == 1)).all()):
        raise ValueError("Replay termination flags must be zero or one.")
    if bool((terminated[:-1] == 1).any()):
        raise ValueError("Replay episode contains a transition after termination.")


class ReplayArchiveWriter:
    """Append-only archive with one bounded background write in flight.

    ``root`` is a container directory. Each writer creates a unique child and
    exposes that actual archive directory as ``root``. Closing stops collection,
    but leaves checkpoint references available for subsequent explicit saves.
    """

    def __init__(self, root, *, capacity, observation_shape, action_dim, chunk_rows=65536):
        self.capacity = _integer(capacity, "capacity")
        self.observation_shape = tuple(observation_shape)
        if not self.observation_shape:
            raise ValueError("observation_shape must not be empty.")
        for dim in self.observation_shape:
            _integer(dim, "observation dimension", minimum=1)
        self.action_dim = _integer(action_dim, "action_dim", minimum=1)
        self.chunk_rows = _integer(chunk_rows, "chunk_rows", minimum=1)
        if self.chunk_rows > 65536:
            raise ValueError("chunk_rows must be at most 65536.")
        self.archive_id = uuid.uuid4().hex
        self.root = Path(root).resolve() / f"archive-{self.archive_id}"
        _mkdir_durable(self.root.parent)
        self.root.mkdir(exist_ok=False)
        _fsync_directory(self.root.parent)
        self._parts = []
        self._pending_rows = 0
        self._rows = 0
        self._episodes = []
        self._chunks = []
        self._next_chunk = 0
        self._executor = None
        self._future = None
        self._error = None
        self._closed = False
        self._snapshots = {}

    def _check_error(self):
        if self._error is not None:
            raise ReplayArchiveError("Replay archive publication previously failed.") from self._error

    def _finish_write(self):
        self._check_error()
        if self._future is not None:
            future, self._future = self._future, None
            try:
                self._chunks.append(future.result())
            except BaseException as exc:
                self._error = exc
                raise ReplayArchiveError("Could not publish replay archive chunk.") from exc

    def _write_chunk(self, payload, relative):
        path = self.root / relative
        _atomic_write(path, lambda stream: torch.save(payload, stream))
        return {
            "path": relative,
            "sha256": _sha256(path),
            "start": payload["start"],
            "stop": payload["stop"],
        }

    def _queue_chunk(self):
        if not self._pending_rows:
            return
        self._finish_write()
        start = self._rows - self._pending_rows
        fields = {
            name: torch.cat([part[name] for part in self._parts], dim=0)
            for name in _FIELDS
        }
        payload = {
            "schema": "ambi-replay-chunk",
            "version": _VERSION,
            "archive_id": self.archive_id,
            "start": start,
            "stop": self._rows,
            "fields": fields,
        }
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-archive")
        relative = f"chunks/chunk-{self._next_chunk:08d}.pt"
        self._next_chunk += 1
        self._future = self._executor.submit(self._write_chunk, payload, relative)
        self._parts = []
        self._pending_rows = 0

    def append_episode(self, td, *, episode_id):
        """Copy one completed episode, including its initial observation row."""
        self._check_error()
        if self._closed:
            raise ReplayArchiveError("Cannot append to a closed replay archive.")
        _integer(episode_id, "episode_id")
        if episode_id != len(self._episodes):
            raise ValueError("Archived episode IDs must be consecutive, starting at zero.")
        fields = dict(td.items())
        supplied_episode = fields.pop("episode", None)
        obs = fields.get("obs")
        rows = int(obs.shape[0]) if torch.is_tensor(obs) and obs.ndim else 0
        _integer(rows, "episode rows", minimum=1)
        _validate_fields(fields, rows, self.observation_shape, self.action_dim, episode=False)
        _validate_episode_values(fields, includes_initial=True)
        if supplied_episode is not None:
            _validate_fields(
                {**fields, "episode": supplied_episode}, rows,
                self.observation_shape, self.action_dim,
            )
            if not bool((supplied_episode == episode_id).all()):
                raise ValueError("Supplied replay episode IDs disagree with episode_id.")
        episode_start = self._rows
        offset = 0
        while offset < rows:
            count = min(rows - offset, self.chunk_rows - self._pending_rows)
            part = {
                name: value[offset:offset + count].detach().clone()
                for name, value in fields.items()
            }
            part["episode"] = torch.full((count,), episode_id, dtype=torch.int64)
            self._parts.append(part)
            self._pending_rows += count
            self._rows += count
            offset += count
            if self._pending_rows == self.chunk_rows:
                self._queue_chunk()
        self._episodes.append({"episode_id": episode_id, "start": episode_start, "stop": self._rows})

    def flush(self):
        """Make every collected row durable and surface background failures."""
        self._check_error()
        self._queue_chunk()
        self._finish_write()

    def checkpoint_reference(self, checkpoint_path, *, step):
        """Durably snapshot membership before publishing the model sidecar."""
        _integer(step, "checkpoint step")
        if step < self._rows - len(self._episodes):
            raise ValueError("Checkpoint step precedes archived real transitions.")
        self.flush()
        key = (step, self._rows)
        if key not in self._snapshots:
            start = max(0, self._rows - self.capacity)
            episodes = [record for record in self._episodes if record["stop"] > start]
            initial_rows = sum(record["start"] >= start for record in episodes)
            manifest = {
                "schema": "ambi-replay-snapshot",
                "version": _VERSION,
                "archive_id": self.archive_id,
                "step": step,
                "capacity": self.capacity,
                "chunk_rows": self.chunk_rows,
                "observation_shape": list(self.observation_shape),
                "action_dim": self.action_dim,
                "total_rows": self._rows,
                "total_episodes": len(self._episodes),
                "total_transitions": self._rows - len(self._episodes),
                "resident_start": start,
                "resident_rows": self._rows - start,
                "resident_transitions": self._rows - start - initial_rows,
                "episodes": episodes,
                "chunks": [record for record in self._chunks if record["stop"] > start],
            }
            relative = f"snapshots/step-{step:012d}-rows-{self._rows:012d}.json"
            path = self.root / relative
            try:
                _write_json(path, manifest)
                self._snapshots[key] = (relative, _sha256(path))
            except BaseException as exc:
                self._error = exc
                raise ReplayArchiveError("Could not publish replay snapshot manifest.") from exc
        relative, digest = self._snapshots[key]
        return {
            "schema": "ambi-replay-reference",
            "version": _VERSION,
            "archive_id": self.archive_id,
            "archive_path": os.path.relpath(self.root, Path(checkpoint_path).resolve().parent),
            "manifest_path": relative,
            "manifest_sha256": digest,
            "step": step,
        }

    def close(self):
        """Flush and release the worker; immutable references remain usable."""
        try:
            self.flush()
        finally:
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None


def _safe_child(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Replay member path must be a nonempty relative path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Replay member path escapes its archive directory.")
    return path


def _checked_hash(path, expected, description):
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"{description} SHA-256 mismatch.")


def _load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Replay metadata must be a JSON object.")
    return value


def _schema(value, name):
    if not isinstance(value, dict) or value.get("schema") != name or value.get("version") != _VERSION:
        raise ValueError(f"Unsupported {name} schema/version.")


class CheckpointReplay:
    """Read-only raw resident replay with copy-out episode fragments."""

    def __init__(self, manifest, fields):
        self._manifest = copy.deepcopy(manifest)
        self._fragments = []
        resident_start = manifest["resident_start"]
        for record in manifest["episodes"]:
            start = max(record["start"], resident_start)
            selection = slice(start - resident_start, record["stop"] - resident_start)
            fragment = {name: value[selection] for name, value in fields.items()}
            _validate_episode_values(fragment, includes_initial=start == record["start"])
            if not bool((fragment["episode"] == record["episode_id"]).all()):
                raise ValueError("Replay chunk episode IDs disagree with its manifest.")
            self._fragments.append({
                "episode_id": record["episode_id"],
                "episode_offset": start - record["start"],
                "global_start": start,
                "fields": fragment,
            })

    @property
    def metadata(self):
        return copy.deepcopy(self._manifest)

    @property
    def num_rows(self):
        return self._manifest["resident_rows"]

    @property
    def num_transitions(self):
        """Resident non-initial rows, matching Buffer's eviction accounting."""
        return self._manifest["resident_transitions"]

    @property
    def capacity(self):
        return self._manifest["capacity"]

    @property
    def num_episodes(self):
        return len(self._fragments)

    @property
    def fragments(self):
        return tuple({
            **{key: value for key, value in fragment.items() if key != "fields"},
            "fields": {name: value.clone() for name, value in fragment["fields"].items()},
        } for fragment in self._fragments)

    def sample_sequences(self, batch_size, horizon, seed=0):
        """Uniform valid starts, sampled with replacement using a private RNG.

        Return the raw ``Buffer.sample`` tuple on CPU: time-major observations,
        actions, rewards, termination flags, and ``None`` for the single task.
        """
        _integer(batch_size, "batch_size", minimum=1)
        _integer(horizon, "horizon", minimum=1)
        _integer(seed, "seed")
        counts = [max(0, len(part["fields"]["obs"]) - horizon) for part in self._fragments]
        total = sum(counts)
        if not total:
            raise ReplayArchiveError(f"Checkpoint replay has no sequences of horizon {horizon}.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        choices = torch.randint(total, (batch_size,), generator=generator)
        boundaries = torch.tensor(counts, dtype=torch.int64).cumsum(0)
        indices = torch.searchsorted(boundaries, choices, right=True)
        samples = []
        for choice, index in zip(choices.tolist(), indices.tolist()):
            start = choice - (int(boundaries[index - 1]) if index else 0)
            fields = self._fragments[index]["fields"]
            samples.append({name: fields[name][start:start + horizon + 1] for name in _FIELDS[:-1]})
        batch = {name: torch.stack([sample[name] for sample in samples], dim=1) for name in _FIELDS[:-1]}
        return (
            batch["obs"], batch["action"][1:].contiguous(),
            batch["reward"][1:].unsqueeze(-1).contiguous(),
            batch["terminated"][1:].unsqueeze(-1).contiguous(), None,
        )


def _read_replay(checkpoint_path, archive_root):
    checkpoint_path = Path(checkpoint_path).resolve()
    sidecar = _load_json(f"{checkpoint_path}.metadata.json")
    reference = sidecar.get("replay")
    if reference is None:
        raise ValueError("Checkpoint has no replay archive reference.")
    _schema(reference, "ambi-replay-reference")
    _checked_hash(checkpoint_path, reference.get("checkpoint_sha256"), "Checkpoint")
    _integer(reference["step"], "reference step")
    checkpoint_metadata = sidecar.get("checkpoint")
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("Checkpoint sidecar is missing checkpoint metadata.")
    if checkpoint_metadata.get("step") != reference["step"]:
        raise ValueError("Replay reference step differs from checkpoint metadata.")
    root = Path(archive_root).resolve() if archive_root is not None else (
        checkpoint_path.parent / reference["archive_path"]
    ).resolve()
    path = _safe_child(root, reference["manifest_path"])
    _checked_hash(path, reference["manifest_sha256"], "Replay manifest")
    manifest = _load_json(path)
    _schema(manifest, "ambi-replay-snapshot")
    if manifest["archive_id"] != reference["archive_id"] or manifest["step"] != reference["step"]:
        raise ValueError("Replay manifest identity differs from checkpoint reference.")
    for name in ("step", "total_rows", "total_episodes", "total_transitions", "resident_start", "resident_rows", "resident_transitions"):
        _integer(manifest[name], name)
    capacity = _integer(manifest["capacity"], "capacity")
    chunk_rows = _integer(manifest["chunk_rows"], "chunk_rows", minimum=1)
    if chunk_rows > 65536:
        raise ValueError("Replay chunk row limit exceeds 65536.")
    shape = tuple(manifest["observation_shape"])
    if not shape:
        raise ValueError("Replay observation shape is empty.")
    for dim in shape:
        _integer(dim, "observation dimension", minimum=1)
    action_dim = _integer(manifest["action_dim"], "action_dim", minimum=1)
    rows, start = manifest["total_rows"], manifest["resident_start"]
    if (
        start != max(0, rows - capacity)
        or manifest["resident_rows"] != rows - start
        or manifest["total_transitions"] != rows - manifest["total_episodes"]
        or manifest["step"] < manifest["total_transitions"]
    ):
        raise ValueError("Replay snapshot row/transition accounting is inconsistent.")
    episodes, chunks = manifest["episodes"], manifest["chunks"]
    if not isinstance(episodes, list) or not isinstance(chunks, list):
        raise ValueError("Replay episode and chunk inventories must be lists.")
    resident_rows = manifest["resident_rows"]
    if bool(resident_rows) != bool(episodes) or bool(resident_rows) != bool(chunks):
        raise ValueError("Replay empty-state inventories are inconsistent.")
    cursor = start
    for index, record in enumerate(episodes):
        for name in ("episode_id", "start", "stop"):
            _integer(record[name], f"episode {name}")
        if (
            record["episode_id"] != manifest["total_episodes"] - len(episodes) + index
            or not record["start"] <= cursor < record["stop"] <= rows
            or (index and record["start"] != cursor)
        ):
            raise ValueError("Replay episode inventory is missing, overlapping, or out of order.")
        cursor = record["stop"]
    initial_rows = sum(record["start"] >= start for record in episodes)
    if cursor != rows or manifest["resident_transitions"] != rows - start - initial_rows:
        raise ValueError("Replay resident episode accounting is inconsistent.")
    parts = {name: [] for name in _FIELDS}
    cursor = start
    for index, record in enumerate(chunks):
        lo, hi = record["start"], record["stop"]
        _integer(lo, "chunk start")
        _integer(hi, "chunk stop")
        if not lo <= cursor < hi <= rows or hi - lo > chunk_rows or (index and lo != cursor):
            raise ValueError("Replay chunks are missing, overlapping, or out of order.")
        path = _safe_child(root, record["path"])
        _checked_hash(path, record["sha256"], "Replay chunk")
        chunk = torch.load(path, map_location="cpu", weights_only=True)
        _schema(chunk, "ambi-replay-chunk")
        if chunk["archive_id"] != manifest["archive_id"] or chunk["start"] != lo or chunk["stop"] != hi:
            raise ValueError("Replay chunk identity differs from manifest.")
        _validate_fields(chunk["fields"], hi - lo, shape, action_dim)
        for name in _FIELDS:
            parts[name].append(chunk["fields"][name][max(start - lo, 0):].clone())
        cursor = hi
    if cursor != rows:
        raise ValueError("Replay chunk inventory ends before the snapshot.")
    fields = {name: torch.cat(values, dim=0) for name, values in parts.items()} if resident_rows else {}
    return CheckpointReplay(manifest, fields)


def load_checkpoint_replay(checkpoint_path, archive_root=None):
    """Verify model/manifest/chunk hashes and load its resident raw replay.

    ``archive_root`` optionally names the actual relocated ``archive-<id>``
    directory. Model weights are hashed but never loaded or modified.
    """
    try:
        return _read_replay(checkpoint_path, archive_root)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ReplayArchiveError(f"Cannot load checkpoint replay: {exc}") from exc
