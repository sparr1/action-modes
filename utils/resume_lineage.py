"""Small atomic checkpoint store for exact segmented training.

The store owns a process-lifetime ``flock`` and publishes immutable generation
directories. A generation becomes selectable only after all payloads and its
manifest are durable, the directory is atomically renamed, and ``LATEST`` is
atomically replaced. Loading never searches for a usable fallback.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable


LINEAGE_SCHEMA_VERSION = 1
GENERATION_SCHEMA_VERSION = 1
LINEAGE_MANIFEST = "LINEAGE.json"
GENERATIONS_DIRECTORY = "generations"
GENERATION_MANIFEST = "manifest.json"
MANIFEST_CHECKSUM = "manifest.sha256"
LATEST_POINTER = "LATEST"
LOCK_FILE = ".lineage.lock"

_GENERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ROLE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class LineageError(RuntimeError):
    """Base class for lineage failures."""


class LineageModeError(LineageError):
    """The requested new/required operation is invalid."""


class LineageLockError(LineageError):
    """The lineage lease could not be acquired or released."""


class LineageConcurrentWriterError(LineageLockError):
    """Another process owns the lineage lease."""


class LineageStorageError(LineageError):
    """A durable filesystem operation failed."""


class LineageCorruptionError(LineageError):
    """A stored manifest or payload is malformed or corrupt."""


class LineageCompatibilityError(LineageError):
    """Stored immutable metadata differs from the requested run."""


class LineageGenerationNotFoundError(LineageError):
    """LATEST or an explicitly selected generation does not exist."""


class LineageGenerationExistsError(LineageError):
    """An immutable generation ID was reused."""


FileWriter = Callable[[Path], None]
FaultInjector = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class GenerationFile:
    path: str
    role: str
    writer: FileWriter


@dataclass(frozen=True)
class GenerationFileRecord:
    path: str
    role: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Generation:
    generation_id: str
    path: Path
    parent_generation: str | None
    metadata: Mapping[str, Any]
    files: tuple[GenerationFileRecord, ...]
    manifest_sha256: str

    def file_path(self, relative_path: str) -> Path:
        normalized = _payload_path(relative_path)
        if normalized not in {record.path for record in self.files}:
            raise LineageGenerationNotFoundError(
                f"Generation {self.generation_id!r} has no file {normalized!r}."
            )
        return self.path / PurePosixPath(normalized)

    def files_for_role(self, role: str) -> tuple[Path, ...]:
        normalized = _role(role)
        return tuple(
            self.path / PurePosixPath(record.path)
            for record in self.files
            if record.role == normalized
        )


def write_bytes(payload: bytes) -> FileWriter:
    frozen = bytes(payload)

    def writer(path: Path) -> None:
        with path.open("xb") as stream:
            stream.write(frozen)

    return writer


class LineageStore:
    """One exclusively leased lineage directory."""

    def __init__(self, root: Path, mode: str, fault_injector: FaultInjector | None):
        self.root = root
        self.mode = mode
        self.generations_path = root / GENERATIONS_DIRECTORY
        self._fault_injector = fault_injector
        self._lock_fd: int | None = None
        self._lineage_metadata: dict[str, Any] | None = None
        self._current: Generation | None = None
        self._validated: dict[str, Generation] = {}

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        mode: str,
        lineage_metadata: Mapping[str, Any] | None = None,
        expected_lineage: Mapping[str, Any] | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> "LineageStore":
        mode = str(mode).strip().lower()
        if mode not in {"new", "required"}:
            raise LineageModeError("mode must be exactly 'new' or 'required'.")
        store = cls(Path(root), mode, fault_injector)
        try:
            if mode == "new":
                if lineage_metadata is None or expected_lineage is not None:
                    raise LineageModeError(
                        "new mode requires lineage_metadata and forbids expected_lineage."
                    )
                store._create(lineage_metadata)
            else:
                if lineage_metadata is not None:
                    raise LineageModeError(
                        "lineage_metadata is only valid in new mode."
                    )
                store._open_existing(expected_lineage)
            return store
        except BaseException:
            store._release_after_failed_open()
            raise

    @property
    def lineage_metadata(self) -> Mapping[str, Any]:
        self._require_open()
        assert self._lineage_metadata is not None
        return copy.deepcopy(self._lineage_metadata)

    def __enter__(self) -> "LineageStore":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except LineageLockError as close_error:
            if exc is None:
                raise
            note = getattr(exc, "add_note", None)
            if callable(note):
                note(f"Additionally failed to release lineage lock: {close_error}")
        return False

    def close(self) -> None:
        fd = self._lock_fd
        if fd is None:
            return
        self._lock_fd = None
        error = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            error = exc
        try:
            os.close(fd)
        except OSError as exc:
            error = error or exc
        if error is not None:
            raise LineageLockError(f"Could not release {self.root / LOCK_FILE}: {error}")

    def publish(
        self,
        generation_id: str,
        *,
        files: Iterable[GenerationFile],
        metadata: Mapping[str, Any],
        source_generation_id: str | None = None,
    ) -> Generation:
        """Publish a generation without rereading payloads just written."""

        self._require_open()
        generation_id = _generation_id(generation_id)
        metadata = _json_mapping(metadata, "generation metadata")
        specifications = _file_specs(files)
        final = self.generations_path / generation_id
        if final.exists() or final.is_symlink():
            raise LineageGenerationExistsError(
                f"Generation {generation_id!r} already exists."
            )

        parent = self._publication_parent(source_generation_id)
        staging: Path | None = None
        try:
            staging = Path(
                tempfile.mkdtemp(
                    dir=self.generations_path,
                    prefix=f".{generation_id}.",
                    suffix=".tmp",
                )
            )
            self._fault("after_staging_created", generation_id=generation_id)
            records = []
            for specification in specifications:
                output = staging / PurePosixPath(specification.path)
                output.parent.mkdir(parents=True, exist_ok=True)
                specification.writer(output)
                _require_regular(output, f"payload {specification.path!r}")
                _fsync_file(output)
                records.append(
                    GenerationFileRecord(
                        path=specification.path,
                        role=specification.role,
                        size=output.stat().st_size,
                        sha256=_sha256(output),
                    )
                )
            self._fault("after_payloads_written", generation_id=generation_id)

            manifest = {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "generation_id": generation_id,
                "parent_generation": parent,
                "metadata": metadata,
                "files": [record.__dict__ for record in records],
            }
            manifest_bytes = _json_bytes(manifest)
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            _write_new_durable(staging / GENERATION_MANIFEST, manifest_bytes)
            _write_new_durable(
                staging / MANIFEST_CHECKSUM,
                f"{manifest_sha256}\n".encode("ascii"),
            )
            _fsync_directories(staging)
            self._fault("before_generation_rename", generation_id=generation_id)
            os.rename(staging, final)
            staging = None
            _fsync_directory(self.generations_path)
            self._fault("after_generation_rename", generation_id=generation_id)
            self._write_latest(generation_id)

            generation = Generation(
                generation_id=generation_id,
                path=final,
                parent_generation=parent,
                metadata=copy.deepcopy(metadata),
                files=tuple(records),
                manifest_sha256=manifest_sha256,
            )
            previous = self._validated.get(parent) if parent is not None else None
            self._validated[generation_id] = generation
            self._current = generation
            self._fault("before_retention", generation_id=generation_id)
            self._prune_known_ancestor(previous, keep={generation_id, parent})
            return generation
        except BaseException as exc:
            if staging is not None:
                try:
                    shutil.rmtree(staging)
                except OSError as cleanup_error:
                    note = getattr(exc, "add_note", None)
                    if callable(note):
                        note(
                            f"Additionally could not remove incomplete generation "
                            f"{staging}: {cleanup_error}"
                        )
            if isinstance(exc, OSError):
                raise LineageStorageError(
                    f"Could not publish generation {generation_id!r}: {exc}"
                ) from exc
            raise

    def load(self, generation_id: str | None = None) -> Generation:
        """Load exactly LATEST or the explicitly named generation."""

        self._require_open()
        try:
            latest_id = self._read_latest()
            latest = self._load_generation(latest_id)
            if generation_id is None:
                selected = latest_id
                generation = latest
            else:
                selected = _generation_id(generation_id)
                if selected not in {latest_id, latest.parent_generation}:
                    raise LineageGenerationNotFoundError(
                        "An explicit rollback may select only LATEST or its "
                        "retained predecessor."
                    )
                generation = latest if selected == latest_id else self._load_generation(selected)
            self._validated[selected] = generation
            if generation_id is None:
                self._current = generation
                if generation.parent_generation is not None:
                    previous = self._load_generation(generation.parent_generation)
                    self._validated[previous.generation_id] = previous
            return generation
        except OSError as exc:
            raise LineageStorageError(f"Could not load a generation: {exc}") from exc

    def _create(self, metadata: Mapping[str, Any]) -> None:
        if self.root.exists() or self.root.is_symlink():
            raise LineageModeError(f"New lineage path already exists: {self.root}")
        normalized = _json_mapping(metadata, "lineage metadata")
        try:
            _mkdir_durable(self.root)
            self.generations_path.mkdir()
            self._acquire_lock()
            _write_new_durable(
                self.root / LINEAGE_MANIFEST,
                _json_bytes(
                    {
                        "schema_version": LINEAGE_SCHEMA_VERSION,
                        "metadata": normalized,
                    }
                ),
            )
            _fsync_directory(self.root)
        except OSError as exc:
            raise LineageStorageError(f"Could not create lineage {self.root}: {exc}") from exc
        self._lineage_metadata = normalized

    def _open_existing(self, expected: Mapping[str, Any] | None) -> None:
        if self.root.is_symlink() or not self.root.is_dir():
            raise LineageModeError(f"Required lineage does not exist: {self.root}")
        if self.generations_path.is_symlink() or not self.generations_path.is_dir():
            raise LineageCorruptionError("Lineage generations directory is missing.")
        self._acquire_lock()
        document = _read_json(self.root / LINEAGE_MANIFEST, "lineage manifest")
        if set(document) != {"schema_version", "metadata"}:
            raise LineageCorruptionError("Lineage manifest fields are invalid.")
        if document["schema_version"] != LINEAGE_SCHEMA_VERSION:
            raise LineageCorruptionError("Unsupported lineage schema version.")
        metadata = document["metadata"]
        if not isinstance(metadata, dict):
            raise LineageCorruptionError("Lineage metadata must be an object.")
        if expected is not None and metadata != _json_mapping(expected, "expected lineage"):
            raise LineageCompatibilityError(
                "Stored lineage metadata does not match the requested run."
            )
        self._lineage_metadata = copy.deepcopy(metadata)

    def _acquire_lock(self) -> None:
        path = self.root / LOCK_FILE
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise LineageConcurrentWriterError(
                    f"Another process owns lineage {self.root}."
                ) from exc
            os.ftruncate(fd, 0)
            record = json.dumps(
                {"pid": os.getpid(), "host": socket.gethostname()}, sort_keys=True
            ).encode()
            os.write(fd, record + b"\n")
            os.fsync(fd)
            self._lock_fd = fd
        except LineageConcurrentWriterError:
            raise
        except OSError as exc:
            raise LineageLockError(f"Could not lock lineage {self.root}: {exc}") from exc

    def _release_after_failed_open(self) -> None:
        if self._lock_fd is None:
            return
        try:
            self.close()
        except LineageLockError:
            pass

    def _publication_parent(self, selected: str | None) -> str | None:
        if selected is not None:
            selected = _generation_id(selected)
            if selected not in self._validated:
                self._validated[selected] = self._load_generation(selected)
            return selected
        if self._current is not None:
            return self._current.generation_id
        try:
            latest = self._read_latest()
        except LineageGenerationNotFoundError:
            if self.mode == "required":
                raise
            return None
        generation = self._load_generation(latest)
        self._validated[latest] = generation
        self._current = generation
        return latest

    def _write_latest(self, generation_id: str) -> None:
        self._fault("before_latest_replace", generation_id=generation_id)
        _atomic_write(self.root / LATEST_POINTER, f"{generation_id}\n".encode("ascii"))
        _fsync_directory(self.root)
        self._fault("after_latest_replace", generation_id=generation_id)

    def _read_latest(self) -> str:
        path = self.root / LATEST_POINTER
        if not path.exists():
            raise LineageGenerationNotFoundError(
                f"Lineage {self.root} has no {LATEST_POINTER}."
            )
        payload = _read_regular(path, LATEST_POINTER, maximum_size=130)
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise LineageCorruptionError("LATEST is not ASCII.") from exc
        if not text.endswith("\n") or text.count("\n") != 1:
            raise LineageCorruptionError("LATEST must contain one ID and a newline.")
        return _stored_generation_id(text[:-1], "LATEST")

    def _load_generation(self, generation_id: str) -> Generation:
        directory = self.generations_path / generation_id
        if directory.is_symlink() or not directory.is_dir():
            raise LineageGenerationNotFoundError(
                f"Generation {generation_id!r} does not exist."
            )
        manifest_path = directory / GENERATION_MANIFEST
        manifest_bytes = _read_regular(manifest_path, "generation manifest")
        expected_manifest_sha = _read_regular(
            directory / MANIFEST_CHECKSUM,
            "generation manifest checksum",
            maximum_size=65,
        )
        actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if expected_manifest_sha != f"{actual_manifest_sha}\n".encode("ascii"):
            raise LineageCorruptionError("Generation manifest checksum differs.")
        manifest = _decode_json(manifest_bytes, "generation manifest")
        if set(manifest) != {
            "schema_version",
            "generation_id",
            "parent_generation",
            "metadata",
            "files",
        }:
            raise LineageCorruptionError("Generation manifest fields are invalid.")
        if manifest["schema_version"] != GENERATION_SCHEMA_VERSION:
            raise LineageCorruptionError("Unsupported generation schema version.")
        if manifest["generation_id"] != generation_id:
            raise LineageCorruptionError("Generation directory and manifest IDs differ.")
        parent = manifest["parent_generation"]
        if parent is not None:
            parent = _stored_generation_id(parent, "parent_generation")
            if parent == generation_id:
                raise LineageCorruptionError("A generation cannot parent itself.")
        metadata = manifest["metadata"]
        raw_files = manifest["files"]
        if not isinstance(metadata, dict) or not isinstance(raw_files, list) or not raw_files:
            raise LineageCorruptionError("Generation metadata/files are malformed.")

        records = []
        seen = set()
        for entry in raw_files:
            if not isinstance(entry, dict) or set(entry) != {
                "path",
                "role",
                "size",
                "sha256",
            }:
                raise LineageCorruptionError("Generation file entry is malformed.")
            path = _stored_payload_path(entry["path"])
            role = _stored_role(entry["role"])
            size = entry["size"]
            digest = entry["sha256"]
            if path in seen:
                raise LineageCorruptionError(f"Duplicate payload {path!r}.")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise LineageCorruptionError(f"Invalid size for {path!r}.")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise LineageCorruptionError(f"Invalid checksum for {path!r}.")
            payload = directory / PurePosixPath(path)
            _require_regular(payload, f"payload {path!r}")
            if payload.stat().st_size != size or _sha256(payload) != digest:
                raise LineageCorruptionError(f"Payload {path!r} failed checksum validation.")
            seen.add(path)
            records.append(GenerationFileRecord(path, role, size, digest))
        return Generation(
            generation_id=generation_id,
            path=directory,
            parent_generation=parent,
            metadata=copy.deepcopy(metadata),
            files=tuple(records),
            manifest_sha256=actual_manifest_sha,
        )

    def _prune_known_ancestor(
        self, parent: Generation | None, *, keep: set[str | None]
    ) -> None:
        if parent is None or parent.parent_generation is None:
            return
        older_id = parent.parent_generation
        if older_id in keep or older_id not in self._validated:
            return
        older = self._validated.pop(older_id)
        try:
            shutil.rmtree(older.path)
            _fsync_directory(self.generations_path)
        except OSError as exc:
            raise LineageStorageError(
                f"Could not prune old generation {older_id!r}: {exc}"
            ) from exc

    def _fault(self, stage: str, **context: Any) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage, context)

    def _require_open(self) -> None:
        if self._lock_fd is None:
            raise LineageLockError(f"Lineage store {self.root} is closed.")


def _generation_id(value: Any) -> str:
    if not isinstance(value, str) or _GENERATION_ID.fullmatch(value) is None:
        raise LineageCompatibilityError("Invalid generation ID.")
    return value


def _stored_generation_id(value: Any, label: str) -> str:
    try:
        return _generation_id(value)
    except LineageCompatibilityError as exc:
        raise LineageCorruptionError(f"Stored {label} is invalid.") from exc


def _payload_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise LineageCompatibilityError("Payload path must be a portable relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise LineageCompatibilityError(f"Unsafe payload path {value!r}.")
    if value in {GENERATION_MANIFEST, MANIFEST_CHECKSUM}:
        raise LineageCompatibilityError(f"{value} is reserved.")
    return value


def _stored_payload_path(value: Any) -> str:
    try:
        return _payload_path(value)
    except LineageCompatibilityError as exc:
        raise LineageCorruptionError("Stored payload path is invalid.") from exc


def _role(value: Any) -> str:
    if not isinstance(value, str) or _ROLE.fullmatch(value) is None:
        raise LineageCompatibilityError("Invalid generation file role.")
    return value


def _stored_role(value: Any) -> str:
    try:
        return _role(value)
    except LineageCompatibilityError as exc:
        raise LineageCorruptionError("Stored generation file role is invalid.") from exc


def _file_specs(files: Iterable[GenerationFile]) -> tuple[GenerationFile, ...]:
    result = []
    seen = set()
    for item in files:
        if not isinstance(item, GenerationFile):
            raise TypeError("Generation files must be GenerationFile values.")
        path = _payload_path(item.path)
        if path in seen or not callable(item.writer):
            raise LineageCompatibilityError(f"Duplicate or invalid writer for {path!r}.")
        seen.add(path)
        result.append(GenerationFile(path, _role(item.role), item.writer))
    if not result:
        raise LineageCompatibilityError("A generation must contain at least one payload.")
    return tuple(sorted(result, key=lambda item: item.path))


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LineageCompatibilityError(f"{label} must be a mapping.")
    try:
        return _decode_json(_json_bytes(dict(value)), label)
    except (TypeError, ValueError) as exc:
        raise LineageCompatibilityError(
            f"{label} must contain finite JSON values: {exc}"
        ) from exc


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def _decode_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), parse_constant=_reject_constant
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise LineageCorruptionError(f"{label} is not valid finite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LineageCorruptionError(f"{label} must be a JSON object.")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return _decode_json(_read_regular(path, label), label)


def _require_regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise LineageCorruptionError(f"Missing {label} at {path}.") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise LineageCorruptionError(f"{label} is not a regular file.")


def _read_regular(path: Path, label: str, maximum_size: int | None = None) -> bytes:
    _require_regular(path, label)
    if maximum_size is not None and path.stat().st_size > maximum_size:
        raise LineageCorruptionError(f"{label} is unexpectedly large.")
    return path.read_bytes()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write(path: Path, payload: bytes) -> None:
    fd = -1
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except BaseException as exc:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                note = getattr(exc, "add_note", None)
                if callable(note):
                    note(f"Additionally could not remove {temporary}: {cleanup_error}")
        raise


def _mkdir_durable(path: Path) -> None:
    """Create a directory chain and sync each new parent entry."""

    missing = []
    current = path
    while not current.exists():
        if current.parent == current:
            raise OSError(f"No existing ancestor for {path}")
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise OSError(f"Lineage ancestor is not a directory: {current}")
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


__all__ = [
    "GENERATION_MANIFEST",
    "GENERATIONS_DIRECTORY",
    "LATEST_POINTER",
    "LINEAGE_MANIFEST",
    "MANIFEST_CHECKSUM",
    "Generation",
    "GenerationFile",
    "GenerationFileRecord",
    "LineageCompatibilityError",
    "LineageConcurrentWriterError",
    "LineageCorruptionError",
    "LineageError",
    "LineageGenerationExistsError",
    "LineageGenerationNotFoundError",
    "LineageLockError",
    "LineageModeError",
    "LineageStorageError",
    "LineageStore",
    "write_bytes",
]
