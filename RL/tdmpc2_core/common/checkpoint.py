"""Exact-state, atomic checkpoint helpers for TD-MPC2 agents.

Periodic checkpoints freeze tensor storage on the training stream before the
caller can mutate parameters again.  CUDA tensors are first cloned on that
stream, then copied into pinned host buffers on a dedicated copy stream.  The
background worker only waits for those copies and serializes CPU-owned data;
it never reads live model or optimizer storage.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from utils.checkpointing import (
    CheckpointTarget,
    fsync_checkpoint_directories,
    fsync_checkpoint_files,
    invalidate_metadata_sidecars,
    write_metadata_atomic,
)


class FrozenCheckpoint:
    """CPU-owned checkpoint state plus any outstanding device-copy events."""

    def __init__(self, state: Any, events=(), keepalive=()):
        self.state = state
        self._events = tuple(events)
        self._keepalive = tuple(keepalive)

    def wait(self):
        for event in self._events:
            event.synchronize()
        self._events = ()
        self._keepalive = ()
        return self.state


def freeze_checkpoint(state: Any) -> FrozenCheckpoint:
    """Take an immutable snapshot without synchronizing CUDA with the host."""

    memo = {}
    cuda_copies = {}

    def freeze(value):
        object_id = id(value)
        if object_id in memo:
            return memo[object_id]

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.device.type == "cuda" and tensor.layout == torch.strided:
                # The clone is ordered after all preceding training work on the
                # tensor's current stream, and owns storage that future optimizer
                # steps cannot mutate.
                frozen = tensor.clone(memory_format=torch.preserve_format)
                host = torch.empty_strided(
                    frozen.size(),
                    frozen.stride(),
                    dtype=frozen.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                cuda_copies.setdefault(frozen.device, []).append((host, frozen))
                result = host
            else:
                # CPU snapshots are cheap immutable copies. Other accelerators
                # currently use their safe synchronous host-transfer path.
                result = tensor.to(device="cpu", copy=True)
            memo[object_id] = result
            return result

        if isinstance(value, dict):
            try:
                result = value.__class__()
            except TypeError:
                result = {}
            memo[object_id] = result
            for key, item in value.items():
                result[copy.deepcopy(key)] = freeze(item)
            return result
        if isinstance(value, list):
            result = []
            memo[object_id] = result
            result.extend(freeze(item) for item in value)
            return result
        if isinstance(value, tuple):
            result = tuple(freeze(item) for item in value)
            memo[object_id] = result
            return result

        result = copy.deepcopy(value)
        memo[object_id] = result
        return result

    frozen_state = freeze(state)
    events = []
    keepalive = []
    for device, copies in cuda_copies.items():
        with torch.cuda.device(device):
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
            copy_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(ready)
                for host, frozen in copies:
                    host.copy_(frozen, non_blocking=True)
                complete = torch.cuda.Event()
                complete.record(copy_stream)
        events.append(complete)
        keepalive.extend((ready, copy_stream))
        keepalive.extend(frozen for _, frozen in copies)
    return FrozenCheckpoint(frozen_state, events=events, keepalive=keepalive)


def _atomic_torch_save(state: Any, fp, *, durable=True):
    """Serialize to a sibling temporary file, then atomically replace ``fp``."""

    if not isinstance(fp, (str, os.PathLike)):
        torch.save(state, fp)
        return fp

    target = Path(fp)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        if durable:
            fsync_checkpoint_files((temporary,))
        os.replace(temporary, target)
        if durable:
            fsync_checkpoint_directories((target,))
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return str(target)


def save_checkpoint(state: Any, fp):
    """Synchronously freeze and durably write a public explicit save."""

    return _atomic_torch_save(freeze_checkpoint(state).wait(), fp)


def _atomic_clone(source, target):
    """Publish an already-written identical checkpoint under another name."""

    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return str(target)


class AsyncCheckpointWriter:
    """Single-slot background serializer with synchronous explicit saves."""

    def __init__(self):
        self._executor = None
        self._future: Future | None = None
        self._pending_signature = None
        self._last_signature = None
        self._last_path = None

    def _ensure_executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="tdmpc2-checkpoint",
            )

    @staticmethod
    def _write(snapshot, path):
        return _atomic_torch_save(snapshot.wait(), path)

    @staticmethod
    def _normalize_publications(publications):
        normalized = []
        for publication in publications:
            if isinstance(publication, CheckpointTarget):
                path = publication.path
                metadata = publication.metadata
            elif isinstance(publication, (tuple, list)) and len(publication) == 2:
                path, metadata = publication
            else:
                path, metadata = publication, None
            normalized.append((os.fspath(path), copy.deepcopy(metadata)))
        if not normalized:
            raise ValueError("At least one checkpoint publication target is required.")
        return tuple(normalized)

    @staticmethod
    def _write_many(snapshot, publications):
        state = snapshot.wait()
        first, _ = publications[0]
        first = Path(first)
        first.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staging = tempfile.mkstemp(
            dir=first.parent,
            prefix=f".{first.name}.",
            suffix=".staging",
        )
        os.close(descriptor)
        staging = Path(staging)
        try:
            _atomic_torch_save(state, staging, durable=False)
            invalidate_metadata_sidecars(path for path, _ in publications)
            os.replace(staging, first)
            paths = [str(first)]
            for path, _ in publications[1:]:
                paths.append(_atomic_clone(first, path))
            fsync_checkpoint_files(path for path, _ in publications)
            fsync_checkpoint_directories(path for path, _ in publications)
            for path, metadata in publications:
                if metadata is not None:
                    write_metadata_atomic(path, metadata)
            return tuple(paths)
        finally:
            try:
                staging.unlink()
            except FileNotFoundError:
                pass

    def enqueue(self, state, path, *, signature=None):
        """Queue one periodic save after making its exact-state snapshot."""

        # Bound memory and surface any previous write error at the next
        # checkpoint boundary instead of accumulating stale snapshots.
        self.flush()
        snapshot = freeze_checkpoint(state)
        self._ensure_executor()
        self._pending_signature = signature
        self._future = self._executor.submit(self._write, snapshot, path)

    def enqueue_many(self, state, publications, *, signature=None):
        """Queue one exact snapshot for several atomically published aliases."""

        self.flush()
        normalized = self._normalize_publications(publications)
        snapshot = freeze_checkpoint(state)
        self._ensure_executor()
        self._pending_signature = signature
        self._future = self._executor.submit(self._write_many, snapshot, normalized)

    def flush(self):
        """Wait until the most recently queued periodic checkpoint is durable."""

        if self._future is None:
            return self._last_path
        future = self._future
        signature = self._pending_signature
        self._future = None
        self._pending_signature = None
        result = future.result()
        path = result[0] if isinstance(result, (tuple, list)) else result
        self._last_path = os.path.abspath(os.fspath(path))
        self._last_signature = signature
        return path

    def save(self, state, path, *, signature=None):
        """Complete an explicit save before returning.

        If the final state is byte-for-byte represented by the last periodic
        snapshot, publish that immutable file rather than snapshotting and
        serializing the same tensors again.
        """

        return self.save_many(state, (path,), signature=signature)[0]

    def save_many(self, state, publications, *, signature=None):
        """Synchronously publish one exact state under one or more names."""

        self.flush()
        normalized = self._normalize_publications(publications)
        targets = tuple(os.path.abspath(os.fspath(path)) for path, _ in normalized)
        if (
            signature is not None
            and signature == self._last_signature
            and self._last_path is not None
            and os.path.exists(self._last_path)
        ):
            invalidate_metadata_sidecars(targets)
            for target in targets:
                if target != self._last_path:
                    _atomic_clone(self._last_path, target)
            fsync_checkpoint_files(targets)
            fsync_checkpoint_directories(targets)
            for target, (_, metadata) in zip(targets, normalized):
                if metadata is not None:
                    write_metadata_atomic(target, metadata)
            self._last_path = targets[0]
            return targets

        absolute_publications = tuple(
            (target, metadata)
            for target, (_, metadata) in zip(targets, normalized)
        )
        result = self._write_many(freeze_checkpoint(state), absolute_publications)
        self._last_path = targets[0]
        self._last_signature = signature
        return tuple(result)

    def invalidate(self):
        """Forget snapshot-reuse metadata after live state is replaced.

        A pending write must finish first; otherwise its later completion would
        repopulate the very cache this method is intended to invalidate.
        """

        self.flush()
        self._last_signature = None
        self._last_path = None

    def shutdown(self):
        """Flush writes and release the background thread; the writer is reusable."""

        try:
            self.flush()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
