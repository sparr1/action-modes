"""Shared policy and atomic publication helpers for training checkpoints.

The tracker in this module deliberately knows nothing about PyTorch, SB3, or
TD-MPC2.  It decides *which* aliases should be published for one immutable
training state; backend adapters serialize that state exactly once and use the
publication helpers below to expose every requested alias.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


CHECKPOINT_METADATA_SCHEMA_VERSION = 1
SUPPORTED_SAVE_STRATEGIES = ("all", "best", "latest", "none")
_STRATEGY_ORDER = ("all", "best", "latest")


def normalize_save_strat(value: str | Sequence[str] | None = None) -> tuple[str, ...]:
    """Normalize a checkpoint strategy string/list into a deterministic tuple.

    ``None`` retains the historical behavior of keeping every numbered
    checkpoint.  ``last`` remains accepted as a compatibility alias for
    ``latest``.  ``none`` is intentionally exclusive so a configuration cannot
    simultaneously request and disable checkpoint publication.
    """

    if value is None:
        raw = ["all"]
    elif isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise TypeError("save_strat must be a string, list, tuple, or null.")

    if not raw:
        raise ValueError("save_strat cannot be empty; use 'none' to disable saves.")

    normalized = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError("Every save_strat entry must be a string.")
        strategy = item.strip().lower()
        if strategy == "last":
            strategy = "latest"
        if strategy not in SUPPORTED_SAVE_STRATEGIES:
            raise ValueError(
                f"Unsupported save_strat entry {item!r}; expected one of "
                f"{SUPPORTED_SAVE_STRATEGIES} (or legacy alias 'last')."
            )
        if strategy not in normalized:
            normalized.append(strategy)

    if "none" in normalized and len(normalized) != 1:
        raise ValueError("save_strat='none' cannot be combined with other strategies.")
    if normalized == ["none"]:
        return ("none",)
    return tuple(strategy for strategy in _STRATEGY_ORDER if strategy in normalized)


@dataclass(frozen=True)
class CheckpointConfig:
    """Resolved checkpoint settings for one algorithm/trial."""

    every: int | None
    strategies: tuple[str, ...]
    best_window: int

    @property
    def enabled(self) -> bool:
        return self.every is not None and self.strategies != ("none",)


def resolve_checkpoint_config(
    trial_run_params: Mapping[str, Any] | None,
    experiment_params: Mapping[str, Any] | None,
) -> CheckpointConfig:
    """Resolve per-algorithm checkpoint values before experiment fallbacks.

    Algorithm configuration values live at the top level of the resolved trial
    dictionary (beside ``alg`` and ``total_steps``).  A key explicitly present
    there wins even when its value is ``None``.  This permits one member of a
    multi-algorithm experiment to disable checkpointing while others inherit
    the experiment cadence.
    """

    trial = trial_run_params or {}
    experiment = experiment_params or {}

    def resolved(key, default=None):
        if key in trial:
            return trial[key]
        return experiment.get(key, default)

    every = resolved("checkpoint_every", None)
    strategies = normalize_save_strat(resolved("save_strat", None))
    best_window = resolved("checkpoint_best_window", 100)

    if every is not None:
        if isinstance(every, bool):
            raise ValueError("checkpoint_every must be a positive integer or null.")
        try:
            numeric_every = float(every)
            every = int(numeric_every)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("checkpoint_every must be a positive integer or null.") from exc
        if not math.isfinite(numeric_every) or every <= 0 or numeric_every != every:
            raise ValueError("checkpoint_every must be a positive integer or null.")

    if isinstance(best_window, bool):
        raise ValueError("checkpoint_best_window must be a positive integer.")
    try:
        numeric_window = float(best_window)
        best_window = int(numeric_window)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint_best_window must be a positive integer.") from exc
    if not math.isfinite(numeric_window) or best_window <= 0 or numeric_window != best_window:
        raise ValueError("checkpoint_best_window must be a positive integer.")

    return CheckpointConfig(every=every, strategies=strategies, best_window=best_window)


@dataclass(frozen=True)
class CheckpointTarget:
    """One public filename and its matching portable metadata document."""

    path: Path
    kind: str
    metadata: dict[str, Any]


def checkpoint_metadata(
    *,
    kind: str,
    step: int,
    episode: int,
    best_score: float | None,
    best_window: int,
    trial_run_params: Mapping[str, Any] | None,
    experiment_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a versioned, self-contained checkpoint sidecar payload."""

    return {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "checkpoint": {
            "kind": str(kind),
            "step": int(step),
            "episode": int(episode),
            "best_score": None if best_score is None else float(best_score),
            "best_window": int(best_window),
        },
        "trial_run_params": copy.deepcopy(dict(trial_run_params or {})),
        "experiment_params": copy.deepcopy(dict(experiment_params or {})),
    }


class CheckpointTracker:
    """Track completed returns and choose composable checkpoint aliases."""

    def __init__(
        self,
        save_freq: int,
        save_path: str | os.PathLike[str],
        name_prefix: str,
        *,
        save_strat: str | Sequence[str] | None = None,
        best_window: int = 100,
        periodic_step_suffix: str = "_steps",
        trial_run_params: Mapping[str, Any] | None = None,
        experiment_params: Mapping[str, Any] | None = None,
    ):
        config = resolve_checkpoint_config(
            {
                "checkpoint_every": save_freq,
                "save_strat": save_strat,
                "checkpoint_best_window": best_window,
            },
            {},
        )
        self.save_freq = int(config.every)
        self.save_path = Path(save_path)
        self.name_prefix = str(name_prefix)
        self.strategies = config.strategies
        self.best_window = config.best_window
        self.periodic_step_suffix = str(periodic_step_suffix)
        self.trial_run_params = copy.deepcopy(dict(trial_run_params or {}))
        self.experiment_params = copy.deepcopy(dict(experiment_params or {}))
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.strategies != ("none",)

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def best_score(self) -> float | None:
        return self._best_score

    @property
    def recent_returns(self) -> tuple[float, ...]:
        return tuple(self._returns)

    def reset(self) -> None:
        self._returns = deque(maxlen=self.best_window)
        self._episode_count = 0
        self._best_score = None

    def state_dict(self) -> dict[str, Any]:
        """Return dynamic policy state; lineage identity owns static config."""
        return {
            "schema_version": 2,
            "episode_count": self._episode_count,
            "best_score": self._best_score,
            "recent_returns": list(self._returns),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore tracker state after strict policy compatibility checks."""
        normalized = self.validate_state_dict(state)
        self._returns = deque(normalized["recent_returns"], maxlen=self.best_window)
        self._episode_count = normalized["episode_count"]
        self._best_score = normalized["best_score"]

    def validate_state_dict(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize resume policy state without mutation."""
        fields = {
            "schema_version",
            "episode_count",
            "best_score",
            "recent_returns",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != fields
            or state.get("schema_version") != 2
        ):
            raise ValueError("Unsupported CheckpointTracker resume schema.")
        episode_count = state.get("episode_count")
        if (
            isinstance(episode_count, bool)
            or not isinstance(episode_count, int)
            or episode_count < 0
        ):
            raise ValueError("CheckpointTracker episode_count is invalid.")
        returns = state.get("recent_returns")
        if (
            not isinstance(returns, list)
            or len(returns) > self.best_window
            or len(returns) > episode_count
        ):
            raise ValueError("CheckpointTracker recent_returns are invalid.")
        normalized_returns = []
        for value in returns:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("CheckpointTracker returns must be finite.")
            normalized_returns.append(value)
        best_score = state.get("best_score")
        if best_score is not None:
            best_score = float(best_score)
            if not math.isfinite(best_score):
                raise ValueError(
                    "CheckpointTracker best_score must be finite or null."
                )
            if "best" not in self.strategies or not normalized_returns:
                raise ValueError(
                    "CheckpointTracker best_score has no compatible best policy/history."
                )

        return {
            "episode_count": episode_count,
            "best_score": best_score,
            "recent_returns": normalized_returns,
        }

    def record_episode_return(self, episode_return: Any) -> None:
        """Record one completed episode, excluding non-finite scores."""

        self._episode_count += 1
        try:
            score = float(episode_return)
        except (TypeError, ValueError, OverflowError):
            return
        if math.isfinite(score):
            self._returns.append(score)

    def _metadata(self, kind: str, step: int) -> dict[str, Any]:
        return checkpoint_metadata(
            kind=kind,
            step=step,
            episode=self._episode_count,
            best_score=self._best_score,
            best_window=self.best_window,
            trial_run_params=self.trial_run_params,
            experiment_params=self.experiment_params,
        )

    def targets(self, step: int, *, final: bool = False) -> tuple[CheckpointTarget, ...]:
        """Return aliases to publish for a periodic or clean-final boundary."""

        step = int(step)
        if not self.enabled or step < 0:
            return ()
        if not final and (step == 0 or step % self.save_freq != 0):
            return ()

        kinds_and_names: list[tuple[str, str]] = []
        if not final and "all" in self.strategies:
            kinds_and_names.append(
                (
                    "periodic",
                    f"{self.name_prefix}_{step}{self.periodic_step_suffix}",
                )
            )

        if "best" in self.strategies and self._returns:
            # Divide before summing so a mathematically finite mean of very
            # large finite returns does not overflow in the intermediate sum.
            window_score = float(
                math.fsum(value / len(self._returns) for value in self._returns)
            )
            if self._best_score is None or window_score > self._best_score:
                self._best_score = window_score
                kinds_and_names.append(("best", f"{self.name_prefix}_best"))

        if "latest" in self.strategies:
            kinds_and_names.append(("latest", f"{self.name_prefix}_latest"))

        return tuple(
            CheckpointTarget(
                path=self.save_path / name,
                kind=kind,
                metadata=self._metadata(kind, step),
            )
            for kind, name in kinds_and_names
        )

    def explicit_target(
        self,
        path: str | os.PathLike[str],
        *,
        step: int,
        episode: int | None = None,
        kind: str = "trial_final",
    ) -> CheckpointTarget:
        """Create metadata for a save requested outside the periodic policy."""

        if episode is None:
            episode = self._episode_count
        metadata = checkpoint_metadata(
            kind=kind,
            step=step,
            episode=episode,
            best_score=self._best_score,
            best_window=self.best_window,
            trial_run_params=self.trial_run_params,
            experiment_params=self.experiment_params,
        )
        return CheckpointTarget(Path(path), kind, metadata)


def explicit_checkpoint_target(
    path: str | os.PathLike[str],
    *,
    step: int,
    episode: int,
    trial_run_params: Mapping[str, Any] | None,
    experiment_params: Mapping[str, Any] | None,
    best_window: int | None = None,
    best_score: float | None = None,
    kind: str = "trial_final",
) -> CheckpointTarget:
    """Build an explicit target when no periodic tracker was configured."""

    if best_window is None:
        best_window = resolve_checkpoint_config(
            trial_run_params, experiment_params
        ).best_window
    return CheckpointTarget(
        Path(path),
        kind,
        checkpoint_metadata(
            kind=kind,
            step=step,
            episode=episode,
            best_score=best_score,
            best_window=best_window,
            trial_run_params=trial_run_params,
            experiment_params=experiment_params,
        ),
    )


def with_extension(path: str | os.PathLike[str], extension: str | None) -> Path:
    """Append a backend extension unless the requested name already has it."""

    result = Path(path)
    if not extension:
        return result
    extension = extension if extension.startswith(".") else f".{extension}"
    if not str(result).endswith(extension):
        result = Path(f"{result}{extension}")
    return result


def metadata_path(checkpoint_path: str | os.PathLike[str]) -> Path:
    """Return ``<actual-checkpoint>.metadata.json`` without replacing suffixes."""

    return Path(f"{os.fspath(checkpoint_path)}.metadata.json")


def _fsync_directory(path: str | os.PathLike[str]) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_checkpoint_files(
    checkpoint_paths: Iterable[str | os.PathLike[str]],
) -> None:
    for path in checkpoint_paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def fsync_checkpoint_directories(
    checkpoint_paths: Iterable[str | os.PathLike[str]],
) -> None:
    for directory in sorted({Path(path).parent for path in checkpoint_paths}):
        _fsync_directory(directory)


def invalidate_metadata_sidecars(
    checkpoint_paths: Iterable[str | os.PathLike[str]],
) -> None:
    """Durably remove old sidecars before replacing their checkpoint aliases."""

    changed_directories = set()
    for checkpoint_path in checkpoint_paths:
        sidecar = metadata_path(checkpoint_path)
        try:
            sidecar.unlink()
        except FileNotFoundError:
            continue
        changed_directories.add(sidecar.parent)
    for directory in sorted(changed_directories):
        _fsync_directory(directory)


def _json_default(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_metadata_atomic(
    checkpoint_path: str | os.PathLike[str], metadata: Mapping[str, Any]
) -> str:
    """Atomically replace a sidecar after its checkpoint is durable."""

    target = metadata_path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return str(target)


def atomic_clone(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> str:
    """Atomically expose an already serialized file under another name."""

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


def publish_checkpoint(
    targets: Iterable[CheckpointTarget],
    serialize: Callable[[Path], str | os.PathLike[str] | None],
    *,
    extension: str | None = None,
) -> tuple[str, ...]:
    """Serialize once, then atomically publish all requested aliases.

    ``serialize`` receives a sibling staging filename whose suffix already
    matches the backend's actual suffix.  It may return the actual written path
    (useful for libraries which adjust filenames) or ``None`` when it wrote the
    supplied path exactly.
    """

    requested = tuple(targets)
    if not requested:
        return ()
    actual_targets = tuple(with_extension(target.path, extension) for target in requested)
    first = actual_targets[0]
    first.parent.mkdir(parents=True, exist_ok=True)
    normalized_extension = ""
    if extension:
        normalized_extension = extension if extension.startswith(".") else f".{extension}"
    descriptor, staging = tempfile.mkstemp(
        dir=first.parent,
        prefix=f".{first.name}.",
        suffix=f".staging{normalized_extension}",
    )
    os.close(descriptor)
    staging_path = Path(staging)
    written_path = staging_path
    try:
        result = serialize(staging_path)
        if result is not None:
            written_path = Path(result)
        if not written_path.is_file():
            appended = with_extension(written_path, extension)
            if appended.is_file():
                written_path = appended
            else:
                raise FileNotFoundError(
                    f"Checkpoint serializer did not create its promised output: {written_path}"
                )
        # A checkpoint and its sidecar cannot be atomically replaced as a pair.
        # Remove old sidecars durably before exposing any new checkpoint bytes,
        # so an interruption can only leave a missing sidecar, never stale
        # metadata describing the new checkpoint.
        invalidate_metadata_sidecars(actual_targets)
        os.replace(written_path, first)
        published = [str(first)]
        for target in actual_targets[1:]:
            published.append(atomic_clone(first, target))
        # Make checkpoint data durable before its directory entries, then
        # publish metadata that claims to describe those durable bytes. After
        # the durable sidecar unlink above, a crash can therefore recover only
        # the new pair or no sidecar.
        fsync_checkpoint_files(actual_targets)
        fsync_checkpoint_directories(actual_targets)
        for target, publication in zip(actual_targets, requested):
            write_metadata_atomic(target, publication.metadata)
        return tuple(published)
    finally:
        cleanup = {staging_path, written_path}
        for candidate in cleanup:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def supports_composable_checkpointing(model: Any) -> bool:
    """Whether a model explicitly opts into the current checkpoint contract."""

    return bool(getattr(model, "supports_composable_checkpointing", False))
