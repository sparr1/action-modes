"""Checkpoint-first W&B logging for one resumable training run.

Metric calls made during training only append to :class:`WandbEventBuffer`.
The buffer is part of the trainer checkpoint, and the caller explicitly
publishes that checkpointed prefix *after* the generation becomes ``LATEST``.
Consequently W&B can be equal to, or behind, the durable trainer state; it is
never allowed to be ahead of it.

The module intentionally uses only ordinary W&B run resumption (stable ``id``
plus ``resume``). It does not use Rewind, local JSONL journals, grouped runs, or
W&B artifacts.
"""

from __future__ import annotations

import copy
import inspect
import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any


WANDB_STATE_SCHEMA_VERSION = 1
ENV_STEP_KEY = "env_step"
EVENT_INDEX_KEY = "resume/event_index"

DEFAULT_REMOTE_VERIFICATION_TIMEOUT_SECONDS = 120.0
DEFAULT_REMOTE_VERIFICATION_POLL_SECONDS = 2.0
DEFAULT_REMOTE_REQUEST_TIMEOUT_SECONDS = 20

_STATE_FIELDS = frozenset({"schema_version", "run_id", "events"})
_EVENT_FIELDS = frozenset({"event_index", "env_step", "payload"})
_RESERVED_PAYLOAD_KEYS = frozenset({ENV_STEP_KEY, EVENT_INDEX_KEY})
_REMOTE_INTERNAL_KEYS = frozenset({"_step", "_timestamp", "_runtime", "_wandb"})
_RUN_ID_RE = re.compile(r"[^/\\#?%:\s]+\Z")


class WandbResumeError(RuntimeError):
    """Base class for exact-resume W&B failures."""


class WandbResumeConfigurationError(WandbResumeError):
    """The requested W&B resume configuration is unsafe."""


class WandbCapabilityError(WandbResumeError):
    """The installed W&B SDK lacks a required public capability."""


class WandbStateError(WandbResumeError, ValueError):
    """Checkpointed W&B events are malformed or inconsistent."""


class WandbReconciliationError(WandbResumeError):
    """Remote history is not the expected checkpoint prefix."""


class WandbRemoteHistoryError(WandbReconciliationError):
    """Remote W&B history is malformed."""


class WandbRemoteAheadError(WandbReconciliationError):
    """Remote W&B contains events beyond the selected checkpoint."""


class WandbRemoteDivergenceError(WandbReconciliationError):
    """Remote W&B differs from the selected checkpoint."""


class WandbRemoteNetworkError(WandbReconciliationError):
    """Remote W&B history could not be fetched."""


class WandbRemoteVerificationTimeout(WandbReconciliationError):
    """Remote W&B remained a proper prefix after blocking finish."""


class WandbRemoteWriteError(WandbResumeError):
    """A committed event could not be queued for W&B upload."""


class WandbFinishError(WandbResumeError):
    """The W&B SDK could not finish its uploader cleanly."""


class WandbInitializationError(WandbResumeError):
    """The stable online W&B run could not be initialized."""


def validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise WandbResumeConfigurationError(
            "A stable W&B run ID must be non-empty and contain no whitespace or URI separators."
        )
    return run_id


def validate_wandb_resume_capabilities(wandb_module: object) -> None:
    """Validate the small public SDK surface used by checkpoint-first resume."""

    init = getattr(wandb_module, "init", None)
    define_metric = getattr(wandb_module, "define_metric", None)
    api = getattr(wandb_module, "Api", None)
    if not callable(init) or not callable(define_metric) or not callable(api):
        raise WandbCapabilityError(
            "Exact W&B resume requires callable init(), define_metric(), and Api()."
        )
    try:
        parameters = inspect.signature(init).parameters
    except (TypeError, ValueError) as exc:
        raise WandbCapabilityError("The W&B init() signature is not introspectable.") from exc
    missing = sorted({"id", "resume"}.difference(parameters))
    if missing:
        raise WandbCapabilityError(
            "The W&B init() signature lacks: " + ", ".join(missing) + "."
        )


def _nonnegative_int(value: object, *, name: str, remote: bool = False) -> int:
    if isinstance(value, bool):
        valid = False
    elif isinstance(value, Integral):
        valid = True
    elif remote and isinstance(value, Real):
        numeric = float(value)
        valid = math.isfinite(numeric) and numeric.is_integer()
    else:
        valid = False
    if not valid or int(value) < 0:
        error = WandbRemoteHistoryError if remote else WandbStateError
        raise error(f"{name} must be a non-negative integer.")
    return int(value)


def _canonical_value(value: object, *, location: str, remote: bool = False) -> object:
    error = WandbRemoteHistoryError if remote else WandbStateError
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return float(value)
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise error(f"Metric value at {location} must be finite.")
        return numeric
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise error(f"Metric mapping key at {location} must be a string.")
            result[key] = _canonical_value(
                child, location=f"{location}.{key}", remote=remote
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(child, location=f"{location}[{index}]", remote=remote)
            for index, child in enumerate(value)
        ]
    raise error(f"Metric value at {location} has unsupported type {type(value).__name__}.")


def _canonical_payload(payload: object, *, remote: bool = False) -> dict[str, object]:
    error = WandbRemoteHistoryError if remote else WandbStateError
    if not isinstance(payload, Mapping):
        raise error("A W&B metric payload must be a mapping.")
    result: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise error("W&B metric payload keys must be strings.")
        if key in _RESERVED_PAYLOAD_KEYS or key.startswith("_"):
            raise error(f"Metric key {key!r} is reserved by W&B resume.")
        result[key] = _canonical_value(value, location=key, remote=remote)
    return result


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class WandbMetricEvent:
    """One checkpointed metric event with explicit W&B and environment steps."""

    event_index: int
    env_step: int
    payload: Mapping[str, object]

    @classmethod
    def create(cls, *, event_index: object, env_step: object, payload: object):
        return cls(
            event_index=_nonnegative_int(event_index, name="event_index"),
            env_step=_nonnegative_int(env_step, name=ENV_STEP_KEY),
            payload=_freeze(_canonical_payload(payload)),
        )

    @classmethod
    def from_record(cls, record: object, *, expected_index: int, previous_env_step: int):
        if not isinstance(record, Mapping) or set(record) != _EVENT_FIELDS:
            raise WandbStateError("W&B event fields do not match the supported schema.")
        event = cls.create(
            event_index=record["event_index"],
            env_step=record["env_step"],
            payload=record["payload"],
        )
        if event.event_index != expected_index:
            raise WandbStateError("W&B event indices must be contiguous from zero.")
        if event.env_step < previous_env_step:
            raise WandbStateError("W&B event env_step values must be nondecreasing.")
        return event

    def to_record(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "env_step": self.env_step,
            "payload": _thaw(self.payload),
        }

    def wandb_payload(self) -> dict[str, object]:
        return {
            **_thaw(self.payload),
            ENV_STEP_KEY: self.env_step,
            EVENT_INDEX_KEY: self.event_index,
        }


class WandbEventBuffer:
    """In-memory metric history serialized as part of the trainer checkpoint."""

    def __init__(self, events: Iterable[WandbMetricEvent] = ()):
        self._events = list(events)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[WandbMetricEvent, ...]:
        return tuple(self._events)

    def append(self, payload: Mapping[str, object], *, env_step: int) -> WandbMetricEvent:
        event = WandbMetricEvent.create(
            event_index=len(self._events), env_step=env_step, payload=payload
        )
        if self._events and event.env_step < self._events[-1].env_step:
            raise WandbStateError("W&B event env_step values must be nondecreasing.")
        self._events.append(event)
        return event

    @classmethod
    def from_records(cls, records: object) -> "WandbEventBuffer":
        if not isinstance(records, (list, tuple)):
            raise WandbStateError("Checkpointed W&B events must be a sequence.")
        events: list[WandbMetricEvent] = []
        previous_env_step = 0
        for index, record in enumerate(records):
            event = WandbMetricEvent.from_record(
                record, expected_index=index, previous_env_step=previous_env_step
            )
            events.append(event)
            previous_env_step = event.env_step
        return cls(events)

    def records(self) -> list[dict[str, object]]:
        return [event.to_record() for event in self._events]


def _decode_checkpoint_state(
    state: object, *, expected_run_id: str | None = None
) -> tuple[str, WandbEventBuffer]:
    if not isinstance(state, Mapping) or set(state) != _STATE_FIELDS:
        raise WandbStateError("W&B checkpoint fields do not match the supported schema.")
    if type(state.get("schema_version")) is not int or state["schema_version"] != WANDB_STATE_SCHEMA_VERSION:
        raise WandbStateError(
            f"Unsupported W&B checkpoint schema version {state.get('schema_version')!r}."
        )
    run_id = validate_run_id(state["run_id"])
    if expected_run_id is not None and run_id != expected_run_id:
        raise WandbStateError(
            f"W&B checkpoint run ID {run_id!r} does not match {expected_run_id!r}."
        )
    return run_id, WandbEventBuffer.from_records(state["events"])


def _checkpoint_state(run_id: str, buffer: WandbEventBuffer) -> dict[str, object]:
    return {
        "schema_version": WANDB_STATE_SCHEMA_VERSION,
        "run_id": validate_run_id(run_id),
        "events": buffer.records(),
    }


def _remote_payload(row: Mapping[str, object]) -> dict[str, object]:
    payload = {
        key: value
        for key, value in row.items()
        if key not in _REMOTE_INTERNAL_KEYS and key not in _RESERVED_PAYLOAD_KEYS
    }
    return _canonical_payload(payload, remote=True)


def validate_remote_prefix(
    events: Iterable[WandbMetricEvent], remote_rows: Iterable[Mapping[str, object]]
) -> int:
    """Return remote length only when it is an exact prefix of ``events``."""

    expected = tuple(events)
    rows = tuple(remote_rows)
    if len(rows) > len(expected):
        raise WandbRemoteAheadError(
            f"Remote W&B has {len(rows)} events but the checkpoint has {len(expected)}. "
            "The selected generation cannot reuse this run ID."
        )
    for index, (event, row) in enumerate(zip(expected, rows)):
        if not isinstance(row, Mapping):
            raise WandbRemoteHistoryError(f"Remote W&B row {index} is not a mapping.")
        missing = [key for key in ("_step", EVENT_INDEX_KEY, ENV_STEP_KEY) if key not in row]
        if missing:
            raise WandbRemoteHistoryError(
                f"Remote W&B row {index} lacks: {', '.join(missing)}."
            )
        internal_step = _nonnegative_int(row["_step"], name="remote _step", remote=True)
        event_index = _nonnegative_int(
            row[EVENT_INDEX_KEY], name="remote event index", remote=True
        )
        env_step = _nonnegative_int(row[ENV_STEP_KEY], name="remote env_step", remote=True)
        if internal_step != index or event_index != index:
            raise WandbRemoteHistoryError(
                "Remote W&B event indices are not contiguous from zero."
            )
        if env_step != event.env_step or _remote_payload(row) != _thaw(event.payload):
            raise WandbRemoteDivergenceError(
                f"Remote W&B event {index} differs from the durable checkpoint."
            )
    return len(rows)


def fetch_remote_history(
    wandb_module: object,
    *,
    entity: str,
    project: str,
    run_id: str,
    api: object | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Fetch the full history of one stable W&B run through the public API."""

    validate_wandb_resume_capabilities(wandb_module)
    run_id = validate_run_id(run_id)
    if not isinstance(entity, str) or not entity:
        raise WandbResumeConfigurationError("W&B entity is required for resume.")
    if not isinstance(project, str) or not project:
        raise WandbResumeConfigurationError("W&B project is required for resume.")
    try:
        client = (
            wandb_module.Api(timeout=DEFAULT_REMOTE_REQUEST_TIMEOUT_SECONDS)
            if api is None
            else api
        )
        get_run = getattr(client, "run", None)
        if not callable(get_run):
            raise WandbCapabilityError("The W&B API client must provide run().")
        remote_run = get_run(f"{entity}/{project}/{run_id}")
        scan_history = getattr(remote_run, "scan_history", None)
        if not callable(scan_history):
            raise WandbCapabilityError("The remote W&B run must provide scan_history().")
        return tuple(scan_history(page_size=1000))
    except (WandbCapabilityError, WandbResumeConfigurationError):
        raise
    except Exception as exc:
        raise WandbRemoteNetworkError(
            f"Could not fetch W&B history for {entity}/{project}/{run_id}."
        ) from exc


def verify_remote_checkpoint(
    wandb_module: object,
    *,
    entity: str,
    project: str,
    checkpoint_state: Mapping[str, object],
    timeout_seconds: float = DEFAULT_REMOTE_VERIFICATION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_REMOTE_VERIFICATION_POLL_SECONDS,
    api: object | None = None,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> tuple[Mapping[str, object], ...]:
    """Poll after ``Run.finish()`` until the committed history is exact."""

    for name, value in (
        ("timeout_seconds", timeout_seconds),
        ("poll_interval_seconds", poll_interval_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) <= 0:
            raise WandbResumeConfigurationError(
                f"W&B verification {name} must be finite and positive."
            )
    timeout = float(timeout_seconds)
    interval = float(poll_interval_seconds)
    if interval > timeout:
        raise WandbResumeConfigurationError(
            "W&B verification poll interval cannot exceed its timeout."
        )

    run_id, buffer = _decode_checkpoint_state(checkpoint_state)
    deadline = monotonic() + timeout
    while True:
        rows = fetch_remote_history(
            wandb_module,
            entity=entity,
            project=project,
            run_id=run_id,
            api=api,
        )
        count = validate_remote_prefix(buffer.events, rows)
        if count == len(buffer):
            return rows
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise WandbRemoteVerificationTimeout(
                f"W&B exposed only {count} of {len(buffer)} committed events after finish()."
            )
        sleep(min(interval, remaining))


@dataclass
class WandbResumeContext:
    """Prepared stable identity, checkpointed events, and verified remote prefix.

    ``resume`` is only for the lineage's current ``LATEST`` generation. An
    explicitly selected older generation must use ``branch`` with a fresh run
    ID even when its remote history happens not to be ahead. The copied history
    is not uploaded until that branch checkpoint is explicitly published after
    becoming ``LATEST``.
    """

    run_id: str
    buffer: WandbEventBuffer
    remote_event_count: int
    committed_event_count: int
    new_run: bool
    directory: Path | None = None

    def __post_init__(self) -> None:
        self.run_id = validate_run_id(self.run_id)
        if not isinstance(self.buffer, WandbEventBuffer):
            raise WandbResumeConfigurationError("W&B context requires a WandbEventBuffer.")
        if not 0 <= self.remote_event_count <= self.committed_event_count <= len(self.buffer):
            raise WandbResumeConfigurationError("W&B context event counts are inconsistent.")
        if self.directory is not None:
            self.directory = Path(self.directory)

    @classmethod
    def new(cls, *, run_id: str, directory: str | Path | None = None):
        return cls(run_id, WandbEventBuffer(), 0, 0, True, directory)

    @classmethod
    def resume(
        cls,
        *,
        run_id: str,
        checkpoint_state: Mapping[str, object],
        remote_rows: Iterable[Mapping[str, object]],
        directory: str | Path | None = None,
    ):
        _, buffer = _decode_checkpoint_state(
            checkpoint_state, expected_run_id=validate_run_id(run_id)
        )
        remote_count = validate_remote_prefix(buffer.events, remote_rows)
        return cls(run_id, buffer, remote_count, len(buffer), False, directory)

    @classmethod
    def branch(
        cls,
        *,
        run_id: str,
        checkpoint_state: Mapping[str, object],
        directory: str | Path | None = None,
    ):
        """Prepare an explicit rollback branch with a fresh physical run ID."""

        _, buffer = _decode_checkpoint_state(checkpoint_state)
        return cls(run_id, buffer, 0, 0, True, directory)

    def checkpoint_state(self) -> dict[str, object]:
        return _checkpoint_state(self.run_id, self.buffer)

    def validate_checkpoint_prefix(self, state: Mapping[str, object]) -> int:
        _, candidate = _decode_checkpoint_state(state, expected_run_id=self.run_id)
        count = len(candidate)
        if count > len(self.buffer) or candidate.events != self.buffer.events[:count]:
            raise WandbStateError("W&B checkpoint is not a prefix of the current event buffer.")
        return count


class CheckpointedWandbRun:
    """W&B run wrapper that buffers first and publishes only committed events."""

    def __init__(
        self,
        run: object,
        context: WandbResumeContext,
        *,
        wandb_module: object,
        entity: str,
        project: str,
    ):
        self._run = run
        self._context = context
        self._wandb_module = wandb_module
        self._entity = entity
        self._project = project
        self._published_count = context.remote_event_count
        self._committed_count = context.committed_event_count
        self._remote_write_error: WandbRemoteWriteError | None = None

    @property
    def raw_run(self) -> object:
        return self._run

    @property
    def buffer(self) -> WandbEventBuffer:
        return self._context.buffer

    @property
    def published_event_count(self) -> int:
        return self._published_count

    @property
    def committed_event_count(self) -> int:
        return self._committed_count

    def checkpoint_state(self) -> dict[str, object]:
        return self._context.checkpoint_state()

    def log(self, payload: Mapping[str, object], *, env_step: int) -> None:
        if self._remote_write_error is not None:
            raise WandbRemoteWriteError(
                "W&B logging is disabled after an earlier committed upload failure."
            ) from self._remote_write_error
        payload = dict(payload)
        if ENV_STEP_KEY in payload:
            supplied = _nonnegative_int(payload.pop(ENV_STEP_KEY), name=ENV_STEP_KEY)
            requested = _nonnegative_int(env_step, name=ENV_STEP_KEY)
            if supplied != requested:
                raise WandbStateError(
                    f"Payload env_step={supplied} does not match log step {requested}."
                )
            env_step = requested
        self.buffer.append(payload, env_step=env_step)

    def _upload_committed_suffix(self) -> None:
        if self._remote_write_error is not None:
            raise WandbRemoteWriteError(
                "W&B upload is disabled after an earlier committed upload failure."
            ) from self._remote_write_error
        for event in self.buffer.events[self._published_count : self._committed_count]:
            try:
                self._run.log(event.wandb_payload(), step=event.event_index)
            except Exception as exc:
                error = WandbRemoteWriteError(
                    f"W&B upload failed for committed event {event.event_index}."
                )
                self._remote_write_error = error
                raise error from exc
            self._published_count += 1

    def publish_committed(self, checkpoint_state: Mapping[str, object]) -> None:
        """Upload a prefix only after its checkpoint has become durable LATEST."""

        count = self._context.validate_checkpoint_prefix(checkpoint_state)
        if count < self._committed_count:
            raise WandbStateError("A W&B committed prefix cannot move backwards.")
        if count < self._published_count:
            raise WandbRemoteAheadError("W&B is ahead of the prefix being committed.")
        self._committed_count = count
        self._upload_committed_suffix()

    def synchronize_resumed_checkpoint(self) -> None:
        """Replay the missing suffix of an already-durable restored checkpoint."""

        self._upload_committed_suffix()

    def finish(self, *, api: object | None = None) -> Any:
        if len(self.buffer) != self._committed_count:
            raise WandbStateError(
                "Cannot finish resumable W&B with uncommitted metric events; publish a fresh "
                "LATEST checkpoint first."
            )
        self._upload_committed_suffix()
        try:
            result = self._run.finish()
        except Exception as exc:
            raise WandbFinishError("W&B finish() failed before upload completion.") from exc
        verify_remote_checkpoint(
            self._wandb_module,
            entity=self._entity,
            project=self._project,
            checkpoint_state=_checkpoint_state(
                self._context.run_id,
                WandbEventBuffer(self.buffer.events[: self._committed_count]),
            ),
            api=api,
        )
        return result

    def abort(self) -> Any:
        """Close the SDK after a primary failure without claiming reconciliation."""

        try:
            return self._run.finish()
        except Exception as exc:
            raise WandbFinishError("W&B finish() also failed during abort.") from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run, name)
