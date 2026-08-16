"""Episode-boundary orchestration for exact segmented TD-MPC2 training."""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import shutil
import signal
import socket
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from RL.tdmpc2_core.common.buffer import Buffer
from utils.checkpointing import CheckpointTracker, metadata_path
from utils.resume_lineage import GenerationFile, LineageStore
from utils.resume_runtime import (
    capture_environment_state,
    capture_global_rng_state,
    restore_environment_state,
    restore_global_rng_state,
    validate_environment_state,
    validate_global_rng_state,
)
from utils.wandb_resume import (
    WandbResumeContext,
    fetch_remote_history,
)
from utils.wandb_utils import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    abort_wandb,
    finish_wandb,
)


RESUME_COMPLETE = 0
RESUME_HANDOFF = 75
TRAINER_ENVELOPE_SCHEMA_VERSION = 2
SEGMENT_SCHEMA_VERSION = 1
HANDOFF_SCHEMA_VERSION = 1
DEFAULT_REPLAY_SHARD_ROWS = 16_384


class ResumeTrainingError(RuntimeError):
    pass


class ResumeStateCorruptionError(ResumeTrainingError):
    pass


class ResumeIncompatibilityError(ResumeTrainingError):
    pass


class ResumeStorageError(ResumeTrainingError):
    pass


def _fsync_directory(path: Path) -> None:
    """Persist directory entries or fail the resume transaction."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ResumeStorageError(
            f"Could not durably sync directory {path}: {exc}"
        ) from exc


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = -1
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError) as exc:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                note = getattr(exc, "add_note", None)
                if callable(note):
                    note(f"Additionally could not remove {temporary}: {cleanup_error}")
        raise ResumeStorageError(f"Could not durably write {path}: {exc}") from exc


def _torch_writer(payload: object):
    def writer(path: Path) -> None:
        try:
            with path.open("xb") as stream:
                torch.save(payload, stream)
        except (OSError, RuntimeError) as exc:
            raise ResumeStorageError(f"Could not serialize {path}: {exc}") from exc

    return writer


def _copy_writer(source: Path):
    def writer(path: Path) -> None:
        try:
            with source.open("rb") as incoming, path.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        except OSError as exc:
            raise ResumeStorageError(f"Could not copy {source}: {exc}") from exc

    return writer


def _json_writer(payload: Mapping[str, Any]):
    frozen = copy.deepcopy(dict(payload))

    def writer(path: Path) -> None:
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(frozen, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        except (OSError, TypeError, ValueError) as exc:
            raise ResumeStorageError(f"Could not serialize {path}: {exc}") from exc

    return writer


def _load_torch(path: Path, label: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
    ) as exc:
        raise ResumeStateCorruptionError(f"Could not decode {label} {path}: {exc}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeStateCorruptionError(f"Could not decode {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumeStateCorruptionError(f"{label} must contain a JSON object.")
    return value


def _envelope(value: object) -> Mapping[str, Any]:
    fields = {
        "schema_version",
        "generation_id",
        "trainer",
        "environment",
        "global_rng",
        "wandb",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ResumeStateCorruptionError("Trainer checkpoint fields are invalid.")
    if value["schema_version"] != TRAINER_ENVELOPE_SCHEMA_VERSION:
        raise ResumeStateCorruptionError("Unsupported trainer checkpoint schema.")
    return value


class TrainingResumeSession:
    """Own one lineage lease, one allocation segment, and checkpoint commits."""

    def __init__(
        self,
        *,
        store: LineageStore,
        mode: str,
        selected_generation,
        source_generation_id: str | None,
        segment_id: str,
        checkpoint_minutes: float,
        drain_after_seconds: float | None,
        segment_started_monotonic: float,
        replay_shard_rows: int = DEFAULT_REPLAY_SHARD_ROWS,
    ) -> None:
        self.store = store
        self.mode = mode
        self.selected_generation = selected_generation
        self._branch_source_generation_id = source_generation_id
        self.segment_id = str(segment_id)
        if not self.segment_id or any(item in self.segment_id for item in "/\\\0"):
            raise ResumeIncompatibilityError("Invalid segment ID.")
        self.segment_dir = store.root / "segments" / self.segment_id
        self.segment_log_dir = self.segment_dir / "logs"
        self.eval_csv_path = self.segment_dir / "evaluation.csv"
        self.checkpoint_interval_seconds = float(checkpoint_minutes) * 60.0
        self.drain_after_seconds = (
            None if drain_after_seconds is None else float(drain_after_seconds)
        )
        if not math.isfinite(self.checkpoint_interval_seconds) or self.checkpoint_interval_seconds <= 0:
            raise ResumeIncompatibilityError("Checkpoint cadence must be positive and finite.")
        if self.drain_after_seconds is not None and (
            not math.isfinite(self.drain_after_seconds) or self.drain_after_seconds <= 0
        ):
            raise ResumeIncompatibilityError("Drain deadline must be positive and finite.")
        self.replay_shard_rows = int(replay_shard_rows)
        if self.replay_shard_rows <= 0:
            raise ResumeIncompatibilityError("Replay shard rows must be positive.")
        self._segment_started = float(segment_started_monotonic)
        if time.monotonic() < self._segment_started:
            raise ResumeIncompatibilityError("The monotonic clock moved backwards.")
        self._last_checkpoint = time.monotonic()
        self._drain_requested = False
        self._signal_handlers: dict[int, Any] = {}
        self._last_generation = selected_generation
        self._segment_start_step = (
            0 if selected_generation is None else int(selected_generation.metadata["global_step"])
        )
        try:
            self.segment_dir.mkdir(parents=True, exist_ok=False)
            self.segment_log_dir.mkdir()
        except OSError as exc:
            raise ResumeStorageError(f"Could not create segment directory: {exc}") from exc
        _atomic_json(
            self.segment_dir / "SEGMENT.json",
            {
                "schema_version": SEGMENT_SCHEMA_VERSION,
                "segment_id": self.segment_id,
                "mode": self.mode,
                "source_generation": (
                    None if selected_generation is None else selected_generation.generation_id
                ),
                "start_global_step": self._segment_start_step,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
                "host": socket.gethostname(),
            },
        )
        # SEGMENT.json's atomic write syncs ``segment_dir``, thereby
        # persisting both the manifest and the new logs/ entry. Walk outward
        # only after the child contents are durable so first-segment creation
        # also persists segments/ in the lineage root.
        _fsync_directory(self.segment_dir.parent)
        _fsync_directory(self.store.root)
        self.install_signal_handlers()

    @classmethod
    def open(
        cls,
        lineage_dir: str | os.PathLike[str],
        *,
        mode: str,
        scientific_identity: Mapping[str, Any],
        total_steps: int,
        checkpoint_minutes: float,
        drain_after_seconds: float | None,
        resume_generation: str | None = None,
        segment_id: str | None = None,
    ) -> "TrainingResumeSession":
        started = time.monotonic()
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0:
            raise ResumeIncompatibilityError("Total steps must be a non-negative integer.")
        mode = str(mode)
        if mode == "new":
            if resume_generation is not None:
                raise ResumeIncompatibilityError("A new lineage cannot select a generation.")
            metadata = {
                "schema_version": 1,
                "training_id": uuid.uuid4().hex,
                # Rollback branches get a fresh ID in their trainer envelope;
                # this is only the original trajectory's genesis ID.
                "initial_wandb_run_id": uuid.uuid4().hex,
                "scientific_identity": copy.deepcopy(dict(scientific_identity)),
                "total_steps": total_steps,
            }
            store = LineageStore.open(
                lineage_dir, mode="new", lineage_metadata=metadata
            )
            selected = None
            source = None
        elif mode == "required":
            store = LineageStore.open(lineage_dir, mode="required")
            try:
                metadata = store.lineage_metadata
                if (
                    metadata.get("schema_version") != 1
                    or metadata.get("scientific_identity") != dict(scientific_identity)
                    or metadata.get("total_steps") != total_steps
                ):
                    raise ResumeIncompatibilityError(
                        "Code, configuration, environment, or target differs from the lineage."
                    )
                if not isinstance(metadata.get("initial_wandb_run_id"), str):
                    raise ResumeStateCorruptionError("Lineage has no initial W&B run ID.")
                latest = store.load()
                if resume_generation is None:
                    selected = latest
                    source = None
                else:
                    if resume_generation == latest.generation_id:
                        raise ResumeIncompatibilityError(
                            "--resume-generation must select the retained predecessor, "
                            "not the current LATEST generation. Omit the option for an "
                            "ordinary one-run continuation."
                        )
                    selected = store.load(resume_generation)
                    source = selected.generation_id
            except BaseException:
                store.close()
                raise
        else:
            raise ResumeIncompatibilityError("Resume mode must be 'new' or 'required'.")

        resolved_segment = segment_id or os.environ.get("AMBI_SEGMENT_ID")
        if not resolved_segment:
            restart = os.environ.get("SLURM_RESTART_COUNT", "0")
            resolved_segment = f"segment-{restart}-{uuid.uuid4().hex[:12]}"
        try:
            return cls(
                store=store,
                mode=mode,
                selected_generation=selected,
                source_generation_id=source,
                segment_id=resolved_segment,
                checkpoint_minutes=checkpoint_minutes,
                drain_after_seconds=drain_after_seconds,
                segment_started_monotonic=started,
            )
        except BaseException:
            store.close()
            raise

    @property
    def lineage_metadata(self) -> Mapping[str, Any]:
        return self.store.lineage_metadata

    @property
    def last_generation(self):
        if self._last_generation is None:
            raise ResumeTrainingError("No durable generation exists yet.")
        return self._last_generation

    def install_signal_handlers(self) -> None:
        if self._signal_handlers:
            return

        def request_drain(_signum, _frame):
            self._drain_requested = True

        for signum in (signal.SIGUSR1, signal.SIGTERM):
            self._signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_drain)

    def restore_signal_handlers(self) -> None:
        for signum, handler in self._signal_handlers.items():
            signal.signal(signum, handler)
        self._signal_handlers.clear()

    def checkpoint_due(self) -> bool:
        return time.monotonic() - self._last_checkpoint >= self.checkpoint_interval_seconds

    def drain_requested(self) -> bool:
        if self.drain_after_seconds is not None and (
            time.monotonic() - self._segment_started >= self.drain_after_seconds
        ):
            self._drain_requested = True
        return self._drain_requested

    @staticmethod
    def _wandb_settings(model) -> tuple[str, str]:
        params = model.custom_params or {}
        if params.get("wandb") is not True or params.get("wandb_mode") != "online":
            raise ResumeIncompatibilityError(
                "Exact resume requires wandb=true and wandb_mode='online'."
            )
        return (
            params.get("wandb_entity", DEFAULT_WANDB_ENTITY),
            params.get("wandb_project", DEFAULT_WANDB_PROJECT),
        )

    def _bind_outputs(self, model) -> None:
        if getattr(model, "_eval_csv_initialized", False):
            raise ResumeIncompatibilityError("Evaluation logging started before resume setup.")
        model._eval_csv_path = str(self.eval_csv_path)
        logger = getattr(model, "alg_logger", None)
        if logger is not None:
            if not callable(getattr(logger, "flush_durable", None)):
                raise ResumeIncompatibilityError("Resume logger lacks flush_durable().")
            if Path(logger.log_dir).absolute() != self.segment_log_dir.absolute():
                raise ResumeIncompatibilityError("Logger is not bound to this segment.")

    def _flush_outputs(self, model) -> None:
        logger = getattr(model, "alg_logger", None)
        if logger is not None:
            try:
                logger.flush_durable()
            except (OSError, RuntimeError) as exc:
                raise ResumeStorageError(f"Could not flush training logs: {exc}") from exc
        if self.eval_csv_path.exists():
            try:
                fd = os.open(self.eval_csv_path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                _fsync_directory(self.eval_csv_path.parent)
            except OSError as exc:
                raise ResumeStorageError(f"Could not flush evaluation log: {exc}") from exc

    def prepare_learner(self, model) -> None:
        """Prepare new state or restore a validated generation; RNG is last."""

        self._bind_outputs(model)
        entity, project = self._wandb_settings(model)
        wandb_directory = self.segment_dir / "wandb"
        wandb_directory.mkdir()
        if self.mode == "new":
            context = WandbResumeContext.new(
                run_id=self.lineage_metadata["initial_wandb_run_id"],
                directory=wandb_directory,
            )
            model._wandb_run = model._init_wandb(resume_context=context)
            model._reset_wandb_window()
            self.publish(model, reason="genesis")
            return

        generation = self.selected_generation
        if generation is None:
            raise ResumeStateCorruptionError("Required mode selected no generation.")
        trainer_files = generation.files_for_role("trainer")
        if len(trainer_files) != 1:
            raise ResumeStateCorruptionError("Generation needs exactly one trainer file.")
        envelope = _envelope(_load_torch(trainer_files[0], "trainer"))
        if envelope["generation_id"] != generation.generation_id:
            raise ResumeStateCorruptionError("Trainer and manifest generation IDs differ.")
        trainer = envelope["trainer"]
        if not isinstance(trainer, Mapping):
            raise ResumeStateCorruptionError("Trainer state must be a mapping.")
        for key, value in (
            ("global_step", trainer.get("global_step")),
            ("completed_episodes", trainer.get("completed_episodes")),
            ("phase", trainer.get("phase")),
        ):
            if generation.metadata.get(key) != value:
                raise ResumeStateCorruptionError(f"Manifest {key} differs from trainer state.")

        replay_files = generation.files_for_role("replay")
        metadata_files = [item for item in replay_files if item.name == "metadata.pt"]
        shard_files = sorted(item for item in replay_files if item.name.startswith("shard-"))
        if len(metadata_files) != 1 or len(replay_files) != len(shard_files) + 1:
            raise ResumeStateCorruptionError("Replay generation inventory is incomplete.")
        candidate_buffer = Buffer(model.cfg, resumable=True)
        replay_metadata = _load_torch(metadata_files[0], "replay metadata")
        try:
            candidate_buffer.load_training_state_shards(
                replay_metadata,
                (_load_torch(path, "replay shard") for path in shard_files),
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ResumeStateCorruptionError(f"Replay state is incompatible: {exc}") from exc

        step = int(trainer["global_step"])
        episodes = int(trainer["completed_episodes"])
        horizon = int(model.cfg.episode_length)
        if (
            candidate_buffer.num_eps != episodes
            or candidate_buffer.total_transitions != step
            or step != episodes * horizon
        ):
            raise ResumeStateCorruptionError(
                "Trainer progress, replay counters, and fixed episode horizon disagree."
            )
        environment_state = envelope["environment"]
        if trainer["phase"] == "before_initial_seeded_reset":
            if environment_state is not None:
                raise ResumeStateCorruptionError("Genesis contains environment state.")
        else:
            validate_environment_state(model.env, environment_state)
        validate_global_rng_state(envelope["global_rng"])

        # This learner is fresh and disposable. Strict component loaders are
        # the compatibility preflight; no training can run on a partial load.
        try:
            model.load_training_state_dict(trainer)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ResumeStateCorruptionError(f"Trainer state is incompatible: {exc}") from exc
        model.buffer = candidate_buffer
        if environment_state is not None:
            restore_environment_state(model.env, environment_state)

        saved_wandb = envelope["wandb"]
        if self._branch_source_generation_id is not None:
            context = WandbResumeContext.branch(
                run_id=uuid.uuid4().hex,
                checkpoint_state=saved_wandb,
                directory=wandb_directory,
            )
        else:
            try:
                import wandb
            except ImportError as exc:
                raise ResumeIncompatibilityError("W&B is required for exact resume.") from exc
            run_id = saved_wandb.get("run_id") if isinstance(saved_wandb, Mapping) else None
            rows = fetch_remote_history(
                wandb, entity=entity, project=project, run_id=run_id
            )
            context = WandbResumeContext.resume(
                run_id=run_id,
                checkpoint_state=saved_wandb,
                remote_rows=rows,
                directory=wandb_directory,
            )
        model._wandb_run = model._init_wandb(resume_context=context)
        restore_global_rng_state(envelope["global_rng"])
        self._last_checkpoint = time.monotonic()

        commit_timing = getattr(model, "_commit_resume_timing_checkpoint", None)
        if callable(commit_timing):
            commit_timing(trainer["wandb"])
        if self._branch_source_generation_id is not None:
            self.publish(model, reason="operator-rollback-branch")

    def _model_files(self, model, generation_id: str) -> list[GenerationFile]:
        tracker = getattr(model, "_checkpointing", None)
        if not isinstance(tracker, CheckpointTracker) or not tracker.enabled:
            return []
        prefix = tracker.name_prefix
        if not prefix or Path(prefix).name != prefix:
            raise ResumeIncompatibilityError("Invalid model checkpoint prefix.")
        lineage = {"schema_version": 1, "generation_id": generation_id}
        files = []
        if "latest" in tracker.strategies:
            relative = f"models/{prefix}_latest"
            target = tracker.explicit_target(
                relative,
                step=int(model._global_step),
                episode=int(model._episode_idx),
                kind="latest",
            )
            files.extend(
                [
                    GenerationFile(relative, "model_latest", _torch_writer(model.agent.checkpoint_state())),
                    GenerationFile(
                        f"{relative}.metadata.json",
                        "model_latest_metadata",
                        _json_writer({**target.metadata, "resume_lineage": lineage}),
                    ),
                ]
            )
        if "best" not in tracker.strategies:
            return files

        relative = f"models/{prefix}_best"
        local = tracker.save_path / f"{prefix}_best"
        local_metadata = metadata_path(local)
        source = metadata = None
        if local.is_file() and local_metadata.is_file():
            source = local
            metadata = _load_json(local_metadata, "best model metadata")
        elif self._last_generation is not None:
            parents = self._last_generation.files_for_role("model_best")
            parent_metadata = self._last_generation.files_for_role("model_best_metadata")
            if len(parents) == len(parent_metadata) == 1:
                source = parents[0]
                metadata = _load_json(parent_metadata[0], "best model metadata")
        if source is not None:
            files.extend(
                [
                    GenerationFile(relative, "model_best", _copy_writer(source)),
                    GenerationFile(
                        f"{relative}.metadata.json",
                        "model_best_metadata",
                        _json_writer({**metadata, "resume_lineage": lineage}),
                    ),
                ]
            )
        return files

    def publish(self, model, *, reason: str):
        """Synchronously commit scientific state, then release metrics to W&B."""

        force_metrics = getattr(model, "_force_resume_metric_boundary", None)
        if not callable(force_metrics):
            raise ResumeIncompatibilityError(
                "Exact-resume learner lacks a metric-boundary hook."
            )
        force_metrics()
        self._flush_outputs(model)
        model.flush_checkpoints()
        generation_id = (
            f"step-{int(model._global_step):012d}-"
            f"episode-{int(model._episode_idx):09d}-{uuid.uuid4().hex[:12]}"
        )
        trainer = model.training_state_dict()
        phase = trainer["phase"]
        environment = (
            None
            if phase == "before_initial_seeded_reset"
            else capture_environment_state(model.env)
        )
        run = model._wandb_run
        if run is None or not callable(getattr(run, "checkpoint_state", None)):
            raise ResumeIncompatibilityError("Learner lacks checkpoint-first W&B state.")
        wandb_state = run.checkpoint_state()
        envelope = {
            "schema_version": TRAINER_ENVELOPE_SCHEMA_VERSION,
            "generation_id": generation_id,
            "trainer": trainer,
            "environment": environment,
            "global_rng": capture_global_rng_state(),
            "wandb": wandb_state,
        }
        replay_metadata = model.buffer.training_state_metadata()
        rows = int(replay_metadata["storage_rows"])
        shard_count = math.ceil(rows / self.replay_shard_rows) if rows else 0
        files = [
            GenerationFile("trainer.pt", "trainer", _torch_writer(envelope)),
            GenerationFile("replay/metadata.pt", "replay", _torch_writer(replay_metadata)),
        ]
        files.extend(self._model_files(model, generation_id))
        for index in range(shard_count):

            def shard_writer(path: Path, shard_index=index) -> None:
                payload = model.buffer.training_state_shard(
                    shard_index, max_rows=self.replay_shard_rows
                )
                _torch_writer(payload)(path)

            files.append(
                GenerationFile(
                    f"replay/shard-{index:06d}.pt", "replay", shard_writer
                )
            )

        parent = self._last_generation
        source = self._branch_source_generation_id
        published = self.store.publish(
            generation_id,
            files=files,
            metadata={
                "global_step": int(model._global_step),
                "completed_episodes": int(model._episode_idx),
                "total_steps": int(model.cfg.steps),
                "segment_id": self.segment_id,
                "segment_start_step": self._segment_start_step,
                "reason": str(reason),
                "phase": phase,
                "replay_rows": rows,
                "wandb_event_count": len(wandb_state["events"]),
                "host": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            source_generation_id=source,
        )
        expected_parent = None if parent is None else parent.generation_id
        if published.parent_generation != expected_parent:
            raise ResumeStateCorruptionError("Published generation parent changed.")
        self._last_generation = published
        self._branch_source_generation_id = None

        # This is the central telemetry invariant: no event reaches W&B until
        # the checkpoint containing it is the durable LATEST generation.
        run.publish_committed(wandb_state)
        commit_timing = getattr(model, "_commit_resume_timing_checkpoint", None)
        if callable(commit_timing):
            commit_timing(trainer["wandb"])
        self._last_checkpoint = time.monotonic()
        return published

    def _finish_wandb(self, model) -> None:
        if model._wandb_run is not None:
            finish_wandb(model._wandb_run)
            model._wandb_run = None

    def abort_wandb(self, model, primary_error: BaseException) -> None:
        if model._wandb_run is None:
            return
        try:
            abort_wandb(model._wandb_run)
        except BaseException as cleanup_error:
            note = getattr(primary_error, "add_note", None)
            if callable(note):
                note(f"Additional W&B cleanup failure: {cleanup_error}")
        finally:
            model._wandb_run = None

    def clean_handoff(self, model, generation) -> int:
        self._finish_wandb(model)
        _atomic_json(
            self.store.root / "HANDOFF.json",
            {
                "schema_version": HANDOFF_SCHEMA_VERSION,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "segment_id": self.segment_id,
                "generation_id": generation.generation_id,
            },
        )
        return RESUME_HANDOFF

    def complete(self, model, generation) -> int:
        self._finish_wandb(model)
        _atomic_json(
            self.store.root / "DONE",
            {
                "schema_version": 1,
                "generation_id": generation.generation_id,
                "global_step": int(model._global_step),
                "target_step": int(model.cfg.steps),
                "segment_id": self.segment_id,
            },
        )
        return RESUME_COMPLETE

    def close(self) -> None:
        self.restore_signal_handlers()
        self.store.close()


__all__ = [
    "DEFAULT_REPLAY_SHARD_ROWS",
    "HANDOFF_SCHEMA_VERSION",
    "RESUME_COMPLETE",
    "RESUME_HANDOFF",
    "ResumeIncompatibilityError",
    "ResumeStateCorruptionError",
    "ResumeStorageError",
    "ResumeTrainingError",
    "TrainingResumeSession",
]
