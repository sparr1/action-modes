"""Small fail-closed checks used by the Oscar Slurm launchers.

Checkpoint payload validation belongs to the training process. The launcher
only proves storage preconditions and the scheduler control records needed to
decide whether to stop or requeue.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


class OscarLauncherError(RuntimeError):
    """A launcher check failed and the job must not requeue."""


class QuotaPreflightError(OscarLauncherError):
    pass


class StoragePreflightError(OscarLauncherError):
    pass


class HandoffVerificationError(OscarLauncherError):
    pass


class DoneVerificationError(OscarLauncherError):
    pass


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _generation_id(value: Any) -> str:
    value = _nonempty_string(value, "generation_id")
    if len(value) > 128 or any(character in value for character in "/\\\0"):
        raise ValueError("generation_id is not a safe directory name")
    return value


def _latest(lineage_dir: Path) -> str:
    try:
        lines = (lineage_dir / "LATEST").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read LATEST in {lineage_dir}: {exc}") from exc
    if len(lines) != 1 or not lines[0]:
        raise ValueError("LATEST must name exactly one generation")
    return _generation_id(lines[0])


def _generation_metadata(lineage_dir: Path, generation: str) -> dict[str, Any]:
    document = _json_object(
        lineage_dir / "generations" / generation / "manifest.json",
        "generation manifest",
    )
    if document.get("generation_id") != generation:
        raise ValueError("generation manifest does not match its directory")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("generation manifest has no metadata object")
    return metadata


def verify_quota_output(
    output: str, allocation: str, filesystem_path: str | None = None
) -> str:
    """Require exactly one selected allocation row ending in exactly ``OK``.

    Some Oscar allocations have separate rows for home and scratch under the
    same allocation name. ``filesystem_path`` selects one such row by an exact
    whitespace-delimited field match; omitting it preserves the stricter
    single-row behavior.
    """

    if not allocation or any(character.isspace() for character in allocation):
        raise QuotaPreflightError("quota allocation must be one non-empty field")
    if filesystem_path is not None and (
        not filesystem_path
        or any(character.isspace() for character in filesystem_path)
    ):
        raise QuotaPreflightError("quota filesystem path must be one non-empty field")
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if allocation not in fields:
            continue
        if filesystem_path is not None and filesystem_path not in fields:
            continue
        rows.append(line.strip())
    if len(rows) != 1:
        selection = (
            f" and filesystem path {filesystem_path!r}"
            if filesystem_path is not None
            else ""
        )
        raise QuotaPreflightError(
            f"expected exactly one checkquota row for {allocation!r}{selection}; "
            f"found {len(rows)}"
        )
    fields = rows[0].split()
    if not fields or fields[-1] != "OK":
        raise QuotaPreflightError(
            f"durable allocation {allocation!r} is not exactly OK: {rows[0]}"
        )
    return rows[0]


def verify_durable_storage(
    *,
    durable_root: str | os.PathLike[str],
    lineage_dir: str | os.PathLike[str],
    metrics: dict[str, float | int] | None = None,
) -> tuple[Path, Path]:
    """Require a lineage below one operator-selected root and fsync-probe it."""

    configured_root = Path(durable_root)
    configured_lineage = Path(lineage_dir)
    if not configured_root.is_absolute() or not configured_lineage.is_absolute():
        raise StoragePreflightError("durable root and lineage directory must be absolute")
    try:
        root = configured_root.resolve(strict=True)
        lineage = configured_lineage.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise StoragePreflightError(f"cannot resolve durable storage paths: {exc}") from exc
    if not root.is_dir() or root.is_symlink():
        raise StoragePreflightError(f"durable root is not a real directory: {root}")
    try:
        relative = lineage.relative_to(root)
    except ValueError as exc:
        raise StoragePreflightError(
            f"lineage directory {lineage} is outside durable root {root}"
        ) from exc
    if not relative.parts:
        raise StoragePreflightError("lineage directory must be below the durable root")

    probe_dir = lineage
    while not probe_dir.exists():
        probe_dir = probe_dir.parent
    if not probe_dir.is_dir() or probe_dir.is_symlink():
        raise StoragePreflightError(
            f"nearest existing lineage ancestor is not a real directory: {probe_dir}"
        )
    try:
        if probe_dir.stat().st_dev != root.stat().st_dev:
            raise StoragePreflightError(
                "lineage ancestor is on a different filesystem from the durable root"
            )
        started = time.perf_counter()
        descriptor, raw_probe = tempfile.mkstemp(prefix=".ambi-fsync-probe-", dir=probe_dir)
        probe = Path(raw_probe)
        directory_descriptor: int | None = None
        try:
            file_started = time.perf_counter()
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(b"AMBI durable storage probe\n")
                stream.flush()
                os.fsync(stream.fileno())
            file_seconds = time.perf_counter() - file_started
            directory_descriptor = os.open(
                probe_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            directory_started = time.perf_counter()
            os.fsync(directory_descriptor)
            probe.unlink()
            os.fsync(directory_descriptor)
            directory_seconds = time.perf_counter() - directory_started
        except BaseException:
            try:
                probe.unlink()
            except FileNotFoundError:
                pass
            raise
        finally:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
    except StoragePreflightError:
        raise
    except OSError as exc:
        raise StoragePreflightError(
            f"durable lineage filesystem failed its write/fsync probe: {exc}"
        ) from exc

    if metrics is not None:
        metrics.clear()
        metrics.update(
            {
                "file_fsync_seconds": float(file_seconds),
                "directory_fsync_seconds": float(directory_seconds),
                "total_seconds": float(time.perf_counter() - started),
            }
        )
    return root, lineage


def verify_handoff(
    *, lineage_dir: str | os.PathLike[str], slurm_job_id: str, segment_id: str
) -> str:
    """Verify only the facts that authorize one scheduler requeue."""

    root = Path(lineage_dir)
    try:
        record = _json_object(root / "HANDOFF.json", "HANDOFF.json")
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("unsupported HANDOFF.json schema")
        if _nonempty_string(record.get("slurm_job_id"), "slurm_job_id") != str(
            slurm_job_id
        ):
            raise ValueError("HANDOFF.json names a stale Slurm job")
        if _nonempty_string(record.get("segment_id"), "segment_id") != str(segment_id):
            raise ValueError("HANDOFF.json names a stale segment")
        generation = _generation_id(record.get("generation_id"))
        if generation != _latest(root):
            raise ValueError("HANDOFF.json generation does not match LATEST")
        metadata = _generation_metadata(root, generation)
        if metadata.get("segment_id") != str(segment_id):
            raise ValueError("LATEST was not published by the handing-off segment")
    except ValueError as exc:
        raise HandoffVerificationError(str(exc)) from exc
    return generation


def verify_done(*, lineage_dir: str | os.PathLike[str]) -> str:
    """Verify the terminal marker, LATEST pointer, and immutable target."""

    root = Path(lineage_dir)
    try:
        record = _json_object(root / "DONE", "DONE")
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("unsupported DONE schema")
        _nonempty_string(record.get("segment_id"), "segment_id")
        generation = _generation_id(record.get("generation_id"))
        if generation != _latest(root):
            raise ValueError("DONE generation does not match LATEST")
        global_step = _nonnegative_integer(record.get("global_step"), "global_step")
        generation_metadata = _generation_metadata(root, generation)
        if generation_metadata.get("segment_id") != record["segment_id"]:
            raise ValueError("DONE segment did not publish LATEST")
        if generation_metadata.get("global_step") != global_step:
            raise ValueError("DONE step does not match the LATEST generation")
        lineage = _json_object(root / "LINEAGE.json", "LINEAGE.json")
        if type(lineage.get("schema_version")) is not int or lineage["schema_version"] != 1:
            raise ValueError("unsupported LINEAGE.json schema")
        metadata = lineage.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("LINEAGE.json has no metadata object")
        target = _nonnegative_integer(metadata.get("total_steps"), "total_steps")
        if global_step != target:
            raise ValueError(
                f"DONE global_step {global_step} does not equal lineage target {target}"
            )
        if "target_step" in record and _nonnegative_integer(
            record["target_step"], "target_step"
        ) != target:
            raise ValueError("DONE target_step does not equal lineage target")
    except ValueError as exc:
        raise DoneVerificationError(str(exc)) from exc
    return generation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    quota = commands.add_parser("quota")
    quota.add_argument("--allocation", required=True)
    quota.add_argument("--filesystem-path")
    storage = commands.add_parser("storage")
    storage.add_argument("--durable-root", required=True)
    storage.add_argument("--lineage-dir", required=True)
    storage.add_argument("--print-metrics", action="store_true")
    handoff = commands.add_parser("handoff")
    handoff.add_argument("--lineage-dir", required=True)
    handoff.add_argument("--slurm-job-id", required=True)
    handoff.add_argument("--segment-id", required=True)
    done = commands.add_parser("done")
    done.add_argument("--lineage-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "quota":
            verify_quota_output(
                sys.stdin.read(), args.allocation, args.filesystem_path
            )
        elif args.command == "storage":
            storage_metrics = {} if args.print_metrics else None
            verify_durable_storage(
                durable_root=args.durable_root,
                lineage_dir=args.lineage_dir,
                metrics=storage_metrics,
            )
            if storage_metrics is not None:
                print(json.dumps(storage_metrics, sort_keys=True, allow_nan=False))
        elif args.command == "handoff":
            verify_handoff(
                lineage_dir=args.lineage_dir,
                slurm_job_id=args.slurm_job_id,
                segment_id=args.segment_id,
            )
        else:
            verify_done(lineage_dir=args.lineage_dir)
    except OscarLauncherError as exc:
        print(f"Oscar resume preflight failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
