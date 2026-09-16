"""Create, append, inspect, and publish checkpoint evaluation curves.

GPU workers only stage completed result pointers. Run the publisher on a CPU
allocation sharing the registry filesystem. Creation and continuation are
explicit operations; matching settings never implicitly select an attempt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import time

from utils.eval_series import Publisher, create_run, load_run, stage_record, validate_identity
from utils.eval_series_data import load_records


def select_records(path, selector=None, inventory=None, source_run=None):
    records = load_records(path, checkpoint_inventory=inventory, source_run=source_run)
    if selector is not None:
        records = [r for r in records if selector in (r.get("selector"), r.get("controller"))]
    if len(records) != 1:
        raise ValueError("Select exactly one controller with --selector; each controller has its own run")
    return records[0]


def scheduler_done(job_ids, expected_jobs=None):
    """Do not finish a watcher merely because accounting is temporarily absent."""
    if not job_ids or any(re.fullmatch(r"\d+(?:_\d+)?", job) is None for job in job_ids):
        raise ValueError("--jobs requires Slurm job IDs")
    query = ",".join(job_ids)
    # -r expands pending arrays so every observed task must later be accounted
    # for. JobIDRaw is a separate numeric allocation ID and cannot be matched to
    # the parent_array_task identifiers supplied by users.
    try:
        active = subprocess.check_output(["squeue", "-h", "-r", "-j", query, "-o", "%i"], text=True, stderr=subprocess.PIPE, timeout=30)
    except subprocess.CalledProcessError as exc:
        if "Invalid job id" not in (exc.stderr or ""):
            raise
        # Jobs may have left the controller while sacct retains their terminal
        # record. Other queue failures must not be mistaken for completion.
        active = exc.output or ""
    accounting = subprocess.check_output([
        "sacct", "-n", "-X", "--array", "-j", query, "-o", "JobID%128,State%64", "-P",
    ], text=True, timeout=30)
    rows = {}
    for line in accounting.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 2 and re.fullmatch(r"\d+(?:_\d+)?", fields[0]):
            rows[fields[0]] = fields[1].strip().split()[0].rstrip("+") if fields[1].strip() else ""
    expected = expected_jobs if expected_jobs is not None else set()
    expected.update(line.strip() for line in active.splitlines() if re.fullmatch(r"\d+(?:_\d+)?", line.strip()))
    expected.update(rows)
    expected.update(job for job in job_ids if "_" in job)
    if active.strip():
        return False
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
                "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    for job in job_ids:
        matching = [jid for jid in expected if jid == job or jid.startswith(job + "_")]
        if not matching or any(rows.get(jid) not in terminal for jid in matching):
            return False
    return True


def publish_watch(run_dir, *, owner=None, watch=False, jobs=(), poll_seconds=15,
                  max_wait_seconds=None, publisher_factory=Publisher):
    if watch and not jobs and max_wait_seconds is None:
        raise ValueError("--watch requires --jobs or --max-wait-seconds")
    if poll_seconds <= 0 or (max_wait_seconds is not None and max_wait_seconds <= 0):
        raise ValueError("Watch intervals must be positive")
    start = time.monotonic()
    expected_jobs = set()
    last = None
    with publisher_factory(run_dir, owner=owner) as publisher:
        while True:
            progress = publisher.publish_pending()
            if progress != last:
                print(json.dumps(progress, sort_keys=True), flush=True)
                last = progress
            if not watch:
                break
            if jobs and scheduler_done(jobs, expected_jobs):
                # Result pointers are atomic and written before the GPU job
                # exits. Drain once more after observing terminal tasks.
                last = publisher.publish_pending()
                break
            if max_wait_seconds is not None and time.monotonic() - start >= max_wait_seconds:
                raise TimeoutError("Publisher watch expired; saved results remain available for retry")
            delay = poll_seconds
            if max_wait_seconds is not None:
                delay = min(delay, max(0, max_wait_seconds - (time.monotonic() - start)))
            time.sleep(delay)
    publication = Path(run_dir) / "publication.json"
    if publication.exists():
        entries = json.loads(publication.read_text())["records"].values()
        last = {"run_id": load_run(run_dir)["run_id"], "accepted": len(entries),
                "published": sum(entry["status"] == "published" for entry in entries),
                "queued": sum(entry["status"] == "queued" for entry in entries)}
    return last


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Explicitly start a new evaluation attempt")
    create.add_argument("--root", type=Path, required=True)
    source = create.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", type=Path, help="Use one completed result as an identity template")
    source.add_argument("--spec", type=Path, help="JSON containing identity and label, prepared before GPU jobs")
    create.add_argument("--attempt-label", required=True)
    create.add_argument("--owner", required=True, help="Stable publication owner, e.g. oscar-rgao48")
    create.add_argument("--project", default="ambi-inner-bench")
    create.add_argument("--entity", default="rwgao_b-brown-university")
    append = sub.add_parser("append", help="Validate a prelaunch identity or append a completed checkpoint to a selected run")
    append.add_argument("run_dir", type=Path)
    append.add_argument("record", type=Path, nargs="?", help="Completed result to append")
    append.add_argument("--spec", type=Path, help="Validate an identity template before launching more checkpoints")
    for p in (create, append):
        p.add_argument("--selector")
        p.add_argument("--checkpoint-inventory", type=Path)
        p.add_argument("--source-run")
    inspect = sub.add_parser("inspect", help="Read an existing run registry")
    inspect.add_argument("run_dir", type=Path)
    publish = sub.add_parser("publish", help="Publish staged results without evaluating anything")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument("--owner")
    publish.add_argument("--watch", action="store_true")
    publish.add_argument("--jobs", nargs="*", default=[])
    publish.add_argument("--poll-seconds", type=float, default=15)
    publish.add_argument("--max-wait-seconds", type=float)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.command == "create":
        record = json.loads(args.spec.read_text()) if args.spec else select_records(
            args.record, args.selector, args.checkpoint_inventory, args.source_run)
        value = create_run(args.root, record, attempt_label=args.attempt_label,
                           project=args.project, entity=args.entity, owner=args.owner)
    elif args.command == "append":
        if (args.record is None) == (args.spec is None):
            argument_parser.error("append requires exactly one completed result or --spec identity template")
        if args.spec is not None:
            value = validate_identity(args.run_dir, json.loads(args.spec.read_text())["identity"])
        else:
            record = select_records(args.record, args.selector, args.checkpoint_inventory, args.source_run)
            value = stage_record(args.run_dir, record)
    elif args.command == "inspect":
        value = load_run(args.run_dir)
    else:
        # Slurm sends this before the CPU allocation expires. Unwind the context
        # so queued SDK rows are flushed and acknowledged while time remains.
        previous = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise KeyboardInterrupt("CPU publisher interrupted; retained results and journal can be resumed")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            value = publish_watch(args.run_dir, owner=args.owner, watch=args.watch,
                                  jobs=args.jobs, poll_seconds=args.poll_seconds,
                                  max_wait_seconds=args.max_wait_seconds)
        finally:
            signal.signal(signal.SIGTERM, previous)
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
