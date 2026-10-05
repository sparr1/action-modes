"""Measure the complete saved-root diagnostic workload, with GPU synchronization.

This wrapper changes no solver settings. It requires every diagnostic family,
writes incremental phase timings, and refuses to certify incomplete evidence.
The scheduler's /usr/bin/time receipt additionally includes Python import time.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import torch

import evaluate_ambi_transfer_diagnostics as evaluator
from utils import transfer_diagnostics as diagnostics
from utils.transfer_diagnostic_coverage import verify_full_bundle


class PhaseTimer:
    """Nested inclusive/exclusive wall times; synchronize at measured boundaries."""

    def __init__(self, stream, synchronize=lambda: None, clock=time.perf_counter):
        self.stream, self.synchronize, self.clock = stream, synchronize, clock
        self.stack = []
        self.stats = defaultdict(lambda: dict(calls=0, seconds=0., exclusive_seconds=0.))

    @contextmanager
    def phase(self, name):
        self.synchronize()
        frame = dict(name=name, start=self.clock(), children=0.)
        self.stack.append(frame)
        self.emit(dict(event="start", phase=name, parent=self.stack[-2]["name"] if len(self.stack)>1 else None))
        status = "failed"
        try:
            yield
            status = "complete"
        finally:
            self.synchronize()
            elapsed = self.clock() - frame["start"]
            self.stack.pop()
            if self.stack:
                self.stack[-1]["children"] += elapsed
            stats = self.stats[name]
            stats["calls"] += 1
            stats["seconds"] += elapsed
            stats["exclusive_seconds"] += elapsed - frame["children"]
            self.emit(dict(event="end", phase=name, status=status, seconds=elapsed,
                           exclusive_seconds=elapsed-frame["children"]))

    def emit(self, event):
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.stream.flush()

    def wrap(self, function, name):
        @wraps(function)
        def measured(*args, **kwargs):
            label = name
            if name == "solve":
                parents = {frame["name"] for frame in self.stack}
                label = ("replanning_solve" if "replanning" in parents else
                         "root_fork_solve" if "root_audit" in parents else "source_solve")
            with self.phase(label):
                return function(*args, **kwargs)
        return measured


def instrument(timer):
    """Restore all temporary wrappers on success or failure."""
    stack = ExitStack()
    for module, name, label in (
        (evaluator, "_make_env", "environment_setup"),
        (evaluator, "_initialize_frozen_model", "model_setup"),
        (evaluator, "_outer_state_digest", "frozen_state_verification"),
        (evaluator, "audit_root", "root_audit"),
        (evaluator, "solve_fork", "solve"),
        (evaluator, "write_json", "json_output"),
        (diagnostics, "solve_fork", "solve"),
        (diagnostics, "audit_snapshot", "snapshot_measurements"),
        (diagnostics, "portability_audit", "portability"),
        (diagnostics, "stationary_audit", "stationary_fit"),
        (diagnostics, "audit_real", "real_prefix_tail"),
        (diagnostics, "audit_replanning", "replanning"),
    ):
        stack.enter_context(patch.object(module, name, timer.wrap(getattr(module, name), label)))
    return stack


def run(args):
    evaluator.validate_options(args)
    if (set(args.data_lanes) != {"common", "natural"} or args.no_target_cross
            or args.capture_rounds is not None or args.real_rollouts < 1
            or args.replan_steps < 2 or not args.save_snapshots or not args.full_trace_probes
            or args.dry_run):
        raise ValueError("Profiling requires both lanes, all rounds, target cross, real tails, "
                         "replanning, full trace probes and saved snapshots (not dry run).")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    hardware = dict(host=platform.node(), device=str(device), python=platform.python_version(),
                    torch=torch.__version__, torch_threads=torch.get_num_threads())
    if device.type == "cuda":
        hardware["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    result = dict(status="failed", hardware=hardware, scope="eager saved-root diagnostics",
                  options={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  timing_semantics="Synchronized wall seconds; nested totals overlap. "
                  "Exclusive seconds can be added. Includes measurement and output overhead; "
                  "process.time additionally includes imports and verification.")
    with (args.profile_dir / "progress.jsonl").open("x") as stream:
        timer = PhaseTimer(stream, synchronize)
        try:
            with timer.phase("evaluation"), instrument(timer):
                evaluator.run(args)
            with timer.phase("coverage_verification"):
                result["coverage"] = verify_full_bundle(args.output_dir)
            result["status"] = "complete"
        finally:
            result["phases"] = dict(timer.stats)
            if device.type == "cuda":
                result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            diagnostics.write_json(args.profile_dir / "timings.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = evaluator.parser()
    parser.description = __doc__
    parser.add_argument("--profile-dir", type=Path, required=True)
    run(parser.parse_args())
