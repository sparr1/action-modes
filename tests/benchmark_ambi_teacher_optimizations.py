"""Standalone ablations of identical-budget teacher arithmetic and concurrency.

Each invocation measures the production full-refresh path in one fresh process.
Feature overrides are private to this benchmark; they never create checkpoints.
"""

import argparse
import heapq
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from RL.tdmpc2_core.batched_teacher import BatchedTeacherCore
from tests import benchmark_ambi_reanalyze as reanalyze_benchmark
from tests.benchmark_ambi_reanalyze import benchmark


FEATURES = {
    "reference": frozenset(),
    "heads": frozenset({"selective_q"}),
    "heads-cache": frozenset({"selective_q", "actor_cache"}),
    "all": frozenset({"selective_q", "actor_cache", "fused_updates"}),
    "first-layer": frozenset({"first_layer_fusion"}),
    "all-first-layer": frozenset({"selective_q", "actor_cache", "fused_updates", "first_layer_fusion"}),
}


def _event_times(event):
    return {
        "name": str(getattr(event, "key", getattr(event, "name", "")))[:1024],
        "device_type": str(event.device_type), "count": int(event.count),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "cpu_time_total_us": float(event.cpu_time_total),
        "self_cuda_time_total_us": float(event.self_cuda_time_total),
        "cuda_time_total_us": float(event.cuda_time_total),
    }


def _trace_refresh(operation, *, device, directory, stem):
    """One additional warmed invocation; never contributes timing samples."""
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / (stem + ".trace.json.gz")
    summary_path = directory / (stem + ".summary.json")
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as profiler:
        started = time.perf_counter()
        with torch.profiler.record_function("ambi_teacher_full_refresh_diagnostic"):
            operation()
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    profiler.export_chrome_trace(str(trace_path))
    averages = [_event_times(event) for event in profiler.key_averages()]
    cpu = [event for event in averages if event["device_type"].endswith("CPU")]
    cuda = [event for event in averages if event["device_type"].endswith("CUDA")]
    summary = {
        "diagnostic_only": True, "timing_samples_include_profiling": False,
        "profiled_refreshes": 1, "profiled_elapsed_seconds": elapsed,
        "chrome_trace": str(trace_path.resolve()),
        "timing_note": (
            "CPU operator CUDA totals include child work; do not add them to raw CUDA events. "
            "CUDA device-event duration sums can exceed wall time when streams overlap."
        ),
        "cpu_operators_by_self_cpu_time": heapq.nlargest(50, cpu, key=lambda event: event["self_cpu_time_total_us"]),
        "cpu_operators_by_self_cuda_time": heapq.nlargest(50, cpu, key=lambda event: event["self_cuda_time_total_us"]),
        "cuda_device_averages": heapq.nlargest(50, cuda, key=lambda event: event["self_cuda_time_total_us"]),
    }
    # Inspect actual device records separately from host operators' attributed
    # GPU time. Keep only aggregate names and twenty slow events in the JSON.
    aggregates, slowest = {}, []
    count, total = 0, 0.
    try:
        for event in profiler.events():
            if not str(event.device_type).endswith("CUDA"):
                continue
            duration = float(event.cuda_time_total)
            count += 1
            total += duration
            key = (str(event.name), int(event.device_index))
            group = aggregates.setdefault(key, {
                "name": key[0][:1024], "device_index": key[1], "count": 0,
                "cuda_time_total_us": 0., "max_duration_us": 0.,
            })
            group["count"] += 1
            group["cuda_time_total_us"] += duration
            group["max_duration_us"] = max(group["max_duration_us"], duration)
            if len(slowest) < 20 or duration > slowest[0][0]:
                row = {"name": key[0][:1024], "device_index": key[1],
                       "duration_us": duration, "start_us": float(event.time_range.start),
                       "end_us": float(event.time_range.end)}
                item = (duration, count, row)
                if len(slowest) < 20:
                    heapq.heappush(slowest, item)
                else:
                    heapq.heapreplace(slowest, item)
        summary.update(
            cuda_device_event_count=count, cuda_device_event_duration_sum_us=total,
            cuda_device_event_aggregates=heapq.nlargest(
                50, aggregates.values(), key=lambda event: event["cuda_time_total_us"],
            ),
            slowest_cuda_device_events=[item[2] for item in sorted(slowest, reverse=True)],
        )
        if not count:
            summary["cuda_device_events_warning"] = "Profiler returned no CUDA device records; GPU kernel coverage is unavailable."
    except Exception as error:
        summary["cuda_device_events_error"] = f"{type(error).__name__}: {error}"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {"diagnostic_only": True, "profiled_elapsed_seconds": elapsed,
            "chrome_trace": str(trace_path.resolve()), "summary": str(summary_path.resolve()),
            **{key: value for key, value in summary.items() if key.endswith(("_error", "_warning"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=FEATURES, default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--roots", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", choices=("tiny", "reference"), default="reference")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--matmul-precision", choices=("high", "highest"), default="high")
    parser.add_argument("--trace-dir", type=Path,
                        help="CUDA only: trace one extra warmed refresh after all timing samples.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers, args.width, args.roots, args.repeats, args.warmup) < 1:
        parser.error("All counts must be positive.")
    if args.workers > 1 and args.device != "cuda":
        parser.error("Concurrent workers require CUDA.")
    if args.trace_dir is not None and args.device != "cuda":
        parser.error("--trace-dir requires --device cuda.")
    args.compile = not args.eager
    args.full_refresh = True
    args.batched_backend = "batched_cuda_graph" if args.device == "cuda" else "batched"
    args.reference_batched = args.features in {"reference", "first-layer"}
    args.batched_workers = args.workers
    args.batched_first_layer_fusion = "first_layer_fusion" in FEATURES[args.features]
    args.parallel_roots = [args.width]
    args.threads = 1
    args.graph_only = False
    args.reuse_graph_pool = False
    args.compile_cache_limit = 128
    initialize = BatchedTeacherCore.__init__
    measure = reanalyze_benchmark._measure
    diagnostics = []

    def with_features(self, *positional, **keywords):
        keywords["optimization_features"] = FEATURES[args.features]
        initialize(self, *positional, **keywords)

    def measured_then_profiled(operation, **keywords):
        outputs, measured = measure(operation, **keywords)
        if args.trace_dir is not None and not diagnostics:
            try:
                diagnostic = _trace_refresh(
                    operation, device=keywords["device"], directory=args.trace_dir,
                    stem=f"{args.features}-workers{args.workers}-width{args.width}-roots{args.roots}",
                )
            except Exception as error:
                # Optional profiler availability must not discard a completed
                # synchronized timing comparison or prevent subsequent arms.
                diagnostic = {"diagnostic_only": True, "error": f"{type(error).__name__}: {error}"}
                print(json.dumps({"profiling_failed": diagnostic}), flush=True)
            diagnostics.append(diagnostic)
        return outputs, measured

    with patch.object(BatchedTeacherCore, "__init__", with_features), \
            patch.object(reanalyze_benchmark, "_measure", measured_then_profiled):
        result = benchmark(args)
    result.update(optimization_features=sorted(FEATURES[args.features]),
                  feature_set=args.features, workers=args.workers,
                  checkpointing=False, wandb=False)
    if diagnostics:
        result["profiling"] = diagnostics[0]
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
