"""Compare loop and vmap critic heads in complete compiled learner updates.

Four fresh-process conditions cross head vectorization with auxiliary PAVE off
or on at all sampled transitions. This benchmark preserves the original model
parameters and optimizers; it does not save trained weights or create W&B runs.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

import benchmark_pave as timing


CASES = {
    "loop_no_pave": {"ensemble_backend": "loop", "pave_enabled": False},
    "vmap_no_pave": {"ensemble_backend": "vmap", "pave_enabled": False},
    "loop_pave": {"ensemble_backend": "loop", "pave_enabled": True},
    "vmap_pave": {"ensemble_backend": "vmap", "pave_enabled": True},
}
COMPARISONS = {
    "vmap_vs_loop_without_pave": ("vmap_no_pave", "loop_no_pave"),
    "vmap_vs_loop_with_pave": ("vmap_pave", "loop_pave"),
    "pave_overhead_loop": ("loop_pave", "loop_no_pave"),
    "pave_overhead_vmap": ("vmap_pave", "vmap_no_pave"),
}


def case_order(repeat):
    names = list(CASES)
    offset = repeat % len(names)
    return names[offset:] + names[:offset]


def case_overrides(case):
    if case not in CASES:
        raise ValueError(f"Unknown ensemble timing case: {case}")
    controls = timing.case_overrides("aux_all" if CASES[case]["pave_enabled"] else "baseline")
    # Keep the same inert sampling control when PAVE is disabled.
    controls["outer_pave_batch_size"] = None
    return controls


def _backend_context(case):
    if CASES[case]["ensemble_backend"] == "vmap":
        from utils.ensemble_vmap_benchmark import vectorized_ensemble_benchmark
        return vectorized_ensemble_benchmark()
    return nullcontext({
        "implementation": "production_modulelist_loop",
        "ordinary_forward": "loop_shared_inputs", "detached_actor_forward": "loop_shared_inputs_detached_parameters",
        "pave_forward": "eager_loop_independent_head_actions_separate_views",
        "parameter_ownership": "original_modulelist_parameters",
        "parameter_stacking": "none", "persistent_stacked_parameters": False, "view_fusion": False,
        "dropout_randomness": "ordinary_per_head_loop_draws",
        "compile_boundary": "production_compiled_ensemble_loop",
    })


def _source():
    identity = timing._source_identity()
    root = Path(__file__).resolve().parent
    for relative in ("benchmark_pave.py", Path(__file__).name, "utils/ensemble_vmap_benchmark.py"):
        identity["files"][relative] = timing._file_sha256(root / relative)
    identity["sha256"] = timing._fingerprint(identity["files"])
    return identity


def run_worker(args):
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise ValueError("Production ensemble timing requires an available CUDA GPU.")
    record = timing._record(args)
    overrides = case_overrides(args.worker_case)
    record["resolved"]["algorithm_config"]["alg_params"].update(overrides)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Install before construction/loading so every compiled wrapper belongs to
    # exactly one backend. Keep the patch through cleanup in this fresh process.
    try:
        with _backend_context(args.worker_case) as backend:
            model = env = None
            error = None
            try:
                source = _source()
                checkpoint_hash = timing._file_sha256(record["path"])
                metadata_hash = timing._file_sha256(record["context"].source)
                bank_hash = timing._file_sha256(record["bank_path"])
                env = timing._make_env(record["resolved"])
                model, run_config = timing._initialize_probe_model(record, env, args.device)
                agent = model.agent
                if agent.cfg.batch_size != 256 or agent.cfg.train_unroll_horizon != 3:
                    raise ValueError("This benchmark requires the saved B=256, H=3 training protocol (768 rows).")
                bank = timing.load_probe_bank(record["bank_path"])
                timing._validate_bank(bank, record, model)
                repeat_seed = (args.seed + 1_000_003 * args.worker_repeat) % 2**32
                seeds = timing.batch_seeds(args.seed, args.worker_repeat, args.warmup_updates + args.updates)
                batches = [timing.sample_probe_windows(
                    bank, horizon=agent.cfg.train_unroll_horizon, batch_size=agent.cfg.batch_size,
                    seed=seed, device=agent.device) for seed in seeds]
                random.seed(repeat_seed)
                np.random.seed(repeat_seed)
                torch.manual_seed(repeat_seed)
                torch.cuda.manual_seed_all(repeat_seed)
                measured = timing.time_updates(
                    agent, batches, warmup_updates=args.warmup_updates,
                    synchronize=lambda: torch.cuda.synchronize(agent.device))
                for path, expected in ((record["path"], checkpoint_hash),
                                       (record["context"].source, metadata_hash),
                                       (record["bank_path"], bank_hash)):
                    if timing._file_sha256(path) != expected:
                        raise RuntimeError(f"Benchmark source input changed during timing: {path}")
                if _source() != source:
                    raise RuntimeError("Benchmark source code changed during timing.")
                config = timing._jsonable(vars(model.cfg))
                properties = torch.cuda.get_device_properties(agent.device)
                result = {
                    "schema": "ambi-ensemble-vmap-complete-update-timing-case", "version": 1,
                    "case": args.worker_case, "repeat": args.worker_repeat,
                    "backend": timing._jsonable(backend), "overrides": overrides,
                    "repeat_seed": repeat_seed, "batch_seeds": seeds,
                    "resolved_cfg": config, "resolved_cfg_sha256": timing._fingerprint(config),
                    "resolved_run_config": run_config, "source_identity": source,
                    "checkpoint": {"path": str(record["path"]), "sha256": checkpoint_hash,
                                   "metadata_path": str(record["context"].source),
                                   "metadata_sha256": metadata_hash},
                    "bank": {"path": str(record["bank_path"]), "sha256": bank_hash,
                             "transition_count": len(bank), "provenance": bank.provenance},
                    "runtime": {
                        "python": platform.python_version(), "torch": torch.__version__, "hostname": platform.node(),
                        "cuda": torch.version.cuda, "device": str(agent.device),
                        "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
                        "gpu_capability": [properties.major, properties.minor],
                        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "compile_backend": "inductor", "compile_strict": True,
                        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cpu_threads": torch.get_num_threads()},
                    "pave_rows": 768 if CASES[args.worker_case]["pave_enabled"] else 0,
                    **measured,
                }
                timing._save(output / "result.json", result)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                timing._close_resources(model, env, primary_error=error)
    except BaseException as exc:
        timing._save(output / "failure.json", {"error": str(exc), "complete": False})
        raise


def validate_results(results, args, source):
    """Reject incomplete or unpaired workers before publishing a comparison."""
    expected = {(repeat, case) for repeat in range(args.repeats) for case in CASES}
    identities = [(row["repeat"], row["case"]) for row in results]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Ensemble timing requires exactly one result for every repeat and case.")
    reference = results[0]
    controls = set(case_overrides("loop_no_pave"))
    reference_cfg = {key: value for key, value in reference["resolved_cfg"].items() if key not in controls}
    for row in results:
        case = row["case"]
        if row["source_identity"] != source:
            raise ValueError("Workers did not use the same benchmark source identity.")
        if any(row[key] != reference[key] for key in ("checkpoint", "bank", "runtime")):
            raise ValueError("Workers did not use the same checkpoint, bank, and GPU runtime.")
        if row["overrides"] != case_overrides(case):
            raise ValueError("Worker case settings differ from the requested protocol.")
        backend = row["backend"]
        if backend["parameter_ownership"] != "original_modulelist_parameters" or backend["persistent_stacked_parameters"] or backend["view_fusion"]:
            raise ValueError("Worker changed parameter ownership or fused observation views.")
        if CASES[case]["ensemble_backend"] == "vmap":
            required = {
                "implementation": "torch_vmap_differentiable_parameter_stack_v1",
                "ordinary_forward": "vmap_shared_inputs",
                "detached_actor_forward": "vmap_shared_inputs_detached_parameters",
                "pave_forward": "eager_vmap_independent_head_actions_separate_views",
                "parameter_stacking": "each_forward_included_in_timing",
                "dropout_randomness": "different_per_head_distribution_equivalent_rng_order_differs",
                "compile_boundary": "pure_tensor_vmap_allow_in_graph_aot_traced", "context_seams_installed": True,
            }
        else:
            required = {
                "implementation": "production_modulelist_loop", "ordinary_forward": "loop_shared_inputs",
                "detached_actor_forward": "loop_shared_inputs_detached_parameters",
                "pave_forward": "eager_loop_independent_head_actions_separate_views",
                "parameter_stacking": "none", "dropout_randomness": "ordinary_per_head_loop_draws",
            }
        if any(backend.get(key) != value for key, value in required.items()):
            raise ValueError("Worker ensemble backend differs from its assigned case.")
        cfg = row["resolved_cfg"]
        if any(cfg.get(key) != value for key, value in row["overrides"].items()):
            raise ValueError("Resolved worker settings differ from its case overrides.")
        if {key: value for key, value in cfg.items() if key not in controls} != reference_cfg:
            raise ValueError("Worker training configurations differ beyond the intended PAVE controls.")
        if row["resolved_cfg_sha256"] != timing._fingerprint(cfg):
            raise ValueError("Worker configuration fingerprint is inconsistent.")
        if row["batch_seeds"] != timing.batch_seeds(args.seed, row["repeat"], args.warmup_updates + args.updates):
            raise ValueError("Worker resident batches are not paired by repeat.")
        if row["repeat_seed"] != (args.seed + 1_000_003 * row["repeat"]) % 2**32:
            raise ValueError("Worker training RNG seed differs from the protocol.")
        if row["start_num_updates"] != reference["start_num_updates"]:
            raise ValueError("Workers did not start from the same checkpoint update count.")
        if row["warmup_updates"] != args.warmup_updates or row["measured_updates"] != args.updates:
            raise ValueError("Worker update counts differ from the protocol.")
        total = args.warmup_updates + args.updates
        if row["num_updates_delta"] != total or row["aux_num_updates_delta"] != total:
            raise ValueError("Worker did not execute every complete primary and auxiliary update.")
        if row["end_num_updates"] - row["start_num_updates"] != total:
            raise ValueError("Worker final update counter is inconsistent.")
        if row["compile_fallback"] or not row["parameters_finite_before_and_after"] or not row["all_update_metrics_finite"]:
            raise ValueError("Worker failed compilation or finiteness requirements.")
        if row["pave_rows"] != (768 if CASES[case]["pave_enabled"] else 0):
            raise ValueError("Worker did not use the requested full PAVE row count.")
        if row["summary"] != timing.summarize_ms(row["timings_ms"]) or len(row["timings_ms"]) != args.updates:
            raise ValueError("Worker timing samples or summary are inconsistent.")


def summarize_cases(results):
    summaries = {}
    indexed = {(row["repeat"], row["case"]): row for row in results}
    for case in CASES:
        rows = sorted((row for row in results if row["case"] == case), key=lambda row: row["repeat"])
        summary = timing.summarize_ms([value for row in rows for value in row["timings_ms"]])
        summary.update(repeats=len(rows), repeat_medians_ms=[row["summary"]["median_ms"] for row in rows],
                       max_peak_allocated_bytes=max(row["memory"]["peak_allocated_bytes"] for row in rows))
        summaries[case] = summary
    comparisons = {}
    repeats = sorted({row["repeat"] for row in results})
    for name, (numerator, denominator) in COMPARISONS.items():
        ratios = [indexed[(repeat, numerator)]["summary"]["median_ms"] /
                  indexed[(repeat, denominator)]["summary"]["median_ms"] for repeat in repeats]
        comparisons[name] = {"numerator": numerator, "denominator": denominator,
                             "paired_median_ratios": ratios, "median_paired_ratio": float(np.median(ratios))}
    return summaries, comparisons


def _report(result):
    lines = ["# Ensemble vmap complete-update timing", "",
             "Synchronized public learner-update latency on GPU-resident batches. Includes world/critic, "
             "actor, temperature, target updates, and learner metrics. Data preparation, warmup, and "
             "validation are excluded. This is not end-to-end training throughput.", "",
             "The vmap cases stack the original parameters on every forward; this work is included in timing. "
             "Ordinary dropout is retained with vmap randomness='different': masks are distribution-equivalent "
             "but are not expected to match the loop cases bit for bit. PAVE dropout remains disabled. "
             "Observation views are not fused.", "",
             "| Case | Median ms | P90 ms | Mean ms | Peak allocated MiB |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for case, row in result["summary"].items():
        lines.append(f"| {case} | {row['median_ms']:.3f} | {row['p90_ms']:.3f} | {row['mean_ms']:.3f} | "
                     f"{row['max_peak_allocated_bytes'] / 2**20:.1f} |")
    lines.extend(["", "Ratios use paired per-repeat medians; values below one mean a faster numerator.", "",
                  "| Comparison | Median paired time ratio |", "| --- | ---: |"])
    for row in result["comparisons"].values():
        lines.append(f"| {row['numerator']} / {row['denominator']} | {row['median_paired_ratio']:.3f}× |")
    lines.extend(["", f"There are {result['protocol']['repeats']} fresh-process repeats per condition, "
                  f"each with {result['protocol']['measured_updates']} measured updates. Consecutive updates "
                  "within a repeat are not independent trials."])
    return "\n".join(lines) + "\n"


def run_benchmark(args):
    timing._record(args)
    source = _source()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "ambi-ensemble-vmap-complete-update-timing", "version": 1, "complete": False,
        "source_identity": source,
        "protocol": {
            "repeats": args.repeats, "warmup_updates": args.warmup_updates, "measured_updates": args.updates,
            "cases": CASES, "auxiliary_weights": timing._AUXILIARY_WEIGHTS, "seed": args.seed,
            "batch_size": 256, "horizon": 3, "pave_batch_size": None, "pave_rows": 768,
            "obs_noise_std": .01, "curvature_margin": 1.,
            "fresh_process_and_checkpoint_per_case": True,
            "optimizer_initialization": "portable_checkpoint_load_restores_saved_optimizer_states",
            "order": [case_order(repeat) for repeat in range(args.repeats)],
            "timing_scope": "synchronized_public_agent_update_latency_on_resident_batch_not_pipelined_throughput",
            "excluded": ["bank_io", "window_sampling", "host_to_device_transfer", "finite_checks", "reporting"],
            "ordinary_compile": "strict_inductor", "pave_execution": "eager",
            "vmap_parameter_stacking": "differentiable_stacks_of_original_parameters_per_forward_included_in_timing",
            "vmap_randomness": "different", "dropout_pairing": "distribution_equivalent_not_bitwise",
            "view_fusion": False,
            "disabled_diagnostics": ["outer_policy_diagnostics", "value_equivalence_diagnostics"]},
        "results": [],
    }
    deadline = time.monotonic() + args.timeout_seconds
    try:
        for repeat in range(args.repeats):
            for case in case_order(repeat):
                child_output = output / f"repeat-{repeat}-{case}"
                command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint.resolve()),
                           "--bank", str(args.bank.resolve()), "--output-dir", str(child_output),
                           "--device", args.device, "--warmup-updates", str(args.warmup_updates),
                           "--updates", str(args.updates), "--seed", str(args.seed),
                           "--worker-case", case, "--worker-repeat", str(repeat)]
                if args.metadata is not None:
                    command.extend(["--metadata", str(args.metadata.resolve())])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Ensemble timing reached its wall-time budget.")
                print(f"Starting repeat {repeat + 1}/{args.repeats}, case {case}", flush=True)
                with (output / f"repeat-{repeat}-{case}.log").open("x") as log:
                    subprocess.run(command, cwd=Path(__file__).resolve().parent,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
                row = json.loads((child_output / "result.json").read_text())
                result["results"].append(row)
                print(f"Completed {case}: median {row['summary']['median_ms']:.3f} ms, "
                      f"P90 {row['summary']['p90_ms']:.3f} ms", flush=True)
        validate_results(result["results"], args, source)
        if _source() != source:
            raise RuntimeError("Benchmark source code changed during the comparison.")
        result["summary"], result["comparisons"] = summarize_cases(result["results"])
        result["complete"] = True
        timing._write_output_atomic(output / "report.md", _report(result), overwrite=False)
        timing._save(output / "results.json", result)
        return result
    except BaseException as exc:
        # Per-case outputs and this aggregate preserve all successful work.
        result["complete"] = False
        timing._save(output / "partial-results.json", result)
        timing._save(output / "failure.json", {"complete": False, "error": str(exc),
                                               "completed_cases": len(result["results"])})
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-updates", type=int, default=10)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout-seconds", type=float, default=2400.)
    parser.add_argument("--worker-case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeat", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.warmup_updates < 1 or args.updates < 1 or args.repeats < 1 or args.worker_repeat < 0:
            raise ValueError("Warmup, update, and repeat counts must be positive.")
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise ValueError("timeout-seconds must be finite and positive.")
        if torch.device(args.device).type != "cuda":
            raise ValueError("The timing CLI requires a CUDA device.")
        (run_worker if args.worker_case is not None else run_benchmark)(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Ensemble timing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
