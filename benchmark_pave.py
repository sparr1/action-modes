"""Time complete compiled outer updates on disposable checkpoint copies.

Four sequential, fresh-process cases compare PAVE off with auxiliary-only
(.005, .005, .0005) at caps 64, 256, and all transitions. Source checkpoint and
bank files remain read-only; only a new results directory is written.
"""

from __future__ import annotations

import argparse
import hashlib
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

from evaluate_ambi_checkpoint import _close_resources, _file_sha256, _jsonable, _make_env, _write_output_atomic
from probe_pave import _inputs, _initialize_probe_model, _source_identity, _validate_bank
from utils.pave_identity import PAVE_COEFFICIENT_KEYS
from utils.pave_probe_data import load_probe_bank, sample_probe_windows


CASES = {"baseline": 64, "aux_64": 64, "aux_256": 256, "aux_all": None}
_AUXILIARY_WEIGHTS = {"mpr": .005, "vfc": .005, "curvature": .0005}


def case_order(repeat):
    names = list(CASES)
    offset = repeat % len(names)
    return names[offset:] + names[:offset]


def batch_seeds(seed, repeat, count):
    repeat_seed = (seed + 1_000_003 * repeat) % 2**32
    return [(repeat_seed + 104729 * index) % 2**32 for index in range(count)]


def case_overrides(case):
    if case not in CASES:
        raise ValueError(f"Unknown timing case: {case}")
    controls = dict.fromkeys(PAVE_COEFFICIENT_KEYS, 0.)
    if case != "baseline":
        controls.update({f"aux_return_pave_{term}_coef": value
                         for term, value in _AUXILIARY_WEIGHTS.items()})
    controls.update(outer_pave_batch_size=CASES[case], outer_pave_obs_noise_std=.01,
                    outer_pave_curvature_margin=1., compile=True, compile_strict=True,
                    wandb=False, outer_policy_diagnostics=False,
                    value_equivalence_diagnostics=False)
    return controls


def summarize_ms(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timing samples must be nonempty, finite, positive milliseconds.")
    return {"count": len(values), "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)), "mean_ms": float(values.mean())}


class _ResidentBatch:
    """The public learner update still owns its host RNG/CUDA graph boundary."""

    def __init__(self, batch):
        self.batch = batch

    def sample(self, *, include_behavior_policy=False):
        ordinary = (*self.batch, None)
        return (*ordinary, None, None, None) if include_behavior_policy else ordinary


def _check_metrics(info):
    values = {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
              for key, value in info.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise RuntimeError("Complete update returned nonfinite metrics.")
    if any(value != 0. for key, value in values.items() if key.startswith("compile_") and "fallback" in key):
        raise RuntimeError("Ordinary training compilation fell back to eager execution.")
    return {key: value for key, value in values.items()
            if key.endswith("loss") or key in {"grad_norm", "num_updates"}}


def _check_parameters(agent):
    if not all(bool(torch.isfinite(parameter).all()) for parameter in agent.parameters()):
        raise RuntimeError("The disposable timing model has nonfinite parameters.")


def _check_compilation(agent):
    if os.environ.get("TORCH_COMPILE_DISABLE", "0").lower() not in {"", "0", "false"} or torch._dynamo.config.disable:
        raise RuntimeError("Compilation is globally disabled; strict Inductor timing would be misleading.")
    region = agent._outer_update_region
    if not (agent.cfg.compile and agent.cfg.compile_strict and region.enabled
            and region.strict and region._compiled is not None and not region.failed):
        raise RuntimeError("The ordinary loss did not execute with strict compilation.")
    for attribute in ("_Qs", "_target_Qs", "_aux_return_Qs", "_target_aux_return_Qs"):
        ensemble = getattr(agent.model, attribute)
        if not ensemble._compile_enabled or not ensemble._compile_strict or ensemble.compile_failed:
            raise RuntimeError(f"{attribute} did not retain strict compiled execution.")


def time_updates(agent, batches, *, warmup_updates, synchronize, require_compilation=True,
                 clock=time.perf_counter):
    """Time only public complete updates over already resident tensors.

    CPU tests inject an eager tiny agent and a no-op synchronize function;
    the command-line worker separately requires a CUDA device and Inductor.
    """
    if warmup_updates < 1 or len(batches) <= warmup_updates:
        raise ValueError("Require warmup updates followed by measured updates.")
    if agent.behavior_policy_kl_enabled or (agent.aux_return is not None
                                           and agent.aux_return.behavior_policy_kl_enabled):
        raise ValueError("The bank has no recorded behavior-policy parameters for active behavior regularization.")
    start_updates = int(agent.num_updates)
    start_aux_updates = int(agent.aux_return.num_updates) if agent.aux_return is not None else None
    _check_parameters(agent)
    agent.model.eval()
    for batch in batches[:warmup_updates]:
        info = agent.update(_ResidentBatch(batch))
        synchronize()
        _check_metrics(info)
    del info
    _check_parameters(agent)
    if require_compilation:
        _check_compilation(agent)
    device = torch.device(agent.device)
    synchronize()
    memory = None
    if device.type == "cuda":
        memory = {"resident_allocated_bytes": torch.cuda.memory_allocated(device),
                  "resident_reserved_bytes": torch.cuda.memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    first_losses = last_losses = None
    for batch in batches[warmup_updates:]:
        resident = _ResidentBatch(batch)
        synchronize()
        start = clock()
        info = agent.update(resident)
        synchronize()
        timings.append((clock() - start) * 1000.)
        # Scalar transfer, finite checks, and bookkeeping are outside timing.
        last_losses = _check_metrics(info)
        if first_losses is None:
            first_losses = last_losses
        del info
    if memory is not None:
        memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        memory["incremental_peak_allocated_bytes"] = memory["peak_allocated_bytes"] - memory["resident_allocated_bytes"]
        memory["incremental_peak_reserved_bytes"] = memory["peak_reserved_bytes"] - memory["resident_reserved_bytes"]
    _check_parameters(agent)
    if require_compilation:
        _check_compilation(agent)
    expected = len(batches)
    if int(agent.num_updates) - start_updates != expected:
        raise RuntimeError("Complete update count differs from the requested protocol.")
    if start_aux_updates is not None and int(agent.aux_return.num_updates) - start_aux_updates != expected:
        raise RuntimeError("Auxiliary update count differs from the requested protocol.")
    return {"timings_ms": timings, "summary": summarize_ms(timings), "memory": memory,
            "warmup_updates": warmup_updates, "measured_updates": len(timings),
            "start_num_updates": start_updates, "end_num_updates": int(agent.num_updates),
            "num_updates_delta": int(agent.num_updates) - start_updates,
            "aux_num_updates_delta": None if start_aux_updates is None else int(agent.aux_return.num_updates) - start_aux_updates,
            "first_measured_losses": first_losses, "last_measured_losses": last_losses,
            "parameters_finite_before_and_after": True, "all_update_metrics_finite": True,
            "compile_fallback": False}


def _record(args):
    probe_args = argparse.Namespace(
        checkpoint=[args.checkpoint], metadata=None if args.metadata is None else [args.metadata],
        bank=[args.bank], collect_episodes=None, warmup_batches=args.warmup_updates,
        seed=args.seed, scales=[1.], critics=["auxiliary"], ordinary_execution="saved",
    )
    record = _inputs(probe_args)[0]
    params = record["resolved"]["algorithm_config"]["alg_params"]
    if params.get("aux_return_mode", "off") == "off":
        raise ValueError("Complete-update timing requires a saved auxiliary return critic.")
    return record


def _fingerprint(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source():
    identity = _source_identity()
    identity["files"][Path(__file__).name] = _file_sha256(Path(__file__))
    identity["sha256"] = _fingerprint(identity["files"])
    return identity


def _save(path, value):
    _write_output_atomic(path, json.dumps(value, indent=2, allow_nan=False) + "\n", overwrite=False)


def run_worker(args):
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise ValueError("Production complete-update timing requires an available CUDA GPU.")
    record = _record(args)
    overrides = case_overrides(args.worker_case)
    record["resolved"]["algorithm_config"]["alg_params"].update(overrides)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = env = None
    error = None
    try:
        checkpoint_hash = _file_sha256(record["path"])
        metadata_hash = _file_sha256(record["context"].source)
        bank_hash = _file_sha256(record["bank_path"])
        env = _make_env(record["resolved"])
        model, run_config = _initialize_probe_model(record, env, args.device)
        agent = model.agent
        bank = load_probe_bank(record["bank_path"])
        _validate_bank(bank, record, model)
        repeat_seed = (args.seed + 1_000_003 * args.worker_repeat) % 2**32
        seeds = batch_seeds(args.seed, args.worker_repeat, args.warmup_updates + args.updates)
        batches = [sample_probe_windows(bank, horizon=agent.cfg.train_unroll_horizon,
                                        batch_size=agent.cfg.batch_size, seed=seed,
                                        device=agent.device) for seed in seeds]
        random.seed(repeat_seed)
        np.random.seed(repeat_seed)
        torch.manual_seed(repeat_seed)
        torch.cuda.manual_seed_all(repeat_seed)
        # Fresh checkpoint restores identical private auxiliary streams in each
        # case; repeat-specific real batches and global draws vary across repeats.
        measured = time_updates(agent, batches, warmup_updates=args.warmup_updates,
                                synchronize=lambda: torch.cuda.synchronize(agent.device))
        for path, expected in ((record["path"], checkpoint_hash),
                               (record["context"].source, metadata_hash),
                               (record["bank_path"], bank_hash)):
            if _file_sha256(path) != expected:
                raise RuntimeError(f"Benchmark source input changed during timing: {path}")
        config = _jsonable(vars(model.cfg))
        properties = torch.cuda.get_device_properties(agent.device)
        result = {
            "schema": "ambi-pave-complete-update-timing-case", "version": 1,
            "case": args.worker_case, "repeat": args.worker_repeat,
            "overrides": overrides, "repeat_seed": repeat_seed, "batch_seeds": seeds,
            "resolved_cfg": config, "resolved_cfg_sha256": _fingerprint(config),
            "resolved_run_config": run_config, "source_identity": _source(),
            "checkpoint": {"path": str(record["path"]), "sha256": checkpoint_hash,
                           "metadata_path": str(record["context"].source),
                           "metadata_sha256": metadata_hash},
            "bank": {"path": str(record["bank_path"]), "sha256": bank_hash,
                     "transition_count": len(bank), "provenance": bank.provenance},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(agent.device),
                        "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
                        "gpu_capability": [properties.major, properties.minor],
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "compile_backend": "inductor", "compile_strict": True},
            "pave_rows": 0 if args.worker_case == "baseline" else min(
                CASES[args.worker_case] or agent.cfg.train_unroll_horizon * agent.cfg.batch_size,
                agent.cfg.train_unroll_horizon * agent.cfg.batch_size),
            **measured,
        }
        _save(output / "result.json", result)
        return result
    except BaseException as exc:
        error = exc
        _save(output / "failure.json", {"error": str(exc), "complete": False})
        raise
    finally:
        _close_resources(model, env, primary_error=error)


def summarize_cases(results):
    summaries = {}
    by_repeat = {(row["repeat"], row["case"]): row for row in results}
    for case in CASES:
        rows = [row for row in results if row["case"] == case]
        values = [value for row in rows for value in row["timings_ms"]]
        summary = summarize_ms(values)
        ratios = [row["summary"]["median_ms"] / by_repeat[(row["repeat"], "baseline")]["summary"]["median_ms"]
                  for row in rows]
        summary.update(repeats=len(rows), repeat_medians_ms=[row["summary"]["median_ms"] for row in rows],
                       paired_median_ratios_to_baseline=ratios,
                       median_paired_ratio_to_baseline=float(np.median(ratios)))
        summaries[case] = summary
    return summaries


def run_benchmark(args):
    _record(args)  # Resolve and validate inputs before creating output.
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "ambi-pave-complete-update-timing", "version": 1, "complete": False,
              "protocol": {"repeats": args.repeats, "warmup_updates": args.warmup_updates,
                           "measured_updates": args.updates, "cases": CASES,
                           "auxiliary_weights": _AUXILIARY_WEIGHTS, "seed": args.seed,
                           "fresh_process_and_checkpoint_per_case": True,
                           "optimizer_initialization": "portable_checkpoint_load_restores_saved_optimizer_states",
                           "order": [case_order(repeat) for repeat in range(args.repeats)],
                           "timing_scope": "synchronized_public_agent_update_latency_on_resident_batch_not_pipelined_throughput",
                           "excluded": ["bank_io", "window_sampling", "host_to_device_transfer", "finite_checks", "reporting"],
                           "ordinary_compile": "strict_inductor", "pave_execution": "eager",
                           "disabled_diagnostics": ["outer_policy_diagnostics", "value_equivalence_diagnostics"]},
              "results": []}
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
                    raise TimeoutError("Complete-update timing reached its wall-time budget.")
                print(f"Starting repeat {repeat + 1}/{args.repeats}, case {case}", flush=True)
                with (output / f"repeat-{repeat}-{case}.log").open("x") as log:
                    subprocess.run(command, cwd=Path(__file__).resolve().parent,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
                row = json.loads((child_output / "result.json").read_text())
                result["results"].append(row)
                print(f"Completed {case}: median {row['summary']['median_ms']:.3f} ms, "
                      f"P90 {row['summary']['p90_ms']:.3f} ms", flush=True)
        result["summary"] = summarize_cases(result["results"])
        result["complete"] = True
        _save(output / "results.json", result)
        lines = ["# PAVE complete-update timing", "",
                 "Includes world/critic, actor, temperature, target updates, and learner metrics. "
                 "Inputs are resident on the GPU; warmup, data preparation, and validation are excluded.", "",
                 "| Case | Median ms | P90 ms | Mean ms | Median paired slowdown |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for case, row in result["summary"].items():
            lines.append(f"| {case} | {row['median_ms']:.3f} | {row['p90_ms']:.3f} | {row['mean_ms']:.3f} | {row['median_paired_ratio_to_baseline']:.3f}× |")
        _write_output_atomic(output / "report.md", "\n".join(lines) + "\n", overwrite=False)
        return result
    except BaseException as exc:
        _save(output / "failure.json", {"complete": False, "error": str(exc), "completed_cases": len(result["results"])})
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
    parser.add_argument("--timeout-seconds", type=float, default=1500.)
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
        print(f"PAVE timing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
