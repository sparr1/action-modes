"""Bounded, unpublished gSDE actor and inner-solve latency measurements.

Run with ``python -m tests.benchmark_ambi_gsde --output /absolute/path.json``.
CUDA timings synchronize explicitly; compilation/first-call costs are separate
from warmed latency. A local source snapshot can provide a comparison using
``--reference-root`` without editing or importing it as the active checkout.
No environments are stepped, outer training is run, or W&B runs are created.
"""

import argparse
from contextlib import ExitStack, contextmanager
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

import torch

from RL.tdmpc2_core import ambi_agent
from RL.tdmpc2_core.common.gsde import GsdeActor
from tests.test_ambi_config_decoupling import _build_cfg


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callable(function, *, device, warmup, steps):
    """Measure wall latency without hiding asynchronous device work."""
    synchronize(device)
    start = time.perf_counter()
    function()
    synchronize(device)
    first_ms = 1000 * (time.perf_counter() - start)
    for _ in range(warmup):
        function()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed, gpu_elapsed = [], []
    for _ in range(steps):
        synchronize(device)
        events = None
        if device.type == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        start = time.perf_counter()
        function()
        if events is not None:
            events[1].record()
        synchronize(device)
        elapsed.append(1000 * (time.perf_counter() - start))
        if events is not None:
            gpu_elapsed.append(events[0].elapsed_time(events[1]))
    return {
        "first_call_ms": first_ms,
        "warmup_calls": warmup,
        "measured_calls": steps,
        "wall_median_ms": statistics.median(elapsed),
        "wall_min_ms": min(elapsed),
        "wall_max_ms": max(elapsed),
        "cuda_median_ms": statistics.median(gpu_elapsed) if gpu_elapsed else None,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def profile_callable(function, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
        function()
        synchronize(device)
    events = sorted(profile.key_averages(), key=lambda event: event.self_cpu_time_total,
                    reverse=True)[:12]
    return [{"operator": event.key, "calls": event.count,
             "self_cpu_us": event.self_cpu_time_total,
             "self_cpu_memory_bytes": event.self_cpu_memory_usage,
             "self_cuda_us": getattr(event, "self_cuda_time_total", 0)}
            for event in events]


def _load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference(root):
    common = root / "RL/tdmpc2_core/common"
    actor = _load_reference(common / "gsde.py", "RL.tdmpc2_core.common._gsde_benchmark_reference")
    world = _load_reference(common / "soft_world_model.py", "RL.tdmpc2_core.common._gsde_world_benchmark_reference")
    world.GsdeActor = actor.GsdeActor
    inner = _load_reference(root / "RL/tdmpc2_core/inner_improvement.py",
                            "RL.tdmpc2_core._gsde_inner_benchmark_reference")
    return {"actor": actor.GsdeActor, "world": world.SoftWorldModel,
            "inner": inner.InnerImprovementEngine}


@contextmanager
def compile_backend(backend):
    if backend == "eager":
        yield
        return
    original_compile = torch.compile

    def selected_backend(function, **kwargs):
        kwargs["backend"] = backend
        return original_compile(function, **kwargs)

    with patch.object(torch, "compile", selected_backend):
        yield


def actor_case(args, actor_type, *, batch, source, matrix, backward, backend):
    torch.manual_seed(args.seed)
    actor = actor_type(args.width, args.width, args.action_dim,
                       feature_source=source).to(args.device).eval()
    x = torch.randn(batch, args.width, device=args.device)
    noise = torch.randn(batch, args.width, args.action_dim, device=args.device) if matrix else None
    forward = actor if backend == "eager" else torch.compile(
        actor, backend=backend, fullgraph=True, dynamic=False,
    )

    def step():
        if backward:
            actor.zero_grad(set_to_none=True)
            mean, log_std, standardized = forward(x, sde_noise=noise)
            loss = mean.square().mean() + log_std.mean()
            if standardized is not None:
                loss = loss + standardized.square().mean()
            loss.backward()
        else:
            with torch.no_grad():
                forward(x, sde_noise=noise)

    result = time_callable(step, device=args.device, warmup=args.warmup, steps=args.steps)
    if args.profile and batch == max(args.batches) and backward:
        result["profile"] = profile_callable(step, args.device)
    return result


def inner_case(args, reference, *, size, backend):
    representative = size == "representative"
    width = args.width if representative else 32
    action_dim = args.action_dim if representative else 3
    count, batch = (128, 256) if representative else (8, 16)
    cfg = _build_cfg(
        model_size=None, device=str(args.device), enc_dim=width, mlp_dim=width,
        latent_dim=width, num_enc_layers=2, simnorm_dim=8, num_bins=101,
        q_representation="distributional", num_q=5, q_num_bins=101,
        q_vmin=-10, q_vmax=10, batch_size=batch, buffer_size=1024,
        train_unroll_horizon=3, outer_planning_horizon=3,
        actor_distribution="gsde", gsde_feature_source="actor",
        outer_actor_entropy_mode="pre_tanh", inner_actor_entropy_mode="pre_tanh",
        ent_coef=0.1, inner_temperature_mode="fixed", inner_temperature=0.1,
        inner_rounds=2, inner_rollouts_per_round=count, inner_rollout_horizon=3,
        inner_updates_per_round=1, inner_batch_size=batch, inner_replay_capacity=count * 6,
        inner_diagnostic_rollouts=0, dropout=0.0, compile=backend != "eager",
        compile_strict=True, inner_cuda_graph=False, episodic=False,
        critic_geometry_diagnostics=False, outer_policy_diagnostics=False,
        wandb=False, wandb_mode="disabled", critic_ensemble_backend="vmap",
    )
    cfg.action_dim = action_dim
    cfg.device = str(args.device)
    torch.manual_seed(args.seed)
    with ExitStack() as stack:
        if reference is not None:
            stack.enter_context(patch.object(ambi_agent, "SoftWorldModel", reference["world"]))
            stack.enter_context(patch.object(ambi_agent, "InnerImprovementEngine", reference["inner"]))
        agent = ambi_agent.AMBITDMPC2Agent(cfg)
    observation = torch.tensor([0.25, -0.1, 0.7], device=args.device)
    first = True

    def step():
        nonlocal first
        agent.act(observation, t0=first, eval_mode=False, collect_diagnostics=False)
        first = False

    with compile_backend(backend):
        result = time_callable(step, device=args.device, warmup=args.inner_warmup,
                               steps=args.inner_steps)
        if args.profile:
            result["profile"] = profile_callable(step, args.device)
    result["shape"] = {"width": width, "action_dim": action_dim, "rounds": 2,
                       "rollouts": count, "horizon": 3, "batch": batch, "q_heads": 5}
    result["inner_compile_fallbacks"] = {
        key: float(value) for key, value in agent.last_inner_metrics.items()
        if "compile" in key and "fallback" in key
    }
    return result


def benchmark(args):
    os.environ["WANDB_MODE"] = "disabled"
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    torch.set_num_threads(args.threads)
    if any(backend != "eager" for backend in args.compile_backends):
        # Bound compiler workers as well as numerical workers. This also avoids
        # the default compiler fork pool in single-thread macOS measurements.
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(args.threads)
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = args.threads
    reference = load_reference(args.reference_root) if args.reference_root is not None else None
    result = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "device": str(args.device),
                        "device_name": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else platform.processor(),
                        "threads": torch.get_num_threads(), "wandb_mode": os.environ["WANDB_MODE"]},
        "settings": {key: str(value) if isinstance(value, (Path, torch.device)) else value
                     for key, value in vars(args).items()},
        "actors": [], "inner_solves": [],
    }
    variants = []
    if reference is not None and "reference" in args.variants:
        variants.append(("reference", reference))
    if "current" in args.variants:
        variants.append(("current", None))
    for label, sources in variants:
        for backend in args.compile_backends:
            if "actors" in args.sections:
                actor_type = GsdeActor if sources is None else sources["actor"]
                for source in args.feature_sources:
                    for batch in args.batches:
                        for matrix in (False, True):
                            for backward in (False, True):
                                row = {"variant": label, "backend": backend, "batch": batch,
                                       "feature_source": source, "matrix": matrix, "backward": backward}
                                try:
                                    row.update(actor_case(args, actor_type, batch=batch, source=source,
                                                          matrix=matrix, backward=backward, backend=backend))
                                except Exception as error:
                                    row["error"] = f"{type(error).__name__}: {error}"
                                result["actors"].append(row)
            if "inner" in args.sections:
                for size in args.inner_sizes:
                    row = {"variant": label, "backend": backend, "size": size}
                    try:
                        row.update(inner_case(args, sources, size=size, backend=backend))
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                    result["inner_solves"].append(row)
            gc.collect()
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--reference-root", type=Path)
    result.add_argument("--variants", nargs="+", choices=("reference", "current"), default=["reference", "current"])
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--width", type=int, default=512)
    result.add_argument("--action-dim", type=int, default=21)
    result.add_argument("--batches", type=int, nargs="+", default=[1, 128, 256, 1024])
    result.add_argument("--feature-sources", nargs="+", choices=("actor", "latent"), default=["actor"])
    result.add_argument("--compile-backends", nargs="+", choices=("eager", "aot_eager", "inductor"), default=["eager"])
    result.add_argument("--sections", nargs="+", choices=("actors", "inner"), default=["actors", "inner"])
    result.add_argument("--inner-sizes", nargs="+", choices=("tiny", "representative"), default=["tiny", "representative"])
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--steps", type=int, default=7)
    result.add_argument("--inner-warmup", type=int, default=1)
    result.add_argument("--inner-steps", type=int, default=3)
    result.add_argument("--seed", type=int, default=41)
    result.add_argument("--profile", action="store_true")
    return result


def main():
    command = parser()
    args = command.parse_args()
    if (min(args.width, args.action_dim, args.threads, args.steps, args.inner_steps, *args.batches) <= 0
            or min(args.warmup, args.inner_warmup) < 0):
        command.error("Dimensions, threads and measured steps must be positive; warmup must be nonnegative.")
    if args.output.exists():
        command.error(f"Output already exists: {args.output}")
    if args.variants == ["reference"] and args.reference_root is None:
        command.error("The reference-only variant requires --reference-root.")
    output = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    errors = sum("error" in row for group in ("actors", "inner_solves") for row in output[group])
    print(json.dumps({"output": str(args.output.resolve()), "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
