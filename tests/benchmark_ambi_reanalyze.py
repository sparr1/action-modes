"""Bounded teacher-solve timing; no environment steps, training runs, or W&B.

CPU smoke: python -m tests.benchmark_ambi_reanalyze --device cpu
GPU recipe: python -m tests.benchmark_ambi_reanalyze --device cuda \
    --profile reference --compile --roots 60 --parallel-roots 1 4 8

Measures resident encoded roots, including fresh resets and teacher extraction.
This excludes observation encoding, replay writeback, and the outer update.
Graph setup is reported separately; graph and native use different RNG protocols.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

os.environ["WANDB_MODE"] = "disabled"

import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.benchmark_aux_return_joint_vmap import _build_cfg, _tiny_params


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_progress(result, output):
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
        temporary.replace(output)


def _measure(operation, *, device, warmup, repeats, roots):
    _sync(device)
    start = time.perf_counter()
    output = operation()
    _sync(device)
    setup = time.perf_counter() - start
    for _ in range(warmup):
        operation()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from torch._dynamo.utils import counters
    before = int(counters["stats"]["unique_graphs"])
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = operation()
        _sync(device)
        samples.append(time.perf_counter() - start)
    after = int(counters["stats"]["unique_graphs"])
    if after != before:
        raise RuntimeError("Compilation occurred during timing; increase --warmup.")
    return output, {
        "first_call_seconds": setup,
        "refresh_seconds": samples,
        "median_refresh_seconds": statistics.median(samples),
        "median_seconds_per_root": statistics.median(samples) / roots,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def benchmark(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this host.")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision(getattr(args, "matmul_precision", "high"))
    cache_limit = getattr(args, "compile_cache_limit", None)
    if cache_limit is not None:
        # Private actors induce separate identity guards in the pinned PyTorch
        # runtime. Widen only this standalone process's bounded benchmark cache.
        torch._dynamo.config.cache_size_limit = cache_limit
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 8 * cache_limit,
        )
    torch.manual_seed(193)
    params = _tiny_params(
        outer_actor_objective="distillation", inner_rounds=2,
        inner_updates_per_round=1, inner_replay_capacity=8,
        compile=args.compile, compile_strict=args.compile,
        inner_temperature_mode="inherit_outer", wandb=False,
    )
    if args.profile == "reference":
        params.pop("inner_updates_per_round")
        params.update(
            enc_dim=256, mlp_dim=512, latent_dim=512, simnorm_dim=8,
            num_q=5, num_bins=101, q_num_bins=101,
            q_representation="distributional", q_vmin=-10, q_vmax=10,
            dropout=0.01, critic_ensemble_backend="vmap",
            train_unroll_horizon=3, outer_planning_horizon=3,
            inner_rounds=8, inner_rollouts_per_round=128,
            inner_rollout_horizon=3, inner_batch_size=256,
            inner_replay_capacity=3072,
            inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
            inner_temperature_updates_per_round=0,
        )
    cfg = _build_cfg(**params)
    cfg.device = str(device)
    if args.profile == "reference":
        cfg.obs_shape = {"state": (67,)}
        cfg.obs_shapes = [cfg.obs_shape]
        cfg.action_dim = 21
        cfg.action_dims = [21]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    if getattr(args, "full_refresh", False):
        return benchmark_refresh(args, agent)
    with torch.no_grad():
        roots = agent.model.encode(torch.randn(args.roots, cfg.obs_shape["state"][0], device=device))
    seeds = [173 + index for index in range(args.roots)]

    def serial(engine, teacher_only):
        means, log_stds = [], []
        for root, seed in zip(roots.split(1), seeds):
            engine.reset_for_evaluation(seed, reuse_action_pool=True)
            if teacher_only:
                teacher, _ = engine.solve_teacher(root, t0=True)
            else:
                engine.act(root, t0=True, eval_mode=True, collect_diagnostics=False,
                           capture_teacher_policy=True, apply_inner_writeback=False)
                teacher = engine.last_teacher_policy
            means.append(teacher["pre_tanh_mean"])
            log_stds.append(teacher["log_std"])
        return torch.stack(means), torch.stack(log_stds)

    result = {
        "device": str(device), "torch": torch.__version__, "platform": platform.platform(),
        "profile": args.profile, "roots": args.roots, "threads": args.threads,
        "compile": args.compile, "warmup": args.warmup, "repeats": args.repeats,
        "compile_cache_limit": torch._dynamo.config.cache_size_limit,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "configuration": vars(cfg).copy(), "arms": {},
        "scope": "encoded roots through teacher extraction; excludes encoder/cache/outer learner",
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    controls = {}
    native_arms = [] if getattr(args, "graph_only", False) else [("legacy_native", False), ("teacher_native", True)]
    for name, fast in native_arms:
        print(json.dumps({"starting_arm": name}), flush=True)
        engine = InnerImprovementEngine(agent)
        outputs, measured = _measure(
            lambda: serial(engine, fast), device=device, warmup=args.warmup,
            repeats=args.repeats, roots=args.roots,
        )
        controls[name] = tuple(t.detach().cpu() for t in outputs)
        result["arms"][name] = measured
        _save_progress(result, getattr(args, "output", None))
        print(json.dumps({"completed_arm": name, **measured}), flush=True)
        del engine, outputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if controls:
        for actual, expected in zip(controls["teacher_native"], controls["legacy_native"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        result["native_teacher_bitwise_parity"] = True
    if device.type == "cuda":
        from RL.tdmpc2_core.parallel_teacher import ParallelTeacherSolver
        from RL.tdmpc2_core.teacher_graph import TEACHER_GRAPH_PROTOCOL
        result["graph_rng_protocol"] = TEACHER_GRAPH_PROTOCOL
        graph_control = None
        for width in args.parallel_roots:
            print(json.dumps({"starting_arm": f"cuda_graph_roots_{width}"}), flush=True)
            cfg.reanalyze_backend = "cuda_graph"
            cfg.reanalyze_parallel_roots = width
            solver = ParallelTeacherSolver(agent)
            outputs, measured = _measure(
                lambda: solver.solve(roots, seeds), device=device,
                warmup=args.warmup, repeats=args.repeats, roots=args.roots,
            )
            current = tuple(t.detach().cpu() for t in outputs[:2])
            if graph_control is None:
                graph_control = current
            else:
                for actual, expected in zip(current, graph_control):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            result["arms"][f"cuda_graph_roots_{width}"] = measured
            _save_progress(result, getattr(args, "output", None))
            print(json.dumps({"completed_arm": f"cuda_graph_roots_{width}", **measured}), flush=True)
            del outputs, solver
            gc.collect()
            torch.cuda.empty_cache()
        result["graph_pool_width_bitwise_parity"] = True
    if "legacy_native" in result["arms"]:
        baseline = result["arms"]["legacy_native"]["median_refresh_seconds"]
        for measured in result["arms"].values():
            measured["speedup_over_legacy_native"] = baseline / measured["median_refresh_seconds"]
    return result


def benchmark_refresh(args, agent):
    """Measure the production refresh including deduplication, encoder and cache."""
    from tensordict import TensorDict
    from RL.tdmpc2_core.common.buffer import Buffer
    from RL.tdmpc2_core.reanalyze import LazyReanalyzer

    cfg, device = agent.cfg, agent.device
    horizon = int(cfg.train_unroll_horizon)
    if args.roots % horizon:
        raise ValueError("--full-refresh requires --roots divisible by the training horizon.")
    sequences = args.roots // horizon
    cfg.reanalyze_batch_size = sequences
    replay_cfg = SimpleNamespace(**vars(cfg))
    replay_cfg.batch_size = sequences
    replay_cfg.buffer_size = replay_cfg.steps = max(4096, args.roots * 20)
    replay = Buffer(replay_cfg)
    rows = replay_cfg.buffer_size
    replay.add(TensorDict({
        "obs": torch.randn(rows, cfg.obs_shape["state"][0]),
        "action": torch.zeros(rows, cfg.action_dim),
        "reward": torch.zeros(rows), "terminated": torch.zeros(rows),
    }, batch_size=[rows]))
    selector = LazyReanalyzer(agent)
    for _ in range(100):
        batch, teacher = replay.sample_distillation()
        obs = batch[0]
        positions, _, _ = selector._positions(obs, teacher)
        if len(positions) == args.roots:
            break
    else:
        raise RuntimeError("Could not sample the requested distinct reanalysis roots.")
    del selector
    result = {
        "device": str(device), "torch": torch.__version__, "platform": platform.platform(),
        "profile": args.profile, "roots": args.roots, "sequences": sequences,
        "threads": args.threads, "compile": args.compile,
        "warmup": args.warmup, "repeats": args.repeats,
        "compile_cache_limit": torch._dynamo.config.cache_size_limit,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "replay_device": str(replay._buffer._storage._storage.device),
        "configuration": vars(cfg).copy(), "arms": {},
        "scope": "production LazyReanalyzer.refresh including deduplication, encoding and replay cache writeback; excludes batch sampling and outer update",
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    arms = [] if getattr(args, "graph_only", False) else [("legacy_native", "native", 1), ("teacher_native", "native", 1)]
    if device.type == "cuda":
        arms += [(f"cuda_graph_roots_{width}", "cuda_graph", width) for width in args.parallel_roots]
    native_control = graph_control = None
    saved_solver = None
    for name, backend, width in arms:
        print(json.dumps({"starting_arm": name}), flush=True)
        cfg.reanalyze_backend, cfg.reanalyze_parallel_roots = backend, width
        reanalyzer = LazyReanalyzer(agent)
        if backend == "cuda_graph" and saved_solver is not None:
            saved_solver.parallel_roots = width
            reanalyzer.parallel_solver = saved_solver
        reused_slots = len(saved_solver.runners) if saved_solver is not None else 0
        if name == "legacy_native":
            engine = reanalyzer.engine
            def legacy_teacher(root, *, t0=True):
                _, counters, _ = engine.act(
                    root, t0=t0, eval_mode=True, collect_diagnostics=False,
                    capture_teacher_policy=True, apply_inner_writeback=False,
                )
                return engine.last_teacher_policy, counters
            engine.solve_teacher = legacy_teacher
        outputs, measured = _measure(
            lambda: reanalyzer.refresh(replay, obs, teacher, agent.outer_version),
            device=device, warmup=args.warmup, repeats=args.repeats, roots=args.roots,
        )
        labels, metrics = outputs
        current = tuple(labels[key].detach().cpu().clone() for key in ("mean", "log_std"))
        if name == "legacy_native":
            native_control = current
        elif name == "teacher_native":
            for actual, expected in zip(current, native_control):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            result["native_teacher_bitwise_parity"] = True
        elif graph_control is None:
            graph_control = current
        else:
            for actual, expected in zip(current, graph_control):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            result["graph_pool_width_bitwise_parity"] = True
        measured["last_refresh_metrics"] = metrics
        measured["reused_graph_slots"] = reused_slots
        if backend == "cuda_graph" and getattr(args, "reuse_graph_pool", False):
            saved_solver = reanalyzer.parallel_solver
        result["arms"][name] = measured
        _save_progress(result, getattr(args, "output", None))
        print(json.dumps({"completed_arm": name, **measured}), flush=True)
        del outputs, labels, reanalyzer
        if name == "legacy_native":
            del engine, legacy_teacher
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if "legacy_native" in result["arms"]:
        baseline = result["arms"]["legacy_native"]["median_refresh_seconds"]
        for measured in result["arms"].values():
            measured["speedup_over_legacy_native"] = baseline / measured["median_refresh_seconds"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile", choices=("tiny", "reference"), default="tiny")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--compile-cache-limit", type=int)
    parser.add_argument("--reuse-graph-pool", action="store_true")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="high")
    parser.add_argument("--roots", type=int, default=6)
    parser.add_argument("--parallel-roots", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.roots, args.threads, args.repeats, *args.parallel_roots) <= 0 or args.warmup < 1:
        parser.error("Counts must be positive, including at least one warmup.")
    if args.compile_cache_limit is not None and args.compile_cache_limit < 1:
        parser.error("--compile-cache-limit must be positive.")
    if args.reuse_graph_pool and not (args.full_refresh and args.graph_only):
        parser.error("--reuse-graph-pool requires --full-refresh --graph-only.")
    if args.graph_only:
        if args.device != "cuda":
            parser.error("--graph-only requires CUDA.")
        from RL.tdmpc2_core.parallel_teacher import ParallelTeacherSolver
        make_runner = ParallelTeacherSolver._make_runner

        def observed_runner(self, root):
            started = time.perf_counter()
            runner = make_runner(self, root)
            print(json.dumps({"captured_slot": len(self.runners) + 1,
                              "pool_width": self.parallel_roots,
                              "setup_seconds": time.perf_counter() - started}), flush=True)
            return runner

        ParallelTeacherSolver._make_runner = observed_runner
    result = benchmark(args)
    rendered = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "configuration"}, indent=2))


if __name__ == "__main__":
    main()
