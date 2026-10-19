"""Bounded PAVE off/separate/fused complete-update timing on resident batches.

For example, on an allocated GPU::

    python -m tests.benchmark_pave_views --device cuda --compile \
        --pave-rows 768 --output /scratch/pave-views.json

Paired synthetic inputs, initialization and RNG; alternating case order. Times
include the public learner update and synchronization, but exclude warmup,
compilation, validation, replay I/O, environment interaction and logging.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from RL.tdmpc2_core import pave
from tests.benchmark_aux_return_joint_vmap import make_agent
from tests.pave_reference import separate_critic_pave_loss


@contextmanager
def implementation(mode):
    original = pave.critic_pave_loss
    try:
        if mode == "separate":
            pave.critic_pave_loss = separate_critic_pave_loss
        yield
    finally:
        pave.critic_pave_loss = original


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure(args, mode):
    agent = make_agent(args, joint=True)
    agent.cfg.outer_pave_batch_size = args.pave_rows
    prefixes = ("outer_pave", "aux_return_pave") if args.critics == "both" else ("aux_return_pave",)
    for prefix in prefixes:
        for term, weight in (("mpr", .005), ("vfc", .005), ("curvature", .0005)):
            setattr(agent.cfg, f"{prefix}_{term}_coef", 0. if mode == "off" else weight)
    return agent


def measure(agent, batch, args, mode):
    device = torch.device(agent.device)
    for _ in range(args.warmup):
        agent._update(*batch)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from torch._dynamo.utils import counters
    graphs_before = int(counters["stats"]["unique_graphs"])
    samples = []
    for _ in range(args.steps):
        synchronize(device)
        started = time.perf_counter()
        info = agent._update(*batch)
        synchronize(device)
        samples.append(1000 * (time.perf_counter() - started))
    graphs_after = int(counters["stats"]["unique_graphs"])
    assert graphs_after == graphs_before, "Compilation during timed updates; increase warmup."
    assert not info["compile_fallback"]
    assert ("aux_return_pave_weighted_loss" in info) == (mode != "off")
    assert info["aux_return_joint_vmap_online"] == info["aux_return_joint_vmap_target"] == 1.
    if mode != "off":
        assert info["aux_return_pave_sample_count"] == min(args.pave_rows, args.batch_size * args.horizon)
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values())
    assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
    if args.compile:
        assert agent._outer_update_region._compiled is not None
    return dict(
        median_ms=statistics.median(samples), mean_ms=statistics.mean(samples), samples_ms=samples,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        unique_graphs_before_timing=graphs_before, unique_graphs_after_timing=graphs_after,
        compile_fallback=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pave-rows", type=int, default=768)
    parser.add_argument("--critics", choices=("auxiliary", "both"), default="auxiliary")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.repeats, args.threads, args.pave_rows) < 1:
        parser.error("Warmup, steps, repeats, threads and PAVE rows must be positive.")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing result.")
    args.aux_mode, args.operator, args.representation = "sac", "sac", "distributional"
    args.heads, args.bins, args.horizon, args.dropout = 5, 101, 3, .01
    args.obs_dim, args.action_dim = 67, 21
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(12345)
    batch = (
        torch.randn(4, args.batch_size, 67, generator=generator).to(device),
        torch.randn(3, args.batch_size, 21, generator=generator).tanh().to(device),
        torch.randn(3, args.batch_size, 1, generator=generator).to(device),
        torch.zeros(3, args.batch_size, 1, device=device),
    )
    torch.manual_seed(9187)
    source = configure(args, "fused")
    with torch.no_grad():
        for name in ("_Qs", "_target_Qs", "_aux_return_Qs", "_target_aux_return_Qs"):
            for head in getattr(source.model, name):
                head[-1].weight.normal_(std=.02)
    initial = deepcopy(source.checkpoint_state())
    del source
    rows = []
    cases = ("off", "separate", "fused")
    for repeat in range(args.repeats):
        order = cases[repeat % 3:] + cases[:repeat % 3]
        for mode in order:
            torch._dynamo.reset()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            torch.manual_seed(1771)
            agent = configure(args, mode)
            agent.load(deepcopy(initial))
            with implementation(mode):
                row = dict(repeat=repeat, mode=mode, **measure(agent, batch, args, mode))
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "samples_ms"}), flush=True)
            del agent
    summary = {mode: statistics.median(row["median_ms"] for row in rows if row["mode"] == mode)
               for mode in cases}
    pairs = []
    for repeat in range(args.repeats):
        values = {row["mode"]: row["median_ms"] for row in rows if row["repeat"] == repeat}
        pairs.append(dict(repeat=repeat, fused_speedup=values["separate"] / values["fused"],
                          fused_time_saved_percent=100 * (1 - values["fused"] / values["separate"]),
                          separate_overhead_ms=values["separate"] - values["off"],
                          fused_overhead_ms=values["fused"] - values["off"]))
    root = Path(__file__).resolve().parents[1]
    files = ["RL/tdmpc2_core/pave.py", "RL/tdmpc2_core/common/ensemble_vmap.py",
             "RL/tdmpc2_core/ambi_agent.py", "tests/pave_reference.py", "tests/benchmark_pave_views.py"]
    result = dict(
        schema="pave-view-batching-timing-v1", complete=True,
        torch=torch.__version__, device=args.device,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        protocol={**vars(args), "output": str(args.output), "source": "synthetic_resident_sequences"},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
        rows=rows, median_update_ms=summary, paired_comparisons=pairs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"median_update_ms": summary, "paired_comparisons": pairs}), flush=True)


if __name__ == "__main__":
    main()
