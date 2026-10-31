"""Compare native compiled regions, a graph-ready control, and one CUDA graph.

All arms use identical fixed roots, initial outer weights and inner work. The
experimental arms share a graph-private RNG protocol; native phase RNG ordering
is intentionally different. Timings include fresh initialization and ordinary
metrics, but exclude environment interaction and outer learning.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import gymnasium as gym
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph, disable_phase_timers

ROOT = Path(__file__).resolve().parents[1]


def make_agent(*, device="cuda", tiny=False, compile_enabled=True):
    run = json.loads((ROOT / "configs/research/ambi_whole_graph_benchmark.json").read_text())
    params = deepcopy(run["alg_params"])
    params.update(device=device, compile=compile_enabled, compile_strict=compile_enabled)
    if tiny:
        params.update(model_size=None, enc_dim=16, mlp_dim=16, latent_dim=8,
                      num_enc_layers=2, simnorm_dim=4, num_bins=5, q_num_bins=5,
                      num_q=2, inner_rounds=2, inner_rollouts_per_round=4,
                      inner_batch_size=4, inner_replay_capacity=24,
                      inner_critic_updates_per_round=2, inner_actor_updates_per_round=1)
    builder = object.__new__(AMBITDMPC2)
    builder.env = gym.make("Pendulum-v1")
    builder.run_params = {**run, "seed": 55, "device": device}
    builder.custom_params = params
    try:
        cfg = builder._build_cfg(params)
    finally:
        builder.env.close()
    obs_dim, action_dim = (3, 1) if tiny else (67, 21)
    cfg.obs_shape = {"state": (obs_dim,)}
    cfg.obs_shapes = [cfg.obs_shape]
    cfg.action_dim, cfg.action_dims, cfg.device = action_dim, [action_dim], device
    cfg.target_entropy = -0.5 * action_dim
    torch.manual_seed(55)
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


def snapshot(engine):
    pool = engine._action_pool
    result = {}
    for name in ("actor", "critic", "critic_target"):
        module = getattr(pool, name)
        result[name] = {key: tensor.detach().clone() for key, tensor in module.state_dict().items()}
    result["alpha"] = pool.log_alpha.detach().clone()
    result["replay"] = pool.replay._storage[:pool.replay.size].detach().clone()
    result["optimizers"] = {
        name: deepcopy(getattr(pool, name).state_dict())
        for name in ("actor_optim", "critic_optim", "temperature_optim")
    }
    return result


def assert_tree_close(a, b, *, rtol=2e-5, atol=2e-6):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_close(a[key], b[key], rtol=rtol, atol=atol)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_close(left, right, rtol=rtol, atol=atol)
    else:
        assert a == b


def materialize(agent, output):
    # Same unavoidable CPU action/metric boundary as the native agent.
    return agent._materialize_action_metrics(*output)


def measure(operation, roots, steps):
    from torch._dynamo.utils import counters
    torch.cuda.synchronize()
    before = int(counters["stats"]["unique_graphs"])
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for index in range(steps):
        operation(roots[index % len(roots)])
    torch.cuda.synchronize()
    seconds, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
    after = int(counters["stats"]["unique_graphs"])
    assert before == after, "Compilation occurred inside a timed window."
    return dict(seconds=seconds, solves_per_second=steps / seconds,
                milliseconds_per_solve=1000 * seconds / steps,
                process_cpu_seconds=cpu, average_cpu_cores=cpu / seconds,
                graphs_before=before, graphs_after=after)


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a scheduled CUDA allocation.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    agents = {name: make_agent(tiny=args.tiny, compile_enabled=not args.eager)
              for name in ("native_regions", "graph_ready_eager", "whole_graph")}
    generator = torch.Generator(device="cpu").manual_seed(1947)
    # Valid SimNorm latents, fixed across every arm and repeated measurement.
    dim = agents["whole_graph"].cfg.latent_dim
    group = agents["whole_graph"].cfg.simnorm_dim
    roots = torch.softmax(torch.randn(4, 1, dim // group, group, generator=generator), -1).reshape(4, 1, dim).cuda()
    outer_before = {name: deepcopy(agent.model.state_dict()) for name, agent in agents.items()}
    alpha_before = {name: agent.log_ent_coef.detach().clone() for name, agent in agents.items()}
    engine = agents["native_regions"].inner_engine
    disable_phase_timers(engine)
    operations = {"native_regions": lambda root: engine.act(root, collect_diagnostics=False)}
    setup = {}
    started = time.perf_counter()
    for _ in range(args.warmup):
        materialize(agents["native_regions"], operations["native_regions"](roots[0]))
    setup["native_regions"] = {"warmup_seconds": time.perf_counter() - started}
    runners = {}
    for name in ("graph_ready_eager", "whole_graph"):
        print("PREPARE " + name, flush=True)
        runner = WholeInnerGraph(agents[name].inner_engine, roots[0], capture=name == "whole_graph",
                                 warmup=args.warmup)
        runners[name] = runner
        operations[name] = runner
        setup[name] = dict(warmup_seconds=runner.warmup_seconds, capture_seconds=runner.capture_seconds)
    # Paired numerical gate includes changing roots and changing outer priors.
    checks = []
    for index in range(3):
        if index == 1:
            with torch.no_grad():
                for name in runners:
                    agent = agents[name]
                    for parameter in agent.model.parameters():
                        parameter.add_(0.0001)
                    agent.log_ent_coef.add_(0.03)
        captured = {}
        states = {}
        outside = torch.cuda.get_rng_state().clone()
        for name, runner in runners.items():
            expected_outer = deepcopy(agents[name].model.state_dict())
            output = runner(roots[index])
            captured[name] = deepcopy(materialize(agents[name], output))
            states[name] = snapshot(agents[name].inner_engine)
            torch.testing.assert_close(torch.cuda.get_rng_state(), outside, rtol=0, atol=0)
            assert_tree_close(agents[name].model.state_dict(), expected_outer, rtol=0, atol=0)
            assert all(p.grad is None for p in agents[name].model.parameters())
        assert_tree_close(captured["graph_ready_eager"], captured["whole_graph"])
        assert_tree_close(states["graph_ready_eager"], states["whole_graph"])
        checks.append({"root": index, "action_state_optimizer_replay_parity": True})
    for name, agent in agents.items():
        # Restore exactly the same outer weights for throughput measurement.
        agent.model.load_state_dict(outer_before[name])
        if name in runners:
            with torch.no_grad():
                agent.log_ent_coef.copy_(alpha_before[name])
    for name in runners:
        runners[name].reset_rng()
    # Both controls must do the complete configured work, with no fallback.
    cfg = agents["whole_graph"].cfg
    expected = {"inner_critic_optimizer_steps": cfg.inner_critic_updates_per_action,
                "inner_actor_optimizer_steps": cfg.inner_actor_updates_per_action,
                "inner_model_steps": cfg.inner_model_step_budget}
    for name, operation in operations.items():
        action, metrics, lengths = materialize(agents[name], operation(roots[0]))
        assert torch.isfinite(action).all()
        assert metrics["inner_compile_fallback"] == 0
        for key, value in expected.items():
            assert metrics[key] == value, (name, key, metrics[key], value)
    measurements = {name: [] for name in agents}
    for repeat in range(args.repeats):
        names = list(agents) if repeat % 2 == 0 else list(reversed(agents))
        for name in names:
            def complete(root, name=name):
                return materialize(agents[name], operations[name](root))
            result = measure(complete, roots, args.steps)
            measurements[name].append(result)
            print("MEASUREMENT " + json.dumps({"arm": name, "repeat": repeat, **result}), flush=True)
    median_ms = {name: statistics.median(row["milliseconds_per_solve"] for row in rows)
                 for name, rows in measurements.items()}
    report = {
        "passed": True, "scope": "complete_inner_solve_plus_action_and_metrics_host_transfer",
        "one_cuda_graph_per_solve": True, "compiler_graph_is_monolithic": False,
        "rng_protocol": "isolated_default_cuda_stream_for_graph_and_matched_control",
        "native_rng_sequence_preserved": False,
        "wandb": False, "compile": not args.eager, "tiny": args.tiny,
        "steps_per_repeat": args.steps, "repeats": args.repeats, "threads": args.threads,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "host": platform.node(),
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
        "workload": {"rounds": cfg.inner_rounds, "horizon": cfg.inner_rollout_horizon,
                     "rollouts_per_round": cfg.inner_rollouts_per_round,
                     "batch_size": cfg.inner_batch_size, **expected},
        "setup": setup, "validation": checks, "measurements": measurements,
        "median_milliseconds_per_solve": median_ms,
        "speedup_vs_native_regions": median_ms["native_regions"] / median_ms["whole_graph"],
        "speedup_vs_graph_ready_control": median_ms["graph_ready_eager"] / median_ms["whole_graph"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print("RESULT " + json.dumps({key: value for key, value in report.items()
                                 if key not in ("measurements", "validation", "setup")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    assert args.steps > 0 and args.repeats > 0 and args.warmup >= 3
    benchmark(args)


if __name__ == "__main__":
    main()
