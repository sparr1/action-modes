"""Scheduled timing of frozen Humanoid mean-action evaluation after graph capture.

The evaluation adapter is benchmark-only: production deterministic evaluation
still uses native regions. No training, W&B, traces, or model-return probes run.
Graph and matched uncaptured control use the same isolated RNG protocol, which
differs from the native phase streams. Returns are timing context only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _initialize_frozen_model, _jsonable, _make_env, _outer_state_digest,
)
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph
from tests.benchmark_whole_inner_graph import assert_tree_close, materialize, snapshot
from utils.ambi_benchmark import solver_seed

ROOT = Path(__file__).resolve().parents[1]


class MeanActionGraph(WholeInnerGraph):
    def _solve(self):
        return self._engine_act(self.root, collect_diagnostics=False, eval_mode=True)


def make_model(checkpoint):
    run = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_h3_j10_k1_reward_only_graph.json").read_text())
    run["alg_params"].update(wandb=False, wandb_mode="disabled", inner_cuda_graph=False)
    resolved = {
        "selector": "benchmark/J10-K1-H3-return",
        "algorithm_config": run,
        "environment": {"id": "DMControl-v0", "params": {
            "task": "humanoid-walk", "obs": "state", "render_mode": None,
        }},
    }
    env = _make_env(resolved)
    model, _ = _initialize_frozen_model(resolved, env, checkpoint, 55, device="cuda")
    cfg = model.cfg
    assert (cfg.inner_rounds, cfg.inner_solve_interval, cfg.inner_rollout_horizon) == (10, 1, 3)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (160, 40)
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert cfg.aux_return_mode == "off" and cfg.wandb is False
    return model, env


def root_for(model, observation):
    with torch.no_grad():
        obs = model._obs_to_tensor(observation).to(model.agent.device).unsqueeze(0)
        return model.agent.model.encode(obs).detach()


def install_adapter(model, runner):
    def act(root, *, eval_mode=False, collect_diagnostics=True, trace=None, **kwargs):
        assert eval_mode and not collect_diagnostics and trace is None
        assert not kwargs.get("apply_inner_writeback", False)
        assert not kwargs.get("return_behavior_policy", False)
        action, metrics, lengths = runner(root)
        return action, dict(metrics, inner_cuda_graph_used=float(runner.capture_enabled)), lengths
    model.agent.inner_engine.act = act


def episode(model, env, *, seed, steps, diagnostics, runner=None):
    from torch._dynamo.utils import counters
    episode_seed = solver_seed(55, "episode", seed)
    model.agent.inner_engine.reset_for_evaluation(episode_seed, reuse_action_pool=True)
    if runner is not None:
        runner.seed = episode_seed
        runner.reset_rng()
    obs, _ = env.reset(seed=seed)
    before = int(counters["stats"]["unique_graphs"])
    torch.cuda.synchronize()
    started, cpu_start = time.perf_counter(), time.process_time()
    latencies, environment_seconds, total_return = [], 0.0, 0.0
    terminated = truncated = False
    for index in range(steps):
        t = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True, episode_start=index == 0,
                                  collect_diagnostics=diagnostics)
        latencies.append(time.perf_counter() - t)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_critic_optimizer_steps"] == 160
        assert metrics["inner_actor_optimizer_steps"] == 40
        assert metrics["inner_model_steps"] == 3840
        assert metrics["inner_compile_fallback"] == 0
        assert np.isfinite(action).all()
        t = time.perf_counter()
        obs, reward, terminated, truncated, _ = env.step(action)
        environment_seconds += time.perf_counter() - t
        total_return += float(reward)
        if terminated or truncated:
            break
    torch.cuda.synchronize()
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_start
    after = int(counters["stats"]["unique_graphs"])
    assert before == after, "Compilation occurred during a timed episode."
    assert steps != 500 or len(latencies) == 500, "Expected a complete 500-decision episode."
    result = {
        "seed": seed, "decisions": len(latencies), "seconds": elapsed,
        "seconds_per_decision": elapsed / len(latencies),
        "control_seconds": sum(latencies), "environment_seconds": environment_seconds,
        "control_median_ms": statistics.median(latencies) * 1000,
        "control_p95_ms": float(np.percentile(latencies, 95)) * 1000,
        "process_cpu_seconds": cpu, "average_cpu_cores": cpu / elapsed,
        "graphs_before": before, "graphs_after": after,
        "terminated": terminated, "truncated": truncated,
        "return_timing_context_only": total_return,
        "blocks_seconds_per_decision": [statistics.mean(latencies[i:i+50])
                                        for i in range(0, len(latencies), 50)],
    }
    print("MEASUREMENT " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-steps", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    assert os.environ.get("WANDB_MODE") == "disabled"
    assert torch.cuda.is_available() and os.environ.get("SLURM_JOB_ID")
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    process_started = time.perf_counter()
    models, envs, digests, setup = {}, {}, {}, {}
    for name in ("native", "control", "graph"):
        t = time.perf_counter()
        models[name], envs[name] = make_model(args.checkpoint)
        digests[name] = _outer_state_digest(models[name])
        setup[name] = {"initialization_seconds": time.perf_counter() - t}
    assert len(set(digests.values())) == 1
    observations = [envs["native"].reset(seed=seed)[0] for seed in (101, 102)]
    native = models["native"]
    # Warm both diagnostics modes so measured windows contain no compilation.
    for diagnostics in (True, False):
        t = time.perf_counter()
        for index in range(4):
            native.predict(observations[index % 2], deterministic=True,
                           episode_start=index == 0, collect_diagnostics=diagnostics)
        setup["native"][f"warmup_diagnostics_{diagnostics}_seconds"] = time.perf_counter() - t
    runners = {}
    for name in ("control", "graph"):
        worker = InnerImprovementEngine(models[name].agent)
        runners[name] = MeanActionGraph(worker, root_for(models[name], observations[0]),
                                        capture=name == "graph", seed=8173, warmup=4)
        setup[name].update(warmup_seconds=runners[name].warmup_seconds,
                           capture_seconds=runners[name].capture_seconds)
    print("SETUP " + json.dumps(setup), flush=True)
    parity = []
    for observation in observations:
        outputs, states = {}, {}
        for name, runner in runners.items():
            model = models[name]
            root = root_for(model, observation)
            outside = torch.cuda.get_rng_state().clone()
            outside_cpu = torch.random.get_rng_state().clone()
            before_rng = runner.rng_state.clone()
            output = materialize(model.agent, runner(root))
            outputs[name] = output
            states[name] = snapshot(runner.engine)
            with torch.no_grad():
                expected_mean, _ = model.agent.model.pi(
                    root, policy=runner.engine._action_pool.actor, deterministic=True)
            torch.testing.assert_close(output[0], expected_mean[0].cpu())
            assert not torch.equal(before_rng, runner.rng_state)
            torch.testing.assert_close(outside, torch.cuda.get_rng_state(), rtol=0, atol=0)
            torch.testing.assert_close(outside_cpu, torch.random.get_rng_state(), rtol=0, atol=0)
        assert_tree_close(outputs["control"], outputs["graph"])
        assert_tree_close(states["control"], states["graph"])
        parity.append({"action_optimizer_replay_state_match": True, "executes_actor_mean": True,
                       "private_rng_advances": True, "global_rng_unchanged": True})
    for name, runner in runners.items():
        install_adapter(models[name], runner)
    measurements = {"native_default_diagnostics": [], "native_minimal": [],
                    "matched_uncaptured_minimal": [], "graph_minimal": []}
    for repeat in range(2):
        for label, diagnostics in (("native_default_diagnostics", True), ("native_minimal", False)):
            print("ARM " + label, flush=True)
            measurements[label].append(episode(native, envs["native"], seed=101+repeat,
                                               steps=args.native_steps, diagnostics=diagnostics))
    measurements["matched_uncaptured_minimal"].append(episode(
        models["control"], envs["control"], seed=101, steps=args.native_steps,
        diagnostics=False, runner=runners["control"]))
    for repeat in range(args.episodes):
        print("ARM graph_full_episode", flush=True)
        measurements["graph_minimal"].append(episode(
            models["graph"], envs["graph"], seed=101+repeat, steps=500,
            diagnostics=False, runner=runners["graph"]))
    for name, model in models.items():
        assert _outer_state_digest(model) == digests[name], "Frozen outer state changed."
    per_decision = {name: sum(x["seconds"] for x in rows) / sum(x["decisions"] for x in rows)
                    for name, rows in measurements.items()}
    report = {
        "passed": True, "wandb": False, "benchmark_only_adapter": True,
        "production_deterministic_eval_uses_whole_graph": False,
        "scope": "frozen closed-loop mean-action controller including real Humanoid environment",
        "excluded": ["inner traces", "model-return probes", "publication", "queue time"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(Path(str(args.checkpoint)+".metadata.json").read_bytes()).hexdigest(),
        "runtime": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "host": platform.node(),
                    "job": os.environ["SLURM_JOB_ID"], "threads": 1,
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
        "resolved_config": _jsonable(vars(models["graph"].cfg)),
        "setup": setup, "parity": parity, "outer_state_unchanged": True,
        "measurements": measurements, "seconds_per_decision": per_decision,
        "projected_500_decision_episode_seconds": {k: v * 500 for k, v in per_decision.items()},
        "speedup_vs_native_minimal": per_decision["native_minimal"] / per_decision["graph_minimal"],
        "speedup_vs_native_default_diagnostics": per_decision["native_default_diagnostics"] / per_decision["graph_minimal"],
        "process_total_seconds": time.perf_counter() - process_started,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "rng_note": "Graph differs from native phase-seeded RNG; only graph versus matched control is numerical parity.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print("RESULT " + json.dumps({k: report[k] for k in (
        "passed", "seconds_per_decision", "projected_500_decision_episode_seconds",
        "speedup_vs_native_minimal", "process_total_seconds")}), flush=True)
    for name in models:
        models[name].close()
        envs[name].close()


if __name__ == "__main__":
    main()
