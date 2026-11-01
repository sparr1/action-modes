"""Scheduled CUDA correctness gate and bounded J10/K1 regularizer benchmark.

Run with ``python -m tests.benchmark_pave_graph --term mpr --graph --output ...``.
The geometry comparison uses identical weights and inputs. Real-training timing
uses the production K1 recipe with short replay/pretraining and W&B disabled;
it is not a learning-quality comparison or a mature-replay runtime estimate.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from RL.tdmpc2_core.pave import outer_pave_loss
from RL.tdmpc2_core.pave_graph import PaveGraph
from slurm.ambi_mppi_training import smoke_recipe
from slurm.ambi_reward_training import GRAPH_MANIFEST, SEED_STEPS, validate_action
from slurm.benchmark_ambi_training_cpu import TrainingWindow
from tests.benchmark_whole_inner_graph import make_agent, assert_tree_close
from tests.test_training_inner_graph import assert_graph_resume_continuation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dmcontrol/algs/ambi_full_h3_j10_k1_reward_only_graph.json"


def controls(term, graph):
    return {f"outer_pave_{term}_coef": .004, "outer_pave_batch_size": None,
            "outer_pave_obs_noise_std": .01, "outer_pave_action_noise_std": .02,
            "outer_pave_cuda_graph": graph, "wandb": False, "wandb_mode": "disabled"}


def geometry_gate(term):
    agent = make_agent(config=CONFIG)
    for key, value in controls(term, True).items():
        setattr(agent.cfg, key, value)
    generator = torch.Generator().manual_seed(747)
    with torch.no_grad():
        for head in agent.model._Qs:
            head[-1].weight.copy_(torch.randn(head[-1].weight.shape, generator=generator) * .025)
            head[-1].bias.copy_(torch.randn(head[-1].bias.shape, generator=generator) * .01)
    parameters = tuple(agent.model._Qs.parameters())
    obs = torch.randn(4, 256, 67, generator=generator, device="cpu").cuda()
    action = torch.randn(3, 256, 21, generator=generator, device="cpu").tanh().cuda()
    # Exercise clamping and noncontiguous replay inputs as well as normal rows.
    action[:, :2] = 1
    action[:, 2:4] = -1
    action = action.transpose(0, 1).contiguous().transpose(0, 1)
    runner = PaveGraph(agent.model, agent.cfg)

    def eager(index):
        return outer_pave_loss(agent.model, obs, action, agent.cfg, index)

    def captured(index):
        return runner(obs, action, index)

    max_error = 0.
    for index in range(1, 4):
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        loss, metrics = eager(index)
        expected = torch.autograd.grad(loss, parameters, allow_unused=True)
        result, info = captured(index)
        assert info.pop("outer_pave_cuda_graph_used") == 1
        actual = torch.autograd.grad(result, parameters, allow_unused=True)
        assert_tree_close((result, info, actual), (loss, metrics, expected), rtol=3e-5, atol=2e-7)
        assert metrics["outer_pave_sample_count"] == 768
        assert metrics[f"outer_pave_{term}_loss"] > 0
        for got, wanted in zip(actual, expected):
            if got is not None:
                max_error = max(max_error, float((got - wanted).abs().max()))
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        with torch.no_grad():
            parameters[0].add_(.0001)
            obs.add_(.07)

    def step(function, index):
        value, _ = function(index)
        return torch.autograd.grad(value, parameters, allow_unused=True)

    for function in (eager, captured):
        for index in range(5):
            step(function, index + 20)
    timings = {"eager": [], "graph": []}
    for repeat in range(4):
        for name in (("eager", "graph") if repeat % 2 == 0 else ("graph", "eager")):
            function = eager if name == "eager" else captured
            torch.cuda.synchronize()
            start = time.perf_counter()
            for index in range(40):
                step(function, 100 + index)
            torch.cuda.synchronize()
            timings[name].append(1000 * (time.perf_counter() - start) / 40)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {"passed": True, "rows": 768, "heads": 5, "q_bins": agent.cfg.q_num_bins,
            "max_absolute_gradient_error": max_error, "milliseconds_per_loss_and_gradient": timings,
            "median_milliseconds": medians, "speedup": medians["eager"] / medians["graph"],
            "scope": "Preparation, encoding, loss, gradients, graph validation and output ownership; no optimizer."}


def training_gate(args):
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    name, manifest, run, overrides = smoke_recipe(0, args.warmup, args.decisions, GRAPH_MANIFEST)
    run["alg_params"].update(controls(args.term, args.graph))
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        learner = AMBITDMPC2(name + "-pave-correctness", env, run["alg_params"], run, manifest)
        agent, cfg = learner.agent, learner.cfg
        assert not cfg.wandb and cfg.wandb_mode == "disabled"
        assert cfg.inner_cuda_graph and cfg.inner_solve_interval == 1 and cfg.inner_rounds == 10
        assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
        assert cfg.outer_pave_batch_size is None and cfg.batch_size == 256 and cfg.train_unroll_horizon == 3
        learner.buffer.enable_resumable_storage()
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log = learner._act_agent, agent.update, learner._log_wandb_step
        active, updates, positive = 0, 0, 0
        counts = {}

        def observed_act(*positional, **kwargs):
            nonlocal active
            step = learner._global_step + 1
            if step == timer.first_step:
                counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before_action(step, learner._num_updates)
            started = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - started)
            validate_action(agent.last_inner_metrics, operator="sac", decision=active, interval=1, rounds=10)
            assert agent.last_inner_metrics["inner_cuda_graph_used"] == 1
            assert torch.isfinite(action).all()
            active += 1
            return action

        def observed_update(*positional, **kwargs):
            nonlocal updates, positive
            metrics = update(*positional, **kwargs)
            assert metrics["outer_pave_sample_count"] == 768
            assert metrics.get("outer_pave_cuda_graph_used", 0) == int(args.graph)
            assert metrics["compile_fallback"] == 0
            assert all(math.isfinite(float(value)) for value in metrics.values())
            positive += int(metrics[f"outer_pave_{args.term}_loss"] > 0)
            updates += 1
            return metrics

        def observed_log(reward, *positional, **kwargs):
            result = log(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log
        learner.learn(run["total_steps"])
        torch.cuda.synchronize()
        counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert counts["after"] == counts["before"], "Compilation occurred in timing window."
        assert active == args.warmup + args.decisions
        assert updates == agent.num_updates == 2 + active and positive > 0
        assert timer.result and timer.result["outer_updates"] == args.decisions
        # The inner graph must remain correct after regularized outer updates,
        # including native diagnostics, changed priors, and recapture on resume.
        assert_graph_resume_continuation(agent)
        # Exact outer continuation at production shapes, after recapturing both
        # subsystems. Replay sampling is outside this agent-state boundary.
        data = learner.buffer.sample()[:4]
        agent.prepare_training_resume_boundary()
        saved = deepcopy(agent.training_state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        expected = agent._update(*data)
        expected_state = deepcopy(agent.checkpoint_state())
        expected_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        agent.load_training_state_dict(saved)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        actual = agent._update(*data)
        assert_tree_close((actual, agent.checkpoint_state()), (expected, expected_state), rtol=0, atol=0)
        assert_tree_close((torch.get_rng_state(), torch.cuda.get_rng_state()), expected_rng, rtol=0, atol=0)
        return {"passed": True, "timing": timer.result, "compile_counts": counts,
                "nonzero_regularizer_updates": positive, "outer_updates": updates,
                "inner_and_outer_exact_continuation": True,
                "overrides": {**overrides, **controls(args.term, args.graph), "total_steps": run["total_steps"]},
                "scope": "Short real training; excludes startup, W&B, checkpoints, mature replay and production pretraining."}
    finally:
        if learner is not None:
            learner.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", choices=("mpr", "action_smoothness"), required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--geometry", action="store_true", help="Run production-shape geometry comparison before training.")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--decisions", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.warmup >= 12 and args.decisions >= 32
    assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000
    assert torch.cuda.is_available(), "Requires a scheduled CUDA allocation."
    os.environ["WANDB_MODE"] = "disabled"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "term": args.term, "graph": args.graph, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda}
    if args.geometry:
        report["geometry"] = geometry_gate(args.term)
        print("GEOMETRY_PASSED " + json.dumps(report["geometry"]), flush=True)
    report["training"] = training_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print("PAVE_GRAPH_TRAINING_PASSED " + json.dumps({key: report[key] for key in ("term", "graph")}), flush=True)


if __name__ == "__main__":
    main()
