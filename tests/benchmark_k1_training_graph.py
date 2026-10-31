"""Short real Humanoid K1 training comparison, including outer learning.

Each worker starts from seed 55, completes a seed episode, then measures full
decisions after active warmup. Graph execution changes the inner RNG sequence,
so this is a throughput comparison, not a return or exact-continuation study.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import gymnasium as gym
import torch
import domains  # noqa: F401

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph, disable_phase_timers

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/research/ambi_whole_graph_k1_reward_benchmark.json"
SEED_STEPS = 500


def recipe(warmup, decisions):
    run = json.loads(RECIPE.read_text())
    overrides = dict(wandb=False, wandb_mode="disabled", seed_steps=SEED_STEPS,
                     pretrain_steps=2, buffer_size=4096)
    run["alg_params"].update(overrides)
    run.update(total_steps=SEED_STEPS + 1 + warmup + decisions,
               checkpoint_every=None, save_strat="none")
    manifest = dict(env_params=dict(task="humanoid-walk", obs="state", render_mode=None),
                    logs="none", save_trials="none", checkpoint_every=None, save_strat="none")
    return run, manifest, overrides


class DecisionTimer:
    def __init__(self, first, count, synchronize):
        self.first, self.last = first, first + count - 1
        self.synchronize = synchronize
        self.seconds, self.inner_seconds, self.updates = [], [], []

    def before(self, step, updates):
        if step == self.first:
            self.synchronize()
            self.start = self.previous = time.perf_counter()
            self.cpu_start = time.process_time()
            self.first_updates = updates

    def after_action(self, step, elapsed):
        if self.first <= step <= self.last:
            self.inner_seconds.append(elapsed)

    def after(self, step, updates):
        if not self.first <= step <= self.last:
            return
        if step == self.last:
            self.synchronize()
        now = time.perf_counter()
        self.seconds.append(now - self.previous)
        self.previous = now
        self.updates.append(updates - self.first_updates)
        assert self.updates[-1] == len(self.seconds), "Expected one outer update per measured decision."
        if step == self.last:
            self.elapsed = now - self.start
            self.cpu = time.process_time() - self.cpu_start


def benchmark(args):
    assert torch.cuda.is_available() and os.environ.get("SLURM_JOB_ID")
    assert args.warmup >= 8 and args.decisions >= 16
    assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    run, manifest, overrides = recipe(args.warmup, args.decisions)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = AMBITDMPC2("k1-graph-throughput", env, run["alg_params"], run, manifest)
    try:
        cfg, agent, engine = learner.cfg, learner.agent, learner.agent.inner_engine
        assert (cfg.inner_rounds, cfg.inner_rollout_horizon) == (8, 3)
        assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size) == (128, 256)
        assert cfg.inner_critic_updates_per_action == 128 and cfg.inner_actor_updates_per_action == 32
        assert cfg.aux_return_mode == "off" and agent.aux_return is None
        assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
        assert getattr(cfg, "inner_solve_interval", 1) == 1
        assert cfg.compile and cfg.compile_strict and not cfg.wandb
        assert not learner._stores_behavior_policy
        learner.buffer.enable_resumable_storage()
        disable_phase_timers(engine)
        timer = DecisionTimer(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        original_act, original_update, original_log = learner._act_agent, agent.update, learner._log_wandb_step
        runner, active, counts = None, 0, {}
        setup = {}

        def graph_act(root, **kwargs):
            assert not kwargs.get("eval_mode") and not kwargs.get("return_behavior_policy")
            assert kwargs.get("trace") is None
            assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
            return runner(root)

        def observed_act(obs, **kwargs):
            nonlocal runner, active
            if args.arm == "whole_graph" and runner is None:
                was_training = agent.model.training
                agent.model.eval()
                try:
                    with torch.no_grad():
                        root = agent.model.encode(obs.to(agent.device).unsqueeze(0)).detach()
                    runner = WholeInnerGraph(engine, root, warmup=4)
                    setup.update(warmup_seconds=runner.warmup_seconds, capture_seconds=runner.capture_seconds)
                    engine.act = graph_act
                finally:
                    agent.model.train(was_training)
            step = learner._global_step + 1
            if step == timer.first:
                counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before(step, learner._num_updates)
            start = time.perf_counter()
            action = original_act(obs, **kwargs)
            timer.after_action(step, time.perf_counter() - start)
            metrics = agent.last_inner_metrics
            assert action.device.type == "cpu" and torch.isfinite(action).all()
            assert metrics["inner_critic_optimizer_steps"] == 128
            assert metrics["inner_actor_optimizer_steps"] == 32
            assert metrics["inner_compile_fallback"] == 0
            assert all(math.isfinite(float(value)) for value in metrics.values())
            active += 1
            return action

        def observed_update(*positional, **kwargs):
            metrics = original_update(*positional, **kwargs)
            assert metrics["compile_fallback"] == 0
            return metrics

        def observed_log(*positional, **kwargs):
            result = original_log(*positional, **kwargs)
            timer.after(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log
        learner.learn(run["total_steps"])
        counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert counts["before"] == counts["after"], "Compilation occurred during measurement."
        assert active == args.warmup + args.decisions
        assert len(timer.seconds) == len(timer.inner_seconds) == args.decisions
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        assert learner.buffer.resumable_storage and str(learner.buffer._storage_device) == "cpu"
        assert all(math.isfinite(float(v)) for v in learner._last_train_metrics.values())
        report = dict(
            passed=True, arm=args.arm, scope="real_training_decisions_environment_replay_outer_updates",
            measured_decisions=args.decisions, measured_outer_updates=timer.updates[-1],
            measured_inner_solves=args.decisions, warmup_active_decisions=args.warmup,
            wall_seconds=timer.elapsed, decisions_per_second=args.decisions / timer.elapsed,
            milliseconds_per_decision=timer.elapsed * 1000 / args.decisions,
            mean_inner_milliseconds=statistics.mean(timer.inner_seconds) * 1000,
            process_cpu_seconds=timer.cpu, average_cpu_cores=timer.cpu / timer.elapsed,
            per_decision_seconds=timer.seconds, per_inner_seconds=timer.inner_seconds,
            compile_counts=counts, setup=setup, replay_storage_device="cpu",
            config_sha256=hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
            workload=dict(K=1, J=8, H=3, N=128, B=256, critic_updates=128, actor_updates=32,
                          outer_target="reward_only", auxiliary_critic=False),
            benchmark_overrides=overrides, wandb=False, checkpointing=False,
            native_rng_sequence_preserved=args.arm == "native_regions",
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0), host=platform.node(),
                         job_id=os.environ["SLURM_JOB_ID"],
                         commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()),
            limitations=["Short early-training window with 500 seed decisions and two pretraining updates.",
                         "Excludes production checkpoint/logging costs and mature replay.",
                         "Graph path uses an isolated RNG sequence; rewards are not a paired scientific comparison."],
        )
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
        print("RESULT " + json.dumps({k: v for k, v in report.items()
                                     if k not in ("per_decision_seconds", "per_inner_seconds")}), flush=True)
    finally:
        learner.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("native_regions", "whole_graph"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--decisions", type=int, default=64)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
