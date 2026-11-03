"""Bounded real-environment distillation training timing, without W&B.

Run each arm in a fresh process on the same allocated GPU. The graph collection
adapter below is benchmark-only; production collection is left unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

os.environ["WANDB_MODE"] = "disabled"

import gymnasium as gym
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.teacher_graph import TeacherGraph
from tests.benchmark_aux_return_joint_vmap import _tiny_params


class CollectionGraph(TeacherGraph):
    """Same fresh graph protocol, retaining sampled execution and teacher stats."""

    def __init__(self, engine, root, **kwargs):
        engine._timer_start = lambda: None
        engine._timer_stop = lambda key, start: None
        super().__init__(engine, root, **kwargs)

    def _solve(self):
        action, metrics, lengths = self.engine.act(
            self.root, t0=True, eval_mode=False, collect_diagnostics=False,
            capture_teacher_policy=True, apply_inner_writeback=False,
        )
        return action, metrics, lengths, self.engine.last_teacher_policy


def install_graph_collection(agent):
    """Install an explicitly non-resumable benchmark adapter on the live engine."""
    live = agent.inner_engine
    generator = torch.Generator().manual_seed(int(agent.cfg.seed) + 193817)
    state = {"runner": None, "calls": 0}

    def act(root, **kwargs):
        if kwargs.get("eval_mode") or kwargs.get("return_behavior_policy") or kwargs.get("trace"):
            raise ValueError("Benchmark graph collection only supports ordinary training actions.")
        if state["runner"] is None:
            state["runner"] = CollectionGraph(InnerImprovementEngine(agent), root)
        runner = state["runner"]
        seed = int(torch.randint(0, 2**31 - 1, (), generator=generator))
        action, metrics, lengths, teacher = runner(root, seed=seed)
        # The ordinary caller materializes action/metrics before another solve.
        # Teacher metadata must own storage across staging into real replay.
        with torch.cuda.stream(runner.stream):
            owned_teacher = {key: value.clone() for key, value in teacher.items()}
        torch.cuda.current_stream(agent.device).wait_stream(runner.stream)
        live.last_teacher_policy = owned_teacher
        live.action_index += 1
        live.state.outer_version = int(agent.outer_version)
        state["calls"] += 1
        return action, dict(metrics), lengths

    live.act = act
    return state


def recipe(args):
    params = _tiny_params(
        model_size=None, enc_dim=256, mlp_dim=512, latent_dim=512, simnorm_dim=8,
        num_q=5, num_bins=101, q_num_bins=101, q_representation="distributional",
        q_vmin=-10, q_vmax=10, dropout=0.01, critic_ensemble_backend="vmap",
        train_unroll_horizon=3, outer_planning_horizon=3,
        inner_rounds=8, inner_rollouts_per_round=128, inner_rollout_horizon=3,
        inner_batch_size=256, inner_replay_capacity=3072,
        inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
        inner_temperature_updates_per_round=0, inner_temperature_mode="inherit_outer",
        outer_actor_objective="distillation", reanalyze_interval=args.refresh_interval,
        reanalyze_batch_size=20, reanalyze_backend=getattr(args, "reanalyze_backend", "cuda_graph") if args.refresh_interval else "native",
        reanalyze_parallel_roots=getattr(args, "parallel_roots", 4), batch_size=256, buffer_size=4096,
        seed_steps=500, pretrain_steps=2, compile=True, compile_strict=True,
        inner_diagnostics_every=1_000_000_000, inner_diagnostic_rollouts=0,
        wandb=False, utd=1,
    )
    params.pop("inner_updates_per_round")
    if args.tiny:
        params.update(
            enc_dim=16, mlp_dim=16, latent_dim=8, simnorm_dim=4,
            num_q=2, num_bins=5, q_num_bins=5, q_representation="scalar", dropout=0.0,
            critic_ensemble_backend="loop", train_unroll_horizon=2,
            outer_planning_horizon=2, inner_rounds=1, inner_rollouts_per_round=2,
            inner_rollout_horizon=2, inner_batch_size=4, inner_replay_capacity=4,
            inner_critic_updates_per_round=1, inner_actor_updates_per_round=1,
            seed_steps=5, pretrain_steps=2, batch_size=3, buffer_size=128,
            compile=False, compile_strict=False,
        )
        if args.device == "cpu":
            params["reanalyze_backend"] = "native"
    params["device"] = args.device
    return params


def benchmark(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    params = recipe(args)
    seed_steps = params["seed_steps"]
    if not args.tiny and args.warmup < 500:
        raise ValueError("Warm up through a complete adapted episode so cached-only actor learning is active.")
    if args.decisions % 10:
        raise ValueError("Measure a multiple of ten decisions to include complete refresh periods.")
    if args.tiny:
        env = gym.make("Pendulum-v1", max_episode_steps=5)
        env_name = "Pendulum-v1"
    else:
        import domains  # noqa: F401
        env = gym.make("DMControl-v0", task="humanoid-walk", obs="state", render_mode=None)
        env_name = "DMControl-v0"
    total = seed_steps + 1 + args.warmup + args.decisions
    run = dict(seed=55, device=args.device, env=env_name, total_steps=total,
               checkpoint_every=None, save_strat="none")
    manifest = dict(logs="none", checkpoint_every=None, save_strat="none", save_trials="none")
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2("distillation-latency", env, params, run, manifest)
        agent, cfg = learner.agent, learner.cfg
        assert agent.ent_coef_optim is not None, "Include outer temperature tuning in both timing arms."
        graph = install_graph_collection(agent) if args.collection == "cuda_graph" else None
        first = seed_steps + args.warmup + 2
        last = first + args.decisions - 1
        actions, decisions, updates, refreshes = [], [], [], []
        timestamps = {}
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
        compile_counts = {}

        def observed_act(*positional, **kwargs):
            step = learner._global_step + 1
            if step == first:
                sync()
                timestamps.update(start=time.perf_counter(), previous=time.perf_counter(),
                                  cpu=time.process_time(), updates=learner._num_updates,
                                  actor_updates=agent.distillation.actor_updates)
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            tick = time.perf_counter()
            action = act(*positional, **kwargs)
            if first <= step <= last:
                actions.append(time.perf_counter() - tick)
                assert torch.isfinite(action).all()
                metrics = agent.last_inner_metrics
                assert metrics["inner_actor_optimizer_steps"] == cfg.inner_actor_updates_per_action
                assert metrics["inner_critic_optimizer_steps"] == cfg.inner_critic_updates_per_action
            return action

        def observed_update(*positional, **kwargs):
            tick = time.perf_counter()
            metrics = update(*positional, **kwargs)
            if first <= learner._global_step <= last:
                updates.append(time.perf_counter() - tick)
                assert not metrics["compile_fallback"]
                if metrics.get("distillation_reanalyze_ran", 0):
                    refreshes.append({k: float(v) for k, v in metrics.items()
                                      if k.startswith("distillation_reanalyze_")})
            return metrics

        def observed_log(*positional, **kwargs):
            result = log_step(*positional, **kwargs)
            step = learner._global_step
            if first <= step <= last:
                if step == last:
                    sync()
                now = time.perf_counter()
                decisions.append(now - timestamps["previous"])
                timestamps["previous"] = now
                if step == last:
                    timestamps.update(end=now, cpu_end=time.process_time())
            if step > seed_steps and (step - seed_steps) % 100 == 0:
                print(json.dumps({"training_step": step, "outer_updates": learner._num_updates,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log
        learner.learn(total)
        compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert compile_counts["before"] == compile_counts["after"], compile_counts
        assert len(decisions) == len(actions) == len(updates) == args.decisions
        assert learner._num_updates - timestamps["updates"] == args.decisions
        actor_updates = agent.distillation.actor_updates - timestamps["actor_updates"]
        assert actor_updates == args.decisions, "Cached-only timings must include actor/temperature learning."
        expected_refreshes = args.decisions // args.refresh_interval if args.refresh_interval else 0
        assert len(refreshes) == expected_refreshes, (len(refreshes), expected_refreshes)
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        seconds = timestamps["end"] - timestamps["start"]
        result = dict(
            collection=args.collection, reanalyze_interval=args.refresh_interval,
            decisions=args.decisions, outer_updates=args.decisions, actor_updates=actor_updates,
            seconds=seconds, seconds_per_decision=seconds/args.decisions,
            decisions_per_second=args.decisions/seconds,
            collection_seconds_mean=statistics.mean(actions),
            learner_seconds_mean=statistics.mean(updates),
            decision_seconds=decisions, collection_seconds=actions, learner_seconds=updates,
            refreshes=refreshes, refresh_count=len(refreshes),
            mean_refresh_seconds=statistics.mean([r["distillation_reanalyze_seconds"] for r in refreshes]) if refreshes else 0.,
            mean_refreshed_roots=statistics.mean([r["distillation_reanalyze_roots"] for r in refreshes]) if refreshes else 0.,
            projected_500k_days=seconds/args.decisions*500000/86400,
            projected_1m_days=seconds/args.decisions*1000000/86400,
            average_cpu_cores=(timestamps["cpu_end"]-timestamps["cpu"])/seconds,
            compile_graph_counts=compile_counts,
            gpu=torch.cuda.get_device_name(0) if args.device=="cuda" else None,
            torch=torch.__version__, source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            warmup_decisions=args.warmup, total_decisions=total,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
            config=vars(cfg), wandb=False, checkpoints=False,
            graph_collection_protocol="benchmark-only sampled execution using per-root seeded graph RNG; no production resume integration" if graph else None,
            scope="Actual env, collection, replay sampling, distillation actor/temperature and model/Q updates; startup, checkpoint I/O and W&B excluded.",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,default=str)+"\n")
        print(json.dumps({key:value for key,value in result.items() if key not in
                         ("config","decision_seconds","collection_seconds","learner_seconds","refreshes")},indent=2))
        return result
    finally:
        if learner is not None:
            learner.close()
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection",choices=("native","cuda_graph"),default="native")
    parser.add_argument("--refresh-interval",type=int,choices=(0,10),default=10)
    parser.add_argument("--reanalyze-backend", choices=("native", "cuda_graph", "batched", "batched_cuda_graph"), default="cuda_graph")
    parser.add_argument("--parallel-roots", type=int, default=4)
    parser.add_argument("--warmup",type=int,default=510)
    parser.add_argument("--decisions",type=int,default=100)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--tiny",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.collection=="cuda_graph" and args.device!="cuda":
        parser.error("Graph collection requires CUDA.")
    benchmark(args)


if __name__=="__main__":
    main()
