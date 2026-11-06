"""Validate and launch the three BOOM x0.01 plus gSDE Humanoid runs on Oscar."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import binding, recipe, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus
from slurm.gsde_noise_scopes_oscar import (
    accelerated_runtime, check_geometry, checkpoint_space, resize_pending, select_plan,
)

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_boom_gsde.json"
CAMPAIGN_ROOT = Path("/oscar/scratch/rgao48/ambi/boom-gsde-20261106")
GPU_PARITY_TEST = "tests/test_ambi_boom_gsde_graph.py"
SCOPES = ("real_steps", "solve", "round")


def check_active_boom(learner):
    """Exercise saved behavior labels and the active loss after steady-state timing.

    The bounded training window ends before a second complete episode. Add its
    real, labeled prefix to this disposable benchmark replay, then deliberately
    raise S for one outer actor check. Neither change reaches production.
    """
    import torch
    agent = learner.agent
    rows = int(learner._episode_len) + 1
    assert rows > agent.cfg.train_unroll_horizon + 1
    staged = learner._episode_staging[:rows]
    assert staged["behavior_policy_valid"][-agent.cfg.train_unroll_horizon:].all()
    learner.buffer.add(staged)
    batch = learner.buffer.sample(include_behavior_policy=True)
    obs, _, _, _, _, mean, log_std, valid = batch
    assert valid.any() and torch.isfinite(mean[valid.expand_as(mean)]).all()
    with torch.no_grad():
        zs = agent.model.encode(obs)
        agent.actor_loss_scale.fill_(3.2)
    metrics = agent._update_actor(zs, behavior_pre_tanh_mean=mean,
                                  behavior_log_std=log_std, behavior_policy_valid=valid)
    torch.cuda.synchronize()
    assert metrics["boom_gate_active"] == 1
    assert metrics["boom_valid_fraction"] > 0
    assert metrics["compile_outer_actor_fallback"] == 0
    assert all(torch.isfinite(parameter).all() for parameter in agent.model._pi.parameters())
    for key in ("boom_alignment_loss", "actor_loss", "actor_grad_norm"):
        assert math.isfinite(float(metrics[key])), (key, metrics[key])
    return {"passed": True, "scale_override": 3.2, "partial_episode_rows": rows,
            "valid_fraction": float(metrics["boom_valid_fraction"]),
            "gate_active": float(metrics["boom_gate_active"]),
            "alignment_loss": float(metrics["boom_alignment_loss"]),
            "actor_grad_norm": float(metrics["actor_grad_norm"])}


def parity_cases(junit_path):
    """Reject a successful pytest exit that merely skipped unavailable CUDA."""
    cases = list(ET.parse(junit_path).getroot().iter("testcase"))
    names = [case.attrib["name"] for case in cases]
    assert len(cases) == 4 and len(set(names)) == 4, names
    assert all(name.startswith("test_cuda_") for name in names), names
    assert not any(case.find(tag) is not None for case in cases
                   for tag in ("skipped", "error", "failure")), names
    return names

def validate_global_parity(receipt, expected_binding, expected_runtime):
    assert receipt["passed"] is True
    assert receipt["binding"] == expected_binding
    assert receipt["runtime"] == expected_runtime
    assert receipt["case_count"] == 4 and len(set(receipt["case_names"])) == 4
    assert receipt["test_node"] == GPU_PARITY_TEST
    assert receipt["failures"] == receipt["errors"] == receipt["skipped"] == 0

def load_global_parity(campaign, current):
    """Verify persisted evidence before reusing a parity job or starting training."""
    receipt = json.loads((campaign / "gpu-parity.json").read_text())
    validate_global_parity(receipt, binding(MANIFEST), current)
    junit_path = Path(receipt["junit_path"])
    assert junit_path.is_relative_to(campaign)
    assert hashlib.sha256(junit_path.read_bytes()).hexdigest() == receipt["junit_sha256"]
    assert parity_cases(junit_path) == receipt["case_names"]
    return receipt

def run_gpu_parity(output, campaign):
    """One gate validates matrix replay, fixed-noise parameter updates and exact boundary continuation on CUDA."""
    junit = output / "gpu-parity.xml"
    log_path = output / "gpu-parity.log"
    # Match the validated trainer's precision and avoid thread oversubscription.
    program = ("import sys,torch,pytest; torch.set_num_threads(1); "
               "torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('high'); "
               "raise SystemExit(pytest.main(sys.argv[1:]))")
    with log_path.open("x") as log:
        subprocess.run([sys.executable, "-c", program, GPU_PARITY_TEST, "-k", "cuda",
                        "-q", "--junitxml=" + str(junit)], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    names = parity_cases(junit)
    receipt = {"passed": True, "binding": binding(MANIFEST),
               "runtime": accelerated_runtime(), "test_node": GPU_PARITY_TEST,
               "case_count": len(names), "case_names": names,
               "failures": 0, "errors": 0, "skipped": 0,
               "junit_path": str(junit), "log_path": str(log_path),
               "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
               "job_id": os.environ["SLURM_JOB_ID"], "wandb": False}
    write_new(campaign / "gpu-parity.json", receipt)
    return receipt

def parity(args):
    """A one-CPU GPU allocation can validate all three arms before sizing jobs."""
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "disabled"
    campaign = Path(os.environ["AMBI_BOOM_GSDE_CAMPAIGN"])
    assert campaign == CAMPAIGN_ROOT
    args.output.mkdir(parents=True, exist_ok=False)
    run_gpu_parity(args.output, campaign)
    print("GPU_PARITY_PASSED", str(campaign / "gpu-parity.json"), flush=True)

def worker(args):
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    assert set(cpus) <= os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    import gymnasium as gym
    import torch
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2
    from main import _checkpoint_replay_kwargs
    from utils.checkpointing import resolve_checkpoint_config

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    execution = accelerated_runtime()
    name, manifest, run = recipe(args.index, MANIFEST)
    checkpoint_config = resolve_checkpoint_config(run, manifest)
    assert checkpoint_config.enabled and checkpoint_config.every == 25000
    assert _checkpoint_replay_kwargs(AMBITDMPC2, checkpoint_config) == {}
    run, manifest = deepcopy(run), deepcopy(manifest)
    graph_enabled = True
    overrides = {"wandb": False, "wandb_mode": "disabled", "seed_steps": 500,
                 "pretrain_steps": 2, "inner_cuda_graph": graph_enabled}
    run["alg_params"].update(overrides)
    warmup, measured = 16, 96
    total = 501 + warmup + measured
    run.update(total_steps=total, checkpoint_every=None, save_strat="none")
    manifest.update(logs="none", save_trials="none", checkpoint_every=None, save_strat="none")
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-disabled-benchmark", env, run["alg_params"], run, manifest)
        agent, cfg = learner.agent, learner.cfg
        assert cfg.action_dim == 21 and cfg.inner_operator == "sac" and not cfg.mpc
        assert cfg.actor_distribution == "gsde" and cfg.gsde_feature_source == "actor"
        assert cfg.outer_actor_objective == "boom" and cfg.outer_boom_coef == .0042
        assert cfg.outer_boom_compile_mode == "reduce-overhead" and cfg.store_behavior_policy
        assert cfg.inner_gsde_noise_scope == SCOPES[args.index]
        assert cfg.inner_gsde_resample_steps == cfg.gsde_execution_resample_steps == 8
        assert (cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.inner_solve_interval) == (3, 8, 1)
        assert cfg.inner_finite_horizon
        assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
        assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
        assert cfg.inner_cuda_graph is graph_enabled
        assert cfg.outer_critic_target == "reward_only" and cfg.aux_return_mode == "off"
        assert cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"
        assert cfg.inner_actor_loss_scale_scope == "action"
        timer = TrainingWindow(502 + warmup, measured, torch.cuda.synchronize)
        act, log = learner._act_agent, learner._log_wandb_step
        graph_counts = {}
        replay_counts = {}

        def observed_act(*positional, **kwargs):
            step = learner._global_step + 1
            if step == timer.first_step:
                graph_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                runner = agent.inner_engine._whole_graph_runner
                replay_counts["before"] = 0 if runner is None else runner.calls
            timer.before_action(step, learner._num_updates)
            t = time.perf_counter()
            before = agent.actor_loss_scale.detach().clone()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - t)
            assert torch.isfinite(action).all()
            behavior = learner._pending_behavior_policy
            assert set(behavior) == {"pre_tanh_mean", "log_std"}
            assert all(value.shape == (21,) and torch.isfinite(value).all() for value in behavior.values())
            inner = agent.last_inner_metrics
            assert inner["inner_actor_optimizer_steps"] == 32
            assert inner["inner_critic_optimizer_steps"] == 128
            assert inner["inner_temperature_optimizer_steps"] == 0
            assert inner["inner_compile_fallback"] == 0
            # Native-only execution need not emit the graph telemetry key;
            # graph-enabled measurements still require an explicit one.
            assert inner.get("inner_cuda_graph_used", 0.0) == float(graph_enabled)
            assert all(math.isfinite(float(v)) for v in inner.values())
            assert math.isclose(inner["inner_actor_loss_scale_initial"], before.item(), rel_tol=1e-6)
            if cfg.inner_actor_loss_scale_update == "per_action":
                assert inner["inner_actor_loss_scale_final"] == inner["inner_actor_loss_scale_initial"]
            torch.testing.assert_close(agent.actor_loss_scale, before, rtol=0, atol=0)
            return action

        def observed_log(*positional, **kwargs):
            result = log(*positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, learner._log_wandb_step = observed_act, observed_log
        learner.learn(total)
        assert timer.result and learner._global_step == total
        graph_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert graph_counts["before"] == graph_counts["after"], graph_counts
        runner = agent.inner_engine._whole_graph_runner
        replay_counts["after"] = 0 if runner is None else runner.calls
        replay_counts["timed"] = replay_counts["after"] - replay_counts["before"]
        assert replay_counts["timed"] == (measured if graph_enabled else 0), replay_counts
        metrics = learner._last_train_metrics
        assert metrics["compile_fallback"] == 0
        assert all(math.isfinite(float(v)) for v in metrics.values())
        for key in ("actor_pre_tanh_entropy", "actor_post_tanh_entropy",
                    "actor_mean_action_abs_ge_0p99_fraction",
                    "actor_entropy_coefficient_joint_units", "actor_q_range_instant"):
            assert key in metrics, key
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        assert learner.buffer._storage_device.type == "cuda"
        # Expensive sparse observations and recapture checks are deliberately
        # outside the steady-state timing window.
        geometry = check_geometry(agent, learner.buffer)
        boom_probe = check_active_boom(learner)
        # Checkpoint at an explicitly prepared boundary; live gSDE draws are not silently omitted.
        agent.prepare_training_resume_boundary()
        with tempfile.TemporaryDirectory(prefix="boom-gsde-checkpoint-") as temp:
            checkpoint = Path(learner.save(temp, "checkpoint"))
            learner.flush_checkpoints()
            assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
            size = checkpoint.stat().st_size
            scale = agent.actor_loss_scale.detach().clone()
            engine = agent.inner_engine
            engine.prepare_training_resume_boundary()
            inner_state = deepcopy(engine.training_state_dict())
            engine.load_training_state_dict(inner_state)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            agent.load(checkpoint)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(agent.actor_loss_scale, scale, rtol=0, atol=0)
        report = {"passed": True, "index": args.index, "name": name,
                  "physical_cpus": len(cpus), "runtime": execution,
                  "peak_rss_kib": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "binding": binding(MANIFEST), "measured": timer.result,
                  "checkpoint_bytes": size, "checkpoint_roundtrip": True,
                  "checkpoint_scale_roundtrip": True, "checkpoint_sidecar": True,
                  "compile_counts": graph_counts, "compile_fallback": False,
                  "inner_cuda_graph": graph_enabled, "graph_replays": replay_counts,
                  "global_gpu_parity_required": graph_enabled,
                  "critic_geometry": geometry, "active_boom_probe": boom_probe,
                  "gpu_replay": True, "metrics": {k: float(v) for k, v in metrics.items()},
                  "diagnostic_episodes": learner._training_diagnostics.episode_metrics(),
                  "seconds": time.perf_counter() - started,
                  "benchmark_overrides": {**overrides, "total_steps": total}, "wandb": False}
        write_new(args.output, report)
    finally:
        if learner is not None:
            learner.close()
        env.close()

def gate(args):
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "disabled"
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = Path(os.environ["AMBI_BOOM_GSDE_CAMPAIGN"])
    assert campaign == CAMPAIGN_ROOT
    if (campaign / "gpu-parity.json").exists():
        load_global_parity(campaign, accelerated_runtime())
    rows = []
    for count in (1, 2):
        output = args.output / f"cpu-{count}.json"
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(count)
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--index", str(args.index),
                   "--worker-cpus", ",".join(map(str, cores[:count])), "--output", str(output)]
        with output.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows.append(json.loads(output.read_text()))
    plan = select_plan(rows)
    bank_bytes = 3 * 40 * max(r["checkpoint_bytes"] for r in rows)
    space = checkpoint_space(args.output, bank_bytes)
    receipt = {"passed": True, "binding": binding(MANIFEST), "runtime": rows[0]["runtime"],
               "plan": plan, "space": space, "cases": [str(args.output/f"cpu-{n}.json") for n in (1,2)],
               "global_gpu_parity_required": all(r["global_gpu_parity_required"] for r in rows),
               "global_gpu_parity_receipt": str(campaign / "gpu-parity.json"),
               "scope": "Real H3/J8/K1 Humanoid training, production model/batch; 96 graph decisions on 1/2 CPUs; geometry, active BOOM and boundary/checkpoint checks outside timing; no W&B."}
    write_new(args.output / "receipt.json", receipt)
    # Measuring a gate never implicitly submits/resizes/releases production.
    # An explicit release target is useful only for a previously held job whose
    # complete allocation the caller has budgeted against live account limits.
    if args.release_job is not None:
        submission = json.loads((campaign / "submission.json").read_text())
        assert submission["commit"] == receipt["binding"]["commit"]
        arm = submission["runs"][args.index]
        assert str(arm["gate_job"]) == os.environ["SLURM_JOB_ID"]
        assert int(arm["job"]) == args.release_job
        applied = resize_pending(args.release_job, plan)
        write_new(args.output / "allocation.json", {"production_job": args.release_job, "plan": plan, "verified": applied})
        print("GATE_PASSED_AND_PRODUCTION_RELEASED", args.release_job, flush=True)
    else:
        print("GATE_PASSED", args.index, json.dumps(plan), flush=True)

def train(args):
    import torch
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    receipt = json.loads(args.gate.read_text())
    assert receipt["passed"] and receipt["binding"] == binding(MANIFEST)
    assert receipt["global_gpu_parity_required"]
    current = accelerated_runtime()
    for key in ("python", "torch", "cuda", "gpu", "executable", "float32_matmul_precision"):
        assert current[key] == receipt["runtime"][key], key
    campaign = Path(os.environ["AMBI_BOOM_GSDE_CAMPAIGN"])
    assert campaign == CAMPAIGN_ROOT
    parity_path = campaign / "gpu-parity.json"
    parity = load_global_parity(campaign, current)
    plan = receipt["plan"]
    assert plan["index"] == args.index
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == plan["cpus"]
    assert os.environ.get("WANDB_RUN_ID")
    args.output.mkdir(parents=True, exist_ok=False)
    space = checkpoint_space(args.output, receipt["space"]["checkpoint_bank_bytes"])
    write_new(args.output / "launch.json", {
        "binding": binding(MANIFEST), "plan": plan, "runtime": current,
        "output": str(args.output), "gate": str(args.gate), "space": space,
        "gpu_parity_receipt": str(parity_path),
        "gpu_parity_junit_sha256": parity["junit_sha256"],
        "job_id": os.environ["SLURM_JOB_ID"], "wandb_run_id": os.environ["WANDB_RUN_ID"]})
    command = [sys.executable, "main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(args.output), "--alg-index", str(args.index), "--trial-index", "0", "--num-runs", "1"]
    # Keep the validated matmul setting in the same process as the trainer.
    sys.argv = command[1:]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "gate", "worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--index", type=int, choices=range(3))
    parser.add_argument("--worker-cpus")
    parser.add_argument("--release-job", type=int)
    args = parser.parse_args()
    {"parity": parity, "gate": gate, "worker": worker, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
