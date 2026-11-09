"""Per-arm GPU validation, measured resource sizing, and six-cell gSDE Oscar launch."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import binding, recipe, runtime, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_gsde_noise_scopes.json"
CAMPAIGN_ROOT = Path("/oscar/scratch/rgao48/ambi/gsde-noise-scopes-20261105")
if os.environ.get("AMBI_GSDE_VARIANT") == "humanoid_run":
    MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_gsde_squashed_real_steps_humanoid_run.json"
    CAMPAIGN_ROOT = Path("/oscar/scratch/rgao48/ambi/gsde-humanoid-run-20261108")
GPU_PARITY_TEST = "tests/test_ambi_gsde_graph.py"


def parity_cases(junit_path):
    """Reject a successful pytest exit that merely skipped unavailable CUDA."""
    cases = list(ET.parse(junit_path).getroot().iter("testcase"))
    names = [case.attrib["name"] for case in cases]
    assert len(cases) == 7 and len(set(names)) == 7, names
    assert all(name.startswith("test_cuda_") for name in names), names
    assert not any(case.find(tag) is not None for case in cases
                   for tag in ("skipped", "error", "failure")), names
    return names


def validate_global_parity(receipt, expected_binding, expected_runtime):
    assert receipt["passed"] is True
    assert receipt["binding"] == expected_binding
    assert receipt["runtime"] == expected_runtime
    assert receipt["case_count"] == 7 and len(set(receipt["case_names"])) == 7
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
    """A one-CPU GPU allocation can validate all six arms before sizing jobs."""
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "disabled"
    campaign = Path(os.environ["AMBI_GSDE_CAMPAIGN"])
    assert campaign == CAMPAIGN_ROOT
    args.output.mkdir(parents=True, exist_ok=False)
    run_gpu_parity(args.output, campaign)
    print("GPU_PARITY_PASSED", str(campaign / "gpu-parity.json"), flush=True)


def check_geometry(agent, buffer):
    """Exercise the real sparse observer outside timing without changing learning."""
    import torch
    from RL.tdmpc2_core.critic_geometry import critic_geometry_diagnostics

    cfg = agent.cfg
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.critic_geometry_batch_size == 64
    assert cfg.critic_geometry_obs_noise_std == .01
    assert cfg.critic_geometry_action_noise_std == .02
    obs, action, _, _, _ = buffer.sample()
    before = {name: tensor.detach().clone() for name, tensor in agent.model.state_dict().items()}
    parameters = tuple(agent.model.parameters())
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    modes = [module.training for module in agent.model.modules()]
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    values = critic_geometry_diagnostics(agent.model, obs, action, cfg, update_number=1000)
    torch.cuda.synchronize()
    assert values["critic_geometry_sample_count"].item() == 64
    for name, tensor in agent.model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert [module.training for module in agent.model.modules()] == modes
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    prefix = "critic_geometry_primary_"
    assert values[prefix + "gradient_scale_valid"].item() in (0., 1.)
    allowed_undefined = set()
    for term in ("mpr", "action_smoothness"):
        assert values[prefix + term + "_loss"].item() >= 0
        valid_fraction = values[prefix + term + "_cosine_valid_fraction"].item()
        assert 0 <= valid_fraction <= 1
        if valid_fraction == 0:
            allowed_undefined.add(prefix + term + "_cosine_similarity")
        if values[prefix + "gradient_scale_valid"].item() == 0:
            allowed_undefined.add(prefix + term + "_relative_gradient_change")
    result = {}
    for key, tensor in values.items():
        value = float(tensor)
        if key in allowed_undefined:
            assert math.isnan(value), (key, value)
            result[key] = None
        else:
            assert math.isfinite(value), (key, value)
            result[key] = value
    return {"passed": True, "update_number": 1000, "parameters_gradients_rng_unchanged": True,
            "metrics": result}


def accelerated_runtime():
    import torch
    torch.set_float32_matmul_precision("high")
    return {**runtime(), "float32_matmul_precision": torch.get_float32_matmul_precision()}


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
    graph_enabled = not args.native_reference
    overrides = {"wandb": False, "wandb_mode": "disabled", "seed_steps": 500,
                 "pretrain_steps": 2, "inner_cuda_graph": graph_enabled}
    run["alg_params"].update(overrides)
    warmup, measured = 16, 32 if args.native_reference else 96
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
        expected_scope = "real_steps" if os.environ.get("AMBI_GSDE_VARIANT") == "humanoid_run" else ("round", "solve", "real_steps")[args.index % 3]
        assert cfg.inner_gsde_noise_scope == expected_scope
        assert cfg.inner_gsde_resample_steps == cfg.gsde_execution_resample_steps == 8
        assert (cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.inner_solve_interval) == (3, 8, 1)
        assert cfg.inner_finite_horizon
        assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode
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
        # Checkpoint at an explicitly prepared boundary; live gSDE draws are not silently omitted.
        agent.prepare_training_resume_boundary()
        with tempfile.TemporaryDirectory(prefix="gsde-checkpoint-") as temp:
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
                  "critic_geometry": geometry,
                  "gpu_replay": True, "metrics": {k: float(v) for k, v in metrics.items()},
                  "diagnostic_episodes": learner._training_diagnostics.episode_metrics(),
                  "seconds": time.perf_counter() - started,
                  "benchmark_overrides": {**overrides, "total_steps": total}, "wandb": False}
        write_new(args.output, report)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def checkpoint_space(root, bank_bytes):
    # Shared-filesystem free space alone does not represent this user's quota.
    quota = subprocess.check_output(["checkquota"], text=True)
    rows = [line.split() for line in quota.splitlines()]
    row = next(r for r in rows if len(r) > 5 and r[0] == "rgao48" and r[1] == "/oscar/scratch")
    quota_free = int((float(row[4]) - float(row[2])) * 2**30)
    disk_free = shutil.disk_usage(root).free
    required = bank_bytes * 1.3 + 2 * 2**30
    assert min(quota_free, disk_free) >= required, (quota_free, disk_free, required)
    return {"quota_free_bytes": quota_free, "filesystem_free_bytes": disk_free,
            "checkpoint_bank_bytes": bank_bytes, "required_with_margin_bytes": required}


def select_plan(rows):
    fastest = min(rows, key=lambda r: r["measured"]["seconds_per_decision"])
    one = next(r for r in rows if r["physical_cpus"] == 1)
    selected = one if one["measured"]["seconds_per_decision"] <= 1.05 * fastest["measured"]["seconds_per_decision"] else fastest
    seconds = selected["measured"]["seconds_per_decision"]
    return {"index": selected["index"], "cpus": selected["physical_cpus"],
            "seconds_per_decision": seconds,
            "wall_hours": math.ceil(seconds * 1000000 * 1.15 / 3600 + .5),
            "checkpoint_bytes": selected["checkpoint_bytes"],
            "memory_gib": max(8, math.ceil(max(r["peak_rss_kib"] for r in rows) / 2**20 * 1.25 + 1))}


def slurm_fields(job):
    return dict(item.split("=", 1) for item in subprocess.check_output(
        ["scontrol", "show", "job", str(job), "-o"], text=True).split() if "=" in item)


def resize_pending(job, plan):
    fields = slurm_fields(job)
    assert fields["JobState"] == "PENDING" and fields["UserId"].startswith("rgao48("), fields
    assert fields["WorkDir"] == str(ROOT), fields["WorkDir"]
    subprocess.run(["scontrol", "hold", str(job)], check=True)
    count = str(plan["cpus"])
    subprocess.run(["scontrol", "update", f"JobId={job}",
                    f"TimeLimit={plan['wall_hours']}:00:00", f"NumCPUs={count}",
                    f"CPUsPerTask={count}", f"TresPerTask=cpu:{count}",
                    f"MinCPUsNode={count}"], check=True)
    fields = slurm_fields(job)
    assert fields["JobState"] == "PENDING"
    for name in ("NumCPUs", "CPUs/Task", "MinCPUsNode"):
        assert fields[name] == count, (name, fields)
    assert fields["TresPerTask"] in {f"cpu:{count}", f"cpu={count}"}, fields
    limit = fields["TimeLimit"]
    days, clock = limit.split("-") if "-" in limit else ("0", limit)
    hh, mm, ss = map(int, clock.split(":"))
    assert int(days) * 24 + hh == plan["wall_hours"] and mm == ss == 0, fields
    subprocess.run(["scontrol", "release", str(job)], check=True)
    return fields


def gate(args):
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "disabled"
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = Path(os.environ["AMBI_GSDE_CAMPAIGN"])
    assert campaign == CAMPAIGN_ROOT
    if (campaign / "gpu-parity.json").exists():
        load_global_parity(campaign, accelerated_runtime())
    elif args.index == 0:
        run_gpu_parity(args.output, campaign)
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
    native_output = args.output / "native-cpu-1.json"
    native_env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        native_env[key] = "1"
    with native_output.with_suffix(".log").open("x") as log:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "worker", "--index", str(args.index),
             "--worker-cpus", str(cores[0]), "--native-reference", "--output", str(native_output)],
            cwd=ROOT, env=native_env, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    native = json.loads(native_output.read_text())
    plan = select_plan(rows)
    bank_bytes = 6 * 40 * max(r["checkpoint_bytes"] for r in rows)
    space = checkpoint_space(args.output, bank_bytes)
    receipt = {"passed": True, "binding": binding(MANIFEST), "runtime": rows[0]["runtime"],
               "plan": plan, "space": space, "cases": [str(args.output/f"cpu-{n}.json") for n in (1,2)],
               "native_reference": str(native_output),
               "graph_speedup_vs_native_one_cpu": native["measured"]["seconds_per_decision"] / rows[0]["measured"]["seconds_per_decision"],
               "global_gpu_parity_required": all(r["global_gpu_parity_required"] for r in rows),
               "global_gpu_parity_receipt": str(campaign / "gpu-parity.json"),
               "scope": "Real H3/J8/K1 Humanoid training, production model/batch; 96 graph decisions on 1/2 CPUs and 32 native reference decisions; geometry and boundary/checkpoint checks outside timing; no W&B."}
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
    campaign = Path(os.environ["AMBI_GSDE_CAMPAIGN"])
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
    parser.add_argument("--index", type=int, choices=range(6))
    parser.add_argument("--worker-cpus")
    parser.add_argument("--native-reference", action="store_true")
    parser.add_argument("--release-job", type=int)
    args = parser.parse_args()
    {"parity": parity, "gate": gate, "worker": worker, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
