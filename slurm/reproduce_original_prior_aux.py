"""Launch the historical SAC prior with a detached auxiliary reward critic.

The gate measures online logging and full-capacity replay in disposable runs.
Production always starts fresh from the versioned recipe and real transitions.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import reproduce_original_prior as base

ROOT = base.ROOT
STEM = "ambi_original_prior_aux_detached_target21_1p5m"
ALG = Path("configs/dmcontrol/algs") / f"{STEM}.json"
MANIFEST = Path("configs/dmcontrol/experiments") / f"{STEM}.json"
ALLOWED_SCIENTIFIC_CHANGES = {
    "RL/AMBITDMPC2.py", "RL/tdmpc2_core/ambi_agent.py",
    "RL/tdmpc2_core/auxiliary_return.py", "RL/tdmpc2_core/common/soft_world_model.py",
    "RL/tdmpc2_core/README.md", "configs/dmcontrol/ORIGINAL_PRIOR_AUX.md", str(ALG), str(MANIFEST),
}


def source_binding():
    if base.git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("A clean source checkout is required.")
    changed = set(base.git("diff", "--name-only", base.ORIGINAL, "--", *base.PROTECTED).splitlines())
    if changed - ALLOWED_SCIENTIFIC_CHANGES:
        raise RuntimeError(f"Unexpected historical changes: {sorted(changed - ALLOWED_SCIENTIFIC_CHANGES)}")
    return {"original_commit": base.ORIGINAL, "launch_commit": base.git("rev-parse", "HEAD"),
            "tree": base.git("rev-parse", "HEAD^{tree}"),
            "changed_scientific_paths": sorted(changed),
            "hashes": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                       for p in (ALG, MANIFEST, Path("environments/dmcontrol/uv.lock"))}}


def recipe(name):
    run = json.loads((ROOT / ALG).read_text())
    manifest = json.loads((ROOT / MANIFEST).read_text())
    run["alg_params"]["wandb_run_name"] = name
    return run, manifest


def configure_threads(count):
    import torch
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)


def choose_resources(cases):
    # Every case times complete online training iterations at both occupancies.
    fastest = min(max(c["early_seconds_per_decision"], c["full_seconds_per_decision"]) for c in cases)
    eligible = [c for c in cases if max(c["early_seconds_per_decision"], c["full_seconds_per_decision"]) <= 1.05 * fastest]
    chosen = min(eligible, key=lambda c: c["cores"])
    rate = max(chosen["early_seconds_per_decision"], chosen["full_seconds_per_decision"])
    seconds = rate * 1_500_000 * 1.20 + chosen["startup_seconds"] + 60 * chosen["checkpoint_seconds"] + 300
    return {"cpus": chosen["cores"], "seconds_per_decision": rate,
            "walltime_minutes": math.ceil(seconds / 900) * 15,
            "throughput_margin": 0.20, "startup_seconds": chosen["startup_seconds"],
            "checkpoint_seconds": chosen["checkpoint_seconds"], "final_io_margin_seconds": 300}


def fill_replay_fixture(buffer):
    """Benchmark-only repetition of real episodes; never called by train()."""
    template = buffer._buffer._storage[:501].clone().to(buffer._storage_device)
    assert tuple(template.batch_size) == (501,)
    assert template["episode"].unique().numel() == 1
    while buffer.size < buffer.capacity:
        count = min(100, math.ceil((buffer.capacity - buffer.size) / 501))
        buffer.load(template.unsqueeze(0).expand(count, 501).clone())
    assert buffer.size == buffer.capacity == 1_000_000
    assert buffer._storage_device.type == "cuda"


def worker(args):
    cpus = [int(x) for x in args.cpus.split(",")]
    if not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError("CPU affinity is outside the allocation.")
    os.sched_setaffinity(0, cpus)
    sys.path.insert(0, str(ROOT))
    import torch
    from main import seed_everything, seed_env_spaces
    from utils.core import build_env
    from RL.AMBITDMPC2 import AMBITDMPC2
    configure_threads(len(cpus))
    output = Path(args.output)
    run, manifest = recipe(f"BENCHMARK ONLY | original + aux | {len(cpus)} CPU | job{os.environ['SLURM_JOB_ID']}")
    run["alg_params"]["wandb_tags"] += ["benchmark-only", "real-then-repeated-replay-fixture", "stop-at5000"]
    manifest["study_note"] = "Disposable timing gate: 5000 real decisions; replay is filled with repeated real episodes at4000. Not a scientific training run."
    identity = base.runtime()
    seed_everything(55)
    env = build_env(run, manifest)
    seed_env_spaces(env, 55)
    learner = None
    timings = {}
    start = time.perf_counter()
    try:
        learner = AMBITDMPC2(STEM, env, run["alg_params"], run, manifest)
        assert learner.cfg.seed_steps == learner.cfg.pretrain_steps == 2500
        assert learner.cfg.aux_return_detach_representation
        assert learner.agent.target_entropy == -21
        assert learner.cfg.inner_operator == "none"
        learner.set_checkpointing(5000, str(output / "checkpoints"), "smoke", save_strat="all")
        act, log_step, checkpoint = learner._act_agent, learner._log_wandb_step, learner._maybe_checkpoint

        def observed_act(*pos, **kw):
            step = learner._global_step + 1
            if step == 4001:
                fill_replay_fixture(learner.buffer)
            if step in (3001, 4501):
                torch.cuda.synchronize()
                name = "early" if step == 3001 else "full"
                timings[name] = {"start": time.perf_counter(), "cpu_start": time.process_time(),
                                 "graphs_before": int(torch._dynamo.utils.counters["stats"]["unique_graphs"])}
                if name == "early":
                    timings["startup_seconds"] = time.perf_counter() - start
            return act(*pos, **kw)

        def observed_log(*pos, **kw):
            result = log_step(*pos, **kw)
            if learner._global_step in (4000, 5000):
                name = "early" if learner._global_step == 4000 else "full"
                if "seconds" not in timings[name]:
                    torch.cuda.synchronize()
                    timings[name].update(seconds=time.perf_counter() - timings[name]["start"],
                                         cpu_seconds=time.process_time() - timings[name]["cpu_start"],
                                         graphs_after=int(torch._dynamo.utils.counters["stats"]["unique_graphs"]))
            return result

        def observed_checkpoint():
            if learner._global_step != 5000:
                return checkpoint()
            t = time.perf_counter()
            checkpoint()
            # Join the normal asynchronous writer before timing storage cost.
            learner._checkpoint_writer.shutdown()
            timings["checkpoint_seconds"] = time.perf_counter() - t
            raise base.BenchmarkComplete()

        learner._act_agent, learner._log_wandb_step, learner._maybe_checkpoint = observed_act, observed_log, observed_checkpoint
        try:
            learner.learn(1_500_000)
        except base.BenchmarkComplete:
            pass
        assert learner._global_step == 5000
        assert learner.buffer._storage_device.type == "cuda"
        assert learner.buffer.size == 1_000_000
        for name in ("early", "full"):
            assert timings[name]["graphs_before"] == timings[name]["graphs_after"], "Compilation inside timing window"
        assert all(torch.isfinite(p).all() for p in learner.agent.model.parameters())
        metrics = learner._last_train_metrics
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
        assert all(float(v) == 0 for k, v in metrics.items() if "fallback" in k)
        assert "aux_return_critic_loss" in metrics
        assert all(float(learner.agent.last_inner_metrics.get(k, 0)) == 0 for k in
                   ("inner_model_steps", "inner_actor_optimizer_steps", "inner_critic_optimizer_steps"))
        saved = output / "checkpoints/smoke_5000"
        metadata = json.loads(Path(str(saved) + ".metadata.json").read_text())
        assert metadata["checkpoint"]["step"] == 5000
        before = {k: v.clone() for k, v in learner.agent.model.state_dict().items() if "aux_return" in k}
        assert before
        learner.agent.load(str(saved))
        assert all(torch.equal(v, learner.agent.model.state_dict()[k]) for k, v in before.items())
        result = {"passed": True, "cores": len(cpus), "runtime": identity,
                  "early_seconds_per_decision": timings["early"]["seconds"] / 1000,
                  "full_seconds_per_decision": timings["full"]["seconds"] / 500,
                  "startup_seconds": timings["startup_seconds"],
                  "checkpoint_seconds": timings["checkpoint_seconds"],
                  "timing_windows": timings, "replay_rows": learner.buffer.size,
                  "checkpoint": str(saved), "checkpoint_bytes": saved.stat().st_size,
                  "auxiliary_checkpoint_reload_passed": True, "online_logging": True,
                  "wandb_run_id": os.environ["WANDB_RUN_ID"], "compile_fallback": False,
                  "total_seconds": time.perf_counter() - start}
        base.write_new(output / "result.json", result)
        print("CPU_CASE_COMPLETE " + json.dumps(result), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def benchmark(args):
    binding = source_binding()
    output, free = base.require_output(args.output)
    cores = base.physical_cpus(os.sched_getaffinity(0))
    if len(cores) < 4:
        raise RuntimeError("Benchmark needs four physical CPUs.")
    cases = []
    for count in (1, 2, 4):
        case = output / f"cpu{count}"
        case.mkdir()
        env = dict(os.environ, OMP_NUM_THREADS=str(count), MKL_NUM_THREADS=str(count),
                   OPENBLAS_NUM_THREADS=str(count), WANDB_MODE="online", WANDB__REQUIRE_CORE="true",
                   WANDB_RUN_ID=f"auxgate-{os.environ['SLURM_JOB_ID']}-c{count}")
        env.pop("WANDB_RESUME", None)
        with (case / "stdout.log").open("x") as handle:
            subprocess.run([sys.executable, __file__, "worker", "--output", str(case),
                            "--cpus", ",".join(map(str, cores[:count]))], cwd=ROOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        cases.append(json.loads((case / "result.json").read_text()))
        print("BENCHMARK_PROGRESS " + json.dumps({"cores": count, "full_seconds_per_decision": cases[-1]["full_seconds_per_decision"]}), flush=True)
    result = {"passed": True, "binding": binding, "cases": cases, "free_bytes_before": free,
              "recommended_resources": choose_resources(cases), "job_id": os.environ["SLURM_JOB_ID"],
              "scope": "Full original warmup/pretraining and online logging; early real replay plus1M-row repeated-real-episode fixture. Fixture never used in production."}
    base.write_new(output / "benchmark.json", result)
    print("BENCHMARK_COMPLETE " + json.dumps(result["recommended_resources"]), flush=True)


def train(args):
    binding = source_binding()
    gate = json.loads(Path(args.gate).read_text())
    if not gate.get("passed") or gate["binding"] != binding:
        raise RuntimeError("Successful benchmark for this exact source is required.")
    if int(os.environ["SLURM_CPUS_PER_TASK"]) != gate["recommended_resources"]["cpus"]:
        raise RuntimeError("CPU allocation does not match benchmark.")
    identity = base.runtime_in_child()
    if any(identity != case["runtime"] for case in gate["cases"]):
        raise RuntimeError("Training runtime differs from benchmark.")
    output, free = base.require_output(args.output)
    # Leave enough space for both this bank and the concurrently running original.
    required = max(c["checkpoint_bytes"] for c in gate["cases"]) * 60 + 4_348_405_800 + 4 * 1024**3
    if free < required:
        raise RuntimeError(f"Insufficient storage for both retained banks: need {required} bytes, have {free}.")
    name = f"Original + detached auxiliary critic | seed55 | target−21 | Hydra L40 | job{os.environ['SLURM_JOB_ID']}"
    run, manifest = recipe(name)
    algdir = output / "configs"
    algdir.mkdir()
    base.write_new(algdir / f"{STEM}.json", run)
    base.write_new(output / "manifest.json", manifest)
    base.write_new(output / "launch.json", {"binding": binding, "runtime": identity,
                   "job_id": os.environ["SLURM_JOB_ID"], "run_name": name,
                   "wandb_run_id": os.environ["WANDB_RUN_ID"], "free_bytes_before": free,
                   "required_bytes_both_banks_with_margin": required,
                   "checkpoint_every": 25000, "retain_all": True, "expected_checkpoints": 60,
                   "seed": 55, "budget_decisions": 1_500_000, "resources": gate["recommended_resources"],
                   "benchmark": str(Path(args.gate).resolve()), "algorithm_config": run})
    subprocess.run([sys.executable, __file__, "entry", "--output", str(output)], cwd=ROOT, check=True)
    models = [p for p in (output / "artifacts").rglob("model:*") if p.is_file() and not p.name.endswith(".metadata.json")]
    steps = sorted(json.loads(Path(str(p) + ".metadata.json").read_text())["checkpoint"]["step"] for p in models)
    if steps != list(range(25000, 1_500_001, 25000)):
        raise RuntimeError(f"Retained checkpoint set is incomplete: {steps}")
    base.write_new(output / "complete.json", {"checkpoints": len(models), "steps": steps,
                                              "bytes": sum(p.stat().st_size for p in models)})


def entry(args):
    configure_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    output = Path(args.output)
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "main.py"), "--run", str(output / "manifest.json"),
                "--alg-dir", str(output / "configs"), "--log-dir", str(output / "artifacts"),
                "--alg-index", "0", "--trial-index", "0", "--num-runs", "1"]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "worker", "train", "entry"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpus")
    parser.add_argument("--gate")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("All workloads require a Slurm allocation.")
    if args.mode == "train" and not args.gate:
        parser.error("train requires --gate")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
