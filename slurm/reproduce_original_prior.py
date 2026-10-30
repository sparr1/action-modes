"""Reproduce u13m14st without changing its historical training implementation.

The benchmark runs the real original learner, warmup, pretraining, and GPU
replay for 3,500 decisions; only logging and checkpoint cadence differ. The
production command invokes the original main.py with publication-only changes.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "253e89b3896ca3891844d797573b00ade338165d"
STEM = "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"
ALG = Path("configs/dmcontrol/algs") / f"{STEM}.json"
MANIFEST = Path("configs/dmcontrol/experiments") / f"{STEM}.json"
PROTECTED = ("RL", "domains", "utils", "environments", "configs", "main.py", "log.py")
PUBLICATION_KEYS = ("wandb_run_name", "wandb_group", "wandb_tags")
MIN_FREE_BYTES = 8 * 1024**3


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_binding():
    """Reject changed scientific files, including changes not in the commit."""
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Launch requires a clean checkout.")
    changed = git("diff", "--name-only", ORIGINAL, "--", *PROTECTED)
    if changed:
        raise RuntimeError(f"Original scientific source changed: {changed}")
    return {
        "original_commit": ORIGINAL,
        "launch_commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "protected_paths": list(PROTECTED),
        "algorithm_sha256": hashlib.sha256((ROOT / ALG).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((ROOT / MANIFEST).read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((ROOT / "environments/dmcontrol/uv.lock").read_bytes()).hexdigest(),
    }


def recipe(run_name):
    run = json.loads((ROOT / ALG).read_text())
    manifest = json.loads((ROOT / MANIFEST).read_text())
    params = run["alg_params"]
    params["wandb_run_name"] = run_name
    params["wandb_group"] = "original-prior-reproduction-20261030"
    params["wandb_tags"] += ["original-253e89b-reproduction", "hydra-l40", "retain-all-60-checkpoints"]
    return run, manifest


def write_new(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def require_output(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Outputs must be outside the source checkout.")
    if path.exists():
        raise FileExistsError(path)
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"Insufficient storage: {free / 1024**3:.2f} GiB free; need at least 8 GiB.")
    path.mkdir(parents=True, exist_ok=False)
    return path, free


def runtime():
    import importlib.metadata
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one working CUDA GPU is required.")
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA L40":
        raise RuntimeError(f"Expected NVIDIA L40, found {gpu}.")
    return {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": gpu,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("dm-control", "mujoco", "gymnasium", "numpy", "torchrl", "tensordict")},
    }


def runtime_in_child():
    # Release the probe's CUDA context before starting the training process.
    code = ("import json, runpy; "
            f"module = runpy.run_path({str(Path(__file__).resolve())!r}); "
            "print(json.dumps(module['runtime']()))")
    return json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))


def physical_cpus(allowed):
    cores = {}
    for cpu in sorted(allowed):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (int((path / "physical_package_id").read_text()), int((path / "core_id").read_text()))
        cores.setdefault(key, cpu)
    return [cores[key] for key in sorted(cores)]


def choose_resources(cases):
    fastest = min(case["seconds_per_decision"] for case in cases)
    eligible = [case for case in cases if case["seconds_per_decision"] <= fastest * 1.05]
    chosen = min(eligible, key=lambda case: case["cores"])
    # Full-runtime estimate includes a 20% throughput margin plus ten minutes
    # for compilation/pretraining and checkpoint publication.
    seconds = chosen["seconds_per_decision"] * 1_500_000 * 1.20 + 600
    return {"cpus": chosen["cores"], "seconds_per_decision": chosen["seconds_per_decision"],
            "walltime_minutes": math.ceil(seconds / 900) * 15,
            "throughput_margin": 0.20, "startup_io_margin_seconds": 600}


class BenchmarkComplete(Exception):
    pass


def worker(args):
    output = Path(args.output)
    cpus = [int(x) for x in args.cpus.split(",")]
    if not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError("Requested affinity is outside Slurm allocation.")
    os.sched_setaffinity(0, cpus)
    sys.path.insert(0, str(ROOT))
    import torch
    from main import seed_everything, seed_env_spaces
    from utils.core import build_env
    from RL.AMBITDMPC2 import AMBITDMPC2
    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    identity = runtime()
    run, manifest = recipe(f"original-prior-offline-smoke-{len(cpus)}cpu")
    run["alg_params"]["wandb"] = False
    run["alg_params"]["wandb_mode"] = "disabled"
    # Keep the original budget and replay capacity. Stop the original loop at
    # 3500 only after its native checkpoint publication has been requested.
    seed_everything(55)
    env = build_env(run, manifest)
    seed_env_spaces(env, 55)
    learner = None
    start_wall = time.perf_counter()
    measurements = {}
    try:
        learner = AMBITDMPC2(STEM, env, run["alg_params"], run, manifest)
        assert learner.cfg.seed_steps == learner.cfg.pretrain_steps == 2500
        assert learner.cfg.inner_operator == "none" and learner.cfg.inner_model_step_budget == 0
        assert learner.cfg.steps == 1_500_000 and learner.buffer.capacity == 1_000_000
        assert learner.cfg.outer_critic_target == "entropy_augmented"
        assert learner.cfg.outer_q_actor_reduction == "min_pair"
        learner.set_checkpointing(3500, str(output / "checkpoints"), "smoke", save_strat="all")
        act, log_step, checkpoint = learner._act_agent, learner._log_wandb_step, learner._maybe_checkpoint
        first, last = 3001, 3500

        def observed_act(*pos, **kw):
            if learner._global_step + 1 == first:
                torch.cuda.synchronize()
                measurements.update(start=time.perf_counter(), cpu_start=time.process_time(),
                                    updates_start=learner._num_updates,
                                    graphs_before=int(torch._dynamo.utils.counters["stats"]["unique_graphs"]))
            return act(*pos, **kw)

        def observed_log(*pos, **kw):
            result = log_step(*pos, **kw)
            if learner._global_step == last:
                torch.cuda.synchronize()
                measurements.update(seconds=time.perf_counter() - measurements["start"],
                                    cpu_seconds=time.process_time() - measurements["cpu_start"],
                                    updates=learner._num_updates - measurements["updates_start"],
                                    graphs_after=int(torch._dynamo.utils.counters["stats"]["unique_graphs"]))
            return result

        def observed_checkpoint():
            checkpoint()
            if learner._global_step == last:
                raise BenchmarkComplete()

        learner._act_agent, learner._log_wandb_step, learner._maybe_checkpoint = observed_act, observed_log, observed_checkpoint
        try:
            learner.learn(1_500_000)
        except BenchmarkComplete:
            pass
        assert learner._global_step == last and measurements["updates"] == 500
        assert measurements["graphs_before"] == measurements["graphs_after"], "Compilation in measured window"
        assert learner.buffer._storage_device.type == "cuda"
        assert all(torch.isfinite(value).all() for value in learner.agent.model.parameters())
        metrics = learner._last_train_metrics
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
        assert float(metrics.get("compile_fallback", 0)) == 0
        assert float(metrics.get("compile_outer_update_fallback", 0)) == 0
        inner = learner.agent.last_inner_metrics
        assert all(float(inner.get(key, 0)) == 0 for key in
                   ("inner_model_steps", "inner_actor_optimizer_steps", "inner_critic_optimizer_steps"))
        checkpoints = list((output / "checkpoints").glob("smoke_3500"))
        assert len(checkpoints) == 1, "Missing native smoke checkpoint"
        saved = checkpoints[0]
        sidecar = Path(str(saved) + ".metadata.json")
        metadata = json.loads(sidecar.read_text())
        assert metadata["checkpoint"]["step"] == 3500
        # Exercise the original loader on the saved model after the timed run.
        learner.agent.load(str(saved))
        assert all(torch.isfinite(value).all() for value in learner.agent.model.parameters())
        result = {
            "passed": True, "cores": len(cpus), "cpu_ids": cpus, "runtime": identity,
            "measured_decisions": 500, "outer_updates": measurements["updates"],
            "seconds_per_decision": measurements["seconds"] / 500,
            "average_cpu_cores_used": measurements["cpu_seconds"] / measurements["seconds"],
            "total_wall_seconds": time.perf_counter() - start_wall,
            "replay_storage_device": str(learner.buffer._storage_device),
            "replay_capacity": learner.buffer.capacity, "compile_fallback": False,
            "compile_graphs": [measurements["graphs_before"], measurements["graphs_after"]],
            "checkpoint": str(saved), "checkpoint_bytes": saved.stat().st_size,
            "checkpoint_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
            "checkpoint_sidecar": str(sidecar), "checkpoint_load_passed": True,
        }
        write_new(output / "result.json", result)
        print("BENCHMARK_CASE " + json.dumps(result), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def benchmark(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Benchmark must run through Slurm.")
    binding = source_binding()
    output, free = require_output(args.output)
    cores = physical_cpus(os.sched_getaffinity(0))
    if len(cores) < 4:
        raise RuntimeError("Benchmark needs four physical CPUs.")
    cases = []
    for count in (1, 2, 4):
        case = output / f"cpu{count}"
        case.mkdir()
        env = dict(os.environ, OMP_NUM_THREADS=str(count), MKL_NUM_THREADS=str(count),
                   OPENBLAS_NUM_THREADS=str(count), WANDB_MODE="disabled")
        with (case / "stdout.log").open("x") as handle:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(case),
                            "--cpus", ",".join(map(str, cores[:count]))], cwd=ROOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        cases.append(json.loads((case / "result.json").read_text()))
        print("CPU_CASE_COMPLETE " + json.dumps(cases[-1]), flush=True)
    result = {"passed": True, "binding": binding, "free_bytes_before": free,
              "cases": cases, "recommended_resources": choose_resources(cases),
              "scope": "Original early replay training, decisions 3001–3500; excludes W&B and checkpoint I/O from throughput window.",
              "job_id": os.environ["SLURM_JOB_ID"]}
    write_new(output / "benchmark.json", result)
    print("BENCHMARK_COMPLETE " + json.dumps(result["recommended_resources"]), flush=True)


def train(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training must run through Slurm.")
    binding = source_binding()
    gate = json.loads(Path(args.gate).read_text())
    if not gate.get("passed") or gate["binding"] != binding:
        raise RuntimeError("Successful benchmark for this exact source is required.")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if cpus != gate["recommended_resources"]["cpus"]:
        raise RuntimeError("CPU allocation differs from benchmark recommendation.")
    identity = runtime_in_child()
    output, free = require_output(args.output)
    name = f"Original u13m14st reproduction | seed55 | Hydra L40 | job{os.environ['SLURM_JOB_ID']}"
    run, manifest = recipe(name)
    algdir = output / "configs"
    algdir.mkdir()
    write_new(algdir / f"{STEM}.json", run)
    write_new(output / "manifest.json", manifest)
    write_new(output / "launch.json", {"binding": binding, "runtime": identity,
              "job_id": os.environ["SLURM_JOB_ID"], "run_name": name,
              "free_bytes_before": free, "expected_checkpoints": 60,
              "checkpoint_every": 25000, "retain_all": True, "seed": 55,
              "budget_decisions": 1_500_000, "benchmark": str(Path(args.gate).resolve()),
              "resources": gate["recommended_resources"],
              "publication_overrides": {key: run["alg_params"][key] for key in PUBLICATION_KEYS}})
    print("LAUNCH " + json.dumps({"output": str(output), "name": name, "free_gib": free / 1024**3}), flush=True)
    subprocess.run([sys.executable, str(ROOT / "main.py"), "--run", str(output / "manifest.json"),
                    "--alg-dir", str(algdir), "--log-dir", str(output / "artifacts"),
                    "--alg-index", "0", "--trial-index", "0", "--num-runs", "1"], cwd=ROOT, check=True)
    models = [x for x in (output / "artifacts").rglob("model:*") if x.is_file() and not x.name.endswith(".metadata.json")]
    if len(models) != 60 or any(not Path(str(x) + ".metadata.json").is_file() for x in models):
        raise RuntimeError(f"Expected 60 retained checkpoints with sidecars, found {len(models)}.")
    write_new(output / "complete.json", {"checkpoints": len(models), "bytes": sum(x.stat().st_size for x in models)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "worker", "train"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpus")
    parser.add_argument("--gate")
    args = parser.parse_args()
    if args.mode == "train" and not args.gate:
        parser.error("train requires --gate")
    {"benchmark": benchmark, "worker": worker, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
