"""Size and run exact-resume Oscar allocations for the Q10 tasks."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_transfer_training import binding, read, recipe, write_new
from slurm.inner_entropy_oscar import accelerated_runtime, slurm_fields


CAMPAIGN = Path("/oscar/scratch/rgao48/ambi/mrsq-q10-suite-20261122")
# Completed matching Q5 times scaled by the measured Dog Stand/Walk Q10 ratios.
PREDICTED_COMPUTE_HOURS = {"humanoid-walk": 84, "humanoid-run": 84, "dog-trot": 85, "dog-run": 86,
                           "humanoid-stand": 85, "dog-walk": 86}
SEED56_TASKS = ("dog-run", "humanoid-run", "humanoid-stand")
MAX_SECONDS_PER_DECISION = 0.38
TIMING_MARGIN = 0.15
STARTUP_ALLOWANCE_HOURS = 0.5


def manifest_path(task, seed=55):
    assert seed == 55 or (seed == 56 and task in SEED56_TASKS)
    return ROOT / "configs/dmcontrol/experiments" / f"ambi_{task.replace('-', '_')}_mrsq_q10_seed{seed}.json"


def validate_recipe(task, seed=55):
    name, manifest, run = recipe(0, manifest_path(task, seed))
    assert manifest["configs"] == [name] and manifest["trials"] == 1
    assert manifest["env_params"] == {"task": task, "obs": "state", "render_mode": None}
    assert run["seed"] == seed and run["total_steps"] == 1_000_000
    assert run["checkpoint_every"] == 25_000 and run["save_strat"] == "all"
    cfg = run["alg_params"]
    required = {"num_q": 10, "q_pair_size": 10, "outer_q_actor_reduction": "mean_all",
                "inner_q_actor_reduction": "mean_all", "outer_q_target_reduction": "min_all",
                "inner_q_target_reduction": "min_all", "mppi_terminal_q_reduction": "min_all",
                "compile": True, "compile_strict": True, "inner_cuda_graph": True,
                "critic_ensemble_backend": "vmap", "critic_geometry_diagnostics": True,
                "critic_geometry_every": 1000, "wandb": True, "wandb_mode": "online"}
    assert all(cfg[key] == value for key, value in required.items())
    return name, run


def check_speed(report, task, *, seed=55, limit=MAX_SECONDS_PER_DECISION):
    assert report["passed"] and report["binding"] == binding(manifest_path(task, seed))
    assert report["runtime"]["gpu"] == "NVIDIA L40S"
    assert report["runtime"]["float32_matmul_precision"] == "high"
    assert report["cpu_replay"] and not report["gpu_replay"]
    assert report["wandb"] is False and report["compile_fallback"] is False
    assert report["graph_replays"]["timed"] == report["measured"]["decisions"] == 128
    assert report["compile_counts"]["before"] == report["compile_counts"]["after"]
    assert report["checkpoint_roundtrip"] and report["critic_geometry"]["passed"]
    elapsed = report["measured"]["seconds_per_decision"]
    if not math.isfinite(elapsed) or not 0 < elapsed <= limit:
        raise RuntimeError(f"{task} measured {elapsed:.4f} s/decision above the {limit:.4f} limit")


def measurement(task, cpus, path, *, seed=55, measure_resume_storage=False):
    command = [sys.executable, str(Path(__file__).resolve()), "benchmark-worker", task,
               "--seed", str(seed), "--output", str(path), "--worker-cpus", ",".join(map(str, cpus))]
    if measure_resume_storage:
        command.append("--measure-resume-storage")
    subprocess.run(command, cwd=ROOT, check=True)
    result = read(path)
    assert result["physical_cpus"] == len(cpus)
    return result


def verify_parity(task, path):
    family = task.split("-", 1)[0]
    environment_test = ("tests/test_dog_walk_mrsq.py::test_real_dog_walk_two_control_steps"
                        if task == "dog-walk" else f"tests/test_mrsq_q5_suite.py::test_real_task_action_repeat[{task}]")
    program = ("import sys,torch,pytest; torch.set_num_threads(1); "
               "torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('high'); "
               "raise SystemExit(pytest.main(sys.argv[1:]))")
    subprocess.run([sys.executable, "-c", program, "-q",
                    f"tests/test_mrsq_q10_suite.py::test_cuda_q10_family_parity[{family}]",
                    environment_test,
                    f"--junitxml={path}"], cwd=ROOT,
                   env=dict(os.environ, AMBI_RUN_REAL_DMCONTROL_TESTS="1"), check=True)
    cases = list(ET.parse(path).getroot().iter("testcase"))
    assert len(cases) == 2
    assert all(case.find(problem) is None for case in cases for problem in ("failure", "error", "skipped"))


def size_allocation(rows, task):
    def seconds_per_decision(row):
        return row["measured"]["seconds_per_decision"]

    eligible = [row for row in rows if seconds_per_decision(row) <= MAX_SECONDS_PER_DECISION]
    if not eligible:
        raise RuntimeError(f"{task} has no eligible CPU/GPU timing within {MAX_SECONDS_PER_DECISION:.3f} s/decision")
    fastest = min(map(seconds_per_decision, eligible))
    selected = min((row for row in eligible if seconds_per_decision(row) <= 1.05 * fastest),
                   key=lambda row: row["physical_cpus"])
    rate = seconds_per_decision(selected)
    # Oscar Q10 Dog Stand 9256526 completed in 304091 seconds. Its short
    # L40S training gate measured 0.302855 seconds per decision.
    overhead = 304_091 / (0.302855 * 1_000_000)
    projected_hours = max(PREDICTED_COMPUTE_HOURS[task], rate * 1_000_000 * overhead / 3600)
    peak_bytes = max(row["peak_host_memory_bytes"] for row in rows)
    plan = {"cpus": selected["physical_cpus"], "seconds_per_decision": rate,
            "memory_gib": max(8, math.ceil(peak_bytes * 1.25 / 2**30 + 2)),
            "wall_hours": math.ceil(projected_hours * (1 + TIMING_MARGIN) + STARTUP_ALLOWANCE_HOURS),
            "timing_margin": TIMING_MARGIN, "startup_allowance_hours": STARTUP_ALLOWANCE_HOURS,
            "observed_baseline_overhead_factor": overhead, "checkpoint_bytes": selected["checkpoint_bytes"]}
    return selected, plan


def benchmark(task, task_root):
    from slurm.benchmark_ambi_training_cpu import physical_core_cpus

    assert os.environ["WANDB_MODE"] == "disabled"
    cpus = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cpus) >= 2
    gate = task_root / "benchmark"
    assert not gate.exists()
    job = os.environ["SLURM_JOB_ID"]
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    output = task_root / "benchmark-attempts" / f"{job}.{restart}"
    output.mkdir(parents=True, exist_ok=False)
    parity = output / "parity.xml"
    verify_parity(task, parity)
    rows = [measurement(task, cpus[:count], output / f"cpu-{count}.json", measure_resume_storage=True)
            for count in (1, 2)]
    for row in rows:
        check_speed(row, task, limit=math.inf)
    assert rows[0]["runtime"] == rows[1]["runtime"]
    selected, plan = size_allocation(rows, task)
    plan.update(passed=True, task=task, binding=binding(manifest_path(task)), runtime=rows[0]["runtime"],
                cpu_replay=True, wandb=False,
                max_seconds_per_decision=min(MAX_SECONDS_PER_DECISION, 1.2 * plan["seconds_per_decision"]),
                parity_sha256=hashlib.sha256(parity.read_bytes()).hexdigest(),
                gate_job=job)
    resume_storage = selected["resume_storage"]
    plan["resume_storage"] = resume_storage
    plan["full_generation_bytes"] = sum(resume_storage[key] for key in (
        "trainer_agent_bytes", "replay_metadata_measured_bytes", "replay_full_estimated_bytes"))
    plan["full_generation_bytes_scope"] = "Serialized agent and full-capacity replay estimate, excluding cumulative W&B history and wrapper metadata"
    write_new(output / "plan.json", plan)
    output.rename(gate)
    print("Q10_BENCHMARK_PASSED", json.dumps(plan), flush=True)


def load_plan(task, output, *, seed=55):
    gate = output / "benchmark"
    plan = read(gate / "plan.json")
    assert plan["passed"] and plan["task"] == task and plan["binding"] == binding(manifest_path(task, seed))
    if seed == 56:
        assert plan["sizing_reference"]["seed"] == 55 and plan["cpus"] == 1, plan
    assert plan["cpu_replay"] and plan["wandb"] is False
    assert hashlib.sha256((gate / "parity.xml").read_bytes()).hexdigest() == plan["parity_sha256"]
    return plan


def allocation(task, output, *, seed=55):
    plan = load_plan(task, output, seed=seed)
    fields = slurm_fields(os.environ["SLURM_JOB_ID"])
    count = str(plan["cpus"])
    assert count == os.environ["SLURM_CPUS_PER_TASK"]
    assert all(fields[key] == count for key in ("NumCPUs", "CPUs/Task", "MinCPUsNode"))
    assert fields["TresPerTask"] in (f"cpu={count}", f"cpu:{count}")
    assert fields["MinMemoryNode"] in (f"{plan['memory_gib']}G", f"{plan['memory_gib'] * 1024}M")
    assert fields["Partition"] == "gpu" and "gres/gpu:l40s:1" in fields["TresPerNode"].split(",")
    assert fields["Requeue"] == "1"
    hours = plan["wall_hours"]
    days, hours_of_day = divmod(hours, 24)
    expected = f"{days}-{hours_of_day:02d}:00:00" if days else f"{hours:02d}:00:00"
    assert fields["TimeLimit"] == expected, (fields["TimeLimit"], expected)
    assert int(os.environ.get("AMBI_TRAIN_WALLTIME_SECONDS", hours * 3600)) == hours * 3600
    print(hours * 3600)


def node_check(task, output, receipt, *, seed=55):
    from slurm.benchmark_ambi_training_cpu import physical_core_cpus

    assert os.environ["WANDB_MODE"] == "disabled"
    plan = load_plan(task, output, seed=seed)
    cpus = physical_core_cpus(os.sched_getaffinity(0))
    count = int(os.environ["SLURM_CPUS_PER_TASK"])
    assert plan["cpus"] == count and len(cpus) >= count
    receipt.parent.mkdir(parents=True, exist_ok=True)
    report = measurement(task, cpus[:count], receipt, seed=seed)
    assert report["runtime"] == plan["runtime"]
    check_speed(report, task, seed=seed, limit=plan["max_seconds_per_decision"])
    print(f"Q10_NODE_THROUGHPUT_PASSED {task} {report['measured']['seconds_per_decision']:.6f}", flush=True)


def task_path(task, campaign=CAMPAIGN, *, seed=55):
    return campaign / (task if seed == 55 else f"{task}-s{seed}")


def lineage_path(task, campaign=CAMPAIGN, *, seed=55):
    return task_path(task, campaign, seed=seed) / "lineage"


def train(args, task_root):
    import torch

    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "online"
    name, run = validate_recipe(args.task, args.seed)
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    runtime = accelerated_runtime()
    assert runtime == load_plan(args.task, task_root, seed=args.seed)["runtime"]
    lineage = lineage_path(args.task, seed=args.seed)
    snapshot_root = task_root / "models"
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "launch.json", {
        "binding": binding(manifest_path(args.task, args.seed)), "task": args.task, "config": name,
        "seed": run["seed"], "runtime": runtime, "resume_mode": args.resume_mode,
        "lineage_dir": str(lineage), "snapshot_root": str(snapshot_root),
        "resume_checkpoint_minutes": 15, "drain_after_seconds": args.drain_seconds,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "segment_id": os.environ["AMBI_SEGMENT_ID"],
    })
    sys.argv = ["main.py", "--run", str(manifest_path(args.task, args.seed)), "--alg-dir", "configs/dmcontrol/algs",
                "--num-runs", "1", "--log-dir", str(args.output), "--lineage-dir", str(lineage),
                "--resume-mode", args.resume_mode, "--resume-wandb-mode", "online",
                "--resume-model-snapshot-root", str(snapshot_root), "--resume-checkpoint-minutes", "15",
                "--drain-after-seconds", str(args.drain_seconds)]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def record_handoff(task, output, *, seed=55):
    lineage = lineage_path(task, seed=seed)
    generation = (lineage / "LATEST").read_text().strip()
    generation_file = lineage / "generations" / generation / "manifest.json"
    metadata = read(generation_file)["metadata"]
    assert metadata["global_step"] > 2500 and metadata["num_updates"] > 0 and metadata["replay_rows"] > 0
    identity = read(lineage / "LINEAGE.json")["metadata"]
    assert metadata["wandb_run_id"] == identity["initial_wandb_run_id"]
    receipt = {"task": task, "generation": generation,
              "run_id": identity["initial_wandb_run_id"], "training_id": identity["training_id"],
              "resume_file_bytes": sum(file.stat().st_size for file in generation_file.parent.rglob("*") if file.is_file()),
              **metadata}
    write_new(output / "bootstrap-handoff.json", receipt)
    print("Q10_BOOTSTRAP_HANDOFF", json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "benchmark-worker", "allocation", "node-check", "record-handoff", "train"))
    parser.add_argument("task", choices=PREDICTED_COMPUTE_HOURS)
    parser.add_argument("--seed", type=int, choices=(55, 56), default=55)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-cpus")
    parser.add_argument("--measure-resume-storage", action="store_true")
    parser.add_argument("--resume-mode", choices=("new", "required"))
    parser.add_argument("--drain-seconds", type=int)
    args = parser.parse_args()
    assert Path(os.environ["AMBI_Q10_CAMPAIGN"]) == CAMPAIGN
    task_root = task_path(args.task, seed=args.seed)
    validate_recipe(args.task, args.seed)
    if args.mode == "benchmark-worker":
        from slurm.posttanh_dog_trot_resume import benchmark_worker

        assert os.environ["WANDB_MODE"] == "disabled" and args.worker_cpus and args.output
        args.index, args.native_reference = 0, False
        benchmark_worker(args, manifest_path=manifest_path(args.task, args.seed),
                         measure_resume_storage=args.measure_resume_storage)
    elif args.mode == "benchmark":
        assert args.seed == 55, "Seed 56 uses the measured seed-55 timing and an assigned-GPU check"
        benchmark(args.task, task_root)
    elif args.mode == "allocation":
        allocation(args.task, task_root, seed=args.seed)
    elif args.mode == "node-check":
        assert args.output is not None and args.output.parent == task_root / "node-checks"
        node_check(args.task, task_root, args.output, seed=args.seed)
    elif args.mode == "record-handoff":
        record_handoff(args.task, task_root, seed=args.seed)
    else:
        assert args.resume_mode and args.output and args.output.parent == task_root / "launches"
        assert args.drain_seconds and args.drain_seconds > 0
        train(args, task_root)


if __name__ == "__main__":
    main()
