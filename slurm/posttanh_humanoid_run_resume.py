"""Restart Humanoid Run seed57 with the native full-state resume contract."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_humanoid_run_seed57_resume.json"
DURABLE_ROOT = Path("/oscar/home/rgao48/ambi-durable")


def validate_recipe():
    from slurm.ambi_transfer_training import recipe
    from slurm.posttanh_humanoid_seed_replicates import validate_recipe as original_recipe

    _, original_manifest, original = original_recipe("run", 1)
    name, manifest, run = recipe(0, MANIFEST)
    comparable = deepcopy(run)
    comparable["name"] = original["name"]
    for key in ("wandb_run_name", "wandb_tags", "wandb_group"):
        comparable["alg_params"][key] = original["alg_params"][key]
    assert comparable == original, "Restart changed scientific parameters"
    assert manifest["configs"] == [name]
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert manifest[key] == original_manifest[key], key
    return name, manifest, run


def train(args):
    import torch
    from slurm.ambi_transfer_training import binding, write_new
    from slurm.inner_entropy_oscar import accelerated_runtime
    from slurm.posttanh_humanoid_seed_replicates import validate_reference

    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "online"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == os.environ["AMBI_EXPECTED_COMMIT"]
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip()
    name, _, run = validate_recipe()
    reference_path, reference = validate_reference("run")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    runtime = accelerated_runtime()
    assert runtime == reference["runtime"], (runtime, reference["runtime"])
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"]).resolve()
    assert lineage.is_relative_to(DURABLE_ROOT.resolve())
    assert args.output.is_absolute() and not args.output.is_relative_to(ROOT)
    assert args.snapshot_root.is_absolute() and not args.snapshot_root.is_relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "launch.json", {
        "binding": binding(MANIFEST), "config": name, "seed": run["seed"],
        "reference_gate": str(reference_path), "runtime": runtime,
        "resume_mode": args.resume_mode, "lineage_dir": str(lineage),
        "resume_checkpoint_minutes": args.checkpoint_minutes,
        "drain_after_seconds": args.drain_seconds,
        "snapshot_root": str(args.snapshot_root),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "segment_id": os.environ["AMBI_SEGMENT_ID"],
    })
    sys.argv = ["main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
                "--num-runs", "1", "--log-dir", str(args.output),
                "--lineage-dir", str(lineage), "--resume-mode", args.resume_mode,
                "--resume-wandb-mode", "online", "--resume-model-snapshot-root", str(args.snapshot_root),
                "--resume-checkpoint-minutes", str(args.checkpoint_minutes),
                "--drain-after-seconds", str(args.drain_seconds)]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def benchmark(args):
    from slurm.ambi_transfer_training import write_new
    from slurm.benchmark_ambi_training_cpu import physical_core_cpus

    assert os.environ.get("WANDB_MODE") == "disabled"
    cpus = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cpus) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for count in (1, 2):
        path = args.output / f"cpu{count}.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "benchmark-worker",
                        "--output", str(path), "--worker-cpus", ",".join(map(str, cpus[:count]))],
                       cwd=ROOT, check=True)
        reports.append(json.loads(path.read_text()))
    fastest = min(row["measured"]["seconds_per_decision"] for row in reports)
    selected = min((row for row in reports if row["measured"]["seconds_per_decision"] <= fastest * 1.05),
                   key=lambda row: row["physical_cpus"])
    write_new(args.output / "plan.json", {"cpus": selected["physical_cpus"],
              "seconds_per_decision": selected["measured"]["seconds_per_decision"],
              "binding": selected["binding"], "runtime": selected["runtime"],
              "cpu_replay": True, "wandb": False})


def verify_allocation(args):
    from slurm.ambi_transfer_training import binding
    plan = json.loads((args.output / "plan.json").read_text())
    assert plan["binding"] == binding(MANIFEST)
    assert plan["cpu_replay"] and plan["wandb"] is False
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == plan["cpus"]
    value = subprocess.check_output(["squeue", "--noheader", "--jobs", os.environ["SLURM_JOB_ID"],
                                     "--format", "%l"], text=True).strip()
    days, clock = value.split("-") if "-" in value else ("0", value)
    parts = list(map(int, clock.split(":")))
    if len(parts) == 2:
        parts.insert(0, 0)
    assert len(parts) == 3
    seconds = int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    assert seconds == int(os.environ["AMBI_TRAIN_WALLTIME_SECONDS"])


def record_handoff(args):
    from slurm.ambi_transfer_training import write_new
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    generation = (lineage / "LATEST").read_text().strip()
    metadata = json.loads((lineage / "generations" / generation / "manifest.json").read_text())["metadata"]
    # The first allocation must exercise the trained state, not only genesis.
    assert metadata["global_step"] > 2500 and metadata["num_updates"] > 0, metadata
    assert metadata["replay_rows"] > 0
    write_new(args.output / "bootstrap-handoff.json", {"generation": generation, **metadata})


def publish(args):
    from slurm.ambi_transfer_training import write_new
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    metadata = json.loads((lineage / "LINEAGE.json").read_text())["metadata"]
    path = args.output / "publication.json"
    if not path.exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        name, _, run = validate_recipe()
        write_new(path, {
            "commit": commit, "output": str(args.output),
            "workspace_name": "Humanoid Run — seed57 resumable restart",
            "protocol": (
                "**Humanoid Run: fresh seed57 restart with full-state resume.**\n\n"
                "Same 1,000,000-decision H3/J8/K1, N128/C16/A4/B256, fixed post-tanh alpha0.0021 "
                "recipe as the interrupted seed57 run. Sampled execution, reward-only critics, "
                "strict compilation, whole-inner CUDA graphs, vmap and TF32-high are preserved. "
                "Critic geometry remains enabled every1,000 outer updates. Native resume uses CPU replay; "
                "this fresh random trajectory need not match the old GPU-replay run.\n\n"
                "Full trainer, replay, optimizer, environment and RNG state is committed every15minutes "
                "at an episode boundary; latest and predecessor are retained in Oscar home storage. "
                "All40 portable model checkpoints are retained every25k decisions in scratch. "
                "The first real training segment is checkpointed and resumed in the same allocation. "
                "Time-limit handoffs are verified before automatic requeue, preserving one W&B run. "
                "Curves update only after their state is durably checkpointed.\n\n"
                f"Source `{commit}`. Lineage `{lineage}`."
            ),
            "runs": [{"id": metadata["initial_wandb_run_id"], "seed": 57,
                      "name": run["alg_params"]["wandb_run_name"], "label": "seed57 resumable restart",
                      "color": "#D55E00", "job": int(os.environ["SLURM_JOB_ID"])}],
            "controls": [
                {"id": "66c28347bd394e8bb8fe7ab7276d94be", "label": "seed55 reference", "color": "#777777"},
                {"id": "7a3481f825e741568cf861afb83e03f6", "label": "seed56 complete", "color": "#0072B2"},
                {"id": "836f419c8d854ed7a250bb91426468ea", "label": "seed57 timed out (original)", "color": "#CC79A7"},
            ],
        })
    subprocess.run([os.environ["AMBI_WORKSPACE_PYTHON"], "slurm/inner_entropy_workspace.py", str(path)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "benchmark-worker", "allocation", "record-handoff", "publish", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-cpus")
    parser.add_argument("--resume-mode", choices=("new", "required"))
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--drain-seconds", type=int)
    parser.add_argument("--checkpoint-minutes", type=int, default=15)
    args = parser.parse_args()
    validate_recipe()
    if args.mode == "benchmark-worker":
        assert os.environ.get("WANDB_MODE") == "disabled"
        from slurm import inner_entropy_oscar
        inner_entropy_oscar.MANIFEST = MANIFEST
        args.index, args.native_reference, args.cpu_replay = 0, False, True
        inner_entropy_oscar.worker(args)
    elif args.mode == "train":
        if args.resume_mode is None or args.snapshot_root is None or not args.drain_seconds or args.drain_seconds <= 0:
            parser.error("train requires resume mode, snapshot root and positive drain seconds")
        if args.checkpoint_minutes <= 0:
            parser.error("checkpoint minutes must be positive")
        train(args)
    else:
        {"benchmark": benchmark, "allocation": verify_allocation,
         "record-handoff": record_handoff, "publish": publish}[args.mode](args)


if __name__ == "__main__":
    main()
