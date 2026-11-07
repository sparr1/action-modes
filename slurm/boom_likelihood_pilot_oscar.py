"""Validate, size, and run the 400k BOOM replay-likelihood diagnostic pilot."""
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
from slurm import posttanh_prior_fixes_oscar as base

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_boom_likelihood_pilot.json"
CAMPAIGN = Path("/oscar/scratch/rgao48/ambi/boom-likelihood-pilot-20261107")
GPU_TESTS = [
    "tests/test_ambi_boom_acceleration.py::test_boom_reduce_overhead_cuda_uses_fresh_samples_and_survives_restore[replay_likelihood-scalar]",
    "tests/test_ambi_boom_acceleration.py::test_boom_reduce_overhead_cuda_uses_fresh_samples_and_survives_restore[replay_likelihood-distributional]",
    "tests/test_boom_replay_likelihood.py::test_cuda_whole_inner_graph_preserves_executed_preimages_through_native_diagnostics",
    "tests/test_boom_diagnostics.py::test_cuda_diagnostics_preserve_compiled_adam_rng_and_graph_storage",
]


def parity_cases(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    names = [case.attrib["name"] for case in cases]
    assert sorted(names) == sorted(node.split("::")[-1] for node in GPU_TESTS), names
    assert not any(case.find(tag) is not None for case in cases for tag in ("skipped", "error", "failure"))
    return names


def select_plan(rows):
    plan = base.select_plan(rows)
    plan["decisions"] = 400000
    plan["wall_hours"] = math.ceil(plan["seconds_per_decision"] * 400000 * 1.25 / 3600 + .5)
    return plan


def release_production(receipt, campaign=CAMPAIGN):
    """Release only this campaign's held dependent job, after fresh evidence."""
    state = json.loads((campaign / "submission.json").read_text())
    arm = state["runs"][0]
    assert receipt["passed"] and state["commit"] == receipt["binding"]["commit"]
    assert len(state["runs"]) == 1 and arm["index"] == 0
    assert str(arm["gate_job"]) == os.environ["SLURM_JOB_ID"]
    job = int(arm["job"])
    fields = base.slurm_fields(job)
    assert fields["Dependency"] == f"afterok:{arm['gate_job']}(unfulfilled)", fields
    applied = base.resize_pending(job, receipt["plan"])
    base.write_new(campaign / "allocation.json", {"production_job": job, "plan": receipt["plan"], "verified": applied})
    print("PRODUCTION_RELEASED", job, flush=True)


def gate(args):
    assert os.environ.get("WANDB_MODE") == "disabled"
    cores = base.physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    junit = args.output / "cuda.xml"
    program = ("import sys,torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); "
               "torch.set_float32_matmul_precision('high'); raise SystemExit(pytest.main(sys.argv[1:]))")
    with (args.output / "cuda.log").open("x") as log:
        subprocess.run([sys.executable, "-c", program, *GPU_TESTS, "-q", "--junitxml=" + str(junit)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    names = parity_cases(junit)
    rows = []
    for count in (1, 2):
        output = args.output / f"cpu-{count}.json"
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(count)
        with output.with_suffix(".log").open("x") as log:
            subprocess.run([sys.executable, __file__, "worker", "--worker-cpus", ",".join(map(str, cores[:count])),
                            "--output", str(output)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        row = json.loads(output.read_text())
        assert row["passed"] and row["boom_gradient_probes"] and not row["wandb"]
        rows.append(row)
    receipt = {
        "passed": True, "binding": base.binding(MANIFEST), "runtime": rows[0]["runtime"],
        "plan": select_plan(rows), "cases": [str(args.output / f"cpu-{n}.json") for n in (1, 2)],
        "space": base.checkpoint_space(args.output, 16 * max(row["checkpoint_bytes"] for row in rows)),
        "cuda_cases": names, "junit_path": str(junit),
        "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
        "graph_native_graph_continuation": all(row["graph_native_graph_continuation"] for row in rows),
        "job_id": os.environ["SLURM_JOB_ID"], "wandb": False,
        "scope": "Fresh CUDA likelihood/diagnostic parity; real H3/J8/K1 Humanoid training on 1/2 physical CPUs, GPU replay, compiled outer update, whole-inner graph, geometry and checkpoints."
    }
    base.write_new(args.output / "receipt.json", receipt)
    print("GATE_PASSED", json.dumps(receipt["plan"]), flush=True)
    if args.release_production:
        release_production(receipt)


def train(args):
    import torch
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    receipt = json.loads(args.gate.read_text())
    assert receipt["passed"] and receipt["binding"] == base.binding(MANIFEST)
    assert receipt["graph_native_graph_continuation"]
    current = base.accelerated_runtime()
    assert current == receipt["runtime"], (current, receipt["runtime"])
    junit = Path(receipt["junit_path"])
    assert junit.is_relative_to(CAMPAIGN)
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == receipt["junit_sha256"]
    assert parity_cases(junit) == receipt["cuda_cases"]
    plan = receipt["plan"]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == plan["cpus"]
    assert os.environ.get("WANDB_RUN_ID") and os.environ["WANDB_MODE"] == "online"
    args.output.mkdir(parents=True, exist_ok=False)
    space = base.checkpoint_space(args.output, receipt["space"]["checkpoint_bank_bytes"])
    base.write_new(args.output / "launch.json", {
        "binding": base.binding(MANIFEST), "plan": plan, "runtime": current,
        "output": str(args.output), "gate": str(args.gate), "space": space,
        "job_id": os.environ["SLURM_JOB_ID"], "wandb_run_id": os.environ["WANDB_RUN_ID"],
    })
    sys.argv = ["main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
                "--log-dir", str(args.output), "--alg-index", "0", "--trial-index", "0", "--num-runs", "1"]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gate", "worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--worker-cpus")
    parser.add_argument("--release-production", action="store_true")
    args = parser.parse_args()
    args.index, args.native_reference = 0, False
    assert Path(os.environ["AMBI_PRIOR_CAMPAIGN"]) == CAMPAIGN
    assert os.environ.get("SLURM_JOB_ID")
    base.MANIFEST, base.CAMPAIGN_ROOT = MANIFEST, CAMPAIGN
    {"gate": gate, "worker": base.worker, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
