"""Fixed-scale BOOM training and a finite, reverse-order checkpoint evaluation."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import posttanh_prior_fixes_oscar as base
from slurm import boom_likelihood_pilot_oscar as pilot
from slurm import graph_checkpoint_campaign as evaluation

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_boom_likelihood_calibrated.json"
PILOT_RECEIPT = Path("/oscar/scratch/rgao48/ambi/boom-likelihood-pilot-20261107/gates/0/receipt.json")
GPU_TESTS = pilot.GPU_TESTS + [
    "tests/test_frozen_graph_evaluation.py::test_cuda_mean_graph_matches_control_and_resets_episode_rng",
]


def verify_training_recipe():
    """The new coefficient and publication labels are the only training changes."""
    _, old_manifest, old = base.recipe(0, pilot.MANIFEST)
    _, new_manifest, new = base.recipe(0, MANIFEST)
    old, new = deepcopy(old), deepcopy(new)
    for row in (old, new):
        row.pop("name")
        row["alg_params"].pop("outer_boom_coef")
        for key in list(row["alg_params"]):
            if key.startswith("wandb_"):
                row["alg_params"].pop(key)
    assert old == new, "Calibrated training changed more than its fixed loss coefficient"
    for key in ("overrides_alg", "env_params", "trials", "checkpoint_every", "save_strat"):
        assert old_manifest[key] == new_manifest[key], key


def evaluation_worker(args):
    """No training/publication: snapshot the production architecture, then reload."""
    assert os.environ["WANDB_MODE"] == "disabled"
    cores = [int(x) for x in args.worker_cpus.split(",")]
    os.sched_setaffinity(0, cores)
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2
    from evaluate_ambi_checkpoint import evaluate_matrix

    torch.set_num_threads(len(cores))
    torch.set_num_interop_threads(1)
    runtime = base.accelerated_runtime()
    args.output.mkdir(parents=True, exist_ok=False)
    name, manifest, run = base.recipe(0, MANIFEST)
    run = deepcopy(run)
    run["alg_params"].update(wandb=False, wandb_mode="disabled")
    env = gym.make(run["env"], **manifest["env_params"])
    learner = AMBITDMPC2(name, env, run["alg_params"], run, manifest)
    try:
        checkpoint = Path(learner.save(str(args.output), "benchmark-untrained"))
        learner.flush_checkpoints()
    finally:
        learner.close()
        env.close()
    del learner, env
    gc.collect()
    torch.cuda.empty_cache()
    started, cpu_started = time.perf_counter(), time.process_time()
    payload = evaluate_matrix(evaluation.MATRIX, checkpoint, seeds=[101, 102], max_steps=32,
                              device="cuda", bundle_dir=args.output / "bundle")
    elapsed, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu_started
    evaluation.validate_result(payload, 32, [101, 102])
    # The complete checkpoint/sidecar and return-only bundle stay local for audit;
    # neither benchmark has a W&B run or enters an evaluation curve.
    base.write_new(args.output / "results.json", payload)
    base.write_new(args.output / "receipt.json", dict(
        passed=True, binding=base.binding(MANIFEST), runtime=runtime, cpus=len(cores),
        seconds=elapsed, cpu_seconds=cpu_elapsed, cpu_utilization=cpu_elapsed / elapsed,
        wandb=False, untrained_snapshot=True,
        results=[{k: r[k] for k in ("selector", "return", "episodes", "outer_state_unchanged")}
                 | {"graph_setup_seconds": r.get("graph_setup_seconds", 0)} for r in payload["results"]]))


def evaluation_plan(rows):
    for row in rows:
        row["seconds_per_decision"] = sum(
            sum(ep["control_seconds"] for ep in result["episodes"]) /
            sum(ep["length"] for ep in result["episodes"])
            for result in row["results"])
    best = min(row["seconds_per_decision"] for row in rows)
    chosen = min((row for row in rows if row["seconds_per_decision"] <= 1.05 * best),
                 key=lambda row: row["cpus"])
    # Measured full startup plus five complete paired episodes; 25% margin and
    # one minute for checkpoint/staging I/O, rounded to five-minute blocks.
    seconds = (chosen["seconds"] + 2500 * chosen["seconds_per_decision"]) * 1.25 + 60
    return {"cpus": chosen["cpus"], "seconds_per_decision": chosen["seconds_per_decision"],
            "wall_minutes": math.ceil(seconds / 300) * 5, "checkpoint_count": 16}


def gate(args):
    assert os.environ["WANDB_MODE"] == "disabled"
    verify_training_recipe()
    old = json.loads(PILOT_RECEIPT.read_text())
    assert old["passed"] and old["plan"]["decisions"] == 400000
    # The sizing data may be reused only with unchanged training kernels.
    changes = subprocess.check_output(["git", "diff", "--name-only", old["binding"]["commit"], "HEAD", "--",
                                       "RL", "domains", "main.py", "environments/dmcontrol/uv.lock"], text=True).splitlines()
    assert set(changes) <= {"RL/tdmpc2_core/frozen_graph.py"}, changes
    args.output.mkdir(parents=True, exist_ok=False)
    junit = args.output / "cuda.xml"
    program = ("import sys,torch,pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); "
               "torch.set_float32_matmul_precision('high'); raise SystemExit(pytest.main(sys.argv[1:]))")
    with (args.output / "cuda.log").open("x") as log:
        subprocess.run([sys.executable, "-c", program, *GPU_TESTS, "-q", "--junitxml=" + str(junit)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    assert len(cases) == len(GPU_TESTS)
    assert not any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error"))
    cores = base.physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2
    rows = []
    for count in (1, 2):
        output = args.output / f"eval-cpu-{count}"
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(count)
        with output.with_suffix(".log").open("x") as log:
            subprocess.run([sys.executable, __file__, "evaluation-worker", "--output", str(output),
                            "--worker-cpus", ",".join(map(str, cores[:count]))],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        row = json.loads((output / "receipt.json").read_text())
        assert row["passed"] and row["runtime"] == old["runtime"] and not row["wandb"]
        rows.append(row)
    base.write_new(args.output / "receipt.json", dict(
        passed=True, binding=base.binding(MANIFEST), runtime=old["runtime"], training_plan=old["plan"],
        reused_training_sizing=str(PILOT_RECEIPT), evaluation_plan=evaluation_plan(rows),
        cuda_cases=[case.attrib["name"] for case in cases],
        junit_sha256=hashlib.sha256(junit.read_bytes()).hexdigest(), wandb=False))
    print("CALIBRATED_GATE_PASSED", flush=True)


def train(args):
    import torch
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    verify_training_recipe()
    gate = json.loads(args.gate.read_text())
    assert gate["passed"]
    current = base.binding(MANIFEST)
    if gate["binding"] != current:
        # Calibration evidence can arrive while GPU validation runs. A later
        # coefficient/label-only commit may reuse the same tested kernels.
        previous = gate["binding"]["commit"]
        changes = subprocess.check_output(["git", "diff", "--name-only", previous, "HEAD"], text=True).splitlines()
        allowed = {str(MANIFEST.relative_to(ROOT)), "configs/research/README.md",
                   "tests/test_boom_calibrated_campaign.py",
                   "configs/dmcontrol/algs/ambi_boom_likelihood_outer_adapt_calibrated.json"}
        assert set(changes) <= allowed, changes
        config_path = "configs/dmcontrol/algs/ambi_boom_likelihood_outer_adapt_calibrated.json"
        old_config = json.loads(subprocess.check_output(["git", "show", previous + ":" + config_path], text=True))
        new_config = json.loads((ROOT / config_path).read_text())
        for config in (old_config, new_config):
            config["alg_params"].pop("outer_boom_coef")
            for key in list(config["alg_params"]):
                if key.startswith("wandb_"):
                    config["alg_params"].pop(key)
        assert old_config == new_config and gate["binding"]["lock_sha256"] == current["lock_sha256"]
    assert base.accelerated_runtime() == gate["runtime"]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == gate["training_plan"]["cpus"]
    assert os.environ["WANDB_MODE"] == "online" and os.environ["WANDB_RUN_ID"]
    args.output.mkdir(parents=True, exist_ok=False)
    space = base.checkpoint_space(args.output, 16 * gate["training_plan"]["checkpoint_bytes"])
    base.write_new(args.output / "launch.json", dict(
        binding=base.binding(MANIFEST), gate=str(args.gate), plan=gate["training_plan"],
        runtime=gate["runtime"], space=space, job_id=os.environ["SLURM_JOB_ID"],
        wandb_run_id=os.environ["WANDB_RUN_ID"]))
    sys.argv = ["main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
                "--log-dir", str(args.output), "--alg-index", "0", "--trial-index", "0", "--num-runs", "1"]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gate", "evaluation-worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--worker-cpus")
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    {"gate": gate, "evaluation-worker": evaluation_worker, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
