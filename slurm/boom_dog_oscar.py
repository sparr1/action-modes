"""Validate and launch the sampled BOOM x0.01 Dog recipes on Oscar."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK = os.environ.get("AMBI_DOG_TASK", "run")
if TASK not in ("walk", "trot", "run"):
    raise ValueError("AMBI_DOG_TASK must be walk, trot, or run")
os.environ["AMBI_PRIOR_VARIANT"] = "dog_" + TASK
from slurm import posttanh_prior_fixes_oscar as base

CAMPAIGN = Path("/oscar/scratch/rgao48/ambi/boom-dog-x001-20261113")
OUTPUT = CAMPAIGN / TASK
SOURCE_COMMIT = "bcaa1f59b8e513d7095b1276d6cd2f34144ab18a"
SOURCE_ROOT = Path("/oscar/scratch/rgao48/ambi/posttanh-prior-strength-20261105")


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def locked_state():
    with (CAMPAIGN / "state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(CAMPAIGN / "submission.json")
        yield state
        temporary = CAMPAIGN / f"submission.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(CAMPAIGN / "submission.json")


def arm(state):
    return next(row for row in state["runs"] if row["task"] == TASK)


def phase(value, **updates):
    with locked_state() as state:
        arm(state).update(phase=value, **updates)


def publish():
    subprocess.run(["flock", str(CAMPAIGN / "workspace.lock"), os.environ["AMBI_WORKSPACE_PYTHON"],
                    "slurm/posttanh_prior_fixes_workspace.py", str(CAMPAIGN / "submission.json")],
                   cwd=ROOT, check=True)


def verify_recipe(task=TASK):
    source = read(ROOT / "configs/dmcontrol/algs/ambi_posttanh_prior_outer_adapt_boom_x0p01.json")
    target = read(ROOT / f"configs/dmcontrol/algs/ambi_boom_x0p01_dog_{task}.json")
    source["alg_params"]["outer_boom_coef"] = .00038
    for value in (source, target):
        value["alg_params"] = {k: v for k, v in value["alg_params"].items() if not k.startswith("wandb")}
    assert source == target, "Only the dimension-specific BOOM coefficient may change"
    original = read(ROOT / "configs/dmcontrol/experiments/ambi_posttanh_prior_strength.json")
    manifest = read(ROOT / f"configs/dmcontrol/experiments/ambi_boom_x0p01_dog_{task}.json")
    for key in ("overrides_alg", "trials", "logs", "save_trials", "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert manifest[key] == original[key], key
    assert manifest["configs"] == [f"ambi_boom_x0p01_dog_{task}"]
    assert manifest["env_params"] == {**original["env_params"], "task": "dog-" + task}


def prepare():
    verify_recipe()
    assert base.CAMPAIGN_ROOT == OUTPUT
    protected = ["RL", "domains", "utils", "environments", "main.py",
                 *sorted({node.split("::")[0] for node in base.GPU_PARITY_TESTS})]
    subprocess.run(["git", "diff", "--exit-code", SOURCE_COMMIT, "HEAD", "--", *protected], cwd=ROOT, check=True)
    old = read(SOURCE_ROOT / "gpu-parity.json")
    assert old["binding"]["commit"] == SOURCE_COMMIT
    base.validate_global_parity(old, old["binding"], old["runtime"])
    junit = Path(old["junit_path"])
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old["junit_sha256"]
    assert base.parity_cases(junit) == old["case_names"]
    for relative, digest in [(old["binding"]["manifest"], old["binding"]["manifest_sha256"]),
                             ("environments/dmcontrol/uv.lock", old["binding"]["lock_sha256"]),
                             *[(f"configs/dmcontrol/algs/{name}.json", value) for name, value in old["binding"]["config_sha256"].items()]]:
        assert hashlib.sha256(subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)).hexdigest() == digest
    OUTPUT.mkdir(exist_ok=False)
    for subdir in ("gates", "cache", "production", "slurm"):
        (OUTPUT / subdir).mkdir()
    copied = OUTPUT / "reused-gpu-parity.xml"
    with copied.open("xb") as stream:
        stream.write(junit.read_bytes())
    evidence = {"original_commit": SOURCE_COMMIT, "original_receipt": str(SOURCE_ROOT / "gpu-parity.json"),
                "protected_source_paths": protected, "fresh_checks": "Dog-shape CUDA parity and task-specific 1/2 CPU plus native checks"}
    base.write_new(OUTPUT / "gpu-parity.json", {**old, "binding": base.binding(base.MANIFEST),
                    "original_binding": old["binding"], "junit_path": str(copied), "source_evidence": evidence})
    base.write_new(OUTPUT / "preflight.json", {"binding": base.binding(base.MANIFEST), "source_evidence": evidence})
    print("PREPARED", TASK, flush=True)


def select_plan(rows, reference):
    plan = base.select_plan(rows)
    measured = plan["seconds_per_decision"]
    # Transfer observed end-to-end overhead from the matched Dog baseline,
    # scaled by the relative measured throughput of the new BOOM learner.
    reference_projection = reference["end_to_end_seconds_per_decision"] * measured / reference["benchmark_seconds_per_decision"]
    estimate = max(measured, reference_projection)
    plan.update(wall_hours=math.ceil(estimate * 1000000 * 1.15 / 3600 + .5),
                memory_gib=max(8, math.ceil(max(r["peak_host_memory_bytes"] for r in rows) * 1.25 / 2**30)),
                timing_margin=.15, startup_allowance_hours=.5,
                projected_end_to_end_seconds_per_decision=estimate, sizing_reference=reference)
    return plan


def gate():
    verify_recipe()
    assert os.environ["WANDB_MODE"] == "disabled" and int(os.environ["SLURM_CPUS_PER_TASK"]) == 2
    row = arm(read(CAMPAIGN / "submission.json"))
    assert row["gate_job"] == int(os.environ["SLURM_JOB_ID"])
    phase("running GPU validation and CPU sizing"); publish()
    output = OUTPUT / "gates/0"; output.mkdir(exist_ok=False)
    current = base.accelerated_runtime()
    parity = base.load_global_parity(OUTPUT, current)
    junit = output / "dog-parity.xml"
    program = "import sys,torch,pytest;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('high');raise SystemExit(pytest.main(sys.argv[1:]))"
    nodes = ["tests/test_boom_dog_campaign.py::test_cuda_dog_capture_matches_uncaptured",
             f"tests/test_boom_dog_campaign.py::test_real_dog_matches_two_control_steps[{TASK}]"]
    with (output / "dog-parity.log").open("x") as log:
        subprocess.run([sys.executable, "-c", program, *nodes, "-q", "--junitxml=" + str(junit)],
                       cwd=ROOT, env=dict(os.environ, AMBI_RUN_REAL_DMCONTROL_TESTS="1"), stdout=log, stderr=subprocess.STDOUT, check=True)
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    assert len(cases) == 2 and not any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error"))
    cores = base.physical_core_cpus(os.sched_getaffinity(0)); assert len(cores) >= 2
    rows = []
    for count, native in ((1, False), (2, False), (1, True)):
        path = output / (("native-" if native else "") + f"cpu-{count}.json")
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(count)
        command = [sys.executable, "slurm/posttanh_prior_fixes_oscar.py", "worker", "--index", "0", "--worker-cpus", ",".join(map(str, cores[:count])), "--output", str(path)]
        if native:
            command.append("--native-reference")
        with path.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(path)
        assert report["passed"] and report["runtime"] == current and report["binding"] == base.binding(base.MANIFEST)
        if not native:
            assert report["graph_native_graph_continuation"] and report["gpu_replay"]
            rows.append(report)
    plan = select_plan(rows, row["sizing_reference"])
    space = base.checkpoint_space(output, 3 * 40 * max(r["checkpoint_bytes"] for r in rows))
    receipt = {"passed": True, "binding": base.binding(base.MANIFEST), "runtime": current,
               "plan": plan, "space": space, "cases": [str(output / f"cpu-{n}.json") for n in (1, 2)],
               "native_reference": str(output / "native-cpu-1.json"), "dog_cuda_parity_junit": str(junit),
               "dog_cuda_parity_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
               "graph_native_graph_continuation": True, "source_evidence": parity["source_evidence"],
               "scope": "Actual Dog task, production model and update dose, 128 graph decisions on 1/2 CPUs, 32 native decisions, checkpoint, geometry and continuation checks; W&B disabled."}
    base.write_new(output / "receipt.json", receipt)
    phase("GPU validation passed; awaiting training submission", plan=plan); publish()
    print("BOOM_DOG_GATE_PASSED", TASK, json.dumps(plan), flush=True)


def train():
    verify_recipe()
    row = arm(read(CAMPAIGN / "submission.json"))
    assert row["job"] == int(os.environ["SLURM_JOB_ID"])
    receipt = read(OUTPUT / "gates/0/receipt.json")
    assert receipt["passed"] and receipt["binding"] == base.binding(base.MANIFEST)
    assert hashlib.sha256(Path(receipt["dog_cuda_parity_junit"]).read_bytes()).hexdigest() == receipt["dog_cuda_parity_sha256"]
    phase("training starting"); publish()
    os.environ.update(WANDB_MODE="online", WANDB_RUN_ID=row["id"])
    base.train(SimpleNamespace(index=0, gate=OUTPUT / "gates/0/receipt.json", output=OUTPUT / "production/0"))
    phase("training finished"); publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "gate", "train", "publish"))
    mode = parser.parse_args().mode
    try:
        globals()[mode]()
    except BaseException:
        if mode in ("gate", "train"):
            phase(mode + " failed; inspect Slurm log")
            publish()
        raise
