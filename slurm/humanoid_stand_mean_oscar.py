"""Launch the unchanged mean-execution BOOM learner on Humanoid Stand."""
import argparse
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
os.environ["AMBI_PRIOR_VARIANT"] = "humanoid_stand_mean"
from slurm import posttanh_prior_fixes_oscar as base

SOURCE_COMMIT = "7427e1c681cbc3f2b2fbc3109855afc7d720e57b"
SOURCE_ROOT = Path("/oscar/scratch/rgao48/ambi/boom-humanoid-run-mean-20261108")
CPU_ROOT = Path("/oscar/scratch/rgao48/ambi/posttanh-prior-fixes-20261104/gates/2")
REFERENCE_SECONDS = 181042.10853
MEAN_TESTS = [
    "tests/test_humanoid_run_mean.py::test_cuda_mean_capture_preserves_inner_state",
    "tests/test_humanoid_run_mean.py::test_cuda_mean_graph_native_graph_and_checkpoint_continuation",
]


def read(path):
    return json.loads(Path(path).read_text())


def verify_recipe():
    folder = ROOT / "configs/dmcontrol"
    source = read(folder / "algs/ambi_boom_x0p01_humanoid_run_mean.json")
    target = read(folder / "algs/ambi_boom_x0p01_humanoid_stand_mean.json")
    assert source["alg_params"]["inner_execution_action"] == "mean"
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items()
                                if not k.startswith("wandb")}
    assert source == target
    original = read(folder / "experiments/ambi_boom_x0p01_humanoid_run_mean.json")
    original["env_params"]["task"] = "humanoid-stand"
    manifest = read(base.MANIFEST)
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key], key
    assert manifest["configs"] == ["ambi_boom_x0p01_humanoid_stand_mean"]


def prepare():
    verify_recipe()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_PRIOR_CAMPAIGN"])
    protected = ["RL", "domains", "utils", "environments", "main.py",
                 "tests/test_humanoid_run_mean.py",
                 *sorted({node.split("::")[0] for node in base.GPU_PARITY_TESTS})]
    subprocess.run(["git", "diff", "--exit-code", SOURCE_COMMIT, "HEAD", "--", *protected], cwd=ROOT, check=True)
    old = read(SOURCE_ROOT / "gpu-parity.json")
    assert old["binding"]["commit"] == SOURCE_COMMIT
    base.validate_global_parity(old, old["binding"], old["runtime"])
    for relative, digest in [(old["binding"]["manifest"], old["binding"]["manifest_sha256"]),
                             ("environments/dmcontrol/uv.lock", old["binding"]["lock_sha256"]),
                             *[(f"configs/dmcontrol/algs/{name}.json", value)
                               for name, value in old["binding"]["config_sha256"].items()]]:
        content = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
    junit = Path(old["junit_path"])
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old["junit_sha256"]
    assert base.parity_cases(junit) == old["case_names"]
    copied = campaign / "reused-sampled-gpu-parity.xml"
    with copied.open("xb") as stream:
        stream.write(junit.read_bytes())
    rows = [read(CPU_ROOT / f"cpu-{count}.json") for count in (1, 2)]
    assert all(row["passed"] and row["runtime"] == old["runtime"] for row in rows)
    assert rows[0]["binding"] == rows[1]["binding"]
    assert base.select_plan(rows)["cpus"] == 1
    mean = read(SOURCE_ROOT / "gates/0/mean-gpu-parity.json")
    assert mean["passed"] and mean["binding"] == old["binding"]
    assert mean["tests"] == MEAN_TESTS and mean["case_count"] == 2 and mean["skipped"] == 0
    mean_junit = SOURCE_ROOT / "gates/0/mean-gpu-parity.xml"
    assert hashlib.sha256(mean_junit.read_bytes()).hexdigest() == mean["junit_sha256"]
    cases = list(ET.parse(mean_junit).getroot().iter("testcase"))
    assert {case.attrib["name"] for case in cases} == {node.split("::")[1] for node in MEAN_TESTS}
    assert not any(case.find(tag) is not None for case in cases for tag in ("skipped", "error", "failure"))
    with (campaign / "reused-mean-gpu-parity.xml").open("xb") as stream:
        stream.write(mean_junit.read_bytes())
    base.write_new(campaign / "reused-mean-gpu-parity.json", mean)
    evidence = {"kind": "reused_unchanged_mean_learner",
                "original_commit": SOURCE_COMMIT, "original_receipt": str(SOURCE_ROOT / "gpu-parity.json"),
                "protected_source_paths": protected,
                "mean_gpu_parity_receipt": str(campaign / "reused-mean-gpu-parity.json"),
                "new_validation": "Fresh full-size Humanoid Stand startup, timing, checkpoint continuation and critic geometry; W&B disabled."}
    base.write_new(campaign / "gpu-parity.json", {
        **old, "binding": base.binding(base.MANIFEST), "junit_path": str(copied),
        "original_binding": old["binding"], "source_evidence": evidence})
    space = base.checkpoint_space(campaign, 40 * max(row["checkpoint_bytes"] for row in rows))
    base.write_new(campaign / "preflight.json", {
        "binding": base.binding(base.MANIFEST), "source_evidence": evidence, "space": space,
        "reference_end_to_end_seconds": REFERENCE_SECONDS,
        "cpu_comparison": [{"cpus": row["physical_cpus"],
                            "seconds_per_decision": row["measured"]["seconds_per_decision"],
                            "average_cpu_cores_used": row["measured"]["average_cpu_cores_used"]} for row in rows]})
    print(json.dumps({"prepared": True, "space": space}), flush=True)


def start():
    verify_recipe()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_PRIOR_CAMPAIGN"])
    assert os.environ["WANDB_MODE"] == "disabled" and os.environ["SLURM_JOB_ID"]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == 1
    preflight = read(campaign / "preflight.json")
    assert preflight["binding"] == base.binding(base.MANIFEST)
    arm = read(campaign / "submission.json")["runs"][0]
    assert str(arm["job"]) == os.environ["SLURM_JOB_ID"]
    gate = campaign / "gates/0"
    gate.mkdir(parents=True, exist_ok=False)
    report_path = gate / "cpu-1.json"
    cores = base.physical_core_cpus(os.sched_getaffinity(0))
    with (gate / "startup-check.log").open("x") as log:
        subprocess.run([sys.executable, str(ROOT / "slurm/posttanh_prior_fixes_oscar.py"), "worker", "--index", "0",
                        "--worker-cpus", str(cores[0]), "--output", str(report_path)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(report_path)
    assert report["passed"] and report["binding"] == preflight["binding"]
    assert report["graph_native_graph_continuation"] and report["gpu_replay"]
    parity = base.load_global_parity(campaign, report["runtime"])
    measured = report["measured"]["seconds_per_decision"]
    hours = math.ceil(max(measured * 1000000, REFERENCE_SECONDS) * 1.15 / 3600 + .5)
    assert hours <= arm["allocated_hours"], (hours, arm["allocated_hours"])
    plan = {"index": 0, "cpus": 1, "seconds_per_decision": measured,
            "wall_hours": hours, "checkpoint_bytes": report["checkpoint_bytes"]}
    receipt = {"passed": True, "binding": report["binding"], "runtime": report["runtime"],
               "plan": plan, "space": preflight["space"], "cases": [str(report_path)],
               "graph_native_graph_continuation": True,
               "source_evidence": parity["source_evidence"],
               "mean_gpu_parity_receipt": str(campaign / "reused-mean-gpu-parity.json"),
               "scope": "Fresh Humanoid Stand mean-execution startup, checkpoint continuation, full-size startup, critic geometry and 128 timed graph decisions; W&B disabled."}
    base.write_new(gate / "receipt.json", receipt)
    print("STARTUP_CHECK_PASSED", json.dumps(plan), flush=True)
    subprocess.run([os.environ["AMBI_WORKSPACE_PYTHON"],
                    "slurm/humanoid_stand_workspace.py", os.environ["AMBI_STAND_WORKSPACE_STATE"]],
                   cwd=ROOT, check=True)
    os.environ.update(WANDB_MODE="online", WANDB_RUN_ID=arm["id"])
    base.train(SimpleNamespace(index=0, gate=gate / "receipt.json", output=campaign / "production/0"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "start"))
    args = parser.parse_args()
    {"prepare": prepare, "start": start}[args.mode]()
