"""Validate and launch a mean-execution counterpart of Humanoid Run."""
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
os.environ["AMBI_ENTROPY_VARIANT"] = "humanoid_run_mean"
from slurm import inner_entropy_oscar as base

SOURCE_COMMIT = "6dd7d3244b060891a7edc77143e3c3ea5c2f4c7e"
SOURCE_ROOT = Path("/oscar/scratch/rgao48/ambi/posttanh-humanoid-run-20261108")
CPU_ROOT = Path("/oscar/scratch/rgao48/ambi/inner-entropy-cudagraph-20261102/gates/4")
REFERENCE_SECONDS = 178307.050734
MEAN_TESTS = [
    "tests/test_humanoid_run_mean.py::test_cuda_mean_capture_preserves_inner_state",
    "tests/test_humanoid_run_mean.py::test_cuda_mean_graph_native_graph_and_checkpoint_continuation",
]


def read(path):
    return json.loads(Path(path).read_text())


def verify_recipe():
    folder = ROOT / "configs/dmcontrol"
    source = read(folder / "algs/ambi_posttanh_outer_adapt_humanoid_run.json")
    target = read(folder / "algs/ambi_posttanh_outer_adapt_humanoid_run_mean.json")
    assert source["alg_params"]["inner_execution_action"] == "policy_sample"
    source["alg_params"]["inner_execution_action"] = "mean"
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items()
                                if not k.startswith("wandb")}
    assert source == target
    original = read(folder / "experiments/ambi_posttanh_outer_adapt_humanoid_run.json")
    manifest = read(base.MANIFEST)
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key], key
    assert manifest["configs"] == ["ambi_posttanh_outer_adapt_humanoid_run_mean"]


def prepare():
    verify_recipe()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_ENTROPY_CAMPAIGN"])
    # The only learner change is the already-tested mean-execution guard from
    # 450aa82. Preserve every equation and RNG implementation in this lineage.
    graph_path = "RL/tdmpc2_core/whole_inner_graph.py"
    original = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{graph_path}"], cwd=ROOT, text=True)
    expected = original.replace('        "inner_execution_action": "policy_sample",\n', '', 1)
    anchor = '    if engine.device.type != "cuda":'
    guard = ('    if getattr(engine.cfg, "inner_execution_action", "policy_sample") not in {"policy_sample", "mean"}:\n'
             '        raise ValueError("Training CUDA graph requires sampled or mean policy execution.")\n')
    expected = expected.replace(anchor, guard + anchor, 1)
    assert expected == (ROOT / graph_path).read_text()
    test_nodes = getattr(base, "GPU_PARITY_TESTS", [getattr(base, "GPU_PARITY_TEST", "")])
    protected = ["RL", "domains", "utils", "environments", "main.py",
                 *sorted({node.split("::")[0] for node in test_nodes})]
    subprocess.run(["git", "diff", "--exit-code", SOURCE_COMMIT, "HEAD", "--", *protected,
                    ":!" + graph_path, ":!RL/tdmpc2_core/README.md"], cwd=ROOT, check=True)
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
    evidence = {"kind": "sampled_parity_reuse_with_additional_mean_gpu_gate",
                "original_commit": SOURCE_COMMIT, "original_receipt": str(SOURCE_ROOT / "gpu-parity.json"),
                "protected_source_paths": protected, "mean_guard_reference": "450aa82ed65eeb887c8e8072a0bf9402a2ae8ef0",
                "new_required_gpu_tests": MEAN_TESTS,
                "new_validation": "Fresh mean-action CUDA parity/continuation tests and production-size startup before training; W&B disabled."}
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


def mean_gpu_gate(gate):
    junit = gate / "mean-gpu-parity.xml"
    program = ("import sys,torch,pytest; torch.set_num_threads(1); "
               "torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('high'); "
               "raise SystemExit(pytest.main(sys.argv[1:]))")
    with (gate / "mean-gpu-parity.log").open("x") as log:
        subprocess.run([sys.executable, "-c", program, *MEAN_TESTS, "-q", "--junitxml=" + str(junit)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    assert len(cases) == 2
    assert {case.attrib["name"] for case in cases} == {node.split("::")[1] for node in MEAN_TESTS}
    assert not any(case.find(tag) is not None for case in cases for tag in ("skipped", "error", "failure"))
    base.write_new(gate / "mean-gpu-parity.json", {
        "passed": True, "binding": base.binding(base.MANIFEST), "tests": MEAN_TESTS,
        "case_count": 2, "skipped": 0, "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
        "job_id": os.environ["SLURM_JOB_ID"], "wandb": False})


def start():
    verify_recipe()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_ENTROPY_CAMPAIGN"])
    assert os.environ["WANDB_MODE"] == "disabled" and os.environ["SLURM_JOB_ID"]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == 1
    preflight = read(campaign / "preflight.json")
    assert preflight["binding"] == base.binding(base.MANIFEST)
    arm = read(campaign / "submission.json")["runs"][0]
    assert str(arm["job"]) == os.environ["SLURM_JOB_ID"]
    gate = campaign / "gates/0"
    gate.mkdir(parents=True, exist_ok=False)
    mean_gpu_gate(gate)
    report_path = gate / "cpu-1.json"
    cores = base.physical_core_cpus(os.sched_getaffinity(0))
    with (gate / "startup-check.log").open("x") as log:
        subprocess.run([sys.executable, str(ROOT / "slurm/inner_entropy_oscar.py"), "worker", "--index", "0",
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
               "mean_gpu_parity_receipt": str(gate / "mean-gpu-parity.json"),
               "scope": "Fresh mean-execution Humanoid Run GPU parity, checkpoint continuation, full-size startup, critic geometry and 128 timed graph decisions; W&B disabled."}
    base.write_new(gate / "receipt.json", receipt)
    print("STARTUP_CHECK_PASSED", json.dumps(plan), flush=True)
    subprocess.run(["flock", str(campaign / "workspace.lock"), os.environ["AMBI_WORKSPACE_PYTHON"],
                    "slurm/inner_entropy_workspace.py", str(campaign / "submission.json")], cwd=ROOT, check=True)
    os.environ.update(WANDB_MODE="online", WANDB_RUN_ID=arm["id"])
    base.train(SimpleNamespace(index=0, gate=gate / "receipt.json", output=campaign / "production/0"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "start"))
    args = parser.parse_args()
    {"prepare": prepare, "start": start}[args.mode]()
