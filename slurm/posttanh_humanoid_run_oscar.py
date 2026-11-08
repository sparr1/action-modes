"""Run the original post-tanh outer-adapt learner on Humanoid Run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["AMBI_ENTROPY_VARIANT"] = "humanoid_run"
from slurm import inner_entropy_oscar as base

SOURCE_COMMIT = "5f6089fe70441b968ed814d1b1e73ddbd3f08a71"
SOURCE_ROOT = Path("/oscar/scratch/rgao48/ambi/inner-entropy-cudagraph-20261102")


def read(path):
    return json.loads(Path(path).read_text())


def verify_recipe():
    folder = ROOT / "configs/dmcontrol"
    source = read(folder / "algs/ambi_inner_entropy_posttanh_outer_adapt.json")
    target = read(folder / "algs/ambi_posttanh_outer_adapt_humanoid_run.json")
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items()
                                if not k.startswith("wandb")}
    assert source == target
    original = read(folder / "experiments/ambi_inner_entropy.json")
    original["env_params"]["task"] = "humanoid-run"
    manifest = read(base.MANIFEST)
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key], key
    assert manifest["configs"] == ["ambi_posttanh_outer_adapt_humanoid_run"]


def prepare():
    verify_recipe()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_ENTROPY_CAMPAIGN"])
    protected = ["RL", "domains", "utils", "environments", "main.py",
                 base.GPU_PARITY_TEST.split("::")[0]]
    subprocess.run(["git", "diff", "--exit-code", SOURCE_COMMIT, "HEAD", "--", *protected],
                   cwd=ROOT, check=True)
    old = read(SOURCE_ROOT / "gpu-parity.json")
    assert old["binding"]["commit"] == SOURCE_COMMIT
    base.validate_global_parity(old, old["binding"], old["runtime"])
    for relative, expected in [(old["binding"]["manifest"], old["binding"]["manifest_sha256"]),
                               ("environments/dmcontrol/uv.lock", old["binding"]["lock_sha256"]),
                               *[(f"configs/dmcontrol/algs/{name}.json", digest)
                                 for name, digest in old["binding"]["config_sha256"].items()]]:
        content = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    junit = Path(old["junit_path"])
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old["junit_sha256"]
    assert base.parity_cases(junit) == old["case_names"]
    copied = campaign / "reused-gpu-parity.xml"
    with copied.open("xb") as stream:
        stream.write(junit.read_bytes())
    evidence = {"kind": "reused_unchanged_learner", "original_commit": SOURCE_COMMIT,
                "original_receipt": str(SOURCE_ROOT / "gpu-parity.json"),
                "protected_source_paths": protected,
                "new_validation": "Fresh full-size Humanoid Run startup and timing before training."}
    base.write_new(campaign / "gpu-parity.json", {
        **old, "binding": base.binding(base.MANIFEST), "junit_path": str(copied),
        "original_binding": old["binding"], "source_evidence": evidence})
    rows = [read(SOURCE_ROOT / "gates/4" / f"cpu-{n}.json") for n in (1, 2)]
    assert all(row["passed"] and row["runtime"] == old["runtime"]
               and row["binding"] == old["binding"] for row in rows)
    plan = base.select_plan(rows)
    assert plan["cpus"] == 1
    space = base.checkpoint_space(campaign, 40 * plan["checkpoint_bytes"])
    base.write_new(campaign / "preflight.json", {
        "binding": base.binding(base.MANIFEST), "source_evidence": evidence,
        "source_plan": plan, "space": space,
        "cpu_comparison": [{"cpus": r["physical_cpus"],
                            "seconds_per_decision": r["measured"]["seconds_per_decision"],
                            "average_cpu_cores_used": r["measured"]["average_cpu_cores_used"]}
                           for r in rows]})
    print(json.dumps({"prepared": True, "source_plan": plan, "space": space}), flush=True)


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
    cpus = base.physical_core_cpus(os.sched_getaffinity(0))
    report_path = gate / "cpu-1.json"
    command = [sys.executable, str(ROOT / "slurm/inner_entropy_oscar.py"),
               "worker", "--index", "0", "--worker-cpus", str(cpus[0]), "--output", str(report_path)]
    with (gate / "startup-check.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(report_path)
    assert report["passed"] and report["binding"] == preflight["binding"]
    assert report["graph_native_graph_continuation"] and report["gpu_replay"]
    parity = base.load_global_parity(campaign, report["runtime"])
    plan = base.select_plan([report])
    plan["wall_hours"] = math.ceil(plan["seconds_per_decision"] * 1000000 * 1.15 / 3600 + .5)
    assert plan["wall_hours"] <= arm["allocated_hours"], (plan, arm["allocated_hours"])
    space = base.checkpoint_space(gate, 40 * report["checkpoint_bytes"])
    receipt = {"passed": True, "binding": report["binding"], "runtime": report["runtime"],
               "plan": plan, "space": space, "cases": [str(report_path)],
               "graph_native_graph_continuation": True,
               "global_gpu_parity_receipt": str(campaign / "gpu-parity.json"),
               "source_evidence": parity["source_evidence"], "cpu_sizing_evidence": preflight,
               "scope": "Fresh full-size Humanoid Run training, 128 timed graph decisions, checkpoint roundtrip, critic geometry, graph/native/graph continuation; W&B disabled."}
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
