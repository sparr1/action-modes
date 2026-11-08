"""Launch the exact sampled-action BOOM x0.01 recipe on Stand or Run."""
import argparse
from copy import deepcopy
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
from slurm import posttanh_prior_fixes_oscar as base

SOURCE_COMMIT = "bcaa1f59b8e513d7095b1276d6cd2f34144ab18a"
SOURCE_ROOT = Path("/oscar/scratch/rgao48/ambi/posttanh-prior-strength-20261105")
CAMPAIGN = Path("/oscar/scratch/rgao48/ambi/boom-humanoid-stand-run-20261108")
# Completed source run, including pretraining, diagnostics and checkpoint I/O.
REFERENCE_SECONDS = 181042.10853


def read(path):
    return json.loads(Path(path).read_text())


def scientific(config):
    config = deepcopy(config)
    config["alg_params"] = {k: v for k, v in config["alg_params"].items()
                            if not k.startswith("wandb")}
    return config


def verify_recipe(task):
    assert task in ("stand", "run")
    assert base.CAMPAIGN_ROOT == CAMPAIGN / task
    manifest = read(base.MANIFEST)
    source = read(ROOT / "configs/dmcontrol/algs/ambi_posttanh_prior_outer_adapt_boom_x0p01.json")
    target = read(ROOT / "configs/dmcontrol/algs" / (manifest["configs"][0] + ".json"))
    assert scientific(source) == scientific(target)
    original = read(ROOT / "configs/dmcontrol/experiments/ambi_posttanh_prior_strength.json")
    for key in ("overrides_alg", "trials", "logs", "save_trials", "checkpoint_every",
                "save_strat", "log_info", "log_type"):
        assert manifest[key] == original[key], key
    assert manifest["env_params"] == {**original["env_params"], "task": "humanoid-" + task}
    assert len(manifest["configs"]) == 1


def prepare(task):
    verify_recipe(task)
    protected = ["RL", "domains", "utils", "environments", "main.py",
                 *sorted({node.split("::")[0] for node in base.GPU_PARITY_TESTS})]
    subprocess.run(["git", "diff", "--exit-code", SOURCE_COMMIT, "HEAD", "--", *protected],
                   cwd=ROOT, check=True)
    old = read(SOURCE_ROOT / "gpu-parity.json")
    assert old["binding"]["commit"] == SOURCE_COMMIT
    base.validate_global_parity(old, old["binding"], old["runtime"])
    junit = Path(old["junit_path"])
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old["junit_sha256"]
    assert base.parity_cases(junit) == old["case_names"]
    for relative, expected in [(old["binding"]["manifest"], old["binding"]["manifest_sha256"]),
                               ("environments/dmcontrol/uv.lock", old["binding"]["lock_sha256"]),
                               *[(f"configs/dmcontrol/algs/{name}.json", digest)
                                 for name, digest in old["binding"]["config_sha256"].items()]]:
        contents = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(contents).hexdigest() == expected
    prior = read(SOURCE_ROOT / "gates/1/receipt.json")
    assert prior["passed"] and prior["binding"] == old["binding"]
    assert prior["runtime"] == old["runtime"]
    sizing = prior["cpu_sizing_evidence"]
    cases = [read(Path(sizing["original_gate"]).parent / f"cpu-{count}.json") for count in (1, 2)]
    assert all(r["passed"] and r["runtime"] == old["runtime"] for r in cases)
    assert base.select_plan(cases)["cpus"] == 1
    evidence = {"kind": "reused_unchanged_learner", "original_commit": SOURCE_COMMIT,
                "original_receipt": str(SOURCE_ROOT / "gpu-parity.json"),
                "protected_source_paths": protected,
                "cpu_sizing_evidence": sizing,
                "new_validation": "Fresh task-specific production-shape smoke and 128 timed decisions before training; W&B disabled."}
    base.CAMPAIGN_ROOT.mkdir(exist_ok=False)
    for subdir in ("gates", "cache", "production"):
        (base.CAMPAIGN_ROOT / subdir).mkdir()
    copied = base.CAMPAIGN_ROOT / "reused-gpu-parity.xml"
    with copied.open("xb") as stream:
        stream.write(junit.read_bytes())
    reused = {**old, "binding": base.binding(base.MANIFEST), "junit_path": str(copied),
              "original_binding": old["binding"], "source_evidence": evidence}
    base.write_new(base.CAMPAIGN_ROOT / "gpu-parity.json", reused)
    space = base.checkpoint_space(base.CAMPAIGN_ROOT, 2 * 40 * prior["plan"]["checkpoint_bytes"])
    base.write_new(base.CAMPAIGN_ROOT / "preflight.json", {
        "binding": base.binding(base.MANIFEST), "source_evidence": evidence,
        "space": space, "reference_end_to_end_seconds": REFERENCE_SECONDS})
    print(json.dumps({"prepared": task, "space": space}), flush=True)


def start(task):
    verify_recipe(task)
    assert os.environ["WANDB_MODE"] == "disabled" and os.environ["SLURM_JOB_ID"]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == 1
    assert Path(os.environ["AMBI_PRIOR_CAMPAIGN"]) == base.CAMPAIGN_ROOT
    preflight = read(base.CAMPAIGN_ROOT / "preflight.json")
    assert preflight["binding"] == base.binding(base.MANIFEST)
    arm = next(r for r in read(CAMPAIGN / "submission.json")["runs"] if r["task"] == task)
    assert str(arm["job"]) == os.environ["SLURM_JOB_ID"]
    gate = base.CAMPAIGN_ROOT / "gates/0"
    gate.mkdir(exist_ok=False)
    report_path = gate / "cpu-1.json"
    cores = base.physical_core_cpus(os.sched_getaffinity(0))
    command = [sys.executable, str(ROOT / "slurm/posttanh_prior_fixes_oscar.py"), "worker",
               "--index", "0", "--worker-cpus", str(cores[0]), "--output", str(report_path)]
    with (gate / "startup-check.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(report_path)
    assert report["passed"] and report["binding"] == preflight["binding"]
    assert report["graph_native_graph_continuation"] and report["gpu_replay"]
    parity = base.load_global_parity(base.CAMPAIGN_ROOT, report["runtime"])
    measured = report["measured"]["seconds_per_decision"]
    # Include the completed run's actual overhead, then allow a 15% timing margin
    # plus half an hour for this job's startup validation.
    hours = math.ceil(max(measured * 1000000, REFERENCE_SECONDS) * 1.15 / 3600 + .5)
    assert hours <= arm["allocated_hours"], (hours, arm["allocated_hours"])
    plan = {"index": 0, "cpus": 1, "seconds_per_decision": measured,
            "wall_hours": hours, "checkpoint_bytes": report["checkpoint_bytes"]}
    receipt = {"passed": True, "binding": report["binding"], "runtime": report["runtime"],
               "plan": plan, "space": preflight["space"], "cases": [str(report_path)],
               "graph_native_graph_continuation": True,
               "source_evidence": parity["source_evidence"],
               "scope": f"Fresh Humanoid {task} smoke, checkpoint roundtrip, critic geometry, graph/native/graph continuation and 128 timed graph decisions; W&B disabled."}
    base.write_new(gate / "receipt.json", receipt)
    print("STARTUP_CHECK_PASSED", task, json.dumps(plan), flush=True)
    subprocess.run(["flock", str(CAMPAIGN / "workspace.lock"), os.environ["AMBI_WORKSPACE_PYTHON"],
                    "slurm/posttanh_prior_fixes_workspace.py", str(CAMPAIGN / "submission.json")],
                   cwd=ROOT, check=True)
    os.environ.update(WANDB_MODE="online", WANDB_RUN_ID=arm["id"])
    base.train(SimpleNamespace(index=0, gate=gate / "receipt.json",
                               output=base.CAMPAIGN_ROOT / "production/0"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "start"))
    parser.add_argument("--task", choices=("stand", "run"), required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else start)(args.task)
