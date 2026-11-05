"""Four coefficient-only AMBI runs, reusing verified unchanged-learner evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["AMBI_PRIOR_VARIANT"] = "strength"
from slurm import posttanh_prior_fixes_oscar as base

ORIGINAL_COMMIT = "342c7f1ec4d55fe18257499431eed335226e335a"
ORIGINAL_ROOT = Path("/oscar/scratch/rgao48/ambi/posttanh-prior-fixes-20261104")
ARMS = (("boom", .1, 2), ("boom", .01, 2),
        ("tdmpc_square", .1, 1), ("tdmpc_square", .01, 1))


def read(path):
    return json.loads(Path(path).read_text())


def verify_variants():
    manifest = read(base.MANIFEST)
    assert len(manifest["configs"]) == 4 and manifest["trials"] == 1
    assert manifest["overrides_alg"]["total_steps"] == 1000000
    assert manifest["checkpoint_every"] == 25000
    for index, (method, multiplier, _) in enumerate(ARMS):
        source = read(ROOT / "configs/dmcontrol/algs" / f"ambi_posttanh_prior_outer_adapt_{method}.json")
        target = read(ROOT / "configs/dmcontrol/algs" / (manifest["configs"][index] + ".json"))
        coefficient = "outer_boom_coef" if method == "boom" else "outer_behavior_policy_kl_coef"
        assert target["alg_params"][coefficient] == source["alg_params"][coefficient] * multiplier
        expected = deepcopy(source)
        expected["alg_params"][coefficient] *= multiplier
        def scientific(value):
            value = deepcopy(value)
            value["alg_params"] = {k: v for k, v in value["alg_params"].items() if not k.startswith("wandb")}
            return value
        assert scientific(target) == scientific(expected), manifest["configs"][index]
    return manifest


def prepare():
    """Read-only Git proof plus new, explicitly attributed validation records."""
    verify_variants()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_PRIOR_CAMPAIGN"])
    protected = ["RL", "domains", "utils", "environments", "main.py",
                 *{node.split("::")[0] for node in base.GPU_PARITY_TESTS}]
    subprocess.run(["git", "diff", "--exit-code", ORIGINAL_COMMIT, "HEAD", "--", *protected],
                   cwd=ROOT, check=True)
    old = read(ORIGINAL_ROOT / "gpu-parity.json")
    assert old["binding"]["commit"] == ORIGINAL_COMMIT
    for relative, expected in [(old["binding"]["manifest"], old["binding"]["manifest_sha256"]),
                               ("environments/dmcontrol/uv.lock", old["binding"]["lock_sha256"]),
                               *[(f"configs/dmcontrol/algs/{name}.json", digest)
                                 for name, digest in old["binding"]["config_sha256"].items()]]:
        content = subprocess.check_output(["git", "show", f"{ORIGINAL_COMMIT}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    base.validate_global_parity(old, old["binding"], old["runtime"])
    junit = Path(old["junit_path"])
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == old["junit_sha256"]
    assert base.parity_cases(junit) == old["case_names"]
    copied = campaign / "reused-gpu-parity.xml"
    with copied.open("xb") as stream:
        stream.write(junit.read_bytes())
    evidence = {"kind": "reused_unchanged_learner", "original_commit": ORIGINAL_COMMIT,
                "original_receipt": str(ORIGINAL_ROOT / "gpu-parity.json"),
                "original_receipt_sha256": hashlib.sha256((ORIGINAL_ROOT / "gpu-parity.json").read_bytes()).hexdigest(),
                "protected_source_paths": protected,
                "new_validation": "Each coefficient receives a fresh production-shape startup check before training."}
    reused = {**old, "binding": base.binding(base.MANIFEST), "junit_path": str(copied),
              "source_evidence": evidence, "original_binding": old["binding"]}
    base.write_new(campaign / "gpu-parity.json", reused)
    plans = []
    for index, (_, _, old_index) in enumerate(ARMS):
        old_gate = read(ORIGINAL_ROOT / "gates" / str(old_index) / "receipt.json")
        assert old_gate["passed"] and old_gate["binding"] == old["binding"]
        assert old_gate["runtime"] == old["runtime"] and old_gate["graph_native_graph_continuation"]
        rows = [read(ORIGINAL_ROOT / "gates" / str(old_index) / f"cpu-{n}.json") for n in (1, 2)]
        assert all(row["passed"] and row["runtime"] == old["runtime"] for row in rows)
        plan = base.select_plan(rows)
        assert plan["cpus"] == 1
        plans.append({**plan, "index": index, "original_index": old_index,
                      "original_gate": str(ORIGINAL_ROOT / "gates" / str(old_index) / "receipt.json"),
                      "cpu_comparison": [{"cpus": r["physical_cpus"],
                                          "seconds_per_decision": r["measured"]["seconds_per_decision"],
                                          "average_cpu_cores_used": r["measured"]["average_cpu_cores_used"]} for r in rows]})
    space = base.checkpoint_space(campaign, 4 * 40 * max(p["checkpoint_bytes"] for p in plans))
    base.write_new(campaign / "preflight.json", {"binding": base.binding(base.MANIFEST),
                   "source_evidence": evidence, "plans": plans, "space": space})
    print(json.dumps({"prepared": True, "plans": plans, "space": space}), flush=True)


def start(index):
    verify_variants()
    campaign = base.CAMPAIGN_ROOT
    assert campaign == Path(os.environ["AMBI_PRIOR_CAMPAIGN"])
    assert os.environ["WANDB_MODE"] == "disabled" and os.environ["SLURM_JOB_ID"]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == 1
    preflight = read(campaign / "preflight.json")
    assert preflight["binding"] == base.binding(base.MANIFEST)
    arm = read(campaign / "submission.json")["runs"][index]
    assert str(arm["job"]) == os.environ["SLURM_JOB_ID"]
    gate = campaign / "gates" / str(index)
    gate.mkdir(exist_ok=False)
    cpus = base.physical_core_cpus(os.sched_getaffinity(0))
    report_path = gate / "cpu-1.json"
    command = [sys.executable, str(ROOT / "slurm/posttanh_prior_fixes_oscar.py"),
               "worker", "--index", str(index), "--worker-cpus", str(cpus[0]), "--output", str(report_path)]
    with (gate / "startup-check.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(report_path)
    assert report["passed"] and report["binding"] == preflight["binding"]
    assert report["graph_native_graph_continuation"] and report["gpu_replay"]
    parity = base.load_global_parity(campaign, report["runtime"])
    plan = base.select_plan([report])
    # Keep the already-budgeted measured reservation; refuse an unexpectedly slow node.
    assert plan["wall_hours"] <= arm["allocated_hours"], (plan, arm["allocated_hours"])
    space = base.checkpoint_space(gate, 4 * 40 * report["checkpoint_bytes"])
    receipt = {"passed": True, "binding": report["binding"], "runtime": report["runtime"],
               "plan": plan, "space": space, "cases": [str(report_path)],
               "graph_native_graph_continuation": True,
               "global_gpu_parity_receipt": str(campaign / "gpu-parity.json"),
               "source_evidence": parity["source_evidence"],
               "cpu_sizing_evidence": preflight["plans"][index],
               "scope": "Fresh coefficient-specific production-shape training, 128 timed graph decisions, checkpoint roundtrip, geometry observation and graph/native/graph continuation; W&B disabled."}
    base.write_new(gate / "receipt.json", receipt)
    print("STARTUP_CHECK_PASSED", index, json.dumps(plan), flush=True)
    subprocess.run(["flock", str(campaign / "workspace.lock"), os.environ["AMBI_WORKSPACE_PYTHON"],
                    "slurm/posttanh_prior_fixes_workspace.py", str(campaign / "submission.json")], cwd=ROOT, check=True)
    os.environ.update(WANDB_MODE="online", WANDB_RUN_ID=arm["id"])
    base.train(SimpleNamespace(index=index, gate=gate / "receipt.json", output=campaign / "production" / str(index)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "start"))
    parser.add_argument("--index", type=int, choices=range(4))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        assert args.index is not None
        start(args.index)
