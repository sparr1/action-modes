"""Launch seeds 56/57 with the validated fc7c433a training implementation."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE = "5f6089fe70441b968ed814d1b1e73ddbd3f08a71"
MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_seeds56_57.json"
ORIGINAL = ROOT / "configs/dmcontrol/experiments/ambi_inner_entropy.json"


def validate_recipe(index):
    from slurm.ambi_transfer_training import recipe

    _, original_manifest, original = recipe(4, ORIGINAL)
    name, manifest, run = recipe(index, MANIFEST)
    assert run["seed"] == 56 + index
    comparable = deepcopy(run)
    comparable["seed"] = original["seed"]
    comparable["name"] = original["name"]
    for key in ("wandb_run_name", "wandb_tags", "wandb_group"):
        comparable["alg_params"][key] = original["alg_params"][key]
    assert comparable == original, "Replication changed more than seed/publication labels"
    for key in ("env_params", "trials", "logs", "save_trials", "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert manifest[key] == original_manifest[key], key
    assert f"seed{run['seed']}" in run["alg_params"]["wandb_run_name"]
    assert f"seed{run['seed']}" in run["alg_params"]["wandb_tags"]
    assert "seed55" not in run["alg_params"]["wandb_tags"]
    return name, manifest, run


def launch(index):
    import torch
    from slurm.ambi_transfer_training import binding, write_new
    from slurm.inner_entropy_oscar import (
        accelerated_runtime, checkpoint_space, parity_cases, validate_global_parity,
    )

    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("WANDB_MODE") == "online"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert commit == os.environ["AMBI_EXPECTED_COMMIT"]
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip()
    # Reuse GPU validation only while the complete training/runtime code is identical.
    subprocess.run(["git", "diff", "--exit-code", REFERENCE, "--", "main.py", "RL", "utils", "domains", "environments",
                    "slurm/inner_entropy_oscar.py", "slurm/ambi_transfer_training.py"], cwd=ROOT, check=True)
    name, _, run = validate_recipe(index)
    original_campaign = Path(os.environ["AMBI_REFERENCE_CAMPAIGN"])
    receipt_path = original_campaign / "gates/4/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    original_binding = binding(ORIGINAL)
    original_binding["commit"] = REFERENCE
    assert receipt["passed"] and receipt["binding"] == original_binding
    assert receipt["graph_native_graph_continuation"]
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    current = accelerated_runtime()
    assert current == receipt["runtime"], (current, receipt["runtime"])
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == receipt["plan"]["cpus"]
    parity = json.loads((original_campaign / "gpu-parity.json").read_text())
    validate_global_parity(parity, original_binding, current)
    junit = Path(parity["junit_path"])
    assert junit.is_relative_to(original_campaign)
    assert hashlib.sha256(junit.read_bytes()).hexdigest() == parity["junit_sha256"]
    assert parity_cases(junit) == parity["case_names"]
    campaign = Path(os.environ["AMBI_ENTROPY_CAMPAIGN"])
    state = json.loads((campaign / "submission.json").read_text())
    spec = state["runs"][index]
    assert spec["seed"] == run["seed"] and spec["id"] == os.environ["WANDB_RUN_ID"]
    assert str(spec["job"]) == os.environ["SLURM_JOB_ID"]
    output = campaign / "production" / f"seed{run['seed']}"
    output.mkdir(parents=True, exist_ok=False)
    space = checkpoint_space(output, 2 * 40 * receipt["plan"]["checkpoint_bytes"])
    write_new(output / "launch.json", {
        "commit": commit, "reference_commit": REFERENCE, "seed": run["seed"],
        "binding": binding(MANIFEST), "runtime": current, "space": space,
        "reference_gate": str(receipt_path), "original_plan": receipt["plan"],
        "job_id": os.environ["SLURM_JOB_ID"], "wandb_run_id": spec["id"],
        "training_code_identical_to_reference": True,
    })
    sys.argv = ["main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
                "--log-dir", str(output), "--alg-index", str(index), "--trial-index", "0", "--num-runs", "1"]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=(0, 1))
    launch(parser.parse_args().index)
