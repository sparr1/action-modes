"""Submit one measured K3 training job after the pinned CUDA and CPU gates pass.

Run in a short CPU allocation with an afterok dependency on both validation jobs.
The training checkout remains pinned even when this submitter uses newer code.
An interrupted/unknown submission must be reconciled manually before any retry.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, *, new=False):
    temporary = path if new else path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not new:
        temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resources(gate, benchmark):
    cases = benchmark["cases"]
    assert len(cases) == 2 and {row["physical_cores"] for row in cases} == {1, 2}
    for row in cases:
        assert row["passed"] and row["binding"] == gate["binding"] and row["runtime"] == gate["runtime"]
        seconds = row["measured"]["seconds_per_decision"]
        assert math.isfinite(seconds) and seconds > 0
    fastest = min(row["measured"]["seconds_per_decision"] for row in cases)
    cpus = min(row["physical_cores"] for row in cases if row["measured"]["seconds_per_decision"] <= fastest * 1.05)
    assert cpus == benchmark["recommended_cpus_within_5_percent_of_fastest"]
    seconds = next(row["measured"]["seconds_per_decision"] for row in cases if row["physical_cores"] == cpus)
    assert math.isfinite(gate["seconds"]) and gate["seconds"] > 0
    wall = math.ceil((seconds * 1_000_000 * 1.15 + 2 * gate["seconds"] + 600) / 3600) * 3600
    return {"cpus": cpus, "memory_gib": 16, "gpus": 1, "seconds_per_decision": seconds,
            "margin_fraction": .15, "startup_seconds": 2 * gate["seconds"], "fixed_margin_seconds": 600,
            "walltime_seconds": wall}


def submit(source_dir, campaign_dir, python, expected_commit):
    source_dir, campaign_dir = Path(source_dir).resolve(), Path(campaign_dir).resolve()
    assert re.fullmatch(r"[0-9a-f]{40}", expected_commit), "Require the full pinned source commit"
    assert source_dir != campaign_dir and source_dir not in campaign_dir.parents
    assert Path(python).is_absolute() and Path(python).is_file()
    identity = {"source_dir": str(source_dir), "campaign_dir": str(campaign_dir),
                "python": str(python), "source_commit": expected_commit}
    receipt_path = campaign_dir / "submission.json"
    if receipt_path.exists():
        old = read(receipt_path)
        assert old["identity"] == identity, "Existing submission targets another identity"
        assert old["status"] == "submitted", "Unknown submission outcome: reconcile the saved intent before retrying"
        return old
    assert not (campaign_dir / "lineage/LINEAGE.json").exists(), "A native lineage already exists"
    assert subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip() == expected_commit
    assert not subprocess.check_output(["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    gate_path, benchmark_path = campaign_dir / "gate/receipt.json", campaign_dir / "cpu-benchmark/results.json"
    # Import the training checkout's own helper in an isolated interpreter. This
    # avoids reusing submitter-checkout modules already present in sys.modules.
    code = ("import json,sys; from slurm.ambi_alpha_carry_k3_training import binding,read,validate_gate; "
            "expected=binding(); validate_gate(read(sys.argv[1]),expected); print(json.dumps(expected))")
    expected = json.loads(subprocess.check_output([str(python), "-c", code, str(gate_path)], cwd=source_dir, text=True))
    gate, benchmark = read(gate_path), read(benchmark_path)
    assert expected["commit"] == expected_commit and gate["binding"] == expected
    assert benchmark["passed"] and benchmark["binding"] == expected
    assert "L40S" in gate["runtime"]["gpu"].upper() and gate["runtime"]["executable"] == str(python)
    sizing = resources(gate, benchmark)
    wall = sizing["walltime_seconds"]
    # A CPU dispatcher must not export its allocation, resource overrides or
    # node-local temporary paths into the independently scheduled GPU job.
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(("SLURM_", "SBATCH_", "SRUN_"))
                 and key not in {"CUDA_VISIBLE_DEVICES", "TMPDIR", "TMP", "TEMP"}}
    environment = {**inherited, "AMBI_PROJECT_DIR": str(source_dir), "AMBI_PYTHON": str(python),
                   "AMBI_EXPECTED_COMMIT": expected_commit, "AMBI_ALPHA_CARRY_K3_OUTPUT": str(campaign_dir),
                   "AMBI_ALPHA_CARRY_K3_MODE": "train", "AMBI_RESUME_MODE": "new",
                   "AMBI_TRAIN_WALLTIME_SECONDS": str(wall), "AMBI_DRAIN_AFTER_SECONDS": str(wall - 1800)}
    log_dir = campaign_dir / "slurm"
    log_dir.mkdir(exist_ok=True)
    command = ["sbatch", "--parsable", "--partition=gpu", "--qos=pri-gpu+", "--gres=gpu:l40s:1",
               f"--cpus-per-task={sizing['cpus']}", "--mem=16G", f"--time={wall // 60}",
               "--export=ALL", "--chdir=" + str(source_dir), "--output=" + str(log_dir / "train-%j.out"),
               "--error=" + str(log_dir / "train-%j.err"), str(source_dir / "slurm/run_ambi_alpha_carry_k3_oscar.sbatch")]
    record = {"identity": identity, "status": "submitting", "binding": expected, "resources": sizing,
              "command": command, "gate_receipt": str(gate_path), "benchmark_receipt": str(benchmark_path)}
    save(receipt_path, record, new=True)  # Exclusive durable intent; never repeat an unknown sbatch.
    acknowledgment = subprocess.check_output(command, env=environment, text=True).strip()
    assert re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", acknowledgment), "Unknown sbatch acknowledgment; reconcile receipt"
    record.update(status="submitted", job_id=acknowledgment.split(";")[0])
    save(receipt_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=os.environ.get("AMBI_PROJECT_DIR"), required=not os.environ.get("AMBI_PROJECT_DIR"))
    parser.add_argument("--campaign-dir", default=os.environ.get("AMBI_ALPHA_CARRY_K3_OUTPUT"), required=not os.environ.get("AMBI_ALPHA_CARRY_K3_OUTPUT"))
    parser.add_argument("--python", default=os.environ.get("AMBI_PYTHON", sys.executable))
    parser.add_argument("--expected-commit", default=os.environ.get("AMBI_EXPECTED_COMMIT"), required=not os.environ.get("AMBI_EXPECTED_COMMIT"))
    args = parser.parse_args()
    print(json.dumps(submit(args.source_dir, args.campaign_dir, args.python, args.expected_commit)), flush=True)


if __name__ == "__main__":
    main()
