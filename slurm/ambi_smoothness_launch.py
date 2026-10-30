"""Bind the Hydra smoothness gate and exact-resume training to one clean recipe.

This launcher never submits or requeues jobs. Exit 75 is a verified clean
handoff requiring an explicit subsequent AMBI_RESUME_MODE=required submission.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import netrc
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MASTER = Path("configs/dmcontrol/experiments/ambi_critic_smoothness_training.json")
ALGS = Path("configs/dmcontrol/algs")
GIB = 1024 ** 3
HANDOFF = 75


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value, *, new=False):
    path = Path(path)
    if new and path.exists():
        raise ValueError("Launch receipt already exists; refuse duplicate invocation")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def recipe_binding(root, index, commit):
    if type(index) is not int or index not in range(4):
        raise ValueError("Cell index must be in 0..3")
    master = read(root / MASTER)
    if len(master["configs"]) != 4 or master.get("trials") != 1:
        raise ValueError("Expected the four-cell, one-trial smoothness campaign")
    name = master["configs"][index]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Unsafe configuration name")
    config = ALGS / (name + ".json")
    individual = MASTER.parent / (name + ".json")
    single = read(root / individual)
    if single.get("configs") != [name] or single.get("trials") != 1:
        raise ValueError("Exact resume requires the matching individual manifest")
    for key in ("env_params", "overrides_alg", "checkpoint_every", "save_strat", "save_trials", "logs", "log_type", "log_info"):
        if single.get(key) != master.get(key):
            raise ValueError(f"Master and individual manifest disagree on {key}")
    run = read(root / config)
    if run.get("total_steps") != 1_000_000 or run.get("seed") != 55:
        raise ValueError("Expected the authorized 1M-decision seed55 recipe")
    binding = {"commit": commit, "manifest": str(MASTER), "manifest_sha256": digest(root / MASTER),
               "config": str(config), "config_sha256": digest(root / config), "alg_index": index,
               "lock_sha256": digest(root / "environments/dmcontrol/uv.lock")}
    return name, individual, binding


def validate_gate(gate, binding, runtime, cpus):
    if gate.get("passed") is not True or not gate.get("cases"):
        raise ValueError("A passed matching GPU gate is required")
    selected = []
    for case in gate["cases"]:
        if (case.get("passed") is not True or case.get("binding") != binding
                or case.get("runtime") != runtime or case.get("compile_fallback") is not False
                or case.get("pave_sample_count") != 768 or case.get("replay_storage_device") != "cpu"):
            raise ValueError("GPU gate binding, runtime or production-shape checks differ")
        seconds = case.get("measured", {}).get("seconds_per_decision")
        if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("GPU gate lacks valid measured decision timing")
        if case.get("physical_cores") == cpus:
            selected.append(case)
    if len(selected) != 1:
        raise ValueError("Production CPU count must match exactly one measured gate case")
    return selected[0]


def allocation_seconds(environment):
    # Query Slurm's actual reservation, rather than trusting a requested value.
    row = subprocess.check_output(["scontrol", "show", "job", environment["SLURM_JOB_ID"], "-o"], text=True)
    match = re.search(r"(?:^|\s)TimeLimit=(\S+)", row)
    if match is None:
        raise ValueError("Cannot read Slurm TimeLimit")
    value = match.group(1)
    days, _, clock = value.rpartition("-")
    fields = clock.split(":")
    if len(fields) != 3 or not all(part.isdigit() for part in fields) or (days and not days.isdigit()):
        raise ValueError("Slurm TimeLimit must be finite D-HH:MM:SS or HH:MM:SS")
    hours, minutes, seconds = map(int, fields)
    return int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds


def command_for(mode, python, index, individual, campaign, artifacts, gate_dir, drain_seconds=None, resume_mode="new"):
    if mode == "gate":
        return [python, "slurm/benchmark_ambi_smoothness.py", "--manifest", str(MASTER),
                "--alg-dir", str(ALGS), "--alg-index", str(index), "--output", str(gate_dir),
                "--cores", "1,2", "--warmup", "12", "--decisions", "30"]
    if mode != "train" or resume_mode not in ("new", "required") or not drain_seconds or drain_seconds <= 0:
        raise ValueError("Invalid training mode or drain deadline")
    return [python, "main.py", "--run", str(individual), "--alg-dir", str(ALGS),
            "--log-dir", str(artifacts / f"cell-{index}"), "--num-runs", "1",
            "--lineage-dir", str(campaign / f"cell-{index}" / "lineage"),
            "--resume-model-snapshot-root", str(artifacts / f"cell-{index}" / "snapshot-bank"),
            "--resume-mode", resume_mode, "--resume-wandb-mode", "online",
            "--resume-checkpoint-minutes", "15", "--drain-after-seconds", str(drain_seconds)]


def existing_credentials():
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    try:
        credentials = netrc.netrc().authenticators(host)
        return bool(credentials and credentials[2])
    except (OSError, netrc.NetrcParseError):
        return False


def outside(path, root):
    return path != root and root not in path.parents


def verify_exit(status, lineage, job_id, previous_segments):
    from utils.oscar_resume_launcher import verify_done, verify_handoff
    if status not in (0, HANDOFF):
        return {"state": "failed", "exit_code": status}
    segments = {path.name for path in (lineage / "segments").iterdir() if path.is_dir()}
    created = segments - previous_segments
    if len(created) != 1:
        raise ValueError("Training must create exactly one new exact-resume segment")
    segment = next(iter(created))
    if status == 0:
        generation = verify_done(lineage_dir=lineage)
        if read(lineage / "DONE")["segment_id"] != segment:
            raise ValueError("DONE belongs to an older training segment")
        return {"state": "complete", "exit_code": 0, "generation": generation}
    generation = verify_handoff(lineage_dir=lineage, slurm_job_id=job_id, segment_id=segment)
    return {"state": "clean_handoff", "exit_code": HANDOFF, "generation": generation,
            "next_action": "Explicitly submit AMBI_RESUME_MODE=required; no automatic requeue"}


def main():
    started = time.monotonic()
    environment = os.environ
    for key in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "AMBI_PROJECT_DIR", "AMBI_PYTHON",
                "AMBI_EXPECTED_COMMIT", "AMBI_CAMPAIGN_ROOT", "AMBI_ARTIFACT_ROOT", "AMBI_MODE", "AMBI_CELL_INDEX"):
        if not environment.get(key):
            raise ValueError(f"Required environment setting missing: {key}")
    if environment.get("SLURM_RESTART_COUNT", "0") != "0":
        raise ValueError("Automatic requeue is disabled")
    root = Path(environment["AMBI_PROJECT_DIR"]).resolve(strict=True)
    if root != ROOT.resolve() or Path.cwd().resolve() != root:
        raise ValueError("Run from the selected source checkout")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (not re.fullmatch(r"[0-9a-f]{40}", environment["AMBI_EXPECTED_COMMIT"])
            or commit != environment["AMBI_EXPECTED_COMMIT"]
            or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True).strip()):
        raise ValueError("Source must be clean at the exact expected commit")
    python = environment["AMBI_PYTHON"]
    if not Path(python).is_absolute() or Path(python).resolve() != Path(sys.executable).resolve():
        raise ValueError("Use the selected interpreter for launcher and training")
    mode, index = environment["AMBI_MODE"], int(environment["AMBI_CELL_INDEX"])
    if mode not in ("gate", "train"):
        raise ValueError("AMBI_MODE must be gate or train")
    name, individual, binding = recipe_binding(root, index, commit)
    campaign, artifacts = (Path(environment[key]) for key in ("AMBI_CAMPAIGN_ROOT", "AMBI_ARTIFACT_ROOT"))
    if not campaign.is_absolute() or not artifacts.is_absolute():
        raise ValueError("Campaign and artifact roots must be absolute")
    campaign, artifacts = campaign.resolve(), artifacts.resolve()
    if (not outside(campaign, root) or not outside(artifacts, root)
            or Path.home().resolve() not in campaign.parents or Path("/ltmp/rgao48") not in artifacts.parents
            or not outside(campaign, artifacts) or not outside(artifacts, campaign)):
        raise ValueError("Use disjoint external home and /ltmp/rgao48 campaign roots")
    campaign.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    free = {"home_bytes": shutil.disk_usage(campaign).free, "artifact_bytes": shutil.disk_usage(artifacts).free}
    if free["home_bytes"] < 12 * GIB or free["artifact_bytes"] < 120 * GIB:
        raise ValueError("Campaign requires at least 12 GiB free home and 120 GiB free artifact space")
    host = socket.gethostname().split(".")[0].lower()
    if host != "gpu2501":
        raise ValueError("Node-local artifact storage requires gpu2501")
    cpus = int(environment["SLURM_CPUS_PER_TASK"])
    if cpus not in (1, 2) or (mode == "gate" and cpus != 2):
        raise ValueError("Gate requires two CPUs; training requires a measured one/two-CPU allocation")
    from slurm.ambi_transfer_training import runtime
    current_runtime = runtime()
    if "L40S" not in current_runtime["gpu"].upper():
        raise ValueError("NVIDIA L40S GPU required")
    if not existing_credentials():
        raise ValueError("No usable existing W&B netrc credential; credential material is not printed")
    cell = campaign / f"cell-{index}"
    cell.mkdir(exist_ok=True)
    gate_dir = Path(environment.get("AMBI_GATE_DIR", str(cell / "gate"))).resolve()
    if not outside(gate_dir, root) or campaign not in gate_dir.parents:
        raise ValueError("Gate output must be below the external home campaign root")
    lineage = cell / "lineage"
    previous_segments = ({p.name for p in (lineage / "segments").iterdir() if p.is_dir()}
                         if (lineage / "segments").is_dir() else set())
    measured = None
    requested_seconds = allocation_seconds(environment)
    drain = None
    if mode == "train":
        requested = int(environment.get("AMBI_TRAIN_WALLTIME_SECONDS", "0"))
        if requested <= 1800 or requested != requested_seconds:
            raise ValueError("AMBI_TRAIN_WALLTIME_SECONDS must match Slurm and leave a 30-minute drain margin")
        measured = validate_gate(read(gate_dir / "results.json"), binding, current_runtime, cpus)
        drain = requested_seconds - 1800 - math.ceil(time.monotonic() - started)
        if environment.get("SLURM_JOB_END_TIME", "").isdigit():
            drain = min(drain, int(environment["SLURM_JOB_END_TIME"]) - math.ceil(time.time()) - 1800)
    command = command_for(mode, python, index, individual, campaign, artifacts, gate_dir,
                          drain_seconds=drain, resume_mode=environment.get("AMBI_RESUME_MODE", "new"))
    receipt = {"schema_version": 1, "mode": mode, "cell_index": index, "config": name,
               "binding": binding, "individual_manifest": str(individual),
               "individual_manifest_sha256": digest(root / individual), "runtime": current_runtime,
               "host": host, "slurm_job_id": environment["SLURM_JOB_ID"], "cpus": cpus,
               "allocation_seconds": requested_seconds, "drain_after_seconds": drain,
               "free_space": free, "netrc_credential_available": True,
               "campaign_root": str(campaign), "artifact_root": str(artifacts),
               "lineage": str(lineage), "gate_dir": str(gate_dir), "command": command,
               "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running"}
    if measured is not None:
        receipt.update(gate_sha256=digest(gate_dir / "results.json"), measured=measured["measured"])
    receipt_path = cell / f"launch-{mode}-{environment['SLURM_JOB_ID']}.json"
    atomic_json(receipt_path, receipt, new=True)
    print("SMOOTHNESS_LAUNCH " + json.dumps(receipt, allow_nan=False), flush=True)
    status = subprocess.run(command, cwd=root, check=False).returncode
    if mode == "gate" and status == 0:
        gate = read(gate_dir / "results.json")
        validate_gate(gate, binding, current_runtime, cpus)
        receipt.update(state="gate_passed", exit_code=0, gate_sha256=digest(gate_dir / "results.json"))
    elif mode == "train":
        receipt.update(verify_exit(status, lineage, environment["SLURM_JOB_ID"], previous_segments))
    else:
        receipt.update(state="failed", exit_code=status)
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(receipt_path, receipt)
    print("SMOOTHNESS_RESULT " + json.dumps({key: receipt[key] for key in ("state", "exit_code")}), flush=True)
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Smoothness launcher failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
