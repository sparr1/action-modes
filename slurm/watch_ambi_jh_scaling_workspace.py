"""Bind the existing J/H scaling workspace after all three training jobs start.

Run as a small CPU job with ``after:<production IDs>``. The watcher owns no
training jobs: it only reads their immutable lineage IDs and updates the already
created comparison view using the pinned publication environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_jh_scaling_workspace as publisher


ROOT = Path("/oscar/scratch/rgao48/ambi/jh-scaling-training/20261028")
LINEAGES = {variant: variant.replace("_", "-") + "-seed55"
            for variant in publisher.VARIANTS if variant != "baseline"}
FAILED_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                 "BOOT_FAIL", "DEADLINE", "REVOKED", "PREEMPTED"}


def read_jobs(path):
    jobs = json.loads(Path(path).read_text())
    if (not isinstance(jobs, dict) or set(jobs) != set(LINEAGES)
            or any(not re.fullmatch(r"[1-9][0-9]*", str(value)) for value in jobs.values())
            or len(set(map(str, jobs.values()))) != 3):
        raise ValueError("Production receipt must name exactly three distinct Slurm jobs.")
    return {variant: str(job) for variant, job in jobs.items()}


def job_states(jobs):
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", ",".join(jobs.values()),
         "--format=JobIDRaw,State,ExitCode"],
        capture_output=True, text=True, check=True, timeout=20)
    states = {}
    for line in result.stdout.splitlines():
        columns = line.split("|")
        if len(columns) >= 3 and columns[0] in jobs.values():
            states[columns[0]] = {"state": columns[1].split()[0].rstrip("+"),
                                  "exit_code": columns[2]}
    return states


def owned_workspace(path):
    receipt = json.loads(Path(path).read_text())
    if (receipt.get("entity"), receipt.get("project"), receipt.get("group"),
            receipt.get("baseline"), receipt.get("name")) != (
            publisher.ENTITY, publisher.PROJECT, publisher.GROUP,
            publisher.BASELINE, publisher.NAME):
        raise ValueError("Workspace receipt belongs to another comparison.")
    publisher._view_id(receipt.get("url", ""))
    return receipt


def lineage_ids(root, commit):
    identities = {}
    for variant, directory in LINEAGES.items():
        path = Path(root) / "lineages" / directory / "LINEAGE.json"
        if not path.exists():
            continue
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("Lineage identity must not pass through a symlink.")
        lineage = json.loads(path.read_text())
        metadata = lineage.get("metadata", {})
        identity = metadata.get("scientific_identity", {})
        science = identity.get("scientific", {})
        digest = hashlib.sha256(json.dumps(
            science, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (lineage.get("schema_version") != 1 or metadata.get("schema_version") != 1
                or metadata.get("total_steps") != 1000000 or identity.get("schema_version") != 1
                or identity.get("fingerprint") != digest):
            raise ValueError(f"Invalid immutable lineage identity for {variant}.")
        source = science.get("source", {})
        if source.get("commit") != commit or source.get("dirty") is not False:
            raise ValueError(f"Lineage source does not match the clean launch commit for {variant}.")
        run_id = metadata.get("initial_wandb_run_id", "")
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError(f"Invalid genesis W&B ID for {variant}.")
        trial = science.get("trial_run_params", {})
        if trial.get("name") != f"ambi_full_{variant}_reward_only":
            raise ValueError(f"Wrong training cell in lineage for {variant}.")
        # The scientific identity deliberately omits these historical defaults;
        # W&B's resolved configuration (checked separately) includes them.
        params = {"aux_return_mode": "off", "inner_actor_initialization": "prior",
                  "inner_critic_initialization": "prior", **trial.get("alg_params", {})}
        publisher._validate_run(SimpleNamespace(id=run_id, config={
            "run_params": trial, "alg_params": params}), variant)
        identities[variant] = run_id
    if len(set(identities.values())) != len(identities):
        raise ValueError("Multiple lineages claim the same W&B identity.")
    return identities


def watch(root, workspace, jobs_path, commit, *, timeout_seconds=1500, poll_seconds=30,
          api_factory=None, installer=publisher.install, read_states=job_states,
          clock=time.monotonic, sleep=time.sleep):
    root, workspace = Path(root), Path(workspace)
    status_path = root / "publication" / "workspace-watch-status.json"
    status = {"state": "starting", "source_commit": commit, "expected_ids": {},
              "visible_ids": {}, "api_verified": False, "browser_verified": False}

    def save():
        status["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        publisher._write(status_path, status)

    try:
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("A complete launch commit is required.")
        if not 0 < timeout_seconds <= 1500 or not 0 < poll_seconds <= 60:
            raise ValueError("Watcher must remain within its bounded CPU reservation.")
        previous = owned_workspace(workspace)
        status["url"] = previous["url"]
        jobs = read_jobs(jobs_path)
        status["production_jobs"] = jobs
        old_expected = previous.get("expected_ids", {})
        installed = {row["variant"]: row["run_id"] for row in previous.get("rows", [])
                     if row.get("variant") in LINEAGES and row.get("run_id")}
        seen = {}
        deadline = clock() + timeout_seconds
        while True:
            status["state"] = "waiting for lineage identities"
            states = read_states(jobs)
            status["job_states"] = states
            failed = {job: row for job, row in states.items()
                      if row["state"] in FAILED_STATES}
            if failed:
                status["failed_jobs"] = failed
                raise ValueError("Production job failed before workspace binding completed.")
            current = lineage_ids(root, commit)
            if any(current.get(variant) != identity for variant, identity in seen.items()):
                raise ValueError("A previously observed immutable lineage identity disappeared or changed.")
            seen.update(current)
            status["expected_ids"] = current
            if len(current) == 3:
                if old_expected and old_expected != current:
                    raise ValueError("Workspace expected IDs differ from the immutable lineages.")
                status["state"] = "waiting for W&B startup"
                try:
                    api = api_factory()
                    rows = publisher.discover_rows(api, expected_ids=current)
                    visible = {row["variant"]: row["run_id"] for row in rows
                               if row["variant"] in LINEAGES and row["run_id"]}
                    if any(visible.get(variant) != identity for variant, identity in installed.items()):
                        raise ValueError("A previously published run disappeared or changed identity.")
                    status["visible_ids"] = visible
                    if visible != installed or (len(visible) == 3 and not previous.get("api_verified")):
                        receipt = installer(workspace, api=api, expected_ids=current)
                        if not receipt.get("api_verified") or receipt.get("url") != previous["url"]:
                            raise ValueError("Publisher did not verify the owned workspace.")
                        installed, previous = visible, receipt
                    if len(visible) == 3 and previous.get("api_verified"):
                        status.update(state="complete", api_verified=True)
                        save()
                        return status
                    status.pop("last_error_type", None)
                except ValueError:
                    raise
                except Exception as error:
                    # Record only the type: transport/auth errors may contain sensitive URLs.
                    status["last_error_type"] = type(error).__name__
            save()
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for validated lineages and visible W&B runs.")
            sleep(min(poll_seconds, remaining))
    except Exception as error:
        status.update(state="failed", error_type=type(error).__name__)
        if isinstance(error, (ValueError, TimeoutError)):
            status["reason"] = str(error)
        save()
        publisher._write(root / "publication" / "workspace-watch-failure.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if (args.root != ROOT or args.workspace != ROOT / "publication" / "workspace.json"
            or args.jobs_json != ROOT / "production-jobs.json"):
        parser.error("Watcher accepts only this fixed campaign's existing receipts.")
    for package, version in (("wandb", "0.28.1"), ("wandb-workspaces", "0.4.5")):
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Use the isolated publication environment: {package}=={version}.")
    import wandb
    result = watch(args.root, args.workspace, args.jobs_json, args.commit,
                   timeout_seconds=args.timeout_seconds, poll_seconds=args.poll_seconds,
                   api_factory=lambda: wandb.Api(timeout=30))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
