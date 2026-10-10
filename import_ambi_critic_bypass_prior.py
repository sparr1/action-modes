"""Import exactly five verified prior episodes before critic-bypass workers start.

This narrow importer reuses return evidence only. Historical controller timing
and unobserved action-bank checks are deliberately absent. The entire five-task
directory is staged and published by one rename after all inputs are validated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile


PROTOCOL = "closed-loop-critic-bypass-v1"
ARMS = ("actor_mean", "learned_q", "model_score", "prior", "mppi_h3")
PRIOR_MANIFEST = Path("/oscar/scratch/rgao48/ambi/aux-return-mppi/20260921-other-backbones/"
    "target10p5_shared/production/prior/step_800000/bundle/manifest.json")
PRIOR_SHA256 = "6754c5c8fae547ac6ed9547c48affcfc7a7b4ef0c695acb9246a0e6dea1f99df"
CHECKPOINT_SHA256 = "62606f9d905be4798db89f916993355ff0c6edc994d2780636edbf876b1726a4"
METADATA_SHA256 = "dc1c98a9b6937b193c2d19821875323dd3cff5097c5f8caf426bbbe1f9842eab"
SOURCE_RUN = "rwgao_b-brown-university/ambi/aux6428346x0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_campaign(campaign, *, smoke):
    identity = {key: value for key, value in campaign.items() if key != "campaign_id"}
    require(campaign.get("campaign_id") == fingerprint(identity), "Campaign identity hash mismatch.")
    require(campaign.get("protocol") == PROTOCOL and campaign.get("smoke") is smoke,
            "Wrong campaign protocol or smoke mode.")
    seeds = [101] if smoke else list(range(101, 121))
    require(campaign.get("seeds") == seeds and isinstance(campaign.get("arms"), list)
            and len(campaign["arms"]) == len(ARMS) and set(campaign["arms"]) == set(ARMS)
            and campaign.get("controller_seed") == 55
            and campaign.get("max_steps") == (4 if smoke else 500), "Campaign episode grid differs.")
    require(campaign.get("checkpoint") == dict(step=800000, sha256=CHECKPOINT_SHA256,
            metadata_sha256=METADATA_SHA256), "Campaign checkpoint differs from the pinned prior.")
    require(isinstance(campaign.get("config"), dict)
            and all(campaign.get(key) == value for key, value in campaign["config"].items()),
            "Resolved campaign config conflicts with identity.")
    for key, count in (("source_commit", 40), ("source_tree", 40), ("config_sha256", 64)):
        require(re.fullmatch(r"[0-9a-f]{%d}" % count, str(campaign.get(key, ""))), f"Invalid {key}.")
    expected = [dict(task_id=f"{arm}-seed-{seed}", arm=arm, env_seed=seed, controller_seed=55)
                for arm in ARMS for seed in seeds]
    tasks = campaign.get("task_list")
    require(isinstance(tasks, list) and len(tasks) == len(expected)
            and sorted(tasks, key=lambda task: task["task_id"]) == sorted(expected, key=lambda task: task["task_id"]),
            "Campaign tasks differ from the exact arm/seed grid.")


def verify_smoke(root, campaign):
    smoke = read(root / "campaign.json")
    verify_campaign(smoke, smoke=True)
    for key in ("source_commit", "source_tree", "config_sha256", "checkpoint", "checkpoint_path"):
        require(smoke.get(key) == campaign.get(key), "Smoke and production differ: " + key)
    pins = {}
    for task in smoke["task_list"]:
        directory = root / "tasks" / task["task_id"]
        result, manifest = read(directory / "result.json"), read(directory / "manifest.json")
        require(result.get("complete") is True and result.get("campaign_id") == smoke["campaign_id"]
                and result.get("protocol") == PROTOCOL
                and all(result.get(k) == v for k, v in task.items()), "Incomplete/conflicting smoke task.")
        require(result.get("episode_length") == 4 and result.get("source_commit") == smoke["source_commit"],
                "Smoke execution evidence differs.")
        checks = result.get("checks", {})
        require(checks.get("outer_state_unchanged") is True and all(v is True for v in checks.values()),
                "Smoke integrity check failed.")
        require(manifest.get("status") == "complete" and manifest.get("result_sha256") == sha256(directory/"result.json"),
                "Smoke completion fingerprint differs.")
        if task["arm"] == "prior":
            require(checks.get("prior_execution_path_parity") is True,
                    "Prior smoke has not verified ordinary/audit action parity.")
        pins[task["task_id"]] = sha256(directory / "result.json")
    return dict(campaign_id=smoke["campaign_id"], root=str(root), result_sha256=pins)


def verified_prior(path, campaign, resolved):
    require(path.resolve() == PRIOR_MANIFEST.resolve(), "Only the audited prior manifest path is allowed.")
    require(sha256(path) == PRIOR_SHA256, "Prior manifest hash changed.")
    source = read(path)
    require(source.get("status") == "complete", "Prior manifest is incomplete.")
    cp = source["checkpoint"]
    require(cp.get("sha256") == CHECKPOINT_SHA256 and cp.get("source_run") == SOURCE_RUN
            and cp.get("path") == campaign["checkpoint_path"]
            and cp["metadata"]["checkpoint"]["step"] == 800000, "Prior checkpoint identity differs.")
    protocol, config = source["protocol"], resolved["algorithm_config"]
    expected = dict(environment=resolved["environment"], env_wrapper=config.get("env_wrapper"),
        env_wrappers=config.get("env_wrappers", []), observation=config["alg_params"]["obs"],
        action_rule="tanh_mean", max_steps=500, controller_seed=55, seed_scheme="sha256-v1")
    require(protocol == expected, "Prior environment or episode/action protocol differs.")
    require(len(source["runs"]) == 1, "Expected one prior controller in the source bundle.")
    run = source["runs"][0]
    require(run.get("status") == "complete" and run.get("selector") == "reference/prior",
            "Expected a completed frozen prior run.")
    params, result = run["resolved_config"], run["result"]
    require(params.get("inner_operator") == "none" and params.get("inner_actor_source") == "sac"
            and params.get("inner_execution_action") == "mean" and params.get("compile") is False,
            "Historical prior controller semantics differ.")
    require(result.get("outer_state_unchanged") is True
            and result.get("outer_updates_before") == result.get("outer_updates_after")
            and result.get("outer_updates_before") is not None, "Prior frozen-state proof is missing.")
    episodes = run["episodes"]
    reported = result["episodes"]
    require(len(episodes) == len(reported) and all(
        all(episode.get(key) == value for key, value in row.items())
        for episode, row in zip(episodes, reported)), "Manifest and result episode evidence disagree.")
    require([e["seed"] for e in episodes] == list(range(101, 106)), "Exactly five prior seeds 101–105 are required.")
    for episode in episodes:
        require(type(episode.get("return")) in (int, float) and math.isfinite(episode["return"]),
                "Prior episode return must be finite.")
        require(episode.get("length") == 500 and episode.get("terminated") is False
                and episode.get("truncated") is True and episode.get("truncated_by_evaluator") is False
                and episode.get("capped") is False, "Prior episode is not a completed 500-decision episode.")
        expected_seed = int(fingerprint(["sha256-v1", 55, "episode", episode["seed"]])[:8], 16)
        require(episode.get("solver_seed") == expected_seed, "Historical controller seed differs.")
    return source, episodes


def import_prior(campaign_root, manifest, smoke_root):
    root, path, smoke_root = map(lambda p: Path(p).resolve(), (campaign_root, manifest, smoke_root))
    target = root / "tasks"
    if target.exists() or target.is_symlink():
        raise FileExistsError("Import must precede workers; the production tasks directory already exists.")
    campaign, resolved = read(root / "campaign.json"), read(root / "resolved.json")
    verify_campaign(campaign, smoke=False)
    smoke_proof = verify_smoke(smoke_root, campaign)
    source, episodes = verified_prior(path, campaign, resolved)
    imported_at = datetime.now(timezone.utc).isoformat()
    # All five records are validated before creating any output directory.
    records = []
    for episode in episodes:
        seed = episode["seed"]
        records.append(dict(schema_version=1, protocol=PROTOCOL, campaign_id=campaign["campaign_id"],
            task_id=f"prior-seed-{seed}", arm="prior", env_seed=seed, controller_seed=55,
            source_commit=campaign["source_commit"], complete=True, reused_reference=True,
            reference_source=str(path), reference_sha256=PRIOR_SHA256,
            reference_code=source["code"], reference_protocol=source["protocol"],
            smoke_proof=smoke_proof, imported_at_utc=imported_at,
            runtime=dict(timing_comparable=False, execution="historical eager prior return reuse"),
            episode_return=episode["return"], episode_length=500, terminated=False,
            truncated=True, evaluator_capped=False, solver_seed=episode["solver_seed"],
            controller_time_mean_s=None, controller_time_p95_s=None, selection_time_mean_s=None,
            selection_time_p95_s=None, controller_times_s=[], selection_times_s=[],
            diagnostic_times_s=[], diagnostics=[], elapsed_seconds=None,
            checks=dict(outer_state_unchanged=True, full_episode=True, finite_returns=True,
                historical_reference_verified=True, prior_execution_path_parity=True,
                source_episodes_consistent=True)))
    staging = Path(tempfile.mkdtemp(prefix=".prior-import-", dir=root))
    try:
        for result in records:
            directory = staging / result["task_id"]
            directory.mkdir()
            write(directory / "result.json", result)
            write(directory / "manifest.json", dict(schema_version=1, protocol=PROTOCOL,
                campaign_id=campaign["campaign_id"], task_id=result["task_id"], status="complete",
                source_commit=campaign["source_commit"], reused_reference=True,
                result_sha256=sha256(directory / "result.json")))
            write(directory / "progress.json", dict(task_id=result["task_id"], arm="prior", env_seed=result["env_seed"],
                controller_seed=55, status="complete", decision=500, episode_return=result["episode_return"],
                reused_reference=True))
        if target.exists() or target.is_symlink():
            raise FileExistsError("A production tasks directory appeared during prior import.")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return dict(imported=5, seeds=list(range(101, 106)), campaign_id=campaign["campaign_id"],
                reference_sha256=PRIOR_SHA256, comparable_timing_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_prior(args.campaign_root, args.manifest, args.smoke_root)))


if __name__ == "__main__":
    main()
