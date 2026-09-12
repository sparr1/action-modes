"""Strict, publication-free consolidation of independently seeded evaluations.

Only complete episode panels and real simulator calibration panels are supported.
Shards must differ only in their requested episode seeds and root-bank identity.
No optimizer or simulator runs here. Raw traces, solves, references and source
manifests are preserved; summaries and the full-panel identity are rebuilt.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from utils.ambi_benchmark import atomic_json, canonical_hash, read_json
from utils.ambi_diagnostic_series import (
    build_diagnostic_record, read_diagnostic_bundle, record_from_model_bundle,
    write_diagnostic_bundle,
)

SEAL = "seed-shard-checksums.json"
RECEIPT = "seed-shard-merge.json"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(directory, name):
    path = Path(name)
    _require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
             "Artifact path must remain inside its bundle")
    target = directory / path
    _require(target.resolve().is_relative_to(directory.resolve()) and not target.is_symlink(),
             "Artifact path escapes its bundle or is a symlink")
    _require(target.is_file(), f"Missing artifact {name}")
    return target


def _seeds(values):
    values = list(values)
    _require(values and all(type(s) is int and 0 <= s < 2**32 for s in values)
             and len(values) == len(set(values)), "Seeds must be unique NumPy seed integers")
    return values


def _partition(groups, expected):
    seen = set()
    for group in groups:
        _seeds(group)
        _require(not seen.intersection(group), "Seed shards overlap")
        seen.update(group)
    _require(seen == set(expected), "Seed shards have missing or unexpected seeds")


def _inputs(sources, output):
    sources = [Path(path).expanduser().resolve() for path in sources]
    output = Path(output).expanduser().resolve()
    _require(len(sources) > 0 and len(set(sources)) == len(sources), "Source bundles must be distinct")
    if output.exists():
        raise FileExistsError(output)
    _require(all(not output.is_relative_to(source) and not source.is_relative_to(output)
                 for source in sources), "Output and source bundles must not contain one another")
    return sources, output


def _files(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "Bundle symlinks are not supported")
        if path.is_file() and path != directory / SEAL:
            result[str(path.relative_to(directory))] = _hash(path)
    return result


def seal_episode_bundle(directory):
    """Hash a completed worker bundle before transferring ownership to a merger.

    Legacy episode bundles have no trace hashes. Workers call this immediately
    after evaluation; merging requires this immutable seal. Repeated sealing is
    read-only and succeeds only if the original hashes still match.
    """
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    _require(manifest.get("status") == "complete", "Cannot seal an incomplete episode bundle")
    for run in manifest.get("runs", []):
        _require(run.get("status") == "complete", "Cannot seal an incomplete episode run")
    files = _files(directory)
    seal = {"schema_version": 1, "kind": "ambi_episode_seed_shard", "files": files}
    seal["sha256"] = canonical_hash(seal)
    if (directory / SEAL).exists():
        _require(read_json(directory / SEAL) == seal, "Episode shard checksum mismatch")
    else:
        atomic_json(directory / SEAL, seal)
    return directory / SEAL


def _read_episode(directory):
    seal = read_json(directory / SEAL)
    _require(seal.get("schema_version") == 1 and seal.get("kind") == "ambi_episode_seed_shard"
             and seal.get("sha256") == canonical_hash({k: v for k, v in seal.items() if k != "sha256"}),
             "Invalid episode shard seal")
    _require(seal.get("files") == _files(directory), "Episode shard checksum mismatch")
    manifest = read_json(directory / "manifest.json")
    _require(manifest.get("schema_version") == 1 and manifest.get("status") == "complete",
             "Episode bundle must be complete")
    _require(manifest.get("runs") and not manifest.get("reference"),
             "Episode merge requires AMBI bundles without native reference provenance")
    _require(not manifest.get("protocol", {}).get("root_bank_id"),
             "Episode merge does not merge observation-bank solves")
    # Existing report validation checks trace coordinates, counters and metrics.
    from report_ambi_benchmark import load_bundles
    load_bundles([directory])
    groups = []
    for run in manifest["runs"]:
        result = run.get("result", {})
        _require(run.get("status") == "complete" and run.get("kind") == "episodes"
                 and not run.get("roots"), "Only completed episode runs can be merged")
        _require(result.get("outer_state_unchanged") is True
                 and type(result.get("outer_updates_before")) is int
                 and result["outer_updates_before"] == result.get("outer_updates_after"),
                 "Frozen outer-state checks failed or are absent")
        seeds = _seeds(result.get("environment_seeds", []))
        episodes = run.get("episodes", [])
        _require([ep.get("seed") for ep in episodes] == seeds, "Episode seeds differ from declared shard seeds")
        returned = {ep["seed"]: ep for ep in result.get("episodes", [])}
        _require(len(returned) == len(seeds) and set(returned) == set(seeds), "Missing returned episodes")
        decisions = set()
        for name in run.get("trace_files", []):
            with gzip.open(_safe(directory, name), "rt") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["phase"] == "decision":
                        coord = (row["episode_id"], row["decision_index"])
                        _require(coord not in decisions, "Duplicate decision trace")
                        decisions.add(coord)
        expected_decisions = set()
        for ep in episodes:
            _require(ep.get("episode_id") == f"seed-{ep['seed']}" and type(ep.get("length")) is int
                     and 0 < ep["length"] <= manifest["protocol"]["max_steps"]
                     and math.isfinite(ep.get("return", math.nan))
                     and (ep.get("terminated") is True or ep.get("truncated") is True),
                     "Incomplete or invalid episode")
            _require(all(ep.get(k) == returned[ep["seed"]].get(k)
                         for k in ("return", "length", "terminated", "truncated", "solver_seed")),
                     "Result episodes disagree with manifest episodes")
            expected_decisions.update((ep["episode_id"], d) for d in range(ep["length"]))
        _require(decisions == expected_decisions, "Missing or unexpected decision traces")
        _require(run.get("config_hash") == canonical_hash(run.get("config")), "Run configuration checksum mismatch")
        if run.get("togo_return_probe"):
            record = record_from_model_bundle(directory, run["selector"], "merge-validation", bootstrap_resamples=1)
            _require(record["status"] == "complete", "Incomplete model-probe rows")
        groups.append(seeds)
    _require(all(group == groups[0] for group in groups), "Selectors have different shard seed panels")
    return manifest, groups[0], seal


def _summary(values):
    values = list(values)
    return {"count": len(values), "sum": sum(values), "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def _combine_stats(items):
    """Pool population moments using counts, never mean shard means."""
    items = [item for item in items if item.get("count", 0)]
    if not items:
        return _summary([])
    count = sum(item["count"] for item in items)
    total = sum(item["sum"] for item in items)
    mean = total / count
    variance = sum(item["count"] * (item["std"] ** 2 + (item["mean"] - mean) ** 2)
                   for item in items) / count
    return {"count": count, "sum": total, "mean": mean, "std": math.sqrt(variance),
            "min": min(item["min"] for item in items), "max": max(item["max"] for item in items)}


def _sum_dicts(items):
    return {key: sum(item.get(key, 0) for item in items) for key in set().union(*items)}


def _round_summaries(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault((row["round_index"], row["actor_updates"], row["critic_updates"]), {})
        for key, value in row["metrics"].items():
            if value is not None:
                group.setdefault(key, []).append(value)
    return [{"round_index": j, "actor_updates": a, "critic_updates": c,
             "metrics": {key: _summary(values) for key, values in sorted(metrics.items())}}
            for (j, a, c), metrics in sorted(groups.items())]


def merge_episode_bundles(source_dirs, output_dir, *, expected_seeds):
    """Produce one complete BenchmarkBundle with all requested seeds and traces."""
    sources, output = _inputs(source_dirs, output_dir)
    expected = _seeds(expected_seeds)
    loaded = [_read_episode(source) for source in sources]
    _partition([seeds for _, seeds, _ in loaded], expected)
    order = {seed: index for index, seed in enumerate(expected)}
    pairs = sorted(zip(sources, loaded), key=lambda pair: min(order[s] for s in pair[1][1]))
    sources, loaded = map(list, zip(*pairs))
    base = loaded[0][0]
    for manifest, _, _ in loaded:
        _require(all(manifest.get(k) == base.get(k) for k in ("checkpoint", "protocol", "code", "metric_catalog")),
                 "Episode shards have incompatible checkpoint, protocol, science or metric semantics")
        _require([run["selector"] for run in manifest["runs"]] == [run["selector"] for run in base["runs"]],
                 "Episode shard selectors differ")
    manifest = copy.deepcopy(base)
    receipt = {"schema_version": 1, "kind": "ambi_episode_seed_merge", "expected_seeds": expected,
               "sources": [{"path": str(source), "seeds": seeds, "seal_sha256": seal["sha256"],
                            "manifest_sha256": seal["files"]["manifest.json"]}
                           for source, (_, seeds, seal) in zip(sources, loaded)]}
    manifest["evaluation_id"] = canonical_hash(receipt)[:32]
    manifest["seed_shard_merge"] = receipt
    manifest["elapsed_seconds"] = sum(item[0].get("elapsed_seconds", 0.) for item in loaded)
    manifest["elapsed_semantics"] = "sum_of_worker_elapsed_not_campaign_wall_time"
    for index, run in enumerate(manifest["runs"]):
        parts = [item[0]["runs"][index] for item in loaded]
        for part in parts:
            for key in ("id", "selector", "config", "config_hash", "kind", "resolved_config", "togo_return_probe", "q_scale"):
                _require(part.get(key) == run.get(key), f"Episode shard run {key} differs")
        run["episodes"] = sorted([ep for part in parts for ep in part["episodes"]], key=lambda ep: order[ep["seed"]])
        run["trace_files"] = [name for part in parts for name in part["trace_files"]]
        _require(len(run["trace_files"]) == len(set(run["trace_files"])), "Trace artifact paths collide")
        for key in ("serialization_seconds", "publication_seconds", "initialization_seconds", "warmup_including_compile_seconds"):
            if any(key in part for part in parts):
                run[key] = sum(part.get(key, 0.) for part in parts)
        for key in ("nonfinite_trace_metrics",):
            if any(key in part for part in parts):
                run[key] = _sum_dicts([part.get(key, {}) for part in parts])
        if run.get("togo_return_probe"):
            run["togo_probe_rows"] = [row for part in parts for row in part["togo_probe_rows"]]
        result = run["result"]
        changing = {"environment_seeds", "episodes", "return", "episode_length", "model_metrics", "bank_metrics",
                    "model_metric_availability", "nonfinite_model_metrics", "nonfinite_trace_metrics",
                    "paired_return_delta_vs_prior", "paired_return_delta_vs_reference", "togo_round_summaries"}
        _require(all({k: v for k, v in part["result"].items() if k not in changing}
                     == {k: v for k, v in result.items() if k not in changing} for part in parts),
                 "Episode shard scientific results differ")
        result["environment_seeds"] = expected
        result["episodes"] = sorted([ep for part in parts for ep in part["result"]["episodes"]], key=lambda ep: order[ep["seed"]])
        result["return"] = _summary(ep["return"] for ep in run["episodes"])
        result["episode_length"] = _summary(ep["length"] for ep in run["episodes"])
        for key in ("model_metrics", "bank_metrics"):
            names = set().union(*(part["result"].get(key, {}) for part in parts))
            result[key] = {name: _combine_stats([part["result"].get(key, {}).get(name, {}) for part in parts])
                           for name in sorted(names)}
        result["model_metric_availability"] = sorted(result["model_metrics"])
        for key in ("nonfinite_model_metrics", "nonfinite_trace_metrics"):
            result[key] = _sum_dicts([part["result"].get(key, {}) for part in parts])
        deltas = [ep["paired_return_delta"] for ep in run["episodes"] if "paired_return_delta" in ep]
        if deltas:
            _require(len(deltas) == len(expected), "Only part of the episode panel has prior references")
            result["paired_return_delta_vs_prior"] = {k: v for k, v in _summary(deltas).items() if k != "sum"}
        if run.get("togo_return_probe"):
            result["togo_round_summaries"] = _round_summaries(run["togo_probe_rows"])
    for run in manifest["runs"]:
        result = run["result"]
        references = [part for part in manifest["runs"] if part["result"].get("comparison") == result.get("comparison")
                      and part["result"].get("variant") == result.get("reference_variant")]
        if "paired_return_delta_vs_reference" in result:
            _require(len(references) == 1, "Missing or ambiguous selected reference")
            values = {ep["seed"]: ep["return"] for ep in references[0]["episodes"]}
            result["paired_return_delta_vs_reference"] = _summary(ep["return"] - values[ep["seed"]]
                                                                  for ep in run["episodes"])
    # All validation precedes creation; an error never leaves a publishable output.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seed-merge-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for index, (source, (_, _, seal)) in enumerate(zip(sources, loaded)):
            for name, digest in seal["files"].items():
                src = _safe(source, name)
                _require(_hash(src) == digest, "Episode artifact changed during merge")
                archived = staging / "shards" / str(index) / name
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, archived)
            shutil.copyfile(source / SEAL, staging / "shards" / str(index) / SEAL)
            for run in loaded[index][0]["runs"]:
                for name in run["trace_files"]:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
        atomic_json(staging / "manifest.json", manifest)
        atomic_json(staging / RECEIPT, receipt)
        seal_episode_bundle(staging)
        _read_episode(staging)
        staging.rename(output)
    return output


def _root_bank(directory, record):
    from utils.ambi_real_calibration import SimulatorSnapshot
    import numpy as np
    bank = read_json(directory / "artifacts/root-bank.json")
    protocol = record["identity"]["protocol"]
    _require(bank.get("schema_version") == 1 and bank.get("kind") == "humanoid_integration_state_bank"
             and bank.get("complete") is True and bank.get("id") == canonical_hash({k: v for k, v in bank.items() if k != "id"})
             and bank.get("id") == protocol.get("root_bank_id")
             and bank.get("checkpoint_sha256") == record["identity"]["checkpoint"]["sha256"]
             and bank.get("protocol") == protocol.get("root_protocol"), "Incompatible or corrupt simulator root bank")
    seeds = _seeds(bank["protocol"]["seeds"])
    expected = {(seed, decision) for seed in seeds for decision in bank["protocol"]["decisions"]}
    actual = [(root["seed"], root["decision_index"]) for root in bank["roots"]]
    _require(len(actual) == len(set(actual)) and set(actual) == expected, "Missing or duplicate simulator roots")
    runtime = None
    for root in bank["roots"]:
        state = SimulatorSnapshot.from_dict(root["snapshot"]).state()["environment"]
        seed, decision = root["seed"], root["decision_index"]
        observation = np.asarray(root["observation"], dtype=np.float32)
        _require(root["episode_id"] == f"seed-{seed}" and root["root_id"] == f"seed-{seed}-decision-{decision}"
                 and root["dtype"] == "float32" and list(observation.shape) == root["shape"]
                 and observation.ndim == 1 and np.isfinite(observation).all()
                 and np.array_equal(observation, state["observation"])
                 and state["step_count"] == state["runtime"]["action_repeat"] * decision,
                 "Simulator root differs from integration snapshot")
        if runtime is None:
            runtime = state["runtime"]
        _require(state["runtime"] == runtime, "Simulator runtime differs within root bank")
    _require([ep["episode_id"] for ep in bank["episodes"]] == [f"seed-{seed}" for seed in seeds]
             and all(type(ep.get("length")) is int and 0 < ep["length"] <= bank["protocol"]["max_steps"]
                     and math.isfinite(ep.get("return", math.nan)) for ep in bank["episodes"]),
             "Root-bank source episodes are incomplete")
    return bank, runtime


def _real_signature(record):
    identity = copy.deepcopy(record["identity"])
    identity["protocol"].pop("root_bank_id")
    identity["protocol"]["root_protocol"].pop("seeds")
    return identity, record["attempt_label"], record["aggregation"]


def _validate_real_artifacts(directory, record, bank):
    """Require each completed solve and prior reference, beyond file hashes."""
    manifest = read_json(directory / "manifest.json")
    files = manifest["artifact_files"]
    protocol = record["identity"]["protocol"]
    expected_solves, expected_refs = set(), set()
    row_key = lambda row: (row["round_index"], row["rollout_repeat"])
    for root in bank["roots"]:
        reference_name = f"artifacts/references/{root['root_id']}.json"
        expected_refs.add(reference_name)
        _require(reference_name in files, "Missing prior reference artifact")
        reference = read_json(_safe(directory, reference_name))
        ref_identity = reference.get("identity", {})
        _require(reference.get("complete") is True
                 and reference.get("sha256") == canonical_hash(reference.get("result"))
                 and ref_identity.get("checkpoint") == bank["checkpoint_sha256"]
                 and ref_identity.get("bank_id") == bank["id"]
                 and ref_identity.get("root_id") == root["root_id"]
                 and ref_identity.get("snapshot_hash") == canonical_hash(root["snapshot"])
                 and ref_identity.get("prefix_action_rule", "sampled") == protocol.get("prefix_action_rule", "sampled"),
                 "Prior reference identity or checksum mismatch")
        real_reference = reference["result"].get("real", {}).get("rows", [])
        _require(len(reference["result"].get("model_rows", [])) == protocol["rollout_repetitions"]
                 and len(real_reference) == protocol["rollout_repetitions"]
                 and all(row.get("rollout_index") == index and row.get("mc_complete") is True
                         and row.get("episode_cutoff_complete") is True for index, row in enumerate(real_reference)),
                 "Prior reference rollout panel is incomplete")
        for repeat in range(protocol["solver_repetitions"]):
            name = f"artifacts/solves/{root['root_id']}-solver-{repeat}.json"
            expected_solves.add(name)
            _require(name in files, "Missing completed solve artifact")
            solve = read_json(_safe(directory, name))
            rows = [row for row in record["rows"] if row["root_id"] == root["root_id"] and row["solver_repeat"] == repeat]
            _require(sorted(solve.get("rows", []), key=row_key) == sorted(rows, key=row_key),
                     "Solve artifact differs from paired rows")
            _require(solve.get("trace_events"), "Solve artifact is missing training trace")
            if protocol["model_probe_rollouts"]:
                probes = solve.get("model_probes", [])
                _require([row["round_index"] for row in probes] == list(range(record["identity"]["setting"]["inner_rounds"] + 1)),
                         "Solve artifact has incomplete model probes")
                expected_probes = [dict(episode_id=root["episode_id"], root_id=root["root_id"],
                                        solver_repeat=repeat, **event)
                                   for event in solve["trace_events"] if event["phase"] == "probe"]
                _require(probes == expected_probes, "Solve model probes differ from training trace")
            for row in rows:
                _require(row.get("policy_noise_seed") == ref_identity.get("noise_seed")
                         and row.get("q_pair_seed") == ref_identity.get("q_pair_seed"),
                         "Solve and prior reference random pairing differ")
    _require({name for name in files if name.startswith("artifacts/solves/")} == expected_solves,
             "Unexpected solve artifacts")
    _require({name for name in files if name.startswith("artifacts/references/")} == expected_refs,
             "Unexpected prior reference artifacts")


def merge_real_bundles(source_dirs, output_dir, *, expected_seeds):
    """Rebuild a full common-root diagnostic identity and episode-cluster intervals."""
    sources, output = _inputs(source_dirs, output_dir)
    expected_seeds = _seeds(expected_seeds)
    records = [read_diagnostic_bundle(source) for source in sources]
    _require(all(record["status"] == "complete" and record["identity"]["scope"] == "common_prior_roots"
                 and record["timing"].get("outer_state_unchanged") is True for record in records),
             "Real shards must be complete with unchanged outer state")
    banks = [_root_bank(source, record) for source, record in zip(sources, records)]
    _partition([bank["protocol"]["seeds"] for bank, _ in banks], expected_seeds)
    order = {seed: index for index, seed in enumerate(expected_seeds)}
    indexed = sorted(zip(sources, records, banks), key=lambda item: min(order[s] for s in item[2][0]["protocol"]["seeds"]))
    sources, records, banks = map(list, zip(*indexed))
    _require(all(_real_signature(record) == _real_signature(records[0]) for record in records),
             "Real shards have incompatible science, setting, attempt or sampling protocol")
    _require(all(runtime == banks[0][1] for _, runtime in banks), "Simulator runtime differs across shards")
    for source, record, (bank, _) in zip(sources, records, banks):
        options = record["identity"]["protocol"]
        plan = {"roots": [{"episode_id": root["episode_id"], "root_id": root["root_id"]} for root in bank["roots"]],
                "solver_repeats": options["solver_repetitions"], "rollout_repeats": options["rollout_repetitions"],
                "rounds": options["rounds"]}
        rebuilt = build_diagnostic_record(record["identity"], record["rows"], plan,
            attempt_label=record["attempt_label"], timing=record["timing"],
            bootstrap_resamples=record["aggregation"]["bootstrap_resamples"],
            bootstrap_seed=record["aggregation"]["bootstrap_seed"])
        _require(rebuilt == record, "Real shard expected coverage or summaries differ from raw measurements")
        _require(all(row.get("mc_complete") is True and row.get("truncated") is False for row in record["rows"]),
                 "Real continuations are incomplete")
        _validate_real_artifacts(source, record, bank)
    merged_bank = copy.deepcopy(banks[0][0])
    merged_bank["protocol"]["seeds"] = expected_seeds
    merged_bank["roots"] = sorted([root for bank, _ in banks for root in bank["roots"]],
                                  key=lambda root: (order[root["seed"]], root["decision_index"]))
    episode_order = {f"seed-{seed}": index for seed, index in order.items()}
    merged_bank["episodes"] = sorted([ep for bank, _ in banks for ep in bank["episodes"]],
                                     key=lambda ep: episode_order[ep["episode_id"]])
    merged_bank["id"] = canonical_hash({k: v for k, v in merged_bank.items() if k != "id"})
    identity = copy.deepcopy(records[0]["identity"])
    identity["protocol"]["root_protocol"] = merged_bank["protocol"]
    identity["protocol"]["root_bank_id"] = merged_bank["id"]
    timing = {}
    for record in records:
        for key, value in record["timing"].items():
            if type(value) in (float, int):
                timing[key] = timing.get(key, 0) + value
    timing.update(outer_state_unchanged=True, elapsed_semantics="sum_of_worker_elapsed_not_campaign_wall_time",
                  parallel_worker_max_elapsed_seconds=max(record["timing"].get("total_elapsed_seconds", 0) for record in records))
    receipt = {"schema_version": 1, "kind": "ambi_real_seed_merge", "expected_seeds": expected_seeds,
               "root_bank_id": merged_bank["id"], "sources": [
                   {"path": str(source), "seeds": bank["protocol"]["seeds"], "root_bank_id": bank["id"],
                    "record_sha256": record["record_sha256"], "manifest_sha256": _hash(source / "manifest.json")}
                   for source, record, (bank, _) in zip(sources, records, banks)]}
    rows = [row for record in records for row in record["rows"]]
    expected = [coordinate for record in records for coordinate in record["expected"]]
    merged = build_diagnostic_record(identity, rows, expected, attempt_label=records[0]["attempt_label"], timing=timing,
        bootstrap_resamples=records[0]["aggregation"]["bootstrap_resamples"], bootstrap_seed=records[0]["aggregation"]["bootstrap_seed"])
    _require(merged["status"] == "complete", "Merged diagnostic panel is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".real-seed-merge-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        atomic_json(temporary / "root-bank.json", merged_bank)
        atomic_json(temporary / RECEIPT, receipt)
        artifacts = {"root-bank.json": temporary / "root-bank.json", RECEIPT: temporary / RECEIPT}
        common = {}
        for index, source in enumerate(sources):
            manifest = read_json(source / "manifest.json")
            for name, digest in manifest["artifact_files"].items():
                src = _safe(source, name)
                _require(_hash(src) == digest, "Real artifact changed during merge")
                artifacts[f"shards/{index}/{name}"] = src
                relative = name.removeprefix("artifacts/")
                if relative.startswith("solves/"):
                    _require(relative not in artifacts, "Solve artifact paths collide")
                    artifacts[relative] = src
                elif relative in ("matrix.json", "checkpoint.metadata.json"):
                    _require(common.setdefault(relative, digest) == digest, "Calibration source metadata differs")
                    artifacts[relative] = src
            for name in ("manifest.json", "paired-rows.jsonl.gz", "report.html", "completion.json"):
                if (source / name).exists():
                    artifacts[f"shards/{index}/{name}"] = _safe(source, name)
        staging = temporary / "bundle"
        write_diagnostic_bundle(staging, merged, artifact_files=artifacts)
        atomic_json(staging / RECEIPT, receipt)
        read_diagnostic_bundle(staging)
        staging.rename(output)
    return output
