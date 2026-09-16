"""Strict, publication-free consolidation of independently seeded evaluations.

Only complete episode panels are supported.
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
    record_from_model_bundle,
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
    _require(manifest.get("runs"), "Episode merge requires completed AMBI runs")
    reference = manifest.get("reference")
    prior_returns = None
    if reference:
        _require(set(reference) == {"path", "manifest_sha256"},
                 "Episode merge supports a hash-pinned AMBI prior reference")
        reference_path = Path(reference["path"])
        _require(reference_path.is_file() and _hash(reference_path) == reference["manifest_sha256"],
                 "Prior reference checksum mismatch")
        from utils.ambi_benchmark import reference_returns
        prior_returns = reference_returns(reference_path, manifest["checkpoint"]["sha256"], manifest["protocol"])
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
            if prior_returns is not None:
                _require(ep.get("seed") in prior_returns and
                         ep.get("paired_return_delta") == ep["return"] - prior_returns[ep["seed"]],
                         "Paired gain differs from the verified prior reference")
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
        _require(all(manifest.get(k) == base.get(k) for k in ("checkpoint", "protocol", "code", "metric_catalog", "reference")),
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
