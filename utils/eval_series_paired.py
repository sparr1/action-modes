"""Append verified prior-relative gains without rewriting published evaluations."""
from copy import deepcopy
from pathlib import Path
import statistics

from utils import eval_series as series


def _require(condition, message):
    if not condition:
        raise series.SeriesError(message)


def paired_measurements(target, prior):
    """Require identical frozen checkpoint, science, environment and seed protocol."""
    target, prior = series.validate_record(target), series.validate_record(prior)
    _require(target["checkpoint"] == prior["checkpoint"], "Prior checkpoint differs")
    for key in ("backbone", "science"):
        _require(target["identity"][key] == prior["identity"][key], "Prior " + key + " differs")
    _require(prior["identity"]["planner"] == {"type": "prior", "action_rule": "tanh_mean"},
             "Reference must be the frozen SAC policy mean")
    protocols = [{k: v for k, v in r["identity"]["protocol"].items() if k != "action_rule"}
                 for r in (target, prior)]
    _require(protocols[0] == protocols[1], "Prior episode protocol differs")
    _require(prior["identity"]["protocol"].get("action_rule") == "tanh_mean",
             "Prior action rule differs")
    seeds = protocols[0].get("environment_seeds", [])
    maximum = protocols[0].get("max_steps")
    _require(seeds and len(set(seeds)) == len(seeds) and type(maximum) is int and maximum > 0,
             "Missing complete episode protocol")
    by_seed = []
    for record in (target, prior):
        _require(record["metrics"].get("eval/frozen_state_unchanged") is True,
                 "Frozen state verification missing")
        _require(sorted(e.get("seed") for e in record["episodes"]) == sorted(seeds),
                 "Missing or duplicate paired seeds")
        for episode in record["episodes"]:
            _require(type(episode.get("length")) is int and 0 < episode["length"] <= maximum
                     and (episode.get("terminated") or episode.get("truncated"))
                     and not episode.get("truncated_by_evaluator") and not episode.get("capped")
                     and episode.get("status", "complete") == "complete", "Incomplete paired episode")
            _require(type(episode.get("return")) in (int, float), "Nonfinite or missing episode return")
            _require(type(episode.get("solver_seed")) is int, "Missing paired solver seed")
        by_seed.append({e["seed"]: e for e in record["episodes"]})
    episodes = []
    for seed in sorted(seeds):
        left, right = (mapping[seed] for mapping in by_seed)
        _require(left["solver_seed"] == right["solver_seed"], "Paired solver seeds differ")
        episodes.append({"seed": seed, "solver_seed": left["solver_seed"],
                         "return": left["return"], "prior_return": right["return"],
                         "paired_return_delta": left["return"] - right["return"]})
    gains = [e["paired_return_delta"] for e in episodes]
    prior_returns = [e["prior_return"] for e in episodes]
    metrics = {"eval/paired_gain_mean": statistics.mean(gains),
               "eval/paired_gain_sample_std": statistics.stdev(gains) if len(gains) > 1 else None,
               "eval/paired_episodes": len(gains),
               "eval/prior_return_mean": statistics.mean(prior_returns),
               "eval/prior_return_sample_std": statistics.stdev(prior_returns) if len(gains) > 1 else None}
    return episodes, metrics


def prepare_paired_reference(run_dir, target_record_id, prior):
    """Read-only preflight; pin both immutable evaluations and all artifact bytes."""
    run_dir = Path(run_dir).resolve()
    registry = series.load_run(run_dir)
    index = series._read(run_dir / "publication.json")
    _require(target_record_id in index["records"], "Unknown target record")
    entry = index["records"][target_record_id]
    _require(entry["status"] == "published" and not entry.get("record_kind"),
             "Target must be an acknowledged original evaluation")
    target_path = run_dir / "records" / (target_record_id + ".json")
    target = series.validate_record(series._read(target_path), registry["identity"])
    _require(target["record_id"] == target_record_id, "Target record ID differs")
    files = {name: series._file_digest(path) for name, path in target["artifact_files"].items()}
    _require(series._record_fingerprint(target, files) == entry["record_sha256"],
             "Published target or artifacts changed")
    _require(not any(key.startswith("eval/paired_") for key in target["metrics"]),
             "Original evaluation already has paired gains")
    prior = series.validate_record(prior)
    episodes, metrics = paired_measurements(target, prior)
    prior_files = {name: series._file_digest(path) for name, path in prior["artifact_files"].items()}
    provenance = {"method": "seed-paired-return-minus-frozen-sac-prior-v1",
                  "target_record_id": target_record_id, "target_record_sha256": entry["record_sha256"],
                  "prior_record_id": prior["record_id"],
                  "prior_record_sha256": series._record_fingerprint(prior, prior_files),
                  "prior_identity": prior["identity"], "prior_provenance": prior["provenance"]}
    return series.validate_record({"record_kind": series.PAIRED_REFERENCE_KIND,
        "record_id": "paired-" + series._digest(provenance),
        "identity": deepcopy(target["identity"]), "checkpoint": deepcopy(target["checkpoint"]),
        "metrics": metrics, "episodes": episodes, "provenance": provenance,
        "artifact_files": {"original-evaluation-record.json": str(target_path),
                           **{"prior/" + name: path for name, path in prior["artifact_files"].items()}},
        "source_result_path": prior["source_result_path"], "label": target["label"] + " | paired prior gains"})


def stage_paired_reference(run_dir, target_record_id, prior):
    """Idempotently stage a supplement; never replace the original checkpoint row."""
    run_dir = Path(run_dir).resolve()
    record = prepare_paired_reference(run_dir, target_record_id, prior)
    files = {name: series._file_digest(path) for name, path in record["artifact_files"].items()}
    fingerprint = series._record_fingerprint(record, files)
    rid = record["record_id"]
    with series._lock(run_dir / ".records.lock"):
        index = series._read(run_dir / "publication.json")
        target = index["records"][target_record_id]
        _require(target["status"] == "published" and target["record_sha256"] ==
                 record["provenance"]["target_record_sha256"], "Target changed during pairing")
        for other_id, entry in index["records"].items():
            if entry.get("target_record_id") == target_record_id:
                _require(other_id == rid and entry["record_sha256"] == fingerprint,
                         "Target already has a different paired reference")
                return {"record_id": rid, "status": entry["status"]}
        _require(rid not in index["records"], "Supplement record ID collision")
        series._atomic_json(run_dir / "records" / (rid + ".json"), record)
        index["records"][rid] = {"status": "staged", "record_kind": series.PAIRED_REFERENCE_KIND,
            "target_record_id": target_record_id, "accepted_order": len(index["records"]),
            "checkpoint_step": record["checkpoint"]["step"],
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "record_sha256": fingerprint, "artifact_sha256": files}
        series._atomic_json(run_dir / "publication.json", index)
    return {"record_id": rid, "status": "staged"}
