"""Validate and publish one explicitly named completed saturation attempt."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

FILES = {"results.json", "report.html", "measurements.jsonl.gz", "mppi-populations.jsonl.gz",
         "matrix.json", "root-bank.json", "checkpoint.metadata.json"}
METRICS = ("exact_boundary_fraction", "near_boundary_fraction", "mean_absolute_action")
MPPI_DISTRIBUTIONS = {"candidates_all", "candidates_prior", "candidates_proposal", "elites_unweighted",
                      "weighted_elite_selection", "optimized_mean"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with gzip.open(path, "rt") as stream:
        yield from (json.loads(line) for line in stream)


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_completed(bundle, *, allow_smoke=False):
    """Reject corrupt or incomplete results before touching W&B or its journal."""
    import numpy as np
    directory = Path(bundle).resolve()
    checksums = _json(directory / "checksums.json")
    _require(set(checksums) == FILES, "Unexpected or missing diagnostic artifacts")
    for name, digest in checksums.items():
        path = directory / name
        _require(path.is_file() and not path.is_symlink() and _hash(path) == digest,
                 f"Artifact checksum mismatch: {name}")
    record = _json(directory / "results.json")
    _require(record.get("schema_version") == 1 and record.get("kind") == "mppi_sac_action_saturation"
             and record.get("status") == "complete" and record.get("outer_state_unchanged") is True,
             "Saturation diagnostic is incomplete or failed frozen-state checks")
    _require(type(record.get("smoke")) is bool and (allow_smoke or not record["smoke"]),
             "Smoke results require explicit --allow-smoke publication")
    _require(record["roots"] == (1 if record["smoke"] else 100)
             and record["solver_repetitions"] == (1 if record["smoke"] else 3)
             and record["sac_rounds"] == [0, 1, 2, 4] and record["sac_policy_samples"] == 1024,
             "Unexpected root, repetition or saved-actor coverage")
    _require(record.get("bootstrap_resamples") == 2000 and record.get("bootstrap_seed") == 0,
             "Unexpected episode-cluster bootstrap protocol")
    bank = _json(directory / "root-bank.json")
    _require(record["root_bank_sha256"] == checksums["root-bank.json"]
             and record["matrix_sha256"] == checksums["matrix.json"]
             and bank["id"] == record["root_bank_id"]
             and bank["id"] == _canonical_hash({key: value for key, value in bank.items() if key != "id"})
             and bank["checkpoint_sha256"] == record["checkpoint_sha256"]
             and bank["protocol"] == record["root_protocol"]
             and record["science"] == bank["protocol"]["science"], "Source identity differs from recorded artifacts")
    _require(_json(directory / "checkpoint.metadata.json")["checkpoint"]["step"] == 200000,
             "Unexpected checkpoint step")
    settings = record["mppi"]
    for key, value in dict(horizon=1, iterations=4, num_samples=128, num_elites=16,
                           num_pi_trajs=6, temperature=.5, min_std=.05, max_std=2., eval_mode=True).items():
        _require(settings.get(key) == value, "MPPI settings differ from the diagnostic protocol")
    roots = bank["roots"][:1] if record["smoke"] else bank["roots"]
    _require(len(roots) == record["roots"], "Root bank does not cover the selected panel")
    roots_by_id = {root["root_id"]: root for root in roots}
    _require(len(roots_by_id) == len(roots), "Duplicate root identity")
    expected = {(root["root_id"], repeat, operator, iteration, distribution)
                for root in roots for repeat in range(record["solver_repetitions"])
                for operator, rounds, distributions in (
                    ("mppi", [1, 2, 3, 4], MPPI_DISTRIBUTIONS),
                    ("sac", [0, 1, 2, 4], {"policy_samples", "policy_mean"}))
                for iteration in rounds for distribution in distributions}
    rows = list(_rows(directory / "measurements.jsonl.gz"))
    seen, grouped = set(), {}
    for row in rows:
        key = row["root_id"], row["solver_repeat"], row["operator"], row["iteration"], row["distribution"]
        _require(key in expected and key not in seen, "Duplicate or unexpected measurement")
        seen.add(key)
        root = roots_by_id[row["root_id"]]
        _require(all(row.get(key) == root[key] for key in ("episode_id", "seed", "decision_index")),
                 "Measurement root metadata differs")
        _require(all(isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])
                     and -1e-7 <= row[metric] <= 1 + 1e-7 for metric in METRICS), "Invalid action-component statistic")
        if row["operator"] == "sac":
            _require(row["actor_updates"] == 4 * row["iteration"]
                     and row["critic_updates"] == 32 * row["iteration"], "Unexpected SAC update schedule")
        grouped.setdefault((row["operator"], row["iteration"], row["distribution"]), {}).setdefault(
            row["episode_id"], {}).setdefault(row["root_id"], []).append(row)
    _require(seen == expected, "Incomplete per-root measurement panel")
    summaries = {(s["operator"], s["iteration"], s["distribution"]): s for s in record["summary"]}
    _require(len(summaries) == len(record["summary"]) and set(summaries) == set(grouped), "Incomplete summary panel")
    for key, episodes in grouped.items():
        summary = summaries[key]
        _require(summary["episodes"] == len(episodes), "Summary source-episode count differs")
        indices = np.random.default_rng(0).integers(0, len(episodes), size=(2000, len(episodes)))
        for metric in METRICS:
            values = [statistics.fmean(statistics.fmean(row[metric] for row in repeats)
                       for repeats in roots.values()) for _, roots in sorted(episodes.items())]
            stored = summary["metrics"][metric]
            _require(stored["episode_values"] == values and stored["mean"] == statistics.fmean(values),
                     "Published summary differs from raw episode aggregation")
            std = statistics.stdev(values) if len(values) > 1 else None
            ci = np.quantile(np.asarray(values)[indices].mean(axis=1), [.025, .975]).tolist() if len(values) > 1 else None
            _require(stored.get("episode_std") == std and stored.get("ci95") == ci,
                     "Published variability differs from the episode-cluster bootstrap")
    population_expected = {(root["root_id"], repeat, iteration) for root in roots
                           for repeat in range(record["solver_repetitions"]) for iteration in range(1, 5)}
    population_seen = set()
    for row in _rows(directory / "mppi-populations.jsonl.gz"):
        key = row["root_id"], row["solver_repeat"], row["iteration"]
        _require(key in population_expected and key not in population_seen, "Unexpected MPPI population")
        population_seen.add(key)
        _require(len(row["actions"]) == 1 and len(row["actions"][0]) == 128
                 and len(row["values"]) == 128 and len(row["elite_indices"]) == 16
                 and len(row["elite_weights"]) == 16, "Incomplete observed MPPI population")
    _require(population_seen == population_expected, "Missing observed MPPI populations")
    return record, checksums


def history_rows(record):
    rows = {}
    for summary in record["summary"]:
        iteration, operator, distribution = summary["iteration"], summary["operator"], summary["distribution"]
        row = rows.setdefault(iteration, {"saturation/round": iteration})
        if operator == "sac":
            row["saturation/sac/actor_updates"] = iteration * 4
            row["saturation/sac/critic_updates"] = iteration * 32
        for metric, stats in summary["metrics"].items():
            prefix = f"saturation/{operator}/{distribution}/{metric}"
            row[prefix + "/mean"] = stats["mean"]
            if stats.get("episode_std") is not None:
                row[prefix + "/episode_std"] = stats["episode_std"]
            if stats.get("ci95") is not None:
                row[prefix + "/ci95_low"], row[prefix + "/ci95_high"] = stats["ci95"]
    return [rows[iteration] for iteration in sorted(rows)]


@contextmanager
def _locked(directory):
    with (directory / ".publication.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another publisher owns this diagnostic") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_journal(path, receipt):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def publish(bundle, *, attempt_label, run_id, entity="rwgao_b-brown-university", project="ambi-inner-bench",
            mode="online", allow_smoke=False, wandb_module=None):
    directory = Path(bundle).resolve()
    _require(isinstance(attempt_label, str) and attempt_label.strip(), "Choose an explicit nonempty attempt label")
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{8,64}", run_id), "Choose an explicit unique W&B run ID")
    _require(entity and project and mode in ("online", "offline", "disabled"), "Invalid W&B target")
    record, checksums = read_completed(directory, allow_smoke=allow_smoke)
    history = history_rows(record)
    target = dict(entity=entity, project=project, mode=mode, run_id=run_id,
                  attempt_label=attempt_label, source_sha256=checksums["results.json"],
                  checksums_sha256=_hash(directory / "checksums.json"))
    with _locked(directory):
        journal = directory / "publication.json"
        if journal.exists():
            previous = _json(journal)
            _require(previous.get("target") == target, "Publication target differs from the existing attempt")
            if previous.get("status") == "complete":
                return previous
            raise ValueError("Previous publication is uncertain; inspect W&B before retrying")
        if wandb_module is None:
            import wandb as wandb_module
        started = time.perf_counter()
        receipt = dict(target=target, status="uncertain")
        _write_journal(journal, receipt)
        run = None
        try:
            run = wandb_module.init(entity=entity, project=project, id=run_id, resume="never", mode=mode,
                name=f"MPPI/SAC boundaries | mey3rxj8 200k | {attempt_label}", job_type="action-saturation",
                tags=["action-saturation", "mppi", "inner-sac", "smoke" if record["smoke"] else "shared-prior-roots"],
                config=dict(saturation_schema=1, attempt_label=attempt_label, checkpoint_sha256=record["checkpoint_sha256"],
                    root_bank_id=record["root_bank_id"], root_protocol=record["root_protocol"], science=record["science"],
                    diagnostic_source_sha256=record["diagnostic_source_sha256"], mppi=record["mppi"],
                    sac_resolved_config=record["sac_resolved_config"], action_rules=record["action_rules"],
                    source_sha256=checksums["results.json"], artifact_sha256=checksums))
            run.define_metric("saturation/round")
            run.define_metric("saturation/*", step_metric="saturation/round")
            for row in history:
                run.log(row)
            artifact = wandb_module.Artifact("action-saturation-" + run_id, type="action-saturation",
                metadata=dict(attempt_label=attempt_label, checkpoint_sha256=record["checkpoint_sha256"],
                              root_bank_id=record["root_bank_id"], artifact_sha256=checksums))
            for name in sorted(FILES | {"checksums.json"}):
                artifact.add_file(str(directory / name), name=name)
            run.log_artifact(artifact)
            final = {key.replace("saturation/", "final/", 1): value for key, value in history[-1].items()
                     if key.endswith("/mean")}
            run.summary.update({**final, "saturation/status": "complete", "saturation/roots": record["roots"],
                "saturation/solver_repetitions": record["solver_repetitions"], "saturation/record_sha256": checksums["results.json"],
                "runtime/evaluation_seconds": record["elapsed_seconds"],
                **{"runtime/" + key: value for key, value in record["timing"].items()},
                "runtime/publication_seconds_before_finish": time.perf_counter() - started})
            run.finish()
            receipt.update(status="complete", wandb_path=f"{entity}/{project}/{run_id}",
                           history_rows=len(history), publication_seconds=time.perf_counter() - started)
            _write_journal(journal, receipt)
            return receipt
        except BaseException:
            if run is not None:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--attempt-label", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--entity", default="rwgao_b-brown-university")
    parser.add_argument("--project", default="ambi-inner-bench")
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--allow-smoke", action="store_true")
    print(json.dumps(publish(**vars(parser.parse_args(argv))), indent=2))


if __name__ == "__main__":
    main()
