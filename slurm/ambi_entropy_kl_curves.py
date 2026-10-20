"""Persistent native checkpoint curves built from immutable evaluation records.

Raw experiment runs remain intact. Only the checkpoint-dependent KL coefficient
is canonicalized into its audited rule; every other identity field must match.
The existing eval_series publisher provides locking, retry reconciliation and
one accepted result per checkpoint. No models or episodes are evaluated here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series

PROFILE = Path(__file__).resolve().parents[1] / "configs/research/ambi_mppi_entropy_kl_curves.json"
METHODS = {
    "prior": ("MPPI-trained | sampled prior", "#6B7280"),
    "entropy": ("MPPI-trained | entropy only | H3 J8", "#0F766E"),
    "kl": ("MPPI-trained | KL only | beta = checkpoint alpha | H3 J8", "#DB2777"),
    "both": ("MPPI-trained | entropy + KL | beta = checkpoint alpha | H3 J8", "#2563EB"),
}
KL_RULE = {"kind": "checkpoint_outer_alpha", "multiplier": 1.0, "within_solve": "fixed"}


def canonical_identity(identity, mode, outer_alpha=None):
    result = deepcopy(identity)
    if mode not in METHODS:
        raise series.SeriesError("Unknown curve objective")
    if mode in ("kl", "both"):
        settings = result["planner"]["settings"]
        beta = settings.pop("inner_outer_policy_kl_coef")
        if (not isinstance(outer_alpha, (int, float)) or not math.isfinite(outer_alpha)
                or outer_alpha <= 0 or not math.isclose(beta, outer_alpha, rel_tol=1e-7, abs_tol=1e-10)):
            raise series.SeriesError("KL beta must equal this checkpoint's saved outer alpha")
        settings["inner_outer_policy_kl_coef_rule"] = deepcopy(KL_RULE)
    return result


def prepare(profile, root=None):
    root = Path(root or profile["root"])
    root.mkdir(parents=True, exist_ok=True)
    with series._lock(root / ".curves.lock"):
        path = root / "catalog.json"
        if path.exists():
            catalog = series._read(path)
            if catalog["family"] != profile["family"]:
                raise series.SeriesError("Existing curve catalog belongs to another comparison")
            for curve in catalog["curves"]:
                series.validate_identity(series.load_run(curve["run_dir"]), profile["identities"][curve["mode"]])
            return catalog
        curves = []
        for mode, (name, color) in METHODS.items():
            registry = series.create_run(root / "runs", dict(identity=profile["identities"][mode], label=name),
                profile["family"], profile["project"], profile["entity"], profile["owner"])
            curves.append(dict(mode=mode, name=name, color=color, run_id=registry["run_id"], run_dir=registry["run_dir"]))
        catalog = {key: profile[key] for key in ("family", "entity", "project", "owner", "workspace_url")}
        catalog.update(schema_version=1, root=str(root.resolve()), curves=curves)
        series._atomic_json(root / "sources.json", profile["sources"])
        series._atomic_json(path, catalog)
        return catalog


def register_campaign(profile, campaign):
    """Explicitly append a campaign's raw sources to the same four curve IDs."""
    catalog = prepare(profile)
    root = Path(catalog["root"])
    with series._lock(root / ".curves.lock"):
        sources = series._read(root / "sources.json")
        for cell in campaign["cells"]:
            registry = series.load_run(cell["run_dir"])
            mode, step = cell["mode"], cell["step"]
            identity = canonical_identity(registry["identity"], mode, cell.get("outer_alpha"))
            if identity != profile["identities"][mode]:
                raise series.SeriesError("Campaign differs from the selected checkpoint curve recipe")
            source = dict(mode=mode, run_dir=cell["run_dir"], run_id=cell["run_id"],
                expected_checkpoint_steps=[step], expected_identity_sha256=registry["identity_sha256"],
                outer_alpha_by_step={str(step): cell["outer_alpha"]})
            existing = next((s for s in sources if s.get("run_id") == cell["run_id"]), None)
            if existing is not None:
                if any(existing.get(k) != source[k] for k in ("mode", "run_dir", "expected_identity_sha256")):
                    raise series.SeriesError("Registered source identity changed")
                if step not in existing["expected_checkpoint_steps"]:
                    existing["expected_checkpoint_steps"].append(step)
                existing.setdefault("outer_alpha_by_step", {})[str(step)] = cell["outer_alpha"]
            else:
                sources.append(source)
        series._atomic_json(root / "sources.json", sources)
    return catalog


def read_source(source):
    if "bundle" in source:
        from utils.eval_series_data import load_records
        manifest = Path(source["bundle"]) / "manifest.json"
        receipt = series._read(source["receipt"])
        if not receipt.get("validated") or receipt.get("smoke") is True:
            raise series.SeriesError("Prior baseline lacks a validated full-episode receipt")
        if (source.get("manifest_sha256") != series._file_digest(manifest)
                or source.get("receipt_sha256") != series._file_digest(source["receipt"])):
            raise series.SeriesError("Audited prior bundle or receipt changed")
        records = load_records(manifest, source_run=source["backbone"], inventory_path=source["inventory_path"])
        records = [r for r in records if r["identity"]["planner"]["type"] == "prior"]
        if len(records) != 1:
            raise series.SeriesError("Prior bundle must select exactly one baseline")
        return [(records[0], dict(source_bundle=str(manifest), source_record_sha256=series._digest(records[0])))]
    registry = series.load_run(source["run_dir"])
    if (registry["run_id"] != source["run_id"]
            or registry["identity_sha256"] != source["expected_identity_sha256"]):
        raise series.SeriesError("Raw source run identity changed")
    result = []
    journal = series._read(Path(source["run_dir"]) / "publication.json")
    for rid, entry in journal["records"].items():
        if entry["checkpoint_step"] not in source["expected_checkpoint_steps"] or entry["status"] != "published":
            continue
        path = Path(source["run_dir"]) / "records" / (rid + ".json")
        record = series.validate_record(series._read(path), registry["identity"])
        if series._record_fingerprint(record, entry["artifact_sha256"]) != entry["record_sha256"]:
            raise series.SeriesError("Published source record differs from its immutable journal")
        result.append((record, dict(source_run_id=registry["run_id"], source_record_path=str(path),
            source_record_sha256=entry["record_sha256"], source_identity_sha256=registry["identity_sha256"])))
    return result


def _episodes(record):
    episodes = sorted(record["episodes"], key=lambda ep: ep["seed"])
    if ([ep["seed"] for ep in episodes] != [101, 102, 103, 104, 105]
            or any(ep["length"] != 500 or ep.get("capped") or ep.get("truncated_by_evaluator") for ep in episodes)
            or record["metrics"].get("eval/frozen_state_unchanged") is not True):
        raise series.SeriesError("Curve points require five complete frozen, paired episodes")
    return episodes


def derive(profile, source, raw, provenance, prior, root):
    mode, step = source["mode"], raw["checkpoint"]["step"]
    alpha = source.get("outer_alpha_by_step", {}).get(str(step))
    identity = canonical_identity(raw["identity"], mode, alpha)
    if identity != profile["identities"][mode]:
        raise series.SeriesError("Backbone, source, protocol or planner differs from the curve")
    if prior["identity"] != profile["identities"]["prior"]:
        raise series.SeriesError("Paired prior backbone, protocol or scientific source differs")
    episodes, prior_episodes = _episodes(raw), _episodes(prior)
    if raw["checkpoint"] != prior["checkpoint"]:
        raise series.SeriesError("Prior and adapted checkpoint hashes differ")
    if any(a["solver_seed"] != b["solver_seed"] for a, b in zip(episodes, prior_episodes)):
        raise series.SeriesError("Prior and adapted solver seeds differ")
    returns = [ep["return"] for ep in episodes]
    gains = [ep["return"] - p["return"] for ep, p in zip(episodes, prior_episodes)]
    metrics = deepcopy(raw["metrics"])
    for key, value in {"eval/return_mean": statistics.mean(returns), "eval/return_sample_std": statistics.stdev(returns)}.items():
        if not math.isclose(metrics[key], value, rel_tol=1e-8, abs_tol=1e-8):
            raise series.SeriesError("Raw return summary disagrees with its five episodes")
    for ep, gain in zip(episodes, gains):
        if (mode != "prior" and ep.get("paired_return_delta") is not None
                and not math.isclose(ep["paired_return_delta"], gain, rel_tol=1e-8, abs_tol=1e-8)):
            raise series.SeriesError("Raw paired improvement differs from the selected sampled prior")
    if mode == "prior":
        # Old prior jobs had different instrumentation/compilation. Their
        # return semantics agree; timing and inner diagnostics are not joined.
        metrics = {k: v for k, v in metrics.items() if k.startswith("eval/")}
    metrics.update({"eval/paired_gain_mean": statistics.mean(gains),
        "eval/paired_gain_sample_std": statistics.stdev(gains), "eval/paired_episodes": 5,
        "reference/return_mean": statistics.mean(ep["return"] for ep in prior_episodes),
        "reference/return_sample_std": statistics.stdev(ep["return"] for ep in prior_episodes)})
    if mode in ("kl", "both"):
        metrics["configuration/inner_outer_policy_kl_coef"] = alpha
        metrics["configuration/checkpoint_outer_alpha"] = alpha
    reference = dict(provenance, source_record_id=raw["record_id"], source_identity=raw["identity"],
        checkpoint=raw["checkpoint"], prior_record_id=prior["record_id"],
        prior_record_sha256=series._digest(prior), coefficient_rule=KL_RULE if mode in ("kl", "both") else None)
    path = Path(root) / "provenance" / f"{mode}-{step}.json"
    if path.exists() and series._read(path) != reference:
        raise series.SeriesError("Checkpoint source changed; select a separate evaluation attempt")
    series._atomic_json(path, reference)
    keep = ("seed", "solver_seed", "length", "return", "terminated", "truncated", "capped")
    compact = [{**{k: ep[k] for k in keep if k in ep}, "paired_return_delta": gain}
               for ep, gain in zip(episodes, gains)]
    rid = series._digest({"family": profile["family"], "mode": mode, "checkpoint": raw["checkpoint"],
        "source_record_id": raw["record_id"]})
    return dict(identity=identity, checkpoint=raw["checkpoint"], metrics=metrics, episodes=compact,
        artifact_files={"source-reference.json": str(path.resolve())}, provenance=reference,
        record_id=rid, source_result_path=raw["source_result_path"], label=METHODS[mode][0])


def stage(profile, catalog, *, require_complete=False):
    root = Path(catalog["root"])
    sources = series._read(root / "sources.json")
    loaded, missing, seen = [], [], {}
    for source in sources:
        rows = read_source(source)
        found = {record["checkpoint"]["step"] for record, _ in rows}
        missing.extend((source["mode"], step) for step in source["expected_checkpoint_steps"] if step not in found)
        for record, provenance in rows:
            key = (source["mode"], record["checkpoint"]["step"])
            if key in seen:
                raise series.SeriesError("Multiple selected attempts for one method/checkpoint")
            seen[key] = record
            loaded.append((source, record, provenance))
    if require_complete and missing:
        raise series.SeriesError(f"Missing published source results: {missing}")
    counts = {mode: 0 for mode in METHODS}
    directories = {r["mode"]: r["run_dir"] for r in catalog["curves"]}
    for source, raw, provenance in sorted(loaded, key=lambda row: (row[1]["checkpoint"]["step"], row[0]["mode"])):
        prior = seen.get(("prior", raw["checkpoint"]["step"]))
        if prior is None:
            raise series.SeriesError("A matched sampled prior is required before a curve point is published")
        record = derive(profile, source, raw, provenance, prior, root)
        series.stage_record(directories[source["mode"]], record)
        counts[source["mode"]] += 1
    return dict(counts=counts, missing=missing)


def publish(profile, catalog, *, require_complete=False, wandb_module=None):
    from slurm.ambi_entropy_kl_curves_layout import verify_installed_catalog
    root = Path(catalog["root"])
    with series._lock(root / ".curve-publisher.lock", blocking=False):
        verify_installed_catalog(catalog, root / "workspace.json", wandb_module)
        result = stage(profile, catalog, require_complete=require_complete)
        for curve in catalog["curves"]:
            with series.Publisher(curve["run_dir"], owner=catalog["owner"], wandb_module=wandb_module) as publisher:
                publisher.run.name = curve["name"]
                publisher.run.config.update({"curve_family": profile["family"], "objective_mode": curve["mode"],
                    "curve_label": curve["name"], "kl_coefficient_rule": KL_RULE if curve["mode"] in ("kl", "both") else None},
                    allow_val_change=True)
                for key in {key for path in (Path(curve["run_dir"]) / "records").glob("*.json")
                            for key in series._read(path)["metrics"]}:
                    publisher.run.define_metric(key, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                pending = sum(mode == curve["mode"] for mode, _ in result["missing"])
                publisher.run.summary.update({"progress/checkpoints_complete": result["counts"][curve["mode"]],
                    "progress/checkpoints_expected": result["counts"][curve["mode"]] + pending,
                    "progress/status": "pending results" if pending else "complete", "progress/failed_cells": 0})
        series._atomic_json(root / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "stage", "publish"))
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    profile = series._read(args.profile)
    catalog = register_campaign(profile, series._read(args.campaign)) if args.campaign else prepare(profile)
    if args.command == "prepare":
        result = catalog
    elif args.command == "stage":
        result = stage(profile, catalog, require_complete=args.require_complete)
    else:
        result = publish(profile, catalog, require_complete=args.require_complete)
    print(series._json(result))


if __name__ == "__main__":
    main()
