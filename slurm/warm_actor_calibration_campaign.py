"""Immutable task inventory for the frozen575k warm actor calibration panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKPOINT_SHA = "0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042"
CELLS = ((3, 6), (3, 8), (3, 10), (1, 8))
SEEDS = tuple(range(101, 106))
DECISIONS = (25, 75, 150, 250, 350, 450)
REPLAN_DECISIONS = (75, 350)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(root, source_campaign, *, smoke=False):
    root, source_campaign = Path(root).resolve(), Path(source_campaign).resolve()
    if (root / "campaign.json").exists():
        raise FileExistsError("Use a new campaign directory.")
    source = read(source_campaign)
    if source["checkpoint_sha256"] != CHECKPOINT_SHA or source["study_protocol"] != "actor-transfer-v2":
        raise ValueError("Expected the corrected frozen575k actor-transfer-v2 source.")
    selected = ((3, 10), (1, 8)) if smoke else CELLS
    seeds = (101,) if smoke else SEEDS
    decisions = (1, 3) if smoke else DECISIONS
    replan_decisions = (1,) if smoke else REPLAN_DECISIONS
    cells, captures, prefixes, replans = [], [], [], []
    for h, j in selected:
        name = f"actor_warm_h{h}_j{j}_c16"
        original, = [cell for cell in source["cells"] if cell["name"] == name]
        cells.append(dict(name=name, source_cell=name, H=h, J=j,
            selector=original["selector"], checkpoint=original["checkpoint"],
            checkpoint_sha256=CHECKPOINT_SHA, metadata_sha256=original["metadata_sha256"],
            expected_config=original["expected_config"]))
        for seed in seeds:
            capture_index = len(captures)
            directory = root / "captures" / name / f"seed-{seed}"
            common = dict(source_cell=name, H=h, J=j, seed=seed,
                selector=original["selector"], checkpoint=original["checkpoint"],
                capture_index=capture_index, capture_directory=str(directory))
            captures.append(dict(common, index=capture_index, decisions=list(decisions),
                                 max_steps=4 if smoke else 500))
            for decision in decisions:
                prefixes.append(dict(common, index=len(prefixes), decision=decision,
                    root_file=str(directory / f"decision-{decision}" / "root.json"),
                    output=str(root / "prefix" / name / f"seed-{seed}" / f"decision-{decision}.json")))
            for decision in replan_decisions:
                replans.append(dict(common, index=len(replans), decision=decision,
                    root_file=str(directory / f"decision-{decision}" / "root.json"),
                    output=str(root / "replan" / name / f"seed-{seed}" / f"decision-{decision}.json")))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    campaign = dict(schema_version=1, protocol="warm-actor-branch-calibration-v1", smoke=smoke,
        name="575k warm actors | matched-state value calibration" + (" | smoke" if smoke else ""),
        group="warm-actor-calibration-575k-20260926", checkpoint_sha256=CHECKPOINT_SHA,
        source_commit=commit, source_directory=str(ROOT),
        source_campaign=str(source_campaign), source_campaign_sha256=hashlib.sha256(source_campaign.read_bytes()).hexdigest(),
        matrix=str(ROOT / "configs/research/ambi_actor_transfer_575k.json"),
        cells=cells, captures=captures, prefixes=prefixes, replans=replans,
        expected_capture_shards=len(captures), expected_prefix_shards=len(prefixes),
        expected_replan_shards=len(replans),
        result_globs=["prefix/*/seed-*/decision-*.json", "replan/*/seed-*/decision-*.json"],
        rollouts=2 if smoke else 32, tail_steps=16 if smoke else 1000,
        controller_seed=55, seeds=list(seeds), decisions=list(decisions),
        replan_decisions=list(replan_decisions),
        overview_run_id=uuid.uuid4().hex,
        wandb_entity="rwgao_b-brown-university",
        wandb_project="ambi-inner-bench-validation" if smoke else "ambi-inner-bench",
        wandb_view_name="nw-nwuserrwgao_b-w")
    write(root / "campaign.json", campaign)
    print(json.dumps({key: campaign[key] for key in ("overview_run_id", "expected_capture_shards", "expected_prefix_shards", "expected_replan_shards")}))
    return campaign


def command_for(campaign, kind, index):
    collection = dict(capture="captures", prefix="prefixes", replan="replans")[kind]
    task = campaign[collection][index]
    shared = ["--checkpoint", task["checkpoint"], "--matrix", campaign["matrix"], "--device", "cuda"]
    if kind == "capture":
        return [sys.executable, str(ROOT / "capture_warm_actors.py"), "capture", *shared,
            "--selector", task["selector"], "--seed", str(task["seed"]),
            "--output", task["capture_directory"], "--max-steps", str(task["max_steps"]),
            "--decisions", *map(str, task["decisions"])]
    if kind == "prefix":
        return [sys.executable, str(ROOT / "evaluate_warm_actor_calibration.py"), *shared,
            "--root", task["root_file"], "--output", task["output"],
            "--rollouts", str(campaign["rollouts"]), "--tail-steps", str(campaign["tail_steps"])]
    return [sys.executable, str(ROOT / "capture_warm_actors.py"), "replan", *shared,
        "--root", task["root_file"], "--output", task["output"]]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source_config(actual, expected):
    """Ignore only placement; source compilation and all science stay pinned."""
    actual = {key: value for key, value in actual.items() if key != "device"}
    expected = {key: value for key, value in expected.items() if key != "device"}
    if actual != expected:
        differences = {key: {"expected": expected.get(key), "actual": actual.get(key)}
                       for key in actual.keys() | expected.keys()
                       if key not in actual or key not in expected or actual[key] != expected[key]}
        raise ValueError(f"Historical source configuration differs: {differences}")


def verify_source_provenance(campaign, task):
    """Validate bytes and the actual config resolver before any source solve."""
    from utils.checkpoint_context import load_checkpoint_context
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.eval_series_data import resolved_checkpoint_config

    cell, = [cell for cell in campaign["cells"] if cell["source_cell"] == task["source_cell"]]
    if task["selector"] != cell["selector"] or task["checkpoint"] != cell["checkpoint"]:
        raise ValueError("Capture task differs from its historical source cell.")
    checkpoint_hash = file_sha256(task["checkpoint"])
    if checkpoint_hash != cell["checkpoint_sha256"] or checkpoint_hash != campaign["checkpoint_sha256"]:
        raise ValueError("Actual checkpoint checksum differs from the historical source.")
    context = load_checkpoint_context(task["checkpoint"])
    if file_sha256(context.source) != cell["metadata_sha256"]:
        raise ValueError("Actual checkpoint metadata checksum differs from the historical source.")
    matrix = load_preset_matrix(campaign["matrix"])
    resolved = resolve_preset(campaign["matrix"], task["selector"], matrix,
                              checkpoint_context=context)
    actual = resolved_checkpoint_config({"metadata": context.metadata}, resolved)
    check_source_config(actual, cell["expected_config"])
    return {"checkpoint_sha256": checkpoint_hash, "metadata_sha256": cell["metadata_sha256"],
            "matrix_sha256": file_sha256(campaign["matrix"]), "config_verified": True}


def verify_captured_source(campaign, task):
    """Reject incomplete or changed source configuration before branch work."""
    cell, = [cell for cell in campaign["cells"] if cell["source_cell"] == task["source_cell"]]
    manifest = read(Path(task["capture_directory"]) / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Source capture is not complete.")
    for key, expected in (("checkpoint_sha256", cell["checkpoint_sha256"]),
                          ("selector", cell["selector"]), ("seed", task["seed"]),
                          ("controller_seed", campaign["controller_seed"]),
                          ("matrix_sha256", file_sha256(campaign["matrix"]))):
        if manifest.get(key) != expected:
            raise ValueError(f"Captured source identity mismatch: {key}")
    check_source_config(manifest["resolved_config"], cell["expected_config"])


def worker(root, kind, index):
    campaign = read(Path(root) / "campaign.json")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != campaign["source_commit"]:
        raise ValueError("Worker source commit differs from the campaign.")
    if kind == "smoke":
        if not campaign["smoke"]:
            raise ValueError("Smoke dispatch requires an explicitly marked smoke campaign.")
        worker(root, "capture", index)
        for collection, branch in (("prefixes", "prefix"), ("replans", "replan")):
            for task in campaign[collection]:
                if task["capture_index"] == index:
                    worker(root, branch, task["index"])
        return
    collection = dict(capture="captures", prefix="prefixes", replan="replans")[kind]
    task = campaign[collection][index]
    if kind == "capture":
        proof = verify_source_provenance(campaign, task)
        print(json.dumps(dict(event="source_provenance_verified", **proof)), flush=True)
    else:
        verify_captured_source(campaign, task)
    command = command_for(campaign, kind, index)
    print(json.dumps(dict(kind=kind, index=index, argv=command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if kind == "capture":
        verify_captured_source(campaign, task)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", required=True)
    prep.add_argument("--source-campaign", required=True)
    prep.add_argument("--smoke", action="store_true")
    run = sub.add_parser("worker")
    run.add_argument("--root", required=True)
    run.add_argument("--kind", choices=("capture", "prefix", "replan", "smoke"), required=True)
    run.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        prepare(args.root, args.source_campaign, smoke=args.smoke)
    else:
        worker(args.root, args.kind, args.index)


if __name__ == "__main__":
    main()
