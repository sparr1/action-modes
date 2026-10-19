"""Pinned, offline H3/J8 checkpoint evaluations; publication is a CPU task."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
STEPS = [100000, 150000, 200000, 400000]
SEEDS = [101, 102, 103, 104, 105]
SOURCES = {
    "mppi_aux": "rwgao_b-brown-university/ambi/mppi7530927x0",
    "sac_aux": "rwgao_b-brown-university/ambi/full7391933x4",
}
SOURCE_DEFINITIONS = {
    "mppi_aux": {"source_run": SOURCES["mppi_aux"], "inner_operator": "mppi",
                 "label": "MPPI-trained backbone"},
    "sac_aux": {"source_run": SOURCES["sac_aux"], "inner_operator": "sac",
                "label": "SAC-trained backbone"},
    "original_aux": {"source_run": "rwgao_b-brown-university/ambi/aux6428346x0",
                     "inner_operator": "none", "label": "Original prior-trained backbone"},
}
MATRIX = "configs/research/ambi_early_h3_j8.json"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify_source(campaign=None):
    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise RuntimeError("Evaluation requires a clean, committed source checkout")
    if campaign and (sha != campaign["source_commit"] or
                     sha256(REPO / MATRIX) != campaign["matrix_sha256"]):
        raise RuntimeError("Campaign source/configuration changed")
    if campaign and campaign.get("selection"):
        selection = campaign["selection"]
        if sha256(REPO / selection["path"]) != selection["sha256"]:
            raise RuntimeError("Campaign checkpoint selection changed")
        validate_campaign_selection(campaign)
    return sha


def verify_checkpoint(row):
    source = SOURCE_DEFINITIONS.get(row["backbone_id"])
    if source is None or row["source_run"] != source["source_run"]:
        raise ValueError("Wrong checkpoint source run")
    for path_key, hash_key in (("checkpoint", "checkpoint_sha256"),
                               ("metadata", "metadata_sha256")):
        if sha256(row[path_key]) != row[hash_key]:
            raise RuntimeError(f"Changed {path_key}: {row[path_key]}")
    metadata = read(row["metadata"])
    if metadata["checkpoint"]["step"] != row["step"]:
        raise ValueError("Checkpoint step and sidecar disagree")
    params = metadata["trial_run_params"]["alg_params"]
    if params["inner_operator"] != source["inner_operator"] or params["aux_return_mode"] != "sac":
        raise ValueError("Unexpected source backbone")


def selection_definition(selection_path=None):
    """Use explicit checkpoint pins without changing the original campaign."""
    if selection_path is None:
        return {"campaign_id": "early-h3-j8-20261018",
                "sources": {key: SOURCE_DEFINITIONS[key] for key in SOURCES},
                "steps": STEPS, "checkpoint_pins": []}, None
    path = Path(selection_path).resolve()
    relative = str(path.relative_to(REPO))
    selection = read(path)
    if selection.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint selection schema")
    sources = selection.get("sources", {})
    steps = selection.get("steps", [])
    if not sources or not isinstance(sources, dict):
        raise ValueError("Checkpoint selection requires explicit sources")
    for backbone, source in sources.items():
        known = SOURCE_DEFINITIONS.get(backbone)
        if known is None or any(source.get(key) != known[key] for key in ("source_run", "inner_operator")):
            raise ValueError("Checkpoint selection has an unknown source identity")
    if (not steps or steps != sorted(set(steps))
            or any(type(step) is not int or step <= 0 for step in steps)):
        raise ValueError("Checkpoint selection steps must be sorted unique positive integers")
    expected = {(backbone, step) for backbone in sources for step in steps}
    pins = selection.get("checkpoint_pins", [])
    if len(pins) != len(expected) or {(row["backbone_id"], row["step"]) for row in pins} != expected:
        raise ValueError("Checkpoint selection requires exactly one hash pin per selected checkpoint")
    for pin in pins:
        for key in ("checkpoint_sha256", "metadata_sha256"):
            value = pin.get(key, "")
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("Checkpoint selection contains an invalid SHA256")
    if not isinstance(selection.get("campaign_id"), str) or not selection["campaign_id"]:
        raise ValueError("Checkpoint selection requires a campaign_id")
    return selection, {"path": relative, "sha256": sha256(path)}


def select_inventory_rows(inventory, selection, *, exact_inventory=False):
    """Resolve only the requested source/step pairs, preserving inventory hashes."""
    sources, steps = selection["sources"], selection["steps"]
    expected = {(backbone, step) for backbone in sources for step in steps}
    source_ids = {source["source_run"]: backbone for backbone, source in sources.items()}
    rows = []
    for item in inventory["checkpoints"]:
        source_run = item.get("source_run", inventory.get("source_run"))
        backbone = item.get("backbone_id", source_ids.get(source_run))
        if (backbone, item["step"]) not in expected:
            continue
        if source_run != sources[backbone]["source_run"]:
            raise ValueError("Wrong checkpoint source run")
        rows.append({**item, "backbone_id": backbone, "source_run": source_run})
    if (len(rows) != len(expected) or {(row["backbone_id"], row["step"]) for row in rows} != expected
            or (exact_inventory and len(inventory["checkpoints"]) != len(expected))):
        message = ("Inventory must contain exactly the selected eight checkpoints" if exact_inventory
                   else "Inventory must contain exactly one of each selected source/checkpoint pair")
        raise ValueError(message)
    pins = {(row["backbone_id"], row["step"]): row for row in selection.get("checkpoint_pins", [])}
    for row in rows:
        pin = pins.get((row["backbone_id"], row["step"]))
        if pin and any(row[key] != pin[key] for key in ("checkpoint_sha256", "metadata_sha256")):
            raise ValueError("Inventory disagrees with selected checkpoint SHA256 pins")
    return rows


def validate_campaign_selection(campaign):
    """Bind a generated manifest back to its tracked, hash-pinned selection."""
    provenance = campaign.get("selection")
    if not provenance:
        return
    selection, actual_provenance = selection_definition(REPO / provenance["path"])
    if provenance != actual_provenance:
        raise ValueError("Campaign checkpoint selection hash changed")
    for key in ("campaign_id", "sources", "steps"):
        if campaign.get(key) != selection[key]:
            raise ValueError(f"Campaign {key} disagrees with checkpoint selection")
    if (campaign.get("matrix") != MATRIX or campaign.get("seeds") != SEEDS
            or campaign.get("controller_seed") != 55 or campaign.get("max_steps") != 500):
        raise ValueError("Campaign protocol disagrees with selected H3/J8 evaluation")
    identities = [(backbone, step, execution)
                  for backbone in selection["sources"] for step in selection["steps"]
                  for execution in ("mean", "sample")]
    cells = campaign["cells"]
    if len(cells) != len(identities):
        raise ValueError("Campaign does not contain exactly the selected cells")
    pins = {(pin["backbone_id"], pin["step"]): pin for pin in selection["checkpoint_pins"]}
    for index, (cell, identity) in enumerate(zip(cells, identities)):
        backbone, step, execution = identity
        if (cell["index"] != index or (cell["backbone_id"], cell["step"], cell["execution"]) != identity
                or cell["source_run"] != selection["sources"][backbone]["source_run"]
                or cell["selector"] != f"early_h3_j8/{execution}"):
            raise ValueError("Campaign cell identity disagrees with checkpoint selection")
        pin = pins[(backbone, step)]
        if any(cell[key] != pin[key] for key in ("checkpoint_sha256", "metadata_sha256")):
            raise ValueError("Campaign cell hash disagrees with checkpoint selection")


def prepare(inventory_path, root, *, selection_path=None):
    root = Path(root).resolve()
    sha = verify_source()
    selection, provenance = selection_definition(selection_path)
    sources = selection["sources"]
    rows = select_inventory_rows(read(inventory_path), selection, exact_inventory=selection_path is None)
    for row in rows:
        verify_checkpoint(row)
    campaign = dict(schema="ambi-early-h3-j8-v1", campaign_id=selection["campaign_id"],
                    source_commit=sha, source_tree=git("rev-parse", "HEAD^{tree}"),
                    matrix=MATRIX, matrix_sha256=sha256(REPO / MATRIX),
                    sources=sources, steps=selection["steps"], selection=provenance,
                    seeds=SEEDS, controller_seed=55, max_steps=500, cells=[])
    if selection.get("include_paired_baselines"):
        campaign["baseline_root"] = str(root / "baselines")
    for backbone in sources:
        inventory = root / "inputs" / f"{backbone}.json"
        selected = sorted((r for r in rows if r["backbone_id"] == backbone), key=lambda r: r["step"])
        write_new(inventory, {"source_run": sources[backbone]["source_run"], "checkpoints": [
            {**row, "path": row["checkpoint"], "sha256": row["checkpoint_sha256"]} for row in selected]})
        for row in selected:
            for execution in ("mean", "sample"):
                index = len(campaign["cells"])
                campaign["cells"].append({**row, "index": index, "execution": execution,
                    "selector": f"early_h3_j8/{execution}", "inventory_path": str(inventory),
                    "output_dir": str(root / "production" / f"cell-{index:02d}")})
    validate_campaign_selection(campaign)
    write_new(root / "campaign.json", campaign)
    return campaign


def evaluator_command(campaign, cell, output, *, smoke_steps=None, eval_run_dir=None):
    seeds = [101] if smoke_steps is not None else campaign["seeds"]
    max_steps = smoke_steps if smoke_steps is not None else campaign["max_steps"]
    command = [sys.executable, str(REPO / "evaluate_ambi_checkpoint.py"),
               "--matrix", str(REPO / campaign["matrix"]),
               "--checkpoint", cell["checkpoint"], "--metadata", cell["metadata"],
               "--preset", cell["selector"], "--device", "cuda",
               "--seeds", *map(str, seeds), "--controller-seed", str(campaign["controller_seed"]),
               "--max-steps", str(max_steps), "--output", str(output / "results.json"),
               "--bundle-dir", str(output / "bundle"),
               "--checkpoint-inventory", cell["inventory_path"]]
    if eval_run_dir:
        command += ["--eval-run-dir", str(eval_run_dir)]
    return command, seeds, max_steps


def run_cell(campaign_path, index, smoke_steps=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    verify_source(campaign)
    cell = campaign["cells"][index]
    assert cell["index"] == index
    verify_checkpoint(cell)
    cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    output = (campaign_path.parent / "smoke" / f"cell-{index:02d}-cpu{cpu}-steps{smoke_steps}"
              if smoke_steps is not None else Path(cell["output_dir"]))
    if output.exists():
        raise FileExistsError(f"Refusing to repeat/overwrite {output}")
    output.mkdir(parents=True)
    if smoke_steps is None:
        publication = read(campaign_path.parent / "publication" / "publication-map.json")
        # Registry allocation must precede compute, but only the CPU publisher
        # stages records after receipt validation. The evaluator's auto-stage
        # path would expose a complete bundle before this validation finishes.
        if not publication["cell_run_dirs"].get(str(index)):
            raise RuntimeError("No publication identity allocated for cell")
    command, seeds, max_steps = evaluator_command(campaign, cell, output,
                                                 smoke_steps=smoke_steps)
    write_new(output / "started.json", {"cell": cell, "command": command,
                                        "job_id": os.environ.get("SLURM_JOB_ID"), "cpus": cpu})
    started = time.perf_counter()
    subprocess.run(command, cwd=REPO, check=True)
    from slurm.ambi_early_h3_j8_validation import validate_result
    validation = validate_result(output / "results.json", output / "bundle",
                                 expected_seeds=seeds, max_steps=max_steps)
    receipt = {"cell_index": index, "smoke": smoke_steps is not None, "validated": True,
               "elapsed_seconds": time.perf_counter() - started,
               "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
               "cpus": cpu, "validation": validation,
               "results_sha256": sha256(output / "results.json"),
               "manifest_sha256": sha256(output / "bundle" / "manifest.json")}
    write_new(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--selection", type=Path,
                   help="Tracked source/checkpoint selection; omission preserves the original two-backbone campaign")
    p = sub.add_parser("run")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--index", required=True, type=int)
    p.add_argument("--smoke-steps", type=int, choices=(32, 500))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.inventory, args.root, selection_path=args.selection)
        print(json.dumps({"cells": len(result["cells"]), "root": str(args.root)}))
    else:
        print(json.dumps(run_cell(args.campaign, args.index, args.smoke_steps), allow_nan=False))


if __name__ == "__main__":
    main()
