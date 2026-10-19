"""Measure frozen actor references for the existing early H3/J8 campaign."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm.ambi_early_h3_j8_campaign import read, sha256, verify_checkpoint, verify_source, write_new
from slurm.ambi_early_h3_j8_publish import load_campaign
from utils.ambi_benchmark import protocol_for, solver_seed

MATRIX = "configs/research/ambi_early_prior_baselines.json"


def validate_result(result, manifest, cell, campaign, *, historical_mean=False,
                    expected_protocol=None):
    if result["checkpoint_sha256"] != cell["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint does not match adapted checkpoint")
    rows = result["results"]
    if len(rows) != 1:
        raise ValueError("Reference cell must contain exactly one execution mode")
    row = rows[0]
    if historical_mean:
        if (cell["execution"] != "mean" or row["selector"] != "reference/prior"
                or row.get("execution_action", "mean") != "mean"
                or result.get("deterministic_execution") is not True):
            raise ValueError("Historical reuse requires the original deterministic prior")
    elif row["selector"] != "early_prior/" + cell["execution"] or row["execution_action"] != cell["execution"]:
        raise ValueError("Reference execution mode does not match adapted execution")
    if row["resolved_config"]["inner_operator"] != "none":
        raise ValueError("Reference must execute the unadapted actor")
    if not row["outer_state_unchanged"] or row["outer_updates_before"] != row["outer_updates_after"]:
        raise ValueError("Reference altered outer state")
    episodes = row["episodes"]
    if len(episodes) != 5 or sorted(ep["seed"] for ep in episodes) != campaign["seeds"]:
        raise ValueError("Reference requires all five paired episode seeds")
    for episode in episodes:
        if episode["solver_seed"] != solver_seed(campaign["controller_seed"], "episode", episode["seed"]):
            raise ValueError("Reference controller seed differs")
        if episode["length"] != 500 or not episode["truncated"] or episode["truncated_by_evaluator"]:
            raise ValueError("Reference requires complete environment-limited episodes")
        for metric in ("inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
                       "inner_temperature_optimizer_steps", "inner_model_steps"):
            if episode["model_metrics"][metric] != 0 or row["model_metrics"][metric]["max"] != 0:
                raise ValueError("Reference performed inner learning or imagination")
    if manifest["status"] != "complete" or manifest["protocol"]["action_rule"] != "tanh_" + cell["execution"]:
        raise ValueError("Reference bundle is incomplete or has a different action protocol")
    if expected_protocol is not None and manifest["protocol"] != expected_protocol:
        raise ValueError("Reference environment/action/seed protocol differs from adapted evaluation")
    return {"passed": True, "seeds": campaign["seeds"], "execution": cell["execution"],
            "episodes": len(episodes), "mean_return": row["return"]["mean"],
            "outer_state_unchanged": True, "inner_updates": 0}


def matched_protocol(cell, campaign):
    from utils.checkpoint_context import load_checkpoint_context
    from utils.ambi_research import resolve_preset
    context = load_checkpoint_context(cell["checkpoint"], metadata_path=cell["metadata"])
    resolved = resolve_preset(REPO / campaign["matrix"], cell["selector"],
                              checkpoint_context=context)
    return protocol_for(resolved, campaign["controller_seed"], campaign["max_steps"])


def reuse_mean(campaign_path, index, output_root, reference_dir=None):
    """Import a validated historical mean prior without rewriting its science."""
    campaign = load_campaign(campaign_path)
    cell = campaign["cells"][index]
    if cell["index"] != index or cell["execution"] != "mean":
        raise ValueError("Historical prior reuse applies only to a matching mean cell")
    importer_commit = verify_source()
    verify_checkpoint(cell)
    declared = cell.get("prior_reference")
    source = Path(reference_dir or declared or "").resolve()
    if reference_dir is None and not declared:
        raise ValueError("A historical prior reference path is required")
    if declared and source != Path(declared).resolve():
        raise ValueError("Historical reference path differs from the campaign")
    bundle = source if (source / "manifest.json").is_file() else source / "bundle"
    result_path = bundle.parent / "results.json"
    manifest_path = bundle / "manifest.json"
    result, manifest = read(result_path), read(manifest_path)
    pin = cell.get("prior_pin", {})
    observed = {"manifest_sha256": sha256(manifest_path),
                "results_sha256": sha256(result_path),
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "metadata_sha256": cell["metadata_sha256"],
                "checkpoint_step": cell["step"], "source_run": cell["source_run"],
                "selector": "reference/prior"}
    for key, value in observed.items():
        if key in pin and pin[key] != value:
            raise ValueError(f"Historical reference pin mismatch: {key}")
    checkpoint = manifest["checkpoint"]
    if (checkpoint["sha256"] != cell["checkpoint_sha256"]
            or checkpoint.get("source_run") != cell["source_run"]
            or checkpoint.get("metadata") != read(cell["metadata"])):
        raise ValueError("Historical checkpoint/source/sidecar does not match the campaign")
    runs = manifest["runs"]
    if (len(runs) != 1 or runs[0]["status"] != "complete"
            or runs[0]["selector"] != "reference/prior"
            or runs[0]["config"]["alg_params"]["inner_operator"] != "none"):
        raise ValueError("Historical bundle must contain exactly the completed unadapted prior")
    row = result["results"][0]
    if (runs[0]["result"]["episodes"] != row["episodes"]
            or runs[0]["result"]["resolved_config"] != row["resolved_config"]):
        raise ValueError("Historical manifest and result disagree")
    returns = [episode["return"] for episode in row["episodes"]]
    if (not all(math.isfinite(value) for value in returns)
            or not math.isclose(sum(returns) / len(returns), row["return"]["mean"],
                                rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Historical returns are invalid or inconsistent")
    validation = validate_result(result, manifest, cell, campaign, historical_mean=True,
                                 expected_protocol=matched_protocol(cell, campaign))
    source_code = manifest["code"]
    if source_code.get("dirty") or not source_code.get("commit"):
        raise ValueError("Historical reference needs its original clean source identity")
    files = {"results.json": result_path}
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError("Historical bundle must not contain symlinks")
        if path.is_file():
            files[str(Path("bundle") / path.relative_to(bundle))] = path
    digests = {name: sha256(path) for name, path in files.items()}
    for relative in runs[0].get("trace_files", []):
        if str(Path("bundle") / relative) not in files:
            raise ValueError("Historical reference is missing a declared trace")
    output = Path(output_root).resolve() / f"cell-{index:02d}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    for relative, path in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(target) != digests[relative]:
            raise ValueError("Historical reference changed while copying")
    receipt = {"validated": True, "smoke": False, "reused": True,
        "cell_index": index, "validation": validation,
        "source_commit": source_code["commit"], "importer_source_commit": importer_commit,
        "adapted_source_commit": campaign["source_commit"],
        "source_run": cell["source_run"], "checkpoint_sha256": cell["checkpoint_sha256"],
        "metadata_sha256": cell["metadata_sha256"], "matrix_sha256": result["matrix_sha256"],
        "results_sha256": observed["results_sha256"],
        "manifest_sha256": observed["manifest_sha256"],
        "reused_from": {"path": str(bundle), "source_code": source_code,
                        "selector": row["selector"], "pins": observed, "files": digests}}
    write_new(output / "receipt.json", receipt)
    return receipt


def run(campaign_path, index, output_root):
    campaign = load_campaign(campaign_path)
    cell = campaign["cells"][index]
    source_commit = verify_source()
    verify_checkpoint(cell)
    output = Path(output_root).resolve() / f"cell-{index:02d}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    command = [sys.executable, str(REPO / "evaluate_ambi_checkpoint.py"),
        "--matrix", str(REPO / MATRIX), "--checkpoint", cell["checkpoint"],
        "--metadata", cell["metadata"], "--preset", "early_prior/" + cell["execution"],
        "--device", "cuda", "--seeds", *map(str, campaign["seeds"]),
        "--controller-seed", str(campaign["controller_seed"]), "--max-steps", "500",
        "--output", str(output / "results.json"), "--bundle-dir", str(output / "bundle"),
        "--checkpoint-inventory", cell["inventory_path"]]
    write_new(output / "started.json", {"cell": cell, "source_commit": source_commit,
        "adapted_source_commit": campaign["source_commit"], "command": command})
    started = time.perf_counter()
    subprocess.run(command, cwd=REPO, check=True)
    result = read(output / "results.json")
    manifest = read(output / "bundle" / "manifest.json")
    validation = validate_result(result, manifest, cell, campaign,
                                 expected_protocol=matched_protocol(cell, campaign))
    receipt = {"validated": True, "smoke": False, "cell_index": index, "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "source_commit": source_commit, "source_run": cell["source_run"],
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "metadata_sha256": cell["metadata_sha256"], "matrix_sha256": sha256(REPO / MATRIX),
        "results_sha256": sha256(output / "results.json"),
        "manifest_sha256": sha256(output / "bundle" / "manifest.json")}
    write_new(output / "receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reuse-reference", type=Path,
                        help="Copy the pinned historical mean reference without rerunning it")
    args = parser.parse_args()
    if args.reuse_reference is not None:
        print(json.dumps(reuse_mean(args.campaign, args.index, args.output_root,
                                   args.reuse_reference)))
    else:
        run(args.campaign, args.index, args.output_root)
