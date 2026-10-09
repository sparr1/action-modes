"""Pin, validate and execute checkpoint-by-setting transfer curve cells on Oscar.

Inventory JSON has schema_version=1, source_run, and checkpoints containing
step, checkpoint, checkpoint_sha256, metadata_sha256, optional metadata path,
and optional prior_reference. The inventory must include the audited 575k
anchor; all sidecars must describe its same training run configuration.
The evaluator and transfer interventions remain unchanged. New per-checkpoint
configuration copies only replace the exact checkpoint identity contract.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ambi_benchmark import atomic_json, canonical_hash, read_json, solver_seed

PROTOCOL = "inner-sac-transfer-checkpoint-curves-v1"
SOURCE_RUN = "rwgao_b-brown-university/ambi/aux6428346x0"
ANCHOR_SHA = "0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042"
ANCHOR_METADATA_SHA = "8acc74b7ad4993050a5cc0d3c4c0860441fceb48f36a79540f5e1943e6b3e1d2"
J6_SELECTION = "h1-j6-post500k-v1"
J6_CHECKPOINT_RANGE = dict(start=525000, stop=2000000, step=25000)
J6_SETTINGS = {"h1_j6_fresh", "h1_j6_bernoulli_a0_c05", "h1_j6_matrix_blend05_actor"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    atomic_json(Path(path), value, overwrite=False)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def science_identity():
    from evaluate_ambi_transfer_campaign import source_identity
    identity = source_identity()
    return {key: identity[key] for key in ("files", "sha256")}


def source(expected_sha=None):
    require(not git("status", "--porcelain", "--untracked-files=normal"), "Campaign requires a clean tested checkout.")
    head = git("rev-parse", "HEAD")
    if expected_sha is not None:
        require(head == expected_sha, "Checkout differs from the explicitly expected commit.")
    return dict(source_commit=head, source_tree=git("rev-parse", "HEAD^{tree}"), source_dir=str(ROOT),
                scientific_source=science_identity())


def validate_configuration(config, selected=None):
    require(config.get("schema_version") == 1 and config.get("protocol") == PROTOCOL, "Unsupported curve configuration.")
    fixed = dict(source_run=SOURCE_RUN, seeds=[101, 102, 103, 104, 105], controller_seed=55, max_steps=500,
                 critic_updates=16, actor_updates=4, rollouts=128, batch_size=256,
                 smoke_seeds=[101, 102], smoke_steps=3, gpu_hardware="L40S")
    require(all(config.get(key) == value for key, value in fixed.items()), "Configuration changed the authorized evaluation protocol.")
    candidates = config.get("candidates", [])
    require(candidates and len({row["setting_id"] for row in candidates}) == len(candidates), "Candidates must be unique and nonempty.")
    selection = config.get("selection_id")
    require(selection in (None, J6_SELECTION), "Unknown versioned curve selection.")
    if selection == J6_SELECTION:
        require(config.get("checkpoint_range") == J6_CHECKPOINT_RANGE,
                "J6 checkpoint range must be 525k through 2M inclusive at 25k intervals.")
        require({row["setting_id"] for row in candidates} == J6_SETTINGS,
                "J6 selection requires fresh SAC, critic Bernoulli50%, and actor matrix shrink50%.")
        require(selected is None or (len(selected) == len(J6_SETTINGS) and set(selected) == J6_SETTINGS),
                "J6 selection must retain all three matched settings.")
    else:
        require(config.get("checkpoint_range") is None, "Checkpoint range requires an explicit versioned selection.")
    lookup = {row["setting_id"]: row for row in candidates}
    for row in candidates:
        require(row["setting_id"] == f"h{row['H']}_j{row['J']}_{row['arm']}", "Candidate identity mismatch.")
        require(row["H"] == 1 and row["J"] in ((6,) if selection == J6_SELECTION else (2, 4)),
                "Shortlist budget differs.")
        require(row.get("role") == ("fresh" if row["arm"] == "fresh" else "transfer"), "Candidate role differs.")
        if row["role"] == "transfer":
            control = lookup.get(row.get("fresh_setting_id"), {})
            require(control.get("role") == "fresh" and (control.get("H"), control.get("J")) == (row["H"], row["J"]),
                    "Transfer candidate requires its matching fresh SAC control.")
    if selected is None:
        return candidates
    require(selected and len(set(selected)) == len(selected), "Selected settings must be unique and nonempty.")
    require(set(selected) <= {row["setting_id"] for row in candidates}, "Unknown selected setting.")
    return [row for row in candidates if row["setting_id"] in selected]


def select_checkpoints(checkpoints, config):
    """Apply an explicit inclusive grid without silently accepting missing checkpoints."""
    interval = config.get("checkpoint_range")
    if interval is None:
        return checkpoints
    require(isinstance(interval, dict) and set(interval) == {"start", "stop", "step"}
            and all(type(value) is int for value in interval.values()), "Malformed checkpoint range.")
    start, stop, step = (interval[key] for key in ("start", "stop", "step"))
    require(start >= 0 and stop >= start and step > 0 and (stop - start) % step == 0,
            "Invalid inclusive checkpoint grid.")
    expected = list(range(start, stop + 1, step))
    require(575000 in expected, "Selected checkpoint grid must include the audited 575k anchor.")
    wanted = set(expected)
    selected = [row for row in checkpoints if row["step"] in wanted]
    require([row["step"] for row in selected] == expected,
            "Inventory does not contain every checkpoint in the requested range.")
    return selected


def validate_inventory(inventory):
    require(inventory.get("schema_version") == 1 and inventory.get("source_run") == SOURCE_RUN,
            "Inventory must identify the audited aux6428346x0 backbone.")
    records = inventory.get("checkpoints", [])
    require(records and len({row["step"] for row in records}) == len(records), "Checkpoint steps must be unique and nonempty.")
    rows, identities = [], set()
    for record in sorted(records, key=lambda row: row["step"]):
        require(type(record["step"]) is int and record["step"] >= 0, "Invalid checkpoint step.")
        row = deepcopy(record)
        row["checkpoint"] = str(Path(row["checkpoint"]).resolve())
        row["metadata"] = str(Path(row.get("metadata", row["checkpoint"] + ".metadata.json")).resolve())
        require(digest(row["checkpoint"]) == row["checkpoint_sha256"], "Checkpoint hash mismatch.")
        require(digest(row["metadata"]) == row["metadata_sha256"], "Checkpoint metadata hash mismatch.")
        metadata = read_json(row["metadata"])
        require(metadata.get("checkpoint", {}).get("step") == row["step"], "Sidecar step differs from inventory.")
        require(metadata.get("schema_version") == 1 and isinstance(metadata.get("trial_run_params"), dict)
                and isinstance(metadata.get("experiment_params"), dict), "Malformed checkpoint sidecar.")
        identity = canonical_hash({key: metadata[key] for key in ("trial_run_params", "experiment_params")})
        identities.add(identity)
        row["backbone_config_sha256"] = identity
        rows.append(row)
    anchors = [row for row in rows if row["step"] == 575000]
    require(len(anchors) == 1 and anchors[0]["checkpoint_sha256"] == ANCHOR_SHA
            and anchors[0]["metadata_sha256"] == ANCHOR_METADATA_SHA, "Inventory lacks the audited 575k anchor.")
    require(len(identities) == 1, "Checkpoint sidecars describe different backbone configurations.")
    require(len({row["checkpoint"] for row in rows}) == len(rows)
            and len({row["checkpoint_sha256"] for row in rows}) == len(rows), "Repeated checkpoint files/weights in inventory.")
    return rows


def prepare(args):
    require(not args.root.exists(), f"Refusing to overwrite {args.root}")
    current = source(args.expected_source_sha)
    config = read_json(args.config)
    candidates = deepcopy(validate_configuration(config, args.settings))
    checkpoints = select_checkpoints(validate_inventory(read_json(args.inventory)), config)
    require(not getattr(args, "reuse_575k_root", None),
            "This five-seed shortlist does not reuse historical transfer results.")
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context
    from utils.transfer_campaign import load_campaign, resolved_cell
    template_path = args.config.resolve().parent / config["discovery_template"]
    template = load_campaign(template_path)
    from utils.transfer_campaign_diagnostics import diagnostic_settings
    template["diagnostics"] = diagnostic_settings(template.get("diagnostics"))
    require(template["diagnostics"] is not None, "Sampled transfer diagnostics must remain enabled.")
    require(template["seeds"] == config["seeds"] and template["controller_seed"] == config["controller_seed"]
            and all(template[key] == config[key] for key in ("critic_updates", "actor_updates", "rollouts", "batch_size", "max_steps")),
            "Shortlist template and curve evaluation protocol disagree.")
    base_path = Path(template.pop("base_matrix_path"))
    base_template = load_preset_matrix(base_path)
    for row in candidates:
        require(row["H"] in template["horizons"] and row["J"] in template["rounds"] and row["arm"] in template["arms"],
                "Candidate is not supported by the unchanged discovery evaluator.")
        row["arm_definition"] = deepcopy(template["arms"][row["arm"]])
    campaign = dict(schema_version=1, protocol=PROTOCOL, **current, source_run=SOURCE_RUN,
        inventory_path=str(args.inventory.resolve()), inventory_sha256=digest(args.inventory),
        configuration_path=str(args.config.resolve()), configuration_sha256=digest(args.config),
        discovery_template_sha256=digest(template_path), base_template_sha256=digest(base_path),
        checkpoints=checkpoints, candidates=candidates, cells=[],
        **{key: config[key] for key in ("seeds", "controller_seed", "max_steps", "smoke_seeds", "smoke_steps", "gpu_hardware")},
        selection="Exploratory 575k-selected candidates; checkpoints are repeated measurements of one trained backbone.",
        prior_reference=read_json(args.inventory).get("prior_reference"),
        diagnostics=deepcopy(template["diagnostics"]))
    if config.get("selection_id") is not None:
        campaign.update(selection_id=config["selection_id"], checkpoint_range=deepcopy(config["checkpoint_range"]),
            selection="Post-500k H1 J6 matched-budget comparison; checkpoints are repeated measurements of one trained backbone.")
    generated = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        step = checkpoint["step"]
        directory = args.root.resolve() / "configs" / f"step-{step}"
        matrix = deepcopy(base_template)
        matrix["checkpoint_steps"] = [step]
        matrix["checkpoint_contract"] = dict(step=step, sha256=checkpoint["checkpoint_sha256"])
        recipe = deepcopy(template)
        recipe.update(base_matrix="base-matrix.json", checkpoint_contract=matrix["checkpoint_contract"])
        recipe_path = directory / "discovery.json"
        generated.extend(((directory / "base-matrix.json", matrix), (recipe_path, recipe)))
        context = load_checkpoint_context(checkpoint["checkpoint"], metadata_path=checkpoint["metadata"])
        base = resolve_preset(base_path, recipe["base_preset"], matrix=matrix, checkpoint_context=context)
        for candidate in candidates:
            resolved = resolved_cell(base, recipe, horizon=candidate["H"], rounds=candidate["J"], arm=candidate["arm"])
            name = f"step-{step}/{candidate['setting_id']}"
            campaign["cells"].append(dict(index=len(campaign["cells"]), name=name, step=step,
                checkpoint_index=checkpoint_index, **candidate, config_path=str(recipe_path),
                resolved_sha256=canonical_hash(resolved), result_dir=str(args.root.resolve() / "settings" / name)))
    smoke_steps = {checkpoints[0]["step"], 575000, checkpoints[-1]["step"]}
    campaign["smoke_indices"] = [row["index"] for row in campaign["cells"] if row["step"] in smoke_steps]
    campaign["smoke_checkpoint_steps"] = sorted(smoke_steps)
    campaign["reused_indices"] = []
    campaign["production_indices"] = [cell["index"] for cell in campaign["cells"]]
    args.root.mkdir(parents=True, exist_ok=False)
    for path, value in generated:
        write(path, value)
    campaign["generated_config_sha256"] = {str(path): digest(path) for path, _ in generated}
    write(args.root / "campaign.json", campaign)
    print(json.dumps(dict(campaign=str(args.root / "campaign.json"), checkpoints=len(checkpoints),
        cells=len(campaign["cells"]), production_indices=campaign["production_indices"],
        smoke_indices=campaign["smoke_indices"], reused_indices=campaign["reused_indices"])), flush=True)
    return campaign


def validate_result(directory, campaign, cell, *, smoke=False, allow_historical=False):
    require(not allow_historical and not cell.get("historical_reuse"),
            "Historical transfer reuse is not authorized for the five-seed shortlist.")
    directory = Path(directory)
    result, manifest = read_json(directory / "results.json"), read_json(directory / "manifest.json")
    checkpoint = campaign["checkpoints"][cell["checkpoint_index"]]
    seeds, limit = (campaign["smoke_seeds"], campaign["smoke_steps"]) if smoke else (campaign["seeds"], campaign["max_steps"])
    expected = dict(protocol="inner-sac-transfer-discovery-v1", cell_id=cell["setting_id"], arm=cell["arm"],
                    horizon=cell["H"], rounds=cell["J"], checkpoint_sha256=checkpoint["checkpoint_sha256"], seeds=seeds, smoke=smoke)
    require(all(result.get(key) == value and manifest.get(key) == value for key, value in expected.items()),
            "Result/manifest differs from assigned checkpoint and setting.")
    require(result.get("complete") is True and result.get("frozen_outer_verified") is True,
            "Result lacks completion/frozen-state verification.")
    require(manifest.get("checkpoint_step") == cell["step"] and manifest.get("controller_seed") == 55
            and manifest.get("max_steps") == limit and manifest.get("compile") is True,
            "Result evaluation budget/compile protocol differs.")
    require(manifest.get("arm_definition") == cell["arm_definition"]
            and canonical_hash(manifest.get("resolved")) == cell["resolved_sha256"], "Resolved controller semantics differ.")
    identity = manifest.get("source", {})
    expected_science = campaign["scientific_source"]
    require(result.get("source") == identity and all(identity.get(key) == expected_science[key]
            for key in ("sha256", "files")), "Result scientific fingerprint differs.")
    require(identity.get("git_head") == campaign["source_commit"], "Result source commit differs.")
    require(manifest.get("metric_policy") == "all_scalars" and manifest.get("metric_coverage") == dict(
        policy="all_scalars", computed_scalars_only=True, extra_solver_probes=False, per_update_traces=False),
        "New curve output must retain every already-computed scalar measurement.")
    require(manifest.get("diagnostics") == campaign["diagnostics"], "Diagnostic settings differ.")
    require(manifest.get("campaign_sha256") == campaign["generated_config_sha256"][cell["config_path"]],
            "Result campaign recipe differs from its pin.")
    episodes = result.get("episodes", [])
    require(len(episodes) == len(seeds) and sorted(row["seed"] for row in episodes) == seeds, "Episode seed panel differs.")
    from utils.transfer_campaign_diagnostics import verify_episode_diagnostics
    for episode in episodes:
        verify_episode_diagnostics(episode, campaign["diagnostics"], smoke=smoke)
        require(not smoke or episode.get("diagnostic_isolation_verified") is True,
                "Diagnostic smoke lacks exact controller/RNG isolation verification.")
        seed = episode["seed"]
        require(episode.get("solver_seed") == episode.get("episode_solver_seed") == solver_seed(55, "episode", seed),
                "Episode solver stream differs.")
        require(episode.get("length") == episode.get("steps") == limit and episode.get("smoke") is smoke,
                "Episode does not contain the full requested decision horizon.")
        require(smoke or (episode.get("truncated") is True and not episode.get("terminated")
                and not episode.get("truncated_by_evaluator")), "Production did not complete a normal 500-decision episode.")
        require(math.isfinite(episode["return"]) and math.isfinite(episode["control_seconds"])
                and episode["control_seconds"] > 0, "Invalid episode return/controller timing.")
        rows = [json.loads(line) for line in (directory / f"decisions-seed-{seed}.jsonl").read_text().splitlines()]
        require(len(rows) == limit and [row["decision"] for row in rows] == list(range(limit))
                and all(row["seed"] == seed for row in rows), "Decision trace is incomplete or belongs to another seed.")
        require(all(math.isfinite(row["reward"]) and math.isfinite(row["control_seconds"]) and row["control_seconds"] >= 0 for row in rows),
                "Decision trace has nonfinite return/timing.")
        require(all(row.get("action") and all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in row["action"]) for row in rows), "Decision trace has missing/nonfinite actions.")
        diagnostic_rows = [row["diagnostics"] for row in rows if "diagnostics" in row]
        require([row.get("decision") for row in diagnostic_rows] == episode["diagnostics"]["completed_decisions"]
                and all(row.get("summary") and all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in row["summary"].values()) for row in diagnostic_rows),
                "Sampled diagnostic decision records are missing or nonfinite.")
        for row in rows:
            metrics = row.get("metrics", {})
            expected_updates = dict(inner_actor_optimizer_steps=4 * cell["J"], inner_critic_optimizer_steps=16 * cell["J"],
                                    inner_model_steps_budget=128 * cell["H"] * cell["J"])
            require(all(metrics.get(key) == value for key, value in expected_updates.items()), "Realized solver update/model budget differs.")
            require(all(key in metrics and isinstance(metrics[key], (int, float)) and math.isfinite(metrics[key])
                for key in ("inner_actor_q_mean", "inner_actor_entropy", "inner_q_mean", "inner_q_target_mean",
                            "inner_td_error_abs_mean", "inner_temperature_optimizer_steps")), "Cheap Q/TD/entropy measurements missing.")
        require(math.isclose(sum(row["reward"] for row in rows), episode["return"], rel_tol=1e-10, abs_tol=1e-8)
                and math.isclose(sum(row["control_seconds"] for row in rows), episode["control_seconds"], rel_tol=1e-10, abs_tol=1e-8),
                "Decision trace totals differ from completed episode.")
    return result, manifest


def receipt_path(root, cell, *, smoke=False):
    return Path(root) / ("smoke-receipts" if smoke else "receipts") / f"{cell['index']}.json"


def make_receipt(root, campaign, cell, directory, *, smoke=False, hardware, historical=False, elapsed_seconds=None):
    require(not historical, "Historical transfer reuse is not authorized for the five-seed shortlist.")
    result, manifest = validate_result(directory, campaign, cell, smoke=smoke)
    seeds = campaign["smoke_seeds"] if smoke else campaign["seeds"]
    files = ["results.json", "manifest.json", *[f"decisions-seed-{seed}.jsonl" for seed in seeds]]
    return dict(schema_version=1, status="complete", campaign_sha256=digest(Path(root) / "campaign.json"),
                cell_index=cell["index"], cell_name=cell["name"], setting_id=cell["setting_id"], checkpoint_step=cell["step"],
                checkpoint_sha256=manifest["checkpoint_sha256"], resolved_sha256=cell["resolved_sha256"],
                result_dir=str(Path(directory).resolve()), file_sha256={name: digest(Path(directory) / name) for name in files},
                source=manifest["source"], current_campaign_source_commit=campaign["source_commit"],
                metric_coverage="all_computed_inner_scalars_and_sampled_diagnostics",
                smoke=smoke, historical_reuse=historical, frozen_outer_verified=True,
                hardware=hardware, elapsed_seconds=elapsed_seconds, slurm_job_id=os.environ.get("SLURM_JOB_ID"), episodes=len(result["episodes"]))


def validate_receipt(root, campaign, cell, *, smoke=False):
    receipt = read_json(receipt_path(root, cell, smoke=smoke))
    directory = Path(root) / "smoke" / cell["name"] if smoke else Path(cell["result_dir"])
    expected = dict(status="complete", campaign_sha256=digest(Path(root) / "campaign.json"), cell_index=cell["index"],
                    cell_name=cell["name"], setting_id=cell["setting_id"], checkpoint_step=cell["step"],
                    checkpoint_sha256=campaign["checkpoints"][cell["checkpoint_index"]]["checkpoint_sha256"],
                    resolved_sha256=cell["resolved_sha256"], smoke=smoke, result_dir=str(directory.resolve()))
    require(all(receipt.get(key) == value for key, value in expected.items()), "Completion receipt differs from campaign/cell.")
    historical = bool(cell.get("historical_reuse")) and not smoke
    require(not historical, "Historical transfer reuse is not authorized for the five-seed shortlist.")
    result, manifest = validate_result(directory, campaign, cell, smoke=smoke)
    require(receipt.get("source") == manifest["source"] and receipt.get("historical_reuse") is historical
            and receipt.get("frozen_outer_verified") is True and "L40S" in receipt.get("hardware", ""), "Receipt provenance/hardware differs.")
    seeds = campaign["smoke_seeds"] if smoke else campaign["seeds"]
    files = {"results.json", "manifest.json", *[f"decisions-seed-{seed}.jsonl" for seed in seeds]}
    require(set(receipt.get("file_sha256", {})) == files and all(digest(directory / name) == expected_digest
            for name, expected_digest in receipt["file_sha256"].items()), "Receipt artifact hash mismatch.")
    return receipt


def worker(args):
    campaign = read_json(args.root / "campaign.json")
    current = source(args.expected_source_sha)
    require(all(campaign[key] == value for key, value in current.items()), "Worker source differs from prepared campaign.")
    require(0 <= args.index < len(campaign["cells"]), "Worker index outside campaign.")
    cell = campaign["cells"][args.index]
    require(cell["index"] == args.index, "Cell index mismatch.")
    checkpoint = campaign["checkpoints"][cell["checkpoint_index"]]
    require(digest(checkpoint["checkpoint"]) == checkpoint["checkpoint_sha256"]
            and digest(checkpoint["metadata"]) == checkpoint["metadata_sha256"], "Checkpoint changed after preparation.")
    require(all(digest(path) == value for path, value in campaign["generated_config_sha256"].items()), "Generated configuration changed.")
    if args.smoke:
        require(args.index in campaign["smoke_indices"], "Unexpected smoke cell.")
    else:
        for index in campaign["smoke_indices"]:
            validate_receipt(args.root, campaign, campaign["cells"][index], smoke=True)
    directory = args.root / "smoke" / cell["name"] if args.smoke else Path(cell["result_dir"])
    require(not directory.exists() and not receipt_path(args.root, cell, smoke=args.smoke).exists(),
            f"Refusing to overwrite existing output/receipt: {directory}")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()
    require("L40S" in hardware, "Comparable controller timing requires an L40S allocation.")
    command = [sys.executable, str(ROOT / "evaluate_ambi_transfer_campaign.py"), "--campaign", cell["config_path"],
        "--checkpoint", checkpoint["checkpoint"], "--metadata", checkpoint["metadata"],
        "--horizon", str(cell["H"]), "--rounds", str(cell["J"]), "--arm", cell["arm"],
        "--output-dir", str(directory), "--device", "cuda", "--controller-seed", "55", "--metric-policy", "all_scalars",
        "--seeds", *map(str, campaign["smoke_seeds"] if args.smoke else campaign["seeds"]),
        "--max-steps", str(campaign["smoke_steps"] if args.smoke else campaign["max_steps"])]
    if args.smoke:
        command.append("--smoke")
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        receipt = make_receipt(args.root, campaign, cell, directory, smoke=args.smoke, hardware=hardware,
                               elapsed_seconds=time.perf_counter() - started)
        write(receipt_path(args.root, cell, smoke=args.smoke), receipt)
    except BaseException as error:
        write(args.root / "failures" / f"{'smoke-' if args.smoke else ''}{args.index}.json",
              dict(error_type=type(error).__name__, error=str(error), source_commit=campaign["source_commit"], time=time.time()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, default=ROOT / "configs/research/ambi_transfer_checkpoint_curves.json")
    prep.add_argument("--inventory", type=Path, required=True)
    prep.add_argument("--settings", nargs="+")
    work = sub.add_parser("worker")
    work.add_argument("--index", type=int, required=True)
    work.add_argument("--smoke", action="store_true")
    for command in (prep, work):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args(argv)
    return (prepare if args.mode == "prepare" else worker)(args)


if __name__ == "__main__":
    main()
