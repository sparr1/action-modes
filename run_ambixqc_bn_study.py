"""Staged, immutable AMBI-XQC BatchNorm study on the pinned shared UTD2 475k prior.

CPU commands prepare/inspect/spec/collect never train or submit jobs. GPU workers
run one explicit condition and stage validated results for the existing publisher.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

import run_ambixqc_inner_475k_screen as screen
from run_ambixqc_mppi_evaluation import file_sha256

ROOT = Path(__file__).resolve().parent
SCHEMA = "ambixqc-bn-study-plan-v1"
RESULT_SCHEMA = "ambixqc-bn-study-results-v1"
MAP_SCHEMA = "ambixqc-bn-study-run-map-v1"
MODES = ("batch_update", "batch_no_update", "running")
ROUTES = ("soft_soft", "return_return")
CHECKPOINT_SHA = screen.CHECKPOINT_SHA
SEEDS = screen.SEEDS
TIE_RULE = "highest_mean_raw_return_then_condition_order"
BASE_SETTINGS = {
    "inner_operator": "xqc", "inner_rollout_horizon": 1,
    "inner_rollouts_per_round": 256, "inner_updates_per_round": 3,
    "inner_batch_size": 256, "inner_replay_capacity": 1024,
    "inner_replay_sampling": "with_replacement", "inner_update_timing": "round",
    "inner_policy_delay": 3, "inner_critic_lr": 5e-5, "inner_temperature_lr": 5e-5,
    "inner_terminal_bootstrap": "outer", "inner_reward_normalization": "frozen_real_scale",
    "inner_diagnostics_every": 1, "inner_actor_bn_mode": "running",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return screen.banks.read_json(path)


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def check_binding(binding):
    path = Path(binding["path"])
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != binding["sha256"]:
        raise ValueError("A pinned study input is missing or changed: " + str(path))
    return path


def immutable_json(path, value):
    """Publish complete JSON without replacing an existing artifact."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if read(path) != value:
            raise FileExistsError("Refusing to replace study artifact: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def condition(mode, route, j, actor_lr):
    if mode not in MODES or route not in ROUTES or j not in (1, 4) or actor_lr not in (1.25e-5, 5e-5):
        raise ValueError("Unsupported study condition.")
    source = "xqc" if route == "soft_soft" else "aux_return"
    rate = "low" if actor_lr == 1.25e-5 else "high"
    name = f"{route}_{mode}_j{j}_actor_{rate}"
    settings = {**BASE_SETTINGS, "inner_critic_bn_mode": mode, "inner_rounds": j,
                "inner_actor_lr": actor_lr, "inner_critic_source": source,
                "inner_horizon_critic_source": source,
                "inner_critic_target": "entropy_augmented" if source == "xqc" else "reward_only"}
    return {"selector": "controller/" + name, "critic_bn_mode": mode, "route": route,
            "rounds": j, "actor_lr": actor_lr, "settings": settings}


def expected_settings(cell):
    return {**cell["settings"], "inner_temperature_mode": "auto",
            "inner_temperature_initialization": "inherit_outer",
            "inner_actor_scope": "action", "inner_critic_scope": "action", "inner_temperature_scope": "action",
            "inner_replay_scope": "action", "inner_actor_optimizer_scope": "action",
            "inner_critic_optimizer_scope": "action", "inner_temperature_optimizer_scope": "action",
            "inner_actor_adaptation": "clone", "inner_critic_adaptation": "clone",
            "inner_rebase_persistent": False, "aux_return_mode": "xqc",
            "aux_return_detach_representation": False, "xqc_utd": 2, "utd": 1}


def matrix_for(cells):
    variants = {cell["selector"].split("/")[1]: {
        "description": f"{cell['route']}; critic BN {cell['critic_bn_mode']}; H1 J{cell['rounds']}; "
                       f"actor LR {cell['actor_lr']}; critic and temperature LR fixed 5e-5.",
        "alg_params": cell["settings"]} for cell in cells}
    variants["prior"] = {"description": "Reference only; reuse completed checkpoint-matched episodes.",
                         "alg_params": {"inner_operator": "none"}}
    return {"schema_version": 1, "description": "Explicit staged AMBI-XQC BN study; no default launch.",
            "base_alg_config": "checkpoint", "shared_alg_params": {},
            "evaluation": {"controller_seed": 12345, "seeds": SEEDS, "max_steps": 500,
                           "default_presets": [], "wandb_project": "ambi-inner-bench"},
            "comparisons": {"controller": {"reference": "prior", "variants": variants}}}


def prepare(stage, output, *, source_sha, selection=None, stage2_results=None):
    if not isinstance(source_sha, str) or not re.fullmatch("[a-f0-9]{40}", source_sha):
        raise ValueError("Provide the exact clean source commit.")
    dependencies, reused = {}, []
    if stage == "smoke":
        # Cover every mode and route plus J4 accounting and independent temperature LR.
        cells = [condition(mode, route, 4, 1.25e-5) for mode in MODES for route in ROUTES]
    elif stage == "stage2":
        if selection is None:
            raise ValueError("Stage 2 requires the validated Stage 1 selection.")
        from run_ambixqc_bn_probe import validate_selection
        validate_selection(selection, source_sha=source_sha)
        selected = read(selection)["selected_critic_bn_mode"]
        if selected not in MODES[1:]:
            raise ValueError("Stage 1 must select batch_no_update or running.")
        dependencies["selection"] = bind(selection)
        cells = [condition(mode, route, 1, 5e-5) for mode in ("batch_update", selected) for route in ROUTES]
    elif stage == "stage3":
        if stage2_results is None:
            raise ValueError("Stage 3 requires all four validated Stage 2 results.")
        result = validate_result_index(stage2_results, source_sha=source_sha)
        if result["stage"] != "stage2":
            raise ValueError("Stage 3 can select only from Stage 2.")
        winner = result["winner"]
        selected = winner["condition"]
        all_cells = [condition(selected["critic_bn_mode"], selected["route"], j, lr)
                     for j in (1, 4) for lr in (1.25e-5, 5e-5)]
        reused = [{"condition": selected, "result": winner, "reason": "Identical J1/actor LR5e-5 Stage 2 result"}]
        cells = [cell for cell in all_cells if cell != selected]
        if len(cells) != 3:
            raise ValueError("Stage 3 must reuse exactly the identical Stage 2 winning condition.")
        dependencies["stage2_results"] = bind(stage2_results)
    else:
        raise ValueError("Stage must be smoke, stage2 or stage3.")
    output = Path(output).resolve()
    matrix_path = output.with_suffix(".matrix.json")
    immutable_json(matrix_path, matrix_for(cells))
    plan = {"schema": SCHEMA, "stage": stage, "source_sha": source_sha,
            "checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_step": screen.STEP,
            "conditions": cells, "reused": reused, "dependencies": dependencies,
            "matrix": bind(matrix_path), "selection_rule": TIE_RULE,
            "environment_seeds": [101, 102] if stage == "smoke" else SEEDS,
            "max_steps": 3 if stage == "smoke" else 500, "controller_seed": 12345}
    plan["plan_sha256"] = digest(plan)
    immutable_json(output, plan)
    return plan


def load_plan(path, *, source_sha=None):
    plan = read(path)
    signed = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("schema") != SCHEMA or plan.get("plan_sha256") != digest(signed)
            or plan.get("checkpoint_sha256") != CHECKPOINT_SHA
            or plan.get("checkpoint_step") != screen.STEP or plan.get("selection_rule") != TIE_RULE
            or source_sha is not None and plan.get("source_sha") != source_sha):
        raise ValueError("Study plan identity, checkpoint or digest differs.")
    stage = plan.get("stage")
    expected_count = {"smoke": 6, "stage2": 4, "stage3": 3}.get(stage)
    cells = plan.get("conditions", [])
    if len(cells) != expected_count or len({c["selector"] for c in cells}) != len(cells):
        raise ValueError("Study plan has missing or duplicate conditions.")
    for cell in cells:
        if cell != condition(cell["critic_bn_mode"], cell["route"], cell["rounds"], cell["actor_lr"]):
            raise ValueError("Study plan changed a controlled condition.")
    if stage == "smoke":
        expected_cells = [condition(mode, route, 4, 1.25e-5) for mode in MODES for route in ROUTES]
        if cells != expected_cells or plan["dependencies"] or plan["reused"]:
            raise ValueError("Smoke plan must cover all modes and routes.")
    elif stage == "stage2":
        if set(plan["dependencies"]) != {"selection"} or plan["reused"]:
            raise ValueError("Stage 2 plan must pin only Stage 1 selection.")
        selected = read(check_binding(plan["dependencies"]["selection"]))["selected_critic_bn_mode"]
        if selected not in MODES[1:] or cells != [condition(mode, route, 1, 5e-5)
                for mode in ("batch_update", selected) for route in ROUTES]:
            raise ValueError("Stage 2 differs from the diagnostic selection.")
    else:
        if set(plan["dependencies"]) != {"stage2_results"} or len(plan["reused"]) != 1:
            raise ValueError("Stage 3 must pin its complete Stage 2 comparison and reused winner.")
        prior = read(check_binding(plan["dependencies"]["stage2_results"]))
        winner = prior["winner"]
        if plan["reused"][0]["result"] != winner or plan["reused"][0]["condition"] != winner["condition"]:
            raise ValueError("Stage 3 reused result differs from Stage 2 winner.")
        selected = winner["condition"]
        expected_cells = [condition(selected["critic_bn_mode"], selected["route"], j, lr)
                          for j in (1, 4) for lr in (1.25e-5, 5e-5)]
        if cells != [cell for cell in expected_cells if cell != selected]:
            raise ValueError("Stage 3 differs from the predeclared factorial.")
    if read(check_binding(plan["matrix"])) != matrix_for(cells):
        raise ValueError("Generated matrix differs from the study plan.")
    if (plan["environment_seeds"] != ([101, 102] if stage == "smoke" else SEEDS)
            or plan["max_steps"] != (3 if stage == "smoke" else 500)
            or plan["controller_seed"] != 12345):
        raise ValueError("Study episode protocol differs.")
    return plan


def select_condition(plan, index):
    if type(index) is not int or not 0 <= index < len(plan["conditions"]):
        raise ValueError("Condition index is outside this explicit plan.")
    return plan["conditions"][index]


def validate_output(path, plan, index):
    path = Path(path)
    saved = read(path / "validation.json")
    cell = select_condition(plan, index)
    if not (path / "PASS").is_file() or (path / "PASS").read_text().strip() != "PASS" or (path.parent / "FAILED").exists():
        raise ValueError("Study evaluation is incomplete or failed.")
    if (saved.get("plan_sha256") != plan["plan_sha256"] or saved.get("index") != index
            or saved.get("source_sha") != plan["source_sha"]):
        raise ValueError("Study result source or condition differs.")
    check_binding(saved["inventory"])
    if saved.get("reference_index"):
        check_binding(saved["reference_index"])
    actual = screen.validate_bundle(path / "bundle", 0, seeds=plan["environment_seeds"],
                                    max_steps=plan["max_steps"], source_sha=plan["source_sha"],
                                    reference_bundle=saved.get("reference_bundle"),
                                    expected_selector=cell["selector"], expected_config=expected_settings(cell))
    if any(saved.get(key) != value for key, value in actual.items()):
        raise ValueError("Study artifacts changed after validation.")
    if plan["stage"] != "smoke" and not actual["prior_reused"]:
        raise ValueError("Full episodes must reuse their paired prior.")
    episodes = read(path / "bundle/manifest.json")["runs"][0]["episodes"]
    return {"index": index, "condition": cell, "output_path": str(path.resolve()),
            "validation": bind(path / "validation.json"), "bundle_manifest": bind(path / "bundle/manifest.json"),
            "mean_return": statistics.mean(ep["return"] for ep in episodes),
            "episode_returns": {str(ep["seed"]): ep["return"] for ep in episodes}}


def collect(plan_path, result_root, output):
    plan = load_plan(plan_path)
    found = {}
    for path in sorted(Path(result_root).rglob("validation.json")):
        saved = read(path)
        if saved.get("plan_sha256") != plan["plan_sha256"]:
            continue
        index = saved.get("index")
        select_condition(plan, index)
        if index in found:
            raise ValueError("Duplicate result for the same study condition.")
        found[index] = validate_output(path.parent, plan, index)
    if set(found) != set(range(len(plan["conditions"]))):
        raise ValueError("All planned conditions must complete before selecting the next stage.")
    entries = [found[index] for index in sorted(found)]
    winner = max(entries, key=lambda row: (row["mean_return"], -row["index"]))
    result = {"schema": RESULT_SCHEMA, "stage": plan["stage"], "source_sha": plan["source_sha"],
              "checkpoint_sha256": CHECKPOINT_SHA, "plan": bind(plan_path),
              "entries": entries, "selection_rule": TIE_RULE, "winner": winner}
    result["results_sha256"] = digest(result)
    immutable_json(output, result)
    return result


def validate_result_index(path, *, source_sha=None):
    result = read(path)
    signed = {k: v for k, v in result.items() if k != "results_sha256"}
    if (result.get("schema") != RESULT_SCHEMA or result.get("results_sha256") != digest(signed)
            or result.get("checkpoint_sha256") != CHECKPOINT_SHA or result.get("selection_rule") != TIE_RULE):
        raise ValueError("Study result-index identity or digest differs.")
    plan = load_plan(check_binding(result["plan"]), source_sha=source_sha)
    if result["source_sha"] != plan["source_sha"] or result["stage"] != plan["stage"]:
        raise ValueError("Study result index differs from its plan.")
    entries = result["entries"]
    if [entry["index"] for entry in entries] != list(range(len(plan["conditions"]))):
        raise ValueError("Study result index must contain each condition exactly once.")
    for entry in entries:
        check_binding(entry["validation"])
        check_binding(entry["bundle_manifest"])
        if validate_output(entry["output_path"], plan, entry["index"]) != entry:
            raise ValueError("Study result index disagrees with its validated bundle.")
    winner = max(entries, key=lambda row: (row["mean_return"], -row["index"]))
    if result["winner"] != winner:
        raise ValueError("Recorded winner differs from the predeclared selection rule.")
    return result


def verify_smokes(root, manifest, source_sha):
    indexes = sorted(Path(root).rglob("validation.json"))
    found, smoke_plan = {}, None
    for path in indexes:
        saved = read(path)
        if saved.get("study_stage") != "smoke":
            continue
        plan = load_plan(check_binding(saved["plan"]), source_sha=source_sha)
        if smoke_plan is not None and plan != smoke_plan:
            raise ValueError("Smoke evidence mixes different plans.")
        smoke_plan = plan
        index = saved["index"]
        if index in found or saved["inventory"]["sha256"] != file_sha256(manifest):
            raise ValueError("Duplicate smoke or different checkpoint inventory.")
        job = path.parent.parent
        if not (job / "PASS").is_file() or (job / "PASS").read_text().strip() != "PASS":
            raise ValueError("GPU smoke job has not completed.")
        runtime = read(job / "runtime.json")
        if (not runtime.get("gpu") or not runtime.get("torch", "").startswith("2.3.1")
                or runtime.get("cuda_device_count") != 1):
            raise ValueError("Smoke lacks verified allocated CUDA runtime evidence.")
        found[index] = validate_output(path.parent, plan, index)
        if index == 0 and not re.search(r"\b[1-9][0-9]* passed\b", (job / "pytest.log").read_text()):
            raise ValueError("CUDA regression tests did not pass.")
    if set(found) != set(range(6)):
        raise ValueError("Require all six critic-BN mode/route GPU smokes.")
    return {"validated_indices": sorted(found), "source_sha": source_sha,
            "smoke_plan_sha256": smoke_plan["plan_sha256"]}


def run_map(path, plan, cell):
    if plan["stage"] == "smoke":
        if path is not None:
            raise ValueError("Smoke cannot assign publication curves.")
        return {}
    if path is None:
        raise ValueError("Production requires an explicit run map.")
    data = read(path)
    mapping = data.get("runs", {})
    if (data.get("schema") != MAP_SCHEMA or data.get("plan_sha256") != plan["plan_sha256"]
            or set(mapping) != {c["selector"] for c in plan["conditions"]}
            or any(not isinstance(v, str) or not Path(v).is_absolute() for v in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("Run map must bind one distinct curve to every planned new condition.")
    from utils.ambi_benchmark import resolve_eval_run_map
    return resolve_eval_run_map([cell["selector"]], run_map={cell["selector"]: mapping[cell["selector"]]})


def prepare_spec(plan_path, index, manifest, output, *, checkpoint_root=None):
    plan = load_plan(plan_path)
    cell = select_condition(plan, index)
    row = screen.select_checkpoint(manifest, checkpoint_root=checkpoint_root)
    from evaluate_ambi_checkpoint import evaluate_matrix
    return evaluate_matrix(plan["matrix"]["path"], row["path"], selectors=[cell["selector"]],
                           checkpoint_inventory=manifest, source_run=row["source_run"], eval_series_spec_dir=output)


def run(plan_path, index, manifest, result_root, *, source_sha, checkpoint_root=None,
        reference_index=None, eval_run_map=None, smoke_root=None, device="cuda"):
    plan = load_plan(plan_path, source_sha=source_sha)
    cell = select_condition(plan, index)
    from utils.ambi_benchmark import code_identity, stage_completed_bundle
    code = code_identity()
    if code.get("dirty") is not False or code.get("commit") != source_sha:
        raise ValueError("Study requires the exact clean tested source commit.")
    row = screen.select_checkpoint(manifest, checkpoint_root=checkpoint_root)
    smoke = plan["stage"] == "smoke"
    if not smoke:
        if smoke_root is None:
            raise ValueError("Production requires validated all-mode GPU smokes.")
        verify_smokes(smoke_root, manifest, source_sha)
        if plan["stage"] == "stage2":
            from run_ambixqc_bn_probe import validate_selection
            validate_selection(check_binding(plan["dependencies"]["selection"]), source_sha=source_sha)
        else:
            validate_result_index(check_binding(plan["dependencies"]["stage2_results"]), source_sha=source_sha)
    reference = screen.select_reference(reference_index, row, manifest) if not smoke else None
    assigned = run_map(eval_run_map, plan, cell)
    output = Path(result_root).resolve() / cell["selector"].split("/")[1]
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Study results must be outside the source checkout.")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"study_stage": plan["stage"], "source_sha": source_sha,
                  "plan": bind(plan_path), "plan_sha256": plan["plan_sha256"], "index": index,
                  "condition": cell, "inventory": bind(manifest), "checkpoint": row,
                  "reference_bundle": str(reference) if reference else None,
                  "reference_index": bind(reference_index) if reference else None, "eval_run_map": assigned}
    immutable_json(output / "provenance.json", provenance)
    from evaluate_ambi_checkpoint import evaluate_matrix
    payload = evaluate_matrix(plan["matrix"]["path"], row["path"], selectors=[cell["selector"]],
                              seeds=plan["environment_seeds"], max_steps=plan["max_steps"],
                              controller_seed=12345, device=device, bundle_dir=output / "bundle",
                              reference_bundle=reference, checkpoint_inventory=manifest,
                              source_run=row["source_run"], eval_run_map=assigned or None, stage_results=False)
    immutable_json(output / "results.json", payload)
    if payload["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Study evaluated a different checkpoint.")
    validation = screen.validate_bundle(output / "bundle", 0, seeds=plan["environment_seeds"],
                                       max_steps=plan["max_steps"], source_sha=source_sha,
                                       reference_bundle=reference, expected_selector=cell["selector"],
                                       expected_config=expected_settings(cell))
    immutable_json(output / "validation.json", {**provenance, **validation})
    if assigned:
        staged = stage_completed_bundle(output / "bundle", assigned, source_run=row["source_run"], inventory_path=manifest)
        if set(staged) != {cell["selector"]} or staged[cell["selector"]]["status"] != "queued":
            raise RuntimeError("Result staging failed; preserve complete bundles for publication recovery.")
    (output / "PASS").write_text("PASS\n")
    return {"output": str(output), "selector": cell["selector"], **validation}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--stage", choices=("smoke", "stage2", "stage3"), required=True)
    p.add_argument("--output-plan", type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--stage2-results", type=Path)
    for name in ("inspect", "spec", "run", "collect"):
        p = commands.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        if name in ("spec", "run"):
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--manifest", type=Path, required=True)
            p.add_argument("--checkpoint-root", type=Path)
        if name == "spec":
            p.add_argument("--spec-dir", type=Path, required=True)
        if name in ("run", "collect"):
            p.add_argument("--result-root", type=Path, required=True)
        if name == "collect":
            p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument("--source-sha", required=True)
            p.add_argument("--reference-index", type=Path)
            p.add_argument("--eval-run-map", type=Path)
            p.add_argument("--smoke-root", type=Path)
            p.add_argument("--device", default="cuda")
    p = commands.add_parser("verify-smokes")
    p.add_argument("--smoke-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        result = prepare(args.pop("stage"), args.pop("output_plan"), **args)
    elif command == "inspect":
        result = load_plan(args["plan"])
    elif command == "spec":
        result = prepare_spec(args["plan"], args["index"], args["manifest"], args["spec_dir"], checkpoint_root=args["checkpoint_root"])
    elif command == "collect":
        result = collect(args["plan"], args["result_root"], args["output"])
    elif command == "verify-smokes":
        result = verify_smokes(args["smoke_root"], args["manifest"], args["source_sha"])
    else:
        args["plan_path"] = args.pop("plan")
        result = run(**args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
