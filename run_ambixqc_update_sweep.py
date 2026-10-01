"""Frozen 475k AMBI-XQC update-dose sweep using the original experiment source.

Only the explicitly selected J/G cells run. Exact J1/G3 and J4/G3 results are
reused. Replay holds every imagined row (at least 1024); the original learner,
optimizer ordering, per-action resets and constant learning rates are unchanged.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import run_ambixqc_critic_lr_study as critic

continuation = critic.continuation
TOOLING_ROOT = Path(__file__).resolve().parent
SOURCE_SHA = critic.SOURCE_SHA
SCHEMA = "ambixqc-update-sweep-v1"
ROUNDS = (1, 2, 4, 6)
UPDATES = (3, 6, 9)
REUSED = {(1, 3), (4, 3)}


def checked_rounds(rounds):
    values = list(rounds)
    if (not values or any(type(j) is not int or j not in ROUNDS for j in values)
            or len(set(values)) != len(values)):
        raise ValueError("Rounds must be a nonempty unique subset of 1, 2, 4, 6")
    return sorted(values)


def conditions(study, rounds=ROUNDS):
    cells = []
    for j in checked_rounds(rounds):
        for g in UPDATES:
            cell = study.condition("running", "return_return", 1 if j == 1 else 4, 5e-5)
            cell.update(rounds=j, updates_per_round=g)
            cell["settings"].update(inner_rounds=j, inner_updates_per_round=g,
                                    inner_replay_capacity=max(1024, 256*j))
            if (j, g) not in REUSED:
                cell["selector"] = f"controller/return_return_running_j{j}_g{g}_actor_high"
            cells.append(cell)
    return cells


def split_conditions(study, rounds=ROUNDS):
    cells = conditions(study, rounds)
    new = [cell for cell in cells if (cell["rounds"], cell["updates_per_round"]) not in REUSED]
    new.sort(key=lambda c: (c["rounds"]*c["updates_per_round"], c["rounds"], c["updates_per_round"]), reverse=True)
    return (new,
            [cell for cell in cells if (cell["rounds"], cell["updates_per_round"]) in REUSED])


def expected_counts(cell):
    settings = cell["settings"]
    slots = settings["inner_rounds"] * settings["inner_updates_per_round"]
    actor = (slots - 1) // settings["inner_policy_delay"] + 1
    return {"critic": slots, "actor": actor, "temperature": actor,
            "model_steps": settings["inner_rounds"] * settings["inner_rollouts_per_round"] * settings["inner_rollout_horizon"],
            "replay_draws": slots * settings["inner_batch_size"]}


def matrix_for(study, cells):
    matrix = study.matrix_for(cells)
    matrix["description"] = "Frozen update-dose sweep; exact compatible J1/G3 and J4/G3 results reused."
    for cell in cells:
        matrix["comparisons"]["controller"]["variants"][cell["selector"].split("/")[1]]["description"] = (
            f"Return/return, running actor/critic BN; H1 J{cell['rounds']} G{cell['updates_per_round']} "
            f"N256 B256 delay3; all LRs5e-5; replay capacity {cell['settings']['inner_replay_capacity']}.")
    return matrix


def validate_bundle(plan, index, root, study, smoke):
    """Original screen checks, with J/G-aware counters and replay work checks.

    The pinned screen validator assumes G3. Keep validation in tooling instead
    of changing or monkeypatching that scientific checkout.
    """
    path = Path(root) / "bundle"
    saved = study.read(path / "manifest.json")
    cell = plan["conditions"][index]
    cfg_expected = study.expected_settings(cell)
    counts = expected_counts(cell)
    seeds = plan["smoke_seeds"] if smoke else plan["environment_seeds"]
    max_steps = plan["smoke_max_steps"] if smoke else plan["max_steps"]
    if (saved.get("status") != "complete" or saved.get("code", {}).get("dirty") is not False
            or saved.get("code", {}).get("commit") != plan["source_sha"]
            or saved.get("checkpoint", {}).get("sha256") != study.CHECKPOINT_SHA
            or saved.get("protocol") != {**study.screen.PROTOCOL, "max_steps": max_steps}
            or len(saved.get("runs", [])) != 1):
        raise ValueError("Bundle source, checkpoint, protocol or completion differs")
    run = saved["runs"][0]
    result = run["result"]
    if (run.get("selector") != cell["selector"] or run.get("status") != "complete"
            or result.get("action_rule") != "tanh_mean" or result.get("outer_state_unchanged") is not True
            or result.get("outer_updates_before") != result.get("outer_updates_after")
            or result.get("nonfinite_model_metrics") or result.get("nonfinite_trace_metrics")):
        raise ValueError("Evaluation changed frozen outer state or did not finish with finite metrics")
    cfg = result["resolved_config"]
    if any(cfg.get(key) != value for key, value in cfg_expected.items()):
        raise ValueError("Resolved update-dose settings differ")
    signature = result.get("checkpoint_evaluation_provenance", {}).get("evaluated_semantic_signature", {})
    if any(signature.get(key) != "running" for key in ("inner_actor_bn_mode", "inner_critic_bn_mode")):
        raise ValueError("Frozen-evaluation provenance does not record running actor/critic BN")
    if [ep["seed"] for ep in run["episodes"]] != list(seeds):
        raise ValueError("Episode seeds differ")
    components = ("critic", "actor", "temperature")
    if run.get("actual_optimizer_steps") != {key: counts[key]*max_steps*len(seeds) for key in components}:
        raise ValueError("Actual run optimizer counts differ")
    from utils.ambi_benchmark import reference_returns, solver_seed
    reference = None if smoke else reference_returns(plan["reference_bundle"], study.CHECKPOINT_SHA, saved["protocol"])
    for episode in run["episodes"]:
        if (episode["length"] != max_steps or not math.isfinite(episode["return"])
                or max_steps == 500 and episode.get("capped")):
            raise ValueError("Episode length, cap or return is invalid")
        if (episode.get("solver_seed") != solver_seed(12345, "episode", episode["seed"])
                or episode.get("actual_optimizer_steps") != {key: counts[key]*max_steps for key in components}):
            raise ValueError("Episode solver seed or optimizer counts differ")
        if reference is not None and not math.isclose(episode["paired_return_delta"],
                episode["return"] - reference[episode["seed"]], rel_tol=0, abs_tol=1e-9):
            raise ValueError("Paired gain differs from the reused prior episode")
    seen, traces = set(), {}
    for relative in run["trace_files"]:
        trace = (path / relative).resolve()
        if path.resolve() not in trace.parents or str(relative) in traces:
            raise ValueError("Trace paths escape the bundle or are duplicated")
        traces[str(relative)] = study.bind(trace)["sha256"]
        with gzip.open(trace, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                identity = (event["episode_id"], event["decision_index"])
                if identity in seen or event["phase"] != "decision" or event["nonfinite"]:
                    raise ValueError("Trace decisions are duplicated, nonfinite or the wrong phase")
                seen.add(identity)
                if (event["critic_updates"], event["actor_updates"], event["temperature_updates"]) != (
                        counts["critic"], counts["actor"], counts["temperature"]):
                    raise ValueError("Actual per-decision optimizer counts differ")
                metrics = event["metrics"]
                required = {"decision/inner_model_steps": counts["model_steps"],
                            "decision/inner_replay_draws": counts["replay_draws"],
                            "decision/inner_buffer_size": counts["model_steps"],
                            "decision/inner_critic_optimizer_steps": counts["critic"],
                            "decision/inner_actor_optimizer_steps": counts["actor"],
                            "decision/inner_temperature_optimizer_steps": counts["temperature"],
                            "decision/inner_compile_fallback": 0,
                            "decision/inner_reward_scale_delta": 0,
                            "decision/inner_reward_normalizer_imagined_updates": 0,
                            "decision/inner_diagnostics_sampled": 1,
                            "decision/inner_critic_source_aux_return": 1,
                            "decision/inner_horizon_critic_source_aux_return": 1,
                            "decision/inner_critic_target_reward_only": 1}
                if any(metrics.get(key) != value for key, value in required.items()):
                    raise ValueError("Trace work, frozen scale, routing or sampling differs")
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values()):
                    raise ValueError("Trace contains nonfinite metrics")
    if seen != {(f"seed-{seed}", decision) for seed in seeds for decision in range(max_steps)}:
        raise ValueError("Trace decisions are missing, extra or misaligned")
    return {"outer_state_unchanged": True, "episodes": len(seeds), "decisions": len(seen),
            "critic_updates_per_decision": counts["critic"], "actor_updates_per_decision": counts["actor"],
            "temperature_updates_per_decision": counts["temperature"], "model_steps_per_decision": counts["model_steps"],
            "actor_bn_mode": "running", "prior_reused": reference is not None,
            "trace_sha256": traces, "bundle_manifest_sha256": study.bind(path / "manifest.json")["sha256"]}


def publication_receipt(run_dir, selector, study, publication, source_sha, entry):
    registry, rid, record = publication._validated_record(run_dir, selector, study.CHECKPOINT_SHA, source_sha)
    path = Path(run_dir) / "bn-study-publication-verified.json"
    expected = {"selector": selector, "checkpoint_sha256": study.CHECKPOINT_SHA, "source_sha": source_sha,
                "run_id": registry["run_id"], "record_id": rid, "record_sha256": record["record_sha256"],
                "accepted": 1, "published": 1}
    raw = study.read(Path(run_dir) / "records" / (rid + ".json"))
    if (record["status"] != "published" or study.read(path) != expected
            or {str(ep["seed"]): ep["return"] for ep in raw["episodes"]} != entry["episode_returns"]):
        raise ValueError("Previous-stage publication or episode evidence differs")
    return study.bind(path), expected


def parent_evidence(args, study, publication):
    parent, stage3_baseline = critic.parent_evidence(args, study, publication)
    stage2 = study.validate_result_index(study.check_binding(parent["original"]["results"]), source_sha=args.source_sha)
    stage2_baseline, = [entry for entry in stage2["entries"]
                       if entry["condition"] == study.condition("running", "return_return", 1, 5e-5)]
    previous = args.previous_stage_root
    state_path, complete_path = previous / "coordinator-state.json", previous / "COMPLETE.json"
    state, complete = study.read(state_path), study.read(complete_path)
    if ({k: v for k, v in complete.items() if k != "workspace"} != state
            or state.get("schema") != critic.SCHEMA or state.get("source_sha") != args.source_sha
            or state.get("progress_run_id") != args.progress_run_id or state.get("stage4_complete") is not True
            or not str(state.get("job", "")).isdigit() or state.get("failures")):
        raise ValueError("Require the completed matching Stage 4 critic-rate study")
    plan_path = study.check_binding(state["plan"])
    previous_plan = critic.load_plan(plan_path, study, args.source_sha, state["tooling"]["commit"])
    if (plan_path != (previous / "stage4-plan.json").resolve() or previous_plan["parent"] != parent
            or previous_plan["reused"] != stage3_baseline
            or previous_plan["tooling"] != state["tooling"] or previous_plan["execution"] != state["execution"]):
        raise ValueError("Previous stage does not share this exact parent and scientific source")
    results_path = study.check_binding(state["stage4_results"])
    result = study.read(results_path)
    if (results_path != (previous / "stage4-results.json").resolve()
            or result.get("schema") != critic.SCHEMA or result.get("stage") != "stage4"
            or result.get("source_sha") != args.source_sha or result.get("plan") != study.bind(plan_path)
            or result.get("reused") != stage3_baseline or len(result.get("entries", [])) != 2):
        raise ValueError("Previous-stage results are incomplete or incompatible")
    receipts, expected_published = [], {}
    for index, cell in enumerate(previous_plan["conditions"]):
        entry = critic.validate_worker(previous_plan, index, state["job"], study)
        if result["entries"][index] != entry:
            raise ValueError("Previous-stage result differs from its completed GPU task")
        run_dir = Path(previous_plan["runs"][cell["selector"]]["path"]).parent
        binding, receipt = publication_receipt(run_dir, cell["selector"], study, publication, args.source_sha, entry)
        receipts.append(binding)
        expected_published[cell["selector"]] = receipt
    if state.get("published") != expected_published:
        raise ValueError("Previous-stage completion lacks acknowledged publication")
    evidence = {"stage3": parent, "stage4": {"state": study.bind(state_path), "complete": study.bind(complete_path),
                "plan": study.bind(plan_path), "results": study.bind(results_path), "publication_receipts": receipts}}
    baseline_entries = [stage2_baseline, stage3_baseline]
    reused = []
    for cell in split_conditions(study, args.rounds)[1]:
        entry, = [item for item in baseline_entries if item["condition"]["selector"] == cell["selector"]
                  and item["condition"]["settings"] == cell["settings"]]
        reused.append({"condition": cell, "result": entry})
    return evidence, reused


def prepare_plan(args, study, publication, provenance):
    evidence, reused = parent_evidence(args, study, publication)
    study.verify_smokes(args.smoke_root, args.manifest, args.source_sha)
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    reference = study.screen.select_reference(args.reference_index, row, args.manifest)
    cells, _ = split_conditions(study, args.rounds)
    matrix_path = args.result_root / "stage5.matrix.json"
    study.immutable_json(matrix_path, matrix_for(study, cells))
    from evaluate_ambi_checkpoint import evaluate_matrix
    runs = {}
    for index, cell in enumerate(cells):
        with tempfile.TemporaryDirectory(dir=args.result_root) as temporary:
            prepared = evaluate_matrix(matrix_path, row["path"], selectors=[cell["selector"]],
                                       checkpoint_inventory=args.manifest, source_run=row["source_run"],
                                       eval_series_spec_dir=Path(temporary) / "specs")
            if set(prepared.get("specs", {})) != {cell["selector"]}:
                raise ValueError("Each update-dose cell must resolve exactly one scientific identity")
            spec = study.read(prepared["specs"][cell["selector"]])
        study.immutable_json(args.result_root / "specs" / f"{index}.json", spec)
        registry = publication.allocate_curve(args.result_root, spec, "stage5", cell["selector"])
        runs[cell["selector"]] = study.bind(Path(registry["run_dir"]) / "run.json")
    plan = {"schema": SCHEMA, "stage": "stage5", "source_sha": args.source_sha, **provenance,
            "campaign": publication.CAMPAIGN, "checkpoint_sha256": study.CHECKPOINT_SHA, "checkpoint_step": 475000,
            "rounds": checked_rounds(args.rounds), "conditions": cells, "reused": reused, "parent": evidence,
            "parent_root": str(args.parent_root), "previous_stage_root": str(args.previous_stage_root),
            "progress_run_id": args.progress_run_id, "result_root": str(args.result_root),
            "matrix": study.bind(matrix_path), "runs": runs,
            "inputs": {key: study.bind(getattr(args, key)) for key in ("manifest", "reference_index")},
            "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else None,
            "reference_bundle": str(reference), "smoke_root": str(args.smoke_root),
            "environment_seeds": study.SEEDS, "controller_seed": 12345, "max_steps": 500,
            "smoke_seeds": [101, 102], "smoke_max_steps": 3,
            "baseline_conditions": 9, "baseline_episodes": 45}
    plan["plan_sha256"] = study.digest(plan)
    path = args.result_root / "stage5-plan.json"
    study.immutable_json(path, plan)
    return path, plan


def load_plan(path, study, source_sha, tooling_sha):
    plan = study.read(path)
    cells, reused = split_conditions(study, plan.get("rounds", []))
    signed = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("schema") != SCHEMA or plan.get("stage") != "stage5"
            or plan.get("campaign") != "ambixqc-bn-study-20260929" or plan.get("plan_sha256") != study.digest(signed)
            or source_sha != SOURCE_SHA or plan.get("source_sha") != source_sha
            or plan.get("tooling", {}).get("commit") != tooling_sha
            or plan.get("execution", {}).get("commit") != source_sha
            or plan.get("checkpoint_sha256") != study.CHECKPOINT_SHA or plan.get("checkpoint_step") != 475000
            or plan.get("rounds") != checked_rounds(plan["rounds"]) or plan.get("conditions") != cells
            or [item.get("condition") for item in plan.get("reused", [])] != reused
            or plan.get("environment_seeds") != study.SEEDS or plan.get("controller_seed") != 12345
            or plan.get("max_steps") != 500 or plan.get("smoke_seeds") != [101, 102] or plan.get("smoke_max_steps") != 3
            or plan.get("baseline_conditions") != 9 or plan.get("baseline_episodes") != 45):
        raise ValueError("Update-dose plan changed source, controlled settings, reuse or protocol")
    critic.bindings_unchanged(plan, study)
    if study.read(plan["matrix"]["path"]) != matrix_for(study, cells):
        raise ValueError("Update-dose matrix differs from its plan")
    if (set(plan["runs"]) != {cell["selector"] for cell in cells}
            or len({binding["path"] for binding in plan["runs"].values()}) != len(cells)):
        raise ValueError("Every new cell requires its own registry; reused cells cannot be allocated")
    for item in plan["reused"]:
        old = item["result"]["condition"]
        if old["settings"] != item["condition"]["settings"] or old["selector"] != item["condition"]["selector"]:
            raise ValueError("Reused result is not exactly compatible")
    return plan


def worker_root(plan, index, job):
    if (type(index) is not int or not 0 <= index < len(plan["conditions"])
            or not re.fullmatch(r"\d+", str(job))):
        raise ValueError("Require an index in the submitted update-dose array")
    return Path(plan["result_root"]) / plan.get("stage", "stage5") / f"job{job}-task{index}"


def evaluate_cell(plan, index, root, study, *, smoke, policy=None):
    policy = policy or sys.modules[__name__]
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_benchmark import stage_completed_bundle
    cell = plan["conditions"][index]
    root.mkdir(parents=True, exist_ok=False)
    manifest = study.check_binding(plan["inputs"]["manifest"])
    row = study.screen.select_checkpoint(manifest, checkpoint_root=plan["checkpoint_root"])
    assigned = None if smoke else {cell["selector"]: str(Path(plan["runs"][cell["selector"]]["path"]).parent)}
    payload = evaluate_matrix(plan["matrix"]["path"], row["path"], selectors=[cell["selector"]],
        seeds=plan["smoke_seeds"] if smoke else plan["environment_seeds"], controller_seed=12345,
        max_steps=plan["smoke_max_steps"] if smoke else 500, device="cuda", bundle_dir=root / "bundle",
        reference_bundle=None if smoke else plan["reference_bundle"], checkpoint_inventory=manifest,
        source_run=row["source_run"], eval_run_map=assigned, stage_results=False)
    if payload["checkpoint_sha256"] != study.CHECKPOINT_SHA:
        raise ValueError("Update-dose worker evaluated a different checkpoint")
    study.immutable_json(root / "results.json", payload)
    study.immutable_json(root / "validation.json", policy.validate_bundle(plan, index, root, study, smoke))
    if not smoke:
        staged = stage_completed_bundle(root / "bundle", assigned, source_run=row["source_run"], inventory_path=manifest)
        if set(staged) != {cell["selector"]} or staged[cell["selector"]]["status"] != "queued":
            raise RuntimeError("Complete update-dose result could not be queued for publication")
    (root / "PASS").write_text("PASS\n")


def parent_arguments(plan):
    return SimpleNamespace(parent_root=Path(plan["parent_root"]),
        previous_stage_root=Path(plan["previous_stage_root"]), rounds=plan["rounds"],
        source_sha=plan["source_sha"], progress_run_id=plan["progress_run_id"])


def worker(args, coordinator, study, publication, provenance, *, policy=None):
    policy = policy or sys.modules[__name__]
    plan = policy.load_plan(args.plan, study, args.source_sha, args.tooling_sha)
    if provenance != {"tooling": plan["tooling"], "execution": plan["execution"]}:
        raise ValueError("Worker checkouts differ from the allocated plan")
    parent_args = policy.parent_arguments(plan)
    evidence, reused = policy.parent_evidence(parent_args, study, publication)
    if (evidence != plan["parent"] or reused != plan["reused"]
            or plan["inputs"] != {key: study.bind(getattr(parent_args, key)) for key in ("manifest", "reference_index")}
            or plan["smoke_root"] != str(parent_args.smoke_root)
            or plan["checkpoint_root"] != (str(parent_args.checkpoint_root) if parent_args.checkpoint_root else None)):
        raise ValueError("Worker parent, inputs or reused baselines differ")
    study.verify_smokes(parent_args.smoke_root, parent_args.manifest, args.source_sha)
    row = study.screen.select_checkpoint(parent_args.manifest, checkpoint_root=parent_args.checkpoint_root)
    if str(study.screen.select_reference(parent_args.reference_index, row, parent_args.manifest)) != plan["reference_bundle"]:
        raise ValueError("Worker paired reference differs")
    job = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
    if os.environ.get("SLURM_ARRAY_TASK_ID") != str(args.index):
        raise ValueError("Worker index differs from its allocated task")
    root = worker_root(plan, args.index, job)
    root.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        if (torch.__version__.split("+")[0] != "2.3.1" or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1 or torch.ones(1, device="cuda").sum().item() != 1):
            raise ValueError("Worker requires locked PyTorch 2.3.1 and one CUDA device")
        study.immutable_json(root / "runtime.json", {"python": sys.version, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "cuda_device_count": 1})
        study.immutable_json(root / "provenance.json", {"plan": study.bind(args.plan), "index": args.index,
            "job": job, **provenance})
        policy.evaluate_cell(plan, args.index, root / "smoke", study, smoke=True)
        # Each evaluation reconstructs and reseeds its controller. Release cyclic
        # model/workspace allocations from the smoke before constructing the full run.
        gc.collect()
        torch.cuda.empty_cache()
        policy.evaluate_cell(plan, args.index, root / "full", study, smoke=False)
        continuation.require_checkout(TOOLING_ROOT, args.tooling_sha)
        coordinator.require_source(args.source_sha)
        (root / "PASS").write_text("PASS\n")
    except Exception as error:
        coordinator.atomic_json(root / "FAILED", {"type": type(error).__name__, "message": str(error)}, overwrite=True)
        raise
    return {"output": str(root), "index": args.index}


def validate_worker(plan, index, job, study, *, policy=None):
    policy = policy or sys.modules[__name__]
    root = worker_root(plan, index, job)
    if (root / "FAILED").exists() or not (root / "PASS").is_file() or (root / "PASS").read_text().strip() != "PASS":
        raise ValueError("Submitted update-dose GPU task did not finish successfully")
    runtime = study.read(root / "runtime.json")
    if not runtime.get("gpu") or not runtime.get("torch", "").startswith("2.3.1") or runtime.get("cuda_device_count") != 1:
        raise ValueError("Missing allocated CUDA runtime evidence")
    provenance = study.read(root / "provenance.json")
    if provenance != {"plan": study.bind(Path(plan["result_root"]) / (plan.get("stage", "stage5") + "-plan.json")), "index": index,
                      "job": str(job), "tooling": plan["tooling"], "execution": plan["execution"]}:
        raise ValueError("Worker provenance differs from the exact submitted plan")
    for phase in ("smoke", "full"):
        directory = root / phase
        if not (directory / "PASS").is_file() or (directory / "PASS").read_text().strip() != "PASS":
            raise ValueError("Missing smoke or full evaluation completion")
        if study.read(directory / "validation.json") != policy.validate_bundle(plan, index, directory, study, phase == "smoke"):
            raise ValueError("Worker bundle changed after validation")
    episodes = study.read(root / "full/bundle/manifest.json")["runs"][0]["episodes"]
    return {"index": index, "condition": plan["conditions"][index], "output_path": str(root),
            "mean_return": statistics.mean(ep["return"] for ep in episodes),
            "episode_returns": {str(ep["seed"]): ep["return"] for ep in episodes},
            "validation": study.bind(root / "full/validation.json"),
            "bundle_manifest": study.bind(root / "full/bundle/manifest.json")}


def submit(args, plan_path, state, coordinator, study, *, policy=None):
    policy = policy or sys.modules[__name__]
    coordinator.require_source(args.source_sha)
    plan = policy.load_plan(plan_path, study, args.source_sha, args.tooling_sha)
    stage = plan.get("stage", "stage5")
    count = len(plan["conditions"])
    if type(args.max_concurrent) is not int or args.max_concurrent < 1:
        raise ValueError("max-concurrent must be a positive integer selected from live cluster capacity")
    inputs = {"plan": study.bind(plan_path), "worker_launcher": study.bind(args.worker_launcher)}
    if state.get("job"):
        if state.get("submission_intent", {}).get("inputs") != inputs or not str(state["job"]).isdigit():
            raise ValueError("Saved array does not match this exact update-dose plan")
        return state["job"]
    if state.get("submission_intent"):
        raise RuntimeError("Uncertain prior sbatch attempt; inspect the named job before retrying")
    state["submission_intent"] = {"inputs": inputs, "unix_time": time.time(),
                                  "job_name": "axqc-update-" + state["plan"]["sha256"][:8]}
    state_path = args.result_root / "coordinator-state.json"
    coordinator.atomic_json(state_path, state, overwrite=True)
    env = dict(os.environ)
    env.update(AMBI_EXECUTION_ROOT=str(args.execution_root), AMBI_SOURCE_SHA=args.source_sha,
               AMBI_TOOLING_SHA=args.tooling_sha, AMBI_TOOLING_ROOT=str(TOOLING_ROOT),
               AMBI_UPDATE_SWEEP_PLAN=str(plan_path), AMBIXQC_PYTHON=sys.executable)
    (args.result_root / "slurm").mkdir(exist_ok=True)
    gpu_options = (["--gres=gpu:1", "--prefer=l40s", "--constraint=a5000"]
                   if args.gpu_type == "prefer_l40s" else ["--gres=gpu:" + args.gpu_type + ":1"])
    command = ["sbatch", "--parsable", "--job-name=" + state["submission_intent"]["job_name"],
               f"--array=0-{count-1}%{min(count,args.max_concurrent)}", "--export=ALL",
               *gpu_options, "--cpus-per-task=6", "--mem=32G", "--time=" + getattr(policy, "WORKER_TIME", "06:00:00"),
               "--output=" + str(args.result_root / "slurm" / (stage + "-%A_%a.out")),
               "--error=" + str(args.result_root / "slurm" / (stage + "-%A_%a.err")), str(args.worker_launcher)]
    job = subprocess.check_output(command, env=env, text=True, timeout=60).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Unrecognized sbatch receipt; inspect the persisted intent before retrying")
    state["job"] = job
    coordinator.atomic_json(state_path, state, overwrite=True)
    print(json.dumps({"stage": stage, "submitted_job": job, "conditions": count,
                      "max_concurrent": min(count,args.max_concurrent)}), flush=True)
    return job


def label_curve(api, registry, cell, publication):
    label = f"475k shared UTD2 · return/running · J{cell['rounds']} G{cell['updates_per_round']} · all LR5e-5"
    # Run.update rewrites summary. A metadata-only mutation preserves even late
    # publication acknowledgements; refresh config before adding display metadata.
    api.flush()
    run = api.run(f"{publication.ENTITY}/{publication.PROJECT}/{registry['run_id']}")
    run.config.update(curve_label=label, update_sweep_id="ambixqc-jg-20260930")
    query = '''mutation LabelUpdateDoseRun($id:String!,$display_name:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,displayName:$display_name,config:$config}){bucket{id displayName}}}'''
    variables = {"id": run.storage_id, "display_name": label, "config": run.json_config}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    if response.get("upsertBucket", {}).get("bucket") != {"id": run.storage_id, "displayName": label}:
        raise RuntimeError("Update-dose display label was not acknowledged")


def publish_finished(args, plan, state, coordinator, study, publication, api, *, timeout=25200, policy=None):
    policy = policy or sys.modules[__name__]
    start = time.monotonic()
    pending = set(range(len(plan["conditions"])))
    observed = {index: set() for index in pending}
    entries, failures = {}, {}
    while pending:
        for index in sorted(pending):
            job = f"{state['job']}_{index}"
            if not coordinator.scheduler_done([job], observed[index]):
                continue
            pending.remove(index)
            try:
                coordinator.wait_jobs([job])
                entries[index] = policy.validate_worker(plan, index, state["job"], study)
                cell = plan["conditions"][index]
                registry = study.read(plan["runs"][cell["selector"]]["path"])
                receipt = publication.publish_curve(registry["run_dir"], cell["selector"], study.CHECKPOINT_SHA,
                                                    source_sha=args.source_sha)
                policy.label_curve(api, registry, cell, publication)
                state["published"][cell["selector"]] = receipt
                coordinator.atomic_json(args.result_root / "coordinator-state.json", state, overwrite=True)
                publication.update_progress(api, args.progress_run_id,
                    conditions_completed=plan.get("baseline_conditions", 9) + len(state["published"]),
                    episodes_completed=plan.get("baseline_episodes", 45) + 5 * len(state["published"]))
            except Exception as error:
                failures[str(index)] = {"type": type(error).__name__, "message": str(error)}
                state["failures"] = failures
                coordinator.atomic_json(args.result_root / "coordinator-state.json", state, overwrite=True)
        if pending:
            if time.monotonic() - start >= timeout:
                raise TimeoutError("Update-dose array did not finish before the coordinator deadline")
            time.sleep(20)
    if failures:
        raise RuntimeError("Update-dose completion/publication failed: " + json.dumps(failures))
    state.pop("failures", None)
    return [entries[index] for index in range(len(plan["conditions"]))]


def coordinate(args, coordinator, study, publication, provenance, *, api):
    workspace = continuation.verify_workspace(api, args.workspace_spec, publication)
    expected = {"schema": SCHEMA, "source_sha": args.source_sha, **provenance,
                "parent_root": str(args.parent_root), "previous_stage_root": str(args.previous_stage_root),
                "rounds": checked_rounds(args.rounds), "max_concurrent": args.max_concurrent,
                "workspace_spec": study.bind(args.workspace_spec), "progress_run_id": args.progress_run_id,
                "gpu_type": args.gpu_type, "worker_launcher": study.bind(args.worker_launcher)}
    state_path = args.result_root / "coordinator-state.json"
    state = study.read(state_path) if state_path.exists() else {**expected, "published": {}}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Update-dose state belongs to different source, scope or publication inputs")
    coordinator.atomic_json(state_path, state, overwrite=True)
    plan_path, plan = prepare_plan(args, study, publication, provenance)
    load_plan(plan_path, study, args.source_sha, args.tooling_sha)
    state["plan"] = study.bind(plan_path)
    for cell in plan["conditions"]:
        label_curve(api, study.read(plan["runs"][cell["selector"]]["path"]), cell, publication)
    total_conditions, total_episodes = 9 + len(plan["conditions"]), 45 + 5*len(plan["conditions"])
    publication.update_progress(api, args.progress_run_id, phase="stage5", conditions_expected=total_conditions,
        episodes_expected=total_episodes, conditions_completed=9 + len(state["published"]),
        episodes_completed=45 + 5*len(state["published"]))
    submit(args, plan_path, state, coordinator, study)
    entries = publish_finished(args, plan, state, coordinator, study, publication, api)
    study.immutable_json(args.result_root / "stage5-results.json", {"schema": SCHEMA, "stage": "stage5",
        "plan": study.bind(plan_path), "source_sha": args.source_sha, "reused": plan["reused"], "entries": entries})
    critic.bindings_unchanged(plan, study)
    continuation.require_checkout(TOOLING_ROOT, args.tooling_sha)
    coordinator.require_source(args.source_sha)
    state["stage5_complete"] = True
    state["stage5_results"] = study.bind(args.result_root / "stage5-results.json")
    coordinator.atomic_json(state_path, state, overwrite=True)
    study.immutable_json(args.result_root / "COMPLETE.json", {**state, "workspace": workspace})
    publication.update_progress(api, args.progress_run_id, phase="complete", stage5_complete=1,
                                conditions_completed=total_conditions, episodes_completed=total_episodes)
    (args.result_root / "FAILED.json").unlink(missing_ok=True)
    return state


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or args.source_sha != SOURCE_SHA:
        raise ValueError("Use a scheduler allocation and original10852a8 experiment source")
    args.execution_root = args.execution_root.resolve()
    provenance = {"tooling": continuation.require_checkout(TOOLING_ROOT, args.tooling_sha),
                  "execution": continuation.require_checkout(args.execution_root, args.source_sha)}
    continuation.require_runtime(args.execution_root)
    coordinator, study, publication = continuation.execution_modules(args.execution_root)
    if args.command == "worker":
        args.plan = args.plan.resolve(strict=True)
        return worker(args, coordinator, study, publication, provenance)
    args.rounds = checked_rounds(args.rounds)
    if type(args.max_concurrent) is not int or args.max_concurrent < 1:
        raise ValueError("max-concurrent must be positive")
    for key in ("parent_root", "previous_stage_root", "result_root", "workspace_spec", "worker_launcher"):
        setattr(args, key, getattr(args, key).resolve())
    if (any(args.result_root.is_relative_to(root) or root.is_relative_to(args.result_root)
            for root in (args.parent_root, args.previous_stage_root))
            or any(args.result_root.is_relative_to(root) for root in (TOOLING_ROOT, args.execution_root))
            or not args.worker_launcher.is_relative_to(TOOLING_ROOT)):
        raise ValueError("Require separate result storage and the pinned tooling worker launcher")
    args.result_root.mkdir(parents=True, exist_ok=True)
    from utils.eval_series import _lock
    import wandb
    with _lock(args.result_root / ".update-sweep.lock", blocking=False):
        api = None
        try:
            api = wandb.Api(timeout=60)
            return coordinate(args, coordinator, study, publication, provenance, api=api)
        except Exception as error:
            coordinator.atomic_json(args.result_root / "FAILED.json", {"type": type(error).__name__,
                                    "message": str(error), **provenance}, overwrite=True)
            try:
                publication.update_progress(api, args.progress_run_id, phase="stage5_failed", failure_type=type(error).__name__)
            except Exception:
                pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("coordinate", "worker"):
        sub = commands.add_parser(command)
        sub.add_argument("--execution-root", type=Path, required=True)
        sub.add_argument("--source-sha", required=True)
        sub.add_argument("--tooling-sha", required=True)
        if command == "coordinate":
            for name in ("parent-root", "previous-stage-root", "result-root", "workspace-spec", "worker-launcher"):
                sub.add_argument("--" + name, type=Path, required=True)
            sub.add_argument("--progress-run-id", required=True)
            sub.add_argument("--gpu-type", default="nvidia_rtx_a5000",
                help="Slurm GPU type, or prefer_l40s for L40S preference with A5000 fallback")
            sub.add_argument("--rounds", nargs="+", type=int, default=list(ROUNDS))
            sub.add_argument("--max-concurrent", type=int, required=True)
        else:
            sub.add_argument("--plan", type=Path, required=True)
            sub.add_argument("--index", type=int, required=True)
    print(json.dumps(run(parser.parse_args(argv)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
