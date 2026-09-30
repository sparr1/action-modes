"""J×G follow-up keeps science fixed, reuses controls and publishes completed cells."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_ambixqc_update_sweep as sweep
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_critic_lr_study import write
from test_ambixqc_inner_475k_screen import make_bundle
from utils.ambi_benchmark import atomic_json, solver_seed


SCIENCE_SHA = "10852a8cb6501f53b800cf50a774e732fac1be90"
TOOLING_SHA = "b" * 40
ROUNDS = (1, 2, 4, 6)
UPDATES = (3, 6, 9)


def coordinate(cell):
    cfg = cell["settings"]
    return cfg["inner_rounds"], cfg["inner_updates_per_round"]


def test_default_grid_only_changes_rounds_updates_and_required_replay_capacity():
    cells = sweep.conditions(study)
    assert len(cells) == 12
    assert len({cell["selector"] for cell in cells}) == 12
    assert {coordinate(cell) for cell in cells} == {(j, g) for j in ROUNDS for g in UPDATES}
    baseline = study.condition("running", "return_return", 4, 5e-5)["settings"]
    for cell in cells:
        cfg, (j, g) = cell["settings"], coordinate(cell)
        assert {key: value for key, value in cfg.items()
                if key not in {"inner_rounds", "inner_updates_per_round", "inner_replay_capacity"}} == {
                    key: value for key, value in baseline.items()
                    if key not in {"inner_rounds", "inner_updates_per_round", "inner_replay_capacity"}}
        assert cfg["inner_replay_capacity"] == max(1024, 256*j)
        assert cfg["inner_actor_lr"] == cfg["inner_critic_lr"] == cfg["inner_temperature_lr"] == 5e-5
        assert cfg["inner_actor_bn_mode"] == cfg["inner_critic_bn_mode"] == "running"
        assert cfg["inner_critic_source"] == cfg["inner_horizon_critic_source"] == "aux_return"
        assert cfg["inner_critic_target"] == "reward_only"
        assert cfg["inner_update_timing"] == "round" and cfg["inner_policy_delay"] == 3
        assert cfg["inner_rollout_horizon"] == 1
        assert cfg["inner_batch_size"] == cfg["inner_rollouts_per_round"] == 256


def test_only_exact_completed_j1_g3_and_j4_g3_are_reused():
    new, reused = sweep.split_conditions(study, ROUNDS)
    assert len(new) == 10 and len(reused) == 2
    assert {coordinate(cell) for cell in reused} == {(1, 3), (4, 3)}
    assert {coordinate(cell) for cell in new} == {(j, g) for j in ROUNDS for g in UPDATES} - {(1, 3), (4, 3)}
    assert len(new) * len(study.SEEDS) == 50
    for cell in reused:
        j, _ = coordinate(cell)
        assert cell["selector"] == study.condition("running", "return_return", j, 5e-5)["selector"]
        assert cell["settings"] == study.condition("running", "return_return", j, 5e-5)["settings"]
    assert [(c["rounds"]*c["updates_per_round"], c["rounds"], c["updates_per_round"]) for c in new] == sorted(
        [(c["rounds"]*c["updates_per_round"], c["rounds"], c["updates_per_round"]) for c in new], reverse=True)


@pytest.mark.parametrize("rounds,new_count,reused_count", [((1,), 2, 1), ((2,), 3, 0), ((4,), 2, 1),
                                                           ((6,), 3, 0), ((1, 4), 4, 2)])
def test_requested_round_subset_never_expands_the_grid(rounds, new_count, reused_count):
    cells = sweep.conditions(study, rounds)
    assert {coordinate(cell) for cell in cells} == {(j, g) for j in rounds for g in UPDATES}
    new, reused = sweep.split_conditions(study, rounds)
    assert (len(new), len(reused)) == (new_count, reused_count)


def test_each_condition_is_an_independent_copy():
    original = sweep.conditions(study)
    changed = sweep.conditions(study)
    changed[0]["settings"]["inner_critic_lr"] = 1.0
    assert changed[1:] == original[1:]
    assert sweep.conditions(study) == original
    assert study.BASE_SETTINGS["inner_critic_lr"] == 5e-5


@pytest.mark.parametrize("rounds", [(), (True,), (0,), (3,), (8,), (16,), (1, 1), (1.0,)])
def test_invalid_round_grid_is_rejected(rounds):
    with pytest.raises(ValueError): sweep.conditions(study, rounds)


def test_round_input_order_has_one_canonical_identity():
    assert sweep.conditions(study, (4, 1)) == sweep.conditions(study, (1, 4))


def test_count_contract_distinguishes_model_data_from_learning_dose():
    for cell in sweep.conditions(study):
        j, g = coordinate(cell)
        assert sweep.expected_counts(cell) == {"critic": j*g, "actor": j*g//3,
            "temperature": j*g//3, "model_steps": 256*j, "replay_draws": 256*j*g}


def test_matrix_only_allocates_new_conditions_and_keeps_episode_protocol():
    new, _ = sweep.split_conditions(study, ROUNDS)
    matrix = sweep.matrix_for(study, new)
    assert matrix["base_alg_config"] == "checkpoint"
    assert matrix["evaluation"]["default_presets"] == []
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 12345
    assert matrix["evaluation"]["max_steps"] == 500
    variants = matrix["comparisons"]["controller"]["variants"]
    assert set(variants) == {cell["selector"].split("/")[1] for cell in new} | {"prior"}
    for cell in new:
        assert variants[cell["selector"].split("/")[1]]["alg_params"] == cell["settings"]


def test_all_cells_resolve_actual_engine_counters_without_eviction(tmp_path):
    import gymnasium as gym
    import numpy as np
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext

    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    cells = sweep.conditions(study)
    path = write(tmp_path / "matrix.json", sweep.matrix_for(study, cells))
    for cell in cells:
        resolved = resolve_preset(path, cell["selector"], checkpoint_context=context)
        model = object.__new__(AMBIXQC)
        model.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32), spec=SimpleNamespace(max_episode_steps=500))
        model.run_params = resolved["algorithm_config"]
        model.custom_params = model.run_params["alg_params"]
        cfg = model._build_cfg({**model.custom_params, "device": "cpu"})
        j, g = coordinate(cell)
        assert cfg.inner_critic_updates_per_action == j*g
        assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == j*g//3
        assert cfg.inner_model_step_budget == 256*j
        assert cfg.inner_replay_capacity >= cfg.inner_model_step_budget
        assert cfg.inner_actor_lr == cfg.inner_critic_lr == cfg.inner_temperature_lr == 5e-5


def bundle_fixture(tmp_path, cell, *, smoke=True):
    root = tmp_path / "cell"
    seeds, steps = ([101, 102], 3) if smoke else (study.SEEDS, 500)
    bundle = root / "bundle"
    manifest = make_bundle(bundle, seeds=seeds, steps=steps)
    run = manifest["runs"][0]
    run["selector"] = cell["selector"]
    manifest["code"] = {"commit": SCIENCE_SHA, "dirty": False}
    run["result"]["resolved_config"] = study.expected_settings(cell)
    run["result"]["checkpoint_evaluation_provenance"]["evaluated_semantic_signature"]["inner_critic_bn_mode"] = "running"
    j, g = coordinate(cell)
    c, a, t = j*g, j*g//3, j*g//3
    run["actual_optimizer_steps"] = {"critic": c*steps*len(seeds), "actor": a*steps*len(seeds), "temperature": t*steps*len(seeds)}
    for episode in run["episodes"]:
        episode["actual_optimizer_steps"] = {"critic": c*steps, "actor": a*steps, "temperature": t*steps}
        episode["paired_return_delta"] = 0.0
        episode["solver_seed"] = solver_seed(12345, "episode", episode["seed"])
    for path in bundle.glob("*.jsonl.gz"):
        with gzip.open(path, "rt") as stream: rows = [json.loads(line) for line in stream]
        for row in rows:
            row.update(critic_updates=c, actor_updates=a, temperature_updates=t)
            row["metrics"].update({"decision/inner_model_steps": 256*j,
                "decision/inner_critic_optimizer_steps": c, "decision/inner_actor_optimizer_steps": a,
                "decision/inner_temperature_optimizer_steps": t, "decision/inner_replay_draws": 256*c,
                "decision/inner_requested_update_slots": c, "decision/inner_update_slots": c,
                "decision/inner_compile_fallback": 0,
                "decision/inner_buffer_capacity": max(1024, 256*j), "decision/inner_buffer_size": 256*j})
        with gzip.open(path, "wt") as stream:
            for row in rows: stream.write(json.dumps(row)+"\n")
    write(bundle / "manifest.json", manifest)
    plan = {"conditions": [cell], "source_sha": SCIENCE_SHA, "smoke_seeds": [101, 102],
            "smoke_max_steps": 3, "environment_seeds": study.SEEDS, "max_steps": 500,
            "reference_bundle": str(tmp_path / "prior")}
    return plan, root, manifest


@pytest.mark.parametrize("j,g", [(j, g) for j in ROUNDS for g in UPDATES])
def test_bundle_validation_accepts_exact_g_aware_work_for_every_cell(tmp_path, j, g):
    cell = next(cell for cell in sweep.conditions(study) if coordinate(cell) == (j, g))
    plan, root, _ = bundle_fixture(tmp_path, cell)
    result = sweep.validate_bundle(plan, 0, root, study, True)
    assert result["decisions"] == 6
    assert result["critic_updates_per_decision"] == j*g
    assert result["actor_updates_per_decision"] == result["temperature_updates_per_decision"] == j*g//3
    assert result["model_steps_per_decision"] == 256*j
    assert len(result["trace_sha256"]) == 2


@pytest.mark.parametrize("problem", ["source", "checkpoint", "critic_bn", "actor_bn", "frozen", "route", "lr", "seeds", "protocol",
                                     "run_counters", "episode_counters", "solver_seed"])
def test_bundle_validation_rejects_scientific_mismatch(tmp_path, problem):
    cell = next(cell for cell in sweep.conditions(study) if coordinate(cell) == (6, 9))
    plan, root, manifest = bundle_fixture(tmp_path, cell)
    run = manifest["runs"][0]
    result = run["result"]
    if problem == "source": manifest["code"]["commit"] = "f"*40
    elif problem == "checkpoint": manifest["checkpoint"]["sha256"] = "f"*64
    elif problem in {"critic_bn", "actor_bn"}:
        result["checkpoint_evaluation_provenance"]["evaluated_semantic_signature"]["inner_"+problem+"_mode"] = "batch_update"
    elif problem == "frozen": result["outer_state_unchanged"] = False
    elif problem == "route": result["resolved_config"]["inner_critic_source"] = "xqc"
    elif problem == "lr": result["resolved_config"]["inner_critic_lr"] = 1e-4
    elif problem == "seeds": run["episodes"].reverse()
    elif problem == "run_counters": run["actual_optimizer_steps"]["critic"] -= 1
    elif problem == "episode_counters": run["episodes"][0]["actual_optimizer_steps"]["actor"] -= 1
    elif problem == "solver_seed": run["episodes"][0]["solver_seed"] += 1
    else: manifest["protocol"]["controller_seed"] = 55
    write(root / "bundle/manifest.json", manifest)
    with pytest.raises(ValueError): sweep.validate_bundle(plan, 0, root, study, True)


@pytest.mark.parametrize("problem", ["old_g3_counts", "actor", "temperature", "model_steps", "scale", "route", "missing", "duplicate", "nonfinite",
                                     "fallback", "replay_draws", "eviction"])
def test_trace_work_alignment_frozen_scale_and_finiteness_are_mandatory(tmp_path, problem):
    cell = next(cell for cell in sweep.conditions(study) if coordinate(cell) == (4, 9))
    plan, root, _ = bundle_fixture(tmp_path, cell)
    trace = root / "bundle/seed-101.jsonl.gz"
    with gzip.open(trace, "rt") as stream: rows = [json.loads(line) for line in stream]
    if problem == "old_g3_counts": rows[0].update(critic_updates=12, actor_updates=4, temperature_updates=4)
    elif problem == "actor": rows[0]["actor_updates"] -= 1
    elif problem == "temperature": rows[0]["temperature_updates"] -= 1
    elif problem == "model_steps": rows[0]["metrics"]["decision/inner_model_steps"] += 256
    elif problem == "scale": rows[0]["metrics"]["decision/inner_reward_scale_delta"] = 1
    elif problem == "route": rows[0]["metrics"]["decision/inner_critic_source_aux_return"] = 0
    elif problem == "missing": rows.pop()
    elif problem == "duplicate": rows.append(deepcopy(rows[0]))
    elif problem == "fallback": rows[0]["metrics"]["decision/inner_compile_fallback"] = 1
    elif problem == "replay_draws": rows[0]["metrics"]["decision/inner_replay_draws"] -= 256
    elif problem == "eviction": rows[0]["metrics"]["decision/inner_buffer_size"] -= 256
    else: rows[0]["metrics"]["broken"] = float("nan")
    with gzip.open(trace, "wt") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")
    with pytest.raises(ValueError): sweep.validate_bundle(plan, 0, root, study, True)


def test_scientific_identities_are_unique_and_reused_controls_keep_original_identity():
    from utils.eval_series_data import planner_identity

    identities = [planner_identity(study.expected_settings(cell), {}, "AMBIXQC/AMBIXQC", "tanh_mean")
                  for cell in sweep.conditions(study)]
    assert len({study.digest(identity) for identity in identities}) == 12
    for cell in sweep.split_conditions(study, ROUNDS)[1]:
        prior = study.condition("running", "return_return", cell["rounds"], 5e-5)
        assert planner_identity(study.expected_settings(cell), {}, "AMBIXQC/AMBIXQC", "tanh_mean") == planner_identity(
            study.expected_settings(prior), {}, "AMBIXQC/AMBIXQC", "tanh_mean")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator

    args = SimpleNamespace(result_root=tmp_path, source_sha=SCIENCE_SHA, tooling_sha=TOOLING_SHA,
        execution_root=tmp_path / "science", parent_root=tmp_path / "stage3", previous_stage_root=tmp_path / "stage4",
        progress_run_id="progress", gpu_type="nvidia_rtx_a5000", rounds=list(ROUNDS), max_concurrent=3,
        manifest=write(tmp_path / "inventory.json", {}), reference_index=write(tmp_path / "reference.json", {}),
        smoke_root=tmp_path / "smokes", checkpoint_root=None,
        worker_launcher=write(tmp_path / "worker.sbatch", {}), workspace_spec=write(tmp_path / "workspace.json", {}))
    provenance = {"tooling": {"commit": TOOLING_SHA}, "execution": {"commit": SCIENCE_SHA}}
    reused = [{"condition": cell, "result": {
        "condition": study.condition("running", "return_return", cell["rounds"], 5e-5),
        "episode_returns": {str(seed): float(seed) for seed in study.SEEDS}}}
        for cell in sweep.split_conditions(study, ROUNDS)[1]]
    evidence = {"stage3": study.bind(write(tmp_path / "old3.json", {})),
                "stage4": study.bind(write(tmp_path / "old4.json", {}))}
    monkeypatch.setattr(sweep, "parent_evidence", lambda *a: (evidence, deepcopy(reused)))
    monkeypatch.setattr(study, "verify_smokes", lambda *a: {})
    monkeypatch.setattr(study.screen, "select_checkpoint", lambda *a, **k: {"path": "checkpoint", "source_run": "prior"})
    monkeypatch.setattr(study.screen, "select_reference", lambda *a: str(tmp_path / "prior"))
    calls = {"specs": [], "allocations": []}
    def evaluate(matrix_path, checkpoint, *, selectors, checkpoint_inventory, source_run, eval_series_spec_dir):
        assert checkpoint == "checkpoint" and checkpoint_inventory == args.manifest and source_run == "prior"
        assert len(selectors) == 1
        calls["specs"].extend(selectors)
        settings = study.read(matrix_path)["comparisons"]["controller"]["variants"][selectors[0].split("/")[1]]["alg_params"]
        path = write(eval_series_spec_dir / "identity.json", {"selector": selectors[0], "settings": settings})
        return {"specs": {selectors[0]: str(path)}}
    def allocate(root, spec, stage, selector):
        assert stage == "stage5" and spec["selector"] == selector
        calls["allocations"].append(selector)
        directory = root / "registry" / selector.split("/")[1]
        study.immutable_json(directory / "run.json", {"run_id": selector.split("/")[1], "run_dir": str(directory)})
        return {"run_dir": str(directory)}
    monkeypatch.setattr(evaluator, "evaluate_matrix", evaluate)
    monkeypatch.setattr(publication, "allocate_curve", allocate)
    path, value = sweep.prepare_plan(args, study, publication, provenance)
    return args, path, value, provenance, calls


def test_prepare_and_retry_allocate_only_ten_new_identities_and_preserve_baseline_results(prepared):
    args, path, value, provenance, calls = prepared
    new, reused = sweep.split_conditions(study, ROUNDS)
    expected = [cell["selector"] for cell in new]
    assert calls["specs"] == calls["allocations"] == expected
    assert not {cell["selector"] for cell in reused}.intersection(expected)
    assert len(value["conditions"]) == 10 and len(value["reused"]) == 2
    assert value["source_sha"] == SCIENCE_SHA != value["tooling"]["commit"]
    assert value["environment_seeds"] == [101, 102, 103, 104, 105]
    assert sweep.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == value
    assert sweep.prepare_plan(args, study, publication, provenance) == (path, value)
    assert calls["allocations"] == expected + expected
    assert len(list((args.result_root / "registry").glob("*/run.json"))) == 10


@pytest.mark.parametrize("problem", ["source", "checkpoint", "rounds", "settings", "capacity", "reuse", "seeds", "matrix", "manifest", "registry", "allocate_reuse"])
def test_plan_rejects_resigned_scope_changes_and_modified_dependencies(prepared, problem):
    _, path, value, _, _ = prepared
    if problem == "source": value["source_sha"] = "c"*40
    elif problem == "checkpoint": value["checkpoint_sha256"] = "f"*64
    elif problem == "rounds": value["rounds"] = [1, 2, 4]
    elif problem == "settings": value["conditions"][0]["settings"]["inner_policy_delay"] = 1
    elif problem == "capacity": value["conditions"][0]["settings"]["inner_replay_capacity"] = 1024
    elif problem == "reuse": value["reused"][0]["result"]["condition"]["settings"]["inner_actor_lr"] = 1.25e-5
    elif problem == "seeds": value["environment_seeds"] = [101, 102]
    elif problem == "matrix": write(value["matrix"]["path"], {"changed": True})
    elif problem == "manifest": write(value["inputs"]["manifest"]["path"], {"changed": True})
    elif problem == "allocate_reuse": value["runs"][value["reused"][0]["condition"]["selector"]] = next(iter(value["runs"].values()))
    else:
        keys = list(value["runs"]); value["runs"][keys[1]] = value["runs"][keys[0]]
    value["plan_sha256"] = study.digest({key: item for key, item in value.items() if key != "plan_sha256"})
    write(path, value)
    with pytest.raises(ValueError): sweep.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


def test_parent_gate_failure_never_allocates_more_curves(prepared, monkeypatch):
    args, _, _, provenance, calls = prepared
    before = list(calls["allocations"])
    def fail(*a): raise ValueError("parent publication incomplete")
    monkeypatch.setattr(sweep, "parent_evidence", fail)
    with pytest.raises(ValueError, match="parent publication"):
        sweep.prepare_plan(args, study, publication, provenance)
    assert calls["allocations"] == before


@pytest.mark.parametrize("index,job", [(True, "123"), (-1, "123"), (10, "123"), (.5, "123"), (0, "123_0"), (0, "")])
def test_worker_bounds_use_new_array_length_and_reject_other_tasks(prepared, index, job):
    value = prepared[2]
    assert sweep.worker_root(value, 9, "123").name == "job123-task9"
    with pytest.raises(ValueError): sweep.worker_root(value, index, job)


@pytest.mark.parametrize("limit,expected", [(3, "0-9%3"), (20, "0-9%10")])
@pytest.mark.parametrize("gpu_type,gpu_flags", [
    ("prefer_l40s", ["--gres=gpu:1", "--prefer=l40s", "--constraint=a5000"]),
    ("l40s", ["--gres=gpu:l40s:1"]),
    ("nvidia_rtx_a5000", ["--gres=gpu:nvidia_rtx_a5000:1"]),
])
def test_submission_uses_actual_array_size_and_requested_live_capacity(prepared, monkeypatch, limit, expected, gpu_type, gpu_flags):
    args, path, _, _, _ = prepared
    args.max_concurrent = limit
    args.gpu_type = gpu_type
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=" + expected in command
        assert [arg for arg in command if arg.startswith(("--gres=", "--prefer=", "--constraint="))] == gpu_flags
        assert "--cpus-per-task=6" in command and "--mem=32G" in command
        assert kwargs["env"]["AMBI_SOURCE_SHA"] == SCIENCE_SHA
        assert kwargs["env"]["AMBI_TOOLING_SHA"] == TOOLING_SHA
        assert kwargs["env"]["AMBI_UPDATE_SWEEP_PLAN"] == str(path)
        return "123;oscar\n"
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    assert sweep.submit(args, path, state, coordinator, study) == "123"
    assert sweep.submit(args, path, state, coordinator, study) == "123"
    write(args.worker_launcher, {"changed": True})
    with pytest.raises(ValueError): sweep.submit(args, path, state, coordinator, study)
    assert len(calls) == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_concurrency_fails_before_submission(prepared, monkeypatch, limit):
    args, path, _, _, _ = prepared
    args.max_concurrent = limit
    calls = []
    monkeypatch.setattr(sweep.subprocess, "check_output", lambda *a, **k: calls.append(a))
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    with pytest.raises(ValueError): sweep.submit(args, path, {"plan": study.bind(path)}, coordinator, study)
    assert calls == []


@pytest.mark.parametrize("failure", ["timeout", "bad_receipt"])
def test_uncertain_submission_is_durable_and_never_resent(prepared, monkeypatch, failure):
    args, path, _, _, _ = prepared
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        saved = study.read(args.result_root / "coordinator-state.json")
        assert saved["submission_intent"]["inputs"]["plan"] == study.bind(path)
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 60)
        return "uncertain receipt"
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)): sweep.submit(args, path, state, coordinator, study)
    with pytest.raises(RuntimeError, match="Uncertain prior sbatch"): sweep.submit(args, path, state, coordinator, study)
    assert len(calls) == 1


@pytest.mark.parametrize("fail_smoke", [False, True])
def test_worker_runs_exact_cell_smoke_before_full_and_releases_smoke_memory(prepared, monkeypatch, fail_smoke):
    args, path, value, provenance, _ = prepared
    args.plan, args.index = path, 9
    def parent_evidence(parent_args, *unused):
        parent_args.manifest = Path(value["inputs"]["manifest"]["path"])
        parent_args.reference_index = Path(value["inputs"]["reference_index"]["path"])
        parent_args.smoke_root = Path(value["smoke_root"])
        parent_args.checkpoint_root = None
        return value["parent"], value["reused"]
    monkeypatch.setattr(sweep, "parent_evidence", parent_evidence)
    monkeypatch.setattr(sweep.continuation, "require_checkout", lambda *a: {})
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "9")
    calls = []
    monkeypatch.setattr(sweep.gc, "collect", lambda: calls.append("gc"))
    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda index: "test GPU", empty_cache=lambda: calls.append("empty_cache"))
    fake_torch = SimpleNamespace(__version__="2.3.1+cu121", cuda=fake_cuda,
        ones=lambda *a, **k: SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    def evaluate(plan, index, root, study, *, smoke):
        calls.append((index, smoke))
        if fail_smoke: raise RuntimeError("smoke failed")
    monkeypatch.setattr(sweep, "evaluate_cell", evaluate)
    coordinator = SimpleNamespace(atomic_json=atomic_json, require_source=lambda sha: None)
    root = sweep.worker_root(value, 9, "123")
    if fail_smoke:
        with pytest.raises(RuntimeError, match="smoke failed"):
            sweep.worker(args, coordinator, study, publication, provenance)
        assert calls == [(9, True)]
        assert (root / "FAILED").is_file() and not (root / "PASS").exists()
    else:
        sweep.worker(args, coordinator, study, publication, provenance)
        assert calls == [(9, True), "gc", "empty_cache", (9, False)]
        assert (root / "PASS").is_file()


@pytest.mark.parametrize("problem", [None, "scheduler", "publication"])
def test_last_cell_publishes_while_earlier_tasks_run_and_partial_success_survives(prepared, monkeypatch, problem):
    args, _, value, _, _ = prepared
    events, updates, polls = [], [], {index: 0 for index in range(10)}
    state = {"job": "123", "published": {}}
    def done(jobs, observed):
        index = int(jobs[0].split("_")[1]); polls[index] += 1
        return index == 9 or polls[index] > 1
    def wait(jobs):
        events.append(("wait", jobs[0]))
        if problem == "scheduler" and jobs == ["123_0"]: raise RuntimeError("GPU failed")
    def publish(run_dir, selector, checkpoint, **kwargs):
        events.append(("publish", selector))
        if problem == "publication" and selector == value["conditions"][0]["selector"]:
            raise RuntimeError("publication uncertain")
        return {"selector": selector, "accepted": 1, "published": 1}
    coordinator = SimpleNamespace(scheduler_done=done, wait_jobs=wait, atomic_json=atomic_json)
    publisher = SimpleNamespace(publish_curve=publish, update_progress=lambda *a, **k: updates.append(k))
    monkeypatch.setattr(sweep, "validate_worker", lambda plan, index, *a: {"index": index})
    monkeypatch.setattr(sweep, "label_curve", lambda *a: None)
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    if problem:
        with pytest.raises(RuntimeError, match="completion/publication failed"):
            sweep.publish_finished(args, value, state, coordinator, study, publisher, object())
    else:
        assert sweep.publish_finished(args, value, state, coordinator, study, publisher, object()) == [
            {"index": index} for index in range(10)]
    first_selector = value["conditions"][9]["selector"]
    assert events.index(("publish", first_selector)) < events.index(("sleep", 20)) < events.index(("wait", "123_0"))
    assert state["published"][first_selector]["published"] == 1
    assert updates[0] == {"conditions_completed": 10, "episodes_completed": 50}
    assert study.read(args.result_root / "coordinator-state.json")["published"] == state["published"]
    if problem:
        assert len(state["published"]) == 9 and "0" in state["failures"]
        assert not (args.result_root / "COMPLETE.json").exists()
    else:
        assert updates[-1] == {"conditions_completed": 19, "episodes_completed": 95}


def test_label_refresh_preserves_published_summary_and_current_config(tmp_path, monkeypatch):
    """A metadata mutation must not rewrite even a concurrently advanced summary."""
    from wandb.apis.public import Run

    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    remote = {"id": "storage-id", "name": "run-id", "displayName": "old label",
        "state": "finished", "tags": [], "description": "", "notes": "", "group": "",
        "config": json.dumps({"campaign_id": {"value": publication.CAMPAIGN}}),
        "summaryMetrics": json.dumps({"study/status": "pending"}), "systemMetrics": "{}"}
    summary_writes, metadata_writes = [], []
    class Client:
        late_summary = None
        def execute(self, query, variable_values=None, **kwargs):
            values = variable_values or {}
            if "summaryMetrics" in values:
                summary_writes.append(values)
                remote["summaryMetrics"] = values["summaryMetrics"]
            elif "config" in values:
                if self.late_summary is not None: remote["summaryMetrics"] = json.dumps(self.late_summary)
                metadata_writes.append(values)
                remote.update(config=values["config"], displayName=values["display_name"])
            else:
                return {"project": {"run": deepcopy(remote)}}
            return {"upsertBucket": {"bucket": {"id": remote["id"], "displayName": remote["displayName"]}}}
    class Api:
        client = Client()
        def __init__(self): self.runs = {}
        def run(self, path):
            if path not in self.runs: self.runs[path] = Run(self.client, *path.split("/"), attrs=deepcopy(remote))
            return self.runs[path]
        def flush(self): self.runs.clear()
    api, registry = Api(), {"run_id": "run-id"}
    cell = next(cell for cell in sweep.conditions(study) if coordinate(cell) == (6, 9))
    sweep.label_curve(api, registry, cell, publication)
    published = {"study/status": "complete", "eval/return_mean": 612.5,
                 "publication/record_id": "accepted-record", "eval/paired_gain_mean": 60.5}
    remote["summaryMetrics"] = json.dumps(published)
    remote["config"] = json.dumps({"campaign_id": {"value": publication.CAMPAIGN},
                                   "publication_added": {"value": "preserve me"}})
    api.client.late_summary = {**published, "publication/late_ack": True}
    sweep.label_curve(api, registry, cell, publication)
    assert summary_writes == []
    assert json.loads(remote["summaryMetrics"]) == api.client.late_summary
    config = json.loads(remote["config"])
    assert config["publication_added"]["value"] == "preserve me"
    assert config["update_sweep_id"]["value"] == "ambixqc-jg-20260930"
    assert "J6 G9" in remote["displayName"] and "all LR5e-5" in remote["displayName"]
    assert len(metadata_writes) == 2
