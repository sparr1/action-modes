"""Expanded J/G grid reuses a complete, acknowledged previous sweep."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_ambixqc_update_sweep as original
import run_ambixqc_update_extension as extension
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_update_sweep import (
    SCIENCE_SHA, TOOLING_SHA, bundle_fixture, coordinate, prepared,
)
from test_ambixqc_critic_lr_study import write
from utils.ambi_benchmark import atomic_json


ROUNDS = (1, 2, 4, 6, 8)
UPDATES = (3, 6, 9, 12)


def test_expanded_grid_adds_only_eight_cells_and_reuses_all_twelve_previous_cells():
    cells = extension.conditions(study)
    new, reused = extension.split_conditions(study)
    assert len(cells) == 20 and len({cell["selector"] for cell in cells}) == 20
    assert {coordinate(cell) for cell in cells} == {(j, g) for j in ROUNDS for g in UPDATES}
    assert len(new) == 8 and len(reused) == 12 and len(new)*len(study.SEEDS) == 40
    assert {coordinate(cell) for cell in new} == {(8, g) for g in UPDATES} | {(j, 12) for j in original.ROUNDS}
    assert reused == original.conditions(study)
    assert [(c["rounds"]*c["updates_per_round"], c["rounds"], c["updates_per_round"]) for c in new] == sorted(
        [(c["rounds"]*c["updates_per_round"], c["rounds"], c["updates_per_round"]) for c in new], reverse=True)


def test_expansion_changes_only_j_g_and_required_capacity_with_same_reset_recipe():
    baseline = study.condition("running", "return_return", 4, 5e-5)["settings"]
    varying = {"inner_rounds", "inner_updates_per_round", "inner_replay_capacity"}
    for cell in extension.conditions(study):
        cfg, (j, g) = cell["settings"], coordinate(cell)
        assert {k: v for k, v in cfg.items() if k not in varying} == {k: v for k, v in baseline.items() if k not in varying}
        assert cfg["inner_replay_capacity"] == max(1024, j*256)
        assert original.expected_counts(cell) == {"critic": j*g, "actor": j*g//3, "temperature": j*g//3,
                                                 "model_steps": 256*j, "replay_draws": 256*j*g}
    maximum = next(c for c in extension.conditions(study) if coordinate(c) == (8, 12))
    assert maximum["settings"]["inner_replay_capacity"] == 2048
    assert original.expected_counts(maximum) == {"critic": 96, "actor": 32, "temperature": 32,
                                               "model_steps": 2048, "replay_draws": 24576}


def test_expanded_configuration_resolves_actual_engine_counts_without_eviction(tmp_path):
    import gymnasium as gym
    import numpy as np
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext

    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    cells, _ = extension.split_conditions(study)
    path = write(tmp_path / "matrix.json", extension.matrix_for(study, cells))
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


@pytest.mark.parametrize("j,g", [(j, 12) for j in (1, 2, 4, 6)] + [(8, g) for g in UPDATES])
def test_new_cells_keep_exact_trace_optimizer_and_model_work_validation(tmp_path, j, g):
    cell = next(c for c in extension.conditions(study) if coordinate(c) == (j, g))
    plan, root, _ = bundle_fixture(tmp_path, cell)
    result = original.validate_bundle(plan, 0, root, study, True)
    assert result["critic_updates_per_decision"] == j*g
    assert result["actor_updates_per_decision"] == result["temperature_updates_per_decision"] == j*g//3
    assert result["model_steps_per_decision"] == 256*j


@pytest.fixture
def completed_parent(prepared, monkeypatch):
    """Real signed legacy plan/results/receipts; bound GPU validation is isolated."""
    args, path, plan, provenance, calls = prepared
    entries, published = [], {}
    for index, cell in enumerate(plan["conditions"]):
        entry = {"index": index, "condition": deepcopy(cell), "mean_return": 600.0 + index,
                 "episode_returns": {str(seed): float(seed + index) for seed in study.SEEDS}}
        entries.append(entry)
        run_path = Path(plan["runs"][cell["selector"]]["path"])
        registry = study.read(run_path)
        record_id = f"record-{index}"
        record_path = write(run_path.parent / "records" / (record_id + ".json"), {
            "episodes": [{"seed": seed, "return": entry["episode_returns"][str(seed)]} for seed in study.SEEDS]})
        saved = {"status": "published", "record_sha256": study.bind(record_path)["sha256"]}
        write(run_path.parent / "publication.json", {"records": {record_id: saved}})
        receipt = {"selector": cell["selector"], "checkpoint_sha256": study.CHECKPOINT_SHA,
                   "source_sha": SCIENCE_SHA, "run_id": registry["run_id"], "record_id": record_id,
                   "record_sha256": saved["record_sha256"], "accepted": 1, "published": 1}
        write(run_path.parent / "bn-study-publication-verified.json", receipt)
        published[cell["selector"]] = receipt
    result_path = write(args.result_root / "stage5-results.json", {
        "schema": original.SCHEMA, "stage": "stage5", "source_sha": SCIENCE_SHA,
        "plan": study.bind(path), "reused": plan["reused"], "entries": entries})
    state = {"schema": original.SCHEMA, "source_sha": SCIENCE_SHA, **provenance,
             "plan": study.bind(path), "stage5_results": study.bind(result_path),
             "stage5_complete": True, "job": "123", "published": published,
             "parent_root": str(args.parent_root), "previous_stage_root": str(args.previous_stage_root),
             "progress_run_id": args.progress_run_id, "rounds": list(original.ROUNDS)}
    write(args.result_root / "coordinator-state.json", state)
    write(args.result_root / "COMPLETE.json", {**state, "workspace": {"verified": True}})
    def validated_record(run_dir, selector, checkpoint_sha, source_sha):
        assert checkpoint_sha == study.CHECKPOINT_SHA and source_sha == SCIENCE_SHA
        registry = study.read(Path(run_dir) / "run.json")
        assert registry["run_id"] == selector.split("/")[1]
        records = study.read(Path(run_dir) / "publication.json")["records"]
        record_id, = records
        return registry, record_id, records[record_id]
    def validate_worker(value, index, job, unused_study):
        assert value == plan and str(job) == "123"
        return deepcopy(entries[index])
    monkeypatch.setattr(publication, "_validated_record", validated_record)
    monkeypatch.setattr(original, "validate_worker", validate_worker)
    def lower_parent(parent_args, *unused):
        for key in ("manifest", "reference_index", "smoke_root", "checkpoint_root"):
            setattr(parent_args, key, getattr(args, key))
        return plan["parent"], plan["reused"]
    monkeypatch.setattr(original, "parent_evidence", lower_parent)
    return args, path, plan, provenance, calls, entries


def test_completed_legacy_plan_still_loads_unchanged(completed_parent):
    args, path, plan, _, _, _ = completed_parent
    before = {p: p.read_bytes() for p in args.result_root.rglob("*.json")}
    assert original.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == plan
    assert len(plan["conditions"]) == 10 and len(plan["reused"]) == 2
    assert {p: p.read_bytes() for p in before} == before


def extension_args(completed_parent):
    parent_args = completed_parent[0]
    args = SimpleNamespace(**vars(parent_args))
    args.completed_sweep_root = parent_args.result_root
    args.result_root = parent_args.result_root.parent / (parent_args.result_root.name + "-extension")
    args.result_root.mkdir(exist_ok=True)
    args.gpu_type = "prefer_l40s"
    args.max_concurrent = 8
    return args


def test_parent_reuses_exact_twelve_results_without_mutating_previous_evidence(completed_parent):
    args = extension_args(completed_parent)
    before = {p: p.read_bytes() for p in args.completed_sweep_root.rglob("*.json")}
    evidence, reused = extension.parent_evidence(args, study, publication)
    assert len(reused) == 12
    assert [item["condition"] for item in reused] == original.conditions(study)
    old = completed_parent[2]
    expected = {item["condition"]["selector"]: item["result"] for item in old["reused"]}
    expected.update({entry["condition"]["selector"]: entry for entry in completed_parent[5]})
    for item in reused: assert item["result"] == expected[item["condition"]["selector"]]
    assert args.manifest.is_file() and args.reference_index.is_file()
    assert evidence
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("problem", ["missing_complete", "not_complete", "source", "state_diverged", "failures",
    "plan_hash", "matrix_hash", "result_hash", "receipt", "publication_status", "published", "episode_returns"])
def test_missing_or_corrupt_completed_parent_cannot_authorize_reuse(completed_parent, problem):
    args = extension_args(completed_parent)
    root, plan = args.completed_sweep_root, completed_parent[2]
    state = study.read(root / "coordinator-state.json")
    run = Path(next(iter(plan["runs"].values()))["path"]).parent
    if problem == "missing_complete": (root / "COMPLETE.json").unlink()
    elif problem == "plan_hash": write(root / "stage5-plan.json", {"changed": True})
    elif problem == "matrix_hash": write(plan["matrix"]["path"], {"changed": True})
    elif problem == "result_hash": write(root / "stage5-results.json", {"changed": True})
    elif problem == "receipt":
        value = study.read(run / "bn-study-publication-verified.json")
        value["published"] = 0
        write(run / "bn-study-publication-verified.json", value)
    elif problem == "publication_status":
        path = run / "publication.json"
        value = study.read(path)
        next(iter(value["records"].values()))["status"] = "pending"
        write(path, value)
    elif problem == "episode_returns":
        path = next((run / "records").glob("*.json"))
        value = study.read(path); value["episodes"][0]["return"] += 1
        write(path, value)
    else:
        if problem == "not_complete": state["stage5_complete"] = False
        elif problem == "source": state["source_sha"] = "f"*40
        elif problem == "failures": state["failures"] = {"0": "GPU failed"}
        elif problem == "published": state["published"] = {}
        else: state["changed_after_completion"] = True
        write(root / "coordinator-state.json", state)
        if problem != "state_diverged": write(root / "COMPLETE.json", {**state, "workspace": {}})
    with pytest.raises((ValueError, FileNotFoundError)):
        extension.parent_evidence(args, study, publication)


@pytest.fixture
def extension_plan(completed_parent, monkeypatch):
    args = extension_args(completed_parent)
    provenance, calls = completed_parent[3:5]
    calls["specs"].clear(); calls["allocations"].clear()
    def allocate(root, spec, stage, selector):
        assert stage == "stage6" and spec["selector"] == selector
        calls["allocations"].append(selector)
        directory = root / "registry" / selector.split("/")[1]
        study.immutable_json(directory / "run.json", {"run_id": selector.split("/")[1], "run_dir": str(directory)})
        return {"run_dir": str(directory)}
    monkeypatch.setattr(publication, "allocate_curve", allocate)
    path, value = extension.prepare_plan(args, study, publication, provenance)
    return args, path, value, provenance, calls


def test_only_eight_new_identities_are_prepared_and_no_previous_controller_is_evaluated(extension_plan):
    args, path, value, _, calls = extension_plan
    new, reused = extension.split_conditions(study)
    expected = [cell["selector"] for cell in new]
    assert calls["specs"] == calls["allocations"] == expected
    assert not set(expected).intersection(cell["selector"] for cell in reused)
    assert path.name == "stage6-plan.json" and value["stage"] == "stage6"
    assert len(value["conditions"]) == 8 and len(value["reused"]) == 12
    assert (value["baseline_conditions"], value["baseline_episodes"]) == (19, 95)
    assert value["source_sha"] == SCIENCE_SHA != value["tooling"]["commit"]
    assert value["environment_seeds"] == [101, 102, 103, 104, 105]
    assert value["controller_seed"] == 12345 and value["max_steps"] == 500
    assert extension.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == value
    assert len(list((args.result_root / "registry").glob("*/run.json"))) == 8


@pytest.mark.parametrize("problem", ["source", "checkpoint", "settings", "capacity", "reuse", "seeds", "matrix", "manifest", "parent_hash", "registry", "allocate_reuse", "baseline"])
def test_extension_plan_rejects_resigned_changes_and_modified_parent_dependencies(extension_plan, problem):
    args, path, value, _, _ = extension_plan
    if problem == "source": value["source_sha"] = "c"*40
    elif problem == "checkpoint": value["checkpoint_sha256"] = "f"*64
    elif problem == "settings": value["conditions"][0]["settings"]["inner_policy_delay"] = 1
    elif problem == "capacity": value["conditions"][0]["settings"]["inner_replay_capacity"] = 1024
    elif problem == "reuse": value["reused"][0]["result"]["condition"]["settings"]["inner_actor_lr"] = 1.25e-5
    elif problem == "seeds": value["environment_seeds"] = [101, 102]
    elif problem == "matrix": write(value["matrix"]["path"], {"changed": True})
    elif problem == "manifest": write(value["inputs"]["manifest"]["path"], {"changed": True})
    elif problem == "parent_hash": write(args.completed_sweep_root / "COMPLETE.json", {"changed": True})
    elif problem == "allocate_reuse": value["runs"][value["reused"][0]["condition"]["selector"]] = next(iter(value["runs"].values()))
    elif problem == "baseline": value["baseline_conditions"] = 9
    else:
        keys = list(value["runs"]); value["runs"][keys[1]] = value["runs"][keys[0]]
    value["plan_sha256"] = study.digest({key: item for key, item in value.items() if key != "plan_sha256"})
    write(path, value)
    with pytest.raises(ValueError): extension.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


@pytest.mark.parametrize("gpu_type,flags", [
    ("prefer_l40s", ["--gres=gpu:1", "--prefer=l40s", "--constraint=a5000"]),
    ("l40s", ["--gres=gpu:l40s:1"]),
])
def test_extension_submits_only_eight_tasks_with_explicit_gpu_policy(extension_plan, monkeypatch, gpu_type, flags):
    args, path, value, _, _ = extension_plan
    args.gpu_type, args.max_concurrent = gpu_type, 20
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=0-7%8" in command
        assert [arg for arg in command if arg.startswith(("--gres=", "--prefer=", "--constraint="))] == flags
        assert "--output=" + str(args.result_root / "slurm/stage6-%A_%a.out") in command
        assert kwargs["env"]["AMBI_UPDATE_SWEEP_PLAN"] == str(path)
        assert kwargs["env"]["AMBI_SOURCE_SHA"] == SCIENCE_SHA
        return "456;oscar\n"
    monkeypatch.setattr(original.subprocess, "check_output", submit)
    assert extension.submit(args, path, state, coordinator, study) == "456"
    assert extension.submit(args, path, state, coordinator, study) == "456"
    assert len(calls) == 1
    assert extension.worker_root(value, 7, "456") == args.result_root / "stage6/job456-task7"
    with pytest.raises(ValueError): extension.worker_root(value, 8, "456")


def test_extension_submission_uncertainty_never_resends_gpu_work(extension_plan, monkeypatch):
    args, path, _, _, _ = extension_plan
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert study.read(args.result_root / "coordinator-state.json")["submission_intent"]["inputs"]["plan"] == study.bind(path)
        raise subprocess.TimeoutExpired(command, 60)
    monkeypatch.setattr(original.subprocess, "check_output", submit)
    with pytest.raises(subprocess.TimeoutExpired): extension.submit(args, path, state, coordinator, study)
    with pytest.raises(RuntimeError, match="Uncertain prior sbatch"): extension.submit(args, path, state, coordinator, study)
    assert len(calls) == 1


@pytest.mark.parametrize("fail_smoke", [False, True])
def test_extension_worker_uses_its_own_parent_and_smoke_gate(extension_plan, monkeypatch, fail_smoke):
    args, path, value, provenance, _ = extension_plan
    args.plan, args.index = path, 7
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setattr(original.continuation, "require_checkout", lambda *a: {})
    calls = []
    monkeypatch.setattr(original.gc, "collect", lambda: calls.append("gc"))
    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda index: "test GPU", empty_cache=lambda: calls.append("empty_cache"))
    fake_torch = SimpleNamespace(__version__="2.3.1+cu121", cuda=fake_cuda,
        ones=lambda *a, **k: SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    def evaluate(plan, index, root, study, *, smoke):
        assert plan["stage"] == "stage6" and plan["conditions"][index] == value["conditions"][7]
        calls.append((index, smoke))
        if fail_smoke: raise RuntimeError("smoke failed")
    monkeypatch.setattr(extension, "evaluate_cell", evaluate)
    coordinator = SimpleNamespace(atomic_json=atomic_json, require_source=lambda sha: None)
    root = extension.worker_root(value, 7, "456")
    if fail_smoke:
        with pytest.raises(RuntimeError, match="smoke failed"):
            extension.worker(args, coordinator, study, publication, provenance)
        assert calls == [(7, True)]
        assert (root / "FAILED").is_file() and not (root / "PASS").exists()
    else:
        extension.worker(args, coordinator, study, publication, provenance)
        assert calls == [(7, True), "gc", "empty_cache", (7, False)]
        assert (root / "PASS").is_file()


@pytest.mark.parametrize("problem", [None, "scheduler", "publication"])
def test_stage6_publishes_each_completed_cell_and_records_19_to_27_progress(extension_plan, monkeypatch, problem):
    args, path, value, provenance, _ = extension_plan
    events, updates, polls = [], [], {index: 0 for index in range(8)}
    monkeypatch.setattr(extension, "prepare_plan", lambda *a: (path, value))
    monkeypatch.setattr(extension.continuation, "verify_workspace", lambda *a: {"verified": True})
    monkeypatch.setattr(extension.continuation, "require_checkout", lambda *a: {})
    def submit(args, plan_path, state, *unused):
        state["job"] = "456"
        return "456"
    monkeypatch.setattr(extension, "submit", submit)
    def done(jobs, observed):
        index = int(jobs[0].split("_")[1]); polls[index] += 1
        return index == 7 or polls[index] > 1
    def wait(jobs):
        events.append(("wait", jobs[0]))
        if problem == "scheduler" and jobs == ["456_0"]: raise RuntimeError("GPU failed")
    def publish(run_dir, selector, checkpoint, **kwargs):
        events.append(("publish", selector))
        if problem == "publication" and selector == value["conditions"][0]["selector"]:
            raise RuntimeError("publication uncertain")
        return {"selector": selector, "accepted": 1, "published": 1}
    coordinator = SimpleNamespace(scheduler_done=done, wait_jobs=wait, atomic_json=atomic_json,
                                  require_source=lambda sha: None)
    monkeypatch.setattr(publication, "publish_curve", publish)
    monkeypatch.setattr(publication, "update_progress", lambda *a, **k: updates.append(k))
    monkeypatch.setattr(extension, "validate_worker", lambda plan, index, *a: {"index": index})
    monkeypatch.setattr(extension, "label_curve", lambda *a: None)
    monkeypatch.setattr(original.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    if problem:
        with pytest.raises(RuntimeError, match="completion/publication failed"):
            extension.coordinate(args, coordinator, study, publication, provenance, api=object())
    else:
        extension.coordinate(args, coordinator, study, publication, provenance, api=object())
    first_selector = value["conditions"][7]["selector"]
    assert events.index(("publish", first_selector)) < events.index(("sleep", 20)) < events.index(("wait", "456_0"))
    state = study.read(args.result_root / "coordinator-state.json")
    assert state["published"][first_selector]["published"] == 1
    assert updates[0] == {"phase": "stage6", "conditions_expected": 27, "episodes_expected": 135,
                          "conditions_completed": 19, "episodes_completed": 95}
    assert updates[1] == {"conditions_completed": 20, "episodes_completed": 100}
    if problem:
        assert len(state["published"]) == 7 and "0" in state["failures"]
        assert not (args.result_root / "COMPLETE.json").exists()
        assert not (args.result_root / "stage6-results.json").exists()
    else:
        complete = study.read(args.result_root / "COMPLETE.json")
        assert complete["stage6_complete"] is True
        assert updates[-1] == {"phase": "complete", "stage6_complete": 1,
                              "conditions_completed": 27, "episodes_completed": 135}
        results = study.read(args.result_root / "stage6-results.json")
        assert results["entries"] == [{"index": index} for index in range(8)]
        assert results["reused"] == value["reused"]
