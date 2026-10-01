"""One actor-rate intervention preserves its completed J2/G6 control."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_ambixqc_actor_lr_study as actor
import run_ambixqc_update_sweep as sweep
import run_ambixqc_update_extension as extension
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_update_sweep import SCIENCE_SHA, TOOLING_SHA, prepared, bundle_fixture
from test_ambixqc_update_extension import completed_parent, extension_args
from test_ambixqc_critic_lr_study import write
from utils.ambi_benchmark import atomic_json


PARENT_SHA = "cffd5b289e8f54b62dae137611deb8f82f134408"
BASELINE_ID = "1a763d57173e4efdb3d228d123232e2f"


def test_exact_one_actor_only_intervention_preserves_all_other_settings():
    baseline = next(c for c in sweep.conditions(study) if (c["rounds"], c["updates_per_round"]) == (2, 6))
    assert actor.baseline_condition(study) == baseline
    new, reused = actor.split_conditions(study)
    assert len(new) == 1 and reused == [baseline]
    assert new == actor.conditions(study)
    assert new[0]["selector"] != baseline["selector"]
    cfg, old = new[0]["settings"], baseline["settings"]
    assert {k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k)} == {"inner_actor_lr"}
    assert cfg["inner_actor_lr"] == 1e-4
    assert cfg["inner_critic_lr"] == cfg["inner_temperature_lr"] == 5e-5
    assert cfg["inner_replay_capacity"] == 1024
    assert cfg["inner_rollout_horizon"] == 1 and cfg["inner_updates_per_round"] == 6
    assert sweep.expected_counts(new[0]) == {"critic": 12, "actor": 4, "temperature": 4,
                                           "model_steps": 512, "replay_draws": 3072}
    assert actor.PARENT_TOOLING_SHA == PARENT_SHA and actor.BASELINE_RUN_ID == BASELINE_ID


def test_actual_engine_resolves_actor_only_learning_rate_change(tmp_path):
    import gymnasium as gym
    import numpy as np
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext

    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    cells = actor.conditions(study)
    path = write(tmp_path / "matrix.json", actor.matrix_for(study, cells))
    resolved = resolve_preset(path, cells[0]["selector"], checkpoint_context=context)
    model = object.__new__(AMBIXQC)
    model.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32), spec=SimpleNamespace(max_episode_steps=500))
    model.run_params = resolved["algorithm_config"]
    model.custom_params = model.run_params["alg_params"]
    cfg = model._build_cfg({**model.custom_params, "device": "cpu"})
    assert cfg.inner_actor_lr == 1e-4 and cfg.inner_critic_lr == cfg.inner_temperature_lr == 5e-5
    assert cfg.inner_critic_updates_per_action == 12
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4
    assert cfg.inner_model_step_budget == 512 and cfg.inner_replay_capacity == 1024
    plan, root, _ = bundle_fixture(tmp_path / "trace", cells[0])
    result = sweep.validate_bundle(plan, 0, root, study, True)
    assert result["critic_updates_per_decision"] == 12 and result["actor_updates_per_decision"] == 4


@pytest.fixture
def actor_parent(completed_parent, monkeypatch):
    args, plan_path, plan, provenance, calls, entries = completed_parent
    baseline = actor.baseline_condition(study)
    run_path = Path(plan["runs"][baseline["selector"]]["path"])
    registry = study.read(run_path); registry["run_id"] = BASELINE_ID
    write(run_path, registry)
    plan["runs"][baseline["selector"]] = study.bind(run_path)
    plan["tooling"]["commit"] = PARENT_SHA
    plan["plan_sha256"] = study.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(plan_path, plan)
    result_path = args.result_root / "stage5-results.json"
    results = study.read(result_path); results["plan"] = study.bind(plan_path)
    write(result_path, results)
    receipt_path = run_path.parent / "bn-study-publication-verified.json"
    receipt = study.read(receipt_path); receipt["run_id"] = BASELINE_ID
    write(receipt_path, receipt)
    state_path = args.result_root / "coordinator-state.json"
    state = study.read(state_path)
    state.update(plan=study.bind(plan_path), stage5_results=study.bind(result_path))
    state["tooling"]["commit"] = PARENT_SHA
    state["published"][baseline["selector"]] = receipt
    write(state_path, state)
    write(args.result_root / "COMPLETE.json", {**state, "workspace": {"verified": True}})
    def validated_record(run_dir, selector, checkpoint_sha, source_sha):
        assert checkpoint_sha == study.CHECKPOINT_SHA and source_sha == SCIENCE_SHA
        registry = study.read(Path(run_dir) / "run.json")
        records = study.read(Path(run_dir) / "publication.json")["records"]
        record_id, = records
        return registry, record_id, records[record_id]
    monkeypatch.setattr(publication, "_validated_record", validated_record)
    return completed_parent


def test_parent_reuses_exact_pinned_baseline_and_never_modifies_its_evidence(actor_parent):
    args = extension_args(actor_parent)
    before = {p: p.read_bytes() for p in args.completed_sweep_root.rglob("*.json")}
    evidence, reused = actor.parent_evidence(args, study, publication)
    assert len(reused) == 1 and reused[0]["condition"] == actor.baseline_condition(study)
    expected, = [e for e in actor_parent[5] if e["condition"] == actor.baseline_condition(study)]
    assert reused[0]["result"] == expected
    assert study.read(evidence["baseline_registry"]["path"])["run_id"] == BASELINE_ID
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("problem", ["missing_complete", "incomplete", "source", "wrong_run", "wrong_tooling", "receipt", "results_hash"])
def test_parent_rejects_missing_incompatible_or_unacknowledged_baseline(actor_parent, monkeypatch, problem):
    args = extension_args(actor_parent)
    root = args.completed_sweep_root
    state = study.read(root / "coordinator-state.json")
    baseline = actor.baseline_condition(study)
    run_path = Path(actor_parent[2]["runs"][baseline["selector"]]["path"])
    if problem == "missing_complete": (root / "COMPLETE.json").unlink()
    elif problem == "results_hash": write(root / "stage5-results.json", {"changed": True})
    elif problem == "receipt":
        path = run_path.parent / "bn-study-publication-verified.json"
        receipt = study.read(path); receipt["published"] = 0; write(path, receipt)
    elif problem in {"wrong_run", "wrong_tooling"}:
        # Keep lower evidence validation intact; mismatch the additional explicit pin.
        monkeypatch.setattr(actor, "BASELINE_RUN_ID" if problem == "wrong_run" else "PARENT_TOOLING_SHA", "wrong")
    else:
        if problem == "incomplete": state["stage5_complete"] = False
        else: state["source_sha"] = "c"*40
        write(root / "coordinator-state.json", state)
        write(root / "COMPLETE.json", {**state, "workspace": {}})
    with pytest.raises((ValueError, FileNotFoundError)): actor.parent_evidence(args, study, publication)


@pytest.fixture
def actor_plan(actor_parent, monkeypatch):
    args = extension_args(actor_parent)
    args.max_concurrent = 1
    provenance = {"tooling": {"commit": TOOLING_SHA}, "execution": {"commit": SCIENCE_SHA}}
    calls = actor_parent[4]; calls["specs"].clear(); calls["allocations"].clear()
    def allocate(root, spec, stage, selector):
        assert stage == "stage7" and spec["selector"] == selector
        calls["allocations"].append(selector)
        directory = root / "registry" / selector.split("/")[1]
        study.immutable_json(directory / "run.json", {"run_id": selector.split("/")[1], "run_dir": str(directory)})
        return {"run_dir": str(directory)}
    monkeypatch.setattr(publication, "allocate_curve", allocate)
    path, value = actor.prepare_plan(args, study, publication, provenance)
    return args, path, value, provenance, calls


def test_preparation_allocates_one_new_identity_and_keeps_five_paired_full_episodes(actor_plan):
    args, path, value, _, calls = actor_plan
    selector = actor.conditions(study)[0]["selector"]
    assert calls["specs"] == calls["allocations"] == [selector]
    assert path.name == "stage7-plan.json" and value["stage"] == "stage7"
    assert len(value["conditions"]) == len(value["reused"]) == 1
    assert (value["baseline_conditions"], value["baseline_episodes"]) == (27, 135)
    assert value["environment_seeds"] == [101, 102, 103, 104, 105]
    assert value["controller_seed"] == 12345 and value["max_steps"] == 500
    assert value["source_sha"] == SCIENCE_SHA != value["tooling"]["commit"]
    assert actor.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == value
    assert len(list((args.result_root / "registry").glob("*/run.json"))) == 1


@pytest.mark.parametrize("problem", ["actor", "critic", "temperature", "checkpoint", "source", "rounds", "seeds", "reuse", "parent", "allocate_baseline"])
def test_resigned_plan_cannot_change_the_single_controlled_intervention(actor_plan, problem):
    args, path, value, _, _ = actor_plan
    if problem in {"actor", "critic", "temperature"}: value["conditions"][0]["settings"][f"inner_{problem}_lr"] = 0.01
    elif problem == "checkpoint": value["checkpoint_sha256"] = "f"*64
    elif problem == "source": value["source_sha"] = "f"*40
    elif problem == "rounds": value["rounds"] = [4]
    elif problem == "seeds": value["environment_seeds"] = [101, 102]
    elif problem == "reuse": value["reused"][0]["result"]["condition"]["settings"]["inner_actor_lr"] = 1e-4
    elif problem == "parent": write(args.completed_sweep_root / "COMPLETE.json", {"changed": True})
    else: value["runs"][value["reused"][0]["condition"]["selector"]] = next(iter(value["runs"].values()))
    value["plan_sha256"] = study.digest({k: v for k, v in value.items() if k != "plan_sha256"})
    write(path, value)
    with pytest.raises(ValueError): actor.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


def test_submission_allocates_exactly_one_gpu_with_short_walltime_and_preferred_policy(actor_plan, monkeypatch):
    args, path, value, _, _ = actor_plan
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=0-0%1" in command and "--time=01:30:00" in command
        assert [arg for arg in command if arg.startswith(("--gres=", "--prefer=", "--constraint="))] == [
            "--gres=gpu:1", "--prefer=l40s", "--constraint=a5000"]
        assert kwargs["env"]["AMBI_SOURCE_SHA"] == SCIENCE_SHA
        assert kwargs["env"]["AMBI_UPDATE_SWEEP_PLAN"] == str(path)
        return "789;oscar\n"
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    assert actor.submit(args, path, state, coordinator, study) == "789"
    assert actor.submit(args, path, state, coordinator, study) == "789"
    assert len(calls) == 1
    assert actor.worker_root(value, 0, "789") == args.result_root / "stage7/job789-task0"
    with pytest.raises(ValueError): actor.worker_root(value, 1, "789")


def test_uncertain_submission_is_not_retransmitted(actor_plan, monkeypatch):
    args, path, _, _, _ = actor_plan
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert study.read(args.result_root / "coordinator-state.json")["submission_intent"]["inputs"]["plan"] == study.bind(path)
        raise subprocess.TimeoutExpired(command, 60)
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    with pytest.raises(subprocess.TimeoutExpired): actor.submit(args, path, state, coordinator, study)
    with pytest.raises(RuntimeError, match="Uncertain prior sbatch"): actor.submit(args, path, state, coordinator, study)
    assert len(calls) == 1


@pytest.mark.parametrize("problem", [None, "smoke", "source", "parent"])
def test_worker_executes_only_new_cell_after_smoke_source_and_parent_guards(actor_plan, monkeypatch, problem):
    args, path, value, provenance, _ = actor_plan
    args.plan, args.index = path, 0
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "789")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(sweep.continuation, "require_checkout", lambda *a: {})
    calls = []
    monkeypatch.setattr(sweep.gc, "collect", lambda: calls.append("gc"))
    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda index: "test GPU", empty_cache=lambda: calls.append("empty_cache"))
    fake_torch = SimpleNamespace(__version__="2.3.1+cu121", cuda=fake_cuda,
        ones=lambda *a, **k: SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    def evaluate(plan, index, root, study, *, smoke):
        assert plan["stage"] == "stage7" and plan["conditions"] == actor.conditions(study)
        calls.append((index, smoke))
        if problem == "smoke": raise RuntimeError("smoke failed")
    monkeypatch.setattr(actor, "evaluate_cell", evaluate)
    if problem == "source": args.source_sha = "f"*40
    elif problem == "parent": write(args.completed_sweep_root / "COMPLETE.json", {"changed": True})
    coordinator = SimpleNamespace(atomic_json=atomic_json, require_source=lambda sha: None)
    root = actor.worker_root(value, 0, "789")
    if problem:
        with pytest.raises((RuntimeError, ValueError)): actor.worker(args, coordinator, study, publication, provenance)
        assert calls == ([(0, True)] if problem == "smoke" else [])
        assert not (root / "PASS").exists()
    else:
        actor.worker(args, coordinator, study, publication, provenance)
        assert calls == [(0, True), "gc", "empty_cache", (0, False)]
        assert (root / "PASS").is_file()


@pytest.mark.parametrize("failure", [False, True])
def test_coordinator_publishes_one_cell_and_advances_27_to_28_only_after_ack(actor_plan, monkeypatch, failure):
    args, path, value, provenance, _ = actor_plan
    updates, published = [], []
    monkeypatch.setattr(actor, "prepare_plan", lambda *a: (path, value))
    monkeypatch.setattr(actor.continuation, "verify_workspace", lambda *a: {"verified": True})
    monkeypatch.setattr(actor.continuation, "require_checkout", lambda *a: {})
    def submit(args, plan_path, state, *unused): state["job"] = "789"; return "789"
    monkeypatch.setattr(actor, "submit", submit)
    def publish(run_dir, selector, checkpoint, **kwargs):
        published.append(selector)
        if failure: raise RuntimeError("publication uncertain")
        return {"selector": selector, "accepted": 1, "published": 1}
    coordinator = SimpleNamespace(scheduler_done=lambda *a: True, wait_jobs=lambda *a: None,
                                  atomic_json=atomic_json, require_source=lambda sha: None)
    monkeypatch.setattr(publication, "publish_curve", publish)
    monkeypatch.setattr(publication, "update_progress", lambda *a, **k: updates.append(k))
    monkeypatch.setattr(actor, "validate_worker", lambda plan, index, *a: {"index": index})
    monkeypatch.setattr(actor, "label_curve", lambda *a: None)
    if failure:
        with pytest.raises(RuntimeError, match="completion/publication failed"):
            actor.coordinate(args, coordinator, study, publication, provenance, api=object())
    else:
        actor.coordinate(args, coordinator, study, publication, provenance, api=object())
    assert published == [value["conditions"][0]["selector"]]
    assert updates[0] == {"phase": "stage7", "conditions_expected": 28, "episodes_expected": 140,
                          "conditions_completed": 27, "episodes_completed": 135}
    if failure:
        assert len(updates) == 1
        assert not (args.result_root / "COMPLETE.json").exists()
        assert not (args.result_root / "stage7-results.json").exists()
    else:
        assert updates[-1] == {"phase": "complete", "stage7_complete": 1,
                              "conditions_completed": 28, "episodes_completed": 140}
        assert study.read(args.result_root / "COMPLETE.json")["stage7_complete"] is True
        results = study.read(args.result_root / "stage7-results.json")
        assert results["entries"] == [{"index": 0}] and results["reused"] == value["reused"]


def test_actor_label_is_explicit_excludes_jg_sweep_and_never_rewrites_summary(tmp_path, monkeypatch):
    from wandb.apis.public import Run

    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    remote = {"id": "storage-id", "name": "run-id", "displayName": "old label",
        "state": "finished", "tags": [], "description": "", "notes": "", "group": "",
        "config": json.dumps({"campaign_id": {"value": publication.CAMPAIGN},
                              "update_sweep_id": {"value": "stale"}}),
        "summaryMetrics": json.dumps({"study/status": "complete", "eval/return_mean": 620}), "systemMetrics": "{}"}
    summary_writes, metadata_writes = [], []
    class Client:
        def execute(self, query, variable_values=None, **kwargs):
            values = variable_values or {}
            if "summaryMetrics" in values:
                summary_writes.append(values)
            elif "config" in values:
                remote["summaryMetrics"] = json.dumps({"study/status": "complete", "eval/return_mean": 620, "late_ack": True})
                metadata_writes.append(values)
                remote.update(config=values["config"], displayName=values["display_name"])
            else: return {"project": {"run": deepcopy(remote)}}
            return {"upsertBucket": {"bucket": {"id": remote["id"], "displayName": remote["displayName"]}}}
    class Api:
        client = Client()
        def __init__(self): self.runs = {}
        def run(self, path):
            if path not in self.runs: self.runs[path] = Run(self.client, *path.split("/"), attrs=deepcopy(remote))
            return self.runs[path]
        def flush(self): self.runs.clear()
    api = Api()
    actor.label_curve(api, {"run_id": "run-id"}, actor.conditions(study)[0], publication)
    assert summary_writes == [] and len(metadata_writes) == 1
    assert json.loads(remote["summaryMetrics"])["late_ack"] is True
    config = json.loads(remote["config"])
    assert "update_sweep_id" not in config
    assert config["actor_lr_study_id"]["value"] == "ambixqc-actor-lr-20261001"
    assert "J2" in remote["displayName"] and "G6" in remote["displayName"]
    assert "actor" in remote["displayName"].lower() and any(rate in remote["displayName"] for rate in ("1e-4", "0.0001"))
    assert "critic" in remote["displayName"].lower() and "temp" in remote["displayName"].lower()
