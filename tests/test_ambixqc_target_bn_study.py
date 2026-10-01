"""One target-BN intervention keeps historical controls in their original namespace."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_ambixqc_target_bn_study as campaign
import run_ambixqc_h2_update_study as previous
import run_ambixqc_update_sweep as sweep
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_h2_update_study import completed_rounds, update_plan
from test_ambixqc_horizon_j_sweep import new_plan as stage9_plan, completed_horizon
from test_ambixqc_horizon_study import horizon_plan, h2_bundle, rewrite_trace
from test_ambixqc_actor_lr_study import actor_parent
from test_ambixqc_update_sweep import SCIENCE_SHA, TOOLING_SHA, prepared
from test_ambixqc_update_extension import completed_parent
from test_ambixqc_critic_lr_study import write
from utils.ambi_benchmark import atomic_json

NEW_SCIENCE_SHA = "c" * 40


@pytest.fixture(autouse=True)
def pinned_new_source(monkeypatch):
    monkeypatch.setattr(campaign, "SOURCE_SHA", NEW_SCIENCE_SHA)


def test_only_target_critic_bn_changes_and_control_is_not_rewritten():
    from utils.eval_series_data import planner_identity

    cells, reused = campaign.split_conditions(study)
    assert len(cells) == len(reused) == 1
    assert reused == campaign.horizon.conditions(study)
    new, old = cells[0], reused[0]
    assert new["selector"] != old["selector"]
    assert new["target_bn_mode"] == "running"
    assert "inner_critic_target_bn_mode" not in old["settings"]
    assert new["settings"] == {**old["settings"], "inner_critic_target_bn_mode": "running"}
    assert campaign.expected_counts(new) == {"critic": 12, "actor": 4, "temperature": 4,
                                              "model_steps": 1024, "replay_draws": 3072}
    old_identity = planner_identity(study.expected_settings(old), {}, "AMBIXQC/AMBIXQC", "tanh_mean")
    new_identity = planner_identity(study.expected_settings(new), {}, "AMBIXQC/AMBIXQC", "tanh_mean")
    assert old_identity != new_identity
    assert new_identity["settings"]["inner_critic_target_bn_mode"] == "running"
    assert "inner_critic_target_bn_mode" not in old_identity["settings"]
    matrix = campaign.matrix_for(study, cells)
    assert matrix["evaluation"]["default_presets"] == []
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 12345 and matrix["evaluation"]["max_steps"] == 500
    variants = matrix["comparisons"]["controller"]["variants"]
    assert set(variants) == {"prior", new["selector"].split("/")[1]}
    assert variants[new["selector"].split("/")[1]]["alg_params"] == new["settings"]


def test_new_matrix_resolves_running_target_bn_with_unchanged_counts(tmp_path):
    import gymnasium as gym
    import numpy as np
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext

    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    cell = campaign.conditions(study)[0]
    path = write(tmp_path / "matrix.json", campaign.matrix_for(study, [cell]))
    resolved = resolve_preset(path, cell["selector"], checkpoint_context=context)
    model = object.__new__(AMBIXQC)
    model.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32), spec=SimpleNamespace(max_episode_steps=500))
    model.run_params = resolved["algorithm_config"]
    model.custom_params = model.run_params["alg_params"]
    cfg = model._build_cfg({**model.custom_params, "device": "cpu"})
    assert cfg.inner_critic_target_bn_mode == "running"
    assert cfg.inner_actor_bn_mode == cfg.inner_critic_bn_mode == "running"
    assert cfg.inner_critic_updates_per_action == 12
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 1024
    assert cfg.inner_actor_lr == cfg.inner_critic_lr == cfg.inner_temperature_lr == 5e-5


@pytest.fixture
def completed_updates(update_plan, monkeypatch):
    args, path, plan, _, calls = update_plan
    plan["tooling"]["commit"] = campaign.PARENT_TOOLING_SHA
    plan["plan_sha256"] = study.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(path, plan)
    entries, published = [], {}
    for index, cell in enumerate(plan["conditions"]):
        entry = {"index": index, "condition": deepcopy(cell), "mean_return": 600.0 + index,
                 "episode_returns": {str(seed): float(seed + 500 + index) for seed in study.SEEDS}}
        entries.append(entry)
        run_path = Path(plan["runs"][cell["selector"]]["path"])
        registry = study.read(run_path)
        record_id = f"stage10-record-{index}"
        record = write(run_path.parent / "records" / (record_id + ".json"), {
            "episodes": [{"seed": seed, "return": entry["episode_returns"][str(seed)]} for seed in study.SEEDS]})
        saved = {"status": "published", "record_sha256": study.bind(record)["sha256"]}
        write(run_path.parent / "publication.json", {"records": {record_id: saved}})
        receipt = {"selector": cell["selector"], "checkpoint_sha256": study.CHECKPOINT_SHA,
            "source_sha": SCIENCE_SHA, "run_id": registry["run_id"], "record_id": record_id,
            "record_sha256": saved["record_sha256"], "accepted": 1, "published": 1}
        write(run_path.parent / "bn-study-publication-verified.json", receipt)
        published[cell["selector"]] = receipt
    results = write(args.result_root / "stage10-results.json", {
        "schema": previous.SCHEMA, "stage": "stage10", "source_sha": SCIENCE_SHA,
        "plan": study.bind(path), "reused": plan["reused"], "entries": entries})
    state = {"schema": previous.SCHEMA, "source_sha": SCIENCE_SHA,
        "tooling": plan["tooling"], "execution": plan["execution"], "plan": study.bind(path),
        "stage10_results": study.bind(results), "stage10_complete": True, "job": "890",
        "published": published, "completed_sweep_root": plan["completed_sweep_root"],
        "progress_run_id": plan["progress_run_id"]}
    write(args.result_root / "coordinator-state.json", state)
    write(args.result_root / "COMPLETE.json", {**state, "workspace": {"verified": True}})
    def validated_worker(value, index, job, unused):
        assert value == plan and job == "890"
        return deepcopy(entries[index])
    monkeypatch.setattr(previous, "validate_worker", validated_worker)
    new_args = SimpleNamespace(**vars(args))
    new_args.completed_sweep_root = args.result_root
    new_args.result_root = args.result_root.parent / (args.result_root.name + "-stage11")
    new_args.result_root.mkdir()
    new_args.source_sha = NEW_SCIENCE_SHA
    new_args.max_concurrent, new_args.gpu_type = 1, "prefer_l40s"
    return new_args, plan, entries, calls


def test_parent_checks_both_old_source_receipts_and_keeps_baseline_immutable(completed_updates):
    args, old, _, _ = completed_updates
    before = {p: p.read_bytes() for p in args.completed_sweep_root.rglob("*.json")}
    evidence, reused = campaign.parent_evidence(args, study, publication)
    assert reused == old["reused"] and len(reused) == 1
    assert evidence["source_sha"] == SCIENCE_SHA != args.source_sha
    assert len(evidence["stage10"]["publication_receipts"]) == 2
    assert study.read(evidence["baseline_registry"]["path"])["run_id"] == campaign.BASELINE_RUN_ID
    assert all(study.read(binding["path"])["source_sha"] == SCIENCE_SHA
               for binding in evidence["stage10"]["publication_receipts"])
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("problem", ["missing_complete", "incomplete", "source", "tooling", "failed", "plan_hash",
    "results_hash", "result_entry", "receipt", "published", "wrong_baseline", "inherited", "inputs", "reference"])
def test_parent_rejects_incomplete_or_changed_historical_evidence(completed_updates, monkeypatch, problem):
    args, plan, _, _ = completed_updates
    root = args.completed_sweep_root
    state = study.read(root / "coordinator-state.json")
    if problem == "missing_complete": (root / "COMPLETE.json").unlink()
    elif problem == "failed": write(root / "FAILED.json", {})
    elif problem == "plan_hash": write(root / "stage10-plan.json", {"changed": True})
    elif problem == "results_hash": write(root / "stage10-results.json", {"changed": True})
    elif problem == "wrong_baseline": monkeypatch.setattr(campaign, "BASELINE_RUN_ID", "wrong")
    elif problem == "receipt":
        directory = Path(plan["runs"][plan["conditions"][1]["selector"]]["path"]).parent
        receipt = study.read(directory / "bn-study-publication-verified.json")
        receipt["source_sha"] = NEW_SCIENCE_SHA
        write(directory / "bn-study-publication-verified.json", receipt)
    elif problem == "inherited": write(plan["parent"]["stage9"]["complete"]["path"], {"changed": True})
    elif problem == "inputs": write(plan["inputs"]["manifest"]["path"], {"changed": True})
    elif problem == "reference": monkeypatch.setattr(study.screen, "select_reference", lambda *a: "wrong")
    else:
        if problem == "incomplete": state["stage10_complete"] = False
        elif problem == "source": state["source_sha"] = NEW_SCIENCE_SHA
        elif problem == "tooling": state["tooling"]["commit"] = "f"*40
        elif problem == "published": state["published"].pop(next(iter(state["published"])))
        else:
            result_path = root / "stage10-results.json"
            result = study.read(result_path); result["entries"][1]["mean_return"] += 1
            write(result_path, result); state["stage10_results"] = study.bind(result_path)
        write(root / "coordinator-state.json", state)
        write(root / "COMPLETE.json", {**state, "workspace": {}})
    with pytest.raises((ValueError, FileNotFoundError)): campaign.parent_evidence(args, study, publication)


@pytest.fixture
def target_plan(completed_updates, monkeypatch):
    args, _, _, calls = completed_updates
    calls["specs"].clear(); calls["allocations"].clear(); calls["smokes"] = []
    monkeypatch.setattr(study, "verify_smokes", lambda *a: calls["smokes"].append(a[2]))
    def allocate(root, spec, stage, selector):
        assert stage == "stage11" and spec["selector"] == selector
        calls["allocations"].append(selector)
        directory = root / "registry" / selector.split("/")[1]
        study.immutable_json(directory / "run.json", {"run_id": selector.split("/")[1], "run_dir": str(directory)})
        return {"run_dir": str(directory)}
    monkeypatch.setattr(publication, "allocate_curve", allocate)
    provenance = {"tooling": {"commit": TOOLING_SHA}, "execution": {"commit": NEW_SCIENCE_SHA}}
    path, plan = campaign.prepare_plan(args, study, publication, provenance)
    return args, path, plan, provenance, calls


def test_one_new_registry_uses_new_source_and_original_smoke_namespace(target_plan):
    args, path, plan, _, calls = target_plan
    assert calls["specs"] == calls["allocations"] == [campaign.conditions(study)[0]["selector"]]
    assert calls["smokes"] and set(calls["smokes"]) == {SCIENCE_SHA}
    assert path.name == "stage11-plan.json" and len(plan["reused"]) == 1
    assert plan["source_sha"] == plan["execution"]["commit"] == NEW_SCIENCE_SHA
    assert plan["parent"]["source_sha"] == SCIENCE_SHA
    assert plan["environment_seeds"] == study.SEEDS and plan["controller_seed"] == 12345 and plan["max_steps"] == 500
    assert (plan["baseline_conditions"], plan["baseline_episodes"]) == (35, 175)
    assert campaign.load_plan(path, study, NEW_SCIENCE_SHA, TOOLING_SHA) == plan
    assert len(list((args.result_root / "registry").glob("*/run.json"))) == 1
    with pytest.raises(ValueError): campaign.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


@pytest.mark.parametrize("problem", ["target_bn", "actor_bn", "updates", "lr", "horizon", "capacity", "source", "baseline", "allocate_baseline"])
def test_resigned_plan_cannot_change_fixed_recipe_source_or_reuse(target_plan, problem):
    _, path, plan, _, _ = target_plan
    keys = {"target_bn": "inner_critic_target_bn_mode", "actor_bn": "inner_actor_bn_mode",
            "updates": "inner_updates_per_round", "lr": "inner_actor_lr",
            "horizon": "inner_rollout_horizon", "capacity": "inner_replay_capacity"}
    if problem in keys: plan["conditions"][0]["settings"][keys[problem]] = "changed"
    elif problem == "source": plan["source_sha"] = SCIENCE_SHA
    elif problem == "baseline": plan["parent"]["baseline_registry"] = next(iter(plan["runs"].values()))
    else: plan["runs"][campaign.baseline_condition(study)["selector"]] = next(iter(plan["runs"].values()))
    plan["plan_sha256"] = study.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(path, plan)
    with pytest.raises(ValueError): campaign.load_plan(path, study, NEW_SCIENCE_SHA, TOOLING_SHA)


@pytest.mark.parametrize("saved,evaluated,trace,valid", [(None, "running", 1, True), ("batch_no_update", "running", 1, True),
    ("running", "running", 1, False), (None, None, 1, False), (None, "batch_no_update", 1, False),
    (None, "running", None, False), (None, "running", 0, False)])
def test_bundle_requires_explicit_new_target_bn_and_historical_saved_default(tmp_path, saved, evaluated, trace, valid):
    plan, root, manifest = h2_bundle(tmp_path)
    cell = campaign.conditions(study)[0]
    plan.update(conditions=[cell], source_sha=NEW_SCIENCE_SHA)
    manifest["code"]["commit"] = NEW_SCIENCE_SHA
    run = manifest["runs"][0]
    run["selector"] = cell["selector"]
    run["result"]["resolved_config"] = study.expected_settings(cell)
    signature = run["result"]["checkpoint_evaluation_provenance"]
    signature.setdefault("saved_semantic_signature", {})
    for key, value in (("saved_semantic_signature", saved), ("evaluated_semantic_signature", evaluated)):
        signature[key].pop("inner_critic_target_bn_mode", None)
        if value is not None: signature[key]["inner_critic_target_bn_mode"] = value
    write(root / "bundle/manifest.json", manifest)
    if trace is not None:
        for path in (root / "bundle").glob("*.jsonl.gz"):
            rewrite_trace(path, lambda row: row["metrics"].update({"decision/inner_critic_target_bn_running": trace}))
    if valid:
        result = campaign.validate_bundle(plan, 0, root, study, True)
        assert result["target_bn_mode"] == "running"
        assert result["critic_updates_per_decision"] == 12
        assert result["actor_updates_per_decision"] == result["temperature_updates_per_decision"] == 4
        assert result["outer_terminal_boundary_rows_per_decision"] == 512
    else:
        with pytest.raises(ValueError, match="Target-BN"): campaign.validate_bundle(plan, 0, root, study, True)


def test_submission_uses_one_worker_and_new_source_once(target_plan, monkeypatch):
    args, path, plan, _, _ = target_plan
    calls, state = [], {"plan": study.bind(path)}
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=0-0%1" in command and "--time=02:00:00" in command
        assert "--gres=gpu:1" in command and "--prefer=l40s" in command and "--constraint=a5000" in command
        assert kwargs["env"]["AMBI_SOURCE_SHA"] == NEW_SCIENCE_SHA
        return "901;oscar\n"
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    assert campaign.submit(args, path, state, coordinator, study) == "901"
    assert campaign.submit(args, path, state, coordinator, study) == "901" and len(calls) == 1
    assert campaign.worker_root(plan, 0, "901") == args.result_root / "stage11/job901-task0"
    with pytest.raises(ValueError): campaign.worker_root(plan, 1, "901")


@pytest.mark.parametrize("failure", [False, True])
def test_only_acknowledged_new_result_advances_campaign_to_36_180(target_plan, monkeypatch, failure):
    args, path, plan, provenance, _ = target_plan
    updates, published = [], []
    monkeypatch.setattr(campaign, "prepare_plan", lambda *a: (path, plan))
    monkeypatch.setattr(campaign.continuation, "verify_workspace", lambda *a: {"verified": True})
    monkeypatch.setattr(campaign.continuation, "require_checkout", lambda *a: {})
    def submit(args, path, state, *unused): state["job"] = "901"; return "901"
    monkeypatch.setattr(campaign, "submit", submit)
    def publish(directory, selector, checkpoint, **kwargs):
        published.append(selector)
        if failure: raise RuntimeError("unacknowledged")
        return {"selector": selector, "accepted": 1, "published": 1}
    monkeypatch.setattr(publication, "publish_curve", publish)
    monkeypatch.setattr(publication, "update_progress", lambda *a, **k: updates.append(k))
    monkeypatch.setattr(campaign, "validate_worker", lambda *a: {"index": 0})
    monkeypatch.setattr(campaign, "label_curve", lambda *a: None)
    coordinator = SimpleNamespace(scheduler_done=lambda *a: True, wait_jobs=lambda *a: None,
                                  atomic_json=atomic_json, require_source=lambda sha: None)
    if failure:
        with pytest.raises(RuntimeError, match="completion/publication failed"):
            campaign.coordinate(args, coordinator, study, publication, provenance, api=object())
    else: campaign.coordinate(args, coordinator, study, publication, provenance, api=object())
    assert published == [plan["conditions"][0]["selector"]]
    assert updates[0] == {"phase": "stage11", "conditions_expected": 36, "episodes_expected": 180,
                          "conditions_completed": 35, "episodes_completed": 175}
    assert updates[-1]["conditions_completed"] == (35 if failure else 36)
    assert updates[-1]["episodes_completed"] == (175 if failure else 180)
    assert (args.result_root / "COMPLETE.json").exists() is not failure


def test_new_label_does_not_join_old_views_or_update_summary():
    class Run:
        storage_id = "new-storage"
        config = {"campaign_id": publication.CAMPAIGN, **{key: "old" for key in (
            "update_sweep_id", "actor_lr_study_id", "horizon_study_id", "horizon_j_study_id", "h2_update_study_id")}}
        @property
        def json_config(self): return json.dumps(self.config)
    run, calls = Run(), []
    class Service:
        def execute_graphql(self, query, variables):
            assert "summary" not in query.lower(); calls.append(variables)
            return {"upsertBucket": {"bucket": {"id": run.storage_id, "displayName": variables["display_name"]}}}
    api = SimpleNamespace(_service_api=Service(), flush=lambda: None, run=lambda path: run)
    campaign.label_curve(api, {"run_id": "new"}, campaign.conditions(study)[0], publication)
    assert set(run.config) == {"campaign_id", "curve_label", "target_bn_study_id"}
    assert run.config["target_bn_study_id"] == campaign.STUDY_ID
    assert len(calls) == 1 and "target BN running" in calls[0]["display_name"]


def test_new_evaluation_callback_uses_new_policy(monkeypatch):
    seen = []
    def called(*args, policy, **kwargs): seen.append(policy)
    monkeypatch.setattr(sweep, "evaluate_cell", called)
    campaign.evaluate_cell({}, 0, Path("unused"), study, smoke=True)
    assert seen == [campaign]
