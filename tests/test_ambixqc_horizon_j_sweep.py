"""H2 collection-round sweep preserves the completed J2 baseline and science."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_ambixqc_horizon_j_sweep as campaign
import run_ambixqc_horizon_study as horizon
import run_ambixqc_update_sweep as sweep
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_horizon_study import horizon_plan, h2_bundle, rewrite_trace
from test_ambixqc_actor_lr_study import actor_parent
from test_ambixqc_update_sweep import SCIENCE_SHA, TOOLING_SHA, prepared, bundle_fixture
from test_ambixqc_update_extension import completed_parent
from test_ambixqc_critic_lr_study import write
from utils.ambi_benchmark import atomic_json


def test_exact_four_new_recipes_preserve_j2_and_only_change_rounds_and_capacity(tmp_path):
    import gymnasium as gym
    import numpy as np
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext
    from utils.eval_series_data import planner_identity

    cells, reused = campaign.split_conditions(study)
    assert [cell["rounds"] for cell in cells] == [8, 6, 4, 1]
    assert reused == horizon.conditions(study)
    baseline = reused[0]
    assert campaign.BASELINE_RUN_ID == "4b2d4437ffd84ca49093e303f794a11b"
    assert campaign.PARENT_TOOLING_SHA == "dd0cd9a8ac5a054ffdd61cd3fa358a48dd6de5ef"
    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    path = write(tmp_path / "matrix.json", campaign.matrix_for(study, cells))
    identities = [planner_identity(study.expected_settings(baseline), {}, "AMBIXQC/AMBIXQC", "tanh_mean")]
    for cell in cells:
        j, cfg, old = cell["rounds"], cell["settings"], baseline["settings"]
        changed = {k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k)}
        assert changed == ({"inner_rounds"} if j == 1 else {"inner_rounds", "inner_replay_capacity"})
        assert cfg["inner_rollout_horizon"] == 2 and cfg["inner_updates_per_round"] == 6
        assert cfg["inner_actor_lr"] == cfg["inner_critic_lr"] == cfg["inner_temperature_lr"] == 5e-5
        assert cfg["inner_replay_capacity"] == max(1024, 512*j)
        assert campaign.expected_counts(cell) == {"critic": 6*j, "actor": 2*j, "temperature": 2*j,
                                                 "model_steps": 512*j, "replay_draws": 1536*j}
        resolved = resolve_preset(path, cell["selector"], checkpoint_context=context)
        model = object.__new__(AMBIXQC)
        model.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32), spec=SimpleNamespace(max_episode_steps=500))
        model.run_params = resolved["algorithm_config"]
        model.custom_params = model.run_params["alg_params"]
        actual = model._build_cfg({**model.custom_params, "device": "cpu"})
        assert actual.inner_model_step_budget == 512*j and actual.inner_replay_capacity >= 512*j
        assert actual.inner_critic_updates_per_action == 6*j
        assert actual.inner_actor_updates_per_action == actual.inner_temperature_updates_per_action == 2*j
        identities.append(planner_identity(study.expected_settings(cell), {}, "AMBIXQC/AMBIXQC", "tanh_mean"))
    assert len({study.digest(identity) for identity in identities}) == 5


@pytest.fixture
def completed_horizon(horizon_plan, monkeypatch):
    args, path, plan, provenance, calls = horizon_plan
    cell = campaign.baseline_condition(study)
    run_path = Path(plan["runs"][cell["selector"]]["path"])
    registry = study.read(run_path); registry["run_id"] = campaign.BASELINE_RUN_ID
    write(run_path, registry)
    plan["runs"][cell["selector"]] = study.bind(run_path)
    plan["tooling"]["commit"] = campaign.PARENT_TOOLING_SHA
    plan["plan_sha256"] = study.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(path, plan)
    entry = {"index": 0, "condition": deepcopy(cell), "mean_return": 603.0,
             "episode_returns": {str(seed): float(seed + 500) for seed in study.SEEDS}}
    record_id = "h2-j2-record"
    record_path = write(run_path.parent / "records" / (record_id + ".json"), {
        "episodes": [{"seed": seed, "return": entry["episode_returns"][str(seed)]} for seed in study.SEEDS]})
    saved = {"status": "published", "record_sha256": study.bind(record_path)["sha256"]}
    write(run_path.parent / "publication.json", {"records": {record_id: saved}})
    receipt = {"selector": cell["selector"], "checkpoint_sha256": study.CHECKPOINT_SHA,
        "source_sha": SCIENCE_SHA, "run_id": registry["run_id"], "record_id": record_id,
        "record_sha256": saved["record_sha256"], "accepted": 1, "published": 1}
    write(run_path.parent / "bn-study-publication-verified.json", receipt)
    result_path = write(args.result_root / "stage8-results.json", {
        "schema": horizon.SCHEMA, "stage": "stage8", "source_sha": SCIENCE_SHA,
        "plan": study.bind(path), "reused": plan["reused"], "entries": [entry]})
    state = {"schema": horizon.SCHEMA, "source_sha": SCIENCE_SHA,
        "tooling": plan["tooling"], "execution": plan["execution"], "plan": study.bind(path),
        "stage8_results": study.bind(result_path), "stage8_complete": True, "job": "456",
        "published": {cell["selector"]: receipt}, "completed_sweep_root": plan["completed_sweep_root"],
        "progress_run_id": plan["progress_run_id"]}
    write(args.result_root / "coordinator-state.json", state)
    write(args.result_root / "COMPLETE.json", {**state, "workspace": {"verified": True}})
    def validated_worker(value, index, job, unused):
        assert value == plan and index == 0 and job == "456"
        return deepcopy(entry)
    monkeypatch.setattr(horizon, "validate_worker", validated_worker)
    new_args = SimpleNamespace(**vars(args))
    new_args.completed_sweep_root = args.result_root
    new_args.result_root = args.result_root.parent / (args.result_root.name + "-stage9")
    new_args.result_root.mkdir()
    new_args.max_concurrent = 4
    return new_args, plan, entry, calls


def test_parent_verifies_and_reuses_exact_completed_stage8_without_mutation(completed_horizon):
    args, plan, entry, _ = completed_horizon
    before = {p: p.read_bytes() for p in args.completed_sweep_root.rglob("*.json")}
    evidence, reused = campaign.parent_evidence(args, study, publication)
    assert reused == [{"condition": campaign.baseline_condition(study), "result": entry}]
    assert study.read(evidence["baseline_registry"]["path"])["run_id"] == campaign.BASELINE_RUN_ID
    assert args.manifest.is_file() and args.reference_index.is_file()
    assert {p: p.read_bytes() for p in before} == before
    assert horizon.load_plan(args.completed_sweep_root / "stage8-plan.json", study,
                             SCIENCE_SHA, campaign.PARENT_TOOLING_SHA) == plan


@pytest.mark.parametrize("problem", ["missing_complete", "incomplete", "source", "tooling", "failed", "plan_hash",
    "results_hash", "result_entry", "receipt", "published", "wrong_run", "parent", "inputs", "reference"])
def test_parent_rejects_incompatible_incomplete_or_changed_evidence(completed_horizon, monkeypatch, problem):
    args, plan, _, _ = completed_horizon
    root = args.completed_sweep_root
    state = study.read(root / "coordinator-state.json")
    if problem == "missing_complete": (root / "COMPLETE.json").unlink()
    elif problem == "failed": write(root / "FAILED.json", {})
    elif problem == "plan_hash": write(root / "stage8-plan.json", {"changed": True})
    elif problem == "results_hash": write(root / "stage8-results.json", {"changed": True})
    elif problem == "wrong_run": monkeypatch.setattr(campaign, "BASELINE_RUN_ID", "wrong")
    elif problem == "receipt":
        run_dir = Path(plan["runs"][campaign.baseline_condition(study)["selector"]]["path"]).parent
        receipt = study.read(run_dir / "bn-study-publication-verified.json"); receipt["published"] = 0
        write(run_dir / "bn-study-publication-verified.json", receipt)
    elif problem == "parent": write(plan["parent"]["completed_sweep"]["stage5"]["complete"]["path"], {"changed": True})
    elif problem == "inputs": write(plan["inputs"]["manifest"]["path"], {"changed": True})
    elif problem == "reference": monkeypatch.setattr(study.screen, "select_reference", lambda *a: "wrong")
    else:
        if problem == "incomplete": state["stage8_complete"] = False
        elif problem == "source": state["source_sha"] = "f"*40
        elif problem == "tooling": state["tooling"]["commit"] = "f"*40
        elif problem == "published": state["published"] = {}
        else:
            result_path = root / "stage8-results.json"
            result = study.read(result_path); result["entries"][0]["mean_return"] += 1
            write(result_path, result); state["stage8_results"] = study.bind(result_path)
        write(root / "coordinator-state.json", state)
        write(root / "COMPLETE.json", {**state, "workspace": {}})
    with pytest.raises((ValueError, FileNotFoundError)): campaign.parent_evidence(args, study, publication)


@pytest.fixture
def new_plan(completed_horizon, monkeypatch):
    args, _, _, calls = completed_horizon
    calls["specs"].clear(); calls["allocations"].clear()
    def allocate(root, spec, stage, selector):
        assert stage == "stage9" and spec["selector"] == selector
        calls["allocations"].append(selector)
        directory = root / "registry" / selector.split("/")[1]
        study.immutable_json(directory / "run.json", {"run_id": selector.split("/")[1], "run_dir": str(directory)})
        return {"run_dir": str(directory)}
    monkeypatch.setattr(publication, "allocate_curve", allocate)
    provenance = {"tooling": {"commit": TOOLING_SHA}, "execution": {"commit": SCIENCE_SHA}}
    path, plan = campaign.prepare_plan(args, study, publication, provenance)
    return args, path, plan, provenance, calls


def test_preparation_allocates_four_new_identities_and_no_baseline(new_plan):
    args, path, plan, _, calls = new_plan
    selectors = [cell["selector"] for cell in campaign.conditions(study)]
    assert calls["specs"] == calls["allocations"] == selectors
    assert path.name == "stage9-plan.json" and plan["rounds"] == [1, 2, 4, 6, 8]
    assert plan["conditions"] == campaign.conditions(study) and len(plan["reused"]) == 1
    assert plan["reused"][0]["condition"] == horizon.conditions(study)[0]
    assert plan["environment_seeds"] == study.SEEDS and plan["controller_seed"] == 12345 and plan["max_steps"] == 500
    assert (plan["baseline_conditions"], plan["baseline_episodes"]) == (29, 145)
    assert campaign.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == plan
    assert len(list((args.result_root / "registry").glob("*/run.json"))) == 4


@pytest.mark.parametrize("problem", ["horizon", "lr", "capacity", "rounds", "allocate_baseline", "reuse", "baseline_registry"])
def test_resigned_plan_cannot_change_recipe_or_reuse(new_plan, problem):
    _, path, plan, _, _ = new_plan
    if problem == "horizon": plan["conditions"][0]["settings"]["inner_rollout_horizon"] = 1
    elif problem == "lr": plan["conditions"][0]["settings"]["inner_actor_lr"] = 1e-4
    elif problem == "capacity": plan["conditions"][0]["settings"]["inner_replay_capacity"] = 1024
    elif problem == "rounds": plan["conditions"].pop()
    elif problem == "reuse": plan["reused"][0]["condition"]["settings"]["inner_rounds"] = 4
    elif problem == "baseline_registry": plan["parent"]["baseline_registry"] = next(iter(plan["runs"].values()))
    else: plan["runs"][campaign.baseline_condition(study)["selector"]] = next(iter(plan["runs"].values()))
    plan["plan_sha256"] = study.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(path, plan)
    with pytest.raises(ValueError): campaign.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


@pytest.mark.parametrize("j", [1, 4, 6, 8])
def test_h2_validation_checks_real_work_for_each_round_count(tmp_path, j):
    cell = next(c for c in campaign.conditions(study) if c["rounds"] == j)
    plan, root, _ = bundle_fixture(tmp_path, cell)
    metrics = {"decision/inner_model_steps": 512*j, "decision/inner_buffer_size": 512*j,
        "decision/inner_buffer_capacity": max(1024, 512*j), "decision/inner_rollout_count": 256*j,
        "decision/inner_rollout_len_min": 2, "decision/inner_rollout_len_mean": 2,
        "decision/inner_rollout_len_max": 2, "decision/inner_rollout_len_std": 0,
        "decision/inner_termination_rate": 0, "decision/inner_terminal_bootstrap_outer": 1,
        "decision/inner_outer_terminal_boundary_rows": 256*j,
        "decision/inner_outer_terminal_policy_evaluations": 1536*j,
        "decision/inner_outer_terminal_q_evaluations": 1536*j,
        "decision/inner_outer_terminal_bootstrap_rows": 761*j}
    for trace in (root / "bundle").glob("*.jsonl.gz"):
        rewrite_trace(trace, lambda row: row["metrics"].update(metrics))
    result = campaign.validate_bundle(plan, 0, root, study, True)
    assert result["model_steps_per_decision"] == 512*j and result["outer_terminal_boundary_rows_per_decision"] == 256*j
    assert result["critic_updates_per_decision"] == 6*j and result["actor_updates_per_decision"] == 2*j
    rewrite_trace(root / "bundle/seed-101.jsonl.gz", lambda row: row["metrics"].update({"decision/inner_outer_terminal_boundary_rows": 512*j}))
    with pytest.raises(ValueError, match="H2 rollout"): campaign.validate_bundle(plan, 0, root, study, True)


def test_shared_h2_helper_preserves_old_validation_and_new_driver_rejects_reused_cell(tmp_path):
    plan, root, _ = h2_bundle(tmp_path)
    expected = horizon.validate_bundle(plan, 0, root, study, True)
    generic = sweep.validate_bundle(plan, 0, root, study, True)
    assert horizon.validate_h2_boundaries(plan, 0, root, study, generic) == expected
    with pytest.raises(ValueError, match="four new"): campaign.validate_bundle(plan, 0, root, study, True)


def test_evaluation_and_prepublication_both_use_new_h2_validator(monkeypatch):
    seen = []
    def called(*args, policy, **kwargs): seen.append(policy)
    monkeypatch.setattr(sweep, "evaluate_cell", called)
    monkeypatch.setattr(sweep, "validate_worker", called)
    campaign.evaluate_cell({}, 0, Path("unused"), study, smoke=True)
    campaign.validate_worker({}, 0, "123", study)
    assert seen == [campaign, campaign]


def test_submission_has_four_jobs_four_hour_limit_and_no_duplicate(new_plan, monkeypatch):
    args, path, plan, _, _ = new_plan
    calls, state = [], {"plan": study.bind(path)}
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=0-3%4" in command and "--time=04:00:00" in command
        assert kwargs["env"]["AMBI_SOURCE_SHA"] == SCIENCE_SHA
        return "789;oscar\n"
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    assert campaign.submit(args, path, state, coordinator, study) == "789"
    assert campaign.submit(args, path, state, coordinator, study) == "789" and len(calls) == 1
    assert campaign.worker_root(plan, 3, "789") == args.result_root / "stage9/job789-task3"
    with pytest.raises(ValueError): campaign.worker_root(plan, 4, "789")


@pytest.mark.parametrize("failure", [False, True])
def test_independent_publication_counts_only_acknowledged_new_cells(new_plan, monkeypatch, failure):
    args, path, plan, provenance, _ = new_plan
    updates, published = [], []
    monkeypatch.setattr(campaign, "prepare_plan", lambda *a: (path, plan))
    monkeypatch.setattr(campaign.continuation, "verify_workspace", lambda *a: {"verified": True})
    monkeypatch.setattr(campaign.continuation, "require_checkout", lambda *a: {})
    def submit(args, plan_path, state, *unused): state["job"] = "789"; return "789"
    monkeypatch.setattr(campaign, "submit", submit)
    def publish(run_dir, selector, checkpoint, **kwargs):
        published.append(selector)
        if failure and selector == plan["conditions"][1]["selector"]: raise RuntimeError("unacknowledged")
        return {"selector": selector, "accepted": 1, "published": 1}
    monkeypatch.setattr(publication, "publish_curve", publish)
    monkeypatch.setattr(publication, "update_progress", lambda *a, **k: updates.append(k))
    monkeypatch.setattr(campaign, "validate_worker", lambda plan, index, *a: {"index": index})
    monkeypatch.setattr(campaign, "label_curve", lambda *a: None)
    coordinator = SimpleNamespace(scheduler_done=lambda *a: True, wait_jobs=lambda *a: None,
                                  atomic_json=atomic_json, require_source=lambda sha: None)
    if failure:
        with pytest.raises(RuntimeError, match="completion/publication failed"):
            campaign.coordinate(args, coordinator, study, publication, provenance, api=object())
    else: campaign.coordinate(args, coordinator, study, publication, provenance, api=object())
    assert published == [cell["selector"] for cell in plan["conditions"]]
    assert updates[0] == {"phase": "stage9", "conditions_expected": 33, "episodes_expected": 165,
                          "conditions_completed": 29, "episodes_completed": 145}
    assert updates[-1]["conditions_completed"] == (32 if failure else 33)
    assert updates[-1]["episodes_completed"] == (160 if failure else 165)
    assert (args.result_root / "COMPLETE.json").exists() is not failure
    if not failure: assert study.read(args.result_root / "stage9-results.json")["reused"] == plan["reused"]


def test_labels_keep_j2_comparison_and_other_sweeps_separate():
    class Run:
        storage_id = "storage"
        config = {"campaign_id": publication.CAMPAIGN, "update_sweep_id": "old", "actor_lr_study_id": "old", "horizon_study_id": "old"}
        @property
        def json_config(self): return json.dumps(self.config)
    run, writes = Run(), []
    class Service:
        def execute_graphql(self, query, variables):
            assert "summary" not in query.lower()
            writes.append(variables)
            return {"upsertBucket": {"bucket": {"id": "storage", "displayName": variables["display_name"]}}}
    api = SimpleNamespace(_service_api=Service(), flush=lambda: None, run=lambda path: run)
    campaign.label_curve(api, {"run_id": "new"}, campaign.conditions(study)[0], publication)
    assert set(run.config) == {"campaign_id", "curve_label", "horizon_j_study_id"}
    assert run.config["horizon_j_study_id"] == "ambixqc-h2-j-20261001"
    assert len(writes) == 1 and "H2 J8 G6" in writes[0]["display_name"]
