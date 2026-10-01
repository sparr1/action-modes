"""H2 G12 comparison separates critic dose from actor/temperature dose."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import run_ambixqc_h2_update_study as campaign
import run_ambixqc_horizon_j_sweep as rounds
import run_ambixqc_horizon_study as horizon
import run_ambixqc_update_sweep as sweep
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_horizon_j_sweep import new_plan as stage9_plan, completed_horizon
from test_ambixqc_horizon_study import horizon_plan, rewrite_trace
from test_ambixqc_actor_lr_study import actor_parent
from test_ambixqc_update_sweep import SCIENCE_SHA, TOOLING_SHA, prepared, bundle_fixture
from test_ambixqc_update_extension import completed_parent
from test_ambixqc_critic_lr_study import write
from utils.ambi_benchmark import atomic_json


def test_exact_two_recipes_resolve_distinct_actor_cadences(tmp_path):
    import gymnasium as gym
    import numpy as np
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext
    from utils.eval_series_data import planner_identity

    cells, reused = campaign.split_conditions(study)
    assert [cell["policy_delay"] for cell in cells] == [6, 3]
    assert reused == horizon.conditions(study)
    baseline = reused[0]
    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    path = write(tmp_path / "matrix.json", campaign.matrix_for(study, cells))
    identities = [planner_identity(study.expected_settings(baseline), {}, "AMBIXQC/AMBIXQC", "tanh_mean")]
    for cell in cells:
        cfg, old, delay = cell["settings"], baseline["settings"], cell["policy_delay"]
        assert {k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k)} == (
            {"inner_updates_per_round", "inner_policy_delay"} if delay == 6 else {"inner_updates_per_round"})
        assert cfg["inner_rollout_horizon"] == cfg["inner_rounds"] == 2
        assert cfg["inner_replay_capacity"] == 1024
        assert cfg["inner_actor_lr"] == cfg["inner_critic_lr"] == cfg["inner_temperature_lr"] == 5e-5
        assert campaign.expected_counts(cell) == {"critic": 24, "actor": 24//delay, "temperature": 24//delay,
                                                 "model_steps": 1024, "replay_draws": 6144}
        resolved = resolve_preset(path, cell["selector"], checkpoint_context=context)
        model = object.__new__(AMBIXQC)
        model.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32), spec=SimpleNamespace(max_episode_steps=500))
        model.run_params = resolved["algorithm_config"]
        model.custom_params = model.run_params["alg_params"]
        actual = model._build_cfg({**model.custom_params, "device": "cpu"})
        assert actual.inner_model_step_budget == actual.inner_replay_capacity == 1024
        assert actual.inner_critic_updates_per_action == 24 and actual.inner_policy_delay == delay
        assert actual.inner_actor_updates_per_action == actual.inner_temperature_updates_per_action == 24//delay
        identities.append(planner_identity(study.expected_settings(cell), {}, "AMBIXQC/AMBIXQC", "tanh_mean"))
    assert len({study.digest(identity) for identity in identities}) == 3


@pytest.fixture
def completed_rounds(stage9_plan, monkeypatch):
    args, path, plan, _, calls = stage9_plan
    plan["tooling"]["commit"] = campaign.PARENT_TOOLING_SHA
    plan["plan_sha256"] = study.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(path, plan)
    entries, published = [], {}
    for index, cell in enumerate(plan["conditions"]):
        entry = {"index": index, "condition": deepcopy(cell), "mean_return": 605.0 + index,
                 "episode_returns": {str(seed): float(seed + 500 + index) for seed in study.SEEDS}}
        entries.append(entry)
        run_path = Path(plan["runs"][cell["selector"]]["path"])
        registry = study.read(run_path)
        rid = f"stage9-record-{index}"
        record = write(run_path.parent / "records" / (rid + ".json"), {
            "episodes": [{"seed": seed, "return": entry["episode_returns"][str(seed)]} for seed in study.SEEDS]})
        saved = {"status": "published", "record_sha256": study.bind(record)["sha256"]}
        write(run_path.parent / "publication.json", {"records": {rid: saved}})
        receipt = {"selector": cell["selector"], "checkpoint_sha256": study.CHECKPOINT_SHA,
            "source_sha": SCIENCE_SHA, "run_id": registry["run_id"], "record_id": rid,
            "record_sha256": saved["record_sha256"], "accepted": 1, "published": 1}
        write(run_path.parent / "bn-study-publication-verified.json", receipt)
        published[cell["selector"]] = receipt
    result_path = write(args.result_root / "stage9-results.json", {
        "schema": rounds.SCHEMA, "stage": "stage9", "source_sha": SCIENCE_SHA,
        "plan": study.bind(path), "reused": plan["reused"], "entries": entries})
    state = {"schema": rounds.SCHEMA, "source_sha": SCIENCE_SHA,
        "tooling": plan["tooling"], "execution": plan["execution"], "plan": study.bind(path),
        "stage9_results": study.bind(result_path), "stage9_complete": True, "job": "789",
        "published": published, "completed_sweep_root": plan["completed_sweep_root"],
        "progress_run_id": plan["progress_run_id"]}
    write(args.result_root / "coordinator-state.json", state)
    write(args.result_root / "COMPLETE.json", {**state, "workspace": {"verified": True}})
    def validated_worker(value, index, job, unused):
        assert value == plan and job == "789"
        return deepcopy(entries[index])
    monkeypatch.setattr(rounds, "validate_worker", validated_worker)
    new_args = SimpleNamespace(**vars(args))
    new_args.completed_sweep_root = args.result_root
    new_args.result_root = args.result_root.parent / (args.result_root.name + "-stage10")
    new_args.result_root.mkdir()
    new_args.max_concurrent, new_args.gpu_type = 2, "l40s"
    return new_args, plan, entries, calls


def test_parent_verifies_all_four_results_but_reuses_only_exact_j2(completed_rounds):
    args, parent, _, _ = completed_rounds
    before = {p: p.read_bytes() for p in args.completed_sweep_root.rglob("*.json")}
    evidence, reused = campaign.parent_evidence(args, study, publication)
    assert reused == parent["reused"] and len(reused) == 1
    assert reused[0]["condition"] == horizon.conditions(study)[0]
    assert study.read(evidence["baseline_registry"]["path"])["run_id"] == "4b2d4437ffd84ca49093e303f794a11b"
    assert len(evidence["stage9"]["publication_receipts"]) == 4
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("problem", ["missing_complete", "incomplete", "source", "tooling", "failed", "plan_hash",
    "results_hash", "result_entry", "receipt", "published", "wrong_baseline", "inherited", "inputs", "reference"])
def test_parent_rejects_missing_incompatible_or_changed_evidence(completed_rounds, monkeypatch, problem):
    args, plan, _, _ = completed_rounds
    root = args.completed_sweep_root
    state = study.read(root / "coordinator-state.json")
    if problem == "missing_complete": (root / "COMPLETE.json").unlink()
    elif problem == "failed": write(root / "FAILED.json", {})
    elif problem == "plan_hash": write(root / "stage9-plan.json", {"changed": True})
    elif problem == "results_hash": write(root / "stage9-results.json", {"changed": True})
    elif problem == "wrong_baseline": monkeypatch.setattr(campaign, "BASELINE_RUN_ID", "wrong")
    elif problem == "receipt":
        run_dir = Path(plan["runs"][plan["conditions"][3]["selector"]]["path"]).parent
        receipt = study.read(run_dir / "bn-study-publication-verified.json"); receipt["published"] = 0
        write(run_dir / "bn-study-publication-verified.json", receipt)
    elif problem == "inherited": write(plan["parent"]["stage8"]["complete"]["path"], {"changed": True})
    elif problem == "inputs": write(plan["inputs"]["manifest"]["path"], {"changed": True})
    elif problem == "reference": monkeypatch.setattr(study.screen, "select_reference", lambda *a: "wrong")
    else:
        if problem == "incomplete": state["stage9_complete"] = False
        elif problem == "source": state["source_sha"] = "f"*40
        elif problem == "tooling": state["tooling"]["commit"] = "f"*40
        elif problem == "published": state["published"].pop(next(iter(state["published"])))
        else:
            result_path = root / "stage9-results.json"
            result = study.read(result_path); result["entries"][3]["mean_return"] += 1
            write(result_path, result); state["stage9_results"] = study.bind(result_path)
        write(root / "coordinator-state.json", state)
        write(root / "COMPLETE.json", {**state, "workspace": {}})
    with pytest.raises((ValueError, FileNotFoundError)): campaign.parent_evidence(args, study, publication)


@pytest.fixture
def update_plan(completed_rounds, monkeypatch):
    args, _, _, calls = completed_rounds
    calls["specs"].clear(); calls["allocations"].clear()
    def allocate(root, spec, stage, selector):
        assert stage == "stage10" and spec["selector"] == selector
        calls["allocations"].append(selector)
        directory = root / "registry" / selector.split("/")[1]
        study.immutable_json(directory / "run.json", {"run_id": selector.split("/")[1], "run_dir": str(directory)})
        return {"run_dir": str(directory)}
    monkeypatch.setattr(publication, "allocate_curve", allocate)
    provenance = {"tooling": {"commit": TOOLING_SHA}, "execution": {"commit": SCIENCE_SHA}}
    path, plan = campaign.prepare_plan(args, study, publication, provenance)
    return args, path, plan, provenance, calls


def test_preparation_allocates_two_new_identities_without_rerunning_control(update_plan):
    args, path, plan, _, calls = update_plan
    assert calls["specs"] == calls["allocations"] == [cell["selector"] for cell in campaign.conditions(study)]
    assert path.name == "stage10-plan.json" and len(plan["reused"]) == 1
    assert plan["environment_seeds"] == study.SEEDS and plan["controller_seed"] == 12345 and plan["max_steps"] == 500
    assert (plan["baseline_conditions"], plan["baseline_episodes"]) == (33, 165)
    assert campaign.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == plan
    assert len(list((args.result_root / "registry").glob("*/run.json"))) == 2


@pytest.mark.parametrize("problem", ["delay", "updates", "lr", "horizon", "capacity", "baseline", "allocate_baseline"])
def test_resigned_plan_cannot_change_fixed_recipe_or_reuse(update_plan, problem):
    _, path, plan, _, _ = update_plan
    keys = {"delay": "inner_policy_delay", "updates": "inner_updates_per_round", "lr": "inner_actor_lr",
            "horizon": "inner_rollout_horizon", "capacity": "inner_replay_capacity"}
    if problem in keys: plan["conditions"][0]["settings"][keys[problem]] = 999
    elif problem == "baseline": plan["parent"]["baseline_registry"] = next(iter(plan["runs"].values()))
    else: plan["runs"][campaign.baseline_condition(study)["selector"]] = next(iter(plan["runs"].values()))
    plan["plan_sha256"] = study.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    write(path, plan)
    with pytest.raises(ValueError): campaign.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


def update_bundle(tmp_path, delay):
    cell = next(c for c in campaign.conditions(study) if c["policy_delay"] == delay)
    plan, root, manifest = bundle_fixture(tmp_path, cell)
    actor_steps = 24//delay
    run = manifest["runs"][0]
    run["actual_optimizer_steps"].update(actor=actor_steps*6, temperature=actor_steps*6)
    for episode in run["episodes"]:
        episode["actual_optimizer_steps"].update(actor=actor_steps*3, temperature=actor_steps*3)
    write(root / "bundle/manifest.json", manifest)
    metrics = {"decision/inner_model_steps": 1024, "decision/inner_buffer_size": 1024,
        "decision/inner_buffer_capacity": 1024, "decision/inner_rollout_count": 512,
        "decision/inner_rollout_len_min": 2, "decision/inner_rollout_len_mean": 2,
        "decision/inner_rollout_len_max": 2, "decision/inner_rollout_len_std": 0,
        "decision/inner_termination_rate": 0, "decision/inner_terminal_bootstrap_outer": 1,
        "decision/inner_outer_terminal_boundary_rows": 512,
        "decision/inner_outer_terminal_policy_evaluations": 6144, "decision/inner_outer_terminal_q_evaluations": 6144,
        "decision/inner_outer_terminal_bootstrap_rows": 3047,
        "decision/inner_actor_optimizer_steps": actor_steps, "decision/inner_temperature_optimizer_steps": actor_steps,
        "decision/inner_policy_delay": delay}
    def update(row):
        row.update(actor_updates=actor_steps, temperature_updates=actor_steps)
        row["metrics"].update(metrics)
    for trace in (root / "bundle").glob("*.jsonl.gz"): rewrite_trace(trace, update)
    return plan, root


@pytest.mark.parametrize("delay", [6, 3])
@pytest.mark.parametrize("bad_metric", [None, "inner_policy_delay", "inner_actor_optimizer_steps",
    "inner_temperature_optimizer_steps", "inner_outer_terminal_boundary_rows", "inner_compile_fallback", "inner_reward_scale_delta"])
def test_validation_enforces_both_update_cadences_and_h2_boundary_contract(tmp_path, delay, bad_metric):
    plan, root = update_bundle(tmp_path, delay)
    if bad_metric:
        rewrite_trace(root / "bundle/seed-101.jsonl.gz", lambda row: row["metrics"].update({"decision/"+bad_metric: 999}))
        with pytest.raises(ValueError): campaign.validate_bundle(plan, 0, root, study, True)
    else:
        value = campaign.validate_bundle(plan, 0, root, study, True)
        assert value["model_steps_per_decision"] == 1024 and value["outer_terminal_boundary_rows_per_decision"] == 512
        assert value["critic_updates_per_decision"] == 24
        assert value["actor_updates_per_decision"] == value["temperature_updates_per_decision"] == 24//delay
        assert value["policy_delay"] == delay


def test_worker_callbacks_use_stage10_validator(monkeypatch):
    seen = []
    def called(*args, policy, **kwargs): seen.append(policy)
    monkeypatch.setattr(sweep, "evaluate_cell", called); monkeypatch.setattr(sweep, "validate_worker", called)
    campaign.evaluate_cell({}, 0, Path("unused"), study, smoke=True)
    campaign.validate_worker({}, 0, "123", study)
    assert seen == [campaign, campaign]


def test_submission_uses_two_l40s_tasks_and_is_idempotent(update_plan, monkeypatch):
    args, path, plan, _, _ = update_plan
    calls, state = [], {"plan": study.bind(path)}
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=0-1%2" in command and "--time=02:00:00" in command and "--gres=gpu:l40s:1" in command
        assert kwargs["env"]["AMBI_SOURCE_SHA"] == SCIENCE_SHA
        return "987;oscar\n"
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    assert campaign.submit(args, path, state, coordinator, study) == "987"
    assert campaign.submit(args, path, state, coordinator, study) == "987" and len(calls) == 1
    assert campaign.worker_root(plan, 1, "987") == args.result_root / "stage10/job987-task1"
    with pytest.raises(ValueError): campaign.worker_root(plan, 2, "987")


@pytest.mark.parametrize("failure", [False, True])
def test_each_acknowledged_new_condition_advances_publication_independently(update_plan, monkeypatch, failure):
    args, path, plan, provenance, _ = update_plan
    updates, published = [], []
    monkeypatch.setattr(campaign, "prepare_plan", lambda *a: (path, plan))
    monkeypatch.setattr(campaign.continuation, "verify_workspace", lambda *a: {"verified": True})
    monkeypatch.setattr(campaign.continuation, "require_checkout", lambda *a: {})
    def submit(args, plan_path, state, *unused): state["job"] = "987"; return "987"
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
    assert updates[0] == {"phase": "stage10", "conditions_expected": 35, "episodes_expected": 175,
                          "conditions_completed": 33, "episodes_completed": 165}
    assert updates[-1]["conditions_completed"] == (34 if failure else 35)
    assert updates[-1]["episodes_completed"] == (170 if failure else 175)
    assert (args.result_root / "COMPLETE.json").exists() is not failure


def test_labels_exclude_prior_panels_and_preserve_summary():
    class Run:
        storage_id = "storage"
        config = {"campaign_id": publication.CAMPAIGN, **{key: "old" for key in (
            "update_sweep_id", "actor_lr_study_id", "horizon_study_id", "horizon_j_study_id")}}
        @property
        def json_config(self): return json.dumps(self.config)
    run, writes = Run(), []
    class Service:
        def execute_graphql(self, query, variables):
            assert "summary" not in query.lower(); writes.append(variables)
            return {"upsertBucket": {"bucket": {"id": "storage", "displayName": variables["display_name"]}}}
    api = SimpleNamespace(_service_api=Service(), flush=lambda: None, run=lambda path: run)
    campaign.label_curve(api, {"run_id": "new"}, campaign.conditions(study)[0], publication)
    assert set(run.config) == {"campaign_id", "curve_label", "h2_update_study_id"}
    assert run.config["h2_update_study_id"] == "ambixqc-h2-updates-20261001"
    assert len(writes) == 1 and "H2 J2 G12 delay6" in writes[0]["display_name"]


def test_wrappers_match_driver_and_preserve_locked_environment():
    worker = study.ROOT / "slurm/run_ambixqc_h2_update_study_oscar.sbatch"
    coordinator = study.ROOT / "slurm/orchestrate_ambixqc_h2_update_study_oscar.sbatch"
    subprocess.run(["bash", "-n", str(worker), str(coordinator)], check=True)
    text = worker.read_text(); cpu = coordinator.read_text()
    for source in (text, cpu):
        assert "run_ambixqc_h2_update_study.py" in source
        assert "LOCK_SHA=f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6" in source
        assert "--untracked-files=all" in source
    assert "#SBATCH --time=02:00:00" in text and "#SBATCH --time=04:00:00" in cpu
    assert '--plan "$AMBI_UPDATE_SWEEP_PLAN" --index "$SLURM_ARRAY_TASK_ID"' in text
    assert "WANDB_MODE=disabled" in text and "WANDB_MODE=online" in cpu
