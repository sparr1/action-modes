"""A single H2 intervention keeps the H1 baseline and boundary evidence intact."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import run_ambixqc_actor_lr_study as actor
import run_ambixqc_horizon_study as horizon
import run_ambixqc_update_sweep as sweep
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_actor_lr_study import actor_parent
from test_ambixqc_update_sweep import SCIENCE_SHA, TOOLING_SHA, prepared, bundle_fixture
from test_ambixqc_update_extension import completed_parent, extension_args
from test_ambixqc_critic_lr_study import write
from utils.ambi_benchmark import atomic_json


def test_horizon_is_the_only_setting_change_and_has_its_own_scientific_identity(tmp_path):
    import gymnasium as gym
    import numpy as np
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext
    from utils.eval_series_data import planner_identity

    baseline = actor.baseline_condition(study)
    cells, reused = horizon.split_conditions(study)
    assert len(cells) == 1 and reused == [baseline]
    cell = cells[0]
    assert cell["selector"] != baseline["selector"]
    cfg, old = cell["settings"], baseline["settings"]
    assert {k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k)} == {"inner_rollout_horizon"}
    assert cfg["inner_rollout_horizon"] == 2 and cfg["inner_replay_capacity"] == 1024
    assert cfg["inner_actor_lr"] == cfg["inner_critic_lr"] == cfg["inner_temperature_lr"] == 5e-5
    assert horizon.expected_counts(cell) == {"critic": 12, "actor": 4, "temperature": 4,
                                           "model_steps": 1024, "replay_draws": 3072}
    identities = [planner_identity(study.expected_settings(c), {}, "AMBIXQC/AMBIXQC", "tanh_mean")
                  for c in (baseline, cell)]
    assert identities[0] != identities[1]
    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    path = write(tmp_path / "matrix.json", horizon.matrix_for(study, cells))
    resolved = resolve_preset(path, cell["selector"], checkpoint_context=context)
    model = object.__new__(AMBIXQC)
    model.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32), spec=SimpleNamespace(max_episode_steps=500))
    model.run_params = resolved["algorithm_config"]
    model.custom_params = model.run_params["alg_params"]
    actual = model._build_cfg({**model.custom_params, "device": "cpu"})
    assert actual.inner_rollout_horizon == 2 and actual.inner_model_step_budget == 1024
    assert actual.inner_critic_updates_per_action == 12
    assert actual.inner_actor_updates_per_action == actual.inner_temperature_updates_per_action == 4
    assert actual.inner_actor_lr == actual.inner_critic_lr == actual.inner_temperature_lr == 5e-5


@pytest.fixture
def horizon_plan(actor_parent, monkeypatch):
    args = extension_args(actor_parent)
    args.max_concurrent = 1
    provenance = {"tooling": {"commit": TOOLING_SHA}, "execution": {"commit": SCIENCE_SHA}}
    calls = actor_parent[4]; calls["specs"].clear(); calls["allocations"].clear()
    def allocate(root, spec, stage, selector):
        assert stage == "stage8" and spec["selector"] == selector
        calls["allocations"].append(selector)
        directory = root / "registry" / selector.split("/")[1]
        study.immutable_json(directory / "run.json", {"run_id": selector.split("/")[1], "run_dir": str(directory)})
        return {"run_dir": str(directory)}
    monkeypatch.setattr(publication, "allocate_curve", allocate)
    before = {p: p.read_bytes() for p in args.completed_sweep_root.rglob("*.json")}
    path, value = horizon.prepare_plan(args, study, publication, provenance)
    assert {p: p.read_bytes() for p in before} == before
    return args, path, value, provenance, calls


def test_plan_reuses_exact_h1_baseline_and_allocates_only_one_h2_condition(horizon_plan):
    args, path, value, _, calls = horizon_plan
    selector = horizon.conditions(study)[0]["selector"]
    assert calls["specs"] == calls["allocations"] == [selector]
    assert path.name == "stage8-plan.json" and value["stage"] == "stage8"
    assert len(value["conditions"]) == len(value["reused"]) == 1
    assert value["reused"][0]["condition"] == horizon.baseline_condition(study)
    assert study.read(value["parent"]["baseline_registry"]["path"])["run_id"] == "1a763d57173e4efdb3d228d123232e2f"
    assert value["environment_seeds"] == [101, 102, 103, 104, 105]
    assert value["controller_seed"] == 12345 and value["max_steps"] == 500
    assert value["smoke_seeds"] == [101, 102] and value["smoke_max_steps"] == 3
    assert (value["baseline_conditions"], value["baseline_episodes"]) == (28, 140)
    assert horizon.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == value
    assert len(list((args.result_root / "registry").glob("*/run.json"))) == 1


@pytest.mark.parametrize("problem", ["horizon", "actor", "critic", "temperature", "checkpoint", "source", "seeds", "reuse", "parent", "allocate_baseline"])
def test_resigned_plan_cannot_change_the_single_horizon_intervention(horizon_plan, problem):
    args, path, value, _, _ = horizon_plan
    if problem == "horizon": value["conditions"][0]["settings"]["inner_rollout_horizon"] = 1
    elif problem in {"actor", "critic", "temperature"}: value["conditions"][0]["settings"][f"inner_{problem}_lr"] = 1e-4
    elif problem == "checkpoint": value["checkpoint_sha256"] = "f"*64
    elif problem == "source": value["source_sha"] = "f"*40
    elif problem == "seeds": value["environment_seeds"] = [101, 102]
    elif problem == "reuse": value["reused"][0]["result"]["condition"]["settings"]["inner_rollout_horizon"] = 2
    elif problem == "parent": write(args.completed_sweep_root / "COMPLETE.json", {"changed": True})
    else: value["runs"][value["reused"][0]["condition"]["selector"]] = next(iter(value["runs"].values()))
    value["plan_sha256"] = study.digest({k: v for k, v in value.items() if k != "plan_sha256"})
    write(path, value)
    with pytest.raises(ValueError): horizon.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


def rewrite_trace(path, transform):
    with gzip.open(path, "rt") as stream: rows = [json.loads(line) for line in stream]
    for row in rows: transform(row)
    with gzip.open(path, "wt") as stream:
        for row in rows: stream.write(json.dumps(row) + "\n")


def h2_bundle(tmp_path, *, smoke=True):
    plan, root, manifest = bundle_fixture(tmp_path, horizon.conditions(study)[0], smoke=smoke)
    required = {"decision/inner_model_steps": 1024, "decision/inner_buffer_size": 1024,
        "decision/inner_rollout_count": 512, "decision/inner_rollout_len_min": 2,
        "decision/inner_rollout_len_mean": 2, "decision/inner_rollout_len_max": 2,
        "decision/inner_rollout_len_std": 0, "decision/inner_termination_rate": 0,
        "decision/inner_terminal_bootstrap_outer": 1, "decision/inner_outer_terminal_boundary_rows": 512,
        "decision/inner_outer_terminal_policy_evaluations": 3072,
        "decision/inner_outer_terminal_q_evaluations": 3072,
        "decision/inner_outer_terminal_bootstrap_rows": 1513}
    for path in (root / "bundle").glob("*.jsonl.gz"):
        rewrite_trace(path, lambda row: row["metrics"].update(required))
    return plan, root, manifest


@pytest.mark.parametrize("sampled", [0, 1513, 3072])
def test_h2_validation_checks_exact_rollout_work_and_inclusive_sample_bounds(tmp_path, sampled):
    plan, root, _ = h2_bundle(tmp_path)
    for path in (root / "bundle").glob("*.jsonl.gz"):
        rewrite_trace(path, lambda row: row["metrics"].update({"decision/inner_outer_terminal_bootstrap_rows": sampled}))
    value = horizon.validate_bundle(plan, 0, root, study, True)
    assert value["rollout_horizon"] == 2 and value["outer_terminal_boundary_rows_per_decision"] == 512
    assert value["model_steps_per_decision"] == 1024
    assert value["critic_updates_per_decision"] == 12 and value["actor_updates_per_decision"] == 4


@pytest.mark.parametrize("metric,value", [
    ("inner_model_steps", 512), ("inner_buffer_size", 512), ("inner_buffer_capacity", 512),
    ("inner_rollout_count", 1024), ("inner_rollout_len_min", 1), ("inner_rollout_len_mean", 1.5),
    ("inner_rollout_len_max", 3), ("inner_rollout_len_std", 1), ("inner_termination_rate", .1),
    ("inner_terminal_bootstrap_outer", 0), ("inner_outer_terminal_boundary_rows", 1024),
    ("inner_outer_terminal_policy_evaluations", 1536), ("inner_outer_terminal_q_evaluations", 1536),
    ("inner_outer_terminal_bootstrap_rows", -1), ("inner_outer_terminal_bootstrap_rows", 3073),
    ("inner_outer_terminal_bootstrap_rows", 1513.5), ("inner_outer_terminal_bootstrap_rows", None),
    ("inner_outer_terminal_bootstrap_rows", True), ("inner_compile_fallback", 1),
    ("inner_reward_scale_delta", 1), ("inner_reward_normalizer_imagined_updates", 1),
    ("inner_critic_target_reward_only", 0),
])
def test_h2_validation_rejects_wrong_boundary_h1_work_or_existing_invariant_failures(tmp_path, metric, value):
    plan, root, _ = h2_bundle(tmp_path)
    def corrupt(row):
        if value is None: row["metrics"].pop("decision/" + metric)
        else: row["metrics"]["decision/" + metric] = value
    rewrite_trace(root / "bundle/seed-101.jsonl.gz", corrupt)
    with pytest.raises(ValueError): horizon.validate_bundle(plan, 0, root, study, True)


def test_evaluate_cell_runs_policy_validation_before_staging_or_pass(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    import utils.ambi_benchmark as benchmark

    plan, source, _ = h2_bundle(tmp_path / "source")
    inventory = write(tmp_path / "inventory.json", {})
    plan.update(inputs={"manifest": study.bind(inventory)}, checkpoint_root=None,
        matrix={"path": str(tmp_path / "matrix.json")}, runs={plan["conditions"][0]["selector"]: {"path": str(tmp_path / "registry/run.json")}})
    row = {"path": "checkpoint.pt", "source_run": {}}
    monkeypatch.setattr(study.screen, "select_checkpoint", lambda *a, **k: row)
    def evaluate(*args, **kwargs):
        shutil.copytree(source / "bundle", kwargs["bundle_dir"])
        assert kwargs["selectors"] == [horizon.conditions(study)[0]["selector"]]
        return {"checkpoint_sha256": study.CHECKPOINT_SHA}
    monkeypatch.setattr(evaluator, "evaluate_matrix", evaluate)
    monkeypatch.setattr(benchmark, "stage_completed_bundle", lambda *a, **k: pytest.fail("Invalid bundle was staged"))
    calls = []
    def reject(*args): calls.append(args[-1]); raise ValueError("H2 policy rejected")
    monkeypatch.setattr(horizon, "validate_bundle", reject)
    root = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="H2 policy rejected"):
        horizon.evaluate_cell(plan, 0, root, study, smoke=False)
    assert calls == [False] and not (root / "PASS").exists()


def test_prepublication_worker_validation_rechecks_h2_boundaries(tmp_path, monkeypatch):
    import utils.ambi_benchmark as benchmark

    plan, smoke, _ = h2_bundle(tmp_path / "smoke-input")
    _, full, full_manifest = h2_bundle(tmp_path / "full-input", smoke=False)
    monkeypatch.setattr(benchmark, "reference_returns", lambda *a: {ep["seed"]: ep["return"] for ep in full_manifest["runs"][0]["episodes"]})
    plan.update(stage="stage8", result_root=str(tmp_path / "result"),
                tooling={"commit": TOOLING_SHA}, execution={"commit": SCIENCE_SHA})
    plan_path = write(Path(plan["result_root"]) / "stage8-plan.json", plan)
    root = horizon.worker_root(plan, 0, "789")
    root.mkdir(parents=True)
    write(root / "runtime.json", {"gpu": "test GPU", "torch": "2.3.1", "cuda_device_count": 1})
    write(root / "provenance.json", {"plan": study.bind(plan_path), "index": 0, "job": "789",
                                   "tooling": plan["tooling"], "execution": plan["execution"]})
    for phase, source in (("smoke", smoke), ("full", full)):
        shutil.copytree(source, root / phase)
        write(root / phase / "validation.json", horizon.validate_bundle(plan, 0, root / phase, study, phase == "smoke"))
        (root / phase / "PASS").write_text("PASS\n")
    (root / "PASS").write_text("PASS\n")
    assert horizon.validate_worker(plan, 0, "789", study)["condition"] == horizon.conditions(study)[0]
    rewrite_trace(root / "full/bundle/seed-101.jsonl.gz", lambda row: row["metrics"].update({"decision/inner_outer_terminal_boundary_rows": 1024}))
    with pytest.raises(ValueError, match="H2 rollout"):
        horizon.validate_worker(plan, 0, "789", study)


def test_submission_allocates_one_h2_worker_with_ninety_minute_limit(horizon_plan, monkeypatch):
    args, path, plan, _, _ = horizon_plan
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=0-0%1" in command and "--time=01:30:00" in command
        assert kwargs["env"]["AMBI_SOURCE_SHA"] == SCIENCE_SHA
        return "789;oscar\n"
    monkeypatch.setattr(sweep.subprocess, "check_output", submit)
    assert horizon.submit(args, path, state, coordinator, study) == "789"
    assert horizon.submit(args, path, state, coordinator, study) == "789" and len(calls) == 1
    assert horizon.worker_root(plan, 0, "789") == args.result_root / "stage8/job789-task0"


def test_coordinator_publishes_only_h2_and_advances_progress_after_ack(horizon_plan, monkeypatch):
    args, path, plan, provenance, _ = horizon_plan
    updates, published = [], []
    monkeypatch.setattr(horizon, "prepare_plan", lambda *a: (path, plan))
    monkeypatch.setattr(horizon.continuation, "verify_workspace", lambda *a: {"verified": True})
    monkeypatch.setattr(horizon.continuation, "require_checkout", lambda *a: {})
    def submit(args, plan_path, state, *unused): state["job"] = "789"; return "789"
    monkeypatch.setattr(horizon, "submit", submit)
    def publish(run_dir, selector, checkpoint, **kwargs):
        published.append(selector)
        return {"selector": selector, "accepted": 1, "published": 1}
    monkeypatch.setattr(publication, "publish_curve", publish)
    monkeypatch.setattr(publication, "update_progress", lambda *a, **k: updates.append(k))
    monkeypatch.setattr(horizon, "validate_worker", lambda plan, index, *a: {"index": index})
    monkeypatch.setattr(horizon, "label_curve", lambda *a: None)
    coordinator = SimpleNamespace(scheduler_done=lambda *a: True, wait_jobs=lambda *a: None,
                                  atomic_json=atomic_json, require_source=lambda sha: None)
    horizon.coordinate(args, coordinator, study, publication, provenance, api=object())
    assert published == [plan["conditions"][0]["selector"]]
    assert updates[0] == {"phase": "stage8", "conditions_expected": 29, "episodes_expected": 145,
                          "conditions_completed": 28, "episodes_completed": 140}
    assert updates[-1] == {"phase": "complete", "stage8_complete": 1, "conditions_completed": 29, "episodes_completed": 145}
    assert study.read(args.result_root / "stage8-results.json")["reused"] == plan["reused"]


def test_h2_label_excludes_other_ablations_without_rewriting_summary(tmp_path, monkeypatch):
    from wandb.apis.public import Run

    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    remote = {"id": "storage-id", "name": "run-id", "displayName": "old", "state": "finished",
        "tags": [], "description": "", "notes": "", "group": "",
        "config": json.dumps({k: {"value": v} for k, v in {
            "campaign_id": publication.CAMPAIGN, "update_sweep_id": "old", "actor_lr_study_id": "old"}.items()}),
        "summaryMetrics": json.dumps({"eval/return_mean": 620}), "systemMetrics": "{}"}
    mutations = []
    class Client:
        def execute(self, query, variable_values=None, **kwargs):
            values = variable_values or {}
            assert "summaryMetrics" not in values
            if "config" in values:
                mutations.append(values)
                remote.update(config=values["config"], displayName=values["display_name"])
                return {"upsertBucket": {"bucket": {"id": remote["id"], "displayName": remote["displayName"]}}}
            return {"project": {"run": deepcopy(remote)}}
    api = SimpleNamespace(client=Client(), flush=lambda: None)
    api.run = lambda path: Run(api.client, *path.split("/"), attrs=deepcopy(remote))
    horizon.label_curve(api, {"run_id": "run-id"}, horizon.conditions(study)[0], publication)
    config = json.loads(remote["config"])
    assert len(mutations) == 1 and "update_sweep_id" not in config and "actor_lr_study_id" not in config
    assert config["horizon_study_id"]["value"] == "ambixqc-horizon-20261001"
    assert "H2 J2 G6" in remote["displayName"] and json.loads(remote["summaryMetrics"]) == {"eval/return_mean": 620}
