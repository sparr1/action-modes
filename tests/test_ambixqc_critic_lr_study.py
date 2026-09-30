"""Critic-rate follow-up preserves its baseline, pairing and publication gates."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_ambixqc_critic_lr_study as followup
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_bn_continuation import parent as stage2_parent
from test_ambixqc_bn_orchestration import _published_record
from test_ambixqc_bn_study import selection, complete
from utils import eval_series as series
from utils.ambi_benchmark import atomic_json


SCIENCE_SHA = followup.SOURCE_SHA
TOOLING_SHA = "b" * 40


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_exact_two_new_cells_change_only_critic_learning_rate():
    baseline = study.condition("running", "return_return", 4, 5e-5)
    original = deepcopy(baseline)
    cells = followup.conditions(study)
    assert len(cells) == 2
    assert len({cell["selector"] for cell in cells}) == 2
    assert baseline["selector"] not in {cell["selector"] for cell in cells}
    assert [cell["settings"]["inner_critic_lr"] for cell in cells] == [1e-4, 2e-4]
    for cell in cells:
        assert {key: value for key, value in cell.items() if key not in {"selector", "settings", "critic_lr"}} == {
            key: value for key, value in baseline.items() if key not in {"selector", "settings"}}
        changed = {key for key in set(cell["settings"]) | set(baseline["settings"])
                   if cell["settings"].get(key) != baseline["settings"].get(key)}
        assert changed == {"inner_critic_lr"}
        settings = cell["settings"]
        assert settings["inner_rounds"] == 4
        assert settings["inner_actor_lr"] == settings["inner_temperature_lr"] == 5e-5
        assert settings["inner_critic_bn_mode"] == settings["inner_actor_bn_mode"] == "running"
        assert settings["inner_critic_source"] == settings["inner_horizon_critic_source"] == "aux_return"
        assert settings["inner_critic_target"] == "reward_only"
        assert settings["inner_terminal_bootstrap"] == "outer"
        assert settings["inner_reward_normalization"] == "frozen_real_scale"
        assert settings["inner_updates_per_round"] == settings["inner_policy_delay"] == 3
        assert settings["inner_replay_capacity"] == 1024
        assert settings["inner_rollout_horizon"] == 1
        assert settings["inner_batch_size"] == settings["inner_rollouts_per_round"] == 256
    assert baseline == original


def test_matrix_contains_only_two_new_conditions_and_honest_rate_descriptions():
    cells = followup.conditions(study)
    matrix = followup.matrix_for(study, cells)
    assert matrix["evaluation"]["default_presets"] == []
    variants = matrix["comparisons"]["controller"]["variants"]
    names = {cell["selector"].split("/")[1] for cell in cells}
    assert set(variants) == names | {"prior"}
    for cell in cells:
        variant = variants[cell["selector"].split("/")[1]]
        assert variant["alg_params"] == cell["settings"]
        assert "critic and temperature LR fixed 5e-5" not in variant["description"]
    assert matrix["comparisons"]["controller"]["reference"] == "prior"


def test_condition_copies_cannot_change_each_other_or_baseline():
    original = followup.conditions(study)
    mutated = followup.conditions(study)
    mutated[0]["settings"]["inner_actor_lr"] = 1
    assert mutated[1] == original[1]
    assert followup.conditions(study) == original
    assert study.BASE_SETTINGS["inner_critic_lr"] == 5e-5


def test_new_matrix_resolves_only_critic_rate_change_from_saved_backbone(tmp_path):
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import CheckpointContext

    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    cells = followup.conditions(study)
    path = write(tmp_path / "matrix.json", followup.matrix_for(study, cells))
    baseline = study.condition("running", "return_return", 4, 5e-5)
    baseline_path = write(tmp_path / "baseline.json", study.matrix_for([baseline]))
    reference = resolve_preset(baseline_path, baseline["selector"], checkpoint_context=context)
    for cell in cells:
        resolved = resolve_preset(path, cell["selector"], checkpoint_context=context)
        assert resolved["environment"] == reference["environment"]
        actual = resolved["algorithm_config"]["alg_params"]
        expected = reference["algorithm_config"]["alg_params"]
        assert {key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)} == {"inner_critic_lr"}
        assert actual["inner_critic_lr"] == cell["settings"]["inner_critic_lr"]
    assert followup.matrix_for(study, cells)["evaluation"] == {
        **study.matrix_for([baseline])["evaluation"], "default_presets": []}


@pytest.fixture
def parent(stage2_parent, monkeypatch):
    args = stage2_parent
    evidence, stage2_results = followup.continuation.parent_evidence(args, study, publication)
    root = args.result_root
    path = root / "stage3-plan.json"
    plan = study.prepare("stage3", path, source_sha=args.source_sha, stage2_results=stage2_results)
    for index in range(3): complete(root / "stage3", path, plan, index, monkeypatch, mean=300+index)
    result_path = root / "stage3-results.json"
    results = study.collect(path, root / "stage3", result_path)
    mapping, published = {}, {}
    for index, entry in enumerate(results["entries"]):
        directory = root / f"record-{index}"
        directory.mkdir()
        record, registry = _published_record(directory)
        selector = entry["condition"]["selector"]
        record["selector"] = selector
        for episode in record["episodes"]: episode["return"] = entry["episode_returns"][str(episode["seed"])]
        series.stage_record(registry["run_dir"], record)
        publication_path = Path(registry["run_dir"]) / "publication.json"
        state = study.read(publication_path)
        saved = state["records"][record["record_id"]]
        saved["status"] = "published"
        write(publication_path, state)
        receipt = {"selector": selector, "checkpoint_sha256": study.CHECKPOINT_SHA,
                   "source_sha": args.source_sha, "run_id": registry["run_id"], "record_id": record["record_id"],
                   "record_sha256": saved["record_sha256"], "accepted": 1, "published": 1}
        write(Path(registry["run_dir"]) / "bn-study-publication-verified.json", receipt)
        mapping[selector], published[selector] = registry["run_dir"], receipt
    write(root / "stage3-run-map.json", {"schema": study.MAP_SCHEMA, "plan_sha256": plan["plan_sha256"], "runs": mapping})
    state = {"schema": followup.continuation.SCHEMA, "campaign": publication.CAMPAIGN,
             "source_sha": args.source_sha, "progress_run_id": args.progress_run_id,
             "stage3_complete": True, "jobs": {"stage3": "123"}, "parent": evidence,
             "stage3_results": study.bind(result_path), "published": published}
    write(root / "coordinator-state.json", state)
    write(root / "COMPLETE.json", {**state, "workspace": {"verified": True}})
    args.parent_root = root
    args.result_root = root.parent / "critic-followup"
    args.result_root.mkdir()
    return args


def test_parent_reuses_exact_baseline_with_matching_return_and_publication_evidence(parent):
    before = {path: path.read_bytes() for path in parent.parent_root.rglob("*.json")}
    evidence, baseline = followup.parent_evidence(parent, study, publication)
    assert baseline["condition"] == study.condition("running", "return_return", 4, 5e-5)
    assert len(baseline["episode_returns"]) == 5
    assert len(evidence["publication_receipts"]) == 3
    assert study.check_binding(evidence["results"]) == parent.parent_root / "stage3-results.json"
    assert parent.manifest.is_file() and parent.reference_index.is_file()
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("problem", ["incomplete", "source", "receipt", "returns", "published", "manifest", "parent_hash"])
def test_parent_rejects_incomplete_changed_or_unacknowledged_evidence(parent, problem):
    root = parent.parent_root
    state = study.read(root / "coordinator-state.json")
    mapping = study.read(root / "stage3-run-map.json")
    run = Path(next(iter(mapping["runs"].values())))
    if problem == "receipt":
        path = run / "bn-study-publication-verified.json"
        value = study.read(path); value["published"] = 0; write(path, value)
    elif problem == "returns":
        path = next((run / "records").glob("*.json"))
        value = study.read(path); value["episodes"][0]["return"] += 1; write(path, value)
    elif problem == "manifest":
        original = study.read(state["parent"]["state"]["path"])
        write(original["inputs"]["manifest"]["path"], {"changed": True})
    else:
        if problem == "incomplete": state["stage3_complete"] = False
        elif problem == "source": state["source_sha"] = "f"*40
        elif problem == "published": state["published"].pop(next(iter(state["published"])))
        else: state["parent"]["state"]["sha256"] = "f"*64
        write(root / "coordinator-state.json", state)
        write(root / "COMPLETE.json", {**state, "workspace": {}})
    with pytest.raises((ValueError, RuntimeError)): followup.parent_evidence(parent, study, publication)


@pytest.fixture
def plan(tmp_path):
    args = SimpleNamespace(result_root=tmp_path, execution_root=tmp_path / "execution", source_sha=SCIENCE_SHA,
                           tooling_sha=TOOLING_SHA, worker_launcher=write(tmp_path / "worker.sbatch", {}),
                           gpu_type="nvidia_rtx_a5000", progress_run_id="progress")
    cells = followup.conditions(study)
    matrix = write(tmp_path / "stage4.matrix.json", followup.matrix_for(study, cells))
    runs = {cell["selector"]: study.bind(write(tmp_path / str(index) / "run.json",
             {"run_id": f"run-{index}", "run_dir": str(tmp_path / str(index))})) for index, cell in enumerate(cells)}
    value = {"schema": followup.SCHEMA, "stage": "stage4", "source_sha": SCIENCE_SHA,
             "campaign": publication.CAMPAIGN,
             "tooling": {"commit": TOOLING_SHA}, "execution": {"commit": SCIENCE_SHA},
             "checkpoint_sha256": study.CHECKPOINT_SHA, "checkpoint_step": 475000,
             "conditions": cells, "reused": {"condition": study.condition("running", "return_return", 4, 5e-5)},
             "matrix": study.bind(matrix), "runs": runs, "result_root": str(tmp_path),
             "inputs": {key: study.bind(write(tmp_path / (key+".json"), {})) for key in ("manifest", "reference_index")},
             "environment_seeds": study.SEEDS, "controller_seed": 12345, "max_steps": 500,
             "smoke_seeds": [101, 102], "smoke_max_steps": 3, "reference_bundle": str(tmp_path / "reference")}
    value["plan_sha256"] = study.digest(value)
    path = write(tmp_path / "stage4-plan.json", value)
    return args, path, value


def test_plan_requires_original_science_and_two_new_five_episode_cells(plan):
    args, path, value = plan
    assert followup.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA) == value
    assert len(value["conditions"]) * len(value["environment_seeds"]) == 10
    with pytest.raises(ValueError): followup.load_plan(path, study, "a"*40, TOOLING_SHA)
    with pytest.raises(ValueError): followup.load_plan(path, study, SCIENCE_SHA, "c"*40)


@pytest.mark.parametrize("problem", ["digest", "actor_lr", "temperature_lr", "critic_lr", "seeds", "horizon", "baseline", "matrix", "registry", "manifest"])
def test_plan_rejects_resigned_scope_changes_and_changed_bound_artifacts(plan, problem):
    _, path, value = plan
    if problem == "digest": value["plan_sha256"] = "f"*64
    elif problem in {"actor_lr", "temperature_lr", "critic_lr"}: value["conditions"][0]["settings"]["inner_"+problem] = 9e-5
    elif problem == "seeds": value["environment_seeds"] = [101, 102]
    elif problem == "horizon": value["conditions"][0]["settings"]["inner_rollout_horizon"] = 2
    elif problem == "baseline": value["reused"]["condition"]["settings"]["inner_critic_lr"] = 1e-4
    elif problem == "matrix": write(value["matrix"]["path"], {})
    elif problem == "registry":
        first, second = [cell["selector"] for cell in value["conditions"]]
        value["runs"][second] = value["runs"][first]
    else: write(value["inputs"]["manifest"]["path"], {"changed": True})
    if problem != "digest": value["plan_sha256"] = study.digest({k: v for k, v in value.items() if k != "plan_sha256"})
    write(path, value)
    with pytest.raises(ValueError): followup.load_plan(path, study, SCIENCE_SHA, TOOLING_SHA)


@pytest.mark.parametrize("index,job", [(True, "123"), (-1, "123"), (2, "123"), (.5, "123"), (0, "123_0"), (0, "")])
def test_worker_bounds_cannot_select_baseline_or_other_array_task(plan, index, job):
    with pytest.raises(ValueError): followup.worker_root(plan[2], index, job)


@pytest.mark.parametrize("failure", ["timeout", "bad_receipt"])
def test_uncertain_submission_is_durable_and_never_resent(plan, monkeypatch, failure):
    args, path, value = plan
    state = {"plan": study.bind(path)}
    calls = []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    def submit(command, **kwargs):
        calls.append(command)
        assert "--array=0-1%2" in command
        saved = study.read(args.result_root / "coordinator-state.json")
        assert saved["submission_intent"]["inputs"]["plan"] == study.bind(path)
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 60)
        return "uncertain receipt"
    monkeypatch.setattr(followup.subprocess, "check_output", submit)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)): followup.submit(args, path, state, coordinator, study)
    with pytest.raises(RuntimeError, match="Uncertain prior sbatch"): followup.submit(args, path, state, coordinator, study)
    assert len(calls) == 1


def test_acknowledged_submission_is_idempotent_and_binds_launcher(plan, monkeypatch):
    args, path, _ = plan
    state, calls = {"plan": study.bind(path)}, []
    coordinator = SimpleNamespace(require_source=lambda sha: None, atomic_json=atomic_json)
    monkeypatch.setattr(followup.subprocess, "check_output", lambda *a, **k: calls.append(a) or "123;oscar\n")
    assert followup.submit(args, path, state, coordinator, study) == "123"
    assert followup.submit(args, path, state, coordinator, study) == "123"
    write(args.worker_launcher, {"changed": True})
    with pytest.raises(ValueError): followup.submit(args, path, state, coordinator, study)
    assert len(calls) == 1


@pytest.mark.parametrize("problem", [None, "scheduler", "publication"])
@pytest.mark.parametrize("first_index", [0, 1])
def test_completed_cell_publishes_before_other_finishes_and_failures_keep_partial_success(plan, monkeypatch, problem, first_index):
    args, _, value = plan
    events, updates = [], []
    state = {"job": "123", "published": {}}
    polls = {0: 0, 1: 0}
    def done(jobs, observed):
        index = int(jobs[0].split("_")[1]); polls[index] += 1
        return index == first_index or polls[index] > 1
    def wait(jobs):
        events.append(("wait", jobs[0]))
        if problem == "scheduler" and jobs == [f"123_{1-first_index}"]: raise RuntimeError("GPU failed")
    def publish(run_dir, selector, checkpoint, **kwargs):
        events.append(("publish", selector))
        if problem == "publication" and selector == value["conditions"][1-first_index]["selector"]:
            raise RuntimeError("publication uncertain")
        return {"selector": selector, "accepted": 1, "published": 1}
    coordinator = SimpleNamespace(scheduler_done=done, wait_jobs=wait, atomic_json=atomic_json)
    publisher = SimpleNamespace(publish_curve=publish, update_progress=lambda *a, **k: updates.append(k))
    monkeypatch.setattr(followup, "validate_worker", lambda plan, index, *a: {"index": index})
    monkeypatch.setattr(followup, "label_curve", lambda *a: None)
    monkeypatch.setattr(followup.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    if problem:
        with pytest.raises(RuntimeError, match="completion/publication failed"):
            followup.publish_finished(args, value, state, coordinator, study, publisher, object())
    else:
        assert followup.publish_finished(args, value, state, coordinator, study, publisher, object()) == [{"index": 0}, {"index": 1}]
    first_selector = value["conditions"][first_index]["selector"]
    assert events.index(("publish", first_selector)) < events.index(("sleep", 20)) < events.index(("wait", f"123_{1-first_index}"))
    assert state["published"][first_selector]["published"] == 1
    assert updates[0] == {"conditions_completed": 8, "episodes_completed": 40}
    if problem:
        assert len(state["published"]) == 1 and str(1-first_index) in state["failures"]
        assert not (args.result_root / "COMPLETE.json").exists()
    else:
        assert updates[-1] == {"conditions_completed": 9, "episodes_completed": 45}


@pytest.mark.parametrize("fail_smoke", [False, True])
def test_worker_runs_exact_cell_smoke_before_full_and_never_runs_full_after_failure(plan, monkeypatch, fail_smoke):
    args, path, value = plan
    args.plan, args.index = path, 0
    value.update(parent_root=str(args.result_root / "parent"), progress_run_id="progress",
                 parent={"verified": True}, smoke_root=str(args.result_root / "smokes"), checkpoint_root=None)
    provenance = {"tooling": value["tooling"], "execution": value["execution"]}
    monkeypatch.setattr(followup, "load_plan", lambda *a: value)
    def parent_evidence(parent_args, *unused):
        parent_args.manifest = Path(value["inputs"]["manifest"]["path"])
        parent_args.reference_index = Path(value["inputs"]["reference_index"]["path"])
        parent_args.smoke_root = Path(value["smoke_root"])
        parent_args.checkpoint_root = None
        return value["parent"], value["reused"]
    monkeypatch.setattr(followup, "parent_evidence", parent_evidence)
    monkeypatch.setattr(study, "verify_smokes", lambda *a: {})
    monkeypatch.setattr(study.screen, "select_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(study.screen, "select_reference", lambda *a: value["reference_bundle"])
    monkeypatch.setattr(followup.continuation, "require_checkout", lambda *a: {})
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda index: "test GPU")
    fake_torch = SimpleNamespace(__version__="2.3.1+cu121", cuda=fake_cuda,
        ones=lambda *a, **k: SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    calls = []
    def evaluate(plan, index, root, study, *, smoke):
        calls.append((index, smoke))
        if fail_smoke: raise RuntimeError("smoke failed")
    monkeypatch.setattr(followup, "evaluate_cell", evaluate)
    coordinator = SimpleNamespace(atomic_json=atomic_json, require_source=lambda sha: None)
    if fail_smoke:
        with pytest.raises(RuntimeError, match="smoke failed"):
            followup.worker(args, coordinator, study, publication, provenance)
        assert calls == [(0, True)]
        assert (followup.worker_root(value, 0, "123") / "FAILED").is_file()
        assert not (followup.worker_root(value, 0, "123") / "PASS").exists()
    else:
        followup.worker(args, coordinator, study, publication, provenance)
        assert calls == [(0, True), (0, False)]
        assert (followup.worker_root(value, 0, "123") / "PASS").is_file()
