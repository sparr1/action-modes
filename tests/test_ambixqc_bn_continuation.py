"""Stage 3 continuation preserves parent science, source and submission ownership."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import continue_ambixqc_bn_study as continuation
import run_ambixqc_bn_study as study
import publish_ambixqc_bn_study as publication
from test_ambixqc_bn_orchestration import _published_record, SHA
from test_ambixqc_bn_study import selection, plan, complete
from test_ambixqc_inner_475k_screen import write
from utils import eval_series as series
from utils.ambi_benchmark import atomic_json


@pytest.mark.parametrize("problem", ["sha", "dirty", "head"])
def test_checkout_identity_requires_exact_clean_commit(tmp_path, monkeypatch, problem):
    def output(command, **kwargs):
        if command[-1] == "HEAD^{tree}": return "c" * 40
        if command[-1] == "HEAD": return ("b" if problem == "head" else "a") * 40
        return " M changed.py" if problem == "dirty" else ""
    monkeypatch.setattr(continuation.subprocess, "check_output", output)
    with pytest.raises(ValueError):
        continuation.require_checkout(tmp_path, "short" if problem == "sha" else SHA)


def test_mixed_execution_imports_are_rejected_before_chdir(tmp_path, monkeypatch):
    cwd = Path.cwd()
    monkeypatch.setitem(sys.modules, "orchestrate_ambixqc_bn_study",
                        SimpleNamespace(__file__=str(tmp_path / "wrong/source.py")))
    (tmp_path / "execution").mkdir()
    with pytest.raises(ValueError, match="another checkout"):
        continuation.execution_modules(tmp_path / "execution")
    assert Path.cwd() == cwd


@pytest.fixture
def parent(tmp_path, selection, monkeypatch):
    root = tmp_path / "parent"
    root.mkdir()
    plan_path = root / "stage2-plan.json"
    value = study.prepare("stage2", plan_path, source_sha=SHA, selection=selection[0])
    for index in range(4):
        complete(root / "stage2", plan_path, value, index, monkeypatch, mean=200+index)
    results = root / "stage2-results.json"
    study.collect(plan_path, root / "stage2", results)
    runs = {}
    for index, cell in enumerate(value["conditions"]):
        directory = root / f"record-{index}"
        directory.mkdir()
        record, registry = _published_record(directory)
        record["selector"] = cell["selector"]
        series.stage_record(registry["run_dir"], record)
        pub_path = Path(registry["run_dir"]) / "publication.json"
        pub = study.read(pub_path)
        entry = pub["records"][record["record_id"]]
        entry["status"] = "published"
        write(pub_path, pub)
        write(Path(registry["run_dir"]) / "bn-study-publication-verified.json", {
            "selector": cell["selector"], "checkpoint_sha256": study.CHECKPOINT_SHA,
            "source_sha": SHA, "run_id": registry["run_id"], "record_id": record["record_id"],
            "record_sha256": entry["record_sha256"], "accepted": 1, "published": 1})
        runs[cell["selector"]] = registry["run_dir"]
    write(root / "stage2-run-map.json", {"schema": study.MAP_SCHEMA,
                                         "plan_sha256": value["plan_sha256"], "runs": runs})
    for name in ("manifest", "reference_index"):
        write(root / (name + ".json"), {"name": name})
    state = {"campaign": publication.CAMPAIGN, "source_sha": SHA, "include_stage3": False,
             "stage2_complete": True, "jobs": {"stage2": "123"}, "progress_run_id": "progress",
             "stage2_results": study.bind(results),
             "inputs": {key: study.bind(root / (key + ".json")) for key in ("manifest", "reference_index")},
             "roots": {"smoke_root": str(root / "smokes"), "checkpoint_root": None}}
    write(root / "coordinator-state.json", state)
    write(root / "COMPLETE.json", {**state, "workspace": {"verified": True}})
    extension = tmp_path / "extension"
    extension.mkdir()
    workspace = tmp_path / "extension-workspace.json"
    write(workspace, {"campaign": publication.CAMPAIGN, "name": continuation.VIEW_NAME, "spec": {}})
    return SimpleNamespace(parent_root=root, result_root=extension, source_sha=SHA, progress_run_id="progress",
                           gpu_type="nvidia_rtx_a5000", workspace_spec=workspace)


def test_parent_requires_matching_complete_results_and_receipts(parent):
    before = {path: path.read_bytes() for path in parent.parent_root.rglob("*.json")}
    evidence, result = continuation.parent_evidence(parent, study, publication)
    assert result == parent.parent_root / "stage2-results.json"
    assert len(evidence["publication_receipts"]) == 4
    assert {path: path.read_bytes() for path in before} == before
    receipt = Path(evidence["publication_receipts"][0]["path"])
    value = study.read(receipt)
    value["published"] = 0
    write(receipt, value)
    with pytest.raises(ValueError, match="acknowledgement"):
        continuation.parent_evidence(parent, study, publication)


@pytest.mark.parametrize("change", ["not_complete", "wrong_source", "stage3_already", "changed_parent"])
def test_incompatible_parent_is_rejected(parent, change):
    path = parent.parent_root / "coordinator-state.json"
    value = study.read(path)
    if change == "not_complete": value["stage2_complete"] = False
    elif change == "wrong_source": value["source_sha"] = "b"*40
    elif change == "stage3_already": value["include_stage3"] = True
    else: value["extra"] = "changed after completion"
    write(path, value)
    if change != "changed_parent": write(parent.parent_root / "COMPLETE.json", {**value, "workspace": {}})
    with pytest.raises(ValueError, match="completed, matching"):
        continuation.parent_evidence(parent, study, publication)


@pytest.mark.parametrize("problem", [None, "wrong_name", "changed_spec", "missing"])
def test_workspace_verification_is_read_only_and_exact(parent, problem):
    proposed = study.read(parent.workspace_spec)
    if problem == "wrong_name":
        proposed["name"] = publication.VIEW_NAME
        write(parent.workspace_spec, proposed)
    calls = []
    node = {"name": continuation.VIEW_NAME, "id": "view1", "spec": {"changed": True} if problem == "changed_spec" else {}}
    def execute(query, variables):
        assert query.startswith("query ")
        calls.append(query)
        return {"project": {"allViews": {"edges": [] if problem == "missing" else [{"node": node}]}}}
    api = SimpleNamespace(_service_api=SimpleNamespace(execute_graphql=execute))
    if problem:
        with pytest.raises(ValueError): continuation.verify_workspace(api, parent.workspace_spec, publication)
    else:
        assert continuation.verify_workspace(api, parent.workspace_spec, publication)["url"] == continuation.VIEW_URL
        assert len(calls) == 1


@pytest.mark.parametrize("fail_publication", [False, True])
def test_only_three_new_cells_execute_and_parent_stays_unchanged(parent, monkeypatch, fail_publication):
    before = {path: path.read_bytes() for path in parent.parent_root.rglob("*.json")}
    monkeypatch.setattr(study, "verify_smokes", lambda *a: {})
    monkeypatch.setattr(study.screen, "select_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(study.screen, "select_reference", lambda *a: None)
    monkeypatch.setattr(continuation, "verify_workspace", lambda *a: {"url": continuation.VIEW_URL, "verified": True})
    updates, submitted, published = [], [], []
    monkeypatch.setattr(publication, "update_progress", lambda *a, **k: updates.append(k))
    def allocate(args, path):
        value = study.load_plan(path, source_sha=args.source_sha)
        mapping = args.result_root / "stage3-run-map.json"
        write(mapping, {"runs": {cell["selector"]: cell["selector"] for cell in value["conditions"]}})
        return value, mapping
    def submit(args, path, mapping, state):
        submitted.append(study.load_plan(path)["stage"])
        state["jobs"]["stage3"] = "456"
        return "456"
    def collect(args, path, job, output):
        value = study.load_plan(path)
        for index in range(3): complete(args.result_root / "stage3", path, value, index, monkeypatch)
        return study.collect(path, args.result_root / "stage3", output)
    def publish(run_dir, selector, checkpoint, **kwargs):
        published.append(selector)
        if fail_publication and len(published) == 2: raise RuntimeError("publication uncertain")
        return {"selector": selector, "published": 1}
    monkeypatch.setattr(publication, "publish_curve", publish)
    coordinator = SimpleNamespace(atomic_json=atomic_json, allocate_stage=allocate, submit_stage=submit,
                                  wait_jobs=lambda jobs: submitted.append(jobs), collect_stage=collect,
                                  require_source=lambda sha: None)
    provenance = {"tooling": {"commit": "b"*40}, "execution": {"commit": SHA}}
    if fail_publication:
        with pytest.raises(RuntimeError, match="uncertain"):
            continuation.continue_stage3(parent, coordinator, study, publication, provenance, api=object())
        assert not (parent.result_root / "COMPLETE.json").exists()
        assert not any(update.get("phase") == "complete" for update in updates)
    else:
        state = continuation.continue_stage3(parent, coordinator, study, publication, provenance, api=object())
        assert state["tooling"]["commit"] != state["source_sha"] == SHA
        assert len(published) == 3 and len(state["published"]) == 3
        assert submitted == ["stage3", ["456_0", "456_1", "456_2"]]
        assert updates[0]["conditions_completed"] == 4 and updates[0]["episodes_completed"] == 20
        assert updates[-1] == {"phase": "complete", "stage3_complete": 1, "conditions_completed": 7, "episodes_completed": 35}
        value = study.load_plan(parent.result_root / "stage3-plan.json")
        assert value["reused"][0]["condition"]["selector"] not in published
    assert {path: path.read_bytes() for path in before} == before


def test_incompatible_extension_state_fails_before_allocation(parent, monkeypatch):
    monkeypatch.setattr(study, "verify_smokes", lambda *a: {})
    monkeypatch.setattr(study.screen, "select_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(study.screen, "select_reference", lambda *a: None)
    monkeypatch.setattr(continuation, "verify_workspace", lambda *a: {})
    write(parent.result_root / "coordinator-state.json", {"source_sha": SHA, "tooling": {"commit": "other"}})
    coordinator = SimpleNamespace(allocate_stage=lambda *a: pytest.fail("must not allocate"))
    with pytest.raises(ValueError, match="Extension state"):
        continuation.continue_stage3(parent, coordinator, study, publication,
                                     {"tooling": {"commit": "b"*40}, "execution": {"commit": SHA}}, api=object())


def test_runtime_failure_is_saved_and_reported_without_changing_parent(parent, monkeypatch):
    parent.execution_root = parent.result_root.parent / "execution"
    parent.execution_root.mkdir()
    parent.tooling_sha = "b" * 40
    monkeypatch.setenv("SLURM_JOB_ID", "789")
    monkeypatch.setattr(continuation, "require_checkout", lambda root, sha: {"root": str(root), "commit": sha})
    monkeypatch.setattr(continuation, "require_runtime", lambda *a: None)
    updates = []
    fake_publication = SimpleNamespace(update_progress=lambda *a, **k: updates.append(k))
    fake_coordinator = SimpleNamespace(atomic_json=atomic_json)
    monkeypatch.setattr(continuation, "execution_modules", lambda *a: (fake_coordinator, study, fake_publication))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda **k: object()))
    def fail(*a, **k): raise RuntimeError("injected failure")
    monkeypatch.setattr(continuation, "continue_stage3", fail)
    original = (parent.parent_root / "COMPLETE.json").read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        continuation.run(parent)
    saved = study.read(parent.result_root / "CONTINUATION_FAILED.json")
    assert saved["type"] == "RuntimeError" and saved["execution"]["commit"] == SHA
    assert updates == [{"phase": "stage3_failed", "failure_type": "RuntimeError"}]
    assert (parent.parent_root / "COMPLETE.json").read_bytes() == original
