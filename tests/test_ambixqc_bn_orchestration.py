"""Coordinator retries preserve submission ownership and truthful publication."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import orchestrate_ambixqc_bn_study as coordinator
import publish_ambixqc_bn_study as publication
import run_ambixqc_bn_study as study
from test_ambixqc_bn_study import selection, plan, complete, SHA
from test_eval_series import FakeWandb, record
from test_ambixqc_inner_475k_screen import write
from utils import eval_series as series


@pytest.fixture
def args(tmp_path):
    for name in ("manifest", "reference_index", "workspace_spec"):
        write(tmp_path / (name + ".json"), {"input": name})
    return SimpleNamespace(result_root=tmp_path, source_sha=SHA, manifest=tmp_path / "manifest.json",
        reference_index=tmp_path / "reference_index.json", workspace_spec=tmp_path / "workspace_spec.json",
        smoke_root=tmp_path / "smoke", probe_root=tmp_path / "probe", checkpoint_root=None,
        initial_jobs=["100", "101"], progress_run_id="progress1", gpu_type="nvidia_rtx_a5000")


def test_wait_jobs_rejects_failed_or_missing_accounting(monkeypatch):
    monkeypatch.setattr(coordinator, "scheduler_done", lambda *a: True)
    for text in ("123_0|COMPLETED|0:0|\n", "123_0|COMPLETED|0:0|\n123_1|FAILED|1:0|\n", ""):
        monkeypatch.setattr(coordinator.subprocess, "check_output", lambda *a, **k: text)
        with pytest.raises(RuntimeError, match="prerequisites failed"):
            coordinator.wait_jobs(["123_0", "123_1"])
    monkeypatch.setattr(coordinator.subprocess, "check_output",
                        lambda *a, **k: "123_0|COMPLETED|0:0|\n123_1|COMPLETED|0:0|\n")
    coordinator.wait_jobs(["123_0", "123_1"])


@pytest.mark.parametrize("field", ["source_sha", "initial_jobs", "progress_run_id", "gpu_type", "manifest"])
def test_coordinator_retry_binds_all_inputs(args, field):
    original = coordinator.coordinator_state(args)
    assert coordinator.coordinator_state(args) == original
    if field == "manifest":
        write(args.manifest, {"changed": True})
    else:
        setattr(args, field, ["102"] if field == "initial_jobs" else "changed")
    with pytest.raises(ValueError, match="different source"):
        coordinator.coordinator_state(args)


@pytest.mark.parametrize("failure", ["timeout", "bad_receipt"])
def test_uncertain_sbatch_is_never_blindly_retried(args, selection, monkeypatch, failure):
    path, value = plan(args.result_root, "stage2", selection=selection[0])
    mapping = args.result_root / "mapping.json"
    write(mapping, {"plan": value["plan_sha256"]})
    monkeypatch.setattr(coordinator, "require_source", lambda *a: None)
    monkeypatch.setattr(study, "run_map", lambda *a: {})
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        saved = study.read(args.result_root / "coordinator-state.json")
        assert saved["submission_intents"]["stage2"]["inputs"]["plan"] == study.bind(path)
        assert "--array=0-3%4" in command
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        return "receipt lost"
    monkeypatch.setattr(coordinator.subprocess, "check_output", submit)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        coordinator.submit_stage(args, path, mapping, {"jobs": {}})
    state = study.read(args.result_root / "coordinator-state.json")
    with pytest.raises(RuntimeError, match="Uncertain prior sbatch"):
        coordinator.submit_stage(args, path, mapping, state)
    assert len(calls) == 1


def test_known_job_retry_checks_exact_plan_and_map(args, selection, monkeypatch):
    path, value = plan(args.result_root, "stage2", selection=selection[0])
    mapping = args.result_root / "mapping.json"
    write(mapping, {"plan": value["plan_sha256"]})
    monkeypatch.setattr(coordinator, "require_source", lambda *a: None)
    monkeypatch.setattr(study, "run_map", lambda *a: {})
    calls = []
    monkeypatch.setattr(coordinator.subprocess, "check_output", lambda *a, **k: calls.append(a) or "123;cluster\n")
    state = {"jobs": {}}
    assert coordinator.submit_stage(args, path, mapping, state) == "123"
    assert coordinator.submit_stage(args, path, mapping, state) == "123"
    assert len(calls) == 1
    write(mapping, {"plan": "changed"})
    with pytest.raises(ValueError, match="exact stage plan"):
        coordinator.submit_stage(args, path, mapping, state)
    assert len(calls) == 1


def test_spec_retry_recomputes_identity_and_rejects_stale_template(args, selection, monkeypatch):
    path, value = plan(args.result_root, "stage2", selection=selection[0])
    monkeypatch.setattr(coordinator, "require_source", lambda *a: None)
    created = []
    def prepare(plan_path, index, manifest, output, *, checkpoint_root=None):
        assert plan_path == path and manifest == args.manifest
        cell = value["conditions"][index]
        spec = {"selector": cell["selector"], "identity": {"index": index}, "label": "test"}
        result = Path(output) / "spec.json"
        result.parent.mkdir(parents=True)
        write(result, spec)
        return {"mode": "evaluation_series_specifications", "specs": {cell["selector"]: str(result)}}
    def allocate(root, spec, stage, selector):
        created.append(spec)
        return {"run_dir": str(root / ("registry-" + str(spec["identity"]["index"]))),
                "run_id": str(spec["identity"]["index"])}
    monkeypatch.setattr(study, "prepare_spec", prepare)
    monkeypatch.setattr(coordinator, "allocate_curve", allocate)
    first = coordinator.allocate_stage(args, path)
    assert coordinator.allocate_stage(args, path) == first
    stale = next((args.result_root / "specs/stage2/0").glob("*.json"))
    payload = study.read(stale)
    payload["identity"] = {"wrong": "controller"}
    write(stale, payload)
    count = len(created)
    with pytest.raises(FileExistsError):
        coordinator.allocate_stage(args, path)
    assert len(created) == count


@pytest.mark.parametrize("problem", ["job_pass", "gpu", "array_job"])
def test_collect_requires_actual_submitted_cuda_jobs(args, selection, monkeypatch, problem):
    path, value = plan(args.result_root, "stage2", selection=selection[0])
    for index in range(4):
        complete(args.result_root / "stage2", path, value, index, monkeypatch)
    result_path = args.result_root / "results.json"
    if problem == "job_pass":
        (args.result_root / "stage2/job123-task0/PASS").unlink()
    elif problem == "gpu":
        write(args.result_root / "stage2/job123-task0/runtime.json", {"gpu": "", "torch": "2.3.1"})
    with pytest.raises((ValueError, FileNotFoundError)):
        coordinator.collect_stage(args, path, "124" if problem == "array_job" else "123", result_path)
    assert not result_path.exists()


def _published_record(tmp_path):
    value = record(tmp_path, 475000, checkpoint={"step": 475000, "sha256": study.CHECKPOINT_SHA},
        selector="controller/test", provenance={"code": {"commit": SHA, "dirty": False}},
        episodes=[{"seed": seed, "return": float(seed), "length": 500, "capped": False}
                  for seed in study.SEEDS])
    registry = series.create_run(tmp_path / "registries", value, publication.CAMPAIGN,
                                 publication.PROJECT, publication.ENTITY, publication.OWNER)
    return value, registry


class SummaryBackend(FakeWandb):
    def init(self, **kwargs):
        run = super().init(**kwargs)
        if not hasattr(run, "summary"):
            run.summary = {}
        return run


def _factory(backend):
    return lambda directory, owner: series.Publisher(directory, owner=owner, wandb_module=backend,
                                                    acknowledgement_timeout=0)


@pytest.mark.parametrize("problem", ["empty", "selector", "checkpoint", "source", "episodes"])
def test_publication_validates_science_before_emitting_history(tmp_path, problem):
    value, registry = _published_record(tmp_path)
    if problem == "selector": value["selector"] = "controller/wrong"
    if problem == "checkpoint": value["checkpoint"]["sha256"] = "b" * 64
    if problem == "source": value["provenance"]["code"]["commit"] = "b" * 40
    if problem == "episodes": value["episodes"].pop()
    if problem != "empty": series.stage_record(registry["run_dir"], value)
    backend = SummaryBackend()
    with pytest.raises((RuntimeError, ValueError)):
        publication.publish_curve(registry["run_dir"], "controller/test", study.CHECKPOINT_SHA,
                                  source_sha=SHA, publisher_factory=_factory(backend))
    assert not backend.runs[registry["run_id"]].rows
    assert not backend.runs[registry["run_id"]].artifacts
    assert not (Path(registry["run_dir"]) / "bn-study-publication-verified.json").exists()


def test_partial_publication_never_claims_complete_and_retry_does_not_duplicate(tmp_path):
    value, registry = _published_record(tmp_path)
    series.stage_record(registry["run_dir"], value)
    backend = SummaryBackend()
    backend.flush = False
    kwargs = dict(source_sha=SHA, publisher_factory=_factory(backend))
    with pytest.raises(series.PublicationUncertainError):
        publication.publish_curve(registry["run_dir"], "controller/test", study.CHECKPOINT_SHA, **kwargs)
    run = backend.runs[registry["run_id"]]
    assert run.summary["study/status"] == "publishing"
    assert not (Path(registry["run_dir"]) / "bn-study-publication-verified.json").exists()
    backend.flush = True
    run.finish()  # Prior history becomes visible; recovery reconciles its slot.
    receipt = publication.publish_curve(registry["run_dir"], "controller/test", study.CHECKPOINT_SHA, **kwargs)
    assert receipt["accepted"] == receipt["published"] == 1
    assert run.summary["study/status"] == "complete"
    assert publication.publish_curve(registry["run_dir"], "controller/test", study.CHECKPOINT_SHA, **kwargs) == receipt
    assert len(run.rows) == 1


@pytest.mark.parametrize("failure", ["scheduler", "smokes"])
def test_prerequisite_failure_cannot_allocate_or_submit(args, monkeypatch, failure):
    monkeypatch.setattr(coordinator, "require_source", lambda *a: None)
    monkeypatch.setattr(study.screen, "select_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(study.screen, "select_reference", lambda *a: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda **k: object()))
    monkeypatch.setattr(coordinator, "install_workspace", lambda *a: {"verified": True})
    monkeypatch.setattr(coordinator, "update_progress", lambda *a, **k: None)
    def failed(*a): raise RuntimeError("failed prerequisite")
    monkeypatch.setattr(coordinator, "wait_jobs", failed if failure == "scheduler" else lambda *a: None)
    monkeypatch.setattr(study, "verify_smokes", failed)
    monkeypatch.setattr(coordinator, "allocate_stage", lambda *a: pytest.fail("allocated before gates"))
    monkeypatch.setattr(coordinator, "submit_stage", lambda *a: pytest.fail("submitted before gates"))
    with pytest.raises(RuntimeError, match="failed prerequisite"):
        coordinator._run_locked(args)
    assert not (args.result_root / "COMPLETE.json").exists()


def test_workspace_api_roundtrip_preserves_panel_ids_and_checks_readback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb_gql", SimpleNamespace(gql=lambda value: value))
    proposal = {"campaign": publication.CAMPAIGN, "name": publication.VIEW_NAME,
                "spec": {"section": {"panelBankConfig": {"sections": [
                    {"name": "Results", "panels": [{"viewType": "RunHistoryLinePlot", "__id__": "new"}]}]}}}}
    path = tmp_path / "workspace.json"
    write(path, proposal)
    old = deepcopy(proposal["spec"])
    old["section"]["panelBankConfig"]["sections"][0]["panels"][0]["__id__"] = "preserved"
    node = {"id": "view1", "name": publication.VIEW_NAME, "spec": json.dumps(old)}
    calls = []
    def execute(query, *, variable_values):
        calls.append(query)
        if query == publication.VIEW_MUTATION:
            node["spec"] = variable_values["spec"]
            return {"upsertView": {"view": {"id": "view1", "name": publication.VIEW_NAME}}}
        return {"project": {"allViews": {"edges": [{"node": node}]}}}
    api = SimpleNamespace(client=SimpleNamespace(execute=execute))
    assert publication.install_workspace(api, path)["verified"]
    assert json.loads(node["spec"])["section"]["panelBankConfig"]["sections"][0]["panels"][0]["__id__"] == "preserved"
    assert calls == [publication.VIEW_QUERY, publication.VIEW_MUTATION, publication.VIEW_QUERY]


@pytest.mark.parametrize("include_stage3,fail_publication", [(False, False), (True, False), (False, True)])
def test_coordinator_scope_and_completion_require_all_publications(args, monkeypatch, include_stage3, fail_publication):
    import run_ambixqc_bn_probe as probe
    args.include_stage3 = include_stage3
    monkeypatch.setattr(coordinator, "require_source", lambda *a: None)
    monkeypatch.setattr(study.screen, "select_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(study.screen, "select_reference", lambda *a: None)
    monkeypatch.setattr(coordinator, "install_workspace", lambda *a: {"verified": True})
    updates, stages, publications, gates = [], [], [], []
    monkeypatch.setattr(coordinator, "update_progress", lambda *a, **k: updates.append(k))
    monkeypatch.setattr(coordinator, "wait_jobs", lambda *a: None)
    monkeypatch.setattr(study, "verify_smokes", lambda *a: gates.append("verified"))
    monkeypatch.setattr(probe, "validate_selection", lambda *a, **k: {
        "selected_critic_bn_mode": "running", "aggregate_scores": {"running": 1.0}})
    class Summary:
        summary = {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def log_artifact(self, artifact): return SimpleNamespace(wait=lambda: None)
    wandb = SimpleNamespace(Api=lambda **k: object(), init=lambda **k: Summary(),
                            Artifact=lambda *a, **k: SimpleNamespace(add_file=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    def prepare(stage, path, **kwargs):
        assert gates == ["verified"]
        stages.append(stage)
    monkeypatch.setattr(study, "prepare", prepare)
    def allocate(args, path):
        stage = path.name.split("-")[0]
        cells = [{"selector": f"controller/{stage}-{i}"} for i in range(4 if stage == "stage2" else 3)]
        mapping = args.result_root / (stage + "-map.json")
        write(mapping, {"runs": {c["selector"]: "registry-" + c["selector"] for c in cells}})
        return {"stage": stage, "conditions": cells}, mapping
    monkeypatch.setattr(coordinator, "allocate_stage", allocate)
    monkeypatch.setattr(coordinator, "submit_stage", lambda *a: "123")
    monkeypatch.setattr(coordinator, "collect_stage", lambda a, p, j, output: write(output, {"validated": True}))
    def publish(*a, **k):
        publications.append(a[1])
        if fail_publication and len(publications) == 2:
            raise RuntimeError("unacknowledged publication")
    monkeypatch.setattr(coordinator, "publish_curve", publish)
    if fail_publication:
        with pytest.raises(RuntimeError, match="unacknowledged"):
            coordinator._run_locked(args)
        assert not (args.result_root / "COMPLETE.json").exists()
        assert not any(u.get("phase") == "complete" for u in updates)
    else:
        coordinator._run_locked(args)
        assert stages == (["stage2", "stage3"] if include_stage3 else ["stage2"])
        count = 7 if include_stage3 else 4
        assert len(publications) == count
        assert updates[-1] == {"phase": "complete", "conditions_completed": count, "episodes_completed": count * 5}
        assert (args.result_root / "COMPLETE.json").exists()
