"""Administrative dispatch owns precise arrays and never retries uncertain jobs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_sac_scale as scientific
from slurm import submit_ambi_closed_loop_sac_scale as dispatch
from utils.ambi_benchmark import solver_seed


SHA = "a" * 40
CHECKPOINT_SHA = "b" * 64


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    cells = scientific.cells()
    prior = dict(checkpoint_step=575000, checkpoint_sha256=CHECKPOINT_SHA,
        episodes=[dict(seed=s, solver_seed=solver_seed(55,"episode",s), length=500,
                       truncated_by_evaluator=False, **{"return":float(s)}) for s in scientific.SEEDS])
    for cell in cells:
        cell.update(directory=str(root / cell["name"]), checkpoint_sha256=CHECKPOINT_SHA,
            initial_alpha=scientific.INITIAL_ALPHA, prior_reference=deepcopy(prior),
            performance_run_id="performance-"+cell["name"], training_run_id="training-"+cell["name"],
            run_dir=str(root/"registry"/cell["name"]))
    campaign = dict(source_commit=SHA, checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_step=575000, source_run=scientific.SOURCE_RUN, cells=cells,
        production_indices=list(range(36)), smoke_indices=[10, 11, 34, 35], smoke_steps=8,
        overview_run_id="overview")
    for index in campaign["smoke_indices"]:
        cell = cells[index]
        directory = root / "smoke" / cell["name"]
        bundle = directory / "bundle"
        bundle.mkdir(parents=True)
        dispatch.write(bundle/"manifest.json", dict(code=dict(commit=SHA,dirty=False),
            runs=[dict(selector=cell["selector"])]))
        (bundle / "trace.jsonl.gz").write_bytes(b"sealed trace bytes")
        timing = dict(decisions=8, initialization_seconds=2., warmup_including_compile_seconds=10.,
            control_seconds=8.*cell["H"]*(2 if cell["P"] == 1 else 1),
            probe_seconds=.8*cell["H"], serialization_seconds=.8)
        receipt = dict(status="complete", smoke=True, cell=cell["name"], selector=cell["selector"],
            **{key:cell[key] for key in ("H", "J", "N", "B", "G", "P", "T")},
            checkpoint_step=575000, checkpoint_sha256=CHECKPOINT_SHA,
            bundle=str(bundle), manifest_sha256=dispatch.digest(bundle/"manifest.json"),
            trace_sha256={"trace.jsonl.gz":dispatch.digest(bundle/"trace.jsonl.gz")}, timing=timing)
        dispatch.write(directory/"worker-completion.json", receipt)
    dispatch.write(root/"campaign.json", campaign)
    return root, campaign


def options(root, phase):
    return SimpleNamespace(root=root, phase=phase, sha=SHA, inventory=root.parent/"inventory.json")


def mock_submissions(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "source_check", lambda expected: expected)
    monkeypatch.setattr("os.chdir", lambda path: None)
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=f"{1000+len(calls)};oscar\n", stderr="")
    monkeypatch.setattr(dispatch.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("actual,dirty", [("c"*40, ""), (SHA, " M source.py")])
def test_dispatch_source_rejects_wrong_commit_or_dirty_checkout(monkeypatch, actual, dirty):
    outputs = iter([actual, dirty])
    monkeypatch.setattr(dispatch, "command", lambda argv: next(outputs))
    with pytest.raises(RuntimeError, match="exact expected commit"):
        dispatch.source_check(SHA)


def test_prepare_only_submits_cpu_metadata_work(tmp_path, monkeypatch):
    calls = mock_submissions(monkeypatch)
    args = options(tmp_path/"campaign", "prepare")
    dispatch.dispatch(args)
    assert len(calls) == 1
    argv = calls[0]
    assert "--partition=batch" in argv and "--cpus-per-task=4" in argv and "--mem=16G" in argv
    assert not any(a.startswith(("--gres=", "--array=")) for a in argv)
    exports = next(a for a in argv if a.startswith("--export="))
    assert "EVAL_MODE=prepare" in exports and f"EXPECTED_ACTION_MODES_SHA={SHA}" in exports
    assert f"CHECKPOINT_INVENTORY={args.inventory}" in exports
    with pytest.raises(RuntimeError, match="Existing dispatch journal"):
        dispatch.dispatch(args)
    assert len(calls) == 1


def test_smoke_scope_and_resources(prepared, monkeypatch):
    root, campaign = prepared
    calls = mock_submissions(monkeypatch)
    dispatch.dispatch(options(root, "smoke"))
    assert len(calls) == 1
    argv = calls[0]
    assert "--partition=gpu-debug" in argv and "--gres=gpu:l40s:1" in argv
    assert "--array=10,11,34,35%2" in argv
    assert "--cpus-per-task=4" in argv and "--mem=32G" in argv
    assert "EVAL_SMOKE=1" in next(a for a in argv if a.startswith("--export="))


@pytest.mark.parametrize("failure", ["rejected", "ambiguous", "exception"])
def test_uncertain_sbatch_preserves_intent_and_prevents_duplicate(tmp_path, monkeypatch, failure):
    journal = tmp_path/"intent.json"
    dispatch.write(journal, dict(jobs={}))
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert dispatch.read(journal)["jobs"]["production"]["status"] == "intent"
        if failure == "exception":
            raise OSError("transport lost")
        return SimpleNamespace(returncode=1 if failure == "rejected" else 0,
            stdout="" if failure == "rejected" else "submitted without confirmed ID", stderr="detail")
    monkeypatch.setattr(dispatch.subprocess, "run", run)
    with pytest.raises((RuntimeError, OSError)):
        dispatch.submit(journal, "production", [], {"EVAL_MODE":"worker"})
    with pytest.raises(RuntimeError, match="already attempted"):
        dispatch.submit(journal, "production", [], {"EVAL_MODE":"worker"})
    assert len(calls) == 1
    assert dispatch.read(journal)["jobs"]["production"]["status"] in ("intent", "uncertain_or_failed")


@pytest.mark.parametrize("value", ["one,two", "one\ntwo"])
def test_bad_export_never_records_or_submits_intent(tmp_path, monkeypatch, value):
    journal = tmp_path/"intent.json"
    dispatch.write(journal, dict(jobs={}))
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: pytest.fail("submitted invalid export"))
    with pytest.raises(ValueError): dispatch.submit(journal, "bad", [], {"VALUE":value})
    assert dispatch.read(journal)["jobs"] == {}


def test_production_dispatches_every_cell_once_without_array_throttle(prepared, monkeypatch):
    root, campaign = prepared
    calls = mock_submissions(monkeypatch)
    dispatch.dispatch(options(root, "production"))
    assert len(calls) == 13
    workers = calls[:-1]
    indices = []
    for argv in workers:
        assert "--partition=gpu" in argv and "--qos=pri-gpu+" in argv and "--gres=gpu:l40s:1" in argv
        assert "--cpus-per-task=4" in argv and "--mem=32G" in argv
        array = next(a.removeprefix("--array=") for a in argv if a.startswith("--array="))
        assert "%" not in array
        group = [int(i) for i in array.split(",")]
        assert len(group) == 3
        assert {campaign["cells"][i]["H"] for i in group} == {1, 2, 3}
        assert len({(campaign["cells"][i]["J"], campaign["cells"][i]["P"]) for i in group}) == 1
        indices.extend(group)
    assert sorted(indices) == list(range(36))
    first = next(a for a in workers[0] if a.startswith("--array="))
    assert {campaign["cells"][int(i)]["J"] for i in first.removeprefix("--array=").split(",")} == {10}
    publisher = calls[-1]
    assert "--partition=batch" in publisher and "--cpus-per-task=8" in publisher and "--mem=64G" in publisher
    assert not any(a.startswith(("--gres=", "--array=")) for a in publisher)
    submission = dispatch.read(root/"submission.json")
    assert len(submission["gpu_job_ids"]) == 12 and submission["settings"] == 36
    assert dispatch.read(root/"timing-estimate.json")["queue_delay_included"] is False
    with pytest.raises(RuntimeError, match="Existing dispatch journal"):
        dispatch.dispatch(options(root, "production"))
    assert len(calls) == 13


@pytest.mark.parametrize("missing", [10, 11, 34, 35])
def test_production_requires_every_smoke_receipt_before_submission(prepared, monkeypatch, missing):
    root, campaign = prepared
    cell = campaign["cells"][missing]
    (root/"smoke"/cell["name"]/"worker-completion.json").unlink()
    calls = mock_submissions(monkeypatch)
    with pytest.raises((FileNotFoundError, ValueError, AssertionError)):
        dispatch.dispatch(options(root, "production"))
    assert not calls and not (root.parent/"dispatch-production.json").exists()


def test_estimate_separates_startup_and_scales_J_conservatively_for_H2(prepared):
    _, campaign = prepared
    estimate = dispatch.estimate(campaign)
    points = {(r["H"], r["J"], r["P"]):r for r in estimate["settings"]}
    for p in (1,5):
        for j in (1,2,4,6,8,10):
            assert points[2,j,p]["estimated_seconds"] == points[3,j,p]["estimated_seconds"]
        assert points[3,1,p]["estimated_seconds"] - 12 == pytest.approx((points[3,10,p]["estimated_seconds"]-12)/10)
    for row in estimate["settings"]:
        assert row["time_limit_minutes"] >= 20
        assert row["time_limit_minutes"]*60 >= row["estimated_seconds"]*1.35+300
    longest = max(r["estimated_seconds"] for r in estimate["settings"])/3600
    assert longest <= estimate["ideal_12_gpu_elapsed_hours"] <= estimate["gpu_hours"]
    assert estimate["ideal_12_gpu_elapsed_hours"] >= estimate["gpu_hours"]/12


@pytest.mark.parametrize("field,value", [("N",128), ("B",2048), ("G",16), ("T",1), ("P",2), ("H",4), ("J",3)])
def test_estimate_rejects_settings_outside_the_concrete_campaign(prepared, field, value):
    _, campaign = prepared
    campaign["cells"][0][field] = value
    with pytest.raises((ValueError, AssertionError)):
        dispatch.estimate(campaign)


@pytest.mark.parametrize("field,value", [("cell","different-cell"), ("selector","sweep/different"),
    ("H",2), ("J",1), ("N",128), ("B",2048), ("G",16), ("P",1), ("T",1)])
def test_estimate_rejects_receipt_from_wrong_setting(prepared, field, value):
    root, campaign = prepared
    cell=campaign["cells"][10]
    path=root/"smoke"/cell["name"]/"worker-completion.json"
    receipt=dispatch.read(path); receipt[field]=value; dispatch.write(path,receipt)
    with pytest.raises((ValueError, AssertionError)):
        dispatch.estimate(campaign)


@pytest.mark.parametrize("field,value", [("control_seconds",float("nan")), ("probe_seconds",-1.),
    ("serialization_seconds",float("inf")), ("initialization_seconds",-1.),
    ("warmup_including_compile_seconds",float("nan")), ("decisions",0)])
def test_estimate_rejects_invalid_measured_time(prepared, field, value):
    root,campaign=prepared; cell=campaign["cells"][10]
    path=root/"smoke"/cell["name"]/"worker-completion.json"
    receipt=dispatch.read(path); receipt["timing"][field]=value
    # Write intentional corruption past the normal finite-only JSON writer.
    path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, AssertionError)):
        dispatch.estimate(campaign)


@pytest.mark.parametrize("indices", [[10,11,34], [10,11,34,34], [0,1,24,25]])
def test_estimate_requires_the_four_full_budget_smoke_cells(prepared, indices):
    _, campaign = prepared
    campaign["smoke_indices"] = indices
    with pytest.raises(ValueError, match="all four"):
        dispatch.estimate(campaign)


@pytest.mark.parametrize("damage", ["dirty", "commit", "selector", "trace"])
def test_estimate_rejects_incompatible_or_changed_smoke_bundle(prepared, damage):
    root,campaign=prepared; cell=campaign["cells"][10]
    path=root/"smoke"/cell["name"]/"worker-completion.json"
    receipt=dispatch.read(path); bundle=Path(receipt["bundle"])
    if damage == "trace":
        (bundle/"trace.jsonl.gz").write_bytes(b"changed")
    else:
        manifest=dispatch.read(bundle/"manifest.json")
        if damage == "dirty": manifest["code"]["dirty"] = True
        elif damage == "commit": manifest["code"]["commit"] = "c"*40
        else: manifest["runs"][0]["selector"] = "sweep/wrong"
        dispatch.write(bundle/"manifest.json", manifest)
        receipt["manifest_sha256"] = dispatch.digest(bundle/"manifest.json")
        dispatch.write(path, receipt)
    with pytest.raises((ValueError, AssertionError)):
        dispatch.estimate(campaign)
