"""Offline publication gates prevent incomplete or mismatched discovery claims."""

from copy import deepcopy
import errno
import json
import sys
from types import SimpleNamespace

import pytest

from slurm.ambi_transfer_discovery_campaign import digest, validate_result
from slurm.ambi_transfer_discovery_publish import snapshot
from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_wandb_results_layout import Service, sample_spec
from utils.ambi_benchmark import solver_seed
from utils import wandb_results_layout as existing_layout
from utils import wandb_transfer_discovery_layout as layout


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    cell = dict(index=0, name="h1_j1_rho_a0_c0", H=1, J=1, arm="rho_a0_c0")
    campaign = dict(source_commit="tested-source", checkpoint_sha256="frozen-checkpoint",
        controller_seed=55, seeds=[101, 102, 103], smoke_seeds=[101, 102],
        max_steps=500, smoke_steps=3, cells=[cell], H=[1], J=[1])
    source = dict(git_head="tested-source", files={"engine.py": "digest"}, sha256="source-digest")
    common = dict(cell_id=cell["name"], arm=cell["arm"], horizon=1, rounds=1,
                  checkpoint_sha256="frozen-checkpoint", seeds=campaign["seeds"], smoke=False,
                  source=source)
    manifest = dict(common, max_steps=500, controller_seed=55)
    episodes = [dict(seed=seed, solver_seed=solver_seed(55, "episode", seed),
        **{"return": float(seed), "length": 500, "control_seconds": 10.,
           "terminated": False, "truncated": True}) for seed in campaign["seeds"]]
    result = dict(common, episodes=episodes, complete=True, frozen_outer_verified=True)
    directory = tmp_path / "settings" / cell["name"]
    _save(tmp_path / "campaign.json", campaign)
    _save(directory / "manifest.json", manifest)
    _save(directory / "results.json", result)
    _save(directory / "worker-completion.json", dict(status="complete", cell=cell, smoke=False,
        source_commit=campaign["source_commit"], campaign_sha256=digest(tmp_path / "campaign.json"),
        result_sha256=digest(directory / "results.json"), manifest_sha256=digest(directory / "manifest.json")))
    return campaign, cell, directory, manifest, result


def test_complete_result_passes_and_publishes_paired_episode_units(tmp_path):
    campaign, cell, directory, _, result = fixture(tmp_path)
    assert validate_result(directory, campaign, cell)[0] == result
    value = snapshot(tmp_path, campaign, False, {})
    assert value["completed"] == 1 and len(value["episodes"]) == 3
    row = value["results"][0]
    assert row["return_mean"] == 102. and row["episodes"] == 3
    assert row["paired_vs_fresh_mean"] == row["paired_vs_fresh_std"] == 0.


def test_partial_outputs_are_progress_only_without_worker_receipt(tmp_path):
    campaign, _, directory, _, _ = fixture(tmp_path)
    (directory / "worker-completion.json").unlink()
    _save(directory / "progress.json", dict(completed_episodes=2, seed=103, decision=123))
    value = snapshot(tmp_path, campaign, True, {})
    assert value["completed"] == 0 and value["episodes"] == []
    assert value["settings"][0]["completed_episodes"] == 2
    assert value["results"][0]["return_mean"] is None
    assert value["results"][0]["paired_vs_fresh_mean"] is None


@pytest.mark.parametrize("corruption", ["frozen", "checkpoint", "source", "cell", "seeds", "incomplete", "nonfinite"])
def test_result_validation_rejects_scientific_binding_errors(tmp_path, corruption):
    campaign, cell, directory, _, result = fixture(tmp_path)
    if corruption == "frozen":
        result["frozen_outer_verified"] = False
    elif corruption == "checkpoint":
        result["checkpoint_sha256"] = "different"
    elif corruption == "source":
        result["source"]["git_head"] = "different"
    elif corruption == "cell":
        result["rounds"] = 8
    elif corruption == "seeds":
        result["episodes"][0]["seed"] = 999
    elif corruption == "incomplete":
        result["episodes"][0].update(length=100, truncated=False)
    elif corruption == "nonfinite":
        result["episodes"][0]["return"] = float("nan")
    _save(directory / "results.json", result)
    with pytest.raises(ValueError):
        validate_result(directory, campaign, cell)


def test_pairing_rejects_changed_solver_seed_even_when_environment_seeds_match(tmp_path):
    campaign, cell, _, _, result = fixture(tmp_path)
    transferred = dict(cell, index=1, name="h1_j1_rho_a1_c0", arm="rho_a1_c0")
    campaign["cells"].append(transferred)
    donor_episodes = deepcopy(result["episodes"])
    donor_episodes[0]["solver_seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        snapshot(tmp_path, campaign, False, {cell["name"]: result["episodes"],
                                          transferred["name"]: donor_episodes})


@pytest.mark.parametrize("corruption", ["result", "manifest"])
def test_completed_receipt_binds_both_result_and_manifest(tmp_path, corruption):
    campaign, _, directory, manifest, result = fixture(tmp_path)
    value = manifest if corruption == "manifest" else result
    value["unreviewed_addition"] = True
    _save(directory / (corruption + ".json" if corruption == "manifest" else "results.json"), value)
    with pytest.raises(ValueError, match="binding"):
        snapshot(tmp_path, campaign, False, {})


def test_layout_preserves_other_sections_views_and_is_idempotent(tmp_path):
    service = Service()
    before_spec = existing_layout.patch_actor_transfer_spec(sample_spec())
    service.views[0]["spec"] = json.dumps(before_spec)
    before_views = deepcopy(service.views)
    kwargs = dict(entity="entity", project="project", receipt_dir=tmp_path,
                  run_id="discovery-overview")
    api = SimpleNamespace(_service_api=service)
    first = layout.ensure_discovery_results_layout(api, **kwargs)
    assert first["status"] == "verified" and first["changed"]
    assert service.views[1:] == before_views[1:]
    installed = json.loads(service.views[0]["spec"])
    assert layout._without_owned(installed) == before_spec
    assert layout.patch_discovery_spec(installed) == installed
    second = layout.ensure_discovery_results_layout(api, **kwargs)
    assert second["changed"] is False and service.writes == 1
    assert second["url"] == "https://wandb.ai/entity/project/runs/discovery-overview?nw=nwuserrwgao_b"


def test_layout_uses_exact_publisher_chart_and_table_keys():
    sections = layout.discovery_sections()
    panels = [panel for section in sections for panel in section["panels"]]
    charts = [p for p in panels if p["viewType"] == "Vega2"]
    keys = [p["config"]["userQuery"]["queryFields"][0]["fields"][0]["args"][0]["value"] for p in charts]
    assert keys == [key + "_table" for key in layout.CHART_KEYS]
    assert tuple(p["config"]["mediaKeys"][0] for p in panels if p["viewType"] == "Media Browser") == layout.TABLE_KEYS
    assert all(section["isOpen"] and not section["isPanelsAuto"] for section in sections)
    intro = panels[0]["config"]["value"]
    assert "Partial episodes are progress only" in intro
    assert "three paired development seeds" in intro


def test_publisher_destination_defaults_and_explicit_override():
    arguments = ["--root", "/campaign", "--publication-root", "/publication", "--gpu-job-id", "123"]
    defaults = publisher.parser().parse_args(arguments)
    assert defaults.entity == "rwgao_b-brown-university"
    assert defaults.project == "ambi-inner-bench"
    selected = publisher.parser().parse_args(arguments + ["--entity", "another-team", "--project", "another-project"])
    assert selected.entity == "another-team" and selected.project == "another-project"


def _publication_fixture(tmp_path):
    root, out = tmp_path / "campaign", tmp_path / "publication"
    campaign = {"source_commit": "original-evaluation-source"}
    _save(root / "campaign.json", campaign)
    return root, out, campaign


def test_publication_state_binds_destination_and_evaluation_source_and_resumes(tmp_path):
    root, out, campaign = _publication_fixture(tmp_path)
    first = publisher.publication_state(root, out, campaign, entity="team", project="ambi-inner-bench")
    path = out / "publication.json"
    before = path.read_bytes()
    assert first["entity"] == "team" and first["project"] == "ambi-inner-bench"
    assert first["campaign_sha256"] == digest(root / "campaign.json")
    assert first["source_commit"] == campaign["source_commit"]
    assert first["campaign_root"] == str(root)
    second = publisher.publication_state(root, out, campaign, entity="team", project="ambi-inner-bench")
    assert second == first and path.read_bytes() == before


@pytest.mark.parametrize("field", ["entity", "project", "campaign_sha256", "source_commit", "campaign_root"])
def test_publication_state_refuses_changed_binding_without_rewriting(tmp_path, field):
    root, out, campaign = _publication_fixture(tmp_path)
    state = publisher.publication_state(root, out, campaign, entity="team", project="ambi-inner-bench")
    state[field] = "different"
    _save(out / "publication.json", state)
    before = (out / "publication.json").read_bytes()
    with pytest.raises(ValueError, match="binding differs"):
        publisher.publication_state(root, out, campaign, entity="team", project="ambi-inner-bench")
    assert (out / "publication.json").read_bytes() == before


def test_legacy_unbound_publication_requires_new_root(tmp_path):
    root, out, campaign = _publication_fixture(tmp_path)
    legacy = dict(campaign_sha256=digest(root / "campaign.json"), run_id="existing-ambi-run",
                  source_commit=campaign["source_commit"], campaign_root=str(root))
    _save(out / "publication.json", legacy)
    before = (out / "publication.json").read_bytes()
    with pytest.raises(ValueError, match="Legacy.*new publication root"):
        publisher.publication_state(root, out, campaign, entity="team", project="ambi-inner-bench")
    assert (out / "publication.json").read_bytes() == before
    new = publisher.publication_state(root, tmp_path / "new-publication", campaign,
                                      entity="team", project="ambi-inner-bench")
    assert new["run_id"] != legacy["run_id"]


def test_destination_override_reaches_wandb_and_layout_without_changing_gpu_outputs(tmp_path, monkeypatch):
    root, out = tmp_path / "campaign", tmp_path / "publication"
    fixture(root)
    original = {str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}
    calls = {}
    run = SimpleNamespace(id="new-reporting-run", summary={}, log=lambda value: None,
        finish=lambda **kwargs: None,
        log_artifact=lambda artifact: SimpleNamespace(wait=lambda: None))

    def initialize(**kwargs):
        calls["init"] = kwargs
        return run

    def install(api, **kwargs):
        calls["layout"] = kwargs
        return {"status": "verified", "url": "https://wandb.ai/selected-team/selected-project/runs/new-reporting-run"}

    fake_wandb = SimpleNamespace(init=initialize, Api=lambda **kwargs: object(),
        Artifact=lambda *args, **kwargs: SimpleNamespace(add_file=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(layout, "ensure_discovery_saved_view", install, raising=False)
    monkeypatch.setattr(publisher, "gpu_jobs_active", lambda jobs: False)
    monkeypatch.setattr(publisher, "payload", lambda *args: {})
    publisher.watch(SimpleNamespace(root=root, publication_root=out, entity="selected-team",
        project="selected-project", gpu_job_id=["123"], poll_seconds=1, once=True))
    for name in ("init", "layout"):
        assert calls[name]["entity"] == "selected-team"
        assert calls[name]["project"] == "selected-project"
    assert calls["init"]["config"]["publication_id"] == calls["init"]["id"]
    assert {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()} == original


@pytest.mark.parametrize("operation_name", ["read", "digest", "write", "snapshot"])
def test_transient_stale_file_handle_recovers_at_publication_io_boundaries(tmp_path, monkeypatch, operation_name):
    campaign, _, _, _, _ = fixture(tmp_path)
    delays, attempts = [], []
    monkeypatch.setattr(publisher.time, "sleep", delays.append)
    if operation_name == "snapshot":
        target = "_validate_result"
        operation = lambda: publisher.snapshot(tmp_path, campaign, False, {})
    elif operation_name == "write":
        target = "atomic_json"
        operation = lambda: publisher.write(tmp_path / "publication" / "progress.json", {"complete": 1})
    else:
        target = "_" + operation_name
        operation = lambda: getattr(publisher, operation_name)(tmp_path / "campaign.json")
    original = getattr(publisher, target)

    def transient(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError(errno.ESTALE, "stale file handle")
        return original(*args, **kwargs)

    monkeypatch.setattr(publisher, target, transient)
    result = operation()
    assert len(attempts) == 2 and delays == [1]
    if operation_name == "snapshot":
        assert result["completed"] == 1
    elif operation_name == "write":
        assert json.loads((tmp_path / "publication" / "progress.json").read_text()) == {"complete": 1}


def test_permanent_stale_file_handle_stops_after_five_attempts_with_original_error(monkeypatch):
    failure = OSError(errno.ESTALE, "persistent stale handle")
    attempts, delays = [], []
    monkeypatch.setattr(publisher.time, "sleep", delays.append)

    def fail():
        attempts.append(True)
        raise failure

    with pytest.raises(OSError) as caught:
        publisher._retry_estale(fail)
    assert caught.value is failure
    assert len(attempts) == 5 and delays == [1, 2, 4, 8]


@pytest.mark.parametrize("failure", [OSError(errno.EACCES, "permission denied"),
                                     OSError(errno.EIO, "I/O failure"), ValueError("invalid receipt")])
def test_other_failures_propagate_immediately_without_retry(monkeypatch, failure):
    attempts, delays = [], []
    monkeypatch.setattr(publisher.time, "sleep", delays.append)

    def fail():
        attempts.append(True)
        raise failure

    with pytest.raises(type(failure)) as caught:
        publisher._retry_estale(fail)
    assert caught.value is failure and len(attempts) == 1 and delays == []


def test_secondary_reporting_failures_do_not_replace_original_failure(tmp_path, monkeypatch, capsys):
    original = ValueError("receipt binding differs")
    calls = []

    def fail_write(*args, **kwargs):
        calls.append("receipt")
        raise OSError(errno.EACCES, "publication directory unavailable")

    def fail_summary(value):
        calls.append("summary")
        raise RuntimeError("W&B unavailable")

    def finish(**kwargs):
        calls.append("finish")
        assert kwargs == {"exit_code": 1}

    monkeypatch.setattr(publisher, "write", fail_write)
    run = SimpleNamespace(summary=SimpleNamespace(update=fail_summary), finish=finish)
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        except BaseException as error:
            publisher._record_failure(run, tmp_path, error)
            raise
    assert caught.value is original
    assert calls == ["receipt", "summary", "finish"]
    assert "Original publisher failure: ValueError: receipt binding differs" in capsys.readouterr().err
