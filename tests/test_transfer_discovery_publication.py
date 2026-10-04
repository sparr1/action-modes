"""Offline publication gates prevent incomplete or mismatched discovery claims."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm.ambi_transfer_discovery_campaign import digest, validate_result
from slurm.ambi_transfer_discovery_publish import snapshot
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
