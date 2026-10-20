"""Only validated, paired production episodes may enter the native entropy-plus-KL view."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_entropy_kl_publish as pub
from slurm import ambi_entropy_kl_episodes as driver
from utils import eval_series as series
from utils import eval_series_data as data


@pytest.fixture
def campaign(tmp_path):
    campaign = dict(campaign_id="entropy-kl-episodes-20261020", source_commit="a"*40,
        matrix_sha256=series._file_digest(pub.MATRIX_PATH), profile_sha256=series._file_digest(pub.PROFILE_PATH),
        seeds=[101,102,103,104,105], controller_seed=55, max_steps=500, execution_action="sample", cells=[])
    profile = series._read(pub.PROFILE_PATH)
    inventory = tmp_path / "inventory.json"
    series._atomic_json(inventory, {"pinned": True})
    for name, (step, _) in pub.CHECKPOINTS.items():
        for token, multiplier in pub.MULTIPLIERS.items():
            beta = pub.OUTER_ALPHA[name] * multiplier
            cell_id = name + "__beta" + token
            source = profile["checkpoints"][name]["source_run"]
            selector = "entropy_kl/" + name + "_beta" + token
            settings = dict(inner_rollout_horizon=3, inner_rounds=8, inner_rollouts_per_round=128,
                inner_batch_size=256, inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
                inner_entropy_enabled=True, inner_temperature_mode="auto", inner_temperature_initialization="inherit_outer",
                inner_temperature_lr=pub.TEMPERATURE_LR, inner_actor_lr=0.0003, inner_critic_lr=0.0003, inner_outer_policy_kl_coef=beta,
                inner_actor_source="sac", inner_horizon_actor_source="sac",
                inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
                inner_sac_critic_target="reward_only")
            identity = dict(backbone=source, science={"version":1,"algorithm":"AMBITDMPC2/AMBITDMPC2","source_sha256":"b"*64},
                planner=dict(type="sac", action_rule="tanh_sample", settings=settings),
                protocol=dict(action_rule="tanh_sample",environment_seeds=campaign["seeds"],max_steps=500,controller_seed=55))
            registry = series.create_run(tmp_path/"runs", dict(identity=identity, selector=selector,label="entropy + KL"),
                campaign["campaign_id"],pub.PROJECT,pub.ENTITY,pub.OWNER)
            references = {}
            for kind in ("reference","baseline"):
                path = tmp_path / name / kind / "manifest.json"
                series._atomic_json(path, {"reference":kind})
                references.update({kind+"_bundle":str(path.parent),kind+"_manifest_sha256":series._file_digest(path)})
            campaign["cells"].append(dict(**profile["checkpoints"][name], index=len(campaign["cells"]), checkpoint_id=name, cell_id=cell_id,
                temperature_lr=pub.TEMPERATURE_LR, beta_multiplier=multiplier, beta=beta, selector=selector, inventory_sha256=series._file_digest(inventory),
                output_dir=str(tmp_path/cell_id/"output"), inventory_path=str(tmp_path/"inventory.json"),
                run_id=registry["run_id"], run_dir=registry["run_dir"], **references))
    path=tmp_path/"campaign.json"
    series._atomic_json(path,campaign)
    assert pub.load_campaign(path)==campaign
    return campaign


def complete(campaign, monkeypatch):
    cell=campaign["cells"][0]
    output=Path(cell["output_dir"])
    manifest=output/"bundle/manifest.json"
    series._atomic_json(manifest,{"runs":[dict(selector=cell["selector"],status="complete",trace_files=["trace.jsonl.gz"])]})
    trace=manifest.parent/"trace.jsonl.gz"
    trace.write_bytes(b"full trace stays remote")
    results=output/"results.json"
    series._atomic_json(results,{"complete":True})
    receipt=dict(validated=True,smoke=False,checkpoint_id=cell["checkpoint_id"],cell_id=cell["cell_id"],temperature_lr=cell["temperature_lr"],beta=cell["beta"],beta_multiplier=cell["beta_multiplier"],source_commit=campaign["source_commit"],
        manifest_sha256=series._file_digest(manifest),results_sha256=series._file_digest(results),
        trace_sha256=driver.trace_hashes(manifest.parent))
    series._atomic_json(output/"receipt.json",receipt)
    record=dict(identity=series.load_run(cell["run_dir"])["identity"],checkpoint={"step":cell["step"],"sha256":cell["checkpoint_sha256"]},
        metrics={**{metric: 7. for _, metric in pub.METRICS}, "eval/episodes":5, "eval/frozen_state_unchanged":True, "work/temperature_updates":80000, "work/actor_updates":80000, "work/critic_updates":320000,
                 "work/environment_decisions":2500, "diagnostics/inner_alpha_initial_mean":pub.OUTER_ALPHA[cell["checkpoint_id"]]},record_id="paired",label="KL",selector=cell["selector"],
        episodes=[dict(seed=s,return_=7.) for s in campaign["seeds"]],provenance={},
        artifact_files={"manifest.json":str(manifest),trace.name:str(trace)},source_result_path=str(manifest))
    record["episodes"]=[{"seed":s,"return":7.,"paired_return_delta":2.} for s in campaign["seeds"]]
    monkeypatch.setattr(data,"load_records",lambda *a,**k:[deepcopy(record)])
    return cell,receipt,record,trace


def test_stages_summary_only_once_and_preserves_exact_prior(campaign,monkeypatch):
    cell,receipt,record,trace=complete(campaign,monkeypatch)
    status=pub.stage_completed(campaign)
    assert status[cell["cell_id"]]=="complete"
    staged=series._read(Path(cell["run_dir"])/"records/paired.json")
    assert "trace.jsonl.gz" not in staged["artifact_files"]
    assert staged["metrics"]["reference/return_mean"]==5.
    index=series._read(staged["artifact_files"]["remote-traces.json"])
    assert index["files"][0]["sha256"]==series._file_digest(trace)
    monkeypatch.setattr(data,"load_records",lambda *a,**k:pytest.fail("repeat normalization"))
    assert pub.stage_completed(campaign)==status
    series._atomic_json(Path(cell["output_dir"])/"receipt.json",{**receipt,"changed":True})
    with pytest.raises(ValueError,match="Previously accepted"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("change",[{"smoke":True},{"validated":False},{"source_commit":"z"*40}])
def test_rejects_smoke_unvalidated_or_wrong_source(campaign,monkeypatch,change):
    cell,receipt,_,_=complete(campaign,monkeypatch)
    series._atomic_json(Path(cell["output_dir"])/"receipt.json",{**receipt,**change})
    with pytest.raises(ValueError,match="validated production"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("change", [{"cell_id":"sac_aux_100k__beta3"}, {"temperature_lr":0.03}, {"beta_multiplier":3}, {"beta":0.01}])
def test_rejects_receipt_from_another_beta(campaign,monkeypatch,change):
    cell,receipt,_,_=complete(campaign,monkeypatch)
    series._atomic_json(Path(cell["output_dir"])/"receipt.json",{**receipt,**change})
    with pytest.raises(ValueError,match="validated production"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("field,value", [("temperature_lr",0.003), ("beta_multiplier",2), ("beta",0.1), ("selector","entropy_kl/sac_aux_100k_beta3"),
    ("step",575000), ("checkpoint_sha256","bad")])
def test_campaign_rejects_wrong_beta_axis_or_pins(campaign,tmp_path,field,value):
    campaign["cells"][0][field]=value
    path=tmp_path/"changed.json"
    series._atomic_json(path,campaign)
    with pytest.raises(ValueError):
        pub.load_campaign(path)


def test_unique_cells_and_labels(campaign,tmp_path):
    rows=pub.EXISTING+pub.curve_rows(campaign)
    assert len({r["name"] for r in rows})==8
    assert len({r["color"] for r in rows})==8
    campaign["cells"][1]=deepcopy(campaign["cells"][0])
    path=tmp_path/"duplicate.json"
    series._atomic_json(path,campaign)
    with pytest.raises(ValueError,match="two KL strengths"):
        pub.load_campaign(path)


def test_registry_checks_entropy_or_beta_before_pending_publication(campaign):
    campaign["cells"][0]["temperature_lr"]=0.3
    with pytest.raises(ValueError,match="entropy-plus-KL controller"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("field,value", [("work/temperature_updates",0), ("eval/frozen_state_unchanged",False)])
def test_requires_temperature_steps_and_frozen_outer(campaign,monkeypatch,field,value):
    _,_,record,_=complete(campaign,monkeypatch)
    record["metrics"][field]=value
    with pytest.raises(ValueError,match="32-temperature-step"):
        pub.stage_completed(campaign)


def test_missing_temperature_metric_is_not_silently_published(campaign,monkeypatch):
    _,_,record,_=complete(campaign,monkeypatch)
    del record["metrics"]["diagnostics/inner_alpha_final_mean"]
    with pytest.raises(ValueError,match="alpha/entropy/KL metrics"):
        pub.stage_completed(campaign)


def test_rejects_changed_files_missing_seed_and_missing_pair(campaign,monkeypatch):
    cell,receipt,record,_=complete(campaign,monkeypatch)
    record["episodes"].pop()
    with pytest.raises(ValueError,match="all five"):
        pub.stage_completed(campaign)
    record["episodes"].append({"seed":105,"return":7.})
    with pytest.raises(ValueError,match="sampled-prior"):
        pub.stage_completed(campaign)
    series._atomic_json(Path(cell["output_dir"])/"results.json",{"changed":True})
    with pytest.raises(ValueError,match="changed"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("require_complete", [False, True])
def test_pending_writes_only_four_new_run_summaries(campaign,tmp_path,monkeypatch,require_complete):
    sessions=[]
    class Publisher:
        def __init__(self,run_dir,*,owner,**kwargs):
            assert owner==pub.OWNER
            self.run=SimpleNamespace(name=None,config={},summary={},define_metric=lambda *a,**k:None)
            sessions.append((run_dir,self.run))
        def __enter__(self): return self
        def __exit__(self,*a): return False
        def publish_pending(self): return {}
    monkeypatch.setattr(series,"Publisher",Publisher)
    result=pub.publish_once(campaign,tmp_path/"pub",workspace_verifier=lambda *a:{"url":"verified"},
                            require_complete=require_complete)
    assert len(sessions)==4
    assert {directory for directory,_ in sessions}=={c["run_dir"] for c in campaign["cells"]}
    assert not result["all_complete"]
    for _,run in sessions:
        assert run.summary["progress/checkpoints_complete"]==0
        assert run.summary["progress/failed_cells"]==int(require_complete)
        assert run.summary["progress/status"]==("failed" if require_complete else "pending results")
        assert ("progress/failure_reason" in run.summary)==require_complete
        assert not any(key.startswith("eval/") for key in run.summary)
    assert set(result["cells"].values())=={"failed" if require_complete else "pending"}


def test_final_cli_fails_after_publishing_terminal_status(campaign,tmp_path,monkeypatch):
    campaign_path=tmp_path/"campaign.json"
    root=tmp_path/"pub"
    calls=[]
    def publish(*args, **kwargs):
        calls.append(kwargs)
        return {"all_complete":False,"cells":{c["checkpoint_id"]:"failed" for c in campaign["cells"]}}
    monkeypatch.setattr(pub,"publish_once",publish)
    monkeypatch.setattr(pub.sys,"argv",["publisher","publish","--campaign",str(campaign_path),
        "--root",str(root),"--once","--require-complete"])
    with pytest.raises(SystemExit,match="marked failed"):
        pub.main()
    assert calls==[{"require_complete":True}]
    assert "marked failed" in series._read(root/"publication-failure.json")["message"]


def test_missing_layout_fails_before_any_wandb_mutation(campaign,tmp_path,monkeypatch):
    monkeypatch.setattr(series,"Publisher",lambda *a,**k:pytest.fail("mutation before layout"))
    with pytest.raises(ValueError,match="native layout"):
        pub.publish_once(campaign,tmp_path/"pub")


def test_native_layout_is_idempotent_and_readback_verified_without_network(campaign,tmp_path,monkeypatch):
    ws=pytest.importorskip("wandb_workspaces.workspaces")
    wr=pytest.importorskip("wandb_workspaces.reports.v2")
    if pub.importlib.metadata.version("wandb")!="0.28.1" or pub.importlib.metadata.version("wandb-workspaces")!="0.4.5":
        pytest.skip("Run layout validation in its pinned separate SDK runtime")
    saved=[]
    monkeypatch.setattr(ws.Workspace,"save",lambda self:saved.append(self))
    monkeypatch.setattr(ws.Workspace,"url",property(lambda self:f"https://wandb.ai/{pub.ENTITY}/{pub.PROJECT}?nw=testview"))
    monkeypatch.setattr(ws.Workspace,"from_url",classmethod(lambda cls,url:saved[-1]))
    path=tmp_path/"workspace.json"
    first=pub.install_workspace(campaign,path)
    assert first["api_verified"] and first["panel_count"]==pub.PANEL_COUNT
    saved[-1].sections.append(ws.Section(name="User notes",panels=[]))
    second=pub.install_workspace(campaign,path)
    assert second["url"]==first["url"]
    assert len(saved[-1].sections)==4
    panels=[p for s in saved[-1].sections for p in s.panels if isinstance(p,wr.LinePlot)]
    assert len(panels)==12 and all(not p.aggregate and len(p.line_marks)==8 for p in panels)
    assert len(second["runs"])==8


def test_rejects_changed_inventory_before_any_pending_publication(campaign):
    series._atomic_json(campaign["cells"][0]["inventory_path"], {"changed": True})
    with pytest.raises(ValueError, match="inventory changed"):
        pub.stage_completed(campaign)


def test_rejects_changed_validated_trace(campaign, monkeypatch):
    _, _, _, trace = complete(campaign, monkeypatch)
    trace.write_bytes(b"different trace")
    with pytest.raises(ValueError, match="validated full trace changed"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("field,value", [
    ("diagnostics/inner_final_outer_policy_kl_mean", float("nan")),
    ("diagnostics/inner_actor_entropy_max_mean", None),
    ("work/actor_updates", 79999),
    ("work/critic_updates", 1),
    ("work/environment_decisions", 2499),
])
def test_joint_diagnostics_and_complete_work_are_required(campaign, monkeypatch, field, value):
    _, _, record, _ = complete(campaign, monkeypatch)
    record["metrics"][field] = value
    with pytest.raises(ValueError, match="alpha/entropy/KL metrics"):
        pub.stage_completed(campaign)


def test_inherited_alpha_is_the_pinned_value(campaign, monkeypatch):
    _, _, record, _ = complete(campaign, monkeypatch)
    record["metrics"]["diagnostics/inner_alpha_initial_mean"] *= 3
    with pytest.raises(ValueError, match="frozen checkpoint"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("field,value", [("source_run", "different/run"), ("checkpoint_sha256", "d" * 64)])
def test_exact_profile_identity_is_required(campaign, tmp_path, field, value):
    campaign["cells"][0][field] = value
    path = tmp_path / "changed.json"
    series._atomic_json(path, campaign)
    with pytest.raises(ValueError, match="pinned source checkpoint"):
        pub.load_campaign(path)


def test_existing_comparison_run_ids_cannot_be_reused(campaign, tmp_path):
    campaign["cells"][0]["run_id"] = pub.EXISTING[0]["run_id"]
    path = tmp_path / "changed.json"
    series._atomic_json(path, campaign)
    with pytest.raises(ValueError, match="existing comparison run IDs"):
        pub.load_campaign(path)


def test_summary_artifact_is_size_bounded(campaign, monkeypatch):
    cell, _, record, _ = complete(campaign, monkeypatch)
    large = Path(cell["output_dir"]) / "large-summary.json"
    with large.open("wb") as handle:
        handle.truncate(16 * 1024 * 1024)
    record["artifact_files"][large.name] = str(large)
    with pytest.raises(ValueError, match="16 MiB"):
        pub.stage_completed(campaign)


def test_all_four_driver_interventions_match_publication_cells(campaign):
    for cell in campaign["cells"]:
        expected = driver.intervention(cell["checkpoint_id"], cell["beta_multiplier"])
        assert all(cell[key] == value for key, value in expected.items())
        pub.validate_registry(cell)
