"""Only validated, paired production episodes may enter the native KL view."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_prior_kl_publish as pub
from utils import eval_series as series
from utils import eval_series_data as data


@pytest.fixture
def campaign(tmp_path):
    campaign = dict(campaign_id="prior-kl-episodes-20261020", source_commit="a"*40,
        seeds=[101,102,103,104,105], controller_seed=55, max_steps=500, execution_action="sample", cells=[])
    for index, name in enumerate(pub.CURVES):
        source = "entity/project/" + name
        selector = "prior_kl/" + name
        identity = dict(backbone=source, science={"version":1,"algorithm":"AMBITDMPC2/AMBITDMPC2","source_sha256":"b"*64},
            planner=dict(type="sac", action_rule="tanh_sample", settings={"inner_rollout_horizon":3,"inner_rounds":8}),
            protocol=dict(action_rule="tanh_sample",environment_seeds=campaign["seeds"],max_steps=500,controller_seed=55))
        registry = series.create_run(tmp_path/"runs", dict(identity=identity, selector=selector,label="KL"),
            campaign["campaign_id"],pub.PROJECT,pub.ENTITY,pub.OWNER)
        references = {}
        for kind in ("reference","baseline"):
            path = tmp_path / name / kind / "manifest.json"
            series._atomic_json(path, {"reference":kind})
            references.update({kind+"_bundle":str(path.parent),kind+"_manifest_sha256":series._file_digest(path)})
        campaign["cells"].append(dict(index=index, checkpoint_id=name, selector=selector, source_run=source,
            step=100000 if name=="sac_aux_100k" else 400000, checkpoint_sha256="c"*64,
            output_dir=str(tmp_path/name/"output"), inventory_path=str(tmp_path/"inventory.json"),
            run_id=registry["run_id"], run_dir=registry["run_dir"], **references))
    path=tmp_path/"campaign.json"
    series._atomic_json(path,campaign)
    assert pub.load_campaign(path)==campaign
    return campaign


def complete(campaign, monkeypatch):
    cell=campaign["cells"][0]
    output=Path(cell["output_dir"])
    manifest=output/"bundle/manifest.json"
    series._atomic_json(manifest,{"runs":[dict(selector=cell["selector"],status="complete",trace_files=["trace.gz"])]})
    trace=manifest.parent/"trace.gz"
    trace.write_bytes(b"full trace stays remote")
    results=output/"results.json"
    series._atomic_json(results,{"complete":True})
    receipt=dict(validated=True,smoke=False,checkpoint_id=cell["checkpoint_id"],source_commit=campaign["source_commit"],
        manifest_sha256=series._file_digest(manifest),results_sha256=series._file_digest(results))
    series._atomic_json(output/"receipt.json",receipt)
    record=dict(identity=series.load_run(cell["run_dir"])["identity"],checkpoint={"step":cell["step"],"sha256":cell["checkpoint_sha256"]},
        metrics={"eval/return_mean":7.,"eval/episodes":5},record_id="paired",label="KL",selector=cell["selector"],
        episodes=[dict(seed=s,return_=7.) for s in campaign["seeds"]],provenance={},
        artifact_files={"manifest.json":str(manifest),trace.name:str(trace)},source_result_path=str(manifest))
    record["episodes"]=[{"seed":s,"return":7.,"paired_return_delta":2.} for s in campaign["seeds"]]
    monkeypatch.setattr(data,"load_records",lambda *a,**k:[deepcopy(record)])
    return cell,receipt,record,trace


def test_stages_summary_only_once_and_preserves_exact_prior(campaign,monkeypatch):
    cell,receipt,record,trace=complete(campaign,monkeypatch)
    status=pub.stage_completed(campaign)
    assert status[cell["checkpoint_id"]]=="complete"
    staged=series._read(Path(cell["run_dir"])/"records/paired.json")
    assert "trace.gz" not in staged["artifact_files"]
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


def test_pending_writes_only_two_new_run_summaries(campaign,tmp_path,monkeypatch):
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
    result=pub.publish_once(campaign,tmp_path/"pub",workspace_verifier=lambda *a:{"url":"verified"})
    assert len(sessions)==2
    assert {directory for directory,_ in sessions}=={c["run_dir"] for c in campaign["cells"]}
    assert not result["all_complete"]
    for _,run in sessions:
        assert run.summary["progress/checkpoints_complete"]==0
        assert not any(key.startswith("eval/") for key in run.summary)


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
    assert first["api_verified"] and first["panel_count"]==11
    saved[-1].sections.append(ws.Section(name="User notes",panels=[]))
    second=pub.install_workspace(campaign,path)
    assert second["url"]==first["url"]
    assert len(saved[-1].sections)==4
    panels=[p for s in saved[-1].sections for p in s.panels if isinstance(p,wr.LinePlot)]
    assert len(panels)==8 and all(not p.aggregate for p in panels)
    assert len(second["runs"])==4
