"""Strict native-XQC aggregation, immutable prior reuse, and publication cleanup."""

import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import summarize_ambixqc_inner_eval as campaign
from utils import ambi_benchmark as storage

NEW_SHA = "1" * 40
SEEDS = [101, 102]


def _saved_config():
    return {"alg": "AMBIXQC/AMBIXQC", "env": "DMControl-v0", "seed": 55, "alg_params": {
        "inner_operator": "none", "inner_rounds": 2, "inner_rollouts_per_round": 32,
        "inner_rollout_horizon": 3, "inner_updates_per_round": 4, "inner_batch_size": 64,
        "inner_replay_capacity": 192, "inner_actor_lr": 5e-5, "inner_critic_lr": 5e-5,
        "xqc_policy_delay": 3, "obs": "state", "wandb": True,
    }}


def _semantic_signature():
    return {"algorithm": "AMBIXQC", "collection_operator": "none", "inner_lifecycle": "fresh_per_action",
            "reward_normalization": "real_discounted_return_only", "actor_arch": [256,256,256,256],
            "inner_schedule": {"rounds":2,"rollouts":32,"horizon":3,"updates":4,"batch_size":64,
                               "replay_capacity":192,"replay_sampling":"with_replacement","actor_lr":5e-5,"critic_lr":5e-5}}


def _make_bundle(path, metadata, checkpoint, *, kind, monkeypatch, reference=None):
    code = {"commit": campaign.REFERENCE_SHA if kind == "prior" else NEW_SHA, "dirty": False,
            "source_sha256":"c"*64, "diff_sha256":"d"*64, "runtime":{"python":"3.10","torch":"2.3.1"}}
    monkeypatch.setattr(storage, "code_identity", lambda: code)
    config = copy.deepcopy(metadata["trial_run_params"])
    if kind == "xqc":
        config["alg_params"].update(campaign.BUDGET)
    resolved = {"selector":f"controller/{kind}", "algorithm_config":config,
                "environment":{"id":"DMControl-v0","params":{"task":"humanoid-walk","obs":"state","render_mode":None}}}
    bundle = storage.BenchmarkBundle(path,checkpoint={**checkpoint,"metadata":metadata},
                                     protocol=storage.protocol_for(resolved,12345,2),reference=reference)
    run = bundle.start_run(resolved,"episodes")
    if reference is not None:
        bundle.manifest["reference"]={"path":"/original/reference/manifest.json","manifest_sha256":checkpoint["reference_manifest_sha256"]}
    run["outer_state_unchanged"]=True
    for seed, prior, gain in ((101,10.,3.),(102,20.,5.)):
        value=prior+(gain if kind=="xqc" else 0)
        elapsed=.2 if kind=="xqc" else .02
        counts=campaign.COUNTS if kind=="xqc" else {key:0 for key in campaign.COUNTS}
        events=[{"episode_id":f"seed-{seed}","decision_index":index,"event_index":0,"phase":"decision","round_index":6,
                 **{f"{key}_updates":count for key,count in counts.items()},"metrics":{
                     "decision/reward":value/2,"decision/control_seconds":elapsed/2,
                     "decision/inner_model_steps":9216 if kind=="xqc" else 0,
                     "decision/inner_reward_normalizer_imagined_updates":0,
                     **{f"decision/inner_{key}_optimizer_steps":count for key,count in counts.items()}}}
                for index in range(2)]
        bundle.episode(run,{"seed":seed,"solver_seed":storage.solver_seed(12345,"episode",seed),"return":value,
                            "length":2,"control_seconds":elapsed,"terminated":False,"truncated":True,
                            "truncated_by_evaluator":True,"model_metrics":{}},events)
    saved_signature=_semantic_signature()
    evaluated_signature=copy.deepcopy(saved_signature)
    if kind=="xqc":
        evaluated_signature["collection_operator"]="xqc"
        evaluated_signature["inner_schedule"].update(rounds=6,rollouts=512,horizon=3,updates=3,batch_size=512,replay_capacity=9216)
    evaluated=copy.deepcopy(config)
    evaluated.update(seed=12345,device="cuda")
    evaluated["alg_params"].update(wandb=False,device="cuda")
    bundle.finish_run(run,{"outer_state_unchanged":True,"outer_updates_before":checkpoint["step"],
                           "outer_updates_after":checkpoint["step"],"controller":kind,"environment_seeds":SEEDS,
                           "controller_seed":12345,"seed_scheme":"sha256-v1","action_rule":"tanh_mean","deterministic_execution":True,
                           "saved_algorithm_config":metadata["trial_run_params"],"evaluated_algorithm_config":evaluated,
                           "resolved_config":{**config["alg_params"],"inner_reward_normalization":"frozen_real_scale"},
                           "checkpoint_evaluation_provenance":{"frozen_evaluation":True,"saved_semantic_signature":saved_signature,
                                                               "evaluated_semantic_signature":evaluated_signature}})
    bundle.finish()
    return bundle


@pytest.fixture
def outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,"STEPS",(50000,100000))
    references,results=tmp_path/"references",tmp_path/"results"
    entries=[]
    for step,digit in ((50000,"a"),(100000,"b")):
        entry={"step":step,"path":f"/saved/{step}.pt","sha256":digit*64,"metadata_sha256":"e"*64,
               "reference_bundle":str(references/"production"/f"step_{step}"/"bundle")}
        metadata={"checkpoint":{"step":step},"trial_run_params":_saved_config()}
        prior=_make_bundle(Path(entry["reference_bundle"]),metadata,entry,kind="prior",monkeypatch=monkeypatch)
        entry["reference_manifest_sha256"]=campaign.file_sha256(prior.path/"manifest.json")
        _make_bundle(results/"production"/f"step_{step}"/"bundle",metadata,entry,kind="xqc",monkeypatch=monkeypatch,reference={101:10.,102:20.})
        entries.append(entry)
    manifest=tmp_path/"inventory.json"
    storage.atomic_json(manifest,{"source_run":campaign.SOURCE_RUN,"checkpoints":entries})
    for entry in entries:
        storage.atomic_json(results/"production"/f'step_{entry["step"]}'/"provenance.json",{
            "checkpoint":entry,"source_run":campaign.SOURCE_RUN,"checkpoint_manifest_sha256":campaign.file_sha256(manifest),
            "matrix_sha256":campaign.file_sha256(campaign.MATRIX),"mode":"production","seeds":SEEDS,"max_steps":2,"controller_seed":12345,
            "numerical_settings":campaign.NUMERICAL_SETTINGS,
            "reference_bundle":entry["reference_bundle"],"reference_manifest_sha256":entry["reference_manifest_sha256"]})
        storage.atomic_json(results/"production"/f'step_{entry["step"]}'/"paired.json",{
            "checkpoint_sha256":entry["sha256"],"matrix_sha256":campaign.file_sha256(campaign.MATRIX),
            "numerical_settings":campaign.NUMERICAL_SETTINGS})
    return manifest,results,references


def _summarize(outputs,**kwargs):
    return campaign.summarize(*outputs[:2],expected_source_sha=NEW_SHA,seeds=SEEDS,max_steps=2,**kwargs)


def _change(outputs,change):
    path=outputs[1]/"production/step_100000/bundle/manifest.json"
    manifest=json.loads(path.read_text())
    change(manifest)
    storage.atomic_json(path,manifest,overwrite=True)


def test_native_inner_xqc_paired_statistics_and_truthful_html(outputs):
    summary=_summarize(outputs)
    assert summary["status"]=="complete" and len(summary["rows"])==2
    assert summary["variant"]=="inner"
    row=summary["rows"][0]
    assert row["prior"]["return_mean"]==15 and row["prior"]["return_std"]==5
    assert row["inner_xqc"]["return_mean"]==19 and row["inner_xqc"]["return_std"]==6
    assert row["paired"]=={"delta_mean":4,"delta_std":1,"deltas":[3,5]}
    assert row["inner_xqc"]["control_seconds_per_decision"]==.1
    assert row["actual_optimizer_steps"]=={"critic":72,"actor":24,"temperature":24}
    rendered=campaign.render_html(summary)
    assert "C18/A6/T6" in rendered and "9,216 model steps" in rendered
    assert "Mean raw episode return" in rendered and "normalized reward units" in rendered
    assert "MPPI" not in rendered and "<script src=" not in rendered
    assert summary["sources"]["reference"]["commit"]==campaign.REFERENCE_SHA
    assert summary["sources"]["inner_xqc"]["commit"]==NEW_SHA
    assert summary["numerical_settings"]==campaign.NUMERICAL_SETTINGS
    assert summary["numerical_settings_scope"]=="new_inner_xqc_evaluations"
    assert "numerical_settings" not in json.loads((outputs[2]/"production/step_50000/bundle/manifest.json").read_text())
    assert row["paired_json_sha256"]==campaign.file_sha256(outputs[1]/"production/step_50000/paired.json")
    assert "inner_reward_normalization" not in summary["inner_settings"]  # Inherited wrapper default, not an authored override.
    json.dumps(summary,allow_nan=False)


@pytest.mark.parametrize("change,match",[
    (lambda m:m.update(status="failed"),"identity"),
    (lambda m:m["code"].update(commit="f"*40),"identity"),
    (lambda m:m["code"].update(dirty=True),"identity"),
    (lambda m:m["code"]["runtime"].update(torch="foreign"),"inconsistent"),
    (lambda m:m["checkpoint"].update(metadata_sha256="a"*64),"identity"),
    (lambda m:m["reference"].update(manifest_sha256="b"*64),"different reference"),
    (lambda m:m["protocol"].update(controller_seed=55),"protocol"),
    (lambda m:m["runs"][0].update(outer_state_unchanged=False),"full outer state"),
    (lambda m:m["runs"][0]["result"].update(outer_updates_after=2),"full outer state"),
    (lambda m:m["runs"][0]["result"].update(deterministic_execution=False),"protocol"),
    (lambda m:m["runs"][0]["result"]["saved_algorithm_config"]["alg_params"].update(inner_actor_lr=.1),"Saved algorithm"),
    (lambda m:m["runs"][0]["config"]["alg_params"].update(inner_actor_lr=.1),"inherit the prior"),
    (lambda m:m["runs"][0]["config"]["alg_params"].update(inner_rounds=2),"inherit the prior"),
    (lambda m:m["runs"][0]["result"]["evaluated_algorithm_config"]["alg_params"].update(inner_batch_size=64),"Evaluated algorithm"),
    (lambda m:m["runs"][0]["result"]["resolved_config"].update(inner_reward_normalization="action_local_imagined"),"Resolved XQC defaults"),
    (lambda m:m["runs"][0]["result"]["resolved_config"].update(inner_actor_lr=.001),"Resolved XQC defaults"),
    (lambda m:m["runs"][0]["result"]["checkpoint_evaluation_provenance"]["evaluated_semantic_signature"].update(actor_arch=[64]),"unsupported outer"),
    (lambda m:m["runs"][0]["result"]["checkpoint_evaluation_provenance"]["evaluated_semantic_signature"].update(reward_normalization="adapted"),"unsupported outer"),
    (lambda m:m["runs"][0]["episodes"][0].update(seed=999),"seeds"),
    (lambda m:m["runs"][0]["episodes"][0].update(solver_seed=1),"mismatched seeds"),
    (lambda m:m["runs"][0]["episodes"][0].update(paired_return_delta=99),"Paired return delta"),
    (lambda m:m["runs"][0]["episodes"][0].update(control_seconds=100),"real-decision measurements"),
    (lambda m:m["runs"][0]["actual_optimizer_steps"].update(actor=72),"optimizer totals"),
])
def test_rejects_unfrozen_foreign_or_misconfigured_inner_results(outputs,change,match):
    _change(outputs,change)
    with pytest.raises(ValueError,match=match):
        _summarize(outputs,allow_partial=True)


@pytest.mark.parametrize("change,match",[
    (lambda r:r.update(actor_updates=18),"C18/A6/T6"),
    (lambda r:r["metrics"].update({"decision/inner_model_steps":12336}),"Measured inner work"),
    (lambda r:r["metrics"].update({"decision/inner_reward_normalizer_imagined_updates":1}),"normalization"),
    (lambda r:r["metrics"].update({"decision/reward":999}),"real-decision measurements"),
    (lambda r:r["metrics"].update({"decision/control_seconds":-1}),"negative"),
])
def test_per_decision_cost_reward_and_timing_are_verified(outputs,change,match):
    path=outputs[1]/"production/step_100000/bundle/controller__xqc/seed-101.jsonl.gz"
    with gzip.open(path,"rt") as stream:
        rows=[json.loads(line) for line in stream]
    change(rows[0])
    with gzip.open(path,"wt") as stream:
        stream.writelines(json.dumps(row)+"\n" for row in rows)
    with pytest.raises(ValueError,match=match):
        _summarize(outputs)


def test_reference_relocation_keeps_hash_validation(outputs,tmp_path):
    moved=tmp_path/"downloaded-references"
    outputs[2].rename(moved)
    summary=_summarize(outputs,reference_root=moved)
    assert summary["rows"][0]["reference_bundle"].startswith(str(moved))
    reference=moved/"production/step_100000/bundle/manifest.json"
    reference.write_text(reference.read_text()+"\n")
    with pytest.raises(ValueError,match="immutable inventory"):
        _summarize(outputs,reference_root=moved)


def test_missing_directories_require_partial_but_partial_never_accepts_incomplete(outputs):
    path=outputs[1]/"production/step_100000"
    path.rename(path.with_name("unselected"))
    with pytest.raises(ValueError,match="Missing checkpoint outputs"):
        _summarize(outputs)
    summary=_summarize(outputs,allow_partial=True)
    assert summary["status"]=="partial" and summary["missing_steps"]==[100000]
    path.mkdir()
    with pytest.raises(FileNotFoundError):
        _summarize(outputs,allow_partial=True)


@pytest.mark.parametrize("filename",["provenance.json","paired.json"])
@pytest.mark.parametrize("settings",[
    None,
    {**campaign.NUMERICAL_SETTINGS,"deterministic_algorithms":False},
    {**campaign.NUMERICAL_SETTINGS,"deterministic_algorithms":1},
    {**campaign.NUMERICAL_SETTINGS,"deterministic_warn_only":True},
    {**campaign.NUMERICAL_SETTINGS,"cudnn_deterministic":False},
    {**campaign.NUMERICAL_SETTINGS,"cudnn_benchmark":True},
    {**campaign.NUMERICAL_SETTINGS,"cublas_workspace_config":":16:8"},
    {**campaign.NUMERICAL_SETTINGS,"device_type":"cpu","cublas_workspace_config":None},
])
def test_production_requires_matching_strict_cuda_numerics_in_both_records(outputs,filename,settings):
    path=outputs[1]/"production/step_100000"/filename
    record=json.loads(path.read_text())
    if settings is None:
        record.pop("numerical_settings")
    else:
        record["numerical_settings"]=settings
    storage.atomic_json(path,record,overwrite=True)
    with pytest.raises(ValueError,match="Production numerical settings"):
        _summarize(outputs,allow_partial=True)


def test_numerical_record_requires_matching_checkpoint_payload(outputs):
    path=outputs[1]/"production/step_100000/paired.json"
    record=json.loads(path.read_text())
    record["checkpoint_sha256"]="f"*64
    storage.atomic_json(path,record,overwrite=True)
    with pytest.raises(ValueError,match="different checkpoint/matrix"):
        _summarize(outputs)


def test_cli_preserves_validated_outputs_when_optional_publication_fails(outputs,tmp_path,monkeypatch):
    summarize=campaign.summarize
    monkeypatch.setattr(campaign,"summarize",lambda *a,**k:summarize(*a,**k,seeds=SEEDS,max_steps=2))
    output,html=tmp_path/"summary.json",tmp_path/"report.html"
    args=["--manifest",str(outputs[0]),"--results-root",str(outputs[1]),"--expected-source-sha",NEW_SHA,
          "--output",str(output),"--html",str(html)]
    assert campaign.main(args)==0
    original=output.read_bytes()
    with pytest.raises(FileExistsError):
        campaign.main(args)
    assert output.read_bytes()==original
    monkeypatch.setattr(campaign,"publish",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError,match="offline"):
        campaign.main([*args,"--overwrite","--wandb"])
    assert json.loads(output.read_text())["status"]=="complete" and html.exists()


def test_optional_publication_stages_validated_raw_bundles(outputs, monkeypatch):
    calls = []
    mapping = {"controller/xqc": "/existing/series"}
    monkeypatch.setattr(storage, "resolve_eval_run_map", lambda *a, **k: mapping)
    monkeypatch.setattr(storage, "stage_completed_bundle", lambda path, runs, **kwargs: calls.append((path, runs, kwargs)) or {})
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *a, **k: pytest.fail("summarizer created a run"))
    summary = _summarize(outputs)
    result = campaign.publish(summary, eval_run_map=mapping, inventory_path=outputs[0])
    assert list(result) == ["50000", "100000"]
    assert len(calls) == 2 and all(call[1] == mapping for call in calls)
    assert calls[0][2]["source_run"] == summary["source_run"]


@pytest.fixture
def outer_outputs(outputs):
    matrix=campaign.campaign_profile("outer_terminal")["matrix"]
    for step in campaign.STEPS:
        destination=outputs[1]/"production"/f"step_{step}"
        bundle=destination/"bundle"
        path=bundle/"manifest.json"
        manifest=json.loads(path.read_text())
        run=manifest["runs"][0]
        run["config"]["alg_params"]["inner_terminal_bootstrap"]="outer"
        run["config_hash"]=storage.canonical_hash(run["config"])
        result=run["result"]
        result["evaluated_algorithm_config"]["alg_params"]["inner_terminal_bootstrap"]="outer"
        result["resolved_config"]["inner_terminal_bootstrap"]="outer"
        # A new loader normalizes a v2 checkpoint; its archived prior-reference
        # signature still lacks the newly introduced default field.
        result["checkpoint_evaluation_provenance"]["saved_semantic_signature"]["inner_terminal_bootstrap"]="inner"
        result["checkpoint_evaluation_provenance"]["evaluated_semantic_signature"]["inner_terminal_bootstrap"]="outer"
        storage.atomic_json(path,manifest,overwrite=True)
        for relative in run["trace_files"]:
            with gzip.open(bundle/relative,"rt") as stream:
                rows=[json.loads(line) for line in stream]
            for row in rows:
                row["metrics"].update({"decision/inner_terminal_bootstrap_outer":1,
                    "decision/inner_outer_terminal_boundary_rows":3072,
                    "decision/inner_outer_terminal_bootstrap_rows":3000,
                    "decision/inner_outer_terminal_policy_evaluations":9216,
                    "decision/inner_outer_terminal_q_evaluations":9216})
            with gzip.open(bundle/relative,"wt") as stream:
                stream.writelines(json.dumps(row)+"\n" for row in rows)
        manifest["metric_catalog"].update(storage.decision_metric_catalog(rows[0]["metrics"],xqc=True))
        storage.atomic_json(path,manifest,overwrite=True)
        for filename in ("provenance.json","paired.json"):
            path=destination/filename
            record=json.loads(path.read_text())
            record.update(variant="outer_terminal",matrix_sha256=campaign.file_sha256(matrix))
            storage.atomic_json(path,record,overwrite=True)
    return outputs


def test_outer_terminal_summary_accepts_legacy_prior_and_has_distinct_labels(outer_outputs,monkeypatch):
    summary=_summarize(outer_outputs,variant="outer_terminal")
    assert summary["variant"]=="outer_terminal"
    assert summary["inner_settings"]["inner_terminal_bootstrap"]=="outer"
    assert summary["rows"][0]["paired"]["delta_mean"]==4
    rendered=campaign.render_html(summary)
    assert "XQC with outer terminal bootstrap" in rendered
    assert "final imagined transition" in rendered and "online critic" in rendered
    assert "adapting inner temperature" in rendered and "running BatchNorm" in rendered
    with pytest.raises(ValueError, match="explicit"):
        campaign.publish(summary, project="test", entity="test")


@pytest.mark.parametrize("filename",["provenance.json","paired.json"])
def test_outer_terminal_cannot_use_missing_or_inner_variant_records(outer_outputs,filename):
    path=outer_outputs[1]/"production/step_100000"/filename
    record=json.loads(path.read_text());record.pop("variant")
    storage.atomic_json(path,record,overwrite=True)
    with pytest.raises(ValueError,match="variant"):
        _summarize(outer_outputs,variant="outer_terminal")


def test_outer_campaign_cannot_be_aggregated_as_native_inner(outer_outputs):
    with pytest.raises(ValueError,match="variant"):
        _summarize(outer_outputs)


@pytest.mark.parametrize("field",["saved_semantic_signature","evaluated_semantic_signature"])
def test_outer_terminal_rejects_misrecorded_checkpoint_semantics(outer_outputs,field):
    def change(manifest):
        manifest["runs"][0]["result"]["checkpoint_evaluation_provenance"][field]["inner_terminal_bootstrap"]="inner" if field.startswith("evaluated") else "outer"
    _change(outer_outputs,change)
    with pytest.raises(ValueError,match="provenance|unsupported"):
        _summarize(outer_outputs,variant="outer_terminal")


@pytest.mark.parametrize("metric,value",[
    ("inner_terminal_bootstrap_outer",0),("inner_outer_terminal_boundary_rows",3000),
    ("inner_outer_terminal_policy_evaluations",3000),("inner_outer_terminal_q_evaluations",3000),
    ("inner_outer_terminal_bootstrap_rows",-1),("inner_outer_terminal_bootstrap_rows",9217),
    ("inner_outer_terminal_bootstrap_rows",1.5),
])
def test_outer_terminal_validates_actual_boundary_and_forward_counts(outer_outputs,metric,value):
    path=outer_outputs[1]/"production/step_100000/bundle/controller__xqc/seed-101.jsonl.gz"
    with gzip.open(path,"rt") as stream:
        rows=[json.loads(line) for line in stream]
    rows[0]["metrics"][f"decision/{metric}"]=value
    with gzip.open(path,"wt") as stream:
        stream.writelines(json.dumps(row)+"\n" for row in rows)
    with pytest.raises(ValueError,match="Outer-terminal decision"):
        _summarize(outer_outputs,variant="outer_terminal")


def test_sampled_terminal_rows_can_be_zero_with_replacement(outer_outputs):
    path=outer_outputs[1]/"production/step_100000/bundle/controller__xqc/seed-101.jsonl.gz"
    with gzip.open(path,"rt") as stream:
        rows=[json.loads(line) for line in stream]
    rows[0]["metrics"]["decision/inner_outer_terminal_bootstrap_rows"]=0
    with gzip.open(path,"wt") as stream:
        stream.writelines(json.dumps(row)+"\n" for row in rows)
    assert _summarize(outer_outputs,variant="outer_terminal")["status"]=="complete"


def test_outer_variant_cli_writes_separate_summary_and_report(outer_outputs,tmp_path,monkeypatch):
    summarize=campaign.summarize
    monkeypatch.setattr(campaign,"summarize",lambda *a,**k:summarize(*a,**k,seeds=SEEDS,max_steps=2))
    output=tmp_path/"outer-summary.json"
    report=tmp_path/"outer-report.html"
    assert campaign.main(["--manifest",str(outer_outputs[0]),"--results-root",str(outer_outputs[1]),
        "--expected-source-sha",NEW_SHA,"--variant","outer_terminal","--output",str(output),"--html",str(report)])==0
    assert json.loads(output.read_text())["variant"]=="outer_terminal"
    assert "outer terminal bootstrap" in report.read_text()


@pytest.fixture
def step_outputs(outer_outputs):
    profile = campaign.campaign_profile("outer_terminal_step")
    for step in campaign.STEPS:
        destination = outer_outputs[1]/"production"/f"step_{step}"
        bundle = destination/"bundle"
        path = bundle/"manifest.json"
        manifest = json.loads(path.read_text())
        run = manifest["runs"][0]
        settings = {"inner_update_timing": "step", "inner_policy_delay": 1}
        run["config"]["alg_params"].update(settings)
        run["config_hash"] = storage.canonical_hash(run["config"])
        result = run["result"]
        result["evaluated_algorithm_config"]["alg_params"].update(settings)
        result["resolved_config"].update(settings)
        provenance = result["checkpoint_evaluation_provenance"]
        provenance["saved_semantic_signature"].update(inner_update_timing="round", inner_policy_delay=3)
        provenance["evaluated_semantic_signature"].update(settings)
        run["actual_optimizer_steps"] = {key: value*len(SEEDS)*2 for key,value in profile["optimizer_steps"].items()}
        for episode in run["episodes"]:
            episode["actual_optimizer_steps"] = {key: value*2 for key,value in profile["optimizer_steps"].items()}
        for relative in run["trace_files"]:
            with gzip.open(bundle/relative,"rt") as stream:
                rows = [json.loads(line) for line in stream]
            for row in rows:
                row.update(actor_updates=18, temperature_updates=18)
                row["metrics"].update({"decision/inner_update_timing_step":1,
                    "decision/inner_policy_delay":1,
                    "decision/inner_updates_per_rollout_step":1, "decision/inner_collection_steps":18,
                    "decision/inner_actor_optimizer_steps":18,"decision/inner_temperature_optimizer_steps":18})
            with gzip.open(bundle/relative,"wt") as stream:
                stream.writelines(json.dumps(row)+"\n" for row in rows)
        manifest["metric_catalog"].update(storage.decision_metric_catalog(rows[0]["metrics"],xqc=True))
        storage.atomic_json(path,manifest,overwrite=True)
        for filename in ("provenance.json","paired.json"):
            path = destination/filename
            record = json.loads(path.read_text())
            record.update(variant="outer_terminal_step",matrix_sha256=campaign.file_sha256(profile["matrix"]))
            storage.atomic_json(path,record,overwrite=True)
    return outer_outputs


def test_step_summary_accepts_old_prior_and_reports_actor_update_every_depth(step_outputs):
    summary = _summarize(step_outputs,variant="outer_terminal_step")
    assert summary["inner_settings"]["inner_update_timing"] == "step"
    assert summary["inner_settings"]["inner_policy_delay"] == 1
    assert summary["optimizer_steps_per_decision"] == {"critic":18,"actor":18,"temperature":18}
    assert summary["rows"][0]["paired"]["delta_mean"] == 4
    rendered = campaign.render_html(summary)
    assert "outer terminal bootstrap and step updates" in rendered
    assert "C18/A18/T18" in rendered and "C18/A6/T6" not in rendered
    assert "one critic, actor, and temperature update" in rendered
    assert "G3 is the total per round" in rendered
    assert "frozen outer learner retains delay 3" in rendered


@pytest.mark.parametrize("metric,value",[("inner_update_timing_step",0),
    ("inner_updates_per_rollout_step",3),("inner_collection_steps",6),
    ("inner_actor_optimizer_steps",6),("inner_policy_delay",3)])
def test_step_summary_requires_actual_interleaving_and_full_actor_dose(step_outputs,metric,value):
    path = step_outputs[1]/"production/step_100000/bundle/controller__xqc/seed-101.jsonl.gz"
    with gzip.open(path,"rt") as stream:
        rows = [json.loads(line) for line in stream]
    rows[0]["metrics"][f"decision/{metric}"] = value
    with gzip.open(path,"wt") as stream:
        stream.writelines(json.dumps(row)+"\n" for row in rows)
    with pytest.raises(ValueError,match="Step-update decision|Measured inner work|Measured inner policy delay"):
        _summarize(step_outputs,variant="outer_terminal_step")


@pytest.mark.parametrize("field",["saved_semantic_signature","evaluated_semantic_signature"])
@pytest.mark.parametrize("setting,wrong",[("inner_update_timing","round"),("inner_policy_delay",3)])
def test_step_summary_rejects_incompatible_saved_or_evaluated_semantics(step_outputs,field,setting,wrong):
    def change(manifest):
        value = wrong if field.startswith("evaluated") else ("step" if setting=="inner_update_timing" else 1)
        manifest["runs"][0]["result"]["checkpoint_evaluation_provenance"][field][setting] = value
    _change(step_outputs,change)
    with pytest.raises(ValueError,match="provenance|unsupported"):
        _summarize(step_outputs,variant="outer_terminal_step")


def test_step_and_round_cannot_be_mislabeled(step_outputs):
    with pytest.raises(ValueError,match="variant"):
        _summarize(step_outputs,variant="outer_terminal")


def test_step_variant_cli_writes_distinct_report(step_outputs,tmp_path,monkeypatch):
    summarize = campaign.summarize
    monkeypatch.setattr(campaign,"summarize",lambda *a,**k:summarize(*a,**k,seeds=SEEDS,max_steps=2))
    output, report = tmp_path/"step-summary.json", tmp_path/"step-report.html"
    campaign.main(["--manifest",str(step_outputs[0]),"--results-root",str(step_outputs[1]),
        "--expected-source-sha",NEW_SHA,"--variant","outer_terminal_step","--output",str(output),"--html",str(report)])
    assert json.loads(output.read_text())["variant"] == "outer_terminal_step"
    assert "C18/A18/T18" in report.read_text()
