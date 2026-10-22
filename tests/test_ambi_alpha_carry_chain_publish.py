"""Dependent episode histories retain their pairing and publication identities."""
from copy import deepcopy
import json
import math
import statistics
from types import SimpleNamespace

import pytest

from slurm import ambi_alpha_carry_chain_publish as pub
from utils import eval_series as series


def campaign():
    return dict(campaign_id="test-chain", source_commit="a"*40, checkpoints={name:{} for name in pub.CHECKPOINTS})


def reference(offset=0., *, rows=True):
    value = dict(episodes=[],rows=[],provenance={"source":"saved-result","offset":offset})
    for seed in range(101,106):
        total = seed-100+offset
        value["episodes"].append(dict(seed=seed,solver_seed=seed*10,length=500,control_seconds=250.,**{"return":total}))
        if rows:
            for decision in range(500):
                value["rows"].append(dict(seed=seed,decision_index=decision,entropy=-8.,target_entropy=-10.5,
                    alpha_initial=.01,alpha_final=.009,reward=total/500,control_seconds=.5,
                    optimizer_step_initial=None,optimizer_step_final=None))
    return value


def references():
    return {"prior":reference(-.5,rows=False),"reset":reference(),"episode_carry":reference(.2)}


def chain():
    value = dict(rows=[],episodes=[])
    for stream in range(5):
        for episode,seed in enumerate(pub.stream_seeds(stream)):
            total = seed-100+stream*10+episode*100
            value["episodes"].append(dict(stream_index=stream,episode_index=episode,seed=seed,
                solver_seed=seed*10,length=500,control_seconds=500.,**{"return":total}))
            for decision in range(500):
                glob=episode*500+decision
                value["rows"].append(dict(stream_index=stream,episode_index=episode,seed=seed,
                    decision_index=decision,global_decision_index=glob,entropy=-8.,target_entropy=-10.5,
                    alpha_initial=.1*math.exp(-glob*.001),alpha_final=.1*math.exp(-(glob+1)*.001),
                    reward=total/500,control_seconds=1.,optimizer_step_initial=glob*32,
                    optimizer_step_final=(glob+1)*32))
    return value


def test_balanced_seed_rotation_and_reused_references_are_not_new_trials():
    source=reference()
    result=pub.reference_chain(source)
    assert len(result["rows"])==7500 and len(result["episodes"])==15
    for episode in range(3):
        assert sorted(ep["seed"] for ep in result["episodes"] if ep["episode_index"]==episode)==list(range(101,106))
    assert result["reused_reference"] is True
    assert result["rows"][500]["seed"]==102 and result["rows"][500]["decision_index"]==0
    assert result["rows"][500]["global_decision_index"]==500
    assert source["rows"][0].get("episode_index") is None


def test_episode_statistics_use_five_histories_not_pooled_fifteen_episodes():
    value=chain()
    points,overall=pub.reduce_episodes(value,references())
    assert [point[pub.EPISODE_AXIS] for point in points]==[1,2,3]
    assert [point["return_mean"] for point in points]==[23.,123.,223.]
    assert all(point["streams"]==5 for point in points)
    assert points[0]["episode_carry_gain_mean"]==pytest.approx(19.8)
    assert points[2]["reset_gain_mean"]==220.
    assert points[0]["prior_gain_mean"]==20.5
    by_stream=[[ep["return"] for ep in value["episodes"] if ep["stream_index"]==stream] for stream in range(5)]
    wanted=statistics.stdev([statistics.mean(values) for values in by_stream])
    pooled=statistics.stdev([ep["return"] for ep in value["episodes"]])
    assert overall["streams"]==5 and overall["episodes_per_stream"]==3
    assert overall["stream_mean_return_sample_std"]==wanted and wanted!=pooled


def test_global_decisions_preserve_alpha_continuity_and_episode_reward_resets():
    points=pub.reduce_decisions(chain())
    assert len(points)==1500 and [points[i][pub.AXIS] for i in (0,499,500,999,1000,1499)]==[1,500,501,1000,1001,1500]
    assert points[500]["alpha_initial"]==points[499]["alpha_final"]
    assert points[500]["boundary_alpha_jump"]==points[1000]["boundary_alpha_jump"]==0
    assert "boundary_alpha_jump" not in points[499]
    assert points[500]["optimizer_step_initial"]==16000
    assert points[500]["episode_cumulative_reward"]==pytest.approx(123/500)
    assert points[500]["chain_cumulative_reward"]==pytest.approx(23+123/500)
    assert points[-1]["chain_cumulative_reward"]==pytest.approx(369)


@pytest.mark.parametrize("failure",["missing","duplicate","seed","global","episode","nonfinite","reward"])
def test_incomplete_or_mispaired_decisions_fail(failure):
    value=chain()
    if failure=="missing":value["rows"].pop()
    elif failure=="duplicate":value["rows"][-1]=deepcopy(value["rows"][0])
    elif failure=="seed":value["rows"][0]["seed"]=105
    elif failure=="global":value["rows"][500]["global_decision_index"]=0
    elif failure=="episode":value["episodes"][0]["length"]=499
    elif failure=="nonfinite":value["rows"][0]["entropy"]=float("nan")
    elif failure=="reward":value["rows"][0]["reward"]+=2
    with pytest.raises(pub.PublicationError):pub.reduce_decisions(value)


def test_episode_pairing_rejects_controller_seed_mismatch():
    refs=references()
    refs["prior"]["episodes"][0]["solver_seed"]+=1
    with pytest.raises(pub.PublicationError,match="controller seeds"):
        pub.reduce_episodes(chain(),refs)


def test_native_episode_values_occur_only_at_three_actual_episode_ends():
    payload=pub.build_payload("sac_aux_100k","chain_carry",chain(),references(),{})
    entry=dict(payload=payload,checkpoint_id="sac_aux_100k")
    rows=pub.history_rows(entry)
    key="episode/sac_aux_100k/return_mean"
    assert [row[pub.AXIS] for row in rows if key in row]==[500,1000,1500]
    assert [row[pub.EPISODE_AXIS] for row in rows if key in row]==[1,2,3]
    prior=pub.reference_chain(references()["prior"],decision_rows=False)
    entry["payload"]=pub.build_payload("sac_aux_100k","prior",prior,references(),{})
    assert len(pub.history_rows(entry))==3
    assert all(pub.AXIS not in row for row in pub.history_rows(entry))


def snapshot():
    source=chain()
    streams=[dict(rows=[row for row in source["rows"] if row["stream_index"]==s],
                  episodes=[ep for ep in source["episodes"] if ep["stream_index"]==s]) for s in range(5)]
    return {"checkpoints":{checkpoint:dict(reference_summaries=references(),streams=deepcopy(streams),
        science_id="science-"+checkpoint,status=[dict(stream_index=s,state="complete",receipt_sha256="a"*64) for s in range(5)])
        for checkpoint in pub.CHECKPOINTS}}


def test_partial_checkpoint_never_publishes_partial_mean_but_shows_progress(monkeypatch):
    from slurm import ambi_alpha_carry_chain as driver
    state=snapshot()
    target=state["checkpoints"]["sac_aux_100k"]
    target["streams"].pop()
    target["status"][-1]["state"]="running"
    monkeypatch.setattr(driver,"publication_snapshot",lambda campaign:state)
    entries=pub.collect(campaign())
    selected=next(e for e in entries if e["checkpoint_id"]=="sac_aux_100k" and e["method"]=="chain_carry")
    assert selected["payload"] is None and selected["status"]=="pending"
    assert selected["progress"]==dict(complete=4,pending=1,failed=0,expected=5)
    final=pub.collect(campaign(),require_complete=True)
    selected=next(e for e in final if e["checkpoint_id"]=="sac_aux_100k" and e["method"]=="chain_carry")
    assert selected["status"]=="failed" and selected["progress"]["failed"]==1


def test_reference_payload_identity_does_not_change_as_new_histories_complete(monkeypatch):
    from slurm import ambi_alpha_carry_chain as driver
    state=snapshot()
    monkeypatch.setattr(driver,"publication_snapshot",lambda campaign:state)
    complete=pub.collect(campaign())
    for checkpoint in state["checkpoints"].values():
        checkpoint["streams"]=[]
        checkpoint["status"]=[dict(stream_index=s,state="pending") for s in range(5)]
    pending=pub.collect(campaign())
    for before,after in zip(complete,pending):
        assert before["run_id"]==after["run_id"]
        if before["method"]!="chain_carry":assert before["payload"]==after["payload"]


def test_dedicated_identities_are_unique_and_protocol_explicit():
    rows=pub.display_rows(campaign())
    assert len(rows)==len({row["run_id"] for row in rows})==8
    for row in rows:
        assert row["identity"]["protocol"]=="run-temperature-chain-v1"
        assert row["identity"]["histories"]==5 and row["identity"]["episodes_per_history"]==3


class Obj(SimpleNamespace):pass
class Filter:
    def __init__(self,name,values):self.name,self.values=name,values
    def to_model(self):return Obj(model_dump=lambda **kwargs:dict(op="IN",key=dict(name=self.name),value=self.values))
class Metric:
    def __init__(self,name):self.name=name
    def isin(self,values):return Filter(self.name,values)
class LinePlot(Obj):pass


def backend(corrupt=None):
    state={}
    class Workspace(Obj):
        @property
        def url(self):return f"https://wandb.ai/{pub.ENTITY}/{pub.PROJECT}?nw=chain-test"
        def save(self):state["work"]=deepcopy(self)
        @classmethod
        def from_url(cls,url):
            value=deepcopy(state["work"])
            if corrupt:corrupt(value)
            return value
    ws=Obj(Workspace=Workspace,Section=Obj,WorkspaceSettings=Obj,RunsetSettings=Obj,RunSettings=Obj)
    wr=Obj(LinePlot=LinePlot,ScalarChart=Obj,MarkdownPanel=Obj,Layout=Obj)
    return (ws,wr,Obj(Metric=Metric)),state


def test_native_layout_preserves_both_axes_true_colors_and_connected_curves(tmp_path):
    sdk,state=backend()
    first=pub.install_workspace(campaign(),tmp_path/"workspace.json",backend=sdk)
    state["work"].sections.append(Obj(name="Personal notes",panels=[]))
    second=pub.install_workspace(campaign(),tmp_path/"workspace.json",backend=sdk)
    assert first["url"]==second["url"] and second["api_verified"]
    assert len(state["work"].sections)==6
    plots=[p for s in state["work"].sections for p in s.panels if isinstance(p,LinePlot)]
    assert len(plots)==38 and {p.x for p in plots}=={pub.AXIS,pub.EPISODE_AXIS}
    assert all(set(p.line_marks.values())=={"solid"} and not p.aggregate for p in plots)
    assert all(len(p.line_titles)==(4 if p.x==pub.EPISODE_AXIS else 3) for p in plots)


@pytest.mark.parametrize("failure",["one_mark","color","metric","axis"])
def test_layout_rejects_dropped_plot_settings(tmp_path,failure):
    def corrupt(work):
        plot=next(p for s in work.sections for p in s.panels if isinstance(p,LinePlot))
        if failure=="one_mark":plot.line_marks.pop(next(iter(plot.line_marks)))
        elif failure=="color":plot.line_colors[next(iter(plot.line_colors))]="#FFFFFF"
        elif failure=="metric":plot.y=["episode/sac_aux_100k/wrong"]
        else:plot.x="Step"
    sdk,_=backend(corrupt)
    with pytest.raises(pub.PublicationError):pub.install_workspace(campaign(),tmp_path/"workspace.json",backend=sdk)
    assert series._read(tmp_path/"workspace.json")["api_verified"] is False


def test_report_has_connected_lines_boundaries_and_embedded_actual_data(tmp_path):
    value=campaign()
    entries=[{**row,"payload":None,"status":"pending","progress":dict(complete=0,pending=5,failed=0)} for row in pub.display_rows(value)]
    path=tmp_path/"report.html"
    data=pub.render_report(value,entries,path)
    html=path.read_text()
    assert "add('polyline'" in html and "[500.5,1000.5]" in html and "not 15 independent trials" in html
    encoded=html.split('<script id="chain-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(encoded)==json.loads(json.dumps(data))


def completed_entry():
    entry=pub.display_rows(campaign())[0]
    refs=references()
    payload=pub.build_payload(entry["checkpoint_id"],"prior",pub.reference_chain(refs["prior"],decision_rows=False),refs,{})
    return {**entry,"payload":payload,"status":"complete","progress":{}}


def sdk_fixture(state):
    class Run:
        disabled=False
        settings=Obj(mode="online")
        def __init__(self):self.step=len(state["history"]);self.summary={}
        def define_metric(self,*args,**kwargs):pass
        def log(self,row,*,step,commit):
            assert step==len(state["history"])
            state["history"].append({**row,"_step":step})
            state["logged"].append(step)
        def finish(self,**kwargs):pass
    def init(**kwargs):
        state["inits"].append(kwargs)
        state["config"]=kwargs["config"]
        return Run()
    remote=Obj(scan_history=lambda **kwargs:deepcopy(state["history"]))
    def remote_run(path):remote.config=state["config"];return remote
    return Obj(init=init,Api=lambda **kwargs:Obj(run=remote_run))


def test_pending_to_complete_publication_is_idempotent_and_preserves_axis_rows(tmp_path):
    entry=completed_entry()
    state=dict(history=[],logged=[],inits=[],config={})
    sdk=sdk_fixture(state)
    pending={**entry,"payload":None,"status":"pending"}
    assert pub.publish_entry(pending,tmp_path,sdk)["rows"]==0
    result=pub.publish_entry(entry,tmp_path,sdk)
    assert result["status"]=="complete" and result["rows"]==3
    assert [row[pub.EPISODE_AXIS] for row in state["history"]]==[1,2,3]
    assert pub.publish_entry(entry,tmp_path,sdk)==result
    assert len(state["inits"])==2 and state["logged"]==[0,1,2]
    entry["payload"]["episode_points"][0]["return_mean"]+=1
    with pytest.raises(pub.PublicationError,match="result changed"):
        pub.publish_entry(entry,tmp_path,sdk)


def test_interrupted_native_history_resumes_only_missing_exact_prefix(tmp_path):
    entry=completed_entry()
    rows=pub.history_rows(entry)
    state=dict(history=[{**rows[0],"_step":0}],logged=[],inits=[],
               config={"chain_identity_sha256":series._digest(entry["identity"])})
    path=tmp_path/"runs"/entry["run_id"]/"publication.json"
    series._atomic_json(path,dict(identity_sha256=series._digest(entry["identity"]),
                                 initialized=True,status="inflight"))
    pub.publish_entry(entry,tmp_path,sdk_fixture(state))
    assert state["logged"]==[1,2] and len(state["history"])==3


def test_failed_first_init_can_retry_only_after_verified_remote_absence(tmp_path):
    entry={**completed_entry(),"payload":None,"status":"pending"}
    state=dict(history=[],logged=[],inits=[],config={})
    sdk=sdk_fixture(state)
    original=sdk.init
    def fail(**kwargs):raise RuntimeError("init failed before remote creation")
    sdk.init=fail
    with pytest.raises(RuntimeError):pub.publish_entry(entry,tmp_path,sdk)
    saved=series._read(tmp_path/"runs"/entry["run_id"]/"publication.json")
    assert saved["init_started"] and not saved["initialized"]
    def missing(path):raise ValueError(f"Could not find run <Run {path} (not found)>")
    sdk.Api=lambda **kwargs:Obj(run=missing)
    sdk.init=original
    assert pub.publish_entry(entry,tmp_path,sdk)["initialized"]
    assert state["inits"][0]["resume"]=="allow"


def test_missing_layout_blocks_publication_before_any_network_or_source_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(pub,"collect",lambda *args,**kwargs:pytest.fail("No source scan before verified layout"))
    with pytest.raises(pub.PublicationError,match="workspace"):
        pub.publish_once(campaign(),tmp_path,wandb_module=Obj())


def test_hardware_provenance_uses_actual_stream_allocation(tmp_path):
    value=campaign()
    value.update(root=str(tmp_path),cells=[dict(index=0,stream_index=0,checkpoint_id="sac_aux_100k",output_dir=str(tmp_path/"production"))])
    series._atomic_json(tmp_path/"production/started.json",{"job_id":"12345"})
    path=tmp_path/"production-hardware/0-12345.json"
    series._atomic_json(path,dict(job_id="12345",gpu_type="l40s",gpu_name="NVIDIA L40S",node="test-node"))
    rows=pub.hardware_provenance(value,"sac_aux_100k")
    assert rows[0]["gpu_name"]=="NVIDIA L40S" and rows[0]["sha256"]==series._file_digest(path)
    series._atomic_json(path,dict(job_id="99999"))
    with pytest.raises(pub.PublicationError,match="different stream allocation"):
        pub.hardware_provenance(value,"sac_aux_100k")
