"""Episode-reset paired preservation and dedicated native publication invariants."""
from copy import deepcopy
import json
import math
import statistics
from types import SimpleNamespace
import pytest
from slurm import ambi_alpha_episode_reset_publish as pub
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
    return {"prior":reference(-.5,rows=False),"reset":reference()}

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

def test_episode_statistics_use_five_histories_not_pooled_fifteen_episodes():
    value=chain()
    points,overall=pub.reduce_episodes(value,references(),chain())
    assert [point[pub.EPISODE_AXIS] for point in points]==[1,2,3]
    assert [point["return_mean"] for point in points]==[23.,123.,223.]
    assert all(point["streams"]==5 for point in points)
    assert points[0]["reset_gain_mean"]==pytest.approx(20.)
    assert points[2]["reset_gain_mean"]==220.
    assert points[0]["prior_gain_mean"]==20.5
    by_stream=[[ep["return"] for ep in value["episodes"] if ep["stream_index"]==stream] for stream in range(5)]
    wanted=statistics.stdev([statistics.mean(values) for values in by_stream])
    pooled=statistics.stdev([ep["return"] for ep in value["episodes"]])
    assert overall["streams"]==5 and overall["episodes_per_stream"]==3
    assert overall["stream_mean_return_sample_std"]==wanted and wanted!=pooled

def test_episode_pairing_rejects_controller_seed_mismatch():
    refs=references()
    refs["prior"]["episodes"][0]["solver_seed"]+=1
    with pytest.raises(pub.PublicationError,match="controller seeds"):
        pub.reduce_episodes(chain(),refs,chain())

def test_native_episode_values_occur_only_at_three_actual_episode_ends():
    payload=pub.build_payload("mppi_aux_500k","episode_reset",chain(),references(),chain(),{})
    entry=dict(payload=payload,checkpoint_id="mppi_aux_500k")
    rows=pub.history_rows(entry)
    key="episode/mppi_aux_500k/return_mean"
    assert [row[pub.AXIS] for row in rows if key in row]==[500,1000,1500]
    assert [row[pub.EPISODE_AXIS] for row in rows if key in row]==[1,2,3]
    prior=pub.reference_chain(references()["prior"],decision_rows=False)
    entry["payload"]=pub.build_payload("mppi_aux_500k","prior",prior,references(),chain(),{})
    assert len(pub.history_rows(entry))==3
    assert all(pub.AXIS not in row for row in pub.history_rows(entry))

def snapshot():
    source=chain()
    streams=[dict(rows=[row for row in source["rows"] if row["stream_index"]==s],
                  episodes=[ep for ep in source["episodes"] if ep["stream_index"]==s]) for s in range(5)]
    return {"checkpoints":{checkpoint:dict(reference_summaries=references(),streams=deepcopy(streams),
        reference_streams=deepcopy(streams),science_id="science-"+checkpoint,status=[dict(stream_index=s,state="complete",receipt_sha256="a"*64) for s in range(5)])
        for checkpoint in pub.CHECKPOINTS}}

def test_partial_checkpoint_never_publishes_partial_mean_but_shows_progress(monkeypatch):
    from slurm import ambi_alpha_episode_reset as driver
    state=snapshot()
    target=state["checkpoints"]["mppi_aux_500k"]
    target["streams"].pop()
    target["status"][-1]["state"]="running"
    monkeypatch.setattr(driver,"publication_snapshot",lambda campaign:state)
    entries=pub.collect(campaign())
    selected=next(e for e in entries if e["checkpoint_id"]=="mppi_aux_500k" and e["method"]=="episode_reset")
    assert selected["payload"] is None and selected["status"]=="pending"
    assert selected["progress"]==dict(complete=4,pending=1,failed=0,expected=5)
    final=pub.collect(campaign(),require_complete=True)
    selected=next(e for e in final if e["checkpoint_id"]=="mppi_aux_500k" and e["method"]=="episode_reset")
    assert selected["status"]=="failed" and selected["progress"]["failed"]==1

def test_reference_payload_identity_does_not_change_as_new_histories_complete(monkeypatch):
    from slurm import ambi_alpha_episode_reset as driver
    state=snapshot()
    monkeypatch.setattr(driver,"publication_snapshot",lambda campaign:state)
    complete=pub.collect(campaign())
    for checkpoint in state["checkpoints"].values():
        checkpoint["streams"]=[]
        checkpoint["status"]=[dict(stream_index=s,state="pending") for s in range(5)]
    pending=pub.collect(campaign())
    for before,after in zip(complete,pending):
        assert before["run_id"]==after["run_id"]
        if before["method"]!="episode_reset":assert before["payload"]==after["payload"]

def test_dedicated_identities_are_unique_and_protocol_explicit():
    rows=pub.display_rows(campaign())
    assert len(rows)==len({row["run_id"] for row in rows})==4
    for row in rows:
        assert row["identity"]["protocol"]=="episode-temperature-reset-v1"
        assert row["identity"]["histories"]==5 and row["identity"]["episodes_per_history"]==3
        expected={"prior":None,"reset":"action","chain_carry":"run","episode_reset":"episode"}[row["method"]]
        assert row["identity"]["method_temperature_scope"]==expected
        assert row["identity"]["method_temperature_optimizer_scope"]==expected

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
    assert len(state["work"].sections)==4
    plots=[p for s in state["work"].sections for p in s.panels if isinstance(p,LinePlot)]
    assert len(plots)==27 and {p.x for p in plots}=={pub.AXIS,pub.EPISODE_AXIS}
    assert all(set(p.line_marks.values())=={"solid"} and not p.aggregate for p in plots)
    assert all(len(p.line_titles)==(4 if p.x==pub.EPISODE_AXIS else 3) for p in plots)

@pytest.mark.parametrize("failure",["one_mark","color","metric","axis"])
def test_layout_rejects_dropped_plot_settings(tmp_path,failure):
    def corrupt(work):
        plot=next(p for s in work.sections for p in s.panels if isinstance(p,LinePlot))
        if failure=="one_mark":plot.line_marks.pop(next(iter(plot.line_marks)))
        elif failure=="color":plot.line_colors[next(iter(plot.line_colors))]="#FFFFFF"
        elif failure=="metric":plot.y=["episode/mppi_aux_500k/wrong"]
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
    payload=pub.build_payload(entry["checkpoint_id"],"prior",pub.reference_chain(refs["prior"],decision_rows=False),refs,chain(),{})
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
    value.update(root=str(tmp_path),cells=[dict(index=0,stream_index=0,checkpoint_id="mppi_aux_500k",output_dir=str(tmp_path/"production"))])
    series._atomic_json(tmp_path/"production/started.json",{"job_id":"12345"})
    path=tmp_path/"production-hardware/0-12345.json"
    series._atomic_json(path,dict(job_id="12345",gpu_type="l40s",gpu_name="NVIDIA L40S",node="test-node"))
    rows=pub.hardware_provenance(value,"mppi_aux_500k")
    assert rows[0]["gpu_name"]=="NVIDIA L40S" and rows[0]["sha256"]==series._file_digest(path)
    series._atomic_json(path,dict(job_id="99999"))
    with pytest.raises(pub.PublicationError,match="different stream allocation"):
        pub.hardware_provenance(value,"mppi_aux_500k")


def test_primary_episode_three_preserves_paired_wins_losses_and_worst_loss():
    value=chain()
    gains=[2.,-1.,0.,4.,-5.]
    for episode in value["episodes"]:
        if episode["episode_index"]==2:
            episode["return"]=episode["seed"]-100+gains[episode["stream_index"]]
    points,overall=pub.reduce_episodes(value,references(),chain())
    primary=points[2]
    assert primary["reset_wins"]==2 and primary["reset_losses"]==2 and primary["reset_ties"]==1
    assert primary["reset_win_fraction"]==.4 and primary["reset_worst_gain"]==-5.
    assert primary["reset_gain_mean"]==0. and primary["reset_gain_sample_std"]==statistics.stdev(gains)
    assert primary["retained_gain_ratio"]==1.
    assert overall["primary_episode"]==3 and overall["primary_reset_worst_gain"]==-5.
    assert overall["primary_reset_gain_sample_std"]==statistics.stdev(gains)


@pytest.mark.parametrize("offset",[0.,1.])
def test_retention_ratio_is_absent_for_zero_or_negative_fresh_improvement(offset):
    refs=references()
    refs["prior"]=reference(offset,rows=False)
    points,overall=pub.reduce_episodes(chain(),refs,chain())
    assert all("retained_gain_ratio" not in point for point in points)
    assert "primary_retained_gain_ratio" not in overall


def test_retention_is_ratio_of_mean_paired_gains_not_mean_individual_ratios():
    refs=references()
    for index,episode in enumerate(refs["prior"]["episodes"]):
        episode["return"]-=index
    points,_=pub.reduce_episodes(chain(),refs,chain())
    assert points[0]["fresh_prior_gain_mean"]==2.5
    assert points[0]["retained_gain_ratio"]==22.5/2.5


def test_no_invented_within_episode_reference_and_protocol_namespace_is_new():
    refs=references()
    assert set(refs)=={"prior","reset"}
    rows=pub.display_rows(campaign())
    assert {row["method"] for row in rows}=={"prior","reset","chain_carry","episode_reset"}
    assert all(row["identity"]["kind"]=="alpha-episode-reset-publication-v1" for row in rows)
    assert all(row["identity"]["primary_episode"]==3 for row in rows)
    refs["episode_carry"]=reference()
    with pytest.raises(pub.PublicationError,match="Exactly"):
        pub.reduce_episodes(chain(),refs,chain())



def test_primary_pairing_uses_measured_history_not_a_repeated_single_episode():
    measured=chain()
    current=deepcopy(measured)
    gains=[3.,-2.,0.,5.,-1.]
    for episode in current["episodes"]:
        if episode["episode_index"]==2:
            episode["return"]+=gains[episode["stream_index"]]
    points,overall=pub.reduce_episodes(current,references(),measured)
    assert [point["chain_carry_gain_mean"] for point in points[:2]]==[0.,0.]
    primary=points[2]
    assert primary["chain_carry_gain_mean"]==statistics.mean(gains)
    assert primary["chain_carry_gain_sample_std"]==statistics.stdev(gains)
    assert (primary["chain_carry_wins"],primary["chain_carry_losses"],primary["chain_carry_ties"])==(2,2,1)
    assert primary["chain_carry_worst_gain"]==-2. and primary["chain_carry_best_gain"]==5.
    assert primary["chain_carry_win_fraction"]==.4
    assert overall["primary_chain_carry_gain_mean"]==1.
    assert overall["primary_episode"]==3


@pytest.mark.parametrize("mutation",["controller_seed","environment_seed","stream","missing","nonfinite"])
def test_measured_carry_reference_pairing_must_be_exact(mutation):
    measured=chain()
    if mutation=="controller_seed":measured["episodes"][1]["solver_seed"]+=1
    elif mutation=="environment_seed":measured["episodes"][1]["seed"]+=1
    elif mutation=="stream":measured["episodes"][1]["stream_index"]=9
    elif mutation=="missing":measured["episodes"].pop()
    else:measured["episodes"][1]["return"]=float("nan")
    with pytest.raises(pub.PublicationError,match="carry|Carry"):
        pub.reduce_episodes(chain(),references(),measured)


def test_reused_carry_is_complete_while_new_reset_pending(monkeypatch):
    from slurm import ambi_alpha_episode_reset as driver
    state=snapshot()
    checkpoint=state["checkpoints"]["mppi_aux_500k"]
    checkpoint["streams"]=[]
    checkpoint["status"]=[dict(stream_index=i,state="pending") for i in range(5)]
    monkeypatch.setattr(driver,"publication_snapshot",lambda campaign:state)
    entries=pub.collect(campaign())
    by_method={entry["method"]:entry for entry in entries}
    assert by_method["chain_carry"]["status"]=="complete"
    assert len(by_method["chain_carry"]["payload"]["decision_points"])==1500
    assert by_method["episode_reset"]["payload"] is None
    assert by_method["episode_reset"]["status"]=="pending"
    assert all(point["chain_carry_gain_mean"]==0 for point in by_method["chain_carry"]["payload"]["episode_points"])


def test_new_run_is_not_labeled_as_a_reused_reference(tmp_path):
    entry=next(row for row in pub.display_rows(campaign()) if row["method"]=="episode_reset")
    entry.update(payload=None,status="pending",progress=dict(complete=0,pending=5,failed=0,expected=5))
    state=dict(history=[],logged=[],inits=[],config={})
    pub.publish_entry(entry,tmp_path,sdk_fixture(state))
    kwargs=state["inits"][0]
    assert kwargs["config"]["reused_reference"] is False
    assert kwargs["config"]["chain_identity"]["reference_reuse"] is False
    assert kwargs["job_type"]=="temperature-episode-reset-evaluation"
    assert kwargs["tags"]==["alpha-episode-reset","episode_reset"]


def test_actual_carry_reference_retains_reference_identity(tmp_path):
    entry=next(row for row in pub.display_rows(campaign()) if row["method"]=="chain_carry")
    entry.update(payload=None,status="pending",progress={})
    state=dict(history=[],logged=[],inits=[],config={})
    pub.publish_entry(entry,tmp_path,sdk_fixture(state))
    assert state["inits"][0]["config"]["reused_reference"] is True
    assert state["inits"][0]["job_type"]=="temperature-episode-reset-reference"


def test_merge_rejects_missing_or_duplicate_measured_reference_histories():
    streams=snapshot()["checkpoints"]["mppi_aux_500k"]["reference_streams"]
    with pytest.raises(pub.PublicationError,match="five"):
        pub.merge_streams(streams[:-1])
    streams[-1]=streams[0]
    with pytest.raises(pub.PublicationError,match="exactly once"):
        pub.merge_streams(streams)


def test_reset_boundary_is_shown_in_native_temperature_curves():
    value=chain()
    for row in value["rows"]:
        local=row["decision_index"]
        row.update(alpha_initial=.1*math.exp(-local*.001),alpha_final=.1*math.exp(-(local+1)*.001),
                   optimizer_step_initial=local*32,optimizer_step_final=(local+1)*32)
    points=pub.reduce_decisions(value)
    assert points[500]["optimizer_step_initial"]==0
    assert points[1000]["optimizer_step_initial"]==0
    assert points[500]["alpha_initial"]==pytest.approx(.1)
    assert points[500]["boundary_alpha_jump"]==pytest.approx(.1-.1*math.exp(-.5))
    assert points[499]["optimizer_step_final"]==16000
