"""Ten actual episode positions, paired histories and immutable native publication."""
from copy import deepcopy
import json
import math
import statistics
from types import SimpleNamespace
import pytest
from slurm import ambi_alpha_long_episodes_publish as pub
from utils import eval_series as series


def campaign():
    return dict(campaign_id="test-long",source_commit="a"*40,checkpoints={"mppi_aux_500k":{}},cells=[])


def chain(method):
    value=dict(rows=[],episodes=[])
    for stream in range(5):
        for episode,seed in enumerate(pub.stream_seeds(stream)):
            total=100.+seed-100+episode*10+stream*3
            if method=="run_carry":total+=stream-2+(episode-4)*(stream+1)
            value["episodes"].append(dict(stream_index=stream,episode_index=episode,seed=seed,
                solver_seed=seed*10,length=500,control_seconds=500.,**{"return":total}))
            for decision in range(500):
                glob=episode*500+decision
                optimizer_decision=glob if method=="run_carry" else decision
                value["rows"].append(dict(stream_index=stream,episode_index=episode,seed=seed,
                    decision_index=decision,global_decision_index=glob,entropy=-8. if decision<100 else -10.5,
                    target_entropy=-10.5,alpha_initial=.1*math.exp(-optimizer_decision*.001),
                    alpha_final=.1*math.exp(-(optimizer_decision+1)*.001),reward=total/500,
                    control_seconds=1.,optimizer_step_initial=optimizer_decision*32,
                    optimizer_step_final=(optimizer_decision+1)*32))
    return value


def all_summaries():
    return {method:chain(method) for method in pub.METHODS}


def split(summary):
    return [dict(rows=[row for row in summary["rows"] if row["stream_index"]==stream],
                 episodes=[ep for ep in summary["episodes"] if ep["stream_index"]==stream]) for stream in range(5)]


def snapshot():
    summaries=all_summaries()
    return {"checkpoints":{"mppi_aux_500k":dict(streams={method:split(summaries[method]) for method in pub.METHODS},
        science_id="long-science",status=[dict(mode=method,stream_index=stream,state="complete")
        for method in pub.METHODS for stream in range(5)])}}


def completed_entry(method="episode_reset"):
    summaries=all_summaries()
    entry=next(row for row in pub.display_rows(campaign()) if row["method"]==method)
    payload=pub.build_payload("mppi_aux_500k",method,summaries[method],summaries,{})
    return {**entry,"payload":payload,"status":"complete","progress":dict(complete=5,pending=0,failed=0,expected=5,paired_comparison_ready=1)}

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

def sdk_fixture(state):
    class Run:
        disabled=False
        settings=Obj(mode="online")
        def __init__(self):self.step=len(state["history"]);self.summary={};state["summary"]=self.summary
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

def test_two_new_sampled_arms_have_ten_actual_positions_and_unique_seed_blocks():
    rows=pub.display_rows(campaign())
    assert {row["method"] for row in rows}=={"episode_reset","run_carry"}
    assert len({row["run_id"] for row in rows})==2
    for row in rows:
        identity=row["identity"]
        assert identity["reference_reuse"] is False
        assert identity["execution_action"]=="sample"
        assert identity["episodes_per_history"]==identity["primary_episode"]==10
        assert identity["histories"]==5
    for stream in range(5):
        seeds=pub.stream_seeds(stream)
        assert len(seeds)==len(set(seeds))==10
        assert set(seeds[:5])==set(range(101,106))
        assert set(seeds[5:])==set(range(106,111))
    for episode in range(10):
        base=101 if episode<5 else 106
        assert {pub.stream_seeds(stream)[episode] for stream in range(5)}==set(range(base,base+5))


def test_episode_ten_primary_uses_five_paired_differences():
    summaries=all_summaries()
    points,overall,comparisons=pub.reduce_episodes(summaries["run_carry"],"run_carry",summaries)
    assert [point[pub.EPISODE_AXIS] for point in points]==list(range(1,11))
    expected=[stream-2+5*(stream+1) for stream in range(5)]
    assert points[-1]["carry_reset_gain_mean"]==statistics.mean(expected)==15.
    assert points[-1]["carry_reset_gain_sample_std"]==statistics.stdev(expected)
    assert points[-1]["carry_reset_wins"]==5 and points[-1]["carry_reset_worst_gain"]==3.
    assert overall["primary_episode"]==10 and overall["streams"]==5
    assert overall["primary_carry_reset_gain_mean"]==15.
    assert len(comparisons)==10 and all(len(item["pairs"])==5 for item in comparisons)
    assert [row["difference"] for row in comparisons[-1]["pairs"]]==expected
    assert "negative control" in overall["first_episode_interpretation"]


def test_last_five_minus_first_five_effect_collapses_within_history_before_sd():
    summaries=all_summaries()
    _,overall,_=pub.reduce_episodes(summaries["run_carry"],"run_carry",summaries)
    expected=[5*(stream+1) for stream in range(5)]
    assert overall["last5_minus_first5_carry_effect_mean"]==15.
    assert overall["last5_minus_first5_carry_effect_sample_std"]==statistics.stdev(expected)
    assert [row["difference"] for row in overall["last5_minus_first5_carry_effect_pairs"]]==expected
    assert "different seed blocks" in overall["window_contrast_interpretation"]


def test_first_last_hundred_entropy_error_and_reward_windows_preserve_transient():
    summaries=all_summaries()
    points,overall,_=pub.reduce_episodes(summaries["episode_reset"],"episode_reset",summaries)
    for point in points:
        assert point["early100_entropy_mean"]==-8.
        assert point["late100_entropy_mean"]==-10.5
        assert point["early100_entropy_error_mean"]==2.5
        assert point["late100_entropy_error_mean"]==0.
        assert point["entropy_mean"]==-10.
        assert point["early100_return_mean"]==pytest.approx(point["return_mean"]/5)
        assert point["late100_return_mean"]==pytest.approx(point["return_mean"]/5)
    assert overall["primary_late100_entropy_mean"]==-10.5


def test_window_sample_sd_uses_five_history_means():
    summaries=all_summaries()
    for row in summaries["episode_reset"]["rows"]:
        if row["decision_index"]<100:row["entropy"]+=row["stream_index"]
    points,_,_=pub.reduce_episodes(summaries["episode_reset"],"episode_reset",summaries)
    assert points[-1]["early100_entropy_sample_std"]==statistics.stdev([-8.,-7.,-6.,-5.,-4.])


def test_overall_distribution_uses_five_ten_episode_means_not_fifty_trials():
    summaries=all_summaries();summary=summaries["run_carry"]
    _,overall,_=pub.reduce_episodes(summary,"run_carry",summaries)
    means=[statistics.mean(ep["return"] for ep in summary["episodes"] if ep["stream_index"]==i) for i in range(5)]
    assert overall["stream_mean_return_sample_std"]==statistics.stdev(means)
    assert overall["stream_mean_return_sample_std"]!=statistics.stdev(ep["return"] for ep in summary["episodes"])
    assert "not fifty independent" in overall["statistical_unit"]


@pytest.mark.parametrize("mutation",["controller_seed","environment_seed","stream","missing"])
def test_pairing_rejects_changed_coordinates(mutation):
    summaries=all_summaries();target=summaries["episode_reset"]
    if mutation=="controller_seed":target["episodes"][0]["solver_seed"]+=1
    elif mutation=="environment_seed":target["episodes"][0]["seed"]+=1
    elif mutation=="stream":target["episodes"][0]["stream_index"]=10
    else:target["episodes"].pop()
    with pytest.raises(pub.PublicationError):
        pub.reduce_episodes(summaries["run_carry"],"run_carry",summaries)


@pytest.mark.parametrize("mutation",["duplicate","missing","old_three_episodes","nonfinite","index"])
def test_full_long_trajectory_requires_actual_complete_decision_coverage(mutation):
    summary=chain("episode_reset")
    if mutation=="duplicate":summary["rows"][-1]=summary["rows"][0]
    elif mutation=="missing":summary["rows"].pop()
    elif mutation=="old_three_episodes":summary["rows"]=[row for row in summary["rows"] if row["episode_index"]<3]
    elif mutation=="nonfinite":summary["rows"][0]["entropy"]=float("nan")
    else:summary["rows"][0]["global_decision_index"]=1
    with pytest.raises(pub.PublicationError):pub.validate_summary(summary)


def test_five_thousand_decisions_and_all_nine_boundaries_are_preserved():
    carry=pub.reduce_decisions(chain("run_carry"));reset=pub.reduce_decisions(chain("episode_reset"))
    assert len(carry)==len(reset)==5000
    assert carry[-1][pub.AXIS]==5000 and carry[-1][pub.EPISODE_AXIS]==10
    assert carry[-1]["optimizer_step_final"]==160000
    assert reset[-1]["optimizer_step_final"]==16000
    for episode in range(1,10):
        decision=episode*500
        assert carry[decision]["boundary_alpha_jump"]==pytest.approx(0.)
        assert reset[decision]["alpha_initial"]==pytest.approx(.1)
        assert reset[decision]["boundary_alpha_jump"]==pytest.approx(.1-.1*math.exp(-.5))
        assert reset[decision]["episode_cumulative_reward"]<reset[decision-1]["episode_cumulative_reward"]
        assert reset[decision]["chain_cumulative_reward"]>reset[decision-1]["chain_cumulative_reward"]


def test_decision_reward_sum_must_match_saved_episode_return():
    summary=chain("run_carry");summary["episodes"][-1]["return"]+=1
    with pytest.raises(pub.PublicationError,match="sum"):
        pub.reduce_decisions(summary)


def test_no_historical_results_are_reused_or_fabricated_when_new_cells_pending(monkeypatch):
    from slurm import ambi_alpha_long_episodes as driver
    state=snapshot();target=state["checkpoints"]["mppi_aux_500k"]
    target["reference_streams"]={method:deepcopy(value) for method,value in target["streams"].items()}
    target["streams"]={method:[] for method in pub.METHODS}
    for row in target["status"]:row["state"]="pending"
    monkeypatch.setattr(driver,"publication_snapshot",lambda campaign:state)
    entries=pub.collect(campaign())
    assert len(entries)==2 and all(entry["payload"] is None for entry in entries)
    assert all(entry["status"]=="pending" and entry["progress"]["complete"]==0 for entry in entries)


def test_both_curves_wait_for_all_ten_paired_cells(monkeypatch):
    from slurm import ambi_alpha_long_episodes as driver
    state=snapshot();target=state["checkpoints"]["mppi_aux_500k"]
    target["streams"]["run_carry"].pop()
    next(row for row in target["status"] if row["mode"]=="run_carry" and row["stream_index"]==4)["state"]="running"
    monkeypatch.setattr(driver,"publication_snapshot",lambda campaign:state)
    entries={entry["method"]:entry for entry in pub.collect(campaign())}
    assert all(entry["payload"] is None and entry["status"]=="pending" for entry in entries.values())
    assert entries["episode_reset"]["progress"]["complete"]==5
    assert entries["run_carry"]["progress"]["complete"]==4
    final={entry["method"]:entry for entry in pub.collect(campaign(),require_complete=True)}
    assert all(entry["status"]=="failed" for entry in final.values())
    assert final["run_carry"]["progress"]["failed"]==1


def test_native_layout_has_two_methods_twenty_seven_panels_and_solid_curves(tmp_path):
    sdk,state=backend();value=campaign()
    first=pub.install_workspace(value,tmp_path/"workspace.json",backend=sdk)
    state["work"].sections.append(Obj(name="Personal notes",panels=[]))
    second=pub.install_workspace(value,tmp_path/"workspace.json",backend=sdk)
    assert first["url"]==second["url"] and second["api_verified"] and second["panel_count"]==27
    assert len(state["work"].sections)==4
    plots=[p for section in state["work"].sections for p in section.panels if isinstance(p,LinePlot)]
    assert len(plots)==23
    for plot in plots:
        metric=plot.y[0].split("/")[-1];expected=pub.plotted_rows(pub.display_rows(value),metric)
        assert len(plot.line_titles)==len(expected)
        assert set(plot.line_marks.values())=={"solid"} and not plot.aggregate
        assert all(plot.line_colors[row["run_id"]+":"+plot.y[0]]==row["color"] for row in expected)


@pytest.mark.parametrize("failure",["mark","color","metric","axis"])
def test_saved_layout_rejects_broken_native_curve_settings(tmp_path,failure):
    def corrupt(work):
        plot=next(p for section in work.sections for p in section.panels if isinstance(p,LinePlot))
        if failure=="mark":plot.line_marks.pop(next(iter(plot.line_marks)))
        elif failure=="color":plot.line_colors[next(iter(plot.line_colors))]="#FFFFFF"
        elif failure=="metric":plot.y=["episode/mppi_aux_500k/not_a_metric"]
        else:plot.x="Step"
    sdk,_=backend(corrupt)
    with pytest.raises(pub.PublicationError):pub.install_workspace(campaign(),tmp_path/"workspace.json",backend=sdk)
    assert not series._read(tmp_path/"workspace.json")["api_verified"]


def test_pending_to_complete_native_history_is_idempotent_and_paired_results_visible(tmp_path):
    entry=completed_entry("run_carry")
    state=dict(history=[],logged=[],inits=[],config={});sdk=sdk_fixture(state)
    assert pub.publish_entry({**entry,"payload":None,"status":"pending"},tmp_path,sdk)["rows"]==0
    result=pub.publish_entry(entry,tmp_path,sdk)
    assert result["status"]=="complete" and result["rows"]==5000
    assert state["config"]["reused_reference"] is False
    assert state["summary"]["comparison/paired_episode_results"]==entry["payload"]["paired_comparisons"]
    assert pub.publish_entry(entry,tmp_path,sdk)==result
    assert len(state["inits"])==2 and len(state["logged"])==5000
    entry["payload"]["episode_points"][-1]["return_mean"]+=1
    with pytest.raises(pub.PublicationError,match="result changed"):
        pub.publish_entry(entry,tmp_path,sdk)


def test_interrupted_native_publication_resumes_exact_prefix(tmp_path):
    entry=completed_entry();rows=pub.history_rows(entry)
    state=dict(history=[{**rows[0],"_step":0}],logged=[],inits=[],config={"chain_identity_sha256":series._digest(entry["identity"])})
    series._atomic_json(tmp_path/"runs"/entry["run_id"]/"publication.json",dict(identity_sha256=series._digest(entry["identity"]),initialized=True,status="inflight"))
    pub.publish_entry(entry,tmp_path,sdk_fixture(state))
    assert state["logged"]==list(range(1,5000)) and len(state["history"])==5000


def test_report_uses_ten_positions_five_thousand_decisions_and_actual_paired_details(tmp_path):
    entry=completed_entry("run_carry");path=tmp_path/"report.html"
    data=pub.render_report(campaign(),[entry],path);html=path.read_text()
    assert "add('polyline'" in html and "data.global_decisions" in html and "data.episodes-1" in html
    assert "Paired histories" in html and "Fifty episodes are not fifty independent trials" in html
    embedded=html.split('<script id="chain-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(embedded)==json.loads(json.dumps(data))
    assert data["episodes"]==10 and data["global_decisions"]==5000
    assert len(data["entries"][0]["payload"]["paired_comparisons"])==10


def test_layout_required_before_source_or_publication_calls(tmp_path,monkeypatch):
    monkeypatch.setattr(pub,"collect",lambda *args,**kwargs:pytest.fail("No source scan before layout"))
    with pytest.raises(pub.PublicationError,match="workspace"):
        pub.publish_once(campaign(),tmp_path,wandb_module=Obj())


def test_native_episode_effects_occur_only_at_ten_actual_episode_ends():
    entry=completed_entry("run_carry");rows=pub.history_rows(entry)
    key="episode/mppi_aux_500k/carry_reset_gain_mean"
    assert [row[pub.AXIS] for row in rows if key in row]==list(range(500,5001,500))
    assert [row[pub.EPISODE_AXIS] for row in rows if key in row]==list(range(1,11))
    assert not any("prior_gain" in key for row in rows for key in row)
