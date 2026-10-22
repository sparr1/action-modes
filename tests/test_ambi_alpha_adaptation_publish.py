"""Scientific result reuse and native decision curves remain distinct and paired."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_alpha_adaptation_publish as pub
from utils import eval_series as series


def campaign():
    value = dict(campaign_id=pub.CAMPAIGN_ID, source_commit="a"*40,
                 seeds=[101, 102, 103, 104, 105], max_steps=500, cells=[])
    for checkpoint in pub.CHECKPOINTS:
        for mode in ("repeat100", "carry"):
            value["cells"].append(dict(checkpoint_id=checkpoint, mode=mode, cell_id=checkpoint+"__"+mode,
                run_id="eval"+str(len(value["cells"])), run_dir="/runs/"+checkpoint+mode))
    return value


def summary(value, *, gain=0):
    return dict(rows=[dict(seed=seed, decision_index=decision, entropy=float(seed)+decision,
                 target_entropy=-10.5, alpha_initial=.01, alpha_final=.005,
                 optimizer_step_final=None, cumulative_reward=decision+1+gain, control_seconds=.5)
                     for seed in value["seeds"] for decision in range(value["max_steps"])],
        episodes=[dict(seed=seed, solver_seed=seed+100, length=value["max_steps"],
                       **{"return": float(seed)+gain}) for seed in value["seeds"]])


def test_decision_reduction_keeps_true_axis_seed_units_and_missing_lifecycle():
    value = campaign()
    reduced = pub.reduce_decisions(summary(value), value)
    assert len(reduced) == 500
    assert reduced[0][pub.AXIS] == 1 and reduced[-1][pub.AXIS] == 500
    assert reduced[0]["entropy"] == 103 and reduced[1]["entropy"] == 104
    assert reduced[0]["entropy_sample_std"] == pytest.approx(1.5811388300841898)
    assert reduced[0]["entropy_error"] == 113.5
    assert "optimizer_step_final" not in reduced[0]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "alpha", "partial", "nonfinite"])
def test_rejects_partial_or_invalid_per_decision_panels(failure):
    value = campaign()
    data = summary(value)
    if failure == "missing": data["rows"].pop()
    elif failure == "duplicate": data["rows"][-1] = deepcopy(data["rows"][0])
    elif failure == "alpha": data["rows"][0]["alpha_final"] = 0
    elif failure == "partial": data["rows"][0]["optimizer_step_final"] = 32
    elif failure == "nonfinite": data["rows"][0]["entropy"] = float("nan")
    with pytest.raises(pub.PublicationError):
        pub.reduce_decisions(data, value)


def test_episode_improvement_is_seed_paired_and_matches_source():
    value = campaign()
    actual, baseline, prior = summary(value, gain=5), summary(value, gain=2), summary(value)
    result = pub.summarize_episodes(actual, baseline, prior, value)
    assert result["return_mean"] == 108
    assert result["prior_gain_mean"] == 5
    assert result["baseline_gain_mean"] == 3
    assert result["baseline_gain_sample_std"] == 0
    assert result["control_seconds_per_decision"] == .5
    prior["episodes"][0]["solver_seed"] += 1
    with pytest.raises(pub.PublicationError, match="controller seeds"):
        pub.summarize_episodes(actual, baseline, prior, value)


def test_display_identity_is_stable_and_never_reuses_evaluation_ids():
    value = campaign()
    rows = pub.display_rows(value)
    assert rows == pub.display_rows(value)
    assert len({row["run_id"] for row in rows}) == 6
    assert not {row["run_id"] for row in rows} & {cell["run_id"] for cell in value["cells"]}
    assert len([row for row in rows if row["reused_result"]]) == 2
    for checkpoint in pub.CHECKPOINTS:
        assert len({row["color"] for row in rows if row["checkpoint_id"] == checkpoint}) == 3


def test_resume_requires_exact_prefix_and_rejects_duplicate_rows():
    expected = [{pub.AXIS: 1, "derived/row_sha256": "a"}, {pub.AXIS: 2, "derived/row_sha256": "b"}]
    actual = [{**expected[0], "_step": 0}]
    assert pub._verify_prefix(actual, expected) == 1
    assert pub._verify_prefix([], expected) == 0
    with pytest.raises(pub.PublicationError):
        pub._verify_prefix(actual + actual, expected)
    with pytest.raises(pub.PublicationError):
        pub._verify_prefix([{**actual[0], pub.AXIS: 99}], expected)


class Obj(SimpleNamespace):
    pass


class Filter:
    def __init__(self, name, values): self.name, self.values = name, values
    def to_model(self):
        return Obj(model_dump=lambda **kwargs: dict(op="IN", key=dict(name=self.name), value=self.values))


class Metric:
    def __init__(self, name): self.name = name
    def isin(self, values): return Filter(self.name, values)


class LinePlot(Obj):
    pass


def fake_layout(corrupt=None):
    state = {}
    class Workspace(Obj):
        @property
        def url(self): return f"https://wandb.ai/{pub.ENTITY}/{pub.PROJECT}?nw=alpha-test"
        def save(self): state["work"] = deepcopy(self)
        @classmethod
        def from_url(cls, url):
            result = deepcopy(state["work"])
            if corrupt: corrupt(result)
            return result
    ws = Obj(Workspace=Workspace, Section=Obj, WorkspaceSettings=Obj, RunsetSettings=Obj, RunSettings=Obj)
    wr = Obj(LinePlot=LinePlot, ScalarChart=Obj, MarkdownPanel=Obj, Layout=Obj)
    return (ws, wr, Obj(Metric=Metric)), state


def test_workspace_is_dedicated_idempotent_and_connects_actual_decisions(tmp_path):
    sdk, state = fake_layout()
    path = tmp_path / "workspace.json"
    first = pub.install_workspace(campaign(), path, backend=sdk)
    state["work"].sections.append(Obj(name="Personal notes", panels=[]))
    second = pub.install_workspace(campaign(), path, backend=sdk)
    assert first["url"] == second["url"] and second["api_verified"]
    assert len(state["work"].sections) == 4
    assert state["work"].sections[0].name == "Personal notes"
    plots = [panel for section in state["work"].sections for panel in section.panels if isinstance(panel, LinePlot)]
    assert len(plots) == 16
    for plot in plots:
        assert plot.x == pub.AXIS and not plot.aggregate and plot.smoothing_type == "none"
        assert set(plot.line_marks.values()) == {"solid"}
        assert len(plot.line_titles) == 3


@pytest.mark.parametrize("field", ["line_marks", "line_colors", "line_titles", "x"])
def test_layout_readback_rejects_lost_native_curve_settings(tmp_path, field):
    def corrupt(work):
        plot = next(panel for section in work.sections for panel in section.panels if isinstance(panel, LinePlot))
        setattr(plot, field, "Step" if field == "x" else {})
    sdk, _ = fake_layout(corrupt)
    path = tmp_path / "workspace.json"
    with pytest.raises(pub.PublicationError):
        pub.install_workspace(campaign(), path, backend=sdk)
    assert series._read(path)["api_verified"] is False


def test_layout_readback_rejects_one_missing_solid_line(tmp_path):
    def corrupt(work):
        plot = next(panel for section in work.sections for panel in section.panels if isinstance(panel, LinePlot))
        plot.line_marks.pop(next(iter(plot.line_marks)))
        assert set(plot.line_marks.values()) == {"solid"}
    sdk, _ = fake_layout(corrupt)
    path = tmp_path / "workspace.json"
    with pytest.raises(pub.PublicationError, match="solid line"):
        pub.install_workspace(campaign(), path, backend=sdk)
    assert series._read(path)["api_verified"] is False


def test_missing_layout_prevents_external_mutation(tmp_path):
    with pytest.raises(pub.PublicationError, match="workspace"):
        pub.publish_once(campaign(), tmp_path, wandb_module=Obj())


def test_html_embeds_actual_lines_and_explicit_pending_without_zeroes(tmp_path):
    value = campaign()
    rows = pub.display_rows(value)
    displays = [{**{k:v for k,v in row.items() if k != "cell"}, "status":"pending", "history":[], "summary":{}}
                for row in rows]
    path = tmp_path / "report.html"
    payload = pub.render_report(value, displays, path)
    text = path.read_text()
    assert "add('polyline'" in text and "Pending measured results" in text
    encoded = text.split('<script id="alpha-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(encoded) == json.loads(json.dumps(payload))
    assert all(not row["summary"] and not row["history"] for row in payload["displays"])


def test_derived_publication_resumes_pending_then_is_idempotent(tmp_path):
    value = campaign()
    row = {k:v for k,v in pub.display_rows(value)[0].items() if k != "cell"}
    row.update(status="pending", history=[], summary={}, provenance={"source":"old-baseline"})
    state = {"history":[], "calls":0}
    class Run:
        disabled = False
        settings = Obj(mode="online")
        def __init__(self, config):
            self.config = config
            self.summary = {}
            self.step = len(state["history"])
        def define_metric(self, *a, **k): pass
        def log(self, data, *, step, commit):
            assert step == len(state["history"])
            state["history"].append({**data,"_step":step})
        def finish(self, **kwargs): pass
    def init(**kwargs):
        state["calls"] += 1
        state["config"] = kwargs["config"]
        return Run(kwargs["config"])
    remote = Obj(scan_history=lambda **kwargs: deepcopy(state["history"]))
    def api_run(path):
        remote.config = state["config"]
        return remote
    sdk = Obj(init=init, Api=lambda **kwargs:Obj(run=api_run))
    first = pub.publish_display(row, tmp_path, sdk)
    assert first["rows"] == 0 and first["status"] == "pending"
    row.update(status="complete", history=pub.reduce_decisions(summary(value), value), summary={"return_mean":103.})
    second = pub.publish_display(row, tmp_path, sdk)
    assert second["rows"] == 500 and state["calls"] == 2
    assert pub.publish_display(row, tmp_path, sdk) == second
    assert state["calls"] == 2 and len(state["history"]) == 500
    row["history"][0]["entropy"] += 1
    with pytest.raises(pub.PublicationError, match="result changed"):
        pub.publish_display(row, tmp_path, sdk)


def test_derived_interrupted_rows_resume_only_unoccupied_prefix(tmp_path):
    value = campaign()
    display = {k:v for k,v in pub.display_rows(value)[0].items() if k != "cell"}
    display.update(status="complete", history=pub.reduce_decisions(summary(value), value), summary={}, provenance={})
    wire = pub._wire_history(display)
    state = {"history":[{**r,"_step":i} for i,r in enumerate(wire[:200])], "logged":[]}
    path = tmp_path / "displays" / display["run_id"] / "publication.json"
    series._atomic_json(path,dict(identity_sha256=series._digest(display["identity"]),initialized=True,status="inflight"))
    remote = Obj(config={"derived_identity_sha256":series._digest(display["identity"])},
                 scan_history=lambda **kwargs:deepcopy(state["history"]))
    run = Obj(step=200,summary={},define_metric=lambda *a,**k:None,finish=lambda **k:None)
    def log(row, *, step, commit):
        state["logged"].append(step)
        state["history"].append({**row,"_step":step})
    run.log = log
    sdk = Obj(init=lambda **kwargs:run,Api=lambda **kwargs:Obj(run=lambda path:remote))
    result = pub.publish_display(display,tmp_path,sdk)
    assert result["rows"] == 500 and state["logged"] == list(range(200,500))


def test_scientific_stage_omits_traces_and_rejects_changed_workload(tmp_path, monkeypatch):
    from slurm import ambi_alpha_adaptation_episodes as driver
    from utils import eval_series_data as data
    value = campaign()
    value["cells"] = [dict(value["cells"][0], output_dir=str(tmp_path/"output"), run_dir=str(tmp_path/"run"),
        source_run="entity/project/backbone", selector="alpha_adaptation/repeat100", step=100000,
        checkpoint_sha256="c"*64, inventory_path=str(tmp_path/"inventory.json"))]
    cell = value["cells"][0]
    series._atomic_json(cell["inventory_path"],{})
    cell["inventory_sha256"] = series._file_digest(cell["inventory_path"])
    series._atomic_json(Path(cell["run_dir"])/"publication.json",{"records":{}})
    output = Path(cell["output_dir"])
    manifest = output/"bundle/manifest.json"
    series._atomic_json(manifest,dict(runs=[dict(selector=cell["selector"],status="complete",trace_files=["full.jsonl.gz"])]))
    trace = manifest.parent/"full.jsonl.gz"
    trace.write_bytes(b"full trace remains on Oscar")
    series._atomic_json(output/"results.json",{})
    series._atomic_json(output/"decision-summary.json",dict(mode="repeat100",checkpoint_id=cell["checkpoint_id"]))
    receipt = dict(validated=True,smoke=False,cell_id=cell["cell_id"],source_commit=value["source_commit"],
        manifest_sha256=series._file_digest(manifest),results_sha256=series._file_digest(output/"results.json"),
        decision_summary_sha256=series._file_digest(output/"decision-summary.json"),
        trace_sha256={trace.name:series._file_digest(trace)})
    series._atomic_json(output/"receipt.json",receipt)
    record = dict(selector=cell["selector"],checkpoint={"step":100000,"sha256":"c"*64},
        episodes=[dict(seed=s) for s in value["seeds"]],provenance={},
        metrics={"eval/frozen_state_unchanged":True,"work/actor_updates":80000,"work/critic_updates":320000,
                 "work/temperature_updates":8000000},artifact_files={"manifest.json":str(manifest),trace.name:str(trace)})
    staged=[]
    monkeypatch.setattr(pub,"validate_registry",lambda *args:None)
    monkeypatch.setattr(driver,"reference_pins",lambda cell:{})
    monkeypatch.setattr(driver,"validate_decisions",lambda *args:None)
    monkeypatch.setattr(data,"load_records",lambda *args,**kwargs:[deepcopy(record)])
    monkeypatch.setattr(series,"stage_record",lambda directory,value:staged.append(value))
    assert pub.stage_completed(value)=={cell["cell_id"]:"complete"}
    assert trace.name not in staged[0]["artifact_files"] and trace.exists()
    assert "decision-summary.json" in staged[0]["artifact_files"]
    assert staged[0]["provenance"]["remote_trace_files"][0]["sha256"]==series._file_digest(trace)
    record["metrics"]["work/temperature_updates"]=80000
    with pytest.raises(pub.PublicationError,match="optimizer update counts"):
        pub.stage_completed(value)


@pytest.mark.parametrize("created_before_failure", [False, True])
def test_failed_first_init_retries_after_verifying_remote_absence_or_identity(tmp_path, created_before_failure):
    value = campaign()
    display = {k:v for k,v in pub.display_rows(value)[0].items() if k != "cell"}
    display.update(status="pending", history=[], summary={}, provenance={})
    state = {"calls":0,"remote":None,"resumes":[]}
    run = Obj(step=0,summary={},define_metric=lambda *a,**k:None,finish=lambda **k:None)
    def init(**kwargs):
        state["calls"] += 1
        state["resumes"].append(kwargs["resume"])
        if state["calls"] == 1:
            if created_before_failure:
                state["remote"] = Obj(config=kwargs["config"],scan_history=lambda **k:[])
            raise RuntimeError("first initialization failed")
        return run
    def api_run(path):
        if state["remote"] is None:
            raise ValueError(f"Could not find run <Run {path} (not found)>")
        return state["remote"]
    sdk = Obj(init=init,Api=lambda **k:Obj(run=api_run))
    with pytest.raises(RuntimeError,match="first initialization"):
        pub.publish_display(display,tmp_path,sdk)
    journal = series._read(tmp_path/"displays"/display["run_id"]/"publication.json")
    assert journal["init_started"] is True and journal["initialized"] is False
    result = pub.publish_display(display,tmp_path,sdk)
    assert result["initialized"] is True
    assert state["resumes"] == ["allow","must" if created_before_failure else "allow"]


def test_uncertain_init_network_failure_does_not_create_or_overwrite_run(tmp_path):
    display = {k:v for k,v in pub.display_rows(campaign())[0].items() if k != "cell"}
    display.update(status="pending",history=[],summary={},provenance={})
    path = tmp_path/"displays"/display["run_id"]/"publication.json"
    series._atomic_json(path,dict(identity_sha256=series._digest(display["identity"]),
                                 init_started=True,initialized=False,status="inflight"))
    def api_run(path): raise RuntimeError("connection refused")
    sdk = Obj(init=lambda **kwargs:pytest.fail("No creation after uncertain network failure"),
              Api=lambda **kwargs:Obj(run=api_run))
    with pytest.raises(RuntimeError,match="connection refused"):
        pub.publish_display(display,tmp_path,sdk)
