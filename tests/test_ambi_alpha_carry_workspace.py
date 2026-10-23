"""Validate native curve identity and fail if the server drops important style."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm import ambi_alpha_carry_workspace as layout


class Object(SimpleNamespace):
    pass


class Filter:
    def __init__(self, name, value):
        self.name, self.value = name, value

    def to_model(self):
        return Object(model_dump=lambda **kwargs: {"op": "=", "key": {"name": self.name}, "value": self.value})


class Metric:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return Filter(self.name, value)


class LinePlot(Object):
    pass


def backend(mutate=None):
    state = {"saved": {}, "created": 0}

    class Workspace(Object):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if kwargs:
                state["created"] += 1
                self.url = f"https://wandb.ai/{layout.ENTITY}/{layout.PROJECT}?nw=view{state['created']}"

        def save(self):
            state["saved"][self.url] = deepcopy(self)

        @classmethod
        def from_url(cls, url):
            work = deepcopy(state["saved"][url])
            if mutate:
                mutate(work)
            return work

    ws = Object(Workspace=Workspace, Section=Object, WorkspaceSettings=Object,
                RunsetSettings=lambda **kwargs: Object(query="", **kwargs), RunSettings=Object)
    wr = Object(LinePlot=LinePlot, MarkdownPanel=Object, Layout=Object)
    return (ws, wr, Object(Metric=Metric)), state


def api(count=8):
    rows = layout.expected_rows()
    runs = [Object(id=f"run{index}", name=row["name"], state="running",
                   summary={"env_step": 1000, "episode/return": 50.})
            for index, row in enumerate(rows[:count])]
    return Object(runs=lambda *args, **kwargs: runs), runs


def plots(work):
    return [panel for section in work.sections for panel in section.panels if isinstance(panel, LinePlot)]


def test_pending_then_registered_runs_reuse_view_and_preserve_user_sections(tmp_path):
    sdk, state = backend()
    receipt = tmp_path / "workspace.json"
    pending = layout.install(receipt, api=api(0)[0], backend=sdk)
    assert pending["api_verified"] and all(row["run_id"] is None for row in pending["rows"])
    url = pending["url"]
    state["saved"][url].sections.append(Object(name="User notes", panels=[Object(text="Preserve")]))
    complete = layout.install(receipt, api=api()[0], backend=sdk)
    assert complete["url"] == url and state["created"] == 1
    saved = state["saved"][url]
    assert saved.sections[-1].panels[0].text == "Preserve"
    assert len(plots(saved)) == len(layout.METRICS)
    assert len(saved.runset_settings.run_settings) == 8
    assert len({value.color for value in saved.runset_settings.run_settings.values()}) == 8
    for plot in plots(saved):
        assert not plot.aggregate and plot.smoothing_type == "none"
        assert set(plot.line_marks.values()) == {"solid"}
        assert len(plot.line_titles) == 8
        metric = plot.y[0]
        expected_axis = next(axis for _, _, name, axis in layout.METRICS if name == metric)
        assert plot.x == expected_axis
    assert layout.DIAGNOSTIC_STEP != layout.STEP


@pytest.mark.parametrize("failure", ["axis", "color", "legend", "group", "marks", "missing"])
def test_invalid_server_readback_fails_with_reusable_receipt(tmp_path, failure):
    def corrupt(work):
        panel = plots(work)[0]
        if failure == "axis":
            panel.x = "Step"
        elif failure == "color":
            work.runset_settings.run_settings["run0"].color = "#FFFFFF"
        elif failure == "legend":
            panel.line_titles = {}
        elif failure == "group":
            work.runset_settings.filters = [Metric("Group") == "another-campaign"]
        elif failure == "marks":
            panel.line_marks = {key: "points" for key in panel.line_marks}
        else:
            work.sections.pop()
    sdk, state = backend(corrupt)
    receipt = tmp_path / "workspace.json"
    with pytest.raises(layout.WorkspaceError):
        layout.install(receipt, api=api()[0], backend=sdk)
    saved = json.loads(receipt.read_text())
    assert saved["url"] and saved["api_verified"] is False
    assert saved["error_type"] == "WorkspaceError" and state["created"] == 1


def test_duplicate_factorial_run_identity_is_not_silently_merged():
    public, runs = api()
    runs.append(deepcopy(runs[0]))
    runs[-1].id = "duplicate"
    with pytest.raises(layout.WorkspaceError, match="Multiple run IDs"):
        layout.discover_rows(public)


def test_unrelated_receipt_is_not_overwritten(tmp_path):
    receipt = tmp_path / "another.json"
    receipt.write_text('{"group":"different"}')
    with pytest.raises(layout.WorkspaceError, match="another campaign"):
        layout.install(receipt, api=api()[0], backend=backend()[0])
    assert receipt.read_text() == '{"group":"different"}'


def test_pinned_sdk_roundtrip_without_network(tmp_path, monkeypatch):
    ws = pytest.importorskip("wandb_workspaces.workspaces")
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    if (layout.importlib.metadata.version("wandb") != "0.28.1"
            or layout.importlib.metadata.version("wandb-workspaces") != "0.4.5"):
        pytest.skip("Requires the isolated pinned workspace runtime")
    state = {}
    url = f"https://wandb.ai/{layout.ENTITY}/{layout.PROJECT}?nw=mockview"
    monkeypatch.setattr(ws.Workspace, "url", property(lambda self: url))
    monkeypatch.setattr(ws.Workspace, "save", lambda self: state.update(work=deepcopy(self)))
    monkeypatch.setattr(ws.Workspace, "from_url", classmethod(lambda cls, value: deepcopy(state["work"])))
    receipt = layout.install(tmp_path / "layout.json", api=api()[0])
    assert receipt["api_verified"]
    model = state["work"]._to_model()
    wire = model.model_dump_json()
    assert layout.DIAGNOSTIC_STEP in wire and "solid" in wire
    assert all(row["name"] in wire for row in layout.expected_rows())
